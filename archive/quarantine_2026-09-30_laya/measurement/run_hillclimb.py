#!/usr/bin/env python3
"""run_hillclimb.py -- the actual selection loop: propose -> train -> score on DEV -> accept iff d > tau.

Harness: the loop container is `autoresearch-win-rtx` (`jsegov/autoresearch-win-rtx`, cloned locally at
the sibling `nanochat-autoresearch` clone, whose path is set by JEVRSI_HARNESS). Its conventions -- a fixed wall-clock generation budget, a
`results.tsv` ledger, and the record-before-reset rejection protocol -- are the substrate JevRSI runs
on. The harness itself is NOT the research object, and its `val_bpb` metric is not ours.

WHAT WAS MISSING
----------------
Everything measured so far describes the RULER: Floor A, Floor B, tau, and the accept rule's
false-accept rate under a true null. None of it is a climb. This script runs the climb.

THE DISCIPLINE IT ENFORCES
  * proposals are a FIXED, pre-registered schedule -- the loop cannot cherry-pick an intervention
    after seeing its score
  * the selection signal is DEV (120 cases / 600 questions), never test
  * tau is measured ON DEV, because tau is split-dependent: dev's 600 questions give a wider paired
    SE than test's 2000, so a test-derived tau would accept noise on dev
  * the incumbent only moves on an accept, so the trajectory is a chain, not a series of one-offs
  * the frozen TEST split is touched ONCE, at the end, to report the chain's true endpoint

Every GPU job runs as its own subprocess that exits before the next begins: the measured margin on
this 12 GiB card is ~1.6 GiB, and co-residency is what produced the original silent stall.
"""
from __future__ import annotations

import os
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from provenance import fingerprint_str, weights_fingerprint

REPO = Path(__file__).resolve().parent.parent
MEAS = REPO / "measurement"
# A historical arm from the retired laya substrate; the path is supplied by the operator
# rather than hardcoded, so the module imports on any machine.
BASE_CKPT = Path(os.environ.get("JEVRSI_BASE_CKPT", "."))

# The config fields that fully determine a training run's outcome. A checkpoint may be reused only
# if the stamp records exactly these values.
CONFIG_KEYS = ("time_budget", "max_steps_ceiling", "batch_cases", "lr", "head_lr",
               "warmup", "weight_decay")

# --- generation budget and viability -----------------------------------------------------------
# The budget is a JevRSI DESIGN PARAMETER, not a copy of the container's 5 minutes. It is set from
# two measured constraints that pull in opposite directions:
#   * long enough that a plausible intervention can exceed tau_dev -- otherwise every generation is
#     powerless by construction, no matter how well tau is calibrated;
#   * short enough to keep 24/7 throughput usable.
# See docs/JevRSI_global_principles_v1.md for the derivation from the measured learning curve.
GENERATION_BUDGET_S = 1200.0      # 20 min of training per generation (excluding setup)
SETUP_ALLOWANCE_S = 150.0         # model load + 4200-item join + first-step compile
HARD_CEILING_S = GENERATION_BUDGET_S + SETUP_ALLOWANCE_S
MIN_STEP_FRACTION = 0.9           # PROVISIONAL -- to be replaced by our own measurement

# WHY THE BAND EXISTS, AND WHY THIS VALUE IS ONLY A PLACEHOLDER
# Under a fixed-time budget the realised step count is an outcome of host speed. If the metric is
# sensitive to step count, two generations with different step counts are not fully comparable, and
# pooling them is the pooled-regression error in miniature. So a band is needed.
#
# Its WIDTH, though, must be measured ON THIS ARM -- laya-multilingual, Top-1 on typed-decisions --
# because that is the only instrument the band is about. The harness's own Floor B
# (`measurement/measure_floor_b_nanochat.py`, nanochat / val_bpb) reports that 99.4% of ITS spread is
# `num_steps` rather than seed, and that number is deliberately NOT used here: it describes a different
# model, dataset and metric, and importing it would be cross-instrument inference -- borrowing an idea
# is fine; borrowing a number is not. JevRSI borrows the harness's FRAMEWORK, not its findings.
#
# CONSEQUENCE, stated plainly: under a fixed-time budget a proposal may change the OPTIMISATION
# (lr, head_lr, warmup, weight decay, loss weighting, data order) but not the COMPUTE PER STEP. A
# proposal that changes batch size / sequence length / width realises far fewer steps in the same
# budget, so it is rejected as NOT_VIABLE rather than scored. That is a real trade against the
# container's premise that any change stays comparable: on this metric, at this scale, it does not --
# and pretending otherwise would silently pool incomparable numbers.


# ---- THE METRIC'S PUBLISHED MEANING BOUNDARY -----------------------------------------------
# These three numbers are NOT ours. They are published by the benchmark's own maintainers in the
# dataset card (`LocalLLaMA/typed-decisions`), retrieved 2026-09-28:
#
#   Perfect scenario understanding  0.704  fitted to the latent factors that generated each case
#   Teacher self-agreement          0.735  a fresh teacher sample scored against gold from the others
#   "Around 0.75 is saturation."    0.750  card text, verbatim
#
# and the operative sentence, also verbatim: "A score much above 0.75 means a model has learned the
# teacher's quirks rather than the task."
#
# WHY THIS IS LOAD-BEARING FOR A CLIMBING LOOP. The loop optimises the metric. Above the teacher
# self-agreement reference the metric rewards reproducing the teacher's *systematic* quirks -- which
# are shared between train and test gold, unlike the teacher's random noise -- so a climbing loop
# will be pulled into that region and will report it as improvement. The boundary is therefore an
# EXTERNAL guardrail: it was published before we measured anything, and we cannot tune it to taste.
# It is also the argument for using these as constants in the driver rather than as prose in a doc:
# a documented intention that no code enforces is a guardrail that does not exist.
#
# REVISED 2026-09-29 after the AnyJev audit. The dataset card labels 0.704 a "factor ceiling" and 0.735 a
# "teacher self-agreement" ceiling, and earlier versions of this file treated them as upper bounds. Public
# measurements on the SAME 400-case / 2,000-decision test split refute that reading: laya-typed-decisions
# scores 0.768, and AnyJev L2 scores 0.786 (Qwen3-4B) and 0.799 (Qwen3-30B-A3B). So 0.735 is the ceiling
# only on systems that reproduce the TEACHER'S OWN samples; a system that predicts the scenario is not
# bound by it. The constants below are therefore reclassified: 0.470 is a real floor, 0.735 and 0.799 are
# reference lines, and nothing here is a prohibition. Source: docs/RESEARCH_GOAL_v2_2026-09-29.md
REF_PRIOR               = 0.470   # ignores the input -- a genuine floor
REF_FACTOR_CEILING      = 0.704   # dataset-card label; a factor reference, NOT a ceiling
REF_TEACHER_SELFAGREE   = 0.735   # teacher self-consistency; a diagnostic trigger, NOT a ceiling
REF_OBSERVED_SOTA       = 0.799   # AnyJev L2, Qwen3-30B-A3B -- the figure to beat
REF_SATURATION          = 0.750   # RETIRED as a ceiling; retained only so old reports stay parseable


def release_generation_weights(gen_dir, keep: bool, say=print) -> int:
    """Release a generation's weights once its row is in the ledger. Returns bytes freed.

    THE ORDERING IS THE POINT, and it is the container's own (program.md, verbatim): "persist the
    attempt first, then `git reset` back to where you started. The reset is what deletes the branch,
    so the record must already be on disk before you run it". Here the reset deletes weights instead
    of a branch, so the same rule applies: this is only ever called AFTER the generation's row has
    been appended to `history`, so no record can be lost by deleting the weights.

    WHY THE WEIGHTS ARE DELETABLE AT ALL -- the part that is easy to get wrong. Every generation
    trains from the SAME fixed base checkpoint with a config delta: the driver passes seed, budget,
    lr, head_lr, warmup, weight_decay -- never an `--init-from` pointing at the incumbent. The loop
    therefore carries the CONFIG forward, not the weights; there is no fine-tuning chain. A
    generation's weights are a pure function of (base revision, config, seed), so retaining them
    stores a regenerable derived artefact. At 1.29 GB each that is the whole difference between a
    constant ~3 GB footprint and 81 GB/day.

    Metrics, predictions and config stay: those are the record, and they are kilobytes.
    """
    w = gen_dir / "model.safetensors"
    if keep or not w.exists():
        return 0
    try:
        n = w.stat().st_size
        w.unlink()
        say(f"      [release] freed {n / 1e9:.2f} GB of weights (record kept; "
            f"regenerable from base + config + seed)")
        return n
    except OSError as e:
        say(f"      [release] could not free {w.name}: {e}")
        return 0


def meaning_zone(acc, ceiling=REF_TEACHER_SELFAGREE):
    """A TRIGGER for the mechanism test -- NOT a ceiling and NOT a prohibition.

    REVISED 2026-09-28 after the test the original version demanded was actually run
    (measurement/mechanism_test.py). The original comment treated teacher self-agreement as a "hard
    stop" past which gains are illegitimate. That was wrong on two counts, both of which the evidence
    now settles:

      1. The benchmark's own declared objective is AGREEMENT WITH THE TEACHER ("A score measures
         agreement with that teacher. It does not measure correctness."). Under that objective the
         theoretical ceiling of an accuracy is 1, and learning the teacher's SYSTEMATIC behaviour is
         the task, not a defect. 0.735 is the teacher's ONE-SAMPLE self-agreement -- a statement
         about label noise, not a bound on what a trained model may legitimately reach. The card
         itself lists models above it (meraGPT 0.768).
      2. The card's warning -- "a score much above 0.75 means a model has learned the teacher's quirks
         rather than the task" -- is an INTERPRETATION with no stated test. The test is constructible:
         gold is the mean of three teacher distributions, so max(target) measures how far the teacher
         agreed with itself on that question. A gain concentrated on the low-agreement items is
         noise-fitting; a gain on the high-agreement items is task-reproduction.

    Run on the available 0.769-0.770 checkpoints, that test found NO material concentration: of the
    +840 correct answers over an untrained head, only 43% sat on low-agreement items while those items
    are 50% of the split, and the high-agreement accuracy reached 0.90. The warning's strong form does
    not hold for these models.

    So this function no longer gates anything. It marks the point at which the mechanism decomposition
    becomes MANDATORY REPORTING alongside the score, because above teacher self-agreement a bare
    accuracy number is uninterpretable on its own -- you must say where the gain came from. Reaching
    above it is a legitimate goal; the requirement is to show the mechanism, not to stop.
    """
    if acc > REF_OBSERVED_SOTA:
        return "above-observed-sota"   # beat a published number; still report the mechanism
    if acc > REF_SATURATION:
        return "above-saturation"     # report the mechanism decomposition
    if acc > ceiling:
        return "above-self-agreement"  # report the mechanism decomposition
    return "below-self-agreement"      # a score alone is informative

# --- time margin, so fluctuation cannot leave the loop with NO viable offspring -----------------
# A budget cut too tight produces a failure mode worse than slow progress: if the host slows down,
# every generation falls under the viability floor, every one is rejected, and the loop spins without
# ever producing a child. That is idling -- running indefinitely while producing no offspring. Three mechanisms:
#
#   (a) a GENEROUS setup allowance. Measured setup is ~15 s (model load + 4.2k-item join + first
#       compile), so a 150 s allowance absorbs a 10x fluctuation before a healthy run is killed.
#   (b) an ADAPTIVE reference. The floor tracks the median of recent HEALTHY realisations instead of a
#       frozen nominal, so a host that is uniformly somewhat slower re-bases rather than failing
#       everything. The reference is not lowered by failures -- only by generations that succeeded.
#   (c) a WATCHDOG. Several independent proposals failing viability back-to-back is evidence about
#       the HOST, not about the proposals (independent bad proposals do not arrive in runs). The loop
#       stops and says so, instead of burning generations silently.
MAX_CONSECUTIVE_NOT_VIABLE = 3


def ceiling_for(time_budget_s: float) -> float:
    """Wall-clock kill line for a generation run with THIS training budget.

    The ceiling must be derived from the budget actually requested, not from the default one. Both
    `train()` and `viability()` previously used the module constant HARD_CEILING_S, which is
    `GENERATION_BUDGET_S + SETUP_ALLOWANCE_S` = 1350 s. That is correct for the loop's own fixed
    budget but wrong for every other caller: the budget-calibration script asks the trainer for a
    1800 s generation, and the constant then kills the child at 1350 s and labels it TIMEOUT -- so the
    calibration could never test any budget above the one already assumed, which is precisely the
    question it exists to answer. The failure was real (observed at 1800 s) and the kill line, not the
    proposal, was the cause.
    """
    return float(time_budget_s) + SETUP_ALLOWANCE_S


def viability(meta: dict | None, wall_s: float,
              ref_steps: int | None,
              time_budget_s: float = GENERATION_BUDGET_S) -> tuple[bool, str, str]:
    """Is this generation a usable measurement?

    Overrunning the budget is EVIDENCE ABOUT THE PROPOSAL, not a resource event: it means the child
    is anomalous, or its optimisation scheme cannot get the job done inside the allotted time. Such a
    generation is rejected on that basis ALONE, and its metric never enters the accept comparison --
    otherwise one lucky noisy draw from a broken proposal reads as an improvement.

    Reason codes mirror the autoresearch container's vocabulary (CRASH / TIMEOUT / WORSE / TIED /
    BREAKS_CONSTRAINT) plus NOT_VIABLE for the step-realisation case.
    """
    ceiling = ceiling_for(time_budget_s)
    if wall_s > ceiling:
        return False, "TIMEOUT", (
            f"wall clock {wall_s:.0f}s exceeded budget+setup allowance {ceiling:.0f}s: the "
            f"child could not complete its generation inside the allotted time")
    if meta is None:
        return False, "CRASH", "no meta.json -- training died before reporting"
    steps = int(meta.get("num_steps") or 0)
    if steps <= 0:
        return False, "CRASH", "zero optimiser steps completed"
    if ref_steps and steps < MIN_STEP_FRACTION * ref_steps:
        return False, "NOT_VIABLE", (
            f"realised {steps} steps against the reference {ref_steps} "
            f"(< {MIN_STEP_FRACTION:.0%}): the scheme cannot get enough optimisation done inside the "
            f"budget, so its metric sits on a different footing and is not comparable")
    return True, "OK", f"{steps} steps, stop_reason={meta.get('stop_reason')}"

# Pre-registered proposal schedule: one intervention per iteration, applied to the INCUMBENT.
# Frozen before execution so no iteration can be chosen after seeing its score.
#
# NOTE the shape change forced by the fixed-time budget: "train longer" is NO LONGER an available
# lever. Under a time budget every generation gets the same wall clock, so an intervention can only
# win by making the OPTIMISATION better, not by making it run longer. That is the container's stated
# purpose, and it is why the old schedule's `steps=3000` / `steps=1500` entries had to be replaced
# rather than rescaled.
#
# WHY `warmup` IS IN HERE, AND WHY THE SCHEDULE-LENGTH AXIS IS NOT.
# A learning-curve measurement produced an accidental but informative contrast: ~2850 steps with the
# cosine stretched over 1200 s scored 0.5750, while ~2968 steps with the cosine over 600 s scored
# 0.5567 -- the same step count, a different schedule length, +1.83 pp apart (above the 1.22 pp dev
# decision SD, but a single seed each, so a hypothesis, not a result; see
# docs/JevRSI_target_backward_derivation.md section 4.5). If it holds, the SCHEDULE rather than the
# step count is the live ingredient.
#   * Within a fixed wall-clock budget the cosine always spans the whole budget, so "longer cosine"
#     is not reachable as a proposal -- only the SHAPE is. `warmup` is exactly that: how much of the
#     fixed budget is spent climbing to peak LR versus decaying from it. Hence its presence here.
#   * A decay-shape axis (cosine vs linear vs WSD) would be the more direct probe, but the trainer
#     exposes no such flag, and adding one changes the training code under test. It is recorded as a
#     candidate, NOT slipped into this frozen schedule after the fact.
#   * `time_budget` is deliberately NOT proposable. It is the invariant that makes generations
#     comparable (claim C1); letting the loop lengthen its own budget would dissolve the fixed-budget
#     premise and re-introduce the step-count confound the budget exists to remove.
SCHEDULE: list[tuple[str, dict]] = [
    ("lr=2e-5",        {"lr": 2e-5}),
    ("head_lr=2e-4",   {"head_lr": 2e-4}),
    ("batch_cases=8",  {"batch_cases": 8}),
    ("warmup=0.05",    {"warmup": 0.05}),
    ("wd=0.01",        {"weight_decay": 0.01}),
]
BASELINE = {"time_budget": GENERATION_BUDGET_S, "max_steps_ceiling": 20000,
            "batch_cases": 4, "lr": 1e-5, "head_lr": 1e-4,
            "warmup": 0.02, "weight_decay": 0.0}
BASE_SEED = 20260921
DEV_N, TEST_N = 600, 2000
B_BOOT = 20000


def run(cmd: list[str], log: Path | None = None,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # The hard ceiling fired: this is not a normal generation, it is a hung or pathological one.
        p = subprocess.CompletedProcess(
            cmd, returncode=124,
            stdout=(e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes)
            else (e.stdout or ""),
            stderr=f"[TIMEOUT] killed after {timeout:.0f}s")
    if log is not None:
        log.write_text(p.stdout + p.stderr, encoding="utf-8")
    return p


def train(seed: int, cfg: dict, out_dir: Path) -> tuple[bool, float, dict | None]:
    """Run one generation. Returns (ok, wall_seconds, meta).

    The hard ceiling is enforced at the SUBPROCESS level, because a proposal that hangs never reaches
    the trainer's own budget check -- and a hung child is exactly the evidence we want recorded.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w = out_dir / "model.safetensors"
    stamp = out_dir / "run_stamp.json"
    meta_p = out_dir / "meta.json"
    want = {"seed": seed} | {k: cfg[k] for k in CONFIG_KEYS}

    if w.exists():
        # A checkpoint may be reused ONLY if THIS config produced it. Reusing on file existence
        # alone means a changed cfg under a stable directory name silently scores the OLD arm -- and
        # the result is then attributed to the new config. Fail loudly instead.
        got = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else None
        if got == want:
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
            print(f"    [reuse] checkpoint matches the requested config")
            return True, 0.0, meta
        print(f"    [STALE] on-disk {got} != requested {want} -- retraining rather than "
              f"misattributing the result")
        w.unlink()

    cmd = [sys.executable, str(MEAS / "train_laya_arm.py"),
           "--seed", str(seed), "--device", "cuda:0",
           "--time-budget", str(cfg["time_budget"]),
           "--max-steps", str(cfg["max_steps_ceiling"]),
           "--batch-cases", str(cfg["batch_cases"]),
           "--lr", str(cfg["lr"]), "--head-lr", str(cfg["head_lr"]),
           "--warmup", str(cfg["warmup"]), "--weight-decay", str(cfg["weight_decay"]),
           "--out-dir", str(out_dir)]
    t0 = time.time()
    # Kill line derived from THIS generation's budget, not from the loop's default. See ceiling_for.
    p = run(cmd, out_dir / "train.log", timeout=ceiling_for(cfg["time_budget"]))
    wall = time.time() - t0
    if p.returncode != 0 or not w.exists():
        print(f"    [FAIL] training exit={p.returncode} after {wall:.0f}s")
        print("      " + "\n      ".join((p.stdout + p.stderr).splitlines()[-5:]))
        return False, wall, None
    stamp.write_text(json.dumps(want, indent=2), encoding="utf-8")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else None
    print(f"    trained in {wall:.0f}s  num_steps={(meta or {}).get('num_steps')}  "
          f"stop={(meta or {}).get('stop_reason')}")
    return True, wall, meta


def evaluate(weights: Path, split: str, out_dir: Path, tag: str) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = out_dir / f"preds_{tag}_{split}.json"
    met = out_dir / f"metrics_{tag}_{split}.json"
    want_fp = fingerprint_str(weights_fingerprint(weights))

    if met.exists():
        # Reuse the numbers ONLY if they were computed from this exact checkpoint. Otherwise a
        # re-trained arm beside a stale metrics file reads as a fresh measurement.
        j = json.loads(met.read_text(encoding="utf-8"))
        got = j.get("weights_fingerprint_str")
        if got == want_fp:
            return j
        print(f"    [STALE] metrics bound to {got} but the checkpoint is {want_fp} "
              f"-- re-evaluating rather than misattributing")

    cmd = [sys.executable, str(MEAS / "eval_laya_split.py"),
           "--weights", str(weights), "--split", split,
           "--out", str(preds), "--metrics-out", str(met),
           "--device", "cuda:0", "--expect-n", str(DEV_N if split == "dev" else TEST_N)]
    p = run(cmd, out_dir / f"eval_{tag}_{split}.log")
    if p.returncode != 0 or not met.exists():
        print(f"    [FAIL] eval({split}) exit={p.returncode}")
        print("      " + "\n      ".join((p.stdout + p.stderr).splitlines()[-5:]))
        return None
    j = json.loads(met.read_text(encoding="utf-8"))
    if j.get("weights_fingerprint_str") != want_fp:
        print(f"    [FAIL] the evaluator did not bind its numbers to the checkpoint "
              f"({j.get('weights_fingerprint_str')} vs {want_fp}) -- refusing to use them")
        return None
    return j


def paired_cluster_se(preds_a: list[dict], preds_b: list[dict], B: int = B_BOOT,
                      seed: int = 20260927) -> dict:
    """Cluster bootstrap on CASES: questions are nested in cases, so a question-level SE is too small."""
    a = {r["id"]: r for r in preds_a}
    b = {r["id"]: r for r in preds_b}
    ids = [k for k in a if k in b]
    ca = np.array([int(np.argmax(a[k]["probs"]) == int(np.argmax(a[k]["target"]))) for k in ids])
    cb = np.array([int(np.argmax(b[k]["probs"]) == int(np.argmax(b[k]["target"]))) for k in ids])
    case = np.array([a[k]["case_id"] for k in ids])
    cases = np.unique(case)
    by_case = {c: np.where(case == c)[0] for c in cases}
    d = cb - ca
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    nc = len(cases)
    for i in range(B):
        pick = rng.integers(0, nc, nc)
        idx = np.concatenate([by_case[cases[j]] for j in pick])
        boots[i] = d[idx].mean()
    return {"delta": float(d.mean()), "se_paired": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_cases": int(nc), "n_questions": int(len(ids))}


def preflight() -> bool:
    """Both gates must pass before any GPU time is spent.

    Kept INSIDE the driver rather than in the launching shell command: gates that live in a shell
    line are bypassed the first time someone runs the driver directly, and the failure is silent.
    """
    ok = True
    for script in ("scripts/check_object_purity.py", "measurement/audit_pipeline.py"):
        p = run([sys.executable, str(REPO / script)])
        tag = "ok  " if p.returncode == 0 else "FAIL"
        print(f"    [{tag}] {script}")
        if p.returncode != 0:
            tail = "\n      ".join((p.stdout + p.stderr).strip().splitlines()[-6:])
            print(f"      {tail}")
            ok = False
    if not ok:
        print("\n[REFUSING TO START] a preflight gate failed -- fix it before burning GPU time.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="measurement/hillclimb")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--iterations", type=int, default=len(SCHEDULE))
    ap.add_argument("--n-decisions", type=int, default=1000,
                    help="planned number of accept decisions over the loop's horizon; tau is "
                         "Bonferroni-corrected over this many. Set it to the loop's real horizon -- "
                         "too large is conservative, but understating it is a validity failure.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="only for re-analysis that spends no GPU time")
    args = ap.parse_args()

    runs = REPO / args.runs
    runs.mkdir(parents=True, exist_ok=True)

    if not args.skip_preflight:
        if not preflight():
            return 1

    print("=" * 78)
    print("JevRSI hill-climb -- selection on DEV, tau measured on DEV, test touched once at the end")
    print("=" * 78)
    print(f"baseline  : {BASELINE}")
    print(f"schedule  : {[s[0] for s in SCHEDULE[:args.iterations]]}")
    print(f"seed      : {args.seed}  (fixed: this loop climbs training CONFIG, not seeds)")
    if args.dry_run:
        for i, (name, delta) in enumerate(SCHEDULE[:args.iterations], 1):
            cfg = dict(BASELINE) | delta
            print(f"  it{i}: {name:16s} -> {cfg}")
        print("\n[dry-run] nothing executed.")
        return 0

    # ---- baseline generation ---------------------------------------------------------------
    print("\n[baseline] training the pre-registered config on train, scoring on dev ...")
    base_dir = runs / "it0_baseline"
    ok, wall, bmeta = train(args.seed, BASELINE, base_dir)
    okv, code, why = viability(bmeta, wall, None, time_budget_s=cfg["time_budget"])
    if not ok or not okv:
        print(f"    [FAIL] the baseline generation is not viable ({code}): {why}")
        return 1
    # The baseline CALIBRATES the host: everything else is judged against what this machine achieved
    # on the reference config inside the same budget.
    ref_steps = int((bmeta or {}).get("num_steps") or 0)
    print(f"    reference realisation: {ref_steps} steps in {wall:.0f}s "
          f"({ref_steps / max(wall, 1):.2f} steps/s)  -> a generation must reach "
          f"≥ {MIN_STEP_FRACTION:.0%} of this to be comparable")
    m0 = evaluate(base_dir / "model.safetensors", "dev", base_dir, "it0")
    if m0 is None:
        return 1
    print(f"    dev acc = {m0['accuracy']:.4f}   skill = {m0['chance_corrected_skill']:.4f}")

    # ---- tau for DEV ----------------------------------------------------------------------
    # Paired against the UNTRAINED arm so the SE reflects the dev split's own question mix.
    ub = runs / "untrained"
    ub.mkdir(parents=True, exist_ok=True)
    mu = evaluate(BASE_CKPT / "model.safetensors", "dev", ub, "untrained")
    if mu is None:
        return 1
    pa = json.loads((runs / "it0_baseline" / "preds_it0_dev.json").read_text(encoding="utf-8"))
    pb = json.loads((ub / "preds_untrained_dev.json").read_text(encoding="utf-8"))
    sev = paired_cluster_se(pb, pa)

    # Floor B is a property of the TRAINER, measured on test at 0.40 pp; it is reused here.
    floor_b = json.loads((MEAS / "floor_b_laya_runs" / "floor_b_laya.json").read_text(
        encoding="utf-8"))["floor_b"]
    # tau MUST come from the single source of truth (measurement/tau_calibration.py), not from a
    # hard-coded z. The driver previously wrote `1.96 *` here, which is the SINGLE-decision factor --
    # so the loop's accept rule silently assumed it would make exactly one decision, whatever its
    # actual horizon. tau_calibration states the opposite requirement in its own header: "A
    # per-decision alpha of 0.025 reaches a family-wise rate near 1 within a few hundred decisions",
    # and lists N = 1000 -> 4.95 pp as "the defensible setting". Two files that disagree about the
    # accept threshold is exactly the class of defect the audit pipeline exists to catch, and it was
    # not caught because the audit checks tau_calibration rather than the driver's use of it.
    #
    # WHAT THIS CHANGES, honestly: applying the horizon correction RAISES tau (more conservative), so
    # it cannot manufacture accepts -- it removes the possibility of an invalid one. It also makes the
    # loop's power problem explicit rather than hidden: at the measured dev SE, tau_dev is already
    # 4.95 pp at H = 1, and 9.98 pp at H = 1000, against a measured single-intervention effect of
    # 1.83 pp. The loop therefore cannot adjudicate micro-interventions at this sample size, and
    # saying so is the point -- see docs/JevRSI_decision_record_2026-09-28.md.
    from tau_calibration import tau as tau_from_source
    n_dec = max(1, int(args.n_decisions))
    se_dev_combined = float(np.sqrt(sev["se_paired"] ** 2 + 2 * floor_b ** 2))
    tau_dev = tau_from_source(se_dev_combined, n_dec)
    se_test = json.loads((MEAS / "floor_b_laya_runs" / "floor_b_laya.json").read_text(
        encoding="utf-8"))["paired_vs_base"]["se_paired"]
    se_test_combined = float(np.sqrt(se_test ** 2 + 2 * floor_b ** 2))
    tau_test = tau_from_source(se_test_combined, n_dec)
    print(f"\n[tau] source = tau_calibration.tau(se, n_decisions={n_dec})  "
          f"(horizon-corrected; the driver no longer hard-codes a z)")
    print(f"      a single-decision z=1.96 would give tau_dev "
          f"{tau_from_source(se_dev_combined, None)*100:.2f} pp / tau_test "
          f"{tau_from_source(se_test_combined, None)*100:.2f} pp")

    print(f"\n[splits are not interchangeable]")
    print(f"    dev : {sev['n_questions']} questions / {sev['n_cases']} cases  -> "
          f"SE_paired {sev['se_paired']*100:.2f} pp  ->  tau_dev  {tau_dev*100:.2f} pp")
    print(f"    test: {TEST_N} questions / 400 cases               -> "
          f"SE_paired {se_test*100:.2f} pp  ->  tau_test {tau_test*100:.2f} pp")
    print(f"    using tau_test for a dev decision would accept "
          f"{(tau_dev/tau_test-1)*100:.0f}% more noise than the split allows")

    # ---- the climb ------------------------------------------------------------------------
    history = []
    incumbent_cfg = dict(BASELINE)
    incumbent_acc = m0["accuracy"]
    # The incumbent's DIRECTORY, not just its config: the loop carries the config forward, but one
    # checkpoint of the current best must survive so the frozen test split can be touched once at the
    # end. Everything else on disk is released as soon as its row is recorded.
    incumbent_dir = base_dir
    # (b) adaptive reference: the median of recent HEALTHY realisations. Only successes extend it, so
    # a degraded host cannot lower its own bar into accepting crippled generations.
    healthy_steps: list[int] = [ref_steps]
    consec_bad = 0
    print(f"\n[climb] accepting iff delta(dev) > tau_dev = {tau_dev*100:.2f} pp")
    print(f"  {'it':>3} {'proposal':16s} {'dev acc':>8} {'delta':>8} {'verdict':>9} {'best':>8}")

    for i, (name, delta) in enumerate(SCHEDULE[:args.iterations], 1):
        cfg = dict(incumbent_cfg) | delta
        d = runs / f"it{i}_{name.replace('=', '')}"
        ref_now = int(np.median(healthy_steps[-5:]))
        print(f"  {i:>3} {name:16s}", end="", flush=True)
        ok, wall, meta = train(args.seed, cfg, d)
        okv, code, why = viability(meta, wall, ref_now, time_budget_s=cfg["time_budget"])
        if not okv:
            # A generation that cannot complete its budget is EVIDENCE about the proposal -- the child
            # is anomalous or its scheme is not viable. Rejected on that basis ALONE, and its metric
            # never enters the accept comparison, so a lucky noisy draw from a broken proposal cannot
            # be read as an improvement.
            consec_bad += 1
            history.append({"it": i, "proposal": name, "config": cfg, "dev_acc": None,
                            "delta": None, "accepted": False, "best": incumbent_acc,
                            "failed": True, "reason_code": code, "reason": why,
                            "wall_s": wall, "num_steps": (meta or {}).get("num_steps")})
            print(f" [{code}] {why[:60]}")
            # (c) watchdog: independent proposals do not fail in runs, so consecutive failures point
            # at the host. Stop and report rather than spinning with no viable offspring.
            if consec_bad >= MAX_CONSECUTIVE_NOT_VIABLE:
                print(f"\n[STOP -- HOST DEGRADED] {consec_bad} consecutive non-viable generations. "
                      f"{consec_bad} independent proposals failing in a row is evidence about the "
                      f"HOST, not the proposals. Halting instead of producing no offspring.")
                (runs / "host_degraded.json").write_text(json.dumps({
                    "consecutive_not_viable": consec_bad,
                    "reference_steps_median": ref_now,
                    "healthy_realisations": healthy_steps,
                    "note": ("The loop stopped because it could not produce a viable child. Check for "
                             "competing GPU/CPU load on the host, then resume -- completed generations "
                             "are cached and will be reused."),
                }, indent=2), encoding="utf-8")
                return 2
            continue
        consec_bad = 0
        steps_now = int((meta or {}).get("num_steps") or 0)
        if steps_now:
            healthy_steps.append(steps_now)
        m = evaluate(d / "model.safetensors", "dev", d, f"it{i}")
        if m is None:
            history.append({"it": i, "proposal": name, "config": cfg,
                            "dev_acc": None, "delta": None, "accepted": False,
                            "best": incumbent_acc, "failed": True,
                            "reason_code": "CRASH",
                            "reason": "the dev evaluation failed to produce metrics",
                            "wall_s": wall, "num_steps": (meta or {}).get("num_steps")})
            print(f" [CRASH] dev evaluation produced no metrics")
            # Record exists -> the weights are disposable. See release_generation_weights.
            release_generation_weights(d, keep=False)
            continue
        acc = m["accuracy"]
        dd = acc - incumbent_acc
        accepted = dd > tau_dev
        # The published meaning boundary is checked on EVERY generation, not only on accepts: a
        # generation can sit in the quirks zone while being rejected on power, and that is also
        # worth knowing -- it means the config space itself reaches into the zone.
        zone = meaning_zone(acc)
        superseded = None
        if accepted:
            superseded = incumbent_dir if incumbent_dir != d else None
            incumbent_cfg, incumbent_acc, incumbent_dir = cfg, acc, d
        history.append({"it": i, "proposal": name, "config": cfg, "dev_acc": acc,
                        "delta": dd, "accepted": bool(accepted), "best": incumbent_acc,
                        "meaning_zone": zone,
                        "quirk_warning": bool(accepted and zone != "task")})
        mark = "" if zone == "below-self-agreement" else f"   <-- {zone}: report mechanism"
        print(f" {acc:>8.4f} {dd*100:>+7.2f}p {'ACCEPT' if accepted else 'reject':>9} "
              f"{incumbent_acc:>8.4f}{mark}")
        # RECORD FIRST, THEN RELEASE -- the container's ordering applied to weights instead of a
        # branch (program.md: "persist the attempt first, then git reset"). The row is in `history`
        # now, so deleting weights cannot lose the record.
        #   * a superseded incumbent is released: the child replaces it, the loop never needs it
        #     again, and keeping it would make the footprint grow with the number of accepts;
        #   * the winner is kept, because the frozen test split is touched once at the end and needs
        #     the current best's weights;
        #   * a rejected generation is released immediately -- its config and metrics are its record.
        # Net footprint is therefore CONSTANT (current best + in-flight), not a function of runtime.
        if superseded is not None:
            release_generation_weights(superseded, keep=False)
        release_generation_weights(d, keep=bool(accepted))

    n_acc = sum(1 for h in history if h.get("accepted"))
    n_quirk = sum(1 for h in history if h.get("quirk_warning"))
    n_zone = sum(1 for h in history if h.get("meaning_zone") not in (None, "below-self-agreement"))
    best_cfg = incumbent_cfg

    # A bare accuracy above teacher self-agreement is uninterpretable on its own: it must be
    # accompanied by WHERE the gain came from. This is a reporting requirement, not a prohibition --
    # reaching this region is a legitimate goal, and the mechanism test run on the available 0.769 /
    # 0.770 checkpoints found the gain broadly distributed rather than concentrated on
    # teacher-disagreement items (measurement/mechanism_test.py, measurement/mechanism_test.json).
    if n_zone:
        print(f"\n[meaning boundary] {n_zone}/{len(history)} generations scored above teacher "
              f"self-agreement ({REF_TEACHER_SELFAGREE}); {n_quirk} of them were ACCEPTED there.")
        print("    -> mandatory: report the mechanism decomposition alongside the score "
              "(measurement/mechanism_test.py)")
        print("       a gain concentrated on LOW-agreement items is teacher-noise fitting; on "
              "HIGH-agreement items it is task-reproduction")
    else:
        print(f"\n[meaning boundary] every scored generation stayed at or below teacher "
              f"self-agreement ({REF_TEACHER_SELFAGREE})")

    # One set of measurements, two rules. The naive chain takes the running max of dev accuracy --
    # what "accept if it scores better" produces -- so it climbs by construction and can never fall.
    # The tau chain moves only on an accept that clears the floor. Their divergence IS the figure.
    naive_chain, tau_chain = [m0["accuracy"]], [m0["accuracy"]]
    nb = tb = m0["accuracy"]
    for h in history:
        a = h.get("dev_acc")
        if a is None:
            naive_chain.append(nb)
            tau_chain.append(tb)
            continue
        nb = max(nb, a)
        if h.get("accepted"):
            tb = a
        naive_chain.append(nb)
        tau_chain.append(tb)
    print(f"\n[one dataset, two rules]")
    print(f"    naive 'accept if better' chain: {m0['accuracy']:.4f} -> {nb:.4f} "
          f"(+{(nb - m0['accuracy']) * 100:.2f} pp)")
    print(f"    tau-gated chain               : {m0['accuracy']:.4f} -> {tb:.4f} "
          f"(+{(tb - m0['accuracy']) * 100:.2f} pp)")

    # ---- the frozen test split, ONCE ------------------------------------------------------
    print(f"\n[test] evaluating the surviving chain on the frozen test split (once) ...")
    # Evaluate the checkpoint that EARNED the accept -- NEVER a retrain of its config. The trainer is
    # not deterministic given a seed (five same-seed runs spread 0.70 pp on test), so a retrain would
    # silently score a DIFFERENT arm than the rule selected, and the reported endpoint would not be
    # the arm whose dev score justified the accept. Evaluating in place also removes a ~7 min retrain.
    acc_hist = [h for h in history if h.get("accepted")]
    if acc_hist:
        last = acc_hist[-1]
        final_dir = runs / f"it{last['it']}_{last['proposal'].replace('=', '')}"
        print(f"    source: {final_dir.name}  (the checkpoint that earned the accept, not retrained)")
    else:
        final_dir = runs / "it0_baseline"
        print("    source: it0_baseline  (no proposal cleared tau_dev)")
    mt = evaluate(final_dir / "model.safetensors", "test", final_dir, "final")
    if mt is None:
        return 1
    test_base = float(json.loads(
        (MEAS / "floor_b_laya_runs" / "floor_b_laya.json").read_text(encoding="utf-8")
    )["base"]["accuracy"])
    print(f"    test acc = {mt['accuracy']:.4f}   (untrained {test_base:.4f})   "
          f"tau_test = {tau_test*100:.2f} pp")

    out = {
        "baseline": BASELINE, "seed": args.seed, "schedule": [s[0] for s in SCHEDULE],
        "tau_dev": tau_dev, "tau_test": tau_test, "floor_b": floor_b,
        "dev_se_paired": sev["se_paired"], "test_se_paired": se_test,
        "baseline_dev": m0["accuracy"], "best_config": best_cfg,
        "best_dev": incumbent_acc,
        "n_accepted": n_acc, "n_iterations": len(history),
        "naive_chain": naive_chain, "tau_chain": tau_chain,
        "naive_end_dev": nb, "tau_end_dev": tb,
        "history": history,
        "test": {"accuracy": mt["accuracy"], "chance_corrected_skill": mt["chance_corrected_skill"],
                 "untrained": test_base},
        "note": ("Proposals are a fixed pre-registered schedule; selection is on dev; tau is "
                 "measured on dev; the frozen test split is scored once at the end."),
    }
    (runs / "hillclimb.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"  accepted {n_acc}/{len(history)} proposals")
    print(f"  dev  : {m0['accuracy']:.4f} -> {incumbent_acc:.4f}")
    print(f"  test : {test_base:.4f} -> {mt['accuracy']:.4f}  (single final measurement)")
    print(f"  -> {runs/'hillclimb.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
