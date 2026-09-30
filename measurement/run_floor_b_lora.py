"""run_floor_b_lora.py -- driver for the pre-registered Floor B on the LoRA seed.

WHAT THIS MEASURES. Floor B is the standard deviation, across runs differing ONLY in random seed, of the
final dev score under one fixed recipe. It is the quantity that sets epsilon, and the only Floor B this
project currently has (0.40 pp) was measured on a different training path, which
measurement/INSTRUMENT_CALIBRATION.json explicitly forbids inheriting silently. Protocol, recipe, decision
rules and falsification conditions are frozen in docs/prereg_floor_b_lora_2026-09-29.md, written before any
replicate of this experiment existed.

DESIGN POINTS THAT ARE NOT OBVIOUS, each of which is a lesson learned the hard way in this project:

  * EVALUATION RUNS IN A SUBPROCESS. An earlier driver held a GPU-resident model while spawning the training
    child, which squeezed the child into an allocator stall: the child sat at step 1 for 632 s with the GPU at
    100% and the log mtime frozen. The parent must not hold VRAM beside the child. Same for evaluation here.

  * EVALUATION IS BATCHED AFTER ALL TRAINING. Same reason. train-all-then-evaluate-all, so no phase ever has
    two GPU residents.

  * IT IS RESUMABLE, AND IT VERIFIES THE SUBJECT MOUNT FIRST. A previous driver died twice on infrastructure:
    once when a USB SSD disconnected mid-batch (ModuleNotFoundError: evaluate_laya_variant), and once when the
    agent runtime rotated its own site-packages out from under a running process (ModuleNotFoundError: laya).
    Both were invisible to the driver until the next replicate. So each replicate's metrics must already exist
    on disk before it is re-run, and the mount is checked up front with bounded retries.

  * A MERGED-CHECKPOINT ACCEPTANCE TEST RUNS FIRST, ON TWO STEPS. The LoRA arm exports a checkpoint with the
    adapter folded into the frozen base so the frozen evaluator can load it strict=True. If that merge is wrong,
    every replicate trains for two hours and then fails at evaluation. Two steps and one evaluation, ~3 minutes,
    converts a 21-hour silent failure into a 3-minute loud one.

Run: python measurement/run_floor_b_lora.py --arm bf16
     python measurement/run_floor_b_lora.py --arm 4bit
"""

# RETIRED OBJECT DECLARATION
# A retired 322M full-parameter arm is a retired research object (2026-09-30). It is named below
# only in a comment recording a MODULE-NOT-FOUND crash caused by this agent's own site-packages
# rotation. Do not work on this object.
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md. This file sits one level below the
# root, so the root is parents[1]; parents[2] would be the root's parent, and every artifact
# read through it would come from a neighbouring directory.
import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The subject repository is an EXTERNAL clone, so no path to it is portable and none can be
# derived from __file__. config/paths.json declares it and paths.py raises rather than falling back
# silently, so a run cannot attribute numbers to a checkpoint it never loaded.
from paths import require, subject                                              # noqa: E402

AUTORESEARCH = Path(str(ROOT))
SUBJECT = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
PY = sys.executable
CONFIG = "agentjev/configs/autoresearch.yaml"
EVALUATOR = "scripts/evaluate_split.py"
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
TEMPERATURES = SUBJECT / "checkpoints" / "autoresearch" / "temperatures.json"
RUNS = AUTORESEARCH / "measurement" / "floor_b_lora_runs"

# The recipe, fixed by the pre-registration. batch_states=1 with grad_accum=16 keeps the EFFECTIVE batch at
# the protocol's 16 while cutting peak activation memory: measured 2.24 GiB against a 12.28 GiB card, versus
# 11.8-11.9 GiB and >45 s/step at the config's batch_states=4 without gradient checkpointing.
BATCH_STATES = 1
GRAD_ACCUM = 16
SEEDS = [20260921, 20260922, 20260923, 20260924, 20260925]

# Keeps a child's console from taking the parent's down with it, and is what makes taskkill /T able to
# address the child as a tree root. 0x200 is CREATE_NEW_GROUP on Windows; the getattr keeps this
# importable on a POSIX host where the attribute does not exist.
CREATE_NEW_GROUP = getattr(subprocess, "CREATE_NEW_GROUP", 0x00000200)


def sh(cmd: list[str], timeout: int, cwd: Path = SUBJECT) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env=env, errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def subject_present(attempts: int = 5, wait: int = 10) -> bool:
    """The subject lives on a removable volume. Confirm it is mounted before doing hours of work."""
    for i in range(attempts):
        if (SUBJECT / "agentjev" / "train.py").exists() and SEED_CKPT.exists():
            if i:
                print(f"  subject mount recovered after {i} retries")
            return True
        print(f"  [warn] subject not visible ({SUBJECT}); retry {i+1}/{attempts} in {wait}s")
        time.sleep(wait)
    return False


def parse_eval_metrics(text: str) -> dict | None:
    """Pull the frozen evaluator's own numbers out of its stdout.

    A missing field is stored as None rather than skipped, so a partially-parsed evaluation is visible in
    the artifact instead of silently looking like a smaller dict.
    """
    out: dict[str, float | None] = {}

    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    out["accuracy"] = grab(r"Accuracy:\s+([0-9.]+)")
    out["chance_corrected"] = grab(r"Chance-corrected:\s+([0-9.]+)")
    out["soft_cross_entropy"] = grab(r"Soft Cross-Entropy:\s+([0-9.]+)")
    out["brier_sum"] = grab(r"Brier Score:\s+([0-9.]+)")
    out["ece"] = grab(r"ECE:\s+([0-9.]+)")
    out["score_mae"] = grab(r"Score MAE:\s+([0-9.]+)")
    for t in ("boolean", "choice", "score"):
        m = re.search(rf"{t}: acc=([0-9.]+)", text)
        out[f"acc_{t}"] = float(m.group(1)) if m else None
    return out if out.get("accuracy") is not None else None


_CHILD_PID: int | None = None       # the training child currently being supervised


def _kill_child_tree(pid: int) -> None:
    """Kill a training child AND anything it spawned.

    taskkill /T rather than a plain kill: the child is a `python -m agentjev.train` that may itself
    have helpers, and a half-killed tree leaves a process still holding VRAM.

    This exists because of a concrete failure. `timeout 3600` sent SIGTERM to the driver, the driver
    died, and its `agentjev.train` grandchild SURVIVED -- still holding 4.4 GB and 100% of the GPU. The
    next replicate then ran at 159 tok/s instead of 1450, a 9x slowdown that looked like a convergence
    problem and was nothing of the kind. A 21-hour batch of ten replicates would have accumulated one
    such orphan per interruption.
    """
    r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  [warn] taskkill rc={r.returncode} for pid {pid}: {r.stdout.strip()[:100]}")


def _install_signal_handlers() -> None:
    """On SIGTERM/SIGINT, take the training child down before this process exits.

    Without this the handler is the *only* thing standing between an interrupted driver and an orphan
    holding the GPU, because the signal that kills the driver does not reach the grandchild on Windows.
    """
    import signal

    def handler(signum, _frame):
        if _CHILD_PID is not None:
            print(f"\n[driver] signal {signum}: taking down training child {_CHILD_PID} first")
            _kill_child_tree(_CHILD_PID)
        raise SystemExit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, handler)
        except (ValueError, OSError):
            pass        # not the main thread, or the platform lacks it; the guard below still applies


def train_one(arm: str, seed: int, steps: int) -> Path:
    global _CHILD_PID
    out_dir = RUNS / arm / f"seed{seed}"
    log = out_dir / "train.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "agentjev.train", "--config", CONFIG,
           "--lora", "--lora-r", "16",
           "--seed", str(seed), "--max-steps", str(steps),
           "--batch-states", str(BATCH_STATES), "--grad-accum", str(GRAD_ACCUM),
           "--save-every-steps", "25",
           # 'auto' resumes from the newest snapshot in out_dir when one exists. Exactness is not
           # assumed: measurement/test_resume_exact.py kills a run mid-flight and requires the resumed
           # steps to match an uninterrupted run exactly. If that test ever fails, this must become
           # "off" and the interrupted replicate must be re-run from step 0.
           "--resume", "auto",
           "--out-dir", str(out_dir)]
    if arm == "4bit":
        cmd.append("--load-4bit")
    print(f"  train {arm} seed={seed} steps={steps} bs={BATCH_STATES} ga={GRAD_ACCUM}", flush=True)
    # Note the log APPEND on resume. A resumed leg writes its own header and step lines; overwriting
    # would destroy the record of how far the first leg got, which is the only evidence of the
    # interruption that floor_b_lora.json later reports.
    fresh = not (out_dir / "train.log").exists() or not list(out_dir.glob("checkpoint-*.pt"))
    t0 = time.time()
    with log.open("w" if fresh else "a", encoding="utf-8") as fh:
        if not fresh:
            fh.write(f"\n===== RESUMED LEG {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            fh.flush()
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.Popen(cmd, cwd=SUBJECT, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, errors="replace", creationflags=CREATE_NEW_GROUP)
        _CHILD_PID = proc.pid
        try:
            rc = proc.wait()
        except BaseException:
            # Covers KeyboardInterrupt, the handler above, and any unexpected exception. The child is
            # killed on every one of those paths, because a leaked child is the failure mode that costs
            # the most time and is the hardest to notice.
            print(f"  [driver] driver interrupted; killing training child {proc.pid}")
            _kill_child_tree(proc.pid)
            raise
        finally:
            _CHILD_PID = None
    # Count resume legs. This is the provenance the pre-registration's disclosure obligation asks for:
    # a reader must be able to see which replicates were interrupted and how many times.
    n_resumes = sum(1 for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                    if l.startswith("===== RESUMED LEG"))
    (out_dir / "resumes.json").write_text(json.dumps(
        {"arm": arm, "seed": seed, "resume_legs": n_resumes,
         "interrupted": n_resumes > 0,
         "note": "n_resumes counts RESUMED LEG headers in train.log; 0 means the replicate ran "
                 "straight through. Resume is bit-exact per measurement/test_resume_exact.py."},
        indent=2), encoding="utf-8")
    print(f"    train rc={rc} in {(time.time()-t0)/60:.1f} min | resume_legs={n_resumes}", flush=True)
    if rc != 0:
        print(f"    [FAIL] training failed; tail of {log.name}:")
        print("      " + "\n      ".join(log.read_text(encoding='utf-8', errors='replace')
                                          .splitlines()[-12:]))
    return out_dir


def prune_checkpoint(d: Path, keep: bool = False) -> None:
    """Delete a replicate's weights once its metrics are on disk.

    Each exported checkpoint is 2.3 GB (the config builds the model in fp32, so 608.5M params x 4 B). Ten
    replicates would be 23 GB of weights whose only purpose was to produce six numbers apiece, and the
    numbers are already written. Metrics files, training logs and the summary are the evidence; the
    weights are not. `--keep-weights` retains them when a run needs to be re-evaluated by hand.
    """
    if keep:
        return
    freed = 0
    for name in ("final.pt", "final.safetensors"):
        f = d / name
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    if freed:
        print(f"    pruned {freed/1e9:.2f} GB of weights (metrics + train.log retained)")


def evaluate_one(arm: str, seed: int, out_dir: Path, keep_weights: bool = False) -> dict | None:
    """Evaluate the exported checkpoint with the FROZEN evaluator, in a subprocess.

    A subprocess, because a parent holding a GPU-resident model squeezes the child into an allocator
    stall -- observed as a child frozen at step 1 for 632 s with the GPU at 100%.
    """
    ckpt = out_dir / "final.safetensors"
    mfile = out_dir / "metrics.json"
    if mfile.exists():
        print(f"  eval {arm} seed={seed}: reusing {mfile.name}")
        return json.loads(mfile.read_text(encoding="utf-8"))
    if not ckpt.exists():
        print(f"  [FAIL] {arm} seed={seed}: no final.safetensors to evaluate")
        return None
    rc, out = sh([PY, EVALUATOR, "--checkpoint", str(ckpt), "--split", "dev",
                  "--temperatures", str(TEMPERATURES)], timeout=1800)
    m = parse_eval_metrics(out)
    if rc != 0 or m is None:
        print(f"  [FAIL] eval {arm} seed={seed} rc={rc}")
        print("      " + "\n      ".join(out.splitlines()[-12:]))
        return None
    m.update({"arm": arm, "seed": seed, "checkpoint": str(ckpt)})
    mfile.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"  eval {arm} seed={seed}: acc={m['accuracy']:.4f} skill={m['chance_corrected']:.4f} "
          f"ce={m['soft_cross_entropy']:.4f}", flush=True)
    prune_checkpoint(out_dir, keep=keep_weights)
    return m


def acceptance_test() -> bool:
    """Two steps, one evaluation. Proves the merged LoRA checkpoint loads into the frozen evaluator.

    Uses seed 0 so it NEVER conflicts with the real SEEDS list (which starts at 20260921). An earlier
    version accidentally ran seed 20260921 -- it cleaned up its metrics.json, so the main loop saw it as
    missing and retrained the same seed, wasting another two minutes. The fix is simple: use a seed that
    does not appear in SEEDS, and remove the entire directory after the test so it cannot linger.
    """
    print("=" * 74)
    print("ACCEPTANCE TEST: does a merged LoRA checkpoint load into the FROZEN evaluator?")
    print("=" * 74)
    arm, seed = "bf16", 0  # seed 0 not in SEEDS; never conflicts with the real loop
    d = RUNS / arm / f"seed{seed}"
    if d.exists():
        shutil.rmtree(d)
    train_one(arm, seed, steps=2)
    m = evaluate_one(arm, seed, d)
    if m is None:
        print("\n[FAIL] acceptance test failed -- the export cannot be evaluated. Refusing to start "
              "21 hours of replicates against a checkpoint the evaluator cannot load.")
        return False
    print(f"\n[OK] merged checkpoint evaluated: acc={m['accuracy']} (2 steps, so the VALUE is meaningless; "
          f"the LOAD is the test)")
    # Remove the ENTIRE directory (not just prune files) so it cannot linger and the main loop
    # never mistakes it for an incomplete replicate.
    shutil.rmtree(d, ignore_errors=True)
    print("     pruned the entire acceptance directory so it cannot interfere with the real loop")
    return True


def summarise(arm: str) -> dict:
    ms = []
    for s in SEEDS:
        f = RUNS / arm / f"seed{s}" / "metrics.json"
        if f.exists():
            ms.append(json.loads(f.read_text(encoding="utf-8")))
    if len(ms) < 2:
        return {"arm": arm, "n": len(ms), "note": "not enough replicates to estimate a dispersion"}
    accs = [m["accuracy"] for m in ms]
    ces = [m["soft_cross_entropy"] for m in ms]
    n = len(accs)
    mean = sum(accs) / n
    sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (n - 1))
    # chi-square interval for a variance at df = n-1; the honest width of "how well do we know Floor B"
    df = n - 1
    lo = (df * sd ** 2) / _chi2_ppf(0.975, df) if sd > 0 else 0.0
    hi = (df * sd ** 2) / _chi2_ppf(0.025, df) if sd > 0 else float("inf")
    return {"arm": arm, "n": n, "df": df, "seeds": [m["seed"] for m in ms],
            "accuracies": accs, "mean_accuracy": round(mean, 6),
            "sd_pp": round(sd * 100, 4), "sd_pp_ci95": [round(math.sqrt(lo) * 100, 4),
                                                         round(math.sqrt(hi) * 100, 4)],
            "soft_ce_mean": round(sum(ces) / n, 6),
            # Provenance, per the pre-registration's disclosure obligation. A replicate that was
            # interrupted and continued is still one draw from the recipe (resume is bit-exact, proven
            # by measurement/test_resume_exact.py), but a reader is entitled to see which ones were.
            "provenance": {str(m["seed"]): (
                json.loads((RUNS / arm / f"seed{m['seed']}" / "resumes.json")
                           .read_text(encoding="utf-8"))["resume_legs"]
                if (RUNS / arm / f"seed{m['seed']}" / "resumes.json").exists() else None)
                for m in ms},
            "n_interrupted": sum(
                1 for m in ms
                if (RUNS / arm / f"seed{m['seed']}" / "resumes.json").exists()
                and json.loads((RUNS / arm / f"seed{m['seed']}" / "resumes.json")
                               .read_text(encoding="utf-8"))["interrupted"]),
            "note": "df is stated because a pilot once overstated a floor by 3.53x at df=1. "
                    "provenance maps seed -> resume_legs (null = record absent)."}


def _chi2_ppf(p: float, df: int) -> float:
    """Chi-square upper quantile, Wilson-Hilferty. Adequate for the df=4 case here."""
    z = _norm_ppf(p)
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def _norm_ppf(p: float) -> float:
    """Acklam-style rational approximation of the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["bf16", "4bit", "both"], default="both")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--skip-acceptance", action="store_true")
    ap.add_argument("--keep-weights", action="store_true",
                    help="keep each 2.3 GB checkpoint after its metrics are written. Off by "
                         "default: 10 replicates would otherwise cost 23 GB of weights whose "
                         "only product is six numbers already on disk.")
    ap.add_argument("--out", default=str(AUTORESEARCH / "measurement" / "floor_b_lora.json"))
    a = ap.parse_args()

    if not subject_present():
        print("[FAIL] subject volume not mounted. Refusing to start.")
        return 2
    RUNS.mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    if not a.skip_acceptance and not (RUNS / "_acceptance" / "metrics.json").exists():
        if not acceptance_test():
            return 1

    arms = ["bf16", "4bit"] if a.arm == "both" else [a.arm]
    results = {}
    for arm in arms:
        print(f"\n{'='*74}\nARM {arm}\n{'='*74}", flush=True)
        if arm == "4bit":
            rc, out = sh([PY, "-c", "import bitsandbytes, torch; "
                                  "w=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); "
                                  "q,s=bitsandbytes.functional.quantize_4bit(w,quant_type='nf4'); "
                                  "bitsandbytes.functional.dequantize_4bit(q,s); print('kernels ok')"],
                         timeout=600)
            if rc != 0 or "kernels ok" not in out:
                print("[SKIP] 4-bit kernels unavailable on this host. The pre-registration's falsification "
                      "condition 1 applies: the secondary question is reported as NOT ANSWERABLE HERE and "
                      "is not substituted with a proxy.")
                results["4bit"] = {"status": "not_answerable", "reason": out.strip()[-300:]}
                continue
        for seed in a.seeds:
            if not subject_present():
                print("[FAIL] subject volume disappeared mid-batch; stopping. Completed replicates are on "
                      "disk and re-running resumes from them.")
                break
            d = RUNS / arm / f"seed{seed}"
            if (d / "metrics.json").exists():
                print(f"  {arm} seed={seed}: already done, skipping")
                continue
            train_one(arm, seed, a.steps)
            evaluate_one(arm, seed, d, keep_weights=a.keep_weights)
        results[arm] = summarise(arm)

    print(f"\n{'='*74}\nSUMMARY\n{'='*74}")
    for arm, r in results.items():
        if r.get("status") == "not_answerable":
            print(f"  {arm:6s} NOT ANSWERABLE ({r['reason'][-90:]})")
        elif "sd_pp" in r:
            print(f"  {arm:6s} n={r['n']} df={r['df']} FloorB(acc) = {r['sd_pp']:.3f} pp "
                  f"95% CI {r['sd_pp_ci95']} | mean acc {r['mean_accuracy']:.4f}")
            print(f"         interrupted replicates: {r['n_interrupted']}/{r['n']} "
                  f"(resume is bit-exact; provenance {r['provenance']})")
        else:
            print(f"  {arm:6s} {r}")
    if "bf16" in results and "sd_pp" in results.get("bf16", {}):
        b, q = results["bf16"], results.get("4bit", {})
        if "sd_pp" in q:
            print(f"\n  4-bit minus BF16 Floor B: {q['sd_pp'] - b['sd_pp']:+.3f} pp")
            print("  The pre-registration's burden of proof is on 4-bit: BF16 already fits the card, so "
                  "quantisation must not inflate the floor beyond the usable headroom to be adopted.")
    Path(a.out).write_text(json.dumps({
        "_what_this_is": "Floor B on the LoRA seed, per docs/prereg_floor_b_lora_2026-09-29.md",
        "prereg": "docs/prereg_floor_b_lora_2026-09-29.md",
        "recipe": {"lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                   "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   "steps": a.steps, "batch_states": BATCH_STATES, "grad_accum": GRAD_ACCUM,
                   "effective_batch": BATCH_STATES * GRAD_ACCUM,
                   "grad_checkpoint": True, "base_dtype": "bf16",
                   "seeds": a.seeds, "eval": "frozen evaluator, dev, L1 temperatures"},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())