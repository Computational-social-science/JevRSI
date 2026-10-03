#!/usr/bin/env python
"""health.py -- the diagnostic chain for a 24/7 RSI pipeline.

WHY THIS EXISTS
    Every failure this project has had was a number or a configuration that no mechanical check was
    watching:

      * a spec reconstructed by hand, wrong in two load-bearing keys (lr_head 10x, option_order
        absent so the run would have trained a position-collapsed head)
      * an OOM read off a run sharing the GPU with another job, then used to justify a deviation
      * a step time of 23 s/step that no run was behind, sizing a job as 9.6 h when it is 1.4 h
      * an audit claiming "byte-identical" from a comparison that had normalised away the difference
      * a watcher reporting NOT RUNNING during the preparation phase of a healthy run

    In each case the cost was GPU-hours and agent tokens spent on a result that could not be used,
    and in each case the fix was not "be more careful" -- it was a check that fails loudly. That is
    what this file is. Running the pipeline without it means paying for the same errors again.

    The chain is deliberately one entry point. `launch` runs `preflight` itself and refuses to start
    if it fails, so the gate cannot be skipped by forgetting to run it.

SUBCOMMANDS
    preflight   Every precondition for a run. Exit 1 blocks a launch. Run before spending GPU time.
    launch      preflight, then start the run with its output captured to <arm>/run.log.
    status      Live state of a running arm. Non-zero exit if the run is not healthy.
    validate    Post-run completeness and plausibility. Non-zero exit if the arm is not usable.
    ledger      Every arm ever run, with environment drift and regression detection across them.

EVERY CHECK NAMES THE FAILURE IT PREVENTS
    Because a check whose purpose is forgotten gets deleted the first time it is inconvenient.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402

# The corpus as their builder emits it, and as the harness counts it after the `synth` source filter.
# n_train_cases is their published number, so a mismatch means the split or the filter moved.
EXPECTED_CORPUS_CASES = 6977
EXPECTED_TRAIN_CASES = 6277
# Wall-clock of arm 0, measured 2026-10-03: 21:13:49 -> 15:20:54. The elapsed check budgets from
# this rather than from a projection, because the projection this replaced was wrong by 14-20x and
# spent eleven hours calling a healthy run stuck.
ARM0_MEASURED_MIN = 1087.0
ARM0_SECONDS_PER_STEP = 41.8
# 2 roles x 2 option orders x 3 targets. A short count means an evaluation leg did not run.
EXPECTED_RECORDS = 12
EXPECTED_ITEMS = 21792
# Retained checkpoints plus the fp32 tower copy; three arms' worth of headroom is a minimum.
MIN_FREE_GIB = 8.0
# GPU states, measured: ~11.8 GB training, ~3.8 GB preparing, <1.5 GB gone.
TRAINING_MIB = 8000
PREPARING_MIB = 1500
# The measured peak of a full-parameter arm on this card: 11,874 / 12,282 MiB. A launch needs this
# much free, not merely an "idle" card -- WDDM reports 26 processes holding a GPU context even when
# nothing is computing, so presence of processes is not the question and capacity is.
NEED_MIB = 11874
# Bounds around the one-step measurement of 2 s and the 1500-step extrapolation of ~4500 s.
# Deliberately wide: this catches an order-of-magnitude error, not a 20% one.
TRAIN_SECONDS_MIN = 600
TRAIN_SECONDS_MAX = 21600
LOSS_MAX = 3.0

OK, WARN, FAIL, SKIP = "[OK]", "[WARN]", "[FAIL]", "[SKIP]"


class Report:
    """Accumulates checks and decides the exit code.

    WARN never fails the run: a warning that blocks a launch gets disabled, and a disabled gate is
    worse than none. FAIL always does.
    """

    def __init__(self, title: str):
        self.title = title
        self.rows: list[tuple[str, str, str, str]] = []
        # Numbers the caller wants persisted alongside the verdict, so validate() is not run twice
        # to obtain both a report and the figures to record.
        self.meta: dict = {}

    def add(self, name: str, status: str, detail: str = "", why: str = "") -> None:
        self.rows.append((name, status, detail, why))

    def ok(self, n, d="", w=""):
        self.add(n, OK, d, w)

    def warn(self, n, d="", w=""):
        self.add(n, WARN, d, w)

    def fail(self, n, d="", w=""):
        self.add(n, FAIL, d, w)

    def skip(self, n, d="", w=""):
        self.add(n, SKIP, d, w)

    def render(self) -> int:
        n_fail = sum(1 for r in self.rows if r[1] == FAIL)
        n_warn = sum(1 for r in self.rows if r[1] == WARN)
        n_ok = sum(1 for r in self.rows if r[1] == OK)
        print(f"\n=== {self.title} ===")
        for name, status, detail, _ in self.rows:
            print(f"  {status} {name}")
            if detail:
                print(f"          {detail}")
        print(f"\n  {n_ok} ok, {n_warn} warn, {n_fail} fail")
        if n_fail:
            print(f"\n  BLOCKING FAILURES — the reasons, because each check exists for one:")
            for name, status, _, why in self.rows:
                if status == FAIL and why:
                    print(f"    - {name}: {why}")
        print(f"\n[{'OK' if not n_fail else 'FAIL'}] {self.title}")
        return 1 if n_fail else 0


# --------------------------------------------------------------------------------------- probes


def sha_file(p: pathlib.Path) -> str:
    """Hash with newlines normalised -- their snapshot is LF, this working tree is CRLF."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def spec_repo_comparison(arm_name: str, spec: pathlib.Path) -> tuple[bool | None, str | None]:
    """Does this arm's frozen spec equal the repo's declared spec for that arm?

    Returns (matches, against). `matches` is None when the repo declares nothing for this arm --
    which means "no claim", not "mismatch".

    The previous form compared against ``config/spec.json``. That path has never existed (the file
    is ``config/arm0_spec.json``), so the is_file() guard made the field None on every launch and
    the check answered nothing. Compared by content, not bytes, for the same reason
    check_edit_guard.py is: the arm's spec carries CRLF on this host and the repo's may not, and
    the question is whether the spec is the same, not how it was written down.
    """
    norm = lambda p: p.read_bytes().replace(b"\r\n", b"\n")
    for cand in (ROOT / "config" / f"{arm_name}_spec.json", ROOT / "config" / spec.name):
        if cand.is_file():
            return norm(spec) == norm(cand), cand.relative_to(ROOT).as_posix()
    return None, None


def env_fingerprint(env: pathlib.Path) -> dict:
    fp = {}
    for p in sorted((env / "rsijev").glob("*.py")):
        fp[f"rsijev/{p.name}"] = sha_file(p)[:16]
    for name in ("run_arm_lib.py", "scripts/release_train.py"):
        f = env / name
        if f.is_file():
            fp[name] = sha_file(f)[:16]
    return fp


def gpu_state() -> tuple[int, str, list[str] | None, int]:
    """(used_mib, util, compute_pids, free_mib).

    `free` is the load-bearing value. On WDDM, `--query-compute-apps` lists every process touching
    the GPU -- 26 of them on this host even when idle, because the desktop, browser and terminal all
    hold a context. A check that refused to launch whenever any process appeared would fire while the
    card was empty, and a gate that cries wolf gets bypassed. What actually decides whether a run can
    proceed is whether its measured 11.87 GB will fit, so that is what is checked.
    """
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    used, free, util = -1, -1, "?"
    if r.returncode != 0:
        # -1 / "?" are visible sentinels: a consumer prints them and a reader sees "unknown".
        # Returning 0 for the memory would read as "an idle GPU".
        return used, util, None, free
    try:
        parts = [p.strip() for p in r.stdout.strip().split(",")]
        used, free, util = int(parts[0]), int(parts[1]), parts[2]
    except Exception:                                            # noqa: BLE001
        return used, util, None, free
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True)
    # None, not []. An empty list means "no process holds a context"; a failed query means "we do
    # not know". They are opposite claims, and the second read as the first is how a busy GPU gets
    # reported as free -- which is the one direction this check must never err in.
    pids = ([p.strip() for p in apps.stdout.splitlines() if p.strip()]
            if apps.returncode == 0 else None)
    return used, util, pids, free


def proc_count(pids: list[str] | None) -> str:
    """Render a compute-process count without turning 'unknown' into 'none'."""
    return "unknown" if pids is None else str(len(pids))


def count_jsonl(path: pathlib.Path) -> int:
    if not path.is_file():
        return -1
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


def ledger_path() -> pathlib.Path:
    return ROOT / "measurement" / "health_ledger.jsonl"


def ledger_read() -> list[dict]:
    p = ledger_path()
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def ledger_append(rec: dict) -> None:
    p = ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def allowed_deviations() -> dict:
    """Import the whitelist from the spec validator so the two cannot disagree."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_arm0_spec as chk
        return getattr(chk, "ALLOWED_DEVIATIONS", {})
    except Exception:                                            # noqa: BLE001
        return {}


# ------------------------------------------------------------------------------------ preflight


def preflight(spec_path: pathlib.Path) -> Report:
    rep = Report(f"preflight  spec={spec_path}")

    # paths. Catches an unset JEVRSI_* override, which paths.py refuses to substitute a default for.
    missing = []
    resolved = {}
    for key in ("backbone", "synth_corpus", "arms", "reference_repo", "published_release", "v1_env"):
        try:
            p = getattr(paths, key)()
            resolved[key] = p
            if not p.exists():
                missing.append(f"{key} -> {p}")
        except SystemExit:
            missing.append(f"{key} (unset, no default by design)")
    if missing:
        rep.fail("paths", "; ".join(missing), "an unset path means a run against the wrong data")
    else:
        rep.ok("paths", f"all {len(resolved)} resolve and exist")

    # spec parses.
    spec = None
    if not spec_path.is_file():
        rep.fail("spec file", f"{spec_path} not found", "nothing can be validated without it")
    else:
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            rep.ok("spec parses", f"{len([k for k in spec if not k.startswith('_')])} live keys")
        except json.JSONDecodeError as e:
            rep.fail("spec parses", str(e)[:90], "the harness would die after loading the model")

    if isinstance(spec, dict):
        # splat hazard: a key inside arch_extra/fit_extra is not a comment, it is a constructor arg.
        splat = {k: sorted(x for x in (v or {}) if x.startswith("_"))
                 for k, v in (("arch_extra", spec.get("arch_extra")),
                              ("fit_extra", spec.get("fit_extra")))}
        bad = {k: v for k, v in splat.items() if v}
        if bad:
            rep.fail("spec splat", f"underscore keys inside dicts that get **-splatted: {bad}",
                     "TypeError: unexpected keyword argument, AFTER the model has loaded")
        else:
            rep.ok("spec splat", "no underscore keys inside arch_extra/fit_extra")

        # the check that caught lr_head and option_order.
        pub = resolved.get("published_release")
        meta_p = (pub / "meta.json") if pub else None
        if meta_p and meta_p.is_file():
            published = json.loads(meta_p.read_text(encoding="utf-8")).get("spec", {})
            allow = allowed_deviations()
            actual = {k: v for k, v in spec.items() if not k.startswith("_")}
            drift = {k: (published.get(k, "<absent>"), actual.get(k, "<absent>"))
                     for k in sorted(set(published) | set(actual))
                     if published.get(k, "<absent>") != actual.get(k, "<absent>") and k not in allow}
            if drift:
                rep.fail("spec vs published", f"{len(drift)} unlisted difference(s): {drift}",
                         "a hand-built spec was wrong twice; only the published one is authoritative")
            else:
                n = len(allow)
                rep.ok("spec vs published", f"matches meta.json field for field"
                                            f"{f' except {n} recorded deviation(s)' if n else ''}")
        else:
            rep.skip("spec vs published", "published meta.json not reachable")

        # a value the harness derives must not be pinned, or it silently contradicts the harness.
        for k, where in (("autocast_bf16", "top level"),):
            if k in spec:
                rep.fail(f"derived key {k}", f"pinned at {where}",
                         "run_arm_lib derives it as not freeze_base; pinning can contradict it")
            else:
                rep.ok(f"derived key {k}", "left to the harness")

    # the guards. A guard that does not run is not a guard.
    for gname in ("check_arm0_spec.py", "check_edit_guard.py", "check_synth_decontamination.py"):
        g = ROOT / "scripts" / gname
        if not g.is_file():
            rep.fail(f"guard {gname}", "missing", "the guard set is part of the pipeline's correctness")
            continue
        r = subprocess.run([sys.executable, str(g)], capture_output=True, text=True,
                           cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
        last = (r.stdout or r.stderr).strip().splitlines()
        tail = last[-1][:88] if last else ""
        if r.returncode == 0:
            rep.ok(f"guard {gname}", tail)
        else:
            rep.fail(f"guard {gname}", tail, "a red guard means the run would produce an unusable result")

    # corpus present and the right size.
    sc = resolved.get("synth_corpus")
    if sc:
        main = sc / "data" / "train.jsonl"
        if not main.is_file():
            candidates = list(sc.rglob("train.jsonl"))
            main = candidates[0] if candidates else main
        n = count_jsonl(main)
        if n < 0:
            rep.fail("corpus", f"train.jsonl not found under {sc}",
                     "a missing corpus wastes the whole run")
        elif n != EXPECTED_CORPUS_CASES:
            rep.warn("corpus", f"{n} cases, expected {EXPECTED_CORPUS_CASES}",
                     "a different corpus means the result is not comparable to the published one")
        else:
            rep.ok("corpus", f"{n} cases, as their builder emits")

    # the environment the run will actually import.
    env = resolved.get("v1_env")
    if env and env.is_dir():
        fp = env_fingerprint(env)
        n_mod = len([k for k in fp if not k.startswith("scripts/")])
        if not (env / "rsijev" / "fit.py").is_file():
            rep.fail("v1_env", f"{env} has no rsijev/fit.py",
                     "running against the checkout HEAD is not a reproduction of the release")
        else:
            rep.ok("v1_env", f"{n_mod} modules, fit.py present")
        # drift against the last arm that ran.
        prev = [r for r in ledger_read() if r.get("kind") == "launch" and r.get("environment")]
        if prev and prev[-1]["environment"] != fp:
            changed = sorted(k for k in set(prev[-1]["environment"]) | set(fp)
                             if prev[-1]["environment"].get(k) != fp.get(k))
            rep.warn("v1_env drift", f"differs from arm {prev[-1].get('arm')}: {changed}",
                     "an arm run in a changed environment is not comparable to earlier arms")
        else:
            rep.ok("v1_env drift", "no earlier arm to compare, or fingerprint unchanged")
    else:
        rep.fail("v1_env", f"{env} missing — run scripts/build_v1_env.py",
                 "HEAD is not the revision that produced v1.0 (fit.py 213->559 lines)")

    # GPU capacity. This is the check for the failure that produced this project's false OOM: a run
    # sharing the card. What matters is not whether other processes exist -- on WDDM they always do --
    # but whether this run's measured peak will fit.
    used, util, pids, free = gpu_state()
    if used < 0:
        rep.warn("gpu", "nvidia-smi unreadable")
    elif free < NEED_MIB:
        rep.fail("gpu capacity", f"{free} MiB free, the run peaks at {NEED_MIB} MiB "
                                 f"({used} MiB in use by {proc_count(pids)} process(es))",
                 "a run that does not fit OOMs partway, and this project's false OOM came from "
                 "sharing the card; free memory before launching")
    elif free < NEED_MIB + 800:
        rep.warn("gpu capacity", f"{free} MiB free against a {NEED_MIB} MiB peak — tight",
                 "other GPU work during the run could push it over")
    else:
        rep.ok("gpu capacity", f"{free} MiB free, run peaks at {NEED_MIB} MiB ({proc_count(pids)} process(es) hold a context)")

    # disk headroom for the checkpoint set.
    try:
        free = shutil.disk_usage(str(ROOT)).free / 2**30
        if free < MIN_FREE_GIB:
            rep.fail("disk", f"{free:.1f} GiB free on the repo volume, need {MIN_FREE_GIB}",
                     "a checkpoint write that fails at hour 1 wastes the hour")
        else:
            rep.ok("disk", f"{free:.1f} GiB free")
    except Exception as e:                                       # noqa: BLE001
        rep.warn("disk", f"unavailable ({type(e).__name__})")

    # inform: the numbers this run will be judged against.
    if isinstance(spec, dict):
        rep.ok("run shape", f"steps={spec.get('steps')} batch={spec.get('batch_size')} "
                            f"seed={spec.get('seed')} lr_head={spec.get('lr_head')} "
                            f"option_order={spec.get('option_order')!r}")
        rep.ok("targets", f"n_train_cases={EXPECTED_TRAIN_CASES} (theirs)  "
                          f"final_loss=0.6075 (theirs)  pooled_top1 zero-shot=0.4025 -> theirs=0.6525")
    return rep


# --------------------------------------------------------------------------------------- status


def status(arm: pathlib.Path) -> Report:
    rep = Report(f"status  {arm.name}")

    # Artifacts FIRST, because whether an idle GPU is a fault depends on whether this arm is done.
    # The previous ordering judged the GPU before knowing that, so a completed arm reported FAIL on
    # its own success -- and the reason it printed ("the harness has NO RESUME") is a property of
    # the NEXT launch, not a fault of the one that just finished.
    ck = arm / "checkpoints" / "tower.safetensors"
    out = sorted(p for p in (arm / "out").glob("*.jsonl")
                 if not p.name.endswith(".items.jsonl")) if (arm / "out").is_dir() else []
    items = sorted((arm / "out").glob("*.items.jsonl")) if (arm / "out").is_dir() else []
    finished = bool(out)

    used, util, pids, free = gpu_state()
    if used >= TRAINING_MIB:
        state = "TRAINING"
    elif used >= PREPARING_MIB:
        state = "PREPARING (fp32 tower copy / corpus encode; the loop has not started)"
    else:
        state = "NOT RUNNING"
    if state != "NOT RUNNING":
        rep.ok("gpu", f"{used} MiB / {util}% -> {state}")
    elif finished:
        rep.ok("gpu", f"{used} MiB / {util}% -> NOT RUNNING, and this arm is done "
                      f"({len(out)} record file(s) written)")
    elif used < 0:
        rep.warn("gpu", "nvidia-smi unreadable; cannot say whether this arm is running")
    else:
        rep.fail("gpu", f"{used} MiB / {util}% -> NOT RUNNING, but this arm has written no "
                        f"evaluation records",
                 "the harness has NO RESUME: a dead run costs all its steps again")

    launch = arm / "LAUNCH.json"
    if launch.is_file():
        L = json.loads(launch.read_text(encoding="utf-8"))
        started = L.get("started_at")
        rep.ok("launch record", f"started {started}  spec {str(L.get('spec_sha256'))[:16]}")
        if started:
            try:
                t0 = datetime.datetime.fromisoformat(started)
                mins = (datetime.datetime.now() - t0).total_seconds() / 60
                # Budget from a MEASUREMENT, never from a projection. Arm 0 (2026-10-03) took 1,087
                # min wall: 62,741 s training = 41.8 s/step, plus ~38 min eval. The value this
                # replaced was ~85 min, derived from a "2-3 s/step" estimate read off a 1-step run
                # whose train_seconds contains only one-time setup -- wrong by 14-20x, and it made a
                # healthy run look stuck for eleven hours.
                budget = ARM0_MEASURED_MIN * 1.25
                if mins > budget:
                    rep.warn("elapsed", f"{mins:.0f} min, past the {budget:.0f} min budget "
                                        f"(arm 0 measured {ARM0_MEASURED_MIN:.0f} min)",
                             "either slower than measured or stuck; check before assuming progress")
                else:
                    rep.ok("elapsed", f"{mins:.0f} min of a {budget:.0f} min budget "
                                      f"(arm 0 measured {ARM0_MEASURED_MIN:.0f} min @ 41.8 s/step)")
            except Exception:                                    # noqa: BLE001
                pass
    else:
        rep.warn("launch record", "no LAUNCH.json — this arm was not started through health.py launch")

    # phase from artifacts. Their absence during training is expected, not a failure.
    # (ck / out / items were computed at the top, because the gpu verdict needs them.)
    if out:
        rep.ok("phase", f"evaluation finished ({len(out)} record file(s), {len(items)} items file(s))")
    elif ck.is_file():
        rep.ok("phase", f"checkpoint written {datetime.datetime.fromtimestamp(ck.stat().st_mtime):%H:%M:%S} "
                        f"-> training done, evaluation running or done")
    elif state == "TRAINING":
        rep.ok("phase", "training (no artifacts expected yet — they are written after training)")
    else:
        rep.warn("phase", "no checkpoint and not training", "confirm the process exists")

    # log scan, if the run was captured.
    log = arm / "run.log"
    if log.is_file():
        txt = log.read_text(encoding="utf-8", errors="replace")
        bad = [pat for pat in ("out of memory", "Traceback", "RuntimeError", "FloatingPointError")
               if pat.lower() in txt.lower()]
        if bad:
            rep.fail("log scan", f"found {bad}", "the run has already failed; do not wait on it")
        else:
            rep.ok("log scan", f"{len(txt.splitlines())} lines, no error patterns")
    else:
        rep.warn("log scan", f"{log} absent — status cannot see the run's own output",
                 "launch through health.py launch to capture output to a file")
    return rep


# ------------------------------------------------------------------------------------- validate


def validate(arm: pathlib.Path) -> Report:
    rep = Report(f"validate  {arm.name}")
    L = {}
    if (arm / "LAUNCH.json").is_file():
        L = json.loads((arm / "LAUNCH.json").read_text(encoding="utf-8"))

    # checkpoint + meta.
    ckdir = arm / "checkpoints"
    meta_p = ckdir / "meta.json"
    ck = ckdir / "tower.safetensors"
    if not ck.is_file() or not meta_p.is_file():
        rep.fail("artifacts", f"checkpoint or meta.json missing under {ckdir}",
                 "the harness writes both between training and evaluation; absence means training died")
        return rep
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    rep.ok("artifacts", f"checkpoint {ck.stat().st_size / 2**30:.2f} GiB + meta.json")

    # the signature their meta.json also carries; a mismatch means the wrong code ran.
    keys = sorted(meta)
    want = {"base_model", "spec", "tapped_layer", "tapped_layer_type", "linear_attn_kernel",
            "code_version", "train_seconds", "n_train_cases", "final_loss", "tower_keys", "excluded"}
    if want - set(keys):
        rep.warn("meta keys", f"missing {sorted(want - set(keys))} vs their 11-key shape")
    else:
        rep.ok("meta keys", f"all {len(want)} keys their meta.json carries")

    n = meta.get("n_train_cases")
    if n == EXPECTED_TRAIN_CASES:
        rep.ok("n_train_cases", f"{n} == theirs {EXPECTED_TRAIN_CASES}")
    else:
        rep.fail("n_train_cases", f"{n} != theirs {EXPECTED_TRAIN_CASES}",
                 "their published number; a mismatch means the split or the source filter moved")

    ts, loss = meta.get("train_seconds"), meta.get("final_loss")
    if isinstance(ts, (int, float)):
        if TRAIN_SECONDS_MIN <= ts <= TRAIN_SECONDS_MAX:
            rep.ok("train_seconds", f"{ts:.0f}s (~{ts/max(meta.get('spec',{}).get('steps',1),1):.2f} s/step)")
        else:
            rep.fail("train_seconds", f"{ts:.0f}s outside [{TRAIN_SECONDS_MIN}, {TRAIN_SECONDS_MAX}]",
                     "either a different configuration ran, or the machine was contended")
    if isinstance(loss, (int, float)):
        if 0 < loss < LOSS_MAX:
            rep.ok("final_loss", f"{loss:.4f}")
        else:
            rep.warn("final_loss", f"{loss} outside a plausible range")
    tk = meta.get("tower_keys")
    rep.ok("tower_keys", f"{tk} (theirs 319 on a different backbone; ~309 expected for Qwen3-0.6B)")

    # records + items.
    out = arm / "out"
    recs = sorted(p for p in out.glob("*.jsonl") if not p.name.endswith(".items.jsonl"))
    items = sorted(out.glob("*.items.jsonl"))
    rows = []
    for p in recs:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if len(rows) == EXPECTED_RECORDS:
        rep.ok("records", f"{len(rows)} == {EXPECTED_RECORDS} (2 roles x 2 option orders x 3 targets)")
    elif rows:
        rep.warn("records", f"{len(rows)} != {EXPECTED_RECORDS}",
                 "a short count means an evaluation leg did not run")
    else:
        rep.fail("records", "none", "evaluation produced nothing; the arm's scores do not exist")

    if items:
        ni = count_jsonl(items[0])
        if ni == EXPECTED_ITEMS:
            rep.ok("items", f"{ni} == {EXPECTED_ITEMS}")
        else:
            rep.warn("items", f"{ni} != {EXPECTED_ITEMS}", "a different evaluation population")
    else:
        rep.warn("items", "no items file — pooled_top1 cannot be recomputed")

    # the metric their release is verified in, against the anchor and their number.
    if items:
        try:
            sys.path.insert(0, str(ROOT))
            from measurement.report_from_items import pooled_top1, load
            rws = load(items[0])
            for role, label in (("control", "zero-shot anchor"), ("candidate", "this arm")):
                t1, nn = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                      and r.get("role") == role
                                      and r.get("option_order") == "canonical"])
                if t1 is not None:
                    rep.ok(f"pooled_top1 {role}", f"{t1:.4f} (n={nn})  [{label}]")
            cand, _ = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                   and r.get("role") == "candidate"
                                   and r.get("option_order") == "canonical"])
            ctrl, _ = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                   and r.get("role") == "control"
                                   and r.get("option_order") == "canonical"])
            if cand is not None and ctrl is not None:
                if cand < ctrl:
                    rep.warn("vs anchor", f"candidate {cand:.4f} < zero-shot {ctrl:.4f}",
                             "expected for few steps; not expected for a finished arm")
                else:
                    rep.ok("vs anchor", f"candidate {cand:.4f} vs zero-shot {ctrl:.4f} "
                                        f"({(cand - ctrl) * 100:+.1f} pp)")
            # the order-gap hypothesis, reported not judged. Their v1.0 scores identically under both
            # orders; this project's zero-shot control shows 3.1 pp, and the gap should shrink with
            # training if it is reading presentation position rather than option content.
            ci, _ = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                 and r.get("role") == "candidate" and r.get("option_order") == "canonical"])
            ri, _ = pooled_top1([r for r in rws if r.get("target") == "typed_decisions"
                                 and r.get("role") == "candidate" and r.get("option_order") == "reversed"])
            if ci is not None and ri is not None:
                rep.ok("order gap", f"{abs(ci - ri) * 100:.1f} pp "
                                    f"(zero-shot 3.1 pp, their v1.0 0.0 pp — hypothesis: should shrink)")
            # recorded in the ledger so the drift and regression views have numbers, not just verdicts
            rep.meta = {"pooled_top1_candidate": cand, "pooled_top1_control": ctrl,
                        "order_gap_pp": (abs(ci - ri) * 100
                                         if (ci is not None and ri is not None) else None),
                        "n_train_cases": n, "final_loss": loss, "train_seconds": ts}
        except Exception as e:                                   # noqa: BLE001
            rep.warn("pooled_top1", f"could not compute ({type(e).__name__}: {str(e)[:60]})")

    # environment drift vs earlier arms — the comparison-validity check.
    if L.get("environment"):
        prev = [r for r in ledger_read() if r.get("kind") == "launch" and r.get("environment")
                and r.get("arm") != arm.name]
        if prev and prev[-1]["environment"] != L["environment"]:
            rep.warn("environment drift", f"differs from {prev[-1].get('arm')}",
                     "arms run in different environments are not comparable; say so wherever they are")
        else:
            rep.ok("environment drift", "same as the previous arm, or no earlier arm")
    return rep


# --------------------------------------------------------------------------------------- ledger


def ledger() -> Report:
    rep = Report("ledger")
    rows = ledger_read()
    if not rows:
        rep.warn("ledger", "empty — no arm has been launched through health.py")
        return rep
    launches = [r for r in rows if r.get("kind") == "launch"]
    validates = [r for r in rows if r.get("kind") == "validate"]
    rep.ok("runs", f"{len(launches)} launch(es), {len(validates)} validation(s)")
    print()
    print(f"  {'arm':10} {'started':20} {'spec':14} {'env':12} {'result':>10}")
    for r in launches:
        arm = r.get("arm", "?")
        v = next((x for x in validates if x.get("arm") == arm), None)
        res = "n/a"
        if v:
            res = f"top1 {v.get('pooled_top1', float('nan')):.4f}" if v.get("pooled_top1") else v.get("status", "?")
        envs = r.get("environment") or {}
        envh = hashlib.sha256(json.dumps(envs, sort_keys=True).encode()).hexdigest()[:10]
        print(f"  {arm:10} {str(r.get('started_at'))[:19]:20} {str(r.get('spec_sha256'))[:12]:14} "
              f"{envh:12} {res:>10}")
    # drift across all launches.
    envs = [json.dumps(r.get("environment"), sort_keys=True) for r in launches if r.get("environment")]
    if len(set(envs)) > 1:
        rep.warn("environment drift", f"{len(set(envs))} distinct environments across "
                                      f"{len(envs)} launches",
                 "arms in different environments are not comparable")
    else:
        rep.ok("environment drift", "one environment across every launch")
    return rep


# ---------------------------------------------------------------------------------------- launch


def launch(a) -> int:
    """preflight, then start. The gate is inside the only supported way to launch."""
    spec = pathlib.Path(a.spec)
    rep = preflight(spec)
    rc = rep.render()
    if rc:
        print("\n[REFUSED] preflight failed; not starting. Fix the failures above.")
        return rc

    arm = pathlib.Path(a.arm_dir)
    arm.mkdir(parents=True, exist_ok=True)
    env_dir = paths.v1_env()
    log = arm / "run.log"
    _srm = spec_repo_comparison(arm.name, spec)
    payload = {
        "arm": arm.name,
        "what": a.what or "RSI-Jev recipe on a Qwen3-0.6B backbone",
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "spec": str(spec),
        # content hash: CRLF normalised to LF, so the number is a property of the spec and not of
        # the machine that hashed it. evidence/arm0_LAUNCH.json holds a raw-byte hash for this same
        # field -- it predates this convention and is left as written, so read the two as different
        # numbers unless you normalise first.
        "spec_sha256": sha_file(spec),
        "spec_sha256_convention": "sha256 over bytes with CRLF normalised to LF",
        "spec_repo_match": _srm[0],
        "spec_repo_match_against": _srm[1],
        "environment": env_fingerprint(env_dir),
        "environment_path": str(env_dir),
        "model": str(paths.backbone()),
        "corpus": a.corpus,
        "seed": a.seed,
        "no_resume": True,
        "targets": {"their_n_train_cases": EXPECTED_TRAIN_CASES,
                    "their_final_loss": 0.6075,
                    "our_zero_shot_typed_decisions_top1": 0.4025,
                    "their_typed_decisions_top1": 0.6525},
    }
    (arm / "LAUNCH.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    ledger_append({"kind": "launch", **payload})

    cmd = (f'cd "{env_dir}" && unset PYTHONPATH && '
           f'export PYTHONPATH="{env_dir};{env_dir}/scripts" && '
           f'python "{env_dir}/scripts/release_train.py" '
           f'--model "{paths.backbone()}" --seed {a.seed} --spec "{spec}" '
           f'--save-dir "{arm / "checkpoints"}" --out "{arm / "out"}" '
           f'--name {arm.name} --corpus "{a.corpus}" 2>&1 | tee "{log}"')
    print(f"\n[STARTING] {arm.name}\n  log -> {log}\n  {cmd}\n")
    r = subprocess.Popen(cmd, shell=True, cwd=str(env_dir))
    print(f"  pid {r.pid}. Monitor with: python scripts/health.py status --arm {arm}")
    return 0


# ------------------------------------------------------------------------------------------ cli


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnostic chain for the JevRSI pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight", help="preconditions for a run; non-zero blocks a launch")
    p.add_argument("--spec", default=str(ROOT / "config" / "arm0_spec.json"))

    p = sub.add_parser("status", help="live state of a running arm")
    p.add_argument("--arm", default="E:/2026-AI4S/arms/arm0")

    p = sub.add_parser("validate", help="post-run completeness and plausibility")
    p.add_argument("--arm", default="E:/2026-AI4S/arms/arm0")

    sub.add_parser("ledger", help="all arms, with drift and regression detection")

    p = sub.add_parser("dashboard", help="export viz/state.js for the live dashboard")
    p.add_argument("--arm", default="E:/2026-AI4S/arms/arm0")

    p = sub.add_parser("launch", help="preflight then start, with output captured")
    p.add_argument("--spec", default=str(ROOT / "config" / "arm0_spec.json"))
    p.add_argument("--arm-dir", default="E:/2026-AI4S/arms/arm0")
    p.add_argument("--corpus", default="E:/2026-AI4S/corpus_rsijev")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--what", default="")

    a = ap.parse_args()
    if a.cmd == "preflight":
        return preflight(pathlib.Path(a.spec)).render()
    if a.cmd == "status":
        return status(pathlib.Path(a.arm)).render()
    if a.cmd == "validate":
        arm = pathlib.Path(a.arm)
        rep = validate(arm)
        rc = rep.render()
        ledger_append({"kind": "validate", "arm": arm.name,
                       "at": datetime.datetime.now().isoformat(timespec="seconds"),
                       "status": "ok" if rc == 0 else "failed",
                       **(rep.meta or {})})
        return rc
    if a.cmd == "ledger":
        return ledger().render()
    if a.cmd == "dashboard":
        import export_health_state
        p = export_health_state.export(pathlib.Path(a.arm))
        print(f"[OK] {p}")
        print(f"     open {ROOT / 'viz' / 'health_dashboard.html'}")
        return 0
    if a.cmd == "launch":
        return launch(a)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
