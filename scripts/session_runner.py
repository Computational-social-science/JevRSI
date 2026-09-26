"""Session runner for the program.md variant study (docs/nanochat_only_design.md).

WHAT A "SESSION" IS
-------------------
The experimental unit is one **agent session**: an executor reads ONE variant of
`program.md` and then iterates on train.py, launching many training jobs. The DV is a
property of that trajectory (best val_bpb@300s, keep rate, reason-code histogram), never
of a single training job -- a single `uv run train.py` does not involve program.md at all.

WHAT THIS SCRIPT ENFORCES
-------------------------
Every one of these is a hard stop, because each corresponds to a documented flaw:

  gate G1  the frozen harness must expose --seed, else S=1 and no seed-generalisation
           claim is admissible (FLAW 7). Checked by asking train.py itself, not by grep.
  D1       the executor must be DECLARED BY NAME in EXECUTORS. There is no default, so a
           session cannot silently run under an undeclared executor -- that would make the
           agent an uncontrolled factor (FLAW 4/9).
  FLAW 11  the host must be idle before the session starts. Concurrent orchestration on
           the training host stole CPU, cost 2 optimizer steps, and moved val_bpb by
           ~0.025 bpb. The regime is recorded for every session.
  FLAW 8   the trajectory directory is required to exist, so rejected attempts leave an
           audit trail rather than vanishing into a git reset.

WHAT THIS SCRIPT DOES **NOT** DO
--------------------------------
It does not implement the executor. Driving a real agent session needs a configured
executor command (see EXECUTORS), and every entry there is currently a placeholder that
refuses to run. `--dry-run` exercises the whole pipeline -- hash-verified assembly,
preflight, ledger, resume -- without launching anything.

Usage:
    python scripts/session_runner.py --preflight --executor qwen25_7b_local
    python scripts/session_runner.py --variant v00 --seed 42 --session-minutes 60 \
        --executor qwen25_7b_local --dry-run
    python scripts/session_runner.py --all --seed 42 --session-minutes 60 \
        --executor qwen25_7b_local --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NANOCHAT = Path("D:/2026-AI4S/nanochat-autoresearch")
VARIANTS_DIR = ROOT / "configs" / "program_variants" / "assembled"
MANIFEST = ROOT / "configs" / "program_variants" / "manifest.json"
LEDGER = ROOT / "runs" / "sessions.jsonl"
LEDGER_LOCK = ROOT / "runs" / ".sessions.lock"

# --- Executor registry (D1) --------------------------------------------------
# The executor is the STRATEGY'S EXECUTOR and is held fixed across the study, so its
# identity is part of the treatment definition and must be recorded with every session.
# `command` is a template rendered with {program_md}, {workdir}, {minutes}, {log}.
# Every entry must be filled in deliberately before a real session can run; an entry
# whose command contains DECLARE_ME is refused by preflight.
EXECUTORS: dict[str, dict] = {
    "qwen25_7b_local": {
        "model": "qwen2.5:7b",
        "backend": "ollama",
        "decoding": {"temperature": 0.0, "seed": None, "num_ctx": 32768},
        "command": "DECLARE_ME_qwen25_7b_local",
        "notes": "Canonical academic model, already local. temperature 0 is a deliberate "
                 "choice: it minimises agent-sampling variance, which is the largest "
                 "unmeasured channel (FLAW 9). Record the seed once decoding supports one.",
    },
    "gemma2_9b_local": {
        "model": "gemma2:9b",
        "backend": "ollama",
        "decoding": {"temperature": 0.0, "seed": None, "num_ctx": 32768},
        "command": "DECLARE_ME_gemma2_9b_local",
        "notes": "Canonical academic model. Available as a SECOND executor for the "
                 "explicitly-confounded replication in FLAW 4; never mixed into the "
                 "primary design, which holds the executor fixed.",
    },
}

_SEED_HELP = re.compile(r"--seed\b")

# --- Load-regime thresholds (FLAW 11) ----------------------------------------
# These three MUST be calibrated on this host; the defaults are provisional. The
# authoritative check is post-hoc (`check_session_throughput`), so a loose gate here is
# safer than a tight one: a gate that always fires gets bypassed with --allow-busy.
GPU_IDLE_MIB = 2600          # a training job holds ~1777 MiB, so above this implies compute
MAX_PYTHON_PROCS = 6         # a bare host has few; concurrent agents push this up
THROUGHPUT_TOLERANCE = 1.15  # >15% departure from the clean baseline counts as contaminated

# Measured 2026-09-26, commit a4123c6, idle host, TIME_BUDGET=300, batch 4, grad-accum 64.
# Two idle-host runs at 44 steps gave val_bpb 0.863803 and 0.866649 (|diff| 0.002846),
# while the 44 -> 42 step gap was ~0.0254 bpb at 0.0127 bpb/step.
BASELINE_CLEAN = {
    "commit": "a4123c6",
    "dt_median_ms": 9151,
    "dt_max_ms": 10544,
    "tok_per_sec_min": 49723,
    "val_bpb": 0.863803,
    "val_bpb_replicate": 0.866649,
    "num_steps": 44,
    "peak_vram_mb": 1777.1,
}


# --- helpers -----------------------------------------------------------------
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"[FAIL] no manifest at {MANIFEST}; run "
                         f"scripts/build_program_variants.py first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def git_out(args: list[str], cwd: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, timeout=60)
        return p.stdout.decode("utf-8", "replace").strip()
    except Exception as exc:  # noqa: BLE001
        return f"<git {args[0]} failed: {exc}>"


# --- preflight ---------------------------------------------------------------
def check_gate_g1() -> dict:
    """Ask train.py whether it accepts --seed. Empirical, not a grep for a string."""
    train = NANOCHAT / "train.py"
    if not train.exists():
        return {"ok": False, "detail": f"train.py not found at {train}"}
    text = train.read_text(encoding="utf-8", errors="replace")
    has_parser_entry = bool(re.search(r'add_argument\(\s*["\']--seed', text))
    has_hardcoded = bool(re.search(r"manual_seed\(\s*42\s*\)", text))
    if not has_parser_entry:
        return {
            "ok": False,
            "detail": "train.py exposes NO --seed argument, so every configuration yields "
                      "exactly one draw and no seed-generalisation claim is admissible "
                      "(FLAW 7). Add a human-controlled --seed outside the agent's "
                      "editable surface and freeze it before running any session.",
            "hardcoded_seed_42_present": has_hardcoded,
        }
    if has_hardcoded:
        return {
            "ok": False,
            "detail": "--seed exists but manual_seed(42) is ALSO still hardcoded; the flag "
                      "may not actually control the seed. Remove the hardcoding.",
        }
    return {"ok": True, "detail": "--seed present and no hardcoded seed remains"}


def check_load_regime() -> dict:
    """FLAW 11 load-regime probe.

    NOTE ON WHY THIS DOES NOT USE `nvidia-smi --query-compute-apps`.
    An earlier version did, and it was useless: on this Windows desktop it returns ~25
    entries (explorer.exe, SearchHost.exe, StartMenuExperienceHost.exe, WeChat,
    Chrome ...) and EVERY one of them reports memory `[N/A]` -- including the running
    training process. So that query carries no discriminating signal here, and gating on
    it would block every session, which trains the operator to pass --allow-busy and
    defeats the gate entirely.

    The contamination mechanism measured on 2026-09-26 was **CPU contention**, not GPU
    occupancy: five parallel subagents slowed the training step (max dt 14,972 ms vs a
    10,544 ms idle-host ceiling), which at a fixed wall-clock budget became 2 fewer
    optimizer steps and ~0.025 bpb. So the useful signals are:

      * GPU memory in use (a training job holds ~1.8 GB; a bare desktop does not), and
      * how many python.exe processes are alive.

    Neither is a perfect gate, so the authoritative check is POST-HOC: every experiment's
    `dt_median_ms` and `tok_per_sec_min` are compared against the frozen clean baseline
    (see BASELINE_CLEAN below). A session whose throughput departs from baseline is
    flagged rather than trusted.
    """
    regime: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        first = p.stdout.strip().splitlines()[0]
        parts = [x.strip() for x in first.split(",")]
        regime["gpu_mem_used_mib"] = int(float(parts[0]))
        regime["gpu_mem_total_mib"] = int(float(parts[1]))
        regime["gpu_util_pct"] = int(float(parts[2]))
    except Exception as exc:  # noqa: BLE001
        regime["nvidia_smi_error"] = str(exc)

    try:
        p = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
                           capture_output=True, text=True, timeout=30)
        regime["python_procs"] = sum(
            1 for ln in p.stdout.splitlines() if "python" in ln.lower()
        )
    except Exception as exc:  # noqa: BLE001
        regime["tasklist_error"] = str(exc)

    mem = regime.get("gpu_mem_used_mib")
    pys = regime.get("python_procs")
    busy_gpu = mem is not None and mem > GPU_IDLE_MIB
    busy_cpu = pys is not None and pys > MAX_PYTHON_PROCS
    regime["busy_gpu"] = busy_gpu
    regime["busy_cpu"] = busy_cpu
    regime["idle"] = not (busy_gpu or busy_cpu)
    return regime


def check_session_throughput(step_metrics: dict, *, fixed_config: bool = True) -> dict:
    """Post-hoc FLAW 11 check: did this session run at clean-baseline throughput?

    `num_steps` is the primary indicator, not the throughput ratios. Validated against
    ground truth on 2026-09-26: the four noise-floor runs split perfectly on step count
    (44, 44 clean; 42, 42 contaminated) while the ratios did not -- repl_2's median dt was
    only 1.015x the clean baseline and its throughput floor 0.94x, so a ratio-only check
    missed it. Step count is the direct consequence of the contamination mechanism: at a
    fixed wall-clock budget, lost throughput *is* lost steps.

    `fixed_config` states whether train.py was held constant across the compared runs.
    It matters: an agent is free to change batch size or model size, which legitimately
    changes `num_steps`. So for **agent sessions** the step-count test is not admissible
    and only the throughput ratios apply; for **replicate/control blocks** at a fixed
    commit it is the strongest evidence available and must be used.
    """
    out: dict = {"baseline": BASELINE_CLEAN, "fixed_config": fixed_config,
                 "violations": []}

    ns = step_metrics.get("num_steps")
    base_ns = BASELINE_CLEAN["num_steps"]
    if ns is None:
        pass
    elif fixed_config and ns < base_ns:
        out["num_steps_delta"] = ns - base_ns
        out["violations"].append(
            f"num_steps {ns} is {base_ns - ns} below the clean baseline {base_ns}: host "
            f"load stole optimizer step(s) at a fixed wall-clock budget"
        )
    elif not fixed_config:
        out["num_steps_note"] = (
            f"num_steps {ns} not tested: config was not held fixed, so a step-count "
            f"change is not attributable to load"
        )

    dt = step_metrics.get("dt_median_ms")
    tok = step_metrics.get("tok_per_sec_min")
    if dt is not None:
        ratio = dt / BASELINE_CLEAN["dt_median_ms"]
        out["dt_median_ratio"] = round(ratio, 3)
        if ratio > THROUGHPUT_TOLERANCE:
            out["violations"].append(
                f"dt_median {dt} ms is {ratio:.2f}x the clean baseline "
                f"{BASELINE_CLEAN['dt_median_ms']} ms"
            )
    if tok is not None:
        ratio = tok / BASELINE_CLEAN["tok_per_sec_min"]
        out["tok_per_sec_min_ratio"] = round(ratio, 3)
        if ratio < 1 / THROUGHPUT_TOLERANCE:
            out["violations"].append(
                f"tok/sec floor {tok:,.0f} is {ratio:.2f}x the clean baseline "
                f"{BASELINE_CLEAN['tok_per_sec_min']:,.0f}"
            )
    out["regime_ok"] = not out["violations"]
    return out


def verify_variant(variant_id: str) -> tuple[Path, dict]:
    """Hash-verify an assembled variant against the frozen manifest."""
    man = load_manifest()
    rec = next((v for v in man["variants"] if v["id"] == variant_id), None)
    if rec is None:
        raise SystemExit(f"[FAIL] variant {variant_id!r} not in the manifest "
                         f"(have: {[v['id'] for v in man['variants']]})")
    path = VARIANTS_DIR / rec["file"]
    if not path.exists():
        raise SystemExit(f"[FAIL] assembled variant missing: {path}")
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise SystemExit(
            f"[FAIL] {variant_id} hash mismatch -- the assembled file has been edited "
            f"since it was frozen.\n  manifest: {rec['sha256']}\n  on disk : {got}\n"
            f"Re-run scripts/build_program_variants.py to regenerate, and note that this "
            f"invalidates any session already run under the old hash."
        )
    return path, rec


def install_variant(variant_id: str, dest: Path = NANOCHAT / "program.md") -> dict:
    """Copy the frozen variant over the harness template, then verify the copy."""
    src, rec = verify_variant(variant_id)
    stock_backup = dest.with_suffix(".md.stock_backup")
    if dest.exists() and not stock_backup.exists():
        shutil.copy2(dest, stock_backup)
    data = src.read_bytes()
    dest.write_bytes(data)
    if sha256_file(dest) != rec["sha256"]:
        raise SystemExit(f"[FAIL] post-install hash mismatch writing {dest}")
    return {"variant": variant_id, "installed_sha256": rec["sha256"],
            "dest": str(dest), "bytes": len(data)}


def restore_stock(dest: Path = NANOCHAT / "program.md") -> bool:
    stock_backup = dest.with_suffix(".md.stock_backup")
    if stock_backup.exists():
        shutil.copy2(stock_backup, dest)
        return True
    return False


# --- session mechanics -------------------------------------------------------
DT_RE = re.compile(r"dt: (\d+)ms \| tok/sec: ([\d,]+)")
VAL_RE = re.compile(r"^val_bpb:\s+([\d.]+)", re.M)
STEPS_RE = re.compile(r"^num_steps:\s+(\d+)", re.M)


def parse_nanochat_log(text: str) -> dict:
    """Extract per-experiment metrics, including the FLAW 11 load covariates."""
    out: dict = {}
    m = VAL_RE.search(text)
    if m:
        out["val_bpb"] = float(m.group(1))
    m = STEPS_RE.search(text)
    if m:
        out["num_steps"] = int(m.group(1))
    dts, toks = [], []
    for dm in DT_RE.finditer(text):
        dts.append(int(dm.group(1)))
        toks.append(float(dm.group(2).replace(",", "")))
    if dts:
        dts_sorted = sorted(dts)
        out["dt_median_ms"] = dts_sorted[len(dts_sorted) // 2]
        out["dt_max_ms"] = max(dts)
        out["tok_per_sec_min"] = min(toks)
        out["n_step_lines"] = len(dts)
    return out


def collect_session(session_dir: Path) -> dict:
    """Read the session's own artifacts: results.tsv plus the trajectory records."""
    res: dict = {"n_experiments": 0, "n_kept": 0, "n_rejected": 0,
                 "reason_codes": {}, "best_val_bpb": None, "trajectory_records": 0}
    tsv = NANOCHAT / "results.tsv"
    if tsv.exists():
        lines = [ln for ln in tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) < 5:
                continue
            res["n_experiments"] += 1
            status, val = parts[3].strip(), parts[1].strip()
            if status == "keep":
                res["n_kept"] += 1
                try:
                    v = float(val)
                    if v > 0 and (res["best_val_bpb"] is None or v < res["best_val_bpb"]):
                        res["best_val_bpb"] = v
                except ValueError:
                    pass
            else:
                res["n_rejected"] += 1
    traj = NANOCHAT / "runs"
    if traj.exists():
        for jf in traj.rglob("trajectory/*.json"):
            res["trajectory_records"] += 1
            try:
                rec = json.loads(jf.read_text(encoding="utf-8"))
                code = rec.get("reason_code", "UNKNOWN")
                res["reason_codes"][code] = res["reason_codes"].get(code, 0) + 1
            except Exception:  # noqa: BLE001
                pass
    return res


def launch_executor(cmd: str, workdir: Path, minutes: int, log: Path) -> dict:
    """Run the executor command for at most `minutes`. Returns a status dict."""
    if "DECLARE_ME" in cmd:
        return {"launched": False,
                "reason": "executor command is a placeholder (DECLARE_ME); declare a real "
                          "executor in EXECUTORS before running a session"}
    with log.open("w", encoding="utf-8") as fh:
        try:
            p = subprocess.run(cmd, shell=True, cwd=str(workdir), stdout=fh,
                               stderr=subprocess.STDOUT, timeout=minutes * 60)
            return {"launched": True, "exit_code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"launched": True, "exit_code": None,
                    "reason": f"session exceeded {minutes} min and was stopped"}


def ledger_cells() -> set[tuple[str, int]]:
    """Cells already recorded by a REAL session.

    Dry-run records are deliberately ignored: a dry run launches nothing and therefore
    proves nothing, so letting it mark a cell as done would silently skip real work.
    """
    if not LEDGER.exists():
        return set()
    done = set()
    for ln in LEDGER.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
            if r.get("dry_run"):
                continue
            done.add((r["variant"], int(r["seed"])))
        except Exception:  # noqa: BLE001
            continue
    return done


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="every variant in the manifest")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--session-minutes", type=int, default=60)
    ap.add_argument("--executor", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except launch the executor")
    ap.add_argument("--resume", action="store_true", help="skip cells already in the ledger")
    ap.add_argument("--preflight", action="store_true", help="run checks and exit")
    ap.add_argument("--allow-busy", action="store_true",
                    help="proceed even if the load regime is not idle (records the violation)")
    args = ap.parse_args()

    print("=" * 74)
    print("session runner -- program.md variant study")
    print("=" * 74)

    man = load_manifest()
    ids = [v["id"] for v in man["variants"]]
    print(f"manifest   : {MANIFEST}")
    print(f"             {man['n_variants']} variants, stock sha256 "
          f"{man['stock_sha256'][:16]}")
    print(f"nanochat   : {NANOCHAT}")
    print(f"ledger     : {LEDGER}")
    print()

    # ---- preflight (always runs; these are hard stops) ----
    blockers: list[str] = []

    g1 = check_gate_g1()
    print(f"[{'OK ' if g1['ok'] else 'G1!'}] gate G1 --seed: {g1['detail']}")
    if not g1["ok"]:
        blockers.append("gate G1")

    if not args.executor:
        print("[FAIL] --executor is required: select one by name from EXECUTORS (D1)")
        blockers.append("executor")
        ex = None
    else:
        ex = EXECUTORS.get(args.executor)
        if ex is None:
            print(f"[FAIL] executor {args.executor!r} is not declared. Known: "
                  f"{list(EXECUTORS)}")
            blockers.append("executor")
        else:
            print(f"[OK ] executor {args.executor}: model={ex['model']} "
                  f"backend={ex['backend']} decoding={ex['decoding']}")
            if "DECLARE_ME" in ex["command"]:
                print(f"[FAIL] executor {args.executor} has no real command "
                      f"(placeholder). Fill in EXECUTORS[...]['command'].")
                blockers.append("executor command")

    regime = check_load_regime()
    idle = regime.get("idle", False)
    gpu_txt = (f"gpu {regime.get('gpu_mem_used_mib', '?')}/{regime.get('gpu_mem_total_mib', '?')} MiB "
               f"util {regime.get('gpu_util_pct', '?')}%")
    print(f"[{'OK ' if idle else 'BUS'}] load regime: {gpu_txt}, "
          f"{regime.get('python_procs', '?')} python.exe "
          f"(thresholds: >{GPU_IDLE_MIB} MiB, >{MAX_PYTHON_PROCS} procs)")
    if not idle:
        print(f"       busy_gpu={regime.get('busy_gpu')} busy_cpu={regime.get('busy_cpu')}")
        if not args.allow_busy:
            print("       -> FLAW 11: refuse to measure under contention "
                  "(override with --allow-busy to record the violation)")
            blockers.append("load regime")

    if not (NANOCHAT / "runs").exists():
        print("[WARN] nanochat runs/ does not exist yet; the trajectory store is created "
              "by program.md setup step 6")
    else:
        print("[OK ] trajectory store present (FLAW 8)")

    print()
    if args.preflight or (blockers and not args.dry_run):
        if blockers:
            print(f"[BLOCKED] {len(blockers)} blocker(s): {', '.join(blockers)}")
            print("Nothing was run. Resolve the blockers above and re-run.")
            return 2
        print("[OK] preflight passed -- the runner is ready for a real session.")
        return 0
    if blockers:
        # --dry-run launches nothing, so it is the one mode allowed to exercise the
        # pipeline while a gate is still unmet. The blockers are still recorded.
        print(f"[WARN] --dry-run proceeding despite {len(blockers)} blocker(s): "
              f"{', '.join(blockers)} -- nothing will be launched.")
        print()
    if ex is None:
        print("[FAIL] --dry-run still requires a declared --executor")
        return 2

    # ---- resolve cells ----
    if args.all:
        chosen = ids
    elif args.variant:
        chosen = args.variant
    else:
        print("[FAIL] pass --variant vNN (repeatable), --all, or --preflight")
        return 2
    if args.seed is None:
        print("[FAIL] --seed is required (it is the session base seed, factor 2)")
        return 2

    done = ledger_cells() if args.resume else set()
    print(f"cells requested: {len(chosen)} variant(s) x seed {args.seed}"
          f"{' (resume: skipping ' + str(len(done)) + ' recorded cell(s))' if args.resume else ''}")
    print()

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    n_done = n_skip = n_fail = 0

    for vid in chosen:
        if (vid, args.seed) in done:
            print(f"  {vid} seed={args.seed}: SKIP (already in ledger)")
            n_skip += 1
            continue

        session_id = f"{vid}_s{args.seed}_{run_stamp}"
        session_dir = ROOT / "runs" / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        try:
            inst = install_variant(vid)
        except SystemExit as exc:
            print(f"  {vid}: FAIL install -> {exc}")
            n_fail += 1
            continue
        print(f"  {vid} seed={args.seed}: installed program.md "
              f"sha256={inst['installed_sha256'][:12]}")

        record = {
            "session_id": session_id,
            "variant": vid,
            "variant_sha256": inst["installed_sha256"],
            "seed": args.seed,
            "executor": args.executor,
            "executor_spec": ex,
            "session_minutes": args.session_minutes,
            "stock_sha256": man["stock_sha256"],
            "load_regime": regime,
            "dry_run": bool(args.dry_run),
            "nanochat_commit": git_out(["rev-parse", "--short", "HEAD"], NANOCHAT),
        }

        if args.dry_run:
            record["status"] = "dry_run"
            print(f"        --dry-run: not launching the executor")
        else:
            log = session_dir / "executor.log"
            out = launch_executor(ex["command"], NANOCHAT, args.session_minutes, log)
            record["executor_result"] = out
            record["status"] = ("completed" if out.get("launched")
                                else "not_launched")
            print(f"        executor: {out}")

        session_metrics = collect_session(session_dir)
        if session_metrics["n_experiments"] == 0:
            print("        [WARN] results.tsv shows 0 experiments -- a session with no "
                  "trajectory cannot contribute to the DV (docs §7.2)")
        record["session_metrics"] = session_metrics

        # Post-hoc FLAW 11 check: the up-front gate is coarse, so the authoritative test
        # is whether the session actually ran at clean-baseline throughput.
        runlog = NANOCHAT / "run.log"
        if runlog.exists():
            step_metrics = parse_nanochat_log(
                runlog.read_text(encoding="utf-8", errors="replace"))
            record["step_metrics"] = step_metrics
            record["throughput_check"] = check_session_throughput(step_metrics)
            if not record["throughput_check"]["regime_ok"]:
                print(f"        [REGIME] contaminated: "
                      f"{record['throughput_check']['violations']}")
            else:
                print(f"        [OK] throughput within "
                      f"{THROUGHPUT_TOLERANCE:.2f}x of the clean baseline")

        record["recorded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

        (session_dir / "session.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        if args.dry_run:
            print("        (dry-run: not written to the ledger, so resume is unaffected)")
        else:
            append_ledger(record)
        n_done += 1
        print(f"        session.json -> {session_dir}")

    restored = restore_stock()
    print()
    print(f"stock program.md restored: {restored}")
    print(f"[OK] {n_done} session(s) recorded, {n_skip} skipped (resume), {n_fail} failed")
    print(f"ledger: {LEDGER}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
