"""Regression tests for scripts/session_runner.py.

The load-regime check is tested against GROUND TRUTH rather than against itself. The four
noise-floor runs of 2026-09-26 have independently established labels -- baseline and
replicate 3 ran on an idle host (44 steps), replicates 1 and 2 ran while five subagents
stole CPU (42 steps) -- and check_session_throughput must reproduce that split. A check
that agrees only with its own expectations would be worthless.

Run: python tests/test_session_runner.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import session_runner as SR  # noqa: E402

LOGS = Path("D:/2026-AI4S/nanochat-autoresearch/.logs")
STOCK = Path("D:/2026-AI4S/nanochat-autoresearch/program.md")

n_pass = n_fail = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[OK] {label}")
    else:
        n_fail += 1
        print(f"[FAIL] {label}" + (f" -- {detail}" if detail else ""))


# --- 1. manifest integrity ---------------------------------------------------
man = json.loads(SR.MANIFEST.read_text(encoding="utf-8"))
v00 = next(v for v in man["variants"] if v["id"] == "v00")
check("manifest declares 13 variants", man["n_variants"] == 13,
      f"got {man['n_variants']}")
check("v00 is flagged identical_to_stock", v00["identical_to_stock"] is True)
check("v00 hash equals the manifest's stock hash",
      v00["sha256"] == man["stock_sha256"])
if STOCK.exists():
    check("v00 hash equals the LIVE program.md on disk",
          v00["sha256"] == SR.sha256_file(STOCK),
          f"manifest {v00['sha256'][:12]} vs disk {SR.sha256_file(STOCK)[:12]}")
check("every non-v00 variant differs from stock",
      all(not v["identical_to_stock"] for v in man["variants"] if v["id"] != "v00"))
check("every variant records exactly the expected slot count",
      all(len(v["slots"]) == len(v["substitutions"]) for v in man["variants"]))


# --- 2. hash verification catches tampering ----------------------------------
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    # copy the assembled dir so we can tamper with a copy, not the frozen artifact
    shutil.copytree(SR.VARIANTS_DIR, tmp / "assembled")
    orig_dir, orig_man = SR.VARIANTS_DIR, SR.MANIFEST
    try:
        SR.VARIANTS_DIR = tmp / "assembled"
        SR.MANIFEST = tmp / "manifest.json"
        shutil.copy(orig_man, SR.MANIFEST)
        p, rec = SR.verify_variant("v01")
        check("verify_variant accepts an untampered variant", rec["id"] == "v01")
        f = SR.VARIANTS_DIR / rec["file"]
        f.write_bytes(f.read_bytes() + b"\n# tampered\n")
        try:
            SR.verify_variant("v01")
            check("verify_variant REJECTS a tampered variant", False,
                  "no exception raised")
        except SystemExit as exc:
            check("verify_variant REJECTS a tampered variant",
                  "hash mismatch" in str(exc))
    finally:
        SR.VARIANTS_DIR, SR.MANIFEST = orig_dir, orig_man


# --- 3. install / restore round trip (custom dest: never touch the live file) -
with tempfile.TemporaryDirectory() as td:
    fake = Path(td) / "program.md"
    fake.write_bytes(STOCK.read_bytes() if STOCK.exists() else b"stock\n")
    before = SR.sha256_file(fake)
    inst = SR.install_variant("v01", dest=fake)
    check("install writes the variant's bytes", SR.sha256_file(fake) == inst["installed_sha256"])
    check("installed content differs from stock", SR.sha256_file(fake) != before)
    check("install preserved a stock backup", fake.with_suffix(".md.stock_backup").exists())
    restored = SR.restore_stock(dest=fake)
    check("restore_stock reports success", restored is True)
    check("restore_stock returns the file to stock", SR.sha256_file(fake) == before)


# --- 4. ledger resume logic ignores dry runs ---------------------------------
with tempfile.TemporaryDirectory() as td:
    orig_ledger = SR.LEDGER
    try:
        SR.LEDGER = Path(td) / "sessions.jsonl"
        SR.LEDGER.write_text(
            json.dumps({"variant": "v00", "seed": 42, "dry_run": True}) + "\n"
            + json.dumps({"variant": "v01", "seed": 42, "dry_run": False}) + "\n"
            + json.dumps({"variant": "v02", "seed": 7}) + "\n",
            encoding="utf-8")
        cells = SR.ledger_cells()
        check("dry-run records are NOT treated as completed work",
              ("v00", 42) not in cells, f"got {cells}")
        check("real records ARE treated as completed work", ("v01", 42) in cells)
        check("a record with no dry_run key counts as real", ("v02", 7) in cells)
    finally:
        SR.LEDGER = orig_ledger


# --- 5. gate G1 is reported as unmet right now (it is the known blocker) ------
g1 = SR.check_gate_g1()
check("gate G1 correctly reports --seed as absent", g1["ok"] is False,
      "if --seed has since been added, update the design doc and expect ok=True")
check("gate G1 names the flaw it enforces", "FLAW 7" in g1["detail"] or
      "seed" in g1["detail"].lower())


# --- 6. GROUND TRUTH: the load-regime check must reproduce the known labels ---
# Independently established by the 2026-09-26 noise floor on commit a4123c6.
GROUND_TRUTH = {
    "baseline": "clean",
    "repl_1": "contaminated",
    "repl_2": "contaminated",
    "repl_3": "clean",
    "repl_4": "clean",
    # repl_5 is the case that broke the linear "bpb per step" story: 43 steps, ONE short of
    # complete, yet val_bpb 0.888926 sits in the truncated cluster rather than between the
    # clusters. It must be flagged, and it is the reason the check tests num_steps < 44
    # rather than modelling a slope.
    "repl_5": "contaminated",
}
if LOGS.exists():
    for name, truth in GROUND_TRUTH.items():
        f = LOGS / f"{name}.log"
        if not f.exists():
            check(f"ground truth {name}: log present", False, f"missing {f}")
            continue
        m = SR.parse_nanochat_log(f.read_text(encoding="utf-8", errors="replace"))
        chk = SR.check_session_throughput(m, fixed_config=True)
        verdict = "clean" if chk["regime_ok"] else "contaminated"
        check(f"ground truth {name} -> {truth}", verdict == truth,
              f"got {verdict}, violations={chk['violations']}")

    # num_steps must be the discriminator: 44 clean vs 42 contaminated
    steps = {}
    for name in GROUND_TRUTH:
        f = LOGS / f"{name}.log"
        if f.exists():
            steps[name] = SR.parse_nanochat_log(
                f.read_text(encoding="utf-8", errors="replace")).get("num_steps")
    clean_steps = {steps[n] for n, t in GROUND_TRUTH.items() if t == "clean" and n in steps}
    dirty_steps = {steps[n] for n, t in GROUND_TRUTH.items() if t == "contaminated" and n in steps}
    check("num_steps separates clean from contaminated with no overlap",
          clean_steps.isdisjoint(dirty_steps),
          f"clean={clean_steps} contaminated={dirty_steps}")

    # and the agent-session mode must refuse to use it
    m2 = SR.parse_nanochat_log((LOGS / "repl_2.log").read_text(encoding="utf-8", errors="replace"))
    chk2 = SR.check_session_throughput(m2, fixed_config=False)
    check("fixed_config=False declines the step-count test (agent sessions)",
          chk2["regime_ok"] is True and "num_steps_note" in chk2)
else:
    check("noise-floor logs available for ground-truth validation", False,
          f"{LOGS} not found")


# --- 7. parse must surface the load covariates --------------------------------
if (LOGS / "repl_1.log").exists():
    m = SR.parse_nanochat_log((LOGS / "repl_1.log").read_text(encoding="utf-8", errors="replace"))
    for key in ("val_bpb", "num_steps", "dt_median_ms", "dt_max_ms",
                "tok_per_sec_min", "n_step_lines"):
        check(f"parse_nanochat_log exposes {key}", key in m)
    check("repl_1 val_bpb parsed", abs(m.get("val_bpb", 0) - 0.889241) < 1e-6,
          f"got {m.get('val_bpb')}")
    check("repl_1 max dt parsed as the contamination spike",
          m.get("dt_max_ms") == 14972, f"got {m.get('dt_max_ms')}")

print()
print(f"[{'OK' if n_fail == 0 else 'FAIL'}] session_runner: {n_pass} checks passed, "
      f"{n_fail} failed")
raise SystemExit(1 if n_fail else 0)
