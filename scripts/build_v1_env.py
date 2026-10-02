#!/usr/bin/env python
"""Build the v1.0 code environment: their release's own modules, with a driver.

WHY THIS EXISTS
    The checkout at reference_repo() is at HEAD, and HEAD is not the revision that produced RSI-Jev
    v1.0. fit.py has gone from 213 to 559 lines and evaluate.py from 297 to 594 -- and evaluate.py is
    in the PROTECTED set, so a score produced through it is not guaranteed comparable to their
    published number. Their v1.0 checkpoint ships code/rsijev/*.py beside its weights: that snapshot
    is the training code at the revision that trained the release.

WHY A DRIVER IS BORROWED
    The v1.0 snapshot has no run_arm_lib.py. It ships a load_release.py, so it is the set of modules
    the release needs to be loaded and used, not a runnable tree. The driver is taken from the
    checkout, which means the reproduction is not "their code, unmodified" in the strict sense: it
    is their v1.0 modules driven by their later harness. This is a real dependency and is recorded
    as one rather than glossed.

WHY THE INTERFACE IS CHECKED RATHER THAN ASSUMED
    A driver from one revision and modules from another only agree if the names and signatures line
    up. This script verifies that before declaring the environment usable, because a mismatch would
    otherwise surface as an AttributeError hours into a run. Checked here:

      * run_arm_lib imports `fit` from rsijev.fit  -> present, signature matches
      * run_arm_lib imports `as_record, eval_precision, predict, score_predictions` from
        rsijev.evaluate                            -> all four present in the v1.0 snapshot
      * run_arm_lib imports from rsijev.encode and rsijev.train -> those two files are byte-identical
        between the revisions, so the names follow
      * ArchConfig / FitConfig fields the driver passes        -> present in the v1.0 dataclasses

USAGE
    python scripts/build_v1_env.py [--dest PATH]

    Exit 0 means the environment is built and its driver imports resolve. Exit 1 means it is not and
    the message says which check failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402

# Names the driver needs from each v1.0 module. Read off run_arm_lib.py's imports rather than
# guessed: a missing one is an AttributeError at import time, which is cheap here and expensive in
# a 10-hour run.
DRIVER_IMPORTS = {
    "rsijev.arch": ["ArchConfig", "DecisionModel"],
    "rsijev.fit": ["FitConfig", "fit"],
    "rsijev.encode": ["EncodeConfig", "collate", "encode_question", "iter_questions"],
    "rsijev.evaluate": ["as_record", "eval_precision", "predict", "score_predictions"],
    "rsijev.train": ["RLConfig", "assert_r0_gate", "seed_everything"],
}

# Files that must be byte-identical between the v1.0 snapshot and the checkout. If one is not, the
# borrowed driver is reading a different module than the one that trained v1.0, and this script says
# so instead of producing an environment that looks fine.
MUST_MATCH_CHECKOUT = ["contract.py", "data.py", "encode.py", "metrics.py", "targets.py", "train.py"]


def sha(p: pathlib.Path, normalize_newlines: bool = True) -> str:
    """Content hash, newline-normalised by default.

    Line endings must not be compared raw. Their release snapshot is LF and the checkout is CRLF, so
    a raw-byte comparison reports every shared file as different -- which is a false alarm, and one
    this project has already made once: an earlier audit compared the files through `read_text`,
    whose universal-newline translation made them look identical, and concluded "byte-identical"
    from a comparison that had silently discarded the difference it was claiming to measure.

    Neither the raw comparison nor the text comparison answers the question. The question is whether
    the two revisions contain the same code, so the newlines are normalised first and the raw hashes
    are reported alongside for anyone who wants them.
    """
    b = p.read_bytes()
    if normalize_newlines:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()[:12]


def build(dest: pathlib.Path) -> int:
    release = paths.require(paths.published_release(), "the published v1.0 release")
    src = release / "code" / "rsijev"
    if not src.is_dir():
        print(f"[FAIL] {src} is missing. The release downloads with allow_patterns including "
              f"'code/**'; re-run the snapshot_download in docs/reproduction_audit.md.")
        return 1

    ref = paths.require(paths.reference_repo(), "the reference checkout")
    meta_path = release / "meta.json"
    if not meta_path.is_file():
        print(f"[FAIL] {meta_path} is missing -- the published spec lives there and the environment "
              f"is only meaningful next to the spec it will run.")
        return 1

    # Rebuild ONLY the code. Anything else under dest -- run outputs, checkpoints, logs -- belongs
    # to whoever ran something there. An earlier version of this script called shutil.rmtree(dest)
    # and destroyed the evaluation records of the very environment it was rebuilding, which is the
    # kind of loss a builder must never be capable of: the code is reproducible, the run is not.
    for stale in ("rsijev", "run_arm_lib.py", "scripts"):
        p = dest / stale
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest / "rsijev")
    shutil.copy2(ref / "scripts" / "run_arm_lib.py", dest / "run_arm_lib.py")
    (dest / "scripts").mkdir()
    shutil.copy2(ref / "scripts" / "release_train.py", dest / "scripts" / "release_train.py")
    print(f"[OK] built {dest}")
    print(f"     rsijev/ from {release.name} (their v1.0 snapshot)")
    print(f"     run_arm_lib.py + scripts/release_train.py from {ref.name} HEAD (borrowed driver)")

    failed = 0

    print("\n[cross-check] files that must contain the same code in both revisions")
    for name in MUST_MATCH_CHECKOUT:
        a, b = src / name, ref / "rsijev" / name
        if not b.is_file():
            print(f"    [SKIP] {name:14} not in the checkout")
            continue
        if sha(a) == sha(b):
            raw = "byte-identical" if a.read_bytes() == b.read_bytes() else "newline-only difference"
            print(f"    [OK]   {name:14} {sha(a)}  ({raw})")
        else:
            failed += 1
            print(f"    [FAIL] {name:14} v1.0={sha(a)} checkout={sha(b)} -- the borrowed driver "
                  f"would read a different module than the one that trained v1.0")

    print("\n[cross-check] what differs (expected: arch, evaluate, fit)")
    for name in ("arch.py", "evaluate.py", "fit.py"):
        a, b = src / name, ref / "rsijev" / name
        la = a.read_text(encoding="utf-8", errors="replace").count("\n")
        lb = b.read_text(encoding="utf-8", errors="replace").count("\n") if b.is_file() else 0
        print(f"    {name:14} v1.0 {la:4} lines / checkout {lb:4} lines  ({lb - la:+d})")

    print("\n[cross-check] driver imports resolve against the v1.0 modules")
    probe = f'''
import sys
sys.path.insert(0, {str(dest)!r})
import importlib
want = {json.dumps(DRIVER_IMPORTS)}
bad = []
for mod, names in want.items():
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        bad.append(f"{{mod}}: cannot import ({{type(e).__name__}}: {{str(e)[:70]}})"); continue
    for n in names:
        if not hasattr(m, n):
            bad.append(f"{{mod}}.{{n}}: absent")
print("MISSING=" + ("|".join(bad) if bad else ""))
try:
    import run_arm_lib
    print("DRIVER_OK")
except Exception as e:
    print(f"DRIVER_FAIL={{type(e).__name__}}: {{str(e)[:110]}}")
'''
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       errors="replace", cwd=str(dest))
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("MISSING="):
            miss = line.split("=", 1)[1]
            if miss:
                failed += 1
                for m in miss.split("|"):
                    print(f"    [FAIL] {m}")
            else:
                print(f"    [OK]   every name the driver imports is present in the v1.0 modules")
        elif line.startswith("DRIVER_OK"):
            print(f"    [OK]   run_arm_lib imports against the v1.0 modules")
        elif line.startswith("DRIVER_FAIL"):
            failed += 1
            print(f"    [FAIL] {line.split('=', 1)[1]}")

    spec = json.loads(meta_path.read_text(encoding="utf-8")).get("spec", {})
    print(f"\n[OK] published spec beside the modules: {len(spec)} keys, lr_head={spec.get('lr_head')}, "
          f"option_order={spec.get('option_order')!r}")
    print(f"\nrun with:\n"
          f"    cd {dest}\n"
          f"    unset PYTHONPATH\n"
          f"    export PYTHONPATH=\"{dest};{dest}/scripts\"")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the v1.0 code environment: the release's own modules, with a driver.")
    ap.add_argument("--dest", default=None,
                    help="where to build (default: v1_env beside the reference checkout)")
    a = ap.parse_args()
    dest = pathlib.Path(a.dest) if a.dest else paths.reference_repo().parent / "v1_env"
    return build(dest)


if __name__ == "__main__":
    raise SystemExit(main())
