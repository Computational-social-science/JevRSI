#!/usr/bin/env python
"""
check_edit_guard.py -- enforce the EDITABLE / PROTECTED split declared in config/edit_guard.json.

WHY THIS EXISTS
    The loop may rewrite the model, the data and the training objective. It may not rewrite what a
    typed decision is, what is scored, or on which split. Without a mechanical check that
    distinction is a convention, and a convention is not what an arm's result rests on.

    The specific hazard here is not hypothetical. In this subject repository:

        train.py            (editable)   imports   model.py
        evaluate_split.py  (protected)   imports   model.py

    RSI-Jev draws the same boundary between arch.py and evaluate.py, but their evaluator does not
    import their architecture and ours does. So the protected surface cannot be a FILE -- it is the
    narrow interface between the editable and the protected, which measurement shows to be the
    `state_dict` key set and shapes. See docs/EDIT_GUARD.md.

WHAT IT CHECKS
    1. COMPLETENESS. Every module under agentjev/ and scripts/ is declared. An undeclared file is a
       FAIL, not a default: a file nobody classified is one whose protection nobody chose, and
       "default to whatever the loop does" is how a scoring change becomes a model change without
       anyone deciding it.
    2. INTEGRITY. Each PROTECTED file's sha256 matches the declaration. A hash rather than a review,
       so it cannot be argued with, and so it cannot change as a side effect of a refactor.
    3. REACHABILITY. Each PROTECTED file exists. A protected path that has moved is a guard that has
       stopped guarding silently.

WHY A HASH AND NOT AN IMPORT RULE
    train.py may import model.py and evaluate_split.py must import model.py. No import rule separates
    those, so the boundary is pinned to content. The cost is that re-declaring is manual; the
    benefit is that re-declaring appears in a diff.

THE NEGATIVE CONTROL IS NOT OPTIONAL
    A guard that has never rejected anything is indistinguishable from a guard that cannot, and the
    difference only surfaces on the run where it matters. `--self-test` injects four defects and
    requires all four to be rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import require, subject  # noqa: E402

DECL = ROOT / "config" / "edit_guard.json"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declaration() -> dict:
    if not DECL.is_file():
        raise SystemExit(f"[fatal] {DECL.name} is missing. The guard reads a declaration, not a "
                         f"convention.")
    return json.loads(DECL.read_text(encoding="utf-8"))


def discover(root: pathlib.Path) -> list[str]:
    """Every module the loop could rewrite, as repo-relative posix paths.

    Scoped to agentjev/ and scripts/, where the loop operates. The benchmark package is excluded
    because it is upstream and the loop has no write path to it; its single entry is protected by
    construction.
    """
    out = []
    for sub in ("agentjev", "scripts"):
        d = root / sub
        if d.is_dir():
            out.extend(p.relative_to(root).as_posix() for p in sorted(d.rglob("*.py")))
    return out


def check(decl: dict, root: pathlib.Path) -> tuple[int, int, list[str]]:
    lines: list[str] = []
    failures = checks = 0
    editable = set(decl["_editable"])
    protected = dict(decl["_protected"])
    declared = editable | set(protected)

    # 1. completeness
    present = discover(root)
    undeclared = [m for m in present if m not in declared]
    for m in undeclared:
        checks += 1
        failures += 1
        lines.append(f"    [FAIL] UNDECLARED  {m}")
        lines.append("           The loop may rewrite this and nobody classified it. Declare it "
                     "EDITABLE or PROTECTED, or move it out of the loop's reach.")
    if not undeclared:
        checks += 1
        lines.append(f"    [PASS] completeness  {len(present)} modules declared "
                     f"({len(editable)} EDITABLE, {len(protected)} PROTECTED)")

    # 2. a declared EDITABLE module that no longer exists is a stale declaration
    for m in sorted(editable):
        checks += 1
        if not (root / m).is_file():
            failures += 1
            lines.append(f"    [FAIL] STALE       {m} is declared EDITABLE but does not exist")
    stale = [m for m in sorted(editable) if not (root / m).is_file()]
    if not stale:
        lines.append(f"    [PASS] editable     all {len(editable)} declared EDITABLE modules present")

    # 3. integrity and reachability
    for rel, want in sorted(protected.items()):
        checks += 1
        p = root / rel
        if not p.is_file():
            failures += 1
            lines.append(f"    [FAIL] MISSING     {rel} -- a protected path that is gone is a guard "
                         f"that has stopped guarding")
            continue
        got = sha256(p)
        if got != want:
            failures += 1
            lines.append(f"    [FAIL] MODIFIED    {rel}")
            lines.append(f"           declared {want[:16]}...  actual {got[:16]}...")
            lines.append("           Changing this file changes what is scored. That is a change to "
                         "the task and goes through a human.")
        else:
            lines.append(f"    [PASS] protected   {rel}  sha256 {got[:12]}...")

    return checks, failures, lines


def self_test(root: pathlib.Path, decl: dict) -> int:
    """Prove the guard rejects. Four injected defects, all four must be caught."""
    print("  Negative control: every injected defect must be REJECTED, and the clean state PASSED.")
    print("  A guard that cannot fail is not a guard.\n")
    victim = "scripts/evaluate_split.py"
    cases = [
        ("a protected file is modified",
         {**decl, "_protected": {**decl["_protected"], victim: "0" * 64}}),
        ("a module is undeclared",
         {**decl, "_editable": [m for m in decl["_editable"] if m != "agentjev/losses.py"]}),
        ("a declared EDITABLE module is missing",
         {**decl, "_editable": [*decl["_editable"], "agentjev/does_not_exist.py"]}),
        ("a protected file is missing",
         {**decl, "_protected": {k: v for k, v in decl["_protected"].items() if k != victim}}),
    ]
    ok = True
    for name, bad in cases:
        _, fails, lines = check(bad, root)
        if fails == 0:
            ok = False
        verdict = "rejected" if fails else "ACCEPTED -- the guard is broken"
        print(f"    {name:44} -> {verdict}")
        for l in lines:
            if "[FAIL]" in l:
                print(f"      {l.strip()[:98]}")
    _, fails, _ = check(decl, root)
    if fails:
        ok = False
    print(f"\n    {'clean state':44} -> {'rejected (WRONG)' if fails else 'passed'}")
    print(f"\n  {'[OK] the guard can fail and does not fail spuriously' if ok else '[FAIL] self-test failed'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Enforce the EDITABLE / PROTECTED split.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the guard rejects a modified, undeclared or missing file")
    a = ap.parse_args()

    decl = declaration()
    root = require(subject(), "the agent-jev subject repository")
    print(f"[edit-guard] declaration: {DECL.relative_to(ROOT)}")
    print(f"[edit-guard] subject    : {root}")

    if a.self_test:
        print()
        return self_test(root, decl)

    checks, failures, lines = check(decl, root)
    print("\n".join(lines))
    print(f"\n[{'OK' if failures == 0 else 'FAIL'}] {checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
