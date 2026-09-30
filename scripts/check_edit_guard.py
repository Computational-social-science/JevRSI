#!/usr/bin/env python
"""
check_edit_guard.py -- enforce the EDITABLE / PROTECTED split declared in config/edit_guard.json.

WHY THIS EXISTS
    The loop is allowed to rewrite the model, the data and the training objective. It is not allowed
    to rewrite what a typed decision is, what is scored, or on which split. Without a mechanical
    check that distinction is a convention, and a convention is not what an arm's result rests on.

    The specific failure this guards against is not hypothetical. `agentjev/model.py` is imported by
    both `train.py` (editable) and `evaluate_split.py` (protected). RSI-Jev draws the same boundary
    between `arch.py` and `evaluate.py`, but their evaluator does not import their architecture. Ours
    does, so the protected surface cannot be a file -- it has to be the narrow interface between
    them, which is the `state_dict` key set and shapes. See docs/EDIT_GUARD.md.

WHAT IT CHECKS, and why each one
    1. COMPLETENESS. Every module in the subject repository is declared EDITABLE or PROTECTED, and an
       undeclared file is a FAIL rather than a default. A file nobody classified is a file whose
       protection nobody chose, and "the default is whatever the loop does" is how a scoring change
       becomes a model change without anybody deciding it.

    2. INTEGRITY. A PROTECTED file's sha256 matches the declaration. This is a hash and not a
       review, so it cannot be argued with, and it does not depend on the file's own docstring --
       a module that misdeclares its own protection is precisely the case the guard exists for.

    3. REACHABILITY. Every PROTECTED file exists and is readable. A protected path that has been
       moved or deleted is a guard that has silently stopped guarding.

WHY A HASH AND NOT AN IMPORT RULE
    `train.py` may import `model.py`, and `evaluate_split.py` must import `model.py`. An import rule
    could not separate those, so the boundary is pinned to content instead. The cost is that
    re-declaring is manual; the benefit is that re-declaring is visible in a diff and cannot happen
    as a side effect of a refactor.

THE NEGATIVE CONTROL
    `--self-test` proves the guard can fail. A guard that has never rejected anything is
    indistinguishable from a guard that cannot, and the difference only shows up on the one run
    where it matters. Run it after changing this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DECL_PATH = ROOT / "config" / "edit_guard.json"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_declaration() -> dict:
    if not DECL_PATH.exists():
        raise SystemExit(f"[fatal] {DECL_PATH.name} is missing. The guard reads a declaration, not a "
                         f"convention. Restore it from git, or re-declare with a human decision "
                         f"recorded.")
    return json.loads(DECL_PATH.read_text(encoding="utf-8"))


def subject_root() -> pathlib.Path:
    """The subject repository root, resolved through the project's single path configuration."""
    sys.path.insert(0, str(ROOT))
    from paths import require, subject  # noqa: E402

    return require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")


def discover_modules(root: pathlib.Path) -> list[str]:
    """Every Python module the loop could plausibly rewrite, as repo-relative posix paths.

    Scoped to `agentjev/` and `scripts/`, which is where the loop operates. The benchmark package is
    not enumerated because it is a separate repository whose every file is protected by
    construction -- it is upstream, and the loop has no write path to it.
    """
    out = []
    for sub in ("agentjev", "scripts"):
        d = root / sub
        if d.is_dir():
            out.extend(p.relative_to(root).as_posix() for p in sorted(d.rglob("*.py")))
    return out


def check(decl: dict, root: pathlib.Path) -> tuple[int, int, list[str]]:
    lines: list[str] = []
    failures = 0
    checks = 0

    editable = set(decl["_editable"])
    protected = dict(decl["_protected"])
    declared = editable | set(protected)

    # --- 1. completeness -------------------------------------------------------------------------
    present = discover_modules(root)
    undeclared = [m for m in present if m not in declared]
    for m in undeclared:
        checks += 1
        failures += 1
        lines.append(f"    [FAIL] UNDECLARED  {m}")
        lines.append(f"           The loop may rewrite this file and nobody classified it. Declare it "
                     f"EDITABLE or PROTECTED; if it is neither, move it out of the loop's reach.")
    if not undeclared:
        checks += 1
        lines.append(f"    [PASS] completeness   {len(present)} modules, all declared "
                     f"({len(editable)} EDITABLE, {len(protected)} PROTECTED)")

    # --- 2. declared-but-absent (an EDITABLE entry that no longer exists is a stale declaration) ---
    for m in sorted(editable):
        checks += 1
        if not (root / m).is_file():
            failures += 1
            lines.append(f"    [FAIL] STALE        {m} is declared EDITABLE but does not exist")
        else:
            lines.append(f"    [PASS] editable      {m}")

    # --- 3. integrity ----------------------------------------------------------------------------
    for rel, want in sorted(protected.items()):
        checks += 1
        p = root / rel
        if not p.is_file():
            failures += 1
            lines.append(f"    [FAIL] MISSING      {rel} -- a protected path that is gone is a guard "
                         f"that has stopped guarding")
            continue
        got = _sha256(p)
        if got != want:
            failures += 1
            lines.append(f"    [FAIL] MODIFIED     {rel}")
            lines.append(f"           declared {want[:16]}...  actual {got[:16]}...")
            lines.append(f"           Changing this file changes what is scored. That is a change to "
                         f"the task and goes through a human; re-declare only with the reason "
                         f"recorded in the commit.")
        else:
            lines.append(f"    [PASS] protected    {rel}  sha256 {got[:12]}...")

    return checks, failures, lines


def self_test(root: pathlib.Path, decl: dict) -> int:
    """Prove the guard rejects. A guard that has never failed is indistinguishable from no guard."""
    print("  Negative control: each case must FAIL, and the clean state must PASS.\n")
    cases = [
        ("a protected file is modified",
         {**decl, "_protected": {**decl["_protected"], "scripts/evaluate_split.py": "0" * 64}}),
        ("a module is undeclared",
         {**decl, "_editable": [m for m in decl["_editable"] if m != "agentjev/losses.py"]}),
        ("a declared EDITABLE module is missing",
         {**decl, "_editable": [*decl["_editable"], "agentjev/does_not_exist.py"]}),
        ("a protected file is missing",
         {**decl, "_protected": {**decl["_protected"], "scripts/evaluate_split.py": None}}),
    ]
    ok = True
    for name, bad in cases:
        d = dict(bad)
        # normalise the None case the way a real missing file presents itself
        p2 = {k: v for k, v in d["_protected"].items() if v is not None}
        d["_protected"] = p2
        _, fails, lines = check(d, root)
        verdict = "rejected" if fails else "ACCEPTED -- the guard is broken"
        if not fails:
            ok = False
        print(f"    {name:42} -> {verdict}")
        for l in lines:
            if "[FAIL]" in l:
                print(f"      {l.strip()[:100]}")

    _, fails, _ = check(decl, root)
    print(f"\n    {'clean state':42} -> {'rejected (WRONG)' if fails else 'passed'}")
    if fails:
        ok = False
    print(f"\n  {'[OK] the guard can fail and does not fail spuriously' if ok else '[FAIL] self-test failed'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Enforce the EDITABLE / PROTECTED split.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the guard rejects a modified, undeclared or missing file")
    a = ap.parse_args()

    decl = load_declaration()
    root = subject_root()

    print(f"[edit-guard] declaration: {DECL_PATH.relative_to(ROOT)}")
    print(f"[edit-guard] subject    : {root}  (from JEVRSI_SUBJECT)")
    if a.self_test:
        print()
        return self_test(root, decl)

    checks, failures, lines = check(decl, root)
    print("\n".join(lines))
    print(f"\n[{'OK' if failures == 0 else 'FAIL'}] {checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
