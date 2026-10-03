#!/usr/bin/env python
"""
check_edit_guard.py -- enforce the EDITABLE / PROTECTED split that the source code declares.

WHY THIS EXISTS
    The RSI-Jev source states the contract in its own README and marks it in every module:

        "Every module here opens by declaring itself EDITABLE or PROTECTED. That is the guard
         rail the autonomous loop runs inside ... an experiment may rewrite the editable modules
         and may not touch the protected ones, so a change to the model, the data or the recipe
         can never quietly become a change to how it is scored."

    An arm may move arch.py, data.py and train.py. It may not move encode.py, evaluate.py,
    targets.py, contract.py or metrics.py. Without a mechanical check that distinction is a
    convention, and a convention is not what an arm's result rests on.

WHAT IS GUARDED
    The code the arms actually execute: <v1_env>/rsijev/*.py. Not a copy, and not a declaration
    about some other repository -- the modules the driver imports when it runs a step.

WHY THE CLASSIFICATION IS READ FROM THE FILES
    The classification is a property of the module, so it is read from the module's own docstring.
    An earlier revision kept the classification in a JSON list instead, which meant a module could
    be edited to say PROTECTED while the list still said EDITABLE, and the guard would report the
    state the list described rather than the state on disk.

WHAT IT CHECKS
    1. COMPLETENESS.   Every module under rsijev/ declares a kind. An undeclared file is a FAIL,
                       not a default: a file nobody classified is one whose protection nobody
                       chose, and "default to whatever the loop does" is how a scoring change
                       becomes a model change without anyone deciding it.
    2. CLASSIFICATION. Each module's declared kind matches the contract in config/edit_guard.json.
                       This is the check the source does not have: it catches an arm that
                       reclassifies a guarded module in order to edit it "legally".
    3. AGREEMENT.      Each module's declaration also matches the reference repository's own copy
                       of that module. A reclassification that also rewrote the contract file is
                       caught here, because it cannot rewrite the reference.
    4. INTEGRITY.      Each guarded module's sha256 matches the recorded hash. A hash rather than a
                       review, so it cannot be argued with, and so it cannot change as a side
                       effect of a refactor.
    5. REACHABILITY.   Every module named by the contract exists. A guarded path that has moved is
                       a guard that has stopped guarding silently.

WHY fit.py IS HASHED WHILE THE SOURCE CALLS IT MIXED
    The source calls fit.py "PROTECTED in its bookkeeping, EDITABLE through its config". Its
    editable surface is the config keys the driver hands it -- those never change this file. So a
    change to fit.py's text is a change to the bookkeeping and goes through a human, and an arm
    that needs new behaviour adds a key through `fit_extra`, exactly as the driver documents.

WHY A HASH AND NOT AN IMPORT RULE
    An import rule cannot separate the two: the driver imports both faces, and the layout decision
    inside encode.py is reachable from every readout. The boundary is pinned to content. The cost
    is that re-declaring is manual; the benefit is that re-declaring appears in a diff.

THE NEGATIVE CONTROL IS NOT OPTIONAL
    A guard that has never rejected anything is indistinguishable from a guard that cannot, and the
    difference only surfaces on the run where it matters. `--self-test` injects five defects --
    modified, undeclared, reclassified, unsigned, stale -- and requires all five to be rejected,
    plus the two hash controls: presentation-blind and content-sensitive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import reference_repo, require, v1_env  # noqa: E402

DECL = ROOT / "config" / "edit_guard.json"


def sha256(path: pathlib.Path) -> str:
    """Content hash. Line-ending presentation is normalised away before hashing.

    Raw bytes are not content: the same file is CRLF after a Windows checkout and LF after a Linux
    one, so a byte hash makes this guard's verdict a function of which machine looked at the file
    rather than of whether the file changed. Measured 2026-10-03 on the subject repository: of
    three PROTECTED files, typed_decisions/experiment.py carried 156 CRLF pairs while the other two
    carried none -- so the byte hash would have reported MODIFIED for a content-identical file the
    moment it crossed a platform, and the honest repair would have looked like relaxing the guard.

    This does not weaken the check: any change to the text of any line still changes the hash.
    """
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


_DOC = re.compile(r'"""(.*?)"""', re.DOTALL)


def declared_kind(path: pathlib.Path) -> str:
    """The kind the module declares about itself, read from its docstring.

    Returns EDITABLE, PROTECTED, MIXED or "?" when there is no declaration. MIXED means the module
    states both words -- the source uses it for fit.py ("PROTECTED in its bookkeeping, EDITABLE
    through its config") and for contract.py / metrics.py ("editable files may not change this"),
    which are protected in effect. The contract file says which reading applies.

    The pattern is unbounded on purpose. A bounded `.{0,200}` cannot reach the closing quotes on the
    longer docstrings and silently reports every module as undeclared -- which reads as a total
    guard failure rather than as the regex bug it is.
    """
    m = _DOC.search(path.read_text(encoding="utf-8", errors="replace"))
    doc = (m.group(1) if m else "").upper()
    has_p, has_e = "PROTECTED" in doc, "EDITABLE" in doc
    if has_p and has_e:
        return "MIXED"
    if has_p:
        return "PROTECTED"
    if has_e:
        return "EDITABLE"
    return "?"


def declaration() -> dict:
    if not DECL.is_file():
        raise SystemExit(f"[fatal] {DECL.name} is missing. The guard reads a declaration, not a "
                         f"convention.")
    return json.loads(DECL.read_text(encoding="utf-8"))


def discover(rsijev: pathlib.Path) -> list[str]:
    """Every module the loop could rewrite, as directory-relative posix names."""
    return sorted(p.name for p in rsijev.glob("*.py"))


def check(decl: dict, rsijev: pathlib.Path,
          ref_rsijev: pathlib.Path | None) -> tuple[int, int, list[str]]:
    lines: list[str] = []
    failures = checks = 0
    contract: dict = decl["_contract"]
    hashes: dict = decl["_protected_sha256"]

    editable = sorted(k for k, v in contract.items() if v == "EDITABLE")
    guarded = sorted(k for k, v in contract.items() if v in ("PROTECTED", "MIXED"))

    # 1. completeness
    present = discover(rsijev)
    undeclared = [m for m in present if m not in contract]
    for m in undeclared:
        checks += 1
        failures += 1
        lines.append(f"    [FAIL] UNDECLARED   {m}")
        lines.append("           The loop may rewrite this and nobody classified it. Declare it "
                     "EDITABLE or PROTECTED, or move it out of the loop's reach.")
    if not undeclared:
        checks += 1
        lines.append(f"    [PASS] completeness {len(present)} modules, all classified "
                     f"({len(editable)} EDITABLE, {len(guarded)} guarded)")

    # 2 + 3. classification, and agreement with the reference
    mismatch = False
    for m in present:
        if m not in contract:
            continue
        checks += 1
        got = declared_kind(rsijev / m)
        want = contract[m]
        if got == "?":
            failures += 1
            mismatch = True
            lines.append(f"    [FAIL] NOCLAIM     {m} declares no kind")
            continue
        if got != want:
            failures += 1
            mismatch = True
            lines.append(f"    [FAIL] RECLASSIFIED {m}")
            lines.append(f"           declares {got}, contract says {want}")
            lines.append("           Reclassifying a guarded module is how a scoring change is "
                         "made to look like a model change.")
            continue
        if ref_rsijev and (ref_rsijev / m).is_file():
            ref_kind = declared_kind(ref_rsijev / m)
            if ref_kind != "?" and ref_kind != got:
                failures += 1
                mismatch = True
                lines.append(f"    [FAIL] DISAGREES   {m} declares {got}, the reference declares "
                             f"{ref_kind}")
    if not mismatch:
        lines.append("    [PASS] classification  every declaration matches the contract"
                     + (" and the reference" if ref_rsijev else " (reference unavailable)"))

    # 4 + 5. integrity and reachability
    for rel in guarded:
        checks += 1
        p = rsijev / rel
        if not p.is_file():
            failures += 1
            lines.append(f"    [FAIL] MISSING     {rel} -- a guarded path that is gone is a guard "
                         f"that has stopped guarding")
            continue
        want = hashes.get(rel)
        if want is None:
            failures += 1
            lines.append(f"    [FAIL] UNSIGNED    {rel} is guarded but has no recorded hash")
            continue
        got = sha256(p)
        if got != want:
            failures += 1
            lines.append(f"    [FAIL] MODIFIED    {rel}")
            lines.append(f"           recorded {want[:16]}...  actual {got[:16]}...")
            lines.append("           Changing this file changes what is scored. That is a change "
                         "to the task and goes through a human.")
        else:
            lines.append(f"    [PASS] protected   {rel}  sha256 {got[:12]}...")

    # stale contract entries: a name that is not on disk
    for m in sorted(contract):
        if not (rsijev / m).is_file():
            checks += 1
            failures += 1
            lines.append(f"    [FAIL] STALE       {m} is in the contract but does not exist")
    return checks, failures, lines


def self_test(rsijev: pathlib.Path, ref_rsijev: pathlib.Path | None, decl: dict) -> int:
    """Prove the guard rejects. Five injected defects, plus two hash controls."""
    print("  Negative control: every injected defect must be REJECTED, and the clean state PASSED.")
    print("  A guard that cannot fail is not a guard.\n")
    victim = "evaluate.py"
    free = "arch.py"
    cases = [
        ("a guarded file is modified",
         {**decl, "_protected_sha256": {**decl["_protected_sha256"], victim: "0" * 64}}),
        ("a module is undeclared",
         {**decl, "_contract": {k: v for k, v in decl["_contract"].items() if k != free}}),
        ("a guarded module is reclassified as EDITABLE",
         {**decl, "_contract": {**decl["_contract"], victim: "EDITABLE"}}),
        ("a guarded module has no recorded hash",
         {**decl, "_protected_sha256": {k: v for k, v in decl["_protected_sha256"].items()
                                        if k != victim}}),
        ("the contract names a module that is not there",
         {**decl, "_contract": {**decl["_contract"], "resurrected.py": "PROTECTED"}}),
    ]
    ok = True
    for name, bad in cases:
        _, fails, lines = check(bad, rsijev, ref_rsijev)
        if fails == 0:
            ok = False
        verdict = "rejected" if fails else "ACCEPTED -- the guard is broken"
        print(f"    {name:48} -> {verdict}")
        for l in lines:
            if "[FAIL]" in l:
                print(f"      {l.strip()[:94]}")

    # The hash controls. Without these, "fewer failures" is the only evidence, and fewer failures
    # is what a broken guard also produces.
    import tempfile
    src = (rsijev / victim).read_bytes().replace(b"\r\n", b"\n")
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        (tdp / "lf.py").write_bytes(src)
        (tdp / "crlf.py").write_bytes(src.replace(b"\n", b"\r\n"))
        (tdp / "edited.py").write_bytes(src.replace(b"def ", b"def  ", 1))
        h_lf, h_crlf, h_edit = (sha256(tdp / "lf.py"), sha256(tdp / "crlf.py"),
                                sha256(tdp / "edited.py"))
    present_ok = h_lf == h_crlf
    content_ok = h_lf != h_edit
    print(f"    {'LF and CRLF forms of one file hash alike':48} -> "
          f"{'accepted (presentation-blind)' if present_ok else 'DIFFER -- still byte-sensitive'}")
    print(f"    {'a one-character edit hashes differently':48} -> "
          f"{'rejected (content-sensitive)' if content_ok else 'ACCEPTED -- the hash is blind'}")
    if not (present_ok and content_ok):
        ok = False

    _, fails, _ = check(decl, rsijev, ref_rsijev)
    if fails:
        ok = False
    print(f"\n    {'clean state':48} -> {'rejected (WRONG)' if fails else 'passed'}")
    print(f"\n  {'[OK] the guard can fail and does not fail spuriously' if ok else '[FAIL] self-test failed'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Enforce the source's EDITABLE / PROTECTED split.")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the guard rejects a modified, undeclared, reclassified, "
                         "unsigned or stale module")
    ap.add_argument("--show-hashes", action="store_true",
                    help="print the content hashes to record in the declaration")
    a = ap.parse_args()

    decl = declaration()
    rsijev = require(v1_env(), "the v1.0 code environment") / "rsijev"
    if not rsijev.is_dir():
        raise SystemExit(f"[fatal] {rsijev} is not a directory -- build it with "
                         f"scripts/build_v1_env.py")
    try:
        ref_rsijev = require(reference_repo(), "the RSI-Jev reference repository") / "rsijev"
        if not ref_rsijev.is_dir():
            ref_rsijev = None
    except SystemExit:
        ref_rsijev = None

    print(f"[edit-guard] declaration: {DECL.relative_to(ROOT)}")
    print(f"[edit-guard] guarded    : {rsijev}")
    print(f"[edit-guard] reference  : {ref_rsijev if ref_rsijev else 'unavailable (skipped)'}")

    if a.show_hashes:
        print()
        for rel in sorted(decl["_protected_sha256"]):
            p = rsijev / rel
            if not p.is_file():
                print(f"  {rel:16} MISSING")
                continue
            got = sha256(p)
            cur = decl["_protected_sha256"][rel]
            print(f"  {rel:16} {got}")
            print(f"  {'':16} recorded {cur}   "
                  f"({'unchanged' if got == cur else 'DIFFERS -> re-declare deliberately'})")
        print("\n  Re-declaring is a human edit to config/edit_guard.json. Per the source's own"
              "\n  contract it is for changing the task, never for making an arm pass.")
        return 0

    if a.self_test:
        print()
        return self_test(rsijev, ref_rsijev, decl)

    checks, failures, lines = check(decl, rsijev, ref_rsijev)
    print("\n".join(lines))
    print(f"\n[{'OK' if failures == 0 else 'FAIL'}] {checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
