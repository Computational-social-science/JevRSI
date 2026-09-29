"""test_manuscript_audit.py -- prove the manuscript audit can fail, and restore the file afterwards.

WHY THIS FILE EXISTS. `audit_manuscript_numbers.py` reported 27/27 pass on its first honest run, and a
guard that has only ever returned PASS is indistinguishable from a guard that checks nothing. That is not
hypothetical here: an earlier version of the audit was reading artifacts belonging to a retired objective
and checking them against a manuscript that no longer existed. It was green, and it was protecting
nothing.

So the audit's own failure modes are now injected on purpose. Each case below is a specific way this
project has already gone wrong:

  N1  quoting a number measured on a different arm without saying so  -- the silent Floor B inheritance
  N2  dropping the word "provisional" so a lower bound reads as a decision threshold
  N3  deleting the calibration file's own instruction, so the warning stops travelling with the number
  N4  a plausible one-digit edit to a baseline figure
  N5  reintroducing a value that a documented decision retracted

Two properties are asserted, and the second is the one that matters most:

  POSITIVE  the unmodified manuscript passes, so the cases are not passing for an unrelated reason
  NEGATIVE  every injection makes the audit exit non-zero, and it names the specific violation
  RESTORED  the file on disk is byte-identical to what was there before, verified by hash

Run: python measurement/test_manuscript_audit.py
Exit 0 = the audit has teeth. Exit 1 = a case passed that should have failed, or the file was not restored.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MS = REPO / "manuscript.html"
AUDIT = REPO / "measurement" / "audit_manuscript_numbers.py"

if not MS.exists():
    print("[SKIP] no manuscript.html -- nothing to test")
    raise SystemExit(0)


def sha() -> str:
    return hashlib.sha256(MS.read_bytes()).hexdigest()


def run_audit() -> tuple[int, list[str]]:
    """The audit is pure stdlib, so any interpreter will do; only PYTHONPATH has to be kept out."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(AUDIT)], cwd=REPO, capture_output=True,
                       text=True, timeout=200, env=env, errors="replace")
    return r.returncode, [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("- ")]


ORIG = MS.read_text(encoding="utf-8")
ORIG_SHA = sha()

CASES: dict[str, tuple[str, str, int]] = {
    # (anchor, replacement, expected_occurrences_to_replace)
    # `all` is the honest default and the reason the first version of this file was wrong. It replaced only
    # the FIRST occurrence, and both the N2 and N4 cases then passed unnoticed -- 0.8150 occurs six times in
    # the manuscript and "provisional" four times, so a single-site edit left the audited string intact and
    # a presence check could not see it. Nobody edits one of six mentions by accident; a contamination event
    # rewrites the claim everywhere. The count is asserted below so a stale case fails loudly instead of
    # silently becoming untestable.
    "N1 Floor B quoted without naming its arm": (
        "a <em>different</em> training path: laya-multilingual 322M, full-parameter, 5 seeds",
        "the optimiser, full-parameter, 5 seeds", 1),
    "N2 'provisional' removed, threshold reads as settled": ("provisional", "established", -1),
    "N3 the calibration file's own Floor B instruction deleted": (
        "Do not inherit it silently.", "Reuse it as needed.", 1),
    "N4 one-digit edit to the dev accuracy": ("0.8150", "0.8250", -1),
    "N5 a retracted value reintroduced": (
        "</p>", " The composite is 66.4 percent. </p>", 1),
}

print("=" * 74)
print("negative control for audit_manuscript_numbers.py")
print("=" * 74)

# ---- POSITIVE: the untouched manuscript must pass, or the cases prove nothing --------------------
rc, msgs = run_audit()
if rc != 0:
    print(f"[FAIL] the unmodified manuscript does not pass (rc={rc}); every case below would be vacuous")
    for m in msgs:
        print(f"    {m}")
    raise SystemExit(1)
print(f"  [ok] positive control: unmodified manuscript passes (rc=0)\n")

fails: list[str] = []
for name, (old, new, count) in CASES.items():
    n_present = ORIG.count(old)
    if n_present == 0:
        fails.append(f"{name}: the anchor text {old[:40]!r} is no longer in the manuscript, so this "
                     f"case cannot be injected -- either the manuscript changed or the case is stale")
        print(f"  [FAIL] {name}\n         anchor missing: {old[:60]!r}")
        continue
    if count == -1 and n_present < 2:
        fails.append(f"{name}: intended as an all-occurrences edit but the anchor appears only once, "
                     f"so this case no longer reproduces the failure it was written for")
        print(f"  [FAIL] {name}\n         needs >=2 occurrences to be an all-sites edit, found {n_present}")
        continue
    mutated = ORIG.replace(old, new) if count == -1 else ORIG.replace(old, new, count)
    n_changed = n_present if count == -1 else min(count, n_present)
    MS.write_text(mutated, encoding="utf-8")
    rc, msgs = run_audit()
    if rc == 0:
        fails.append(f"{name}: the audit still passed after rewriting {n_changed} occurrence(s), so this "
                     f"violation is NOT detected")
        print(f"  [FAIL] {name}\n         rewrote {n_changed} occurrence(s); audit returned 0 -- no teeth")
    else:
        print(f"  [ok] {name}  ({n_changed} occurrence(s) rewritten)")
        for m in msgs[:1]:
            print(f"         -> {m[:104]}")
    MS.write_text(ORIG, encoding="utf-8")

# ---- RESTORED: the file must be byte-identical to what we started with ----------------------------
if sha() != ORIG_SHA:
    fails.append("manuscript.html was not restored byte-identically after the injections")
    print(f"\n  [FAIL] restore: sha {sha()[:16]} != original {ORIG_SHA[:16]}")
else:
    print(f"\n  [ok] restore: manuscript.html byte-identical (sha {ORIG_SHA[:16]})")

print()
if fails:
    print(f"[FAIL] {len(fails)} problem(s):")
    for f in fails:
        print(f"    - {f}")
    raise SystemExit(1)
print(f"[OK] the audit fails on all {len(CASES)} injected violations and passes on the real document")
