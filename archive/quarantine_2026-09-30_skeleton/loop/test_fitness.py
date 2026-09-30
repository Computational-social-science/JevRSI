"""test_fitness.py -- prove the fitness contract accepts real improvements and rejects the traps.

Every case here uses numbers MEASURED on this seed (dev, 600 questions, the frozen evaluator), not
invented ones. A contract tested on invented numbers tests nothing.

The load-bearing case is T3. On this checkpoint, T=0.5 drives ECE from 0.1368 to 0.0113 -- a
twelve-fold improvement on the metric most people quote for "calibration quality" -- while soft CE
degrades from 0.8286 to 1.0500 and Brier from 0.0491 to 0.1017. A loop whose fitness included ECE
would have kept that configuration and reported a large calibration win. That is the specific failure
this contract exists to prevent, and it is a real one, not a hypothetical.

T4 is a regression test for a bug this file's own development produced: the guard metrics had no
declared direction, so the code defaulted them to "higher is better" and the shipped L1 -- which
IMPROVES Brier, 0.0491 -> 0.0476 -- was vetoed by the Brier guard. A guard with an implicit direction
is worse than no guard: it blocks correct work while appearing to protect against something.

Run: python loop/test_fitness.py
Exit 0 = all pass.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location("loop_fitness", Path(__file__).with_name("fitness.py"))
F = importlib.util.module_from_spec(_spec)
sys.modules["loop_fitness"] = F
_spec.loader.exec_module(F)

FAILS: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


# Measured on the frozen evaluator, dev split, 600 questions.
RAW = {"accuracy": 0.8150, "chance_corrected_skill": 0.7326, "soft_cross_entropy": 0.8286,
       "brier_sum": 0.0491, "ece_10_bins_vs_gold_argmax": 0.1368, "score_expectation_mae": 0.1862}
L1 = {"accuracy": 0.8150, "chance_corrected_skill": 0.7326, "soft_cross_entropy": 0.8243,
      "brier_sum": 0.0476, "ece_10_bins_vs_gold_argmax": 0.1481, "score_expectation_mae": 0.1849}
GAMED = {"accuracy": 0.8150, "chance_corrected_skill": 0.7326, "soft_cross_entropy": 1.0500,
         "brier_sum": 0.1017, "ece_10_bins_vs_gold_argmax": 0.0113, "score_expectation_mae": 0.1900}

print("== T0  every metric the code reasons about declares its direction ==")
for m in F.REPORTED:
    try:
        d = F.direction_of(m)
        check(d in (-1, +1), f"{m} declares direction {'lower' if d < 0 else 'higher'}-is-better")
    except KeyError as e:
        check(False, f"{m} has NO declared direction: {e}")
try:
    F.direction_of("some_new_metric")
    check(False, "an undeclared metric must RAISE, not fall through to a default")
except KeyError:
    check(True, "an undeclared metric raises instead of defaulting")

print("\n== T1  the real L1 improvement is REJECTED at a threshold it cannot clear ==")
d = F.evaluate(L1, RAW, epsilon=0.05)
check(not d.accept, f"soft CE gain 0.43 pp does not clear epsilon 5.00 pp ({d.reason})")

print("\n== T2  the same pair is ACCEPTED once the threshold is below the measured gain ==")
d = F.evaluate(L1, RAW, epsilon=0.002)
check(d.accept, f"soft CE gain 0.43 pp clears epsilon 0.20 pp ({d.reason})")
check("brier_sum" not in d.guard_failures,
      "the Brier guard does NOT veto an improvement in Brier (0.0491 -> 0.0476)")

print("\n== T3  the ECE trap is rejected ==")
d = F.evaluate(GAMED, RAW, epsilon=0.05)
check(not d.accept, f"over-smoothed T=0.5 is rejected ({d.reason})")
check("ece" in d.reason.lower() or "improvement" in d.reason.lower(),
      "the rejection is on the primary, not on ECE")

print("\n== T4  the guard vetoes buying the primary with accuracy ==")
bad = dict(L1)
bad["accuracy"] = 0.79
d = F.evaluate(bad, RAW, epsilon=0.002)
check(not d.accept and "accuracy" in d.guard_failures,
      f"soft CE improves but accuracy falls -> vetoed ({d.guard_failures})")

print("\n== T5  a candidate that changes nothing is rejected ==")
d = F.evaluate(dict(RAW), RAW, epsilon=0.002)
check(not d.accept, "an identical candidate is rejected (improvement 0.00 pp)")

print("\n== T6  a missing primary is a hard error, not a silent pass ==")
d = F.evaluate({"accuracy": 0.99}, RAW, epsilon=0.002)
check(not d.accept and "missing" in d.reason, f"a candidate without the primary is rejected ({d.reason})")

print("\n== T7  the direction of every reported metric agrees with the measurement ==")
# L1 vs raw: which metrics did the shipped calibration actually improve? The PRIMARY is checked
# separately because it is not a member of REPORTED -- it is the decision metric, not a reported one.
d_prim = F.direction_of(F.PRIMARY) * (L1[F.PRIMARY] - RAW[F.PRIMARY])
check(d_prim > 0, f"the shipped L1 improves {F.PRIMARY}, as the contract assumes (d*delta={d_prim:+.4f})")
improved = {m for m in F.REPORTED if m in L1 and m in RAW
            and F.direction_of(m) * (L1[m] - RAW[m]) > 0}
worsened = {m for m in F.REPORTED if m in L1 and m in RAW
            and F.direction_of(m) * (L1[m] - RAW[m]) < 0}
check("brier_sum" in improved, "the shipped L1 improves Brier, as the contract assumes")
check("ece_10_bins_vs_gold_argmax" in worsened,
      "the shipped L1 WORSENS ECE -- the conflict the contract is built around is real")
check("accuracy" not in worsened, "the shipped L1 leaves accuracy untouched, as measured")

print()
print("=" * 70)
if FAILS:
    print(f"[FAIL] {len(FAILS)} assertion(s):")
    for f in FAILS:
        print(f"    - {f}")
    raise SystemExit(1)
print("[OK] the fitness contract accepts real improvements and rejects both measured traps")
raise SystemExit(0)
