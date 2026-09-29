"""loop/fitness.py -- the fitness contract: WHICH metric decides keep/revert.

WHY THIS FILE EXISTS
--------------------
The proposal names a fitness function seven times ("fitness_M", "Delta fitness >= eps_M",
"best-so-far on medium set M") and never defines it. Two things then go wrong, and both are silent.

  1. Top-1 cannot see the P0 family. Temperature scaling is a monotone transform of the logits, so
     softmax(z/T) and softmax(z) have the SAME argmax for every item. Measured on this seed: raw
     (T=1.0) and the shipped L1 temperatures both give accuracy 0.8150 and chance-corrected 0.7326,
     to the digit. A fitness built on accuracy is therefore BLIND to every calibration candidate the
     proposal ranks as priority 0.

  2. ECE points the other way. On the same logits, the shipped L1 temperature IMPROVES soft CE
     (0.8286 -> 0.8243) and Brier (0.0491 -> 0.0476) while making ECE WORSE (0.1368 -> 0.1481), on
     all three question types. So a fitness that averages accuracy with ECE receives two signals
     that disagree in sign, and the loop's decisions become a function of an arbitrary weighting that
     nobody wrote down.

THE CONTRACT
------------
    PRIMARY   soft cross-entropy against the teacher distribution. Lower is better.
              It is a strictly proper scoring rule, so it cannot be gamed by collapsing toward the
              marginal or by sharpening: both are penalised. That is the property that matters, because
              the earlier alternative (ECE) is a binned statistic and IS gameable -- on this very
              checkpoint, T=0.5 drives ECE to 0.0113 while soft CE degrades from 0.8286 to 1.0500 and
              Brier from 0.0491 to 0.1017. A loop optimising ECE would have kept that.

    REPORTED  accuracy, chance-corrected skill, Brier, ECE, score MAE -- all computed, none of them
              allowed to influence a decision.

The rule is one-directional on purpose: a candidate must not be able to improve the primary by
degrading a reported metric past its baseline. A configuration that buys soft CE with a collapsed
mean confidence is rejected even though the primary improved.

Run: python loop/fitness.py --explain
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PRIMARY = "soft_cross_entropy"

# DIRECTION IS DECLARED, NEVER INFERRED. An earlier revision kept a set of "lower is better" metrics
# containing only the primary and defaulted every other metric to "higher is better". That silently
# inverted the Brier guard: the shipped L1 lowers Brier from 0.0491 to 0.0476, which is an
# IMPROVEMENT, and the guard rejected it. A guard with an implicit direction is worse than no guard,
# because it vetoes correct work while looking like it is protecting something. Every metric the code
# reasons about now states its direction explicitly.
LOWER_IS_BETTER = {
    "soft_cross_entropy",
    "brier_sum",
    "ece_10_bins_vs_gold_argmax",
    "score_expectation_mae",
}

# Reported but never decisive. `baseline_guard` marks the ones a candidate may not degrade while
# buying the primary -- ECE is excluded from the guard because it is in direct conflict with the
# primary on this checkpoint (measured), and guarding it would forbid every calibration candidate.
REPORTED = ("accuracy", "chance_corrected_skill", "brier_sum", "ece_10_bins_vs_gold_argmax",
            "score_expectation_mae")
BASELINE_GUARD = ("accuracy", "brier_sum")


def direction_of(metric: str) -> int:
    """+1 if higher is better, -1 if lower is better. Raises for an unknown metric.

    Raising is the point: an undeclared metric must not fall through to a default, because the default
    is what produced the inverted-Brier veto.
    """
    if metric in LOWER_IS_BETTER:
        return -1
    if metric in ("accuracy", "chance_corrected_skill"):
        return +1
    raise KeyError(f"metric {metric!r} has no declared direction; add it to LOWER_IS_BETTER or to "
                   f"the higher-is-better tuple rather than relying on a default")

# Measured evidence behind the contract, so the contract is falsifiable rather than asserted.
EVIDENCE = {
    "temperature_does_not_change_top1": {
        "claim": "accuracy and chance-corrected skill are identical for T=1.0 and the shipped L1",
        "measured": {"accuracy": [0.8150, 0.8150], "chance_corrected_skill": [0.7326, 0.7326]},
        "consequence": "a fitness on Top-1 is blind to the entire P0 calibration family",
    },
    "ece_conflicts_with_proper_scoring_rules": {
        "claim": "the shipped L1 improves soft CE and Brier while worsening ECE, on all 3 types",
        "measured": {"soft_cross_entropy": [0.8286, 0.8243], "brier_sum": [0.0491, 0.0476],
                     "ece": [0.1368, 0.1481],
                     "ece_by_type": {"boolean": [0.1052, 0.1139], "choice": [0.1279, 0.1352],
                                     "score": [0.1732, 0.1888]}},
        "consequence": "averaging ECE into the fitness yields two signals of opposite sign",
    },
    "ece_is_gameable_on_this_checkpoint": {
        "claim": "T=0.5 drives ECE to 0.0113 while both proper scoring rules degrade",
        "measured": {"T": [1.0, 0.5], "ece": [0.1368, 0.0113],
                     "soft_cross_entropy": [0.8286, 1.0500], "brier_sum": [0.0491, 0.1017],
                     "mean_confidence": [0.6782, 0.8060]},
        "consequence": "a loop optimising ECE would have kept the over-smoothed configuration",
    },
}


@dataclass
class Decision:
    accept: bool
    primary_delta: float
    epsilon: float
    reason: str
    guard_failures: list[str]


def evaluate(candidate: dict, incumbent: dict, epsilon: float) -> Decision:
    """Decide keep/revert for one candidate against its parent.

    `candidate` and `incumbent` are the metric dicts the frozen evaluator produces. The decision is
    made on the PRIMARY alone, then the guard is applied as a veto -- never as a weighted term,
    because a weighted term is a hidden hyperparameter and this project has already been bitten by
    one (the proposal's undefined fitness).
    """
    if PRIMARY not in candidate or PRIMARY not in incumbent:
        missing = [m for m in (PRIMARY,) if m not in candidate or m not in incumbent]
        return Decision(False, float("nan"), epsilon,
                         f"primary metric {PRIMARY} missing from {missing}", [])

    c, i = float(candidate[PRIMARY]), float(incumbent[PRIMARY])
    delta = c - i                      # negative = better, since LOWER_IS_BETTER
    improvement = -delta               # positive = better

    if improvement <= epsilon:
        return Decision(False, delta, epsilon,
                        f"improvement {improvement * 100:+.2f} pp does not clear epsilon "
                        f"{epsilon * 100:.2f} pp", [])

    guard_failures = []
    for m in BASELINE_GUARD:
        if m in candidate and m in incumbent:
            # `d * delta` is POSITIVE when the candidate moved in the GOOD direction, because d is -1
            # for lower-is-better and +1 for higher-is-better. So a guard fires when that product is
            # NEGATIVE. (Getting this sign backwards is what made an earlier revision veto the shipped
            # L1 for "degrading" a Brier it had actually improved.)
            d = direction_of(m)
            if d * (float(candidate[m]) - float(incumbent[m])) < 0:
                guard_failures.append(m)

    if guard_failures:
        return Decision(False, delta, epsilon,
                        f"primary improved by {improvement * 100:+.2f} pp but the guard metric(s) "
                        f"{', '.join(guard_failures)} degraded", guard_failures)

    return Decision(True, delta, epsilon,
                    f"primary improved by {improvement * 100:+.2f} pp > epsilon "
                    f"{epsilon * 100:.2f} pp, guards intact", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--candidate", default=None, help="JSON file with the candidate's metrics")
    ap.add_argument("--incumbent", default=None, help="JSON file with the incumbent's metrics")
    ap.add_argument("--epsilon", type=float, default=0.05)
    a = ap.parse_args()

    if a.explain or not (a.candidate and a.incumbent):
        print("=" * 74)
        print("fitness contract")
        print("=" * 74)
        print(f"  PRIMARY  {PRIMARY}   (lower is better; a strictly proper scoring rule)")
        print(f"  REPORTED {', '.join(REPORTED)}")
        print(f"  GUARD    {', '.join(BASELINE_GUARD)}  (may not degrade to buy the primary)")
        print()
        for k, v in EVIDENCE.items():
            print(f"  - {k}")
            print(f"      {v['claim']}")
            print(f"      -> {v['consequence']}")
        print()
        print("  A keep/revert decision is made on the primary ALONE, then the guard applies as a")
        print("  veto. It is never a weighted term: a weight is an unwritten hyperparameter.")
        return 0

    cand = json.loads(Path(a.candidate).read_text(encoding="utf-8"))
    inc = json.loads(Path(a.incumbent).read_text(encoding="utf-8"))
    d = evaluate(cand, inc, a.epsilon)
    print(json.dumps({"accept": d.accept, "primary_delta_pp": d.primary_delta * 100,
                      "epsilon_pp": d.epsilon * 100, "reason": d.reason,
                      "guard_failures": d.guard_failures}, indent=2))
    return 0 if d.accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
