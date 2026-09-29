"""loop/epsilon.py -- derive the two decision thresholds the live proposal leaves undefined.

THE GAP THIS FILLS
------------------
The proposal's multi-fidelity rule reads:

    "Evaluate on proxy set P. If improvement over parent is below eps_proxy, discard immediately.
     Otherwise evaluate on medium set M. Keep only if delta_fitness >= eps_M and the lower bound
     of a bootstrap 90% CI is positive."

Both epsilons are named and neither is defined. They are not free parameters: each is a decision
threshold, and a decision threshold set by taste is a selection operator on noise. This project has
already measured what that costs -- on this benchmark the naive rule ("accept if better") fired on
5 of 10 pure-noise comparisons, so about half of all accepts would be noise.

So both are derived here from two measured quantities, and the derivation is explicit enough to
audit:

    eps = z * SE_paired * sqrt(1 + r) * multiplicity

where
    SE_paired  the paired standard error of a difference on that split (questions nest inside
               cases, so this is a CLUSTER bootstrap, not a binomial SE -- the measured design
               effect on the 400-case test split is 1.384, i.e. the binomial SE understates it by
               ~32%);
    r          the replica count, since a candidate is compared against a parent that is itself one
               draw: the variance of the difference is var(candidate) + var(parent);
    z          1.96 for a two-sided 95% rule, 1.645 for a two-sided 90% rule;
    multiplicity
               Bonferroni over the horizon H, because a loop that runs H decisions pays for all H.

FLOOR A TRANSFERS, FLOOR B DOES NOT
-----------------------------------
Floor A (measurement sampling) is a property of the SPLIT: the same 400 cases nest the same way
whatever model is scored on them, so its SE carries over to a new seed. Floor B (run-to-run SD) is
a property of the TRAINING PATH, and QLoRA on a different backbone has different numerical noise, so
it must be re-measured on the new seed before its threshold is used. `require_floor_b` defaults to
True for exactly that reason: it refuses to hand out a threshold built on another model's noise.

Run: python loop/epsilon.py --split dev --replicas 1 --horizon 1000
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CALIB = REPO / "measurement" / "INSTRUMENT_CALIBRATION.json"

# Two-sided normal quantiles. Hard-coded rather than pulled from scipy so this module has no
# dependency beyond the standard library, and so the numbers are auditable by eye.
Z = {0.90: 1.6449, 0.95: 1.959964}


@dataclass
class Epsilon:
    split: str
    eps: float
    se_paired: float
    replicas: int
    horizon: int
    confidence: float
    floor_b_pp: float | None
    floor_a_pp: float | None
    bonferroni: float
    note: str

    @property
    def eps_pp(self) -> float:
        return self.eps * 100


def load_calibration() -> dict:
    return json.loads(CALIB.read_text(encoding="utf-8"))


def bonferroni_z(confidence: float, horizon: int) -> tuple[float, float]:
    """Return (per-decision alpha, z) after Bonferroni over `horizon` decisions.

    The correction is what makes a long loop honest. A loop that runs 1000 decisions at a nominal
    5% spends a family-wise error rate of about 1 - 0.95**1000, which is 1.0 to three decimals: a
    rule with no multiplicity control accepts essentially everything it sees.
    """
    alpha = (1.0 - confidence) / max(1, horizon)
    # z for a two-sided interval at this (already divided) per-decision level
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return alpha, z


def derive(split: str, se_paired: float, floor_b: float | None, replicas: int,
           horizon: int, confidence: float, require_floor_b: bool = True) -> Epsilon:
    if floor_b is None and require_floor_b:
        raise ValueError(
            "Floor B (run-to-run SD) is required and was not supplied. It is a property of the "
            "TRAINING PATH, not of the split: a QLoRA adapter on a different backbone has different "
            "numerical noise, so the previous arm's value does not transfer. Measure it on this seed "
            "(>= 3 seeds) or pass require_floor_b=False to proceed deliberately without it.")
    # var of a difference against a parent that is itself one draw
    se = se_paired * math.sqrt(1.0 + max(1, replicas))
    if floor_b is not None:
        se = math.sqrt(se ** 2 + floor_b ** 2)
    alpha, z = bonferroni_z(confidence, horizon)
    eps = z * se
    return Epsilon(
        split=split, eps=eps, se_paired=se, replicas=replicas, horizon=horizon,
        confidence=confidence,
        floor_b_pp=None if floor_b is None else floor_b * 100,
        floor_a_pp=se_paired * 100,
        bonferroni=alpha,
        note="eps = z_bonf * sqrt(se_paired^2 * (1 + replicas) + floor_b^2). "
             "The (1 + replicas) term is the parent's own variance; the Bonferroni factor pays for "
             "every decision the loop will make, not just this one.")


def from_calibration(split: str, replicas: int, horizon: int, confidence: float,
                     require_floor_b: bool = True) -> Epsilon:
    """Derive from the frozen instrument calibration, for splits whose SE is already measured."""
    c = load_calibration()
    if split == "test":
        se = c["floor_A_measurement_sampling"]["se_pp"] / 100
    elif split == "dev":
        se = c["dev_split_resolution"]["acc_se_pp"] / 100
    else:
        raise ValueError(f"no measured SE for split {split!r}; measure it with "
                         f"floor_a_cluster_bootstrap.py first")
    fb = c["floor_B_run_to_run"]["sd"] if not require_floor_b else None
    return derive(split, se, fb, replicas, horizon, confidence, require_floor_b=require_floor_b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--replicas", type=int, default=1,
                    help="how many independent evaluations the candidate is an average of")
    ap.add_argument("--horizon", type=int, default=1000,
                    help="decisions the loop plans to make; drives the Bonferroni correction")
    ap.add_argument("--confidence", type=float, default=0.95, choices=(0.90, 0.95))
    ap.add_argument("--se-paired", type=float, default=None,
                    help="override the measured paired SE (fraction)")
    ap.add_argument("--floor-b", type=float, default=None,
                    help="run-to-run SD (fraction); omit to derive without it")
    ap.add_argument("--allow-missing-floor-b", action="store_true",
                    help="proceed without Floor B -- deliberate, and recorded in the output")
    a = ap.parse_args()

    try:
        if a.se_paired is not None:
            e = derive(a.split, a.se_paired, a.floor_b, a.replicas, a.horizon, a.confidence,
                       require_floor_b=a.floor_b is not None or not a.allow_missing_floor_b)
        else:
            e = from_calibration(a.split, a.replicas, a.horizon, a.confidence,
                                 require_floor_b=not a.allow_missing_floor_b)
    except ValueError as ex:
        print(f"[BLOCKED] {ex}")
        return 2

    print("=" * 74)
    print(f"decision threshold for split {e.split!r}  "
          f"(H={e.horizon}, {e.confidence:.0%} family, {e.replicas} replica(s))")
    print("=" * 74)
    print(f"  paired SE (measured)      {e.floor_a_pp:6.2f} pp")
    print(f"  floor B (run-to-run)      "
          f"{'n/a -- derived without it' if e.floor_b_pp is None else f'{e.floor_b_pp:6.2f} pp'}")
    print(f"  SE after replica term     {e.se_paired * 100:6.2f} pp")
    print(f"  Bonferroni per-decision   {e.bonferroni:.3e}  (family {e.confidence:.0%} over {e.horizon})")
    print(f"  EPSILON                   {e.eps_pp:6.2f} pp")
    print()
    print(f"  {e.note}")

    # what the naive rule would have cost, from the measured audit
    c = load_calibration()
    ar = c["accept_rule_audit"]
    print()
    print("  For reference, on this benchmark the naive rule (accept if better) fired on "
          f"{ar['naive_rule_delta_gt_0']['fired']}/{ar['naive_rule_delta_gt_0']['of']} "
          f"pure-noise comparisons; the threshold rule fired on "
          f"{ar['tau_rule_delta_gt_tau']['fired']}/{ar['tau_rule_delta_gt_tau']['of']}.")
    print(json.dumps(asdict(e), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
