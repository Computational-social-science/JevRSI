"""Regression guards for the NaN-contamination class in ``oasp.stats``.

This class of bug bit the project four separate times, each in a different metric:

1. M2 counted ``nan != nan`` as an IPS flip (NaN holes read as sensitivity).
2. M2's permutation null was degenerate by construction for M3.
3. M4/M5 counted a NaN sign comparison as an order flip (flip rate 1.0 on data where
   one model led on every variant).
4. M4/M5's rank-τ leg fed a 2-element rank vector into Kendall's τ; any tie made it
   constant, τ returned NaN, and ``np.mean`` propagated that into the headline number.

The shared root cause: **a missing value silently becoming a data point**. These tests
assert absence of the bad outcome, not merely presence of a good one.

Run: ``python tests/test_stats.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasp import stats  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  [OK]   {msg}")
    else:
        print(f"  [FAIL] {msg}")
        FAILS.append(msg)


def grid(model_acc: dict[str, list[float]], *, n_items: int = 60,
         n_variants: int | None = None, split: str = "dev") -> pd.DataFrame:
    """Build a synthetic results frame where each model's per-variant accuracy is exact.

    Achievement is deterministic: for variant v, a model gets ``round(acc * n_items)``
    items right and the rest wrong, so ``mean(correct)`` equals the target exactly.
    """
    n_variants = n_variants or len(next(iter(model_acc.values())))
    rows = []
    for model, accs in model_acc.items():
        for v in range(n_variants):
            n_right = round(accs[v] * n_items)
            for i in range(n_items):
                rows.append({
                    "model": model,
                    "variant": f"v{v:02d}",
                    "split": split,
                    "item_id": f"it{i}",
                    "correct": i < n_right,
                    "parse_ok": True,
                })
    return pd.DataFrame(rows)


print("=== 1. complete 2-model grid: no NaN anywhere, rank-tau explicitly undefined ===")
# 32 variants, well-behaved accuracies, two models that anti-correlate.
va = [0.70 + 0.01 * (i % 5) for i in range(32)]
vb = [0.80 - 0.01 * (i % 5) for i in range(32)]
r2 = stats.metric_m4_m5_rank_stability(grid({"A": va, "B": vb}), split="dev")
check(r2.get("mean_pairwise_tau_accuracy_profile") is not None,
      "mean_pairwise_tau_accuracy_profile is a number, not None")
check(r2.get("n_valid_tau_pairs") == 1, "one model pair evaluated")
check(r2.get("mean_tau_between_variant_rankings") is None,
      "rank-tau is None for 2 models (not NaN, which would render as 'nan')")
check(bool(r2.get("rank_tau_undefined_reason")),
      "undefined rank-tau carries an explicit reason")
check("nan" not in str(r2.get("mean_pairwise_tau_accuracy_profile")).lower(),
      "no literal NaN leaked into the reported τ")

print("\n=== 2. ties are excluded from the flip denominator, not counted as flips ===")
# 32 variants: 2 exact ties, then B wins 6, then A wins the remaining 24.
# A mean = (2*.5 + 6*.4 + 24*.6)/32 = 0.55625 ; B mean = 0.44375, so the reference
# order is A>B and the flips are exactly the 6 variants B wins.
vc, vd = [], []
for v in range(32):
    if v < 2:
        vc.append(0.50); vd.append(0.50)
    elif v < 8:
        vc.append(0.40); vd.append(0.60)
    else:
        vc.append(0.60); vd.append(0.40)
r3 = stats.metric_m4_m5_rank_stability(grid({"A": vc, "B": vd}), split="dev")
check(r3.get("n_tied_comparisons") == 2, "the 2 tied variants are counted as ties")
check(r3.get("n_decided_comparisons") == 30, "ties are excluded from the denominator (32-2)")
check(r3.get("model_pair_order_flip_rate") == round(6 / 30, 4),
      f"flip rate = 6/30 = 0.2 (got {r3.get('model_pair_order_flip_rate')})")
check(r3.get("flip_rate_undefined_reason") is None,
      "no undefined-reason when the reference order is decided")

print("\n=== 2b. models tied on average -> flip rate is UNDEFINED, never 0.0 ===")
# Equal means: there is no reference ranking, so "flips" has no meaning. Returning
# 0.0 here would dress an undefined quantity as a reassuring finding.
ve = [0.60, 0.40] * 16
vf = [0.40, 0.60] * 16
r3b = stats.metric_m4_m5_rank_stability(grid({"A": ve, "B": vf}), split="dev")
check(r3b.get("model_pair_order_flip_rate") is None,
      f"flip rate is None, not 0.0 (got {r3b.get('model_pair_order_flip_rate')})")
check(bool(r3b.get("flip_rate_undefined_reason")),
      "the undefined flip rate carries an explicit reason")
check(r3b.get("n_ref_tied_model_pairs") == 1, "the reference-tied pair is counted")

print("\n=== 3. 3-model grid: rank-tau becomes defined ===")
r4 = stats.metric_m4_m5_rank_stability(
    grid({"A": va, "B": vb, "C": [0.60] * 32}), split="dev"
)
check(r4.get("mean_tau_between_variant_rankings") is not None,
      "rank-tau is defined once >=3 models exist")
check(r4.get("rank_tau_undefined_reason") is None,
      "no undefined-reason is emitted when the leg is computable")

print("\n=== 4. incomplete grid: NaN holes must not become data points ===")
df = grid({"A": va, "B": vb})
df = df[~((df["model"] == "B") & (df["variant"].isin(["v00", "v01", "v02"])))]
r5 = stats.metric_m4_m5_rank_stability(df, split="dev")
check(r5.get("n_variants") == 29,
      f"incomplete variants dropped from the shared grid (got {r5.get('n_variants')})")
check(r5.get("model_pair_order_flip_rate") is not None,
      "flip rate still computable after dropping incomplete variants")
for k, v in r5.items():
    if isinstance(v, float):
        check(v == v, f"field {k} is not NaN")

print("\n=== 5. capability_floor emits the interior verdict ===")
fl = stats.capability_floor(grid({"A": va, "B": vb}), chance=0.25, split="dev")
check(fl["A"]["verdict"] == "INTERIOR", f"A is INTERIOR (got {fl['A']['verdict']})")
check(fl["A"]["eligible_for_h1"] is True, "INTERIOR models are H1-eligible")
ceil = stats.capability_floor(grid({"D": [1.0] * 32}), chance=0.25, split="dev")
check(ceil["D"]["verdict"] == "CEILING_BOUND", "a 1.000 model is CEILING_BOUND, not eligible")
check(ceil["D"]["eligible_for_h1"] is False, "CEILING_BOUND models are excluded from H1")
flr = stats.capability_floor(grid({"E": [0.05] * 32}), chance=0.25, split="dev")
check(flr["E"]["verdict"] == "FLOOR_BOUND", "a below-chance model is FLOOR_BOUND")

print()
if FAILS:
    print(f"[FAIL] stats: {len(FAILS)} check(s) failed")
    for f in FAILS:
        print(f"   - {f}")
    sys.exit(1)
print("[OK] stats: all checks passed")
