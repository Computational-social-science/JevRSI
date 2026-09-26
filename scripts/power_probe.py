"""How many items does the spread CI actually need?

Adding the equivalence test (docs/critical_review_2026-09-26.md FLAW 1) made the
*upper confidence bound* of the spread the decision quantity. That bound turned out
to be 0.2167 on a margin of 0.05 — so the study cannot assert equivalence OR its
negation. This script measures how the CI width scales with the number of items by
subsampling the existing run, giving a concrete item requirement instead of a
hand-wave.

Usage: python scripts/power_probe.py [run_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasp import config, stats  # noqa: E402

run_id = sys.argv[1] if len(sys.argv) > 1 else "pilot_v2_challenge"
REPS = 200
rng = np.random.default_rng(12345)

df = stats.load_results(config.RUNS_DIR / run_id)
df = df[df["split"] == "dev"]

print(f"run={run_id}  items={df['item_id'].nunique()}  "
      f"variants={df['variant'].nunique()}  models={sorted(df['model'].unique())}\n")
print(f"{'n_items':>8} {'model':<12} {'mean range':>11} {'CI lo':>8} {'CI hi':>8} "
      f"{'width':>8} {'hi/margin':>10}")
print("-" * 74)

all_items = sorted(df["item_id"].unique())
margin = config.DESIGN["equivalence_margin"]

for n in (15, 30, 45, 60):
    if n > len(all_items):
        continue
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        ranges: list[float] = []
        for _ in range(REPS):
            pick = set(rng.choice(all_items, size=n, replace=False).tolist())
            s = sub[sub["item_id"].isin(pick)]
            acc = s.groupby("variant", observed=True)["correct"].mean()
            if len(acc) < 2:
                continue
            ranges.append(float(acc.max() - acc.min()))
        if not ranges:
            continue
        # The item-resample distribution of the range IS the sampling distribution of
        # the estimator, so its percentiles give the CI directly. A nested bootstrap
        # inside each replicate would add cost without adding information.
        lo = float(np.percentile(ranges, 2.5))
        hi = float(np.percentile(ranges, 97.5))
        print(f"{n:>8} {model:<12} {np.mean(ranges):>11.4f} {lo:>8.4f} {hi:>8.4f} "
              f"{hi - lo:>8.4f} {hi / margin:>10.2f}")

print()
print(f"equivalence margin = {margin}; a model is declared equivalent only if the")
print("spread's UPPER 95% bound falls below it. 'hi/margin' is how many times too")
print("wide the bound still is.")
