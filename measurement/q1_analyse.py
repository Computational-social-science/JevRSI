#!/usr/bin/env python3
"""q1_analyse.py -- decide Q1 on the pre-registered criterion, with the pairing made explicit.

WHY A SEPARATE ANALYSIS STEP. The Q1 run produces per-question predictions WITH two argmaxes recorded
(`raw_argmax`, `debiased_argmax`) on the SAME cases, so the comparison is paired by construction. That
matters: the decision threshold tau was derived from a PAIRED standard error (floor_b_laya_prereg.md,
tau = 3.11 pp on test, 4.95 pp on dev at H = 1000), so scoring the two arms as if they were independent
samples would use the wrong variance and quietly shrink the threshold.

THE CRITERION, FIXED BEFORE THE NUMBERS EXIST (RESEARCH_GOAL_v2 section 7):
  * Q1 is supported only if the paired improvement exceeds the split's tau.
  * The mechanism claim is the ORDER-FLIP RATE, not the accuracy. A gain without a flip-rate drop would
    mean something other than position debiasing caused it.
  * The intervention is a no-op BY CONSTRUCTION for the boolean and score types. Any movement there is a
    bug, not a finding, and is reported as such.

It also reports the McNemar exact test on the discordant pairs, because "how many items changed answer"
and "in which direction" are separate questions and a mean can hide a symmetric churn.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Pre-registered, from floor_b_laya_prereg.md / tau_calibration.py. Quoted here rather than recomputed
# so the criterion cannot drift with the data.
TAU = {"dev": 0.0495, "test": 0.0311}      # decision threshold, fraction
TAU_SRC = ("measurement/floor_b_laya_prereg.md (test 3.11 pp); "
           "measurement/tau_calibration.py (dev 4.95 pp at H=1000)")


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Two-sided exact binomial p on the discordant pairs. Returns (p, n_discordant)."""
    n = b + c
    if n == 0:
        return 1.0, 0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", required=True, choices=tuple(TAU))
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-n", type=int, default=None)
    a = ap.parse_args()

    preds = json.loads(Path(a.preds).read_text(encoding="utf-8"))
    if a.expect_n is not None and len(preds) != a.expect_n:
        print(f"[FAIL] expected {a.expect_n} questions, got {len(preds)} -- refusing to score a "
              f"partial run")
        return 1

    tau = TAU[a.split]
    n = len(preds)
    print("=" * 74)
    print(f"Q1 verdict -- readout debiasing on {a.split} (n={n})")
    print("=" * 74)
    print(f"  pre-registered criterion: paired improvement > tau = {tau*100:.2f} pp")
    print(f"  tau source: {TAU_SRC}")
    print()

    # ---- per-item outcome, split by whether the intervention could possibly act ----------------
    buckets: dict[str, dict] = {}
    for p in preds:
        t = p["type"]
        b = buckets.setdefault(t, {"n": 0, "raw_hit": 0, "deb_hit": 0, "changed": 0,
                                   "raw_wrong_to_right": 0, "raw_right_to_wrong": 0})
        y = int(np.argmax(np.asarray(p["target"], dtype=np.float64)))
        r_ok = int(p["raw_argmax"]) == y
        d_ok = int(p["debiased_argmax"]) == y
        b["n"] += 1
        b["raw_hit"] += r_ok
        b["deb_hit"] += d_ok
        if p["raw_argmax"] != p["debiased_argmax"]:
            b["changed"] += 1
            b["raw_wrong_to_right"] += int((not r_ok) and d_ok)
            b["raw_right_to_wrong"] += int(r_ok and (not d_ok))

    print(f"  {'type':9s} {'n':>5s} {'raw':>8s} {'debias':>8s} {'delta':>9s} {'changed':>8s} "
          f"{'w->r':>6s} {'r->w':>6s}")
    tot = {"raw": 0, "deb": 0, "w2r": 0, "r2w": 0}
    for t in sorted(buckets):
        b = buckets[t]
        ra, da = b["raw_hit"] / b["n"], b["deb_hit"] / b["n"]
        print(f"  {t:9s} {b['n']:5d} {ra:8.4f} {da:8.4f} {(da-ra)*100:+8.2f}pp "
              f"{b['changed']:8d} {b['raw_wrong_to_right']:6d} {b['raw_right_to_wrong']:6d}")
        tot["raw"] += b["raw_hit"]; tot["deb"] += b["deb_hit"]
        tot["w2r"] += b["raw_wrong_to_right"]; tot["r2w"] += b["raw_right_to_wrong"]

    raw_acc, deb_acc = tot["raw"] / n, tot["deb"] / n
    delta = deb_acc - raw_acc
    print()
    print(f"  POOLED     {n:5d} {raw_acc:8.4f} {deb_acc:8.4f} {delta*100:+8.2f}pp")

    # ---- the paired test -----------------------------------------------------------------------
    p_mc, n_disc = mcnemar_exact(tot["w2r"], tot["r2w"])
    # Paired SE of the accuracy difference, from the discordant-pair rate. Under the null the two
    # arms agree on most items, so the variance is driven by the discordant fraction; using the
    # unpaired binomial SE instead would overstate it and inflate tau.
    if n:
        se_paired = math.sqrt(max(tot["w2r"] + tot["r2w"], 1)) / n
    else:
        se_paired = float("nan")
    ci = (delta - 1.96 * se_paired, delta + 1.96 * se_paired)

    print()
    print("  paired comparison (same cases, same weights, same process):")
    print(f"    discordant pairs      : {n_disc}  (wrong->right {tot['w2r']}, right->wrong {tot['r2w']})")
    print(f"    McNemar exact p       : {p_mc:.4g}")
    print(f"    paired SE of delta    : {se_paired*100:.3f} pp")
    print(f"    95% CI on delta       : [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp")
    print(f"    pre-registered tau    : {tau*100:+.2f} pp")
    print()

    supported = (delta > tau) and (ci[0] > 0)
    consistent = (ci[0] > tau) or (ci[1] < tau)
    if supported:
        verdict = "SUPPORTED -- improvement exceeds the pre-registered threshold"
    elif consistent:
        verdict = ("NOT SUPPORTED, and the interval excludes tau -- the effect is resolvable and it is "
                   "not large enough")
    else:
        verdict = ("INCONCLUSIVE -- the interval spans tau; this run cannot decide the question "
                   "at this n")
    print(f"  VERDICT: {verdict}")

    # ---- the pre-registered mechanism prediction -----------------------------------------------
    print()
    print("  mechanism check (pre-registered):")
    for t in ("noul", "boolean", "score"):
        if t in buckets and buckets[t]["changed"]:
            print(f"    [FAIL] {t}: {buckets[t]['changed']} argmax change(s) -- the intervention is a "
                  f"no-op for this type by construction, so this is a bug")
        elif t in buckets:
            print(f"    ok   {t}: 0 argmax changes, as predicted")
    if "choice" in buckets:
        c = buckets["choice"]
        if c["changed"]:
            print(f"    ok   choice: {c['changed']}/{c['n']} argmax changed -- the intervention acts "
                  f"where it should")
        else:
            print(f"    [note] choice: 0 argmax changes; the readout may already be order-invariant")

    result = {"split": a.split, "n": n, "raw_accuracy": raw_acc, "debiased_accuracy": deb_acc,
              "delta": delta, "tau": tau, "ci95_paired": list(ci), "paired_se": se_paired,
              "mcnemar_p": p_mc, "discordant": n_disc, "supported": bool(supported),
              "verdict": verdict, "by_type": buckets, "tau_source": TAU_SRC}
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
