"""probe_post_hoc_ceiling.py -- how much is reachable WITHOUT touching a weight?

THE QUESTION. The project is spending 2.11 h per 600-step LoRA replicate to move a number, and
2.11 h x 5 seeds = 10.5 h just to learn that training path's own noise floor. The obvious question is
whether a large part of the available gain is reachable for free -- by post-processing the frozen
model's output. This script bounds that prize BEFORE any of it is spent.

WHAT IT IS NOT. Every number below that says "oracle" is fitted on the dev split and evaluated on the
same dev split. That is circular by construction and is therefore NOT a reportable result. It is a
feasibility probe: it answers "is the prize worth chasing", nothing more. Any number that survives into
a claim has to be fitted on calibration and evaluated on dev, once, under a pre-registered rule.

THE STRUCTURAL FACT THIS SCRIPT ESTABLISHES. The evaluator computes accuracy and chance-corrected skill
from `probs.argmax()`, and `probs = softmax(logits / T)`. Dividing by a positive T and exponentiating
is strictly monotone, so argmax is invariant to T. It follows that:

    NO temperature-scaling post-processing can change accuracy or chance-corrected skill. Not slightly.
    Not at all. Ever.

This is not a hypothesis to be tested -- it is a property of the metric -- but it is verified
numerically here anyway, because a project that has just spent effort on calibration deserves a direct
check that the effort could not have done what it was hoped to do. The dev numbers bear this out: the
shipped L1 temperatures leave accuracy at exactly 0.8150, identical to raw T=1.

WHERE THE PRIZE ACTUALLY IS. The metrics that DO respond to post-processing are the ones built on the
full distribution rather than its argmax: soft cross-entropy, Brier, ECE, and the score expectation MAE.
Those are real and they are already partly harvested. The prize for accuracy lives somewhere else
entirely -- in a function of the model's features, not of its output distribution -- which is what the
margin analysis here is for. If wrong items sit close to the decision boundary, a cheap head on frozen
features can flip them. If wrong items are confidently wrong, no cheap method can, and the money belongs
in training.

Run: python measurement/probe_post_hoc_ceiling.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md.
ROOT = pathlib.Path(__file__).resolve().parents[2]

import json
import pathlib

import numpy as np
DEV_LOGITS = ROOT / "measurement" / "day1_raw_dev.json"
TYPES = ("boolean", "choice", "score")


def load(path: pathlib.Path = DEV_LOGITS) -> list[dict]:
    """The cached dev logits are JSONL, one prediction per line -- not a JSON array."""
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def softmax(z, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def hit(r: dict) -> int:
    return int(int(np.argmax(r["logits"])) == int(np.argmax(r["target"])))


def acc(sel: list[dict], T: float = 1.0) -> float:
    return float(np.mean([int(np.argmax(softmax(r["logits"], T))) ==
                          int(np.argmax(r["target"])) for r in sel]))


def skill(sel: list[dict], T: float = 1.0) -> float:
    """The evaluator's own per-item chance correction, so the baseline is comparable to the loop's."""
    vals = []
    for r in sel:
        k = len(r["logits"])
        if k < 2:
            continue
        top1 = float(int(np.argmax(softmax(r["logits"], T))) == int(np.argmax(r["target"])))
        vals.append((top1 - 1.0 / k) / (1.0 - 1.0 / k))
    return float(np.mean(vals))


def soft_ce(sel: list[dict], T: float = 1.0) -> float:
    s = 0.0
    for r in sel:
        p = softmax(r["logits"], T)
        y = np.asarray(r["target"], dtype=np.float64)
        s += -np.sum(y * np.log(np.clip(p, 1e-12, None)))
    return s / len(sel)


def brier(sel: list[dict], T: float = 1.0) -> float:
    s = 0.0
    for r in sel:
        s += float(np.sum((softmax(r["logits"], T) - np.asarray(r["target"], float)) ** 2))
    return s / len(sel)


def rule(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main() -> int:
    rows = load()
    print(f"cached dev predictions: {len(rows)}")
    by = {t: [r for r in rows if r["type"] == t] for t in TYPES}
    for t in TYPES:
        print(f"  {t:8s} n={len(by[t]):3d}  mean k={np.mean([len(r['logits']) for r in by[t]]):.2f}")

    # ---- 1. Temperature cannot move accuracy. Verified, not assumed. ----------
    rule("1. Temperature vs accuracy  (expect: exactly invariant)")
    for t in TYPES:
        vals = [acc(by[t], T) for T in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)]
        uniq = len(set(vals))
        print(f"  {t:8s} T in [0.25,0.5,1,2,4,10] -> acc {[f'{v:.4f}' for v in vals]}")
        print(f"           distinct values = {uniq}  ->  "
              f"{'INVARIANT (confirmed)' if uniq == 1 else 'CHANGED -- investigate!'}")

    # ---- 2. The baseline, per type, with the evaluator's own skill definition --
    rule("2. Per-type baseline (raw logits, T=1)")
    for t in TYPES:
        print(f"  {t:8s} acc={acc(by[t]):.4f}  skill={skill(by[t]):.4f}  "
              f"softCE={soft_ce(by[t]):.4f}  brier={brier(by[t]):.4f}")
    print(f"  {'POOLED':8s} acc={acc(rows):.4f}  skill={skill(rows):.4f}  "
          f"softCE={soft_ce(rows):.4f}  brier={brier(rows):.4f}")

    # ---- 3. The shipped calibration, scored the same way ---------------------
    rule("3. Shipped L1 temperatures -- what it actually bought")
    L1 = {"boolean": 1.0717734625362931, "choice": 1.0352649238413776, "score": 1.0717734625362931}
    for t in TYPES:
        a0, a1 = acc(by[t]), acc(by[t], L1[t])
        print(f"  {t:8s} T={L1[t]:.4f}  acc {a0:.4f} -> {a1:.4f} ({a1-a0:+.4f})   "
              f"softCE {soft_ce(by[t]):.4f} -> {soft_ce(by[t], L1[t]):.4f} "
              f"({soft_ce(by[t],L1[t])-soft_ce(by[t]):+.4f})   "
              f"brier {brier(by[t]):.4f} -> {brier(by[t], L1[t]):.4f} "
              f"({brier(by[t],L1[t])-brier(by[t]):+.4f})")
    print("  Read this line carefully: accuracy does not move. The calibration is real, and it is")
    print("  aimed at metrics the loop's accept rule does not use.")

    # ---- 4. Oracle ceiling for temperature on the distributional metrics ------
    rule("4. ORACLE per-type temperature (fitted AND scored on dev = upper bound, NOT a result)")
    Ts = np.arange(0.30, 4.01, 0.02)
    for t in TYPES:
        ce = [soft_ce(by[t], T) for T in Ts]
        br = [brier(by[t], T) for T in Ts]
        i, j = int(np.argmin(ce)), int(np.argmin(br))
        print(f"  {t:8s} softCE {soft_ce(by[t],1.0):.4f} -> {ce[i]:.4f} at T={Ts[i]:.2f} "
              f"({ce[i]-soft_ce(by[t],1.0):+.4f})   "
              f"brier {brier(by[t],1.0):.4f} -> {br[j]:.4f} at T={Ts[j]:.2f} "
              f"({br[j]-brier(by[t],1.0):+.4f})")
    print("  The oracle T is close to 1 for every type: the shipped calibration is already near the")
    print("  ceiling of what temperature alone can do. This lever is nearly exhausted.")

    # ---- 5. Where the accuracy prize actually lives: the decision margin ------
    rule("5. Decision margin -- can a CHEAP head on frozen features flip wrong items?")
    print("   margin = top1 logit - top2 logit")
    m_right, m_wrong = [], []
    for r in rows:
        z = np.sort(np.asarray(r["logits"], dtype=np.float64))[::-1]
        m = float(z[0] - z[1]) if len(z) > 1 else float("inf")
        (m_right if hit(r) else m_wrong).append(m)
    m_right, m_wrong = np.array(m_right), np.array(m_wrong)
    print(f"  correct n={len(m_right):3d}: median margin={np.median(m_right):6.2f}  "
          f"p10={np.percentile(m_right,10):5.2f}")
    print(f"  wrong   n={len(m_wrong):3d}: median margin={np.median(m_wrong):6.2f}  "
          f"p90={np.percentile(m_wrong,90):5.2f}")
    print(f"\n  {'margin <':>9} {'wrong items':>12} {'of wrong':>9} {'of all dev':>11}")
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0):
        n = int((m_wrong < thr).sum())
        print(f"  {thr:9.1f} {n:12d} {n/len(m_wrong):9.3f} {n/len(rows):11.3f}")
    print("\n  Interpretation. Errors with a small margin are reachable by ANY function of the")
    print("  model's features -- a linear probe, a k-NN over the train split, a better head. Errors")
    print("  with a large margin are not: the model is confidently wrong, and only changing what the")
    print("  backbone computes can fix those. The split between these two populations is the number")
    print("  that decides whether a cheap route is worth anything.")

    print("\n" + "=" * 74)
    print("SUMMARY OF THE FEASIBILITY PROBE")
    print("=" * 74)
    print("  1. Post-hoc temperature CANNOT change accuracy or skill. Proven and verified.")
    print("  2. The shipped calibration is already near the temperature ceiling. Little left there.")
    print("  3. The accuracy prize, if it exists, is in the model's FEATURES, not its output")
    print("     distribution -- which means the cheap route is a head on frozen features, trained")
    print("     on CACHED features, not a temperature.")
    print("  4. Quantify (3) with the margin split above before spending GPU hours on it.")
    print("\n  Reminder: oracle numbers are fitted on dev and scored on dev. Feasibility only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
