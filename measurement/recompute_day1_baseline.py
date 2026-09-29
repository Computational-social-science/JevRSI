"""recompute_day1_baseline.py -- turn the Day-1 baseline from transcribed stdout into a machine-readable fact.

THE PROBLEM THIS FIXES. The manuscript audit could only check the baseline numbers for PRESENCE, because
their only source was the frozen evaluator's stdout, transcribed by hand into docs/DAY1_PROGRESS.md. A
presence check cannot catch a wrong value as long as the right value appears somewhere else in the document,
and the negative control proved it: editing one of six occurrences of 0.8150 changed nothing. That is a
weak link in the provenance chain, and the audit's own output said so.

THE FIX. measurement/day1_raw_dev.json holds the per-question LOGITS for all 600 dev questions, not a
summary. Every metric in the manuscript is therefore recomputable -- and it is recomputed here by importing
`compute_metrics`, `probabilities` and `chance_corrected_skill` from the frozen evaluator itself rather than
reimplementing them. Reimplementing would be the wrong fix: a second implementation of ECE is a second
definition of ECE, and the two would drift exactly where the manuscript is most sensitive.

Both arms come out of the same file:
    raw  -- probabilities(logits, 1.0)
    L1   -- probabilities(logits, temperature[type]) using the shipped temperatures.json

The temperatures are applied to the SAME logits, so the raw-vs-L1 comparison the manuscript makes in
section 2.2 is a comparison on identical inputs, which is the only form in which it means anything.

A recomputation that disagrees with the recorded stdout is itself a finding and is reported as one: it would
mean either the logits file and the log describe different runs, or the metric code has changed since.

Run: python measurement/recompute_day1_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = Path("../agent-jev/scripts/evaluate_split.py")
PREDS = REPO / "measurement" / "day1_raw_dev.json"
TEMPS = REPO / "loop" / "temperatures.json"      # raw arm, T = 1.0 by construction
SHIPPED = Path("../agent-jev/checkpoints/autoresearch/temperatures.json")
OUT = REPO / "measurement" / "day1_baseline_metrics.json"

# What the frozen evaluator printed on 2026-09-29. Kept so a disagreement is visible rather than silent.
RECORDED_STDOUT = {
    "raw": {"accuracy": 0.8150, "chance_corrected": 0.7326, "soft_cross_entropy": 0.8286,
            "brier_sum": 0.0491, "ece": 0.1368, "score_expectation_mae": 0.1862},
    "L1": {"accuracy": 0.8150, "chance_corrected": 0.7326, "soft_cross_entropy": 0.8243,
           "brier_sum": 0.0476, "ece": 0.1481, "score_expectation_mae": 0.1849},
}

if not EVAL.exists():
    print(f"[FAIL] frozen evaluator not found at {EVAL}")
    raise SystemExit(1)
if not PREDS.exists():
    print(f"[FAIL] saved logits not found at {PREDS}")
    raise SystemExit(1)

sys.path.insert(0, str(EVAL.parent))
import importlib.util

spec = importlib.util.spec_from_file_location("frozen_evaluator", EVAL)
if spec is None or spec.loader is None:
    print(f"[FAIL] cannot load the frozen evaluator from {EVAL}")
    raise SystemExit(1)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

rows = [json.loads(l) for l in PREDS.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"loaded {len(rows)} dev questions from {PREDS.name}")
by_type: dict[str, int] = {}
for r in rows:
    by_type[r["type"]] = by_type.get(r["type"], 0) + 1
print(f"  by type: {by_type}")


def arm(temps: dict[str, float] | None, label: str) -> dict:
    """Recompute one arm from the saved logits, using the evaluator's own metric functions."""
    preds = []
    for r in rows:
        t = 1.0 if temps is None else temps[r["type"]]
        preds.append({**r, "probs": ev.probabilities(r["logits"], t)})
    m = ev.compute_metrics(preds)
    rep = ev.report(preds)
    return {
        "arm": label,
        "temperatures": temps or {"(all)": 1.0},
        "n": m["n"],
        "accuracy": round(m["accuracy"], 6),
        "chance_corrected": round(ev.chance_corrected_skill(preds), 6),
        "soft_cross_entropy": round(m["soft_cross_entropy"], 6),
        "brier_sum": round(m["brier_sum"], 6),
        "ece_10_bins_vs_gold_argmax": round(m["ece_10_bins_vs_gold_argmax"], 6),
        "score_expectation_mae": round(m["score_expectation_mae"], 6),
        "by_type": {k: {kk: round(vv, 6) for kk, vv in v.items() if isinstance(vv, (int, float))}
                    for k, v in rep["by_type"].items()},
    }


def temps_from(path: Path, name: str) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    # temperatures.json may nest under a per-type record or be a bare scalar map
    if all(isinstance(v, dict) and "temperature" in v for v in d.values()):
        return {k: float(v["temperature"]) for k, v in d.items()}
    return {k: float(v) for k, v in d.items()}


raw = arm(None, "raw (T=1.0)")
L1 = arm(temps_from(SHIPPED, "shipped L1"), "shipped L1")

print("\n" + "=" * 72)
print(f"{'metric':28s} {'raw':>10s} {'L1':>10s} {'recorded raw':>14s} {'recorded L1':>12s}")
print("=" * 72)
drift: list[str] = []
for k in ("accuracy", "chance_corrected", "soft_cross_entropy", "brier_sum",
          "ece_10_bins_vs_gold_argmax", "score_expectation_mae"):
    rk = {"ece_10_bins_vs_gold_argmax": "ece", "score_expectation_mae": "score_expectation_mae"}.get(k, k)
    rr, rl = RECORDED_STDOUT["raw"].get(rk), RECORDED_STDOUT["L1"].get(rk)
    print(f"{k:28s} {raw[k]:10.4f} {L1[k]:10.4f} {rr if rr is not None else '-':>14} "
          f"{rl if rl is not None else '-':>12}")
    for got, want, who in ((raw[k], rr, "raw"), (L1[k], rl, "L1")):
        if want is not None and abs(got - want) > 5e-5:
            drift.append(f"{who}.{k}: recomputed {got:.4f} vs recorded {want:.4f}")

print("=" * 72)
if drift:
    print("[FAIL] recomputation disagrees with the recorded baseline -- one of these is true:")
    for d in drift:
        print(f"    - {d}")
    print("    (a) the logits file and the log describe different runs, or")
    print("    (b) the metric code changed since the baseline was recorded.")
else:
    print("[OK] every recomputed metric matches the recorded baseline to 4 decimal places")

OUT.write_text(json.dumps({
    "_what_this_is": "Day-1 baseline metrics recomputed from saved logits by the frozen evaluator's own "
                     "metric functions. This replaces transcribed stdout as the audit's source for them.",
    "_source_logits": str(PREDS),
    "_metric_code": str(EVAL),
    "_recomputation_agrees_with_recorded_stdout": not drift,
    "arms": {"raw": raw, "L1": L1},
}, indent=2), encoding="utf-8")
print(f"\nwritten: {OUT}")
raise SystemExit(1 if drift else 0)
