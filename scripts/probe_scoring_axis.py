"""First measurement on the scoring axis (paper A, arXiv 2608.21382).

Compares the two read-outs of the SAME item under the SAME wording:
  (1) generated-text parsing   — what our whole pilot used
  (2) per-option log-likelihood — the axis paper A calls load-bearing

If the two disagree on a non-trivial share of items, then any harness-fragility
number computed from text parsing alone is confounded with the read-out choice.

Usage: python scripts/probe_scoring_axis.py [n_items] [model_key]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasp import benchmark, config, llm  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
model_key = sys.argv[2] if len(sys.argv) > 2 else "mistral_7b"
spec = config.TARGET_MODELS[model_key]
items = benchmark.load_items("arc_challenge")[:n]

agree = 0
text_ok = 0
lp_ok = 0
text_correct = 0
lp_correct = 0
rows = []

for it in items:
    stem = benchmark.render_question_block(it)

    # (1) generated text, exactly as the pilot scored it
    c = llm.chat(
        spec,
        f"{stem}\nAnswer with the option letter only.",
        temperature=0.0,
        seed=42,
        num_predict=8,
        timeout=300,
    )
    t_pred = None
    if c.ok:
        text_ok += 1
        from oasp import evaluator  # local import: keep probe standalone

        try:
            # extract_choice returns (label_or_None, matched_flag) — unpack it,
            # or every comparison below silently compares a tuple to a string and
            # reports a fake 0% agreement.
            t_pred, _matched = evaluator.extract_choice(c.text, it.labels)
        except Exception:
            t_pred = None

    # (2) per-option log-likelihood
    s = llm.score_options_ollama(spec.model_id, stem, it.labels, timeout=300)
    lp_pred = s.predicted if s.ok else None
    if s.ok:
        lp_ok += 1

    same = (t_pred is not None and lp_pred is not None and t_pred == lp_pred)
    agree += int(same)
    text_correct += int(t_pred == it.answer_key)
    lp_correct += int(lp_pred == it.answer_key)
    rows.append((it.item_id, it.answer_key, t_pred, lp_pred, same, round(s.margin() or 0, 2)))

print(f"model={model_key}  n={n}")
print(f"  text-parse ok     : {text_ok}/{n}")
print(f"  logprob ok        : {lp_ok}/{n}")
print(f"  AGREEMENT         : {agree}/{n}  ({agree / n:.1%})")
print(f"  acc text-parse    : {text_correct}/{n} = {text_correct / n:.3f}")
print(f"  acc logprob       : {lp_correct}/{n} = {lp_correct / n:.3f}")
print()
print(f"  {'item':<12} {'key':<4} {'text':<5} {'logpr':<6} {'same':<5} margin")
for r in rows:
    print(f"  {str(r[0])[:12]:<12} {r[1]:<4} {str(r[2]):<5} {str(r[3]):<6} "
          f"{str(r[4]):<5} {r[5]}")
