# v1.0 environment, published spec, 1 step — evaluation records

Produced by: the v1.0 modules (their release's snapshot) + the published spec verbatim,
Qwen3-0.6B, seed 17, steps 1. Reproduced twice, agreeing to full float precision.

| target | role | option_order | pooled AURC | minDS | noul | choice | score |
|---|---|---|---|---|---|---|---|
| in_distribution | control | canonical | 0.3258 | 31.34 | 0.6048 | 0.5836 | 0.4240 |
| mmlu_pro_1k | control | canonical | 0.7146 | 8.36 | — | 0.2110 | — |
| typed_decisions | control | canonical | 0.4919 | -21.43 | 0.5567 | 0.3767 | 0.3063 |
| in_distribution | control | reversed | 0.4074 | 24.58 | 0.5083 | 0.5447 | 0.3114 |
| mmlu_pro_1k | control | reversed | 0.8238 | 0.81 | — | 0.1460 | — |
| typed_decisions | control | reversed | 0.5193 | -23.38 | 0.4933 | 0.3667 | 0.2838 |
| in_distribution | candidate | canonical | 0.5111 | 13.83 | 0.5342 | 0.2702 | 0.3039 |
| mmlu_pro_1k | candidate | canonical | 0.8631 | -2.21 | — | 0.1200 | — |
| typed_decisions | candidate | canonical | 0.5653 | -28.25 | 0.4817 | 0.3417 | 0.3438 |
| in_distribution | candidate | reversed | 0.5079 | 10.27 | 0.5280 | 0.2808 | 0.3340 |
| mmlu_pro_1k | candidate | reversed | 0.8448 | -2.21 | — | 0.1200 | — |
| typed_decisions | candidate | reversed | 0.5718 | -40.26 | 0.4967 | 0.2800 | 0.3237 |

## What the run's own stdout confirms

```
tower moved: max |dw| over probe tensors = 5.007e-06     <- their guard fired
trained 6277 cases in 3s  loss [1.2721]                  <- n_train_cases matches their 6277
saved release checkpoint -> (309 tower tensors)
use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False.
                                                          ^ grad_checkpointing IS active
```

The two runs agreed **to full float precision**. That establishes **determinism, not correctness**,
and the distinction matters enough to state here: `planted-truth` — *"a pipeline that is consistently
wrong is still consistent. Internal agreement, stability under iterations, and smooth residuals are
properties of the machinery, not of the answer."* An earlier version of this file offered the
agreement as sufficient evidence; it is not, and the reason is that no independent route to these
numbers exists yet (see `docs/arm0_acceptance_gate.md`, "What this gate cannot do"). What the
agreement does buy: a single run is enough to *compare* against, because repeating it changes
nothing.

The control rows are **arm 0's anchors**: `typed_decisions` AURC 0.4919, noul 0.5567, choice 0.3767,
score 0.3063 under canonical.

The candidate is **worse than the control on nearly every cell**. That is what one step does — the
readout head is freshly initialised and a single optimiser step has not yet undone the damage. It is
a pipeline sanity check, not a result about the recipe, and must not be reported as one.

## Finding: option-order sensitivity is not produced by training

`eval_option_orders: ["canonical", "reversed"]` was omitted from every earlier spec of this project.
It is doing real work, and for a sharper reason than "a trained head collapses onto position":

| | canonical | reversed | gap |
|---|---|---|---|
| **control** `in_distribution` noul | 0.6048 | 0.5083 | **+9.6 pp** |
| **control** `in_distribution` score | 0.4240 | 0.3114 | **+11.3 pp** |
| **control** `in_distribution` pooled AURC | 0.3258 | 0.4074 | **−8.2 pp** |
| **control** `typed_decisions` noul | 0.5567 | 0.4933 | **+6.3 pp** |
| candidate `typed_decisions` choice | 0.3417 | 0.2800 | +6.2 pp |

The **zero-shot control shows the larger gaps**. Order sensitivity is therefore a property of the
base model and the readout, not something training introduces — and in the cells measured here,
training *reduced* it. Two consequences:

1. **Any single-order number is unreliable**, for the control as much as for a trained arm. Every
   comparison must use the paired evaluation, or state which order it used.
2. It **vindicates the reference's design.** The setting reads like defensive plumbing and is
   load-bearing: without it this project would have compared arms across an uncontrolled
   presentation-order variable worth up to 11 pp on the same model and the same questions.

Caveat kept explicit: the control scores through `LogprobReadout` and the candidate through the
trained `option_xattn` head, so those two rows are not the same architecture and the comparison
*between* them is not like-for-like. The within-row canonical-vs-reversed gaps are unaffected,
because both sides of each gap use the same readout.

## Method check: the order gap is not an index artefact

Before reading anything into the canonical-vs-reversed gaps, the two sides have to be shown to be
comparable. They are, and the check is in the release's own code rather than in an assumption:

- `evaluate.py`'s `predict()` calls `unpermute_logits(model(**batch), batch["option_perm"], ...)` —
  the same inverse mapping `fit.py` applies before its loss. So every prediction is expressed in the
  **canonical** index space regardless of the order the model was shown.
- `score_predictions` takes the gold from `gold_label(q, c.gold[q.key])` — also canonical.
- The records store `pred` and `gold` as **option names** (`pred='neutral_report'`,
  `gold='dry_sarcastic'`), not indices, so the comparison cannot silently pair one space against the
  other.

Therefore the only thing that differs between a canonical and a reversed evaluation is **the order in
which the model saw the options**. That is what makes the gap interpretable as a position sensitivity
rather than as a bookkeeping difference, and it is the premise the learning-signal hypothesis in
`docs/reproduction_audit.md` rests on.

What the check does **not** establish: that `pooled_top1` computed from these rows uses byte-identical
tie-breaking to theirs. Their `verify.json` reports it but ships no per-question rows to compare
against, so the definition is matched by construction (`pred == gold`) and not verified against their
implementation. Numbers in the 0.65 range carry no exact ties, so the risk is small and is recorded
rather than dismissed.

## Full records

Per-question rows (21,792): `E:/2026-AI4S/arms/arm0_probe_v1/out/arm0probe.items.jsonl` (6.7 MB,
kept off-repository); regenerate with the command in `docs/reproduction_audit.md`.
