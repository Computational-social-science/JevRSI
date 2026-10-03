# Arm 0 result — 2026-10-03

The first arm of the JevRSI curve. Their v1.0 recipe, their corpus, their v1.0 code, **one intended
difference: the backbone is 0.6B instead of 2B.**

## The run

| | |
|---|---|
| started | 2026-10-02 21:13:49 |
| finished | 2026-10-03 15:20:54 |
| **wall** | **18 h 07 min** |
| training | 62,741 s (17 h 26 min) per `checkpoints/meta.json` |
| eval | ~38 min |
| step time | **41.8 s/step** (62,741 / 1500) |
| steps | 1,500 completed, no interruption |
| peak VRAM | 11,807 / 12,282 MiB = 96%, flat, no OOM |
| `n_train_cases` | **6,277 — matches the published run exactly** |
| `tower_keys` | 309 |
| `tapped_layer` | 27, `full_attention (final, normed)` |

## The result

`pooled_top1` on `typed_decisions`, the metric the release publishes, measured zero-shot and after
arm 0 on the identical item set (n=2,000):

| | canonical | reversed | order gap |
|---|---|---|---|
| zero-shot control | **0.4025** | 0.3715 | **+3.1 pp** |
| **arm 0 (0.6B)** | **0.5775** | 0.5850 | **−0.7 pp** |
| their v1.0 (2B) | **0.6525** | 0.6525 | 0.0 pp |

**+17.5 pp over the zero-shot anchor, recovering 70% of the headroom** (0.1750 of the 0.2500 that
separates the anchor from their published 2B number). At 0.6B against their 2B, arm 0 lands at
**88.5% of their absolute score**.

The control column is not decoration: 0.4025 is exactly the value this repository measured for the
zero-shot anchor before arm 0 existed, so the pipeline reproduces its own prior number.

## Two secondary findings

### The order gap collapsed

| | canonical vs reversed |
|---|---|
| zero-shot | **+3.1 pp** |
| 1-step probe | +2.2 pp |
| **arm 0** | **−0.7 pp** |
| their v1.0 | 0.0 pp |

Recorded before arm 0 as a hypothesis: the gap is presentation-order sensitivity, and it should
converge to zero as the readout learns content rather than position. A monotone sequence across four
independent measurements supports it. **Not yet a result** — n=2,000 gives a standard error of about
1.1 pp per arm, so −0.7 pp is not distinguishable from 0.0 pp, and the claim "the gap goes to zero"
and the claim "the gap is now below the resolution of this measurement" are different claims.

### `mmlu_pro_1k` did not move

| | canonical | reversed |
|---|---|---|
| zero-shot | 0.2110 | 0.1460 |
| arm 0 | 0.2110 | 0.2230 |

0.2110 = 211/1,000 in both. The canonical agreement to four decimals is coincidence — the reversed
arm moved +7.7 pp — but the picture is consistent: **training on the synth corpus moved the synth
task and left the general benchmark where it was.** Their v1.0 reached 0.3550 on the 2B.
The honest reading is that arm 0 bought no general capability, which is what a head-plus-tiny-LR
objective should produce — but this is one benchmark and one seed.

## What this is not

- **Not the curve.** It is one point. The deliverable is how far the loop carries a 0.6B model across
  arms, and arm 0 is the first.
- **Not a speed measurement of the reference.** 62,741 s vs their 908 s is 69x, on an RTX 4070
  against an H100 for a 3.3x larger model. The ratio is not a property of the recipe.
- **Not the end of the timing correction.** This session estimated 2–3 s/step from a 1-step run whose
  `train_seconds` contains only one-time setup, and from a contended 3-step probe. The true value is
  **41.8 s/step**, and the estimate was wrong by 14–20x. The estimate should never have been
  published, and the fact that the run finished at all does not repair it.

## Reproduce

```
checkpoints/tower.safetensors   1.76 GB   309 tensors
checkpoints/scorer.safetensors    29 MB
out/arm0.items.jsonl             6.7 MB   21,792 per-question records
out/arm0.jsonl                    92 KB   12 summary records
```

```bash
python measurement/report_from_items.py E:/2026-AI4S/arms/arm0/out/arm0.items.jsonl
```

The spec is field-for-field identical to the release's published `meta.json`, verified 2026-10-03
across steps, batch_size, lr_head, lr_base, seed, readout, option_order, option_pool, freeze_base,
freeze_embeddings, max_options and min_trainable_tower.
