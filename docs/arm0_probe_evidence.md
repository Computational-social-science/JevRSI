# Arm 0 — probe evidence

Every number here was produced by RSI-Jev's own code, unmodified, at
`code_version 5ac6dcd`, on this host. Probes are diagnostic; arm 0 itself is the 1,500-step run.

---

## The frozen control, measured three times

The `control` role is `zero_shot_logprob` with `trainable: 0` — the frozen base read out by its own
logprobs. It is the bar any arm has to clear before it can be said to have learned anything.

| probe | noul | choice | score |
|---|---|---|---|
| 3-step | 0.5717 | 0.3750 | 0.3125 |
| 1-step, `freeze_base: true` | 0.5567 | 0.3767 | 0.3063 |
| 30-step | 0.5567 | 0.3767 | 0.3063 |

Spread 1.5 pp on noul, 0.2 pp on choice, 0.6 pp on score. **The control is stable enough to be a
baseline**, which is the only property this project needs from it. Their v1.0 record tabulates the
equivalent control at 0.3775 pooled for a 2B tower; ours is a different number on a different model
and the two are not comparable.

Majority baselines on typed-decisions test: noul 0.5067, choice 0.4867, score 0.3400.

---

## Learning is happening, and it is measurable at 2% of the schedule

30 steps is 2% of the 1,500-step recipe. That is enough to see the mechanism work.

| quantity | step 0 | step 29 |
|---|---|---|
| soft cross-entropy | 1.2698 | **0.9869** (−22%) |

| target | metric | 3-step | **30-step** | control |
|---|---|---|---|---|
| in_distribution | minDS | +16.38 | **+38.66** | +31.34 |
| typed_decisions | AURC | 0.6026 | **0.4817** | 0.4919 |
| typed_decisions | noul acc | — | **0.6333** | 0.5567 |
| typed_decisions | choice acc | — | **0.3483** | 0.3767 |
| typed_decisions | score acc | — | **0.3812** | 0.3063 |

**Two things cross the frozen control for the first time at 30 steps**: in-distribution minDS
(+38.66 against +31.34) and typed-decisions AURC (0.4817 against 0.4919). Per type, noul is +7.67 pp
and score +7.50 pp over the control; choice is −2.83 pp, still below its own majority baseline of
0.4867.

**None of this is arm 0's result.** 30 steps is 2% of the schedule, one seed, and the model is still
below the majority baseline on two of three types. What it establishes is that the pipeline learns,
that the control is a real bar rather than a formality, and that the run is worth 9.6 hours.

---

## Step time: 23 s/step, and one contaminated measurement

| condition | measured | 1,500 steps |
|---|---|---|
| `freeze_base: true` (head only) | 1 s/step | 0.4 h |
| **`freeze_base: false` (the real arm)** | **23 s/step** | **9.6 h** |

> **CORRECTED 2026-10-02. The 23 s/step figure has no traceable source and is contradicted by a
> direct measurement. Use ~2-3 s/step, and ~1.25 h for 1,500 steps plus ~8 min of setup.**
>
> The clean number comes from the 1-step run through the v1.0 modules with the published spec at
> `freeze_base: false`, whose own `meta.json` records `train_seconds: 2`. It is consistent with the
> only other uncontended run: the 3-step probe's `train_seconds: 467` decomposes as ~458 s of one-off
> cost (model load, the fp32 tower copy, corpus encoding) plus a ~9 s loop. Their own 2B on an H100
> records `908 / 1500 = 0.61 s/step`, so the ratio is 4.9x in the direction the hardware predicts.
>
> Why the wrong number survived: it was written down as a measurement and then reused for planning
> without ever being traced back to a run. Every step-time error on this project has been a number
> read off a run whose conditions did not match the question -- this one is a number read off **no
> run at all**, which is the same failure with the provenance removed entirely. The rule stands and
> is now applied to the figure itself: a step time with no run attached is not a measurement.
>
> Consequence for scheduling: arm 0 is a ~1.4-hour job, not a 9.6-hour one. It was launched believing
> the larger figure, which cost nothing here but would have decided against launching it.

> **CORRECTED 2026-10-02 (second correction). The 23 s/step figure had no run behind it; the
> replacement is also not a rate. `train_seconds: 2` from a 1-step run is a LOWER BOUND, not
> s/step.** Arm 0 was launched on the belief that 1,500 steps would take ~75 minutes. It passed 118
> minutes still training, so the true rate is **at least 4.4 s/step** and the projection was low by
> a factor of two or more.
>
> Why a 1-step measurement cannot give a rate: the first step does not share the steady state's
> conditions. CUDA autotunes kernels on first execution, the allocator warms up, and gradient
> checkpointing installs its hooks during the first forward pass. A one-step timing measures that
> setup, not a step.
>
> The exact figure will come from arm 0's own `meta.json`, which records `train_seconds` for all 1,500
> steps. Until that lands, **no projection from this project should be trusted for scheduling** -- and
> the failure mode to watch for is the same one as before: a real number, from a real run, whose
> conditions were not the ones being asked about.

The 22 s/step gap is the 8.88 GiB optimiser state, traversed in the backward pass and written back
to every step.

**A 30-step run reported `train_seconds: 3207`, or 107 s/step, and that number is discarded.** It
ran concurrently with the eval_batch_size necessity test and the two contended for the card. It is
recorded here rather than deleted because it is the shape of error this project has now made three
times — reading a number off a run whose conditions were not the ones being asked about:

1. `train_seconds: 467` over 3 steps, divided by 3 → 156 s/step, charging every one-off cost to
   every step. Wrong by an order of magnitude.
2. `trained 6277 cases in 1s` from a `freeze_base: true` run → "25 minutes". Right number, wrong
   condition; the real arm does not freeze the tower.
3. `train_seconds: 3207` from a run sharing the GPU → 107 s/step. Discarded for contention.

---

## eval_batch_size: 8, and why the first two explanations were both wrong

> **RETRACTED 2026-10-02 — see [`eval_batch_size_correction.md`](eval_batch_size_correction.md).**
> The claim below is withdrawn. The OOM it rests on came from the same contended run whose
> `train_seconds` this document discards four paragraphs earlier; one invalidated run cannot yield
> one number kept and another discarded. Re-measured on an idle card, `eval_batch_size: 32` does
> **not** OOM — it occupies 94% of VRAM and takes >61 minutes against 8's 15.6. The setting stays 8;
> the reason changes from impossibility to headroom and speed. The text below is kept as written.

| | `eval_batch_size: 32` | `eval_batch_size: 8` |
|---|---|---|
| **`freeze_base: true`** | completes, zero OOM | completes, zero OOM |
| **`freeze_base: false`** | **`torch.AcceleratorError: CUDA error: out of memory`**, 0 files | completes, zero OOM |

**The fault is the interaction.** Neither condition alone triggers it; both are required. In the
failing cell training had already completed and their own guard had already fired
(`tower moved: max |dw| = 5.007e-06`); the process died afterwards, during evaluation, having
written nothing.

The two wrong explanations are kept because they are the reusable part:

- *"Evaluation lacks `grad_checkpointing`, so 32 is too big."* Plausible, untested, and written into
  the spec before it was tested.
- *"32 is fine; the optimiser state alone was the fault."* From a `freeze_base: true` control that
  completed at 32. **That control removed half the cause**, so it could only return a false
  all-clear — and a retraction built on it was wrong in the same direction as the claim it retracted.

An isolation that removes the phenomenon cannot report on the phenomenon.

`eval_batch_size: 8` changes how many questions are scored per forward pass and nothing else — not
what is scored, not the metric, not the training, not the model. It is pinned in
`scripts/check_arm0_spec.py` with the interaction recorded, because a vaguer reason is what let two
consecutive commits contradict each other.

`batch_size: 16` is untouched and pinned for the opposite reason: it is their number, and nothing
measured here justifies changing it.

---

## Run conditions, all measured

```
code_version         5ac6dcd                      their commit, unmodified
tapped_layer         27                           Qwen3-0.6B has 28 layers
tapped_layer_type    full_attention (final, normed)   their 2B taps layer 23, also full_attention
linear_attn_kernel   torch-reference/torch-2.13.0+cu126   no fla involved
excluded             embed_tokens (frozen)
trainable            447,815,681                  their own count
tower moved          max |dw| = 5.007e-06          their own guard, it fires
peak VRAM            11,874 / 12,282 MiB = 96.7%, flat
```

`readout_layer: -1` landing on a full-attention layer on both towers is the single result that makes
"run their code on our backbone" a reproduction rather than a rewrite. Their arch is written against
Qwen3.5's hybrid stack, where only every 4th layer can attend freely, so a tap that lands on the
wrong attention type is a failure mode they warn about explicitly. Ours is uniform, and the tap is
the same kind of layer.
