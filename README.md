# JevRSI

> **Reproduce the RSI-Jev self-improvement loop on a Qwen3-0.6B backbone. The deliverable is the
> curve, not a score.**

---

## The question

[RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) moved a 2B typed-decision model across four
releases using a loop of agents that propose hypotheses, register predictions before spending GPU
time, run the experiments, and retire their own champions when the evidence says to. **204 arms.**
It published the failures alongside the successes.

This project runs that same loop against a **0.6B** backbone.

> Given the same loop, how far does it carry a smaller tower before it saturates, and what does the
> arm-by-arm record explain?

A loop that needed four releases to move 2B should have more headroom at 0.6B, so the curve should be
**longer**. If it is the same length, the loop's progress is set by the loop rather than by the
backbone — a sharper claim, and one this hardware can actually test.

## Why this is not a score comparison

| | RSI-Jev | JevRSI |
|---|---|---|
| Backbone | Qwen3.5-2B-Base | **Qwen3-0.6B** |
| Hardware | H100 80 GB | RTX 4070, 12 GB |
| Stack | torch 2.7.1+cu128 / fla 0.5.2 | torch 2.10 |
| v1.0 held-out | 0.662 | not yet measured |

Their numbers are **not reproducible on this hardware and are not claimed to be.** Every figure
carries the project and the machine that produced it. The two sets may share a table; they are never
averaged or differenced.

## The task

One forward pass returns a probability for every option of a typed question about a document —
yes/no, pick-one-of-k, rate-on-a-rubric — and generates nothing. A decision about a document already
read costs about 10 ms.

## Adopted, and not

**Adopted unchanged — the guard rail.** Every module declares itself EDITABLE or PROTECTED, and the
declaration lives in the module rather than in a list beside it. The loop may rewrite the model
(`arch.py`), the data (`data.py`) and the training objective (`train.py`). It may not rewrite what a
typed decision *is* (`contract.py`), what is scored (`evaluate.py`, `metrics.py`, `targets.py`) or how
the input is laid out (`encode.py`); `fit.py` is protected in its bookkeeping and editable through its
config. A change to scoring is a change to the task and goes through a human.
`scripts/check_edit_guard.py` enforces it against the code the arms actually execute, and checks the
declarations against the reference repository's own copy of each module so that reclassifying one is
a failure rather than a redefinition.

**Adopted — the distillation corpus.** `n4ze3m/typed-decisions-synth`, built by the source's own
`scripts/build_synth_corpus.py`: **6,977 documents / 24,320 questions**, each carrying a teacher's full
probability distribution (`deepseek-v4.1-flash`, mean of three), plus a separate
**`synth_adjacent.jsonl`** of 437 documents the builder sets aside "so an arm can" choose whether to
include them. Fitting the distribution rather than the label is the substance of their v1.0, and it
is what arm 0 trained on: 6,277 cases after the spec's source filter, the same count their own v1.0
records.

**Adopted — the v1.0 recipe.** 1,500 steps × batch 16; tower 5e-6 cosine, head 1e-4 constant;
options shuffled per example.

**Not adopted — their kernel stack.** `flash-linear-attention` on this GPU is a different code path,
and their README warns the backward pass is wrong on some architectures.

## Pre-registered predictions

Written before any arm runs, so the curve cannot be narrated into existence afterwards. Each is
falsifiable by a measurement this project can make.

**Where each stands.** The verdicts below are recorded against the numbers that produced them, and
a prediction recorded as unsupported stays in the list rather than being edited out of it.

- **P1 Headroom — CONFIRMED.** The 0.6B starting point scores below RSI-Jev's 2B v1.0 on a held-out
  set, measured by our own code: zero-shot `typed_decisions` pooled top-1 **0.4025**, and after the
  v1.0 recipe at 1,500 steps **0.5775**. Their 2B v1.0 is 0.6525 (their held-out figure 0.662). Both
  our points sit below it, so **+0.075** remains between our arm 0 and their v1.0.
- **P2 Curve length — open, and the source's own record predicts falsification.** The gate is
  +0.020 on a 3-seed mean. Reading their published scale trend: the 2B's total loop gain was
  **+0.1400** and the 0.8B's **+0.0616** — so the loop's leverage itself scales with the backbone,
  and a log-linear extrapolation to 0.6B gives **+0.0370**, which is room for **1.8** arms at a
  +0.020 gate. That is ≤ 4, which is the stated falsification condition. This is an extrapolation
  and not a measurement; it is the one claim a 0.6B run would settle.
- **P3 A different failing set — supported at 0.8B by the source's record.** Their 0.8B line's keeper
  "is a different recipe" from the one that kept at 2B. Confirming it at 0.6B needs our own arm.
- **P4 The bf16 scorer defect does not reproduce — CONFIRMED.** Measured: our scorer opens with a
  normaliser that casts to fp32 before the multiply, so autocast(bf16) leaves a relative error of
  5.8e-03 and does not change the argmax. Their seven failed experiments were a real failure of
  *their* scorer, not a property of the task.
- **P5 Position-collapse is the first failure mode — NOT SUPPORTED.** The order gap *closed* after
  training rather than opening: canonical − reversed was **+3.10 pp** zero-shot (0.4025 / 0.3715) and
  **−0.75 pp** after arm 0 (0.5775 / 0.5850). A readout collapsing onto option *k* would widen that
  gap. Their record agrees: "Order averaging at inference is null: +0.001, canonical plus reversed."

The verdicts in P1 and P4 cost no GPU time, because both questions were already answered by
measurements this project had made for other reasons. That is the point of writing them down before
the runs: two of five resolved without a run, one resolved against the prediction, and only two
needed the loop at all.


## Licence and provenance

Code MIT. RSI-Jev (Shanghua Gao) is MIT and not affiliated with TypeSafe AI. This project reuses its
ideas, loop structure and public data, and claims none of its work as ours. Base weights follow their
own licences.
