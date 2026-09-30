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

**Adopted unchanged — the guard rail.** Every module declares itself EDITABLE or PROTECTED. The loop
may rewrite the model, the data and the training objective. It may not rewrite what a typed decision
*is*, what is scored, or on which split. A change to scoring is a change to the task and goes through
a human. See [`docs/EDIT_GUARD.md`](docs/EDIT_GUARD.md).

**Adopted — the distillation corpus.** `n4ze3m/typed-decisions-synth`: 6,682 documents, 23,317
questions, each carrying a teacher's full probability distribution (`deepseek-v4.1-flash`, mean of
three). Fitting the distribution rather than the label is the substance of their v1.0.

**Adopted — the v1.0 recipe.** 1,500 steps × batch 16; tower 5e-6 cosine, head 1e-4 constant;
options shuffled per example.

**Not adopted — their kernel stack.** `flash-linear-attention` on this GPU is a different code path,
and their README warns the backward pass is wrong on some architectures.

## Pre-registered predictions

Written before any arm runs, so the curve cannot be narrated into existence afterwards. Each is
falsifiable by a measurement this project can make.

- **P1 Headroom.** The 0.6B starting point scores below RSI-Jev's 2B v1.0 (0.662) on a held-out set,
  measured by our own code.
- **P2 Curve length.** More arms clear the gate than the four releases RSI-Jev needed. *Falsified if
  the plateau arrives in ≤ 4 accepted arms.*
- **P3 A different failing set.** Arms that kept at 2B fail at 0.6B, and the reverse.
- **P4 The bf16 scorer defect does not reproduce.** Measured: our scorer opens with a normaliser that
  casts to fp32 before the multiply, so autocast(bf16) leaves a relative error of 5.8e-03 and does
  not change the argmax. Their seven failed experiments were a real failure of *their* scorer, not a
  property of the task.
- **P5 Position-collapse is the first failure mode to appear**, because smaller towers are where a
  causal encoder most easily collapses onto option *k*. Their record: one early run picked the last
  option on 800 of 800 score questions.

## Licence and provenance

Code MIT. RSI-Jev (Shanghua Gao) is MIT and not affiliated with TypeSafe AI. This project reuses its
ideas, loop structure and public data, and claims none of its work as ours. Base weights follow their
own licences.
