# Correction: `eval_batch_size` was justified by a contaminated measurement

Written 2026-10-02. This retracts a claim this repository made twice and pinning a wrong reason in
three places. The setting itself does not change; the reason it is kept does.

---

## What was claimed

> `eval_batch_size: 32` OOMs when full-parameter training has left the 8.88 GiB optimiser state
> resident; the same 32 completes when the tower is frozen. **The fault is the interaction.**

with a two-cell table:

| | `eval_batch_size: 32` | `eval_batch_size: 8` |
|---|---|---|
| `freeze_base: true` | completes, zero OOM | completes, zero OOM |
| `freeze_base: false` | **`torch.AcceleratorError: CUDA error: out of memory`**, 0 files | completes, zero OOM |

## Why it is withdrawn

**The run that produced the OOM was sharing the card.** `docs/arm0_probe_evidence.md` already says so,
in the paragraph immediately above the claim:

> **A 30-step run reported `train_seconds: 3207`, or 107 s/step, and that number is discarded.** It
> ran concurrently with the eval_batch_size necessity test and the two contended for the card.

The timestamps agree: the 30-step spec was written at 22:13, the `freeze_base: false, eval_batch_size:
32` spec at 22:18. They overlap.

So one run, on a contended card, produced **two** numbers:

| number | what happened to it |
|---|---|
| `train_seconds: 3207` → 107 s/step | **discarded as contended** |
| `CUDA error: out of memory` | **kept as a hardware limit** |

That is not a defensible split. If contention is enough to make the duration meaningless, it is
enough to make the allocation failure meaningless — a card with another 6 GB job on it is not a card
reporting its own capacity. The project threw away one number from a run it had declared invalid and
kept the other. **The error was not in either judgement; it was in applying the invalidation to one
output of a run and not the other.**

## What an idle card says

Re-run with the v1.0 modules and the published spec, `freeze_base: false`, `eval_batch_size: 32`, on
an otherwise idle GPU:

| | |
|---|---|
| training | **completed** — checkpoint written, `tower.safetensors` + `scorer.safetensors` |
| evaluation | **ran 61+ minutes without an OOM**, killed by hand |
| peak VRAM | **11,583 MiB / 12,282 MiB = 94%** |
| would-have-taken | unknown; >61 min against `eval_batch_size: 8`'s **15 min 36 s** |

So **32 does not OOM here.** The interaction story is gone.

But 32 is not free, and the reason is not subtle. The evaluation is 21,792 items (10,896 questions ×
2 roles × 2 option orders). At `eval_batch_size: 8` that is 2,724 batches in 15.6 minutes — **0.34 s
per batch**. At 32 it is 681 batches, which at the same per-batch cost would be 3.9 minutes; it spent
over 61. **At 94% of VRAM the allocator is thrashing**, and that is the whole of the difference.

## What changes, and what does not

**Does not change:** `config/arm0_spec.json` still says `eval_batch_size: 8`.

**Changes:** the recorded reason, from *"32 is impossible on this card"* to *"32 is possible but sits
at 94% of VRAM and takes >4× as long; 8 is kept for headroom and speed."* That is a different kind of
claim — a preference with a measurement, not a hard limit — and it is the honest one.

`eval_batch_size` remains evaluation-only. It changes how many questions are scored per forward pass
and nothing else: not what is scored, not the metric, not the training, not the model.

## Still open

`eval_batch_size: 32` was never run to completion, so **whether the batch size changes any score is
untested.** If 8 and 32 return the same numbers, the deviation is cosmetic and either value is
defensible; if they differ, the evaluation is batch-size sensitive and that is a finding about the
evaluation rather than about this host. The pinned number is a preference until that is measured, and
it is labelled as one.

## The pattern, now at four

This project has repeatedly read a number off a run whose conditions were not the ones being asked
about — a step time divided out of a 3-step total, a duration from a frozen-tower run, a duration
from a contended run, and now an OOM from the same contended run. The first three discarded the
number. **This one kept a second output of a run already discarded**, which is the same error with
one more step in it, and it survived two commits because the claim was stated in a table that looked
like a clean two-by-two.

The defence that works is not care. It is to apply an invalidation to the **run**, not to the
individual number that happened to look wrong — and to keep every output of a run under one verdict.
