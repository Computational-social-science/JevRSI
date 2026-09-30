# Pre-registration — Floor B on the Qwen3-0.6B + LoRA seed

**Written 2026-09-29, before any replicate of this experiment was run.**
No accuracy, no SD, no comparison between the two arms existed when this file was written. The only
numbers available at the time are the ones already published elsewhere in this repository, and none of them
is an outcome of this experiment.

## Why this experiment exists

`ε(dev) = 12.38 pp` currently circulating in this project is **provisional and optimistic**. It was derived
with a Floor B of 0.40 pp that was measured on a *different training path* — a retired 322M
full-parameter arm. `measurement/INSTRUMENT_CALIBRATION.json` says in its own words:

> Re-derive Floor B on its OWN seed, because 0.40 pp is a property of this pipeline's training path.
> Do not inherit it silently.

Floor B enters the threshold in a variance term added in quadrature, so a borrowed value that is too small
makes the threshold too small. The accept rule cannot be armed until this seed's own Floor B exists, and no
claim of improvement can be made against a threshold that was never measured for the thing being improved.

## What is being measured

**Floor B** = the standard deviation, across runs differing *only* in random seed, of the final dev-split
score produced by one fixed training recipe.

Both arms run the identical recipe. The arms differ in exactly one thing: whether the frozen backbone is held
in BF16 or quantised to 4-bit. Nothing else varies — same data order, same step count, same effective batch,
same LR schedule, same evaluation.

## Recipe (fixed by this document, not by the code)

| Item | Value | Source |
|---|---|---|
| backbone | Qwen3-0.6B, local copy | `configs/autoresearch.yaml` |
| seed weights | `checkpoints/autoresearch/final.safetensors`, loaded `strict=True` | pre-existing checkpoint |
| adapter | LoRA r=16, α=32, dropout 0.05, on `q,k,v,o,gate,up,down` | this document |
| trainable | adapter + the set-transformer decision head; backbone otherwise frozen | this document |
| steps | 600 | `protocol.json`, unchanged |
| batch | `batch_states=4`, `grad_accum=4` (effective 16) | `protocol.json`, unchanged |
| LR | embed/bottom/middle/top 1e-5, head 1e-4, layer-wise decay | `protocol.json`, unchanged |
| schedule | cosine, 2% warmup, grad clip 1.0 | `protocol.json`, unchanged |
| loss weights | brier 0.1, others 0 | `protocol.json`, unchanged |
| data | train split only; dev/calibration weights 0 | verified case-by-case against `protocol.json` |
| replicates | 5 seeds per arm | this document |
| seeds | 20260921, 20260922, 20260923, 20260924, 20260925 | this document |
| evaluation | frozen evaluator, dev split, 600 questions, L1 temperatures | `scripts/evaluate_split.py` |

**5 seeds per arm, not more.** The SD is estimated at df = 4, and this project has already been bitten by
quoting a floor measured at df = 1 (it overstated by 3.53×). 5 is the minimum that yields a usable df; the
width of the resulting CI is reported alongside the SD and is part of the result, not an afterthought.

## Primary question

What is this seed's Floor B, and therefore what is the real ε?

**Decision rule, fixed now:** Floor B is the sample SD of the 5 final dev scores within an arm, reported with
its df and with a 95% CI from the chi-square interval. ε is then re-derived by `loop/epsilon.py` with that
Floor B and nothing else changed. The provisional 12.38 pp figure is **superseded** by whatever comes out,
in either direction. If the new ε is *smaller*, that is still a result and is reported as one.

## Secondary question

Does 4-bit quantisation cost anything measurable?

Both arms are scored on the same dev split with the same evaluator. The comparison is reported as the
difference in Floor B, with a CI. **This is a comparison of noise, not of accuracy** — a quantised arm could
score better or worse on the mean and that is not the question being asked.

Decision rule, fixed now: 4-bit is adopted **only if** it reduces the measured VRAM requirement enough to
matter *and* does not inflate Floor B beyond the point where ε becomes larger than the usable headroom. The
VRAM side is already settled by `loop/probe_lora_feasibility.py`: BF16 + LoRA peaks at 7.80 GiB against an
11.99 GiB card, so the memory argument for quantisation does not exist on this host. Therefore the burden of
proof is entirely on the noise side, and the default expectation is **BF16 without quantisation**.

## Falsification conditions, fixed now

This experiment can fail, and each way it can fail is written down in advance:

1. **The two arms are not comparable** — if the 4-bit arm cannot be made to run (no `bitsandbytes` kernel on
   this host, or a load error), the secondary question is reported as *not answerable here* and NOT
   substituted with a proxy. The primary question is unaffected.
2. **LoRA does not train** — if the loss does not decrease, or the adapter receives no gradient, the recipe is
   reported as broken. This is not evidence about LoRA; it is a defect in the harness.
3. **Floor B is larger than the usable headroom** — if the measured ε exceeds the ~20 pp of headroom above
   the seed, then **no experiment in this project can detect a real improvement on dev**, and that is the
   finding. It is reported as a negative result about the design, not as a failure of the measurement.
4. **The replicates disagree with the seed's own starting point** — if training makes the seed *worse* on
   dev, Floor B is still measurable (it is a dispersion, not a gain) but the recipe is unsuitable for the
   loop, and that is reported separately.

## What this experiment cannot establish

- It says nothing about the frozen test split. Test is touched once, at the end, under a pre-registered rule.
- It is not a claim of improvement. Nothing here improves anything.
- It does not generalise beyond this benchmark, this backbone and this adapter rank.
- Five replicates bound the SD loosely. The reported CI is wide, and the width is the honest answer to "how
  well do we know Floor B", not a detail to be dropped because the point estimate is what we wanted.

## Recorded at write time

- Day-1 seed baseline, dev, frozen evaluator: accuracy 0.8150, chance-corrected 0.7326, soft CE 0.8243,
  Brier 0.0476, ECE 0.1481, score MAE 0.1849. (Source: `measurement/day1_baseline_metrics.json`,
  recomputed from saved logits.)
- Borrowed Floor B, *not* this seed's: 0.40 pp (a retired 322M full-parameter arm, df = 4).
- Hardware: full-parameter 13.26 GiB (does not fit); BF16 + LoRA r16 peak 7.80 GiB (fits, 4.20 GiB spare).
- `bitsandbytes`: not installed at the time of writing. The 4-bit arm is contingent on that changing.

---

# Amendment 1 — batch composition, and crash recovery

**Added 2026-09-29, still before any replicate of this experiment has completed.** At the time of writing
this amendment, zero of the ten planned replicates had finished, and no accuracy or SD from this experiment
exists. Both changes below are therefore recorded before the fact, which is the only time an amendment to a
pre-registration is legitimate. Neither change was made in response to a result.

## A1.1 — `batch_states=1, grad_accum=16` replaces `batch_states=4, grad_accum=4`

**What changes.** The recipe table above says `batch_states=4, grad_accum=4`. The runs use
`batch_states=1, grad_accum=16`.

**Why.** The EFFECTIVE batch is 16 either way, so the optimiser sees the same number of states per step and
the LR schedule is unchanged. What differs is the micro-batch composition: 4 sequences of 4 states per
forward pass, versus 16 passes of 1 state. Measured peak memory was 11.8–11.9 GiB and >45 s per optimiser
step at `batch_states=4` without gradient checkpointing, against a 11.99 GiB usable card. In other words the
pre-registered setting did not run on this hardware, and the binding constraint was activation memory, not
optimiser state.

**What this does and does not affect.** It does not change the quantity being measured, provided — and only
provided — *every* replicate uses the same setting. Floor B is a dispersion of whole runs; if some replicates
ran at 4×4 and others at 1×16, the measured spread would be the spread of two different recipes wearing one
recipe's name. All ten replicates therefore use 1×16, and the arm labels in `floor_b_lora.json` record it.

**Honest caveat.** Micro-batch composition does perturb the floating-point accumulation order, so runs at
1×16 are not bit-comparable to runs at 4×4. The pre-registered 4×4 numbers do not exist and cannot be
recovered. Any future comparison against a differently-batched measurement must be attributed, not merged.

## A1.2 — exact crash recovery is part of the recipe

**What changes.** A replicate may be interrupted and continued from a periodic snapshot. Training accepts
`--resume auto|off|<path>`; snapshots are written every 25 steps and the two newest are retained.

**Why this needed writing down.** A replicate is 600 steps at a **measured 12.6 s/step = 2.11 h**, and there are
ten of them, so the batch is ~21 h on a machine whose E: volume disconnected twice in the day this was
written. Without recovery, each disconnection costs up to 2.11 h. With it, the cost is at most 25 steps
(~5 min).

**A correction worth recording.** The first draft of this amendment put the rate at 33.5 s/step and the
batch at ~56 h. That figure came from a run that was sharing the GPU with an orphan left behind by an
earlier `timeout`, and it was wrong by a factor of 2.7. The number above is re-measured on an idle GPU.
The episode is kept here because the pre-registration's own subject is *noise*, and a throughput figure
contaminated by contention is exactly the kind of plausible-looking wrong number this experiment exists
to prevent.

**The measurement that showed 216 h/seed is invalid.** `measurement/profile_train_step.py` projected
1299 s/optimizer step and a 12.95 GiB peak (headroom −0.96 GiB, i.e. "this recipe OOMs"). Both are
artifacts: `time_one()` calls `reset_peak_memory_stats()` and then times `n_micro` microbatches with **no
warmup**, so the first call's cuBLAS kernel selection, lazy CUDA module loading and allocator first-touch
dominate a 3-sample mean. Steady state is 0.79 s/microbatch and 5.3 GB. Had that profile been trusted, the
pre-registered recipe would have been rejected on a hardware limit it does not hit. Any future timing
taken from this script must add a warmup loop and raise `n_micro` before its numbers are used.

**The condition that makes it legitimate.** A continued replicate must be *one draw from the recipe*, not
two. That requires the data order, the dropout stream and the accumulation-window position all to be
restored, and it requires that this be demonstrated rather than asserted. `measurement/test_resume_exact.py`
is therefore part of this pre-registration, not a convenience: it runs a seed straight through, kills a
second run by `taskkill /F /T` once `checkpoint-3.pt` lands, resumes it, and requires the post-resume steps to
match the uninterrupted run's loss and LR **as exact strings, with no tolerance**. A test with a tolerance
would not be testing the property that matters.

If that test fails, resume is disabled and a replicate that was interrupted is re-run from step 0 rather
than continued.

**Disclosure obligation.** `floor_b_lora.json` records, per replicate, whether it was interrupted and how
many times. A Floor B built partly from continued runs is reported as such, alongside the uninterrupted
runs, so a reader can see the provenance of every number in the SD.
