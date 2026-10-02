# Reproduction audit — is the pipeline on the reference's track?

Written 2026-10-02, after re-reading the reference rather than the notes about it.

**Short answer: no, it was not.** Two settings in the arm-0 spec were wrong and one of them would
have changed the run, and the code being executed was not the revision that produced the release we
are trying to reproduce. Both are fixed. The check that would have caught them is now in place and
fails on the old spec.

---

## What was wrong, and how it was found

The reference publishes, beside the weights of every release, three things this project had not
looked at:

| File | What it is |
|---|---|
| `meta.json` | the **spec that produced that checkpoint**, plus the layer it tapped and the loss it ended at |
| `code/rsijev/*.py` | the module snapshot that checkpoint shipped with |
| `verify.json` | the release's own verification record |

That is the difference between reproducing a release and reproducing a description of one. It was
not read until now, so the spec in this repository had been **reconstructed** — from the checkout's
`DEFAULTS` plus the release notes — and a reconstruction is only as good as the two sources it is
built from.

```
snapshot_download("shgao/rsi-jev-v1.0-qwen3.5-2b",
                  allow_patterns=["meta.json", "verify.json", "code/**"])
```

### Error 1 — `lr_head` was off by 10×

| | value |
|---|---|
| this repository had | `1e-3` |
| the published spec | **`1e-4`** |

The reasoning that produced `1e-3` is worth recording because it was confident and it was wrong:

> the release notes say 1e-4, the code's `DEFAULTS` say 1e-3, and the code is what ran.

The second clause does not follow from the first. `DEFAULTS` in a checkout reads out the default of
**whatever revision the checkout is at**, and the checkout's `HEAD` is not the revision that
produced v1.0 — `fit.py` has grown from 213 lines to 559 since. The published spec settles it: 1e-4,
and the release notes were right.

This is the same mistake as reading a step time off a run whose conditions were not the ones being
asked about: a number is not evidence until you know what produced it.

### Error 2 — `option_order` was missing entirely, and this one changes the run

The published spec sets:

```json
"option_order": "shuffled",
"eval_option_orders": ["canonical", "reversed"]
```

The module default is `canonical`. Our spec did not mention the key, so arm 0 would have trained on
options in corpus order. Their own record names the consequence:

> Options shuffled per example, because a causal encoder shows option *k* only options 1…*k*−1, and a
> head trained on a fixed order collapses onto position — one early run picked the last option on
> **800 of 800** score questions.

So the run that was launched at 23:29 would not merely have been mis-documented; it would have been
the documented failure mode. It was stopped before completing, so nothing is invalidated, but the
prior reading of it as "on track" was not supportable.

`eval_option_orders` is the matching evaluation-side setting: scoring under both orders is how a
position artefact shows up as a *gap between two scores* rather than as a score that is quietly low.

### Error 3 — the code being run is a later revision

| module | v1.0 shipped | checkout `HEAD` | |
|---|---|---|---|
| `fit.py` | 213 lines | 559 lines | **+346** |
| `arch.py` | 539 lines | 817 lines | **+278** |
| `evaluate.py` | 297 lines | 594 lines | **+297** |
| `contract.py` `data.py` `encode.py` `metrics.py` `targets.py` `train.py` | — | — | identical |

`evaluate.py` is in the **PROTECTED** set — it defines what is scored and on which split — and it
has roughly doubled since v1.0. Two consequences, in order of severity:

1. A score produced through the current evaluator is not guaranteed comparable to their published
   v1.0 number, because the evaluator is not the one that produced it.
2. Everything the checkout adds to `fit.py` is v1.0-or-later functionality: `_retention_*` (their
   v2.0 retention mechanism), `load_init`, `_bucketed_stream`, `_text_tower_embed` (their v4.0
   vision tower). None is active under the v1.0 spec, but "not active" is an argument, not a
   measurement.

Their v1.0 `fit.py` is not a stripped loader — it contains the backward pass, the optimizer, the
training loop, bf16 autocast and the cosine schedule, and its `fit()` signature is **identical** to
the checkout's. It is the training code, at the revision that trained v1.0.

---

## What is fixed

**The spec is now copied from `meta.json`, not reconstructed.** `config/arm0_spec.json` is their
published spec field for field, with one recorded deviation:

| key | theirs | ours | why |
|---|---|---|---|
| `eval_batch_size` | 32 | 8 | **not** the OOM it was first justified by — that came from a contended run, now withdrawn; measured idle, 32 does not OOM but sits at 94% of VRAM and takes >61 min against 8's 15.6. See [eval_batch_size_correction.md](eval_batch_size_correction.md) |

**`check_arm0_spec.py` now diffs the spec against the published `meta.json` live.** This is the
check that was structurally unable to exist while the spec was a reconstruction. Any difference from
the published spec fails unless it appears in `ALLOWED_DEVIATIONS` with the measurement behind it.
Verified by negative control — restoring `lr_head: 1e-3` produces

```
[FAIL] DRIFT       lr_head: published 0.0001 but ours is 0.001
```

83/83 checks pass on the corrected spec.

**A v1.0 code environment exists.** `E:/2026-AI4S/v1_env/` holds the published `rsijev/` package
with the checkout's `run_arm_lib.py` as the driver. Interface compatibility was checked by
inspection rather than assumed: `run_arm_lib` needs `fit(model, tok, cases, enc, cfg, *, max_options,
seed, device)`, `ArchConfig(**8 keys + arch_extra)`, and `from rsijev.evaluate import as_record,
eval_precision, predict, score_predictions`; all are present in the v1.0 snapshot, and `encode.py`,
`train.py` and `contract.py` — which supply the rest of its imports — are byte-identical between the
two revisions.

The v1.0 snapshot has no `run_arm_lib.py` of its own. Borrowing the checkout's driver is a real
dependency and is recorded as one, not passed off as "their code unmodified".

---

## Verification — the corrected pipeline, executed

A 1-step run through the v1.0 modules with the published spec, on Qwen3-0.6B, writes a checkpoint
whose `meta.json` matches theirs **key for key** — all 11 top-level keys, same set — and whose
`n_train_cases` is **6277 against their 6277**. That is the corpus and the split aligned end to end,
not asserted.

| | theirs (v1.0, Qwen3.5-2B) | ours (v1.0 code, Qwen3-0.6B) |
|---|---|---|
| `n_train_cases` | 6,277 | **6,277** |
| `tapped_layer_type` | full_attention (final, normed) | full_attention (final, normed) |
| `tapped_layer` | 23 of 24 | 27 of 28 |
| `tower_keys` | 319 | 309 |
| `linear_attn_kernel` | **fla-0.5.2 / torch-2.7.1+cu128** | **torch-reference / torch-2.13.0+cu126** |
| `meta.json` top-level keys | 11 | 11, identical set |

The last two rows are the honest limits of this reproduction and are recorded rather than glossed.
`tower_keys` differs because Qwen3.5 and Qwen3 do not have the same module set. The attention kernel
differs because FLA is not installed here, so their linear-attention layers run through PyTorch's
reference path. **A `linear_attn_kernel` difference is a difference in the numerics of every
linear-attention layer**, so a checkpoint from this host is not bit-comparable to theirs, and the
comparison has to rest on `final_loss` and on the scored metrics at a stated tolerance rather than on
weight agreement.

### `eval_option_orders` is not decoration

Scoring the candidate under both orders, one step in:

| target | canonical | reversed | gap |
|---|---|---|---|
| **choice** | 0.3417 | **0.2800** | **−6.2 pp** |
| score | 0.3438 | 0.3237 | −2.0 pp |
| noul | 0.4817 | 0.4967 | +1.5 pp |

A 6.2 pp swing on `choice` between two orderings of the same options is a position artefact of the
size the reference warns about. This setting mattered and its absence would have gone unnoticed —
which is the argument for reading the published spec rather than a description of it.

The zero-shot control reproduces too: `typed_decisions` AURC **0.4919**, identical to the value
measured through the checkout's modules, so the two code revisions agree on the evaluation path that
matters for this number.

---

## What this changes about the plan

### The environment is now built, not assembled

`scripts/build_v1_env.py` produces the v1.0 environment and refuses to declare it usable until it
has checked that the borrowed driver's imports resolve against the v1.0 modules. It also compares
the two revisions file by file — and building it caught a mistake in this audit's own evidence.

The audit had compared the shared modules through `read_text`, whose universal-newline translation
turned LF and CRLF into the same string, and concluded **"byte-identical"** from a comparison that
had silently discarded the difference it was claiming to measure. Compared as raw bytes they differ;
compared with newlines normalised they are identical. The release snapshot is LF, the checkout is
CRLF, and the six shared files differ in **nothing but line endings**. The conclusion survives; the
evidence for it did not, and the builder now reports which of the three cases each file is in:

```
[OK]   contract.py    d412f9dd98f2  (newline-only difference)
[OK]   data.py        be9e87c76b8f  (newline-only difference)
...
[OK]   every name the driver imports is present in the v1.0 modules
[OK]   run_arm_lib imports against the v1.0 modules
```

That is a third variety of the same failure: a comparison that reports agreement because it
normalised away the thing it was measuring.

### The plan

The claim "RSI-Jev's code, unmodified, with only the backbone changed" was true of the substitute
statement — the checkout is unmodified — and misleading as a description of the reproduction, since
the checkout is not the code that produced the thing being reproduced. The plan becomes:

1. **Arm 0 runs in `v1_env`** — the v1.0 modules, the published spec, Qwen3-0.6B. The backbone
   remains the only substantive variable.

   ```bash
   python scripts/build_v1_env.py            # idempotent; verifies before it declares success
   cd E:/2026-AI4S/v1_env
   unset PYTHONPATH
   export PYTHONPATH="E:/2026-AI4S/v1_env;E:/2026-AI4S/v1_env/scripts"
   python scripts/release_train.py \
     --model E:/2026-AI4S/jev_repro/models/Qwen3-0.6B --seed 17 \
     --spec E:/2026-AI4S/arms/arm0/spec.json \
     --save-dir E:/2026-AI4S/arms/arm0/checkpoints --out E:/2026-AI4S/arms/arm0/out \
     --name arm0 --corpus E:/2026-AI4S/corpus_rsijev
   ```
2. **The first comparison is `final_loss` against 0.6075**, their published v1.0 endpoint, and
   `n_train_cases` against 6,277. Both are in `meta.json`, so both are checkable without asking
   whether two eval harnesses agree.
3. **Only after that**, their published *score* is a fair target — and its own `verify.json` gives
   the number to compare against through the v1.0 evaluator rather than the current one.

---

## The habit this audit is really about

Three times on this project a number has been read off something that was not the thing being asked
about: a step time divided out of a 3-step total, a run whose tower was frozen, a duration from a run
contending for the card. Each was a measurement of the right quantity under the wrong conditions.

Reading `DEFAULTS` out of a later revision and treating it as "what ran" is the fourth instance, and
it is the same shape: it is a real value, with a real provenance, used as though its provenance did
not matter.

The defence that works is not care. It is to make the reference itself the source — `meta.json` is
shipped *by* the release, in the same directory as the weights — and to make the check mechanical,
so that being wrong about it stops being possible rather than stopping being likely.
