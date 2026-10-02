# Arm 0 — host feasibility, measured

Three things had to be established before arm 0 could be launched, none of which a config file
can tell you. All three were measured on this host on 2026-09-30.

---

## 1. Does a 0.6B tower train full-parameter in 12 GB?

**Yes, and with almost no margin.**

Measured peak during a real 3-step run of *their* code, sampling `nvidia-smi` every 5 s:

| t | used |
|---|---|
| 5 s | 1,912 MiB (weights resident) |
| 35 s | 8,035 MiB (optimiser state allocated) |
| 90 s | 11,817 MiB |
| 130 s | **11,872 MiB**, flat thereafter |

**Peak 11,872 / 12,282 MiB = 96%.** Flat, not still climbing.

The arithmetic agrees. Trainable parameters with `freeze_embeddings: true`:

```
total            751.6 M
input embedding  155.6 M   frozen
trainable        596.0 M   -> min_trainable_tower = 100 M is satisfied 6x over
```

At fp32 weights and a plain `torch.optim.AdamW` — which is what `rsijev/fit.py` constructs, with no
offload path anywhere in their code — the optimiser term is 16 B/param:

```
596.0 M x 16 B = 8.88 GiB   + 1.40 GiB weights resident  =  10.28 GiB
```

leaving ~1.7 GiB for activations, which is what the measured peak spends.

### Why this is tight rather than comfortable

`batch_size: 16` is *their* number, set for an H100 80 GB. Their own code says activations at fp32
for 24 layers at batch 16 used **79 GB**, which is why `autocast_bf16` exists and why
`run_arm_lib` derives it as `not freeze_base`. We get the same saving: weights stay fp32, the
forward runs under bf16 autocast. That is what puts the peak at 96% rather than far past 100%.

**410 MiB of headroom is not a margin.** If arm 0 OOMs, `batch_size` is the first thing to move and
the second is a CPU-resident optimiser master. This host has **127.7 GiB RAM, 99.5 GiB available,
32 logical cores**, so the second option costs PCIe bandwidth and nothing else — but it would be an
adaptation, and an adaptation has to be recorded as one rather than slipped in.

---

## 2. Where does `readout_layer: -1` land on a 0.6B tower?

**On the same kind of layer it lands on in theirs.**

Their `arch.py` is written against Qwen3.5's hybrid stack — `linear_attention x3` then
`full_attention`, so only every 4th layer can attend freely — and it carries a warning that a
`readout_layer` index is off by one from `layer_types`, landing on the wrong *attention type*
rather than a neighbouring depth. On their 2B, `-1` taps layer 23, a full-attention layer.

Measured on Qwen3-0.6B:

```
n_layers     28
layer_types  ['full_attention'] x 28       uniform, no hybrid
readout_layer=-1  ->  (27, 'full_attention (final, normed)')
```

**Same attention type.** Their architectural assumption carries over; the tap differs in depth
(27 against 23) and that is a depth difference, not a kind difference. This is the single most
important result in this file, because it is what makes "run their code on our backbone" a
reproduction rather than a rewrite.

---

## 3. Does their code run on this host's stack at all?

**Yes.**

```
torch 2.13.0+cu126, transformers 5.3.0, CUDA 12.6
rsijev.arch, rsijev.fit, rsijev.train, rsijev.contract   all import
flash-linear-attention                                  NOT a hard dependency of those four
targets: load_mmlu_pro_1k() 1000 items, load_typed_decisions('test') 400, ('train') 1200
```

Their `requirements-repro.txt` pins `torch 2.7.1+cu128` and warns that numbers are only comparable
within one kernel stack. That warning is about their *numbers*, not about whether the code runs.
Our runs are ours, measured here, and are never differenced against theirs.

## 4. End-to-end: their code ran, on our backbone, unmodified

A 3-step probe through `scripts/release_train.py` -- their entry point, no file of theirs edited:

```
python scripts/release_train.py --model E:/2026-AI4S/jev_repro/models/Qwen3-0.6B \
    --seed 17 --spec <spec> --save-dir <ckpt> --out <out> --name probe \
    --corpus E:/2026-AI4S/corpus_rsijev
```

What came back, read from `ckpt/meta.json`:

| field | value | what it settles |
|---|---|---|
| `code_version` | `5ac6dcd` | the reference checkout's own commit. Unmodified. |
| `tapped_layer` | `27` | matches the read-only probe |
| `tapped_layer_type` | `full_attention (final, normed)` | same attention type as their 2B |
| `linear_attn_kernel` | `torch-reference/torch-2.13.0+cu126` | no `fla` involved; pure torch path |
| `excluded` | `["embed_tokens (frozen; taken from base_model)"]` | `freeze_embeddings` took effect |
| `train_seconds` | `467` | 3 steps |
| `final_loss` | `2.1995` | trained, not diverged |
| `n_train_cases` | `6,277` | cases after the calibration split |
| `tower.safetensors` | 1.76 GB | the tower was written, so it was trained |
| `scorer.safetensors` | 29.4 MB | the cross-attention head was written |

**A 1.79 GB checkpoint on disk is the proof that full-parameter training happened.** A frozen-tower
arm would have written a ~29 MB head and nothing else.

### Step time — three measurements, and two corrections

The path to a real number went through two wrong ones, both recorded because the errors are the
reusable part.

**First attempt, wrong by an order of magnitude.** `train_seconds: 467` over 3 steps was divided by
3, giving 156 s/step and a projected 32-65 h. The 467 s is dominated by costs that do not scale with
steps -- corpus encoding, candidate vectors, writing 1.79 GB -- and dividing a total by a small
sample charges all of it to the training. A total divided by a small sample is a bound, not a
measurement.

**Second attempt, wrong in the other direction.** A control printed the line their trainer emits:

```
trained 6277 cases in 1s  loss [1.2714]
```

and 1,500 steps was reported as about 25 minutes. That run had `freeze_base: true`. The real arm
does not: `freeze_base: false` is what reproduces v1.0, and their own control table records that
every frozen variant of theirs stalled below the majority baseline.

**The measurement that settles it**, at `freeze_base: false` and therefore the real arm's condition:

```
trained 6277 cases in 23s  loss [1.2698]
tower moved: max |dw| over probe tensors = 5.007e-06
```

| condition | measured | 1,500 steps |
|---|---|---|
| `freeze_base: true` (head only) | 1 s/step | 0.4 h |
| **`freeze_base: false` (the real arm)** | **23 s/step** | **9.6 h** |

The 22 s/step difference is the 8.88 GiB optimiser state: every step traverses it in the backward
pass and writes back to it. It is also, incidentally, the reason the evaluation OOM in the 3-step
probe happened when it did.

`tower moved: max |dw| = 5.007e-06` is their own guard reporting that the tower weights actually
changed. That is the check that makes a full-parameter claim mean anything, and it fires.

Their 1,500 steps at 2B took about 13 minutes on an H100 80 GB. Ours at 0.6B on a 4070 is about 9.6
hours, roughly 44x longer. The ratio carries no scientific meaning -- different hardware, different
precision stack, different model size -- but it is the honest cost of this arm on this card, and it
is a one-afternoon job rather than an overnight one.

---

### Evaluation OOMs where training does not — an interaction, and two wrong diagnoses

The 3-step probe finished with exit 0 and wrote every record, but the CUDA allocator had been
failing during evaluation:

```
memory allocation failed with OOM on device 0 while trying to allocate 134217728 bytes
(free: 0, total: 12,878,086,144)
```

Getting to the real cause took two wrong answers, both of which had already been written into the
spec before they were tested.

**Wrong answer 1 — "evaluation lacks `grad_checkpointing`, and 32 is too big for 12 GB."** Plausible,
untested, and acted on.

**Wrong answer 2 — "32 is fine; the fault was the optimiser state alone."** This came from a control
at `steps: 1` with `freeze_base: true`, which completed all three targets at 32 with zero OOM:

```
saved release checkpoint -> ...\ck32 (309 tower tensors)
[native/canonical] typed_decisions  minDS -31.10 (ctrl -21.75)  AURC 0.6026 (ctrl 0.4888)
```

That looked like a clean refutation, and the previous commit retracted the claim on its strength. The
control was invalid: `freeze_base: true` removes the 8.88 GiB optimiser state, which is **half the
fault**, so it could only ever return a false all-clear. An isolation that removes the phenomenon
cannot report on the phenomenon.

**The measurement that settles it** keeps the real arm's condition — `freeze_base: false`, so the
optimiser state is resident — and varies only the key under test:

> **RETRACTED 2026-10-02 — see [`eval_batch_size_correction.md`](eval_batch_size_correction.md).**
> The table below and the conclusion drawn from it are withdrawn. The OOM came from the same
> contended run whose `train_seconds` [`arm0_probe_evidence.md`](arm0_probe_evidence.md) discards;
> an invalidated run cannot yield one number kept and another discarded. Re-measured on an idle card,
> `eval_batch_size: 32` does **not** OOM — it occupies 94% of VRAM (11,583 / 12,282 MiB) and takes
> >61 minutes where 8 takes 15.6. **The setting stays 8; the reason changes** from "necessary"
> to "kept for headroom and speed". The text below is kept as written.

| | `eval_batch_size: 32` | `eval_batch_size: 8` |
|---|---|---|
| **`freeze_base: true`** | completes, zero OOM | completes, zero OOM |
| **`freeze_base: false`** | **`torch.AcceleratorError: CUDA error: out of memory`**, 0 files written | — |

**The fault is the interaction.** Neither condition alone triggers it and both are required. In the
failing cell, training had already completed and their own guard had already fired
(`tower moved: max |dw| = 5.007e-06`); the process died afterwards, during evaluation, having
written nothing.

`eval_batch_size: 8` is therefore **necessary, and measured** — and the reason is recorded as the
interaction rather than as either half of it, because a vaguer story would let the next reader
"restore" 32 against an argument that does not hold. It changes how many questions are scored per
forward pass and nothing else: not what is scored, not the metric, not the training, not the model.
`batch_size: 16` is untouched, and pinned in the validator for the opposite reason — it is *their*
number and nothing measured here justifies changing it.

### Two probes that did not measure what they intended

- `steps: 0` is not "skip training" in their code. It is a load check, and `fit()` rejects it
  without `init_from`: `ValueError: steps=0 is only meaningful with init_from (a load check)`. The
  work-around is `steps: 1`, not `steps: 0`.
- The `freeze_base: true` isolation removed the cause of the OOM it was meant to study. A negative
  result from an experiment that removed the phenomenon is not evidence of its absence — and the
  error survived a retraction, because the retraction was built on the same invalid control.

### What the probe scored, and what it does not mean

`out/probe.jsonl` holds six records: three targets x two roles. The `control` role is
`zero_shot_logprob` with `trainable: 0` -- the frozen base read out by its own logprobs, which is
the control their own v1.0 record tabulates. The `candidate` role reports `trainable: 447,815,681`.

| type | candidate (3 steps) | control (zero-shot) | majority base | delta |
|---|---|---|---|---|
| noul | 0.6117 | 0.5717 | 0.5067 | +0.0400 |
| choice | 0.3333 | 0.3750 | 0.4867 | -0.0417 |
| score | 0.2325 | 0.3125 | 0.3400 | -0.0800 |

| pooled | candidate | control |
|---|---|---|
| aurc | 0.5620 | 0.4888 |
| min_decision_score | -29.87 | -21.75 |
| coverage_at_5 | 0.0000 | 0.0135 |

**This is a 3-step run and its scores mean almost nothing about arm 0.** Three steps at batch 16 is
48 questions of gradient. The numbers are consistent with a model that has not yet learned the task:
`coverage_at_5` of exactly zero means it never puts 5% of its decisions in its own top confidence
bin, and `min_decision_score` of -29.87 against a control's -21.75 says the same thing in the
decision-metric form their evaluator uses. A pipeline that were broken would fail differently --
`TypeError` at construction, as the previous attempt did.

What the control column *is* worth: it is an independent measurement of the frozen Qwen3-0.6B on
this benchmark, and it is the number any later arm has to beat before it can be said to have learned
anything. Their v1.0 record tabulates the equivalent control at 0.3775 pooled for a 2B tower; ours is
a different number on a different model and the two are not comparable.

---

## 5. One defect this project shipped and fixed

The first probe died with `TypeError: ArchConfig.__init__() got an unexpected keyword argument
'_xattn_heads'` -- after the 2.28 GB model had loaded. The spec had carried its own notes as
`_`-prefixed keys inside `arch_extra`, and `**dict(cfg["arch_extra"])` splats every one of them into
a dataclass constructor. An underscore does not make a key a comment there.

`scripts/check_arm0_spec.py` now fails on an underscore key inside either extra, with two negative
controls for it. The general lesson is the one the check encodes: **notes belong at the top level of
the spec, where nothing splats them.**

---


```
GPU    1 x NVIDIA GeForce RTX 4070, 12,282 MiB (11.99 GiB)
CPU    Intel Family 6 Model 183, 32 logical
RAM    127.7 GiB total, 99.5 GiB available
Swap   19.0 GiB
```

The reference project ran on an H100 80 GB. Ours fits, at 96% of a 12 GB card.

### "63.9 GB of GPU memory" is not GPU memory

Task Manager's GPU panel shows two numbers and they mean different things:

| | value | what it is |
|---|---|---|
| **Dedicated GPU memory** | **12.28 GB** | real VRAM. What CUDA allocates from, what `nvidia-smi` reports. |
| Shared GPU memory | 63.85 GB | WDDM borrowing system RAM to stand in for VRAM |

**63.85 GB is exactly half of this host's 127.75 GiB of RAM**, which is how the shared figure is
derived, and it is why the number looks large. It is not reachable by CUDA: `torch.cuda.mem_get_info`
reports the dedicated 12 GB, and an allocation that does not fit there fails rather than spilling
into system RAM. A training run sized against "63.9 GB" would OOM on the first optimiser step.

The 99.5 GiB of *actual* free RAM is a real resource, and it is the raw material for CPU offload --
which RSI-Jev does not implement. The `CPUOffload` class this project once had lives in the other
repository, `agent-jev`'s `train.py`, not in the reference code; an earlier note in this project
attributed it to RSI-Jev and was wrong. Offload is therefore an adaptation this project would have
to make and record as one, and at 96% with a flat peak it is not currently needed.

---

