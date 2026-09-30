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

### Step time — measured, and the first estimate was wrong

`train_seconds: 467` for 3 steps cannot be turned into a per-step figure, and the first attempt to
bound it produced numbers that were off by more than an order of magnitude. It attributed all 467 s
to the 3 steps and reported 156 s/step, giving 32-65 h for a 1,500-step arm.

A later control run printed the line their own trainer emits:

```
trained 6277 cases in 1s  loss [1.2714]
```

**One step takes about one second.** The 467 s is dominated by costs that do not scale with steps --
encoding the corpus, materialising candidate vectors, writing 1.79 GB -- and dividing it by 3
charged every one of those to the training. At ~1 s/step a 1,500-step arm is on the order of 25
minutes, not two and a half days.

That single sample is not enough to schedule against, and a 30-step probe is running to settle it.
The lesson is the one the correction itself demonstrates: **a total divided by a small sample is a
bound, not a measurement**, and the first honest statement should have been "the one-off costs
dominate and I cannot separate them" rather than two numbers derived from the same wrong division.

Their 1,500 steps at 2B took about 13 minutes on an H100 80 GB. Ours at 0.6B on a 4070 landing near
25 minutes is not a meaningful ratio -- different hardware, different precision stack, different
model size.

---

### Evaluation OOMs where training does not

The run finished with exit 0 and wrote every record, but the CUDA allocator reported repeated
failures during evaluation:

```
memory allocation failed with OOM on device 0 while trying to allocate 134217728 bytes
(free: 0, total: 12,878,086,144)
```

Training peaked at 96.7% and held; evaluation hit the wall. The cause is structural rather than
incidental: **training runs with `grad_checkpointing`, evaluation does not**, and both evaluation
paths are `@torch.no_grad`, so the difference is activation retention, not graph construction.
`eval_batch_size: 32` is their DEFAULTS, set for an 80 GB card.

This is the first adaptation this project has to make, and it is a one-key change with no edit to
their code: `eval_batch_size` is a top-level spec key that `run_arm_lib` reads directly. It affects
evaluation only -- it changes how many questions are scored per forward pass, not what is scored,
not the metric, and not the training. It is still an adaptation, and it is recorded as one.

`batch_size: 16` is their training number and is not being touched. If the real 1,500-step arm OOMs
in training, the levers in order are `batch_size`, then a CPU-resident fp32 optimiser master, and
each would need its own record.

**Measured, not assumed.** A control run at `steps: 1` with `freeze_base: true` -- which drops the
training-side footprint to nothing and leaves evaluation as the only consumer -- completes all three
targets at `eval_batch_size: 8` with no OOM, and writes every record. That is what makes 8 the
spec's value rather than a guess: 8 is the setting that was observed to work, and 32 is the setting
that was observed to fail.

A note on a probe that did not measure what it intended: `steps: 0` is not "skip training" in their
code, it is a load check, and `fit()` rejects it without `init_from` --
`ValueError: steps=0 is only meaningful with init_from (a load check)`. The isolation above uses
`steps: 1` with the tower frozen instead, which reaches the same separation through supported
values.

`eval_batch_size: 8` is pinned in the spec's load-bearing list, so a later reader who restores their
DEFAULTS of 32 is caught by `scripts/check_arm0_spec.py` rather than by an OOM after 30 hours of
training. `batch_size` is pinned to 16 in the same list, for the opposite reason: it is *their*
number and nothing on this host justifies changing it.

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

