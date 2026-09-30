# RSI-JevRSI: Reproducing the RSI-Jev Self-Improvement Curve on a Qwen3-0.6B Backbone

> **Research objective, restated from zero on 2026-09-30.**
> Supersedes `RTX4070_SelfEvolving_Jev_Research_Proposal.md`, which targeted a different
> objective and is retained only as history (Rule 4). Nothing in it constrains this work.

---

## 0. The objective, in one paragraph

Take the self-improving research loop that produced [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev)
(204 registered arms, four releases, v1.0 → v3.0), run it against a **Qwen3-0.6B** backbone instead
of Qwen3.5-2B, and **measure the curve it produces**. The deliverable is not a model that beats
theirs. It is the curve: how far a self-improving loop carries a small backbone before it
saturates, and whether the arm-by-arm record explains why.

**The hypothesis being tested** is one the smaller backbone makes sharp. A loop that improved a
2B tower across four releases should have more headroom on a 0.6B tower, so the curve should be
**longer** — more arms accepted, a later plateau, and a different set of arms failing. If the curve
turns out to be the same length, that is evidence that the loop's progress is set by the loop
rather than by the backbone, which is a different and more interesting claim than the one we
started with.

---

## 1. What is being reproduced, exactly

Not the models. RSI-Jev's checkpoints were produced on an H100 80 GB with
`torch 2.7.1+cu128 / flash-linear-attention 0.5.2 / Qwen3.5-2B-Base`. This host is an
RTX 4070 (12 GB) with torch 2.10. Their v2.0 recipe alone is 5,774 steps at 2B. **The published
numbers are not reproducible here and no claim will be made that they are.**

What is reproducible, and is the actual object of study:

| Element | Their form | Reproducible here? |
|---|---|---|
| The **loop** — propose, register prediction, run, gate, retire | AutoScientists | **Yes.** This is the object. |
| The **EDITABLE / PROTECTED split** (`arch`/`data`/`train` editable; `contract`/`evaluate`/`metrics`/`targets`/`encode` protected) | `rsijev/README.md` | **Yes**, and it is the single most valuable idea in the repository. |
| The **teacher-distillation corpus** `n4ze3m/typed-decisions-synth`, 6,977 documents with a full probability distribution per question | their v1.0 data | **Yes** — verified downloadable, download started 2026-09-30. |
| The **v1.0 recipe** (1500 steps × batch 16, tower 5e-6 cosine, head 1e-4 constant, options shuffled, fp32 scorer) | `versions/v1.0.md` | **Yes**, at 0.6B. |
| The **v2.1 optimiser change** (`lower_layers_n=8`, `lower_layers_lr_scale=0.1`) | `fit_extra` | **Yes** — and it is one of their few kept arms. |
| The **v3.0 listwise RL** (NDCG@5 over 16 candidates, Plackett-Luce, LOO baseline, KL anchor) | `rsijev/rl2.py` | **Structurally.** Needs a reward; the sources are not published. |
| The **fla kernel stack** | pinned | **No.** Different hardware, and their README warns of wrong gradients on some GPUs. |
| Their **published numbers** | 0.662 / 0.796 / 0.756 | **No.** Not re-derivable on this host. |

**Rule 4 governs every comparison.** A number we produce is ours. A number they published is
theirs, computed on their hardware, their split, their stack. The two may be *placed in the same
table* and must never be averaged, differenced, or described as "an improvement on".

---

## 2. What this project does NOT inherit

Stated explicitly because the previous objective's vocabulary is still in this repository's
history and names do not shape reasoning correctly:

- **No harness-first posture.** The prior proposal made post-hoc calibration the primary route.
  Measured on 2026-09-30, that family is exhausted on this architecture: every per-type and
  per-question map is argmax-invariant because all candidates of one question share its type
  (see `measurement/harness_saturation_2026-09-30.json`). RSI-Jev's own confidence head is
  described the same way — "it never changes which option wins". Both parties to this
  benchmark agree the lever is not there. It is not the objective of this work.
- **No τ threshold on the parameter level.** The revoked threshold (`τ_dev = 4.949 pp`) was
  derived from a floor measured on a different, retired model. It is void and stays void. The
  replacement floor has not been measured yet, and until it is, the inner level is not armed.
- **No split reuse.** RSI-Jev's benchmark train split was used by their v2.0 onward, so training on
  it is *not* a novel arm for us — it is the starting condition. Their "+0.12 from training on the
  benchmark's own train split" cannot be claimed, reproduced or re-earned here. **What is
  re-earnable is everything after that point.**

The useful consequence: our curve starts from a different place, so it is genuinely ours.

---

## 3. The starting point, and why the smaller backbone is the whole point

| | RSI-Jev | This project |
|---|---|---|
| Backbone | Qwen3.5-2B-Base | **Qwen3-0.6B** |
| Trained | all tower weights except frozen input embedding, + 7.3M cross-attention scorer | to be decided; the subject repo's permutation-equivariant set encoder is the candidate |
| Hardware | H100 80 GB | RTX 4070 12 GB |
| v1.0 held-out | 0.662 | **unknown — not yet measured** |

The comparison that will be made is **not** 0.6B vs 2B on a score. It is:

> *Given the same loop, how much further does it carry the smaller backbone, and where does it stop?*

That question is answerable on one GPU and is worth more than the score would be.

---

## 4. Pre-registered predictions

Written **before** any arm is run, so that the curve cannot be narrated into existence afterwards.
Each is falsifiable by a measurement this project can actually make.

- **P1 — Headroom.** The Qwen3-0.6B starting point scores below RSI-Jev's 2B v1.0 (0.662) on the
  same held-out set, under our own measurement code.
- **P2 — Curve length.** The number of arms that clear the gate is larger for 0.6B than the four
  releases RSI-Jev needed for 2B. *Falsified if our plateau arrives in ≤ 4 accepted arms.*
- **P3 — The failing set is different.** Arms that kept for 2B fail for 0.6B, and vice versa.
  A permutation-invariance arm and a "train on the benchmark's train split" arm both have known
  outcomes in RSI-Jev's record; they are not open questions here.
- **P4 — The bf16 scorer bug does not reproduce.** Measured 2026-09-30: `ScalarScorer` opens with
  an `RMSNorm` that casts to fp32 *before* the multiply, so autocast(bf16) leaves a relative error
  of 5.8e-03 and does not change the argmax. Their seven failed experiments were a real failure of
  *their* scorer, not a property of the task.
- **P5 — Position-collapse is the first failure mode to appear**, because the 0.8B-scale towers are
  where a causal encoder most easily collapses onto option *k*. Their record: "one early run picked
  the last option on 800 of 800 score questions". If our curve fails early, this is the most likely
  reason and the first thing to check.

---

## 5. Non-negotiables, carried forward because they were paid for

These are not from the previous objective's *plan*; they are engineering facts learned by hitting
them.

1. **Contamination is mechanical, not judgemental.** `scripts/check_object_purity.py` is the
   authority. A human reading a file and deciding it is clean is exactly how the previous rollback
   happened.
2. **English in every artefact**, including comments. `scripts/check_language.py`.
3. **No absolute paths in code.** `config/paths.json` + environment variables.
4. **A claim without a floor is not a claim.** Every arm carries its own measured noise floor, and
   an unarmed level stays unarmed.
5. **History is immutable.** Incident records, run stamps and broken scripts are not rewritten.
6. **Destructive operations are not delegated.** Migrate file by file, validating with `ast.parse`.
7. **A negative result is a result** and is written down with the measurement that killed it.

---

## 6. What happens first

1. **Measure the starting point.** Qwen3-0.6B + the existing head, on a held-out split this
   project owns. Everything else depends on knowing where the curve begins, and it has never been
   measured — the 0.7955 in this repository is a *dev-derived working set*, and treating it as a
   baseline would place the whole curve on an inflated origin.
2. **Install the EDITABLE / PROTECTED split.** The loop may rewrite the model, the data and the
   training objective. It may not rewrite what a typed decision is, what is scored, or on which
   split. This is the guard rail that makes an arm's result mean anything, and it is the one piece
   of RSI-Jev's engineering we adopt unchanged.
3. **Then, and only then, run arms.** Each arm registers its prediction before GPU time is spent,
   and each is published as a negative unless it clears the gate.

---

## 7. Provenance

- RSI-Jev is MIT licensed; checkpoints follow their base model's licence. Not affiliated with
  TypeSafe AI. We reuse its **ideas, loop structure and public data**, and we do not present
  anything here as their work or as endorsed by them.
- Every number in this repository is labelled with the project that measured it and the hardware
  that measured it.
