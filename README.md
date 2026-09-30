# JevRSI

> **JevRSI** — reproducing the RSI-Jev self-improvement curve on a **Qwen3-0.6B** backbone.
> The deliverable is the curve, not a score.
>
> The objective in full is [`docs/RESEARCH_OBJECTIVE.md`](docs/RESEARCH_OBJECTIVE.md).
> It supersedes `RTX4070_SelfEvolving_Jev_Research_Proposal.md`, which targeted a different
> objective and is retained only as history.

## What this is

[RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) improved a 2B typed-decision model across four
releases with a loop of AI agents that propose hypotheses, register their predictions before spending
GPU time, run the experiments, and retire their own champions when the evidence says to. **204
arms.** It published the arms that failed alongside the ones that worked.

This project runs that same loop against a **smaller backbone** and asks what it does there.

> **The question.** Given the same loop, how far does it carry a 0.6B tower before it saturates, and
> what does the arm-by-arm record explain?

A loop that took four releases to move a 2B model should have more headroom on a 0.6B one, so the
curve should be **longer**. If it turns out to be the same length, that is evidence the loop's
progress is set by the loop rather than by the backbone — a different and more interesting claim
than the one we started with.

## Why this is not a score comparison

| | RSI-Jev | JevRSI |
|---|---|---|
| Backbone | Qwen3.5-2B-Base | **Qwen3-0.6B** |
| Hardware | H100 80 GB | RTX 4070, 12 GB |
| Stack | torch 2.7.1+cu128 / fla 0.5.2 | torch 2.10 |
| v1.0 held-out | 0.662 | not yet measured |

Their published numbers are **not reproducible on this hardware and are not claimed to be.** Every
number is labelled with the project that measured it and the machine that measured it. The two sets
may appear in one table and are never averaged or differenced.

## What is adopted, and what is not

**Adopted unchanged** — the guard rail, the most valuable idea in their repository. Modules are
declared **EDITABLE** or **PROTECTED**. The loop may rewrite the model, the data and the training
objective. It may not rewrite what a typed decision *is*, what is scored, or on which split. A change
to scoring is a change to the task, and goes through a human.

**Adopted** — the teacher-distillation corpus `n4ze3m/typed-decisions-synth`: 6,682 documents,
23,319 questions, each carrying a full teacher probability distribution (`deepseek-v4.1-flash`, mean
of three). Fitting the distribution rather than the label is the substance of their v1.0, and it is
the one asset we did not previously have.

**Adopted** — the v1.0 recipe (1,500 steps × batch 16, tower 5e-6 cosine, head 1e-4 constant,
options shuffled per example) and v2.1's optimiser change (`lower_layers_n=8`,
`lower_layers_lr_scale=0.1`).

**Not adopted** — their kernel stack. `flash-linear-attention` on this GPU is a different code path,
and their own README warns the backward pass is wrong on some architectures.

## The task

A single forward pass returns a probability for every option of a typed question about a document —
yes/no, pick-one-of-k, rate-on-a-rubric — and nothing is generated. One decision about a document
already read costs about 10 ms.

## Pre-registered predictions

Written before any arm runs, so the curve cannot be narrated into existence afterwards. Each is
falsifiable by a measurement this project can make.

- **P1 Headroom.** The Qwen3-0.6B starting point scores below RSI-Jev's 2B v1.0 (0.662) on the same
  held-out set, under our own measurement code.
- **P2 Curve length.** More arms clear the gate than the four releases RSI-Jev needed for 2B.
  *Falsified if the plateau arrives in ≤ 4 accepted arms.*
- **P3 A different failing set.** Arms that kept for 2B fail for 0.6B, and vice versa.
- **P4 The bf16 scorer defect does not reproduce.** Measured: `ScalarScorer` opens with an `RMSNorm`
  that casts to fp32 *before* the multiply, so autocast(bf16) leaves a relative error of 5.8e-03 and
  does not change the argmax. Their seven failed experiments were a real failure of *their* scorer,
  not a property of the task.
- **P5 Position-collapse is the first failure mode to appear**, because smaller towers are where a
  causal encoder most easily collapses onto option *k*. Their record: "one early run picked the last
  option on 800 of 800 score questions."

## Already measured

**Post-hoc calibration cannot move the decision on this architecture, and that is structural.** All
candidates of one question share its type, so a per-type map is a single monotone transform over that
question's candidates and cannot reorder them. Five stages spanning a nine-fold range in calibration
error (ECE 0.3715 → 0.0408) return one identical accuracy, 0.7955. The only family that moves the
argmax is a per-position prior, and a 61-point scan of it tops out at **+1.14 pp** against a
pre-registered threshold of **10.685 pp**.

RSI-Jev's own confidence head is described the same way — it rescales without changing which option
wins. Two parties to this benchmark agree the lever is absent. That is why this project is about the
parameter loop and not the harness.

Numbers in `measurement/harness_saturation_2026-09-30.json`, on a **dev-derived working split** — not
a held-out set.

## Layout

```
autoresearch/
├── docs/RESEARCH_OBJECTIVE.md                        the objective, the predictions, what is not inherited
├── adaptation/                                       gates and adapters, no controller
│   ├── _ecosystem.py                                 loads pure helpers from agent-jev by path
│   └── gate_tau.py                                   refuses to arm a revoked threshold
├── config/paths.json                                 every machine-specific location, declared once
├── harness/                                          reference implementation of the calibration stages
├── measurement/                                      floors, thresholds, feature cache, negative results
├── scripts/                                          the six guards
├── docs/                                             decisions, pre-registration, rules, terminology
└── archive/                                          retired and superseded material, with manifests
```

## Gates — all six must pass before a cycle spends GPU

```bash
python scripts/check_object_purity.py     # 15 retired objects stay absent
python scripts/check_language.py          # English is the language of every result
python scripts/check_terminology.py       # the lineage stays correctly attributed
python scripts/check_relative_paths.py    # no machine paths, AND derived roots land correctly
python scripts/check_split_disjoint.py    # fit / filter / decision sets are three different things
python adaptation/gate_tau.py             # refuses to arm a revoked or unregistered threshold
```

Each prints its exemptions in force on every run, so an exemption list cannot grow unnoticed.

## Rules

The full set is in [`docs/PROJECT_RULES.md`](docs/PROJECT_RULES.md) and is binding. Four worth
restating:

1. **A measurement is not a claim until its noise is measured.** A borrowed floor may plan
   *provisionally* if labelled at the point of use; it may never appear in an accept threshold or a
   comparison table un-measured.
2. **State the df beside every SD.** A pilot once overstated the run-to-run floor by 3.53x because it
   was measured at df = 1.
3. **Destructive operations are not delegated.** Files are moved, never deleted, when reclassifying
   work -- and any agent given a reclassification task gets an explicit target and a verification
   step.
4. **The agent does not retire the owner's objective.** An audit may report that a cited figure has
   no source; whether a stated objective is still pursued is the owner's decision.

## Retired

Fifteen research objects are retired and asserted absent by `check_object_purity.py`. They are
listed with reasons in `CURRENT_OBJECT.md`. Do not work on, extend, or cite any of them as live.
