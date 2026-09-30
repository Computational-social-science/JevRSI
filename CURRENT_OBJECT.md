# CURRENT OBJECT — read this first

> **The objective is [`docs/RESEARCH_OBJECTIVE.md`](docs/RESEARCH_OBJECTIVE.md).**
> Nothing in this repository outranks it. If any file disagrees, that file is either retired or
> superseded.

**Last revised 2026-09-30.** Prior revisions are preserved unmodified under `archive/`.

> **CHANGED 2026-09-30 — the objective is restated from zero.**
> The project is now **JevRSI**: reproduce the
> [RSI-Jev](https://github.com/Shanghua-Gao/RSI-Jev) self-improvement loop against a
> **Qwen3-0.6B** backbone and deliver the curve, not a score.
> `RTX4070_SelfEvolving_Jev_Research_Proposal.md` targeted harness-first optimisation of a retired
> model; it is superseded and kept only as history.
>
> **CHANGED 2026-09-30 — measured, and it is why the objective moved.**
> Every per-type and per-question calibration family is argmax-invariant on this architecture: all
> candidates of one question share its type, so a per-type map is one monotone transform over that
> question's candidates. Five stages spanning ECE 0.3715 → 0.0408 return one identical accuracy,
> 0.7955. The only family that moves the argmax, a per-position prior, tops out at **+1.14 pp** over
> a 61-point scan against a pre-registered threshold of **10.685 pp**.
> RSI-Jev's own confidence head is described the same way. The lever is absent, so it is not the
> objective. Evidence: `measurement/harness_saturation_2026-09-30.json`.
>
> **CHANGED 2026-09-30 — the controller is the ecosystem's, not ours.**
> We drive `agent-jev/scripts/autoresearch_agent.py` (479 lines: pre-registered τ, dev/shadow
> two-signal scheme, crash classification with a circuit breaker, trajectory persistence). Our own
> `pipeline/` and `loop/` orchestrators are **retired** — `archive/quarantine_2026-09-30_skeleton/`.
> Our layer is now `adaptation/`, which contains checks and adapters, never a second controller.
> See "How to tell you are off-target" below for what this rules out.

---

## What we are working on

A **six-week, single-RTX-4070, zero-external-API** self-evolving optimisation framework for the Jev
decision model, governed by six scientific rules: pre-registered held-out criterion, fixed resource
envelope, end-to-end reproducibility, honest placement against concurrent work, first-class negative
results, and dual independent review before any success claim.

| | |
|---|---|
| **Objective** | `RTX4070_SelfEvolving_Jev_Research_Proposal.md` — the six rules and the three phases are binding |
| **Seed model** | AgentJev-0.6B = Qwen3-0.6B backbone (**no LM head**) + permutation-equivariant candidate head |
| **Subject repository** | `E:/2026-AI4S/agent-jev` — clone of `github.com/malevrigns/agent-jev`, `main @ a965ca8f` |
| **Benchmark** | `LocalLLaMA/typed-decisions` — official test split 400 cases / 2000 questions |
| **Primitives** | Boolean · Choice · Score. Realised `k` is `[2]` / `[4,5]` / `[4,5]` — **measured**; do not quote the API's 2–255 range |
| **Loop harness** | `jsegov/autoresearch-win-rtx`, a Windows fork of karpathy/autoresearch. Borrow its **ideas and iteration framework only**; do not reproduce its work and do not retrain nanochat |

**Hardware reality, measured on this host** — not copied from the proposal. The proposal budgets QLoRA
at ≤ 5 GB and estimates ~5 min per training run. Measured here: **5.33 GiB steady state** for BF16 +
LoRA r16 with gradient checkpointing, at **12.6 s/step**, so a 600-step run is **2.11 h, not 5 min**.
The proposal's 4-bit QLoRA path has never been run on this host; `measurement/lora_feasibility.json`
and the prereg record what was actually measured.

---

## Read these next, in this order

1. `RTX4070_SelfEvolving_Jev_Research_Proposal.md` — the objective
2. `docs/prereg_floor_b_lora_2026-09-29.md` — the only pre-registration written under this objective
3. `docs/DECISIONS.md` — what was decided, and why
4. `docs/DAY1_BASELINE_2026-09-29.md` — the seed's measured starting point

## Current experiment

**Floor B on the LoRA seed** — the seed-to-seed dispersion of the final dev score under one fixed
recipe. It is the quantity that sets ε, and the 0.40 pp previously in circulation was measured on a
**different substrate**; `measurement/INSTRUMENT_CALIBRATION.json` forbids inheriting it silently.

- Pre-registration: `docs/prereg_floor_b_lora_2026-09-29.md` (frozen before any replicate existed)
- Driver: `measurement/run_floor_b_lora.py` — resumable with bit-exact resume, kills its child tree on interrupt
- Artifacts: `measurement/floor_b_lora_runs/`, `measurement/floor_b_lora.json`
- **No Floor B number exists yet.** Until one does, ε remains provisional and no accept rule is armed.

## Global project rules — `docs/PROJECT_RULES.md` is binding

**Six rules, recorded 2026-09-29.** They are enforced by mechanical checks, because a rule that is only
written down is a rule that some future session eventually breaks without noticing.

1. **Nature-level scientific standards for every document.** Every number traceable to an artifact;
   provenance explicit; the substrate named; a claimed absence stated as an absence; negative results
   first-class; uncertainty always attached; re-running must reproduce.
2. **English is the language of every result** — manuscripts, comments, docstrings, commit messages, and
   the prose fields of JSON. The only permitted non-English is a *quoted foreign-language term that is
   functionally load-bearing* (the live case: the purity guard matches two Chinese terms because a source
   document used them, and deleting the Chinese form would blind it to the vocabulary it exists to catch).
   `archive/` and binary sources are exempt — history stays as written, and a source is not a result.
3. **Vocabulary is checked, not assumed** — see `docs/TERMINOLOGY.md`.
4. **Destructive operations are not delegated.** Files are moved, never deleted, when reclassifying work,
   and any subagent given a reclassification task receives an explicit move target and a verification
   step. Three agents asked to quarantine retired work deleted roughly 92 files instead, and because
   none had been committed, nothing was recoverable. A clean tree looks exactly like a correct one.
5. **A measurement is not a claim until its noise is measured.** A borrowed Floor B, a nominal runtime or
   a spot-checked timing may PROVISIONALLY plan and must be labelled at the point of use; it may not
   appear in a success criterion, an accept threshold, or a comparison table unre-measured.
6. **A run that stops cleanly is a success** — stopped, abandoned and "no gain" are written up as results.

### The gate — all three must pass before a cycle spends GPU

```bash
python scripts/check_object_purity.py   # retired objects stay absent
python scripts/check_terminology.py      # the lineage stays correctly attributed
python scripts/check_language.py        # English is the language of every result
```

Each prints its exemptions in force on every run, so an exemption list cannot grow unnoticed. The
language guard is deliberately awkward to satisfy: a genuine citation is added to `EXEMPTIONS` in
`scripts/check_language.py` with a written reason, never by widening a skip list.

## Vocabulary is not free — read `docs/TERMINOLOGY.md` before naming anything

**"autoresearch" means [karpathy/autoresearch](https://github.com/karpathy/autoresearch).** Our
Windows-local instrument is `jsegov/autoresearch-win-rtx`, a fork of it. We borrow its ideas and
iteration framework; we do not reproduce its work, and retraining that harness's own substrate is not
our goal.

**`bilevel autoresearch` is not a thing, and is banned.** An agent coined it on 2026-09-29 while
naming the pipeline skeleton, welding together two unrelated concepts: a real borrowed repository
(karpathy/autoresearch) and bilevel optimization, a technique this project does not use. It has been
removed from every file. Our two-level schedule is called **the outer level / the inner level**, which
is a structural description and implies no lineage from the harness.

`python scripts/check_terminology.py` fails on the compound and on a few adjacent misattributions
(claiming the fork as ours, claiming we forked it, making that substrate's retraining a stated goal). It exists
because the failure is silent — a wrong name reads naturally, nothing errors, and the next session
inherits it as vocabulary.

Two guards, both must pass before a cycle spends GPU:

```bash
python scripts/check_object_purity.py   # retired objects stay absent
python scripts/check_terminology.py      # the lineage stays correctly attributed
```

## What we took from the harness — and what we did not

Taken: **one mutable surface per cycle** (the harness lets the agent touch only `train.py`; ours is
`pipeline/routes.py`, where every candidate must name a declared route so its cost, axis and gate
status are known before it runs); **an append-only log that rebuilds all control state**; **a cool-down
after consecutive failures of one module family** plus a failure log the selector must consult;
**persist the attempt before the reset** (the rejection protocol, which this project has not yet
exercised).

Not taken: **the 5-minute number.** The principle — a fixed budget makes experiments mutually
comparable — is kept, but 5 minutes is not this project's budget. Measured here: a harness cycle is
**~10 ms** and a 600-step parameter cycle is **2.11 h**. That 760,000× ratio is why the harness is the
primary route and the parameter route sits behind a measured saturation gate rather than a schedule.
Also not taken: the harness's own substrate, its perplexity benchmark and its synthetic-corpus training
set, and the removed
Linux/H100 path.

---

## Two things this page previously got wrong

**Correction 1 — "harness" was misread as prompt engineering.** An earlier revision of this page
stated that harness methods were "structurally impossible here" because AgentJev has no LM head. That
test was correct but the question was wrong: it tested the *prompting* space, not the harness the
objective defines. The proposal's §0.3 defines the harness for a scoring model as the layer around
the frozen backbone — **calibration, scoring pipeline, inference backend** — and none of that needs a
prompt. Scalar temperature scaling is argmax-invariant; that fact kills *one* calibration family, not
the calibration axis. Vector temperature, isotonic/Platt, per-(model, question) transforms and
permutation-robustness transforms are not scalar monotone maps and are live levers.

**Correction 2 — fitness is a four-axis composite, and only one axis was being worked.** The
composite is Intelligence, Calibration, Speed, Cost. The whole of 2026-09-29 was spent on
Intelligence (accuracy, chance-corrected skill) while Calibration, Speed and Cost went unmeasured. The
seed's own numbers show the cost of that: **the shipped L1 calibration made ECE worse, 0.1368 →
0.1481.** The Calibration axis — which the proposal names as a central driver for Jev-class models —
was being made worse by the one calibration artefact in the repository, and nobody noticed because
nobody was looking at that axis.

**Correction 3 — the most expensive route had become the default.** A 10.5 h LoRA Floor B batch was
running as the primary experiment. The objective makes parameter updates a *gated secondary* lever,
invoked only after harness families plateau. The schedule was inverted.

---

## The four axes and where the measured headroom is

Everything below is measured on this host, on the seed, dev split, frozen evaluator.

| axis | metric | seed (raw) | shipped L1 | headroom |
|---|---|---|---|---|
| Intelligence | accuracy | 0.8150 | 0.8150 | **not reachable by scalar temperature** — argmax is invariant |
| Intelligence | chance-corrected | 0.7326 | 0.7326 | same |
| **Calibration** | **ECE** | **0.1368** | **0.1481** | **large, and currently regressing** |
| Calibration | soft CE | 0.8286 | 0.8243 | near-exhausted (oracle T within 0.3 of shipped) |
| Calibration | Brier | 0.0491 | 0.0476 | near-exhausted |
| Calibration | score MAE | 0.1862 | 0.1849 | barely moved by a scalar T |
| Speed | s/step | 12.6 clean | — | **~10x observed from backend effects alone** |
| Cost | W | unmeasured | — | never instrumented |

Two measured facts make the harness-first posture concrete rather than doctrinal:

1. **87.4% of errors sit within margin 1.0** of the decision boundary (wrong-item median margin 0.38
   against 1.06 for correct items). These are near-ties, reachable by *any* function of the features —
   which is what vector temperature and per-question transforms are.
2. **A 2.37M head refit from scratch on cached frozen features lands 1.50 pp from the shipped head, in
   16 seconds** (`measurement/head_offline.json`). The decision behaviour lives in 0.40% of the
   parameters. A 596M backbone update is a very expensive way to move something a small module already
   mostly determines.

And the ε problem dissolves on this route: a deterministic harness has **no training variance**, so
Floor B is 0 by construction for the dominant route, and ε reduces to a paired bootstrap on the dev
split. The 12.38 pp provisional ε — which depends on a 0.40 pp Floor B borrowed from a *different
substrate* — stops being load-bearing.

---

## Current experiment

**None running.** The LoRA Floor B batch was stopped: it measures the dispersion of a route the
objective gates behind harness saturation, and the schedule was inverted. Its snapshots
(`measurement/floor_b_lora_runs/`) make it resumable at 25-step granularity if the P1 gate is ever
opened, and the pre-registration `docs/prereg_floor_b_lora_2026-09-29.md` stays frozen either way.

**Next: instrument the four axes**, because three of them have never been measured and MAP-Elites
cannot bin on an axis that has no number. Then the first P0 harness mutation batch.

---

## What is retired — do not work on these

Archived under `archive/`, unmodified. None of it is the current object. If you are reading,
extending, or citing any of it as live, stop.

| id | what it was | why it is not the object |
|---|---|---|
| **R1** | text-prompt optimisation | a scoring model has no LM head and is never prompted |
| **R2** | collapse under repeated self-generated data | a data-policy topic on a retired substrate, not training-code search |
| **R3** | extremal-construction topic selection | a wrong turn: an example in a brief, adopted as the topic |
| **R4** | a candidate downstream benchmark | superseded by `typed-decisions` |
| **R5** | a second 322M substrate used as a low-start arm | **a different model on a different training path.** Its floors are void — see the revocation below |
| **R6** | a separate project on a generative task's failure mode | a different project; nothing transfers to a non-generative decision model |
| **R7** | the project's earlier name | retired; the guard asserts its absence |
| **R8–R14** | see `scripts/check_object_purity.py` | superseded topics, each with a written reason at the point of enforcement |

**The authoritative list is `RETIRED_CONCEPTS` in `scripts/check_object_purity.py`**, not this table.
Keeping the vocabulary out of the prose is deliberate: a document that spells out a retired concept
can be misread as introducing it, and the guard already carries the patterns. This table records
*categories*; the guard records *strings*.
### ⛔ The borrowed run-to-run floor is REVOKED — not merely "do not inherit"

**This stopped being a warning on 2026-09-30. It became a revocation.**

`Floor B = 0.40 pp` was measured on a **retired 322M full-parameter arm, 5 seeds** — replicate
accuracies 0.526–0.536. The live seed is **AgentJev-0.6B, 596M, QLoRA**, dev baseline **0.7955**.
Different model, different scale, different training path.

`docs/prereg_floor_b_lora_2026-09-29.md` already forbade it:

> Re-derive Floor B on its OWN seed … **Do not inherit it silently.** The accept rule **cannot be
> armed** until this seed's own Floor B exists.

**It was armed anyway.** `τ_dev = 4.949 pp` was written into `measurement/noise_floor.json`, verified
by `adaptation/gate_tau.py`, and would have driven every keep/discard decision.

| Floor | Value | Status |
|---|---|---|
| **A** — measurement sampling | 1.0668 pp | **Valid.** A property of the benchmark (case-level cluster bootstrap over 400 cases), independent of which model is trained. |
| **B** — run-to-run | 0.40 pp | **VOID.** A property of a retired substrate's training path. |
| **τ_dev** | 4.949 pp | **NOT ARMED.** |

`measurement/INSTRUMENT_CALIBRATION.json` carries `_REVOKED_2026-09-30` with the full reasoning, and
`adaptation/gate_tau.py` **refuses to arm** while that key is present. The scripts and artefacts of that retired arm
were **deleted**, not retained as evidence.

**The transferable finding survives the revocation**, because it is about method, not numbers: a naive
accept-if-improved rule fires on ~50–60% of pure-noise comparisons, where the τ rule fired on 0/10.
The discipline transfers; the number does not.

> **The lesson, recorded because it generalises.** `gate_tau.py` originally checked that the
> threshold *agreed with* `INSTRUMENT_CALIBRATION.json` — and reported PASS, because it did agree,
> exactly. **Agreement is not validity.** A check that compares a number to a file cannot tell whether
> the file should exist. The gate now checks revocation *before* comparison.

---

## How to tell you are off-target

- you are working on any **R1–R14** topic above, or on the separate generative-task project
- you are editing or referencing a text **prompt** or **prompt template**
- you are treating the borrowed harness's substrate as the object of study rather than as an instrument
- you are quoting **Floor B = 0.40 pp** or **τ = 3.11 / 4.949 pp** as this seed's numbers
- you are citing a **retired 322M arm's** accuracy as an AgentJev result
- **you are writing a new orchestrator, selector, or cycle loop.** The ecosystem's
  `scripts/autoresearch_agent.py` is the controller. `adaptation/` holds checks and adapters; a
  second controller is how the two drift, and the drift is invisible until a number disagrees
- you are **copying a function out of the ecosystem instead of loading it** — see
  `adaptation/_ecosystem.py`, which exists so there is exactly one `load_tau`
- you are editing anything under `archive/`
- you are looking for a file that is now at `archive/...` and not finding it live

### Mechanical check

All five gates must pass before a cycle spends GPU:

```bash
python scripts/check_object_purity.py     # 15 retired objects stay absent
python scripts/check_language.py           # English is the language of every result
python scripts/check_terminology.py        # the lineage stays correctly attributed
python scripts/check_relative_paths.py     # no machine paths, AND derived roots land correctly
python scripts/check_split_disjoint.py     # fit / filter / decision sets are three different things
python adaptation/gate_tau.py              # refuses to arm a revoked or unregistered threshold
```

Each prints its exemptions in force on every run, so an exemption list cannot grow unnoticed.

> **A guard proves absence, not correctness.** `check_relative_paths.py` verified that no literal
> path remained while **twelve files carried a `parents[2]` index** — pointing at the project root's
> *parent*. They were syntactically valid, imported cleanly, and would have read every artefact from
> a neighbouring directory. Removing a literal is half of Rule 1; the other half is that the derived
> replacement lands where it should. Hence the depth check inside that guard.

---

## Brief

`project-2.docx` (md5 `ababd76a34adc2376bcb2d81da390c9c`, 24,679 B) is the JevRSI re-location. The
`Problem*.docx` files belong to the previous object; `Problem.docx` at the repo root is byte-identical
to `Problem-4.docx` and is superseded.