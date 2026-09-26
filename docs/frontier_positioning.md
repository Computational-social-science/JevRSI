# Frontier positioning — what is already published, and what is still open

**Date:** 2026-09-26 · **Trigger:** Problem-2.docx rev (2026-only benchmarks, canonical
academic models, frontier-connected)

This document exists because a literature check found that **most of the original framing is
already published in 2026, and two 2026 papers have already run a larger version of our exact
experiment.** That is not a reason to stop; it is a reason to re-aim. Everything below is
grounded in papers retrieved via the arXiv MCP on 2026-09-26.

> **Scope note (added 2026-09-26; FLAW 2 of [`critical_review_2026-09-26.md`](critical_review_2026-09-26.md)).**
> The MCQ/ARC study positioned below is the **pilot**, and it measures **construct A**
> (measurement-instrument method variance: reproducibility of a reported score under
> evaluator-authored surface variation). It is **not** the project's object. The project's object
> is **construct B** (the landing form of an optimization strategy: whether the researcher's
> hand-authored template determines the result), carried by `nanochat`'s `program.md`, which is
> the primary object of study. Construct B has not been tested yet. Nothing below is deleted;
> claims about "the project" are to be read as claims about the pilot.
>
> **DV note (added 2026-09-26; FLAW 6).** The construct-B dependent variable is written
> **`val_bpb@300s`** — validation bits per byte read at a **fixed 300-second wall-clock training
> budget**, hence compute-normalised. Cross-hardware comparisons and comparisons against
> fixed-step runs are **void**.

---

## 1. The four papers that occupy our intended ground

| # | paper | what it already did | what it takes from us |
|---|---|---|---|
| **A** | **2608.21382** *There Is No Neutral Harness: Modern LLM Leaderboards Are Manufactured by Config-Fragile Items* (2026-07-17) | **The fragility grid**: 12 open-weight instruct models × 4 families × **3,679 items** × 4 benchmarks (ARC, HellaSwag, MMLU, TruthfulQA) × **26 equally defensible harness configurations**, one correctness bit per (model, item, config). gemma4-31b spans **31–89%** on harness alone. Config-fragile items carry **95.7%** of a pairwise gap. **4 of 12 models reach rank 1** under some config. | **the entire design** — this is our crossed model × item × variant grid, at 10× scale, with the harness as an explicit independent variable |
| **B** | **2607.09665** *Format Sensitivity Index* (2026-05-02) | Defines **FSI** = accuracy range induced by wrapper choice, plus **PSI** = the same range in *parseability*. 140k generations, 7 QA tasks, 5 wrapper families, 4 models 7B–72B. Reports FSI varies **>30×** across models and is **largely explained by compliance failures**. | **our M1 metric** (we called it "range"; it already has a name and a paper), and the compliance control we lacked |
| **C** | **2603.13285** *Brittlebench* (2026-02-27, Meta/EPFL — Romanou, Ibrahim, Sinha, Williams et al.) | A **framework to disentangle data-induced difficulty from prompt-related variability**; semantics-preserving perturbations; degradation up to **12%**; one perturbation flips rankings in **63%** of cases; perturbations explain **up to half** of performance variance. | **our H1 framing**, the variance decomposition, and a released evaluation pipeline |
| **D** | **2605.08522** *Coordinates of Capability: A Unified MTMM-Geometric Framework* (2026-05-08) | SoK formalizing **nine** instability metrics (Paraphrase Instability, Drift Score, Overton Width, Pluralism Score, …) and factorizing behaviour into three orthogonal latent dimensions, explicitly to **separate method variance (prompt sensitivity) from true capability**. | **our entire statistical programme** — the rigorous framework for exactly the confound we were trying to isolate |

Related: **2604.11581** *Hidden Measurement Error in LLM Pipelines* (2026-04) — standard CIs ignore
judge-model/temperature variability (this is our M6 noise floor, already published);
**2509.01790** *Flaw or Artifact? Rethinking Prompt Sensitivity* (2025-09) — asks our question
directly; **2609.03261** and **2609.29445** — measurement-instrument artifacts;
**2609.25352** SSP-Bench — motivated by benchmarks "failing to capture sensitivity to linguistic
variation".

**Conclusion on novelty.** HPFE, as originally framed ("a single hand-chosen prompt is a heuristic
sample point whose reported score differs from the class"), is **not novel in 2026**. Papers A and
C have bigger grids and more models; papers B and D have named the metrics and the framework.

---

## 2. Our pilot result is *explained* by paper B — and that is the opening

`pilot_v2_challenge` (ARC-Challenge, 60 dev items × 32 surface variants, T=0, seed 42), analysed
2026-09-26. Both models now sit **inside** the interior band (introduced following pilot 1,
defined on baseline accuracy alone, applied unchanged to pilot 2 — a **post-pilot-1 criterion,
NOT pre-registered**), so this is the first *valid* H1 test in the project:

| model | acc | FSI (range) | M3 perm p (spread) | η²_prompt | parse rate |
|---|---|---|---|---|---|
| gemma3_4b | 0.7922 | 0.0667 | **0.9510** | 0.0022 (p=0.93) | 1.000 |
| mistral_7b | 0.7427 | **0.1167** | **0.3703** | 0.0029 (p=0.68) | 1.000 |

The observed spread (6.7 / 11.7 pts) is **at or below** the within-item shuffle null (9.4 / 10.3
pts). So across 32 surface-equivalent prompts the accuracy range is **indistinguishable from
item-level Bernoulli noise**.

Paper B supplies the mechanism: **FSI "is largely explained by compliance failures."** Both models
here have **parse rate 1.000** — zero compliance variance. With no compliance variance there is no
FSI to find. Our null is not a failure of the harness; it is a *predicted* consequence of perfect
compliance.

**This is the sharpest testable claim we own:**

> **H1′ — Compliance-mediated fragility.** Harness/prompt-induced accuracy variance (FSI) is
> mediated by output-format compliance (PSI), not by semantic or surface prompt variation. When
> PSI = 0, FSI collapses to item-level sampling noise, and a leaderboard difference computed from a
> single wrapper is *correct* to within noise.

If H1′ holds it is a **boundary condition on papers A–D**: it says when their effect should vanish.
That is a real contribution — a scope limit on a live result is publishable, and cheap to test
because it only needs the PSI axis that we did not vary.

## 3. The axis nobody in 2026 has covered: contamination

**All four papers run on pre-cutoff benchmarks.** Paper A uses ARC (2018), HellaSwag (2019),
MMLU (2020), TruthfulQA (2021). Paper C uses "a suite of popular benchmarks". Papers B and D
likewise. For a 2024–2025 model, every one of those item sets was in pretraining.

That matters for the *mechanism*. If an item's answer was memorized, the model retrieves it
regardless of wrapper wording — so fragility should be **structurally suppressed** on contaminated
items and **revealed** on clean ones. Nobody has tested this. It also means the leaderboard
fragility reported in 2026 may be an artifact of *contaminated* benchmarks, which is the opposite
of the usual contamination concern: contamination can *hide* fragility.

**H1″ — Contamination-masked fragility.** FSI/PSI are larger on post-cutoff items than on
pre-cutoff items of matched difficulty, because memorization pins the answer and suppresses
wrapper sensitivity.

We can test H1″ today, and essentially nobody else can cheaply:

| asset | why it is rare |
|---|---|
| `benchmarks/arc_challenge` — 2018, **pre-cutoff** for every target | SHA-256 frozen, already collected (2,738 cells) |
| `benchmarks/supergpqa` — **2025-02**, expert-written, **post-cutoff** for every target | 26,529 items frozen, chance **measured** at 0.1059 (not the nominal 0.25) |
| `benchmarks/mmlu_pro` — 2024-06, **partial** contamination (built from MMLU items) | 12,032 items frozen, chance 0.1109 (variable-length options) |
| 24/7 local 4B–9B compute at zero marginal cost | the big papers sample a handful of wrappers per item; we can fill the grid |

## 3b. First measurement on the scoring axis — and it partly contradicts paper A

`scripts/probe_scoring_axis.py`, run 2026-09-26 on **mistral:7b-instruct-v0.3**, 16 frozen
ARC-Challenge items, temperature 0. Both read-outs see the *same* prompt wording, so the only
thing that varies is how the answer is read off.

| read-out | items scored | accuracy |
|---|---|---|
| generated-text parsing (what the whole pilot used) | 16/16 | **0.812** |
| per-option log-likelihood argmax | 16/16 | **0.812** |

**Item-level agreement 15/16 = 93.8%.** So on a model with **perfect compliance**, changing the
scoring axis leaves the accuracy *level* **exactly unchanged** (Δ = 0.000) while flipping one item
in sixteen.

This is a **scope limit on paper A's headline** ("the scoring choice … is the load-bearing axis").
It is not a refutation: paper A's models include many with non-trivial compliance failure, and
paper B finds FSI "is largely explained by compliance failures". Our result says the scoring axis
is **contingent on compliance failure** — when parsing is clean, the two read-outs agree, and the
axis is inert. That is consistent with H1′ and it sharpens it:

> H1′ (refined): the load-bearing-ness of the scoring choice is itself mediated by PSI. With
> PSI = 0 the axis is inert; the axis only carries variance once compliance starts to fail.

**Free by-product:** log-probability **margins** are recorded per item (e.g. 7.85 vs 0.95 in the
sample). Low-margin items are candidate fragile items, and the one item where the read-outs
disagreed had a margin of 0.95 — the smallest observed. Margin is therefore a cheap per-item
fragility predictor that the text-parse pipeline cannot produce at all. Testing whether margin
predicts the M3 permutation outcome is the natural next step.

---

## 4. Revised programme

**Central question.** *Is harness fragility a property of the model, or of the item's contamination
status?*

**Pre-registered predictions**

| id | prediction | test |
|---|---|---|
| **H1′** | FSI ≈ 0 when PSI = 0; FSI tracked by PSI when PSI > 0 | vary the **scoring axis** (generated-text parse vs per-option likelihood) and the output-schema axis to induce compliance failure; regress FSI on PSI |
| **H1″** | FSI(post-cutoff) > FSI(pre-cutoff) at matched baseline difficulty | ARC-Challenge vs SuperGPQA, difficulty-matched subsampling |
| **H2** | search over the harness beats the sampled upper tail on held-out items | unchanged — still the load-bearing test |

**Design changes forced by the literature**

1. **Add the scoring axis.** Paper A: *"the scoring choice, not the option order that protocols
   usually fix, is the load-bearing axis."* Our 180-point space varied verb/order/label/format and
   scored one way — generated text. We therefore **never varied the axis that carries the effect**.
   This is the single most important fix.
2. **Adopt FSI and PSI as the metric names** (paper B) rather than a private M1. Report **PSI
   alongside every FSI** — an FSI without PSI is uninterpretable.
3. **Token-controlled wrappers** (paper B). Without length matching, FSI measures verbosity.
4. **Report M3 as a permutation test against the within-item shuffle null**, which we already do —
   and note it is conservative; paper D's MTMM decomposition is the upgrade path.
5. **Item-level records**, as in paper A, so fragility localises to items. We already store
   per-cell rows; the fragility grid falls out of them.
6. **Drop the "new effect name" ambition.** Do not rename what papers B/C/D have already named.

**Target models (Problem-2 rev — canonical academic baselines, all on the Ollama registry, verified
HTTP 200 on 2026-09-26):** `llama3.1:8b-instruct-q4_K_M` (2024-07), `qwen2.5:7b-instruct-q4_K_M`
(2024-09), `gemma2:9b-instruct-q4_K_M` (2024-06), `mistral:7b-instruct-v0.3-q4_K_M` (2024-05),
`olmo2:7b` (2024-11). Small models are **under-studied** in papers A–D (which use 7B–72B and
frontier), and our interior-band screen shows 7B models have ample headroom for fragility.

**On the 2026-benchmark requirement.** Brittlebench (paper C, 2026-02) is the only 2026 *scored
benchmark* for this construct, and its data is **not on the HF mirror** (HTTP 401 under every
candidate repo id). Most 2026 benchmarks retrieved (EHR, protein, audio, forecasting, agentic) are
not MCQ and cannot carry a mechanical harness grid. Rather than adopt a 2026 benchmark whose format
breaks mechanical scoring, we **keep the contamination contrast** — which is a stronger response to
the contamination concern than recency alone, and is the axis the 2026 literature has not covered.

---

## 5. What is genuinely ours

1. **The mediation claim (H1′)** — a boundary condition on papers A–D, cheap to test.
2. **The contamination axis (H1″)** — untested by any 2026 paper, and we already hold the frozen
   pre-/post-cutoff pair with recorded SHA-256 and measured chance levels.
3. **A clean, well-powered negative result** — 32 surface variants, two interior-band models, spread
   indistinguishable from noise (p=0.95, p=0.37). Reported as a negative, which the brief demands.
4. **Full local reproducibility at zero cost** — every number regenerates from frozen JSONL on a
   single RTX 4070, with no API key anywhere in the pipeline.
