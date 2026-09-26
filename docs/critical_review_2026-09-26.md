# Critical review of the research design — 2026-09-26

Requested: re-examine the goals, find the logical flaws, fix them. This document is a
**self-audit**, written adversarially. It supersedes any earlier framing where they conflict.

---

## 0. The goals, restated as claims that could be false

Recovered from `Problem.docx` + `Problem-2.docx`:

| id | claim | status |
|---|---|---|
| **C1** | The object is the **landing form (落地形态)** of optimization strategies, not benchmarks | accepted |
| **C2** | CoT / self-consistency / ReAct / ToT "all collapse to one hand-authored static template scored once" | accepted as a *description* |
| **C3** | A static prompt is therefore a **heuristic sample point**, not a solution | accepted |
| **C4** | The gap between that sample point and the class it represents is measurable and material → **HPFE** | **falsifiable, currently not supported where tested** |
| **C5** | Searching the harness beats the sampled upper tail on held-out data → **OASP** | untested |

---

## FLAW 1 (most severe). The statistical test is the wrong shape for the hypothesis

The decision rule compares the observed spread across variants against a **within-item
shuffle null** and rejects if the observed is *larger*. But C3/C4 predict the variants are
**equivalent**, which means the predicted spread is *smaller* than an independence null —
not equal to it.

The data says exactly that. gemma3_4b, ARC-Challenge: observed spread **0.0833** vs null
spread mean **0.1144**, p = **0.9915**.

Under a correctly-specified null, p should be ≈ 0.5. A p of 0.99 means the observed statistic
sits at the **1st percentile** of the null: the variants agree with each other **far more
tightly than chance relabeling would produce**. That is *evidence for equivalence*, i.e.
against HPFE — and I have been reporting it as an inconclusive null.

**Two errors follow:**

1. **Interpretation error.** "Spread indistinguishable from noise" is wrong for gemma3_4b.
   The correct statement is "spread is *tighter* than independence predicts — consistent with
   the variants being behaviourally near-duplicates."
2. **Test-shape error.** A difference test can only ever *fail to reject* when the truth is
   equivalence. It can never *establish* equivalence. Establishing it requires an
   **equivalence test (TOST)**: reject non-equivalence when the spread's upper confidence
   bound falls below a pre-specified margin. Without it, a genuine finding of "the harness
   does not matter" is structurally unreportable as such.

**Fix.** Report both, and make equivalence the primary claim:
- difference test → "is the spread larger than independence?" (answers HPFE)
- equivalence test → "is the spread smaller than margin δ?" (answers C3/C4)
- δ must be pre-specified from the **noise floor**, not chosen after seeing the data.

**This flaw was in the original protocol and survived four separate NaN-bug fixes because
those fixes all addressed `None`/NaN handling and none re-examined the test's direction.**

## FLAW 2. Two different constructs are being run as if they were one

The brief targets **the optimization strategy's own template** (C2). The MCQ study varies
**evaluation-harness surface features** (verb, option order, label style). Those are not the
same thing:

| | construct | who authors it | what it measures |
|---|---|---|---|
| **A** | measurement-instrument variance | the *evaluator* | reproducibility of a reported score |
| **B** | optimization-strategy landing form | the *researcher* | whether the strategy's template determines its result |

The MCQ work is **A**. Papers B (FSI) and A (fragility grid) own A completely. The brief is
**B**, and B is where `program.md` lives — the README literally calls it the file "the human
edits", to be iterated on to find the "research org code".

**Consequence.** The MCQ study is not the project. It is a *methodological pilot* that built
the instrument, the statistics, and the bug fixes — and produced a null on A that the 2026
literature already predicted (FSI is compliance-mediated; our parse rate was 1.000, so FSI ≈ 0
by construction). **The project's object is B, and B has not been tested yet.**

**Fix.** Re-label the MCQ work as the pilot it is (`pilot_*` run ids already say so — keep
that discipline), and make `program.md` the primary object.

## FLAW 3. The interior-band rule is being described better than it was

`implementation_plan.md` says the selection rule is "fixed here, before the confirmatory run".
True — but it was introduced **after** pilot 1 revealed ceiling compression on ARC-Easy, i.e.
after seeing data from a related measurement. It is a defensible, *baseline-accuracy-only*
criterion applied to a fresh benchmark — but calling it pre-registered overstates its status.

**Fix.** State the provenance explicitly: *"the band was introduced following pilot 1, is
defined on baseline accuracy alone, and was applied unchanged to pilot 2."* Never "pre-registered".

## FLAW 4. "全程免费" (fully free) is violated by the agent, not by training

Local training is free; the GPU is already owned. But executing the `program.md` loop requires
an **agent**, and a frontier API agent is a per-token cost. To hold the free constraint, the
agent must be **local** (~27B class via llama.cpp/Ollama).

That is not merely a budget fix, it is a **design decision**: the agent is a *factor*. A local
27B agent and a frontier agent are different treatments, and `program.md` effects are unlikely
to transfer between them. Report the agent's identity as part of the treatment, never as an
implementation detail.

## FLAW 5. "Academic star models" has an unresolved referent

Problem-2 requires 学术名星模型. In the MCQ design this meant the *evaluated* LLMs
(Llama-3.1-8B, Qwen2.5-7B, …). In the nanochat design the trained artifact is a 50M GPT with no
academic identity, so the constraint must bind elsewhere.

**Proposed resolution (needs your confirmation):** the constraint binds the **agent** — the LLM
whose research behaviour is being studied — because that is the model whose properties enter
the scientific claim. The 50M nanochat GPT is the *task*, not the subject. If instead you want
the star-model constraint on the MCQ/evaluated side, then the two studies have separate model
tables and must not be pooled.

## FLAW 6. The DV is compute-normalised, and that must be stated, not assumed

`val_bpb` at a fixed 300 s wall clock is the honest objective, but it is **not** comparable to
`val_bpb` at fixed steps or on other hardware. A variant that grows the model gets fewer steps;
a variant that shrinks it gets more. Any claim must therefore be phrased as
**"val_bpb@300s"**, and cross-hardware comparison is void.

## FLAW 7 (blocking). N = 1 per configuration, so no comparison is yet falsifiable

`train.py` hardcodes `torch.manual_seed(42)` and exposes **no `--seed` flag**. Consequences:

- Every configuration yields exactly **one** draw.
- I have **no estimate of run-to-run variance**, so "variant A beats variant B" is
  unfalsifiable — the identical defect the protocol's M6 rule exists to prevent.
- The seed is *inside the agent's editable file*, so an agent that touches RNG handling silently
  changes the noise model.

**This is the same lesson already learned on the MCQ side** (M6 noise floor) and not carried
across. A noise floor is now being measured empirically (5 identical runs). If it is non-zero,
replicates exist and the crossing is available. If it is zero (fully deterministic), then the
seed must be **human-controlled and frozen**, outside the agent's edit surface.

## FLAW 8. The reject path destroys the search trajectory

`program.md` instructs `git reset` on rejection, so a rejected hypothesis leaves **only a
`results.tsv` row** — no code, no diff. But the object of study is the *search*, and the
rejected branches are the search's most informative part (they are what distinguishes searching
from guessing).

**Fix.** Before any reset, persist `git diff` + the agent's stated reason to
`runs/<id>/trajectory/`. This is exactly the "blind reason codes" requirement from the brief,
and my own `oasp/git_memory.py` had the same blind spot.

## FLAW 9. Variance is nested, and the design must say which level is which

Three distinct sources of variability exist and are currently confounded:

1. **training seed** (init, data order) — intrinsic
2. **agent sampling** (temperature, prompt realisation) — extrinsic
3. **`program.md` variant** — the treatment

A single run per variant cannot separate them. The design must be **crossed**:
variants × agent replicates × training seeds, with a variance decomposition — the same M3-style
η² decomposition already built for the MCQ side, but with levels correctly nested.

## FLAW 10 (found only by fixing FLAW 1). The study is underpowered, and I was overclaiming

Implementing the equivalence test made the spread's **upper confidence bound** the decision
quantity. It is **0.2167** against a margin of **0.05**.

So on `pilot_v2_challenge`, *all three* statements are simultaneously true:

| test | gemma3_4b | mistral_7b |
|---|---|---|
| difference (spread > independence null) | p = 0.9915, not rejected | p = 0.3703, not rejected |
| equivalence (upper CI < 0.05) | **not established** (upper 0.2167) | **not established** (upper 0.2167) |
| spread > 2× noise floor | yes | yes |

**The design cannot decide.** The 95% CI on the spread spans roughly 5 → 22 accuracy points,
which contains both "practically interchangeable" and "materially different". Reporting this as
"H1 NOT SUPPORTED" — which I did, twice — **overclaims**: it asserts a negative the design
cannot support. The correct statement is **"underpowered to decide"**.

`scripts/power_probe.py` quantifies the shortfall by resampling items:

| n_items | mean spread (gemma3_4b) | CI width | upper ÷ margin |
|---|---|---|---|
| 15 | 0.1920 | 0.1350 | 5.33 |
| 30 | 0.1388 | 0.1000 | 4.00 |
| 45 | 0.1059 | 0.0667 | 2.67 |
| 60 | 0.0833 | (degenerate — n = N) | 1.67 |

Two consequences:

1. **The reported spread is itself noise-inflated.** It falls monotonically with n (0.192 → 0.083
   from 15 → 60 items), tracking ~1/√n. The asymptote is *below* the measured 0.0833, so every
   spread figure quoted so far is an **upper bound on the true spread**. This strengthens the
   prompt-insensitivity reading rather than weakening it.
2. **A concrete item requirement.** To bring the estimated spread to the margin needs
   ≈ 60 × (0.0833/0.05)² ≈ **170 items**; to place the *upper bound* below it, **~250–300**.
   The current 60 is 3–5× short, and no amount of additional variants fixes it — only items do.

**This flaw was invisible before FLAW 1 was fixed.** Under p-value framing the number "0.9915"
reads as "no effect"; only the equivalence framing exposes that the interval is too wide to
say anything. Fixing the test's shape surfaced the power problem immediately.

---

## Corrections to apply now

| # | correction | where |
|---|---|---|
| 1 | add TOST equivalence test; make it the primary claim for the equivalence hypothesis | `oasp/stats.py` |
| 2 | relabel MCQ work as pilot; make `program.md` the primary object | `docs/` |
| 3 | state the interior-band rule's true provenance | `implementation_plan.md` |
| 4 | record the agent identity as a treatment variable; require a local agent | protocol |
| 5 | resolve the star-model referent (needs your decision) | protocol |
| 6 | phrase the DV as val_bpb@300s and forbid cross-hardware claims | protocol |
| 7 | measure the noise floor; if deterministic, add a **human-controlled** `--seed` to the frozen harness | `train.py` (harness change, not agent change) |
| 8 | persist diffs + reason codes before every reset | agent loop |
| 9 | specify the crossed design and the variance decomposition | protocol |

## FLAW 11 (self-inflicted, found by instrumenting the instrument). Ambient load is a confound and my own orchestration produced it

While the 5-replicate noise floor ran, I dispatched 5 parallel subagents on the same host. The
per-step training logs show the damage:

| run | steps | median dt | max dt | median tok/s | min tok/s |
|---|---|---|---|---|---|
| baseline (host idle) | 44 | 9,151 ms | 10,544 ms | 57,293 | 49,723 |
| replicate 1 (subagents active) | 42 | 9,752 ms | 14,972 ms | 53,764 | 35,017 |
| replicate 2 (subagents tailing off) | 42 | 9,290 ms | 11,191 ms | 56,439 | 46,848 |

Replicate 1's worst step took **14,972 ms** against a 10,544 ms ceiling on the idle host, and its
minimum throughput was **35,017 tok/s** against 49,723. Because the budget is a fixed **wall-clock**
300 s, lost throughput converts directly into **fewer optimizer steps** (44 → 42), which converts
into worse `val_bpb` at the measured ~0.0127 bpb/step slope.

**So the first two replicates measure my own CPU load, not the instrument's intrinsic noise.** The
numbers are not wrong, but their interpretation must be split:

- **contaminated** replicates bound the effect of *ambient load* — a real threat for a 24/7
  unattended campaign, which will never enjoy a truly idle host;
- **clean** replicates (3–5, host idle) bound *intrinsic GPU/driver nondeterminism*.

**Fix.** Every confirmatory run, and every noise-floor run, must execute under a **declared load
regime**, with `dt` and tok/s logged as monitored covariates. No concurrent agent orchestration on
the training host during a measurement block. `num_steps` is then the unit of exchange between host
load and `val_bpb`.

This is the same class as FLAW 7 (an uncontrolled channel silently entering the DV) and was caught
only because throughput was instrumented per step. Had only the final `val_bpb` been logged, the
42-vs-44 discrepancy would have looked like ordinary randomness.

---

## What is *not* wrong

- The instrument is sound: frozen SHA-256 stimuli, mechanical scoring, append-only logs,
  resumable runner, no LLM judge anywhere.
- The negative result on ARC is real for what it measures (construct A), and is now
  *explained* rather than merely null.
- The bug-fixing discipline (four NaN-class defects, 48 regression checks, corrected numbers
  stated openly) is the reason these flaws are visible at all.
