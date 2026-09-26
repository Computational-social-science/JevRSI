# nanochat-only design — pre-registration

**Status:** pre-registration. Written 2026-09-26 **before any `program.md` variant has been run**.
**Implements:** the decision argued in §1, and corrections #1–#3 of
`docs/critical_review_2026-09-26.md`.
**Supersedes:** the MCQ/ARC study as the *project*; that study is retained only as the
methodological pilot (construct A) that built this instrument.
**Companion:** `docs/variance_design.md` holds the full variance-components algebra; this
document holds the **design decision and the run plan** and defers to it on estimation.

---

## §1 The decision: nanochat alone carries the research goal

**Question.** Can the research goal be met using only nanochat — no additional models?

**Answer: yes**, with one clarification and one prerequisite.

### 1.1 What "without additional models" can mean

| reading | meaning | verdict |
|---|---|---|
| **(a)** nanochat is the only **object of study** | the MCQ/ARC apparatus is dropped | **adopted** |
| **(b)** the **executor** may not be an LLM either | loop driven by a script or a human | possible; see §3 |
| **(c)** no *new* local models | reuse models already present | satisfied by `qwen2.5:7b`, `gemma2:9b`, `mistral:7b` |

We adopt **(a)**, and hold the executor **fixed** (§3), which makes the study satisfiable under
(a) alone without depending on (c).

### 1.2 Why nanochat alone is not merely sufficient but *better*

| dimension | MCQ/ARC route | **nanochat route** |
|---|---|---|
| construct | A — measurement-instrument method variance | **B — the landing form of an optimization strategy** |
| is B represented? | no; B is the brief's actual object | **`program.md` *is* B, literally materialised as a file** |
| compliance confound | parse rate 1.000 ⇒ FSI ≈ 0 by construction, which is why the null was uninformative | **no parser exists ⇒ PSI ≡ 0 structurally** |
| contamination | must be argued | **impossible — trains from scratch** |
| cost | 2 models × 32 variants × 300 items per run | **measured 140 runs/day, zero marginal cost** |
| power | CI on the spread spanned 5 → 22 points (FLAW 10) | **continuous DV; see §5** |

The decisive argument is construct identity. The brief's complaint is that "CoT /
self-consistency / ReAct / ToT all collapse to one hand-authored static template scored once."
In this repository that is **not a metaphor**:

- the template is `program.md` — a single file, **hand-authored by the human**, never varied;
- it is **scored once** per run, on one number;
- the strategies inside it collapse into that one file.

### 1.3 The category distinction that must be stated explicitly

The **optimization strategy belongs to the agent, not to nanochat.**

- `program.md` = the strategy's **landing form** (the manipulated variable);
- the **agent** executing it = the strategy's **executor** (held fixed, §3);
- nanochat = the **testbed** that scores the outcome;
- `val_bpb@300s` = the **DV**.

Treating nanochat's own training as "the optimization strategy" would be a category error. The
50M from-scratch GPT has no strategy; it is the artifact being optimized.

### 1.4 What the star-model constraint binds

`Problem-2.docx` requires canonical academic models. Under this design the constraint is
satisfied by the **executor's identity**, not by the trained artifact: the agent model is drawn
from the canonical set already local (`qwen2.5:7b`, `gemma2:9b`, `mistral:7b`). The 50M nanochat
GPT is the *task*, not the *subject*. This removes the need for the MCQ apparatus.

### 1.5 What is given up — stated honestly

The MCQ study's only non-redundant asset was the star-model requirement, now satisfied by §1.4.
Everything else it produced was either already covered by 2026 work (construct A, papers A/B) or
underpowered (FLAW 10). It is **demoted to pilot**, and its real output — the instrument, the
statistics, and eleven documented flaws — carries over intact.

---

## §2 The claim under test

> **C-n.** The `val_bpb@300s` produced by the single hand-authored `program.md` is a
> **heuristic sample point**: it is a draw from the distribution over equally-defensible variants
> of that template, and the gap between it and the class distribution (**HPFE**) is larger than
> the instrument's measured noise.

> **C-o (OASP).** Searching the `program.md` space recovers gain that **generalises to held-out
> seeds**, beyond what the single template achieves.

C-n is the primary claim. C-o is a second phase, run only if C-n holds.

---

## §3 Design decisions (frozen)

| id | decision | rationale |
|---|---|---|
| **D1** | **Executor held fixed** (one canonical agent model, one decoding configuration) | HPFE is a property of the *template*. The agent is an instrument, not a factor. This also removes the largest unmeasured variance channel (FLAW 9) and makes the study satisfiable with no agent-model factor at all. |
| **D2** | **`program.md` is the only manipulated variable.** The harness contract (§4) is invariant. | Otherwise a "variant" could change the rules of the game rather than the strategy. |
| **D3** | **The DV is `val_bpb@300s` with `num_steps` as a mandatory covariate.** | Measured: identical code produced 44 and 42 steps, a ~0.0254 bpb gap at a 0.0127 bpb/step slope. A variant that merely changes throughput must not appear to change learning. |
| **D4** | **Declared load regime**: no concurrent agent orchestration on the training host during any measurement block; per-step `dt` and tok/s logged as monitored covariates. | FLAW 11: five parallel subagents produced a 14,972 ms worst step against a 10,544 ms idle-host ceiling, converting lost throughput into two fewer steps. |
| **D5** | **Variants are frozen and hashed before the first run**; outcome-blind authorship declared. | Pre-registration. The spread is the result, not the winner. |
| **D6** | **Held-out seeds** for C-o; the search never sees them. | Otherwise the search's gain is in-sample by construction. |

### 3.1 Prerequisite gate **G1** — the single blocking item

`train.py` hardcodes the seed and exposes **no** `--seed`:

```
train.py:894,895    torch.manual_seed(42) / torch.cuda.manual_seed(42)
train.py:1055,1056  torch.manual_seed(42) / torch.cuda.manual_seed(42)
train.py:1219,1220  argparse exposes only --smoke-test and --dataset
```

With S = 1 there is no seed-generalisation claim of any kind. **A human-controlled `--seed` must
be added outside the agent's editable surface and frozen before the study starts.** This is a
harness change, not an agent change.

---

## §4 The invariant harness contract

Every variant shares these verbatim; they are the *rules of the game*, not the strategy:

1. `prepare.py` is read-only; `evaluate_bpb` is the ground-truth metric.
2. No new dependencies; only `pyproject.toml` is available.
3. The fixed 300 s wall-clock training budget.
4. The `results.tsv` schema (`commit val_bpb memory_gb status description`, tab-separated).
5. The trajectory protocol: **record, then reset** — reason codes
   `WORSE|TIED|CRASH|TIMEOUT|BREAKS_CONSTRAINT|OTHER`, `runs/<run_id>/trajectory/<NNNN>.{diff,json}`,
   written before `git reset --hard`, never `git clean -x`.
6. The `val_bpb:` log line and the summary block format.
7. The `NEVER STOP` autonomy clause.

**Only the strategy block varies** (§6). The core is stored once with a slot marker, so the
difference between any two variants is exactly their strategy block — this is what makes the
manipulation auditable rather than impressionistic.

---

## §5 Measured instrument characteristics (final, n = 6)

From the completed noise-floor block of 2026-09-26 — identical code, commit `a4123c6`,
`TIME_BUDGET=300`, grad-accum 64, seed 42, six runs:

| run | `num_steps` | tokens | `val_bpb@300s` | host | max dt | min tok/s |
|---|---|---|---|---|---|---|
| baseline | 44 | 23.1 M | 0.863803 | idle | 10,544 ms | 49,723 |
| replicate 3 | 44 | 23.1 M | 0.866649 | idle | 10,127 ms | 51,769 |
| replicate 4 | 44 | 23.1 M | 0.866307 | idle | 10,296 ms | 50,919 |
| replicate 5 | **43** | 22.5 M | 0.888926 | loaded | 10,504 ms | 49,911 |
| replicate 1 | 42 | 22.0 M | 0.889241 | subagents | 14,972 ms | 35,017 |
| replicate 2 | 42 | 22.0 M | 0.888302 | subagents | 11,191 ms | 46,848 |

### 5.1 The distribution is BIMODAL, not a slope

The runs split cleanly into two clusters, and `num_steps` separates them with no overlap:

| cluster | steps | n | mean `val_bpb` | within-cluster SD |
|---|---|---|---|---|
| **complete** | 44 | 3 | 0.865586 | 0.001554 |
| **truncated** | 42–43 | 3 | 0.888823 | 0.000478 |

- **between-regime gap: 0.023237 bpb** (~2.7% of baseline)
- **pooled within-regime SD: 0.00115 bpb**
- ratio: **≈20×**

An earlier version of this section reported a **"0.0127 bpb/step slope"** derived from the single
42-vs-44 comparison. **That was wrong.** Replicate 5 lands at 43 steps — one step short — yet
sits in the *truncated* cluster at 0.888926, not between the clusters. The penalty is a
**cliff, not a slope**, and is not proportional to the number of missing steps.

### 5.2 Mechanism: the LR schedule is normalised over the run's own length

Per-step logs show why the cliff is so steep. The learning-rate multiplier at a given *absolute*
step differs between runs of different lengths:

| step | complete (44 steps) | truncated (42–43 steps) |
|---|---|---|
| 30 | 0.840 | 0.750 |
| 35 | 0.540 | 0.430 |
| 40 | 0.230 | 0.120 |
| 42 | 0.110 | 0.000 |

Training losses are **identical to four decimals up to ~step 35** (3.1946 / 3.1959 / 3.1980 in the
complete cluster vs 3.1976 / 3.1962 / 3.1933 in the truncated one) and diverge only afterwards,
ending at ≈2.825 vs ≈2.87–2.90.

So host load does not merely remove tokens: it **compresses the LR schedule**, lowering the
learning rate at every absolute step. The chain is a two-stage mediation —
**load → fewer steps → compressed schedule → less learning** — and that is why a *one-step*
difference costs the full ~0.023 bpb rather than one step's worth of tokens.

### 5.3 What this forces

- **`num_steps` is a valid regime indicator but not a linear covariate.** Flagging
  `num_steps < 44` is correct — it catches 42 *and* 43 — while extrapolating a per-step slope is
  not. The negative estimate for `dbpb/dstep` used in the §7 mediation estimand is therefore
  unreliable and must be estimated from a controlled 42/43/44 sweep, or the mediation dropped.
- **intrinsic noise ≈ 0.0012 bpb pooled.** Anything below that is unmeasurable on this instrument.
- **the load effect ≈ 0.023 bpb ≈ 20× that**, so the regime must be *controlled*, not merely
  recorded. One concurrent agent is enough: replicate 5 was disturbed by nothing more than this
  session's own light work.
- **Equivalence margin: ±0.005 bpb** — ≈4× the pooled within-regime SD, and ≈0.6% of the baseline.
  The earlier ±0.001 is below 1× SD and is unreachable by construction.

---

## §6 The equivalence class: what may vary

Twelve strategy blocks, each a choice a competent researcher could defensibly make either way,
plus the stock template as the reference point. The manipulated dimensions are:

| # | dimension | the defensible either/or |
|---|---|---|
| 1 | **idea-generation regime** | combine near-misses vs. one hypothesis per attempt |
| 2 | **exploration schedule** | breadth-first sweep vs. depth on the current best |
| 3 | **keep-threshold** | any improvement vs. improvement exceeding the noise floor |
| 4 | **simplicity pressure** | strict line-count penalty vs. pure val_bpb |
| 5 | **change granularity** | one isolated change vs. bundled related changes |
| 6 | **prioritisation order** | architecture-first vs. optimizer/schedule-first |
| 7 | **VRAM posture** | conservative headroom vs. use whatever helps |
| 8 | **rollback trigger** | never rewind vs. rewind after N consecutive failures |
| 9 | **planning formalism** | think-in-code vs. written hypothesis before each attempt |
| 10 | **logging verbosity** | terse description vs. reasoning + expected direction |
| 11 | **reference use** | read papers cited in the code vs. purely empirical |
| 12 | **near-miss harvesting** | discard near-misses vs. revisit the best near-miss later |

`v00_stock` is the current `program.md` **verbatim** — the single heuristic sample point the
brief criticises. HPFE is measured against it.

---

## §7 Run plan and budget

### 7.0 CORRECTION — the unit of analysis is the agent session, not a training run

An earlier version of this section budgeted "V × S = 65 runs at 8.6 min/run = 9.3 hours".
**That was wrong**, and the error is recorded rather than quietly fixed because it is the kind
that wastes a week of GPU.

A single `uv run train.py` does not involve `program.md` at all. The template is read by the
**agent**, which then decides what to change and launches **many** training jobs. So:

- the experimental **unit** is one **agent session** — the trajectory of experiments the agent
  produces while following one variant of the template;
- the **DV** is a property of that trajectory (§7.2), not of one job;
- the **seed** factor applies to the session's base seed, which fixes init and data order for
  every job inside it.

Cost therefore scales with sessions, not with cells:

```
per-experiment cost   8.6 min   (measured, idle host, FLAW 11 regime)
session length T      1.0 h     -> ~7 experiments per session
cells                 V x S
wall time             cells x T

V=13, S=1             13 h      ~1 overnight block
V=13, S=3             39 h      ~1.6 days
V=13, S=5             65 h      ~2.7 days
```

A 24/7 local campaign makes S=5 affordable, but it is **three days of continuous GPU**, not one
night. **Staging is therefore mandatory**: run S=1 first to establish the spread and to prove the
runner itself sound, then extend S only for the variants whose spread warrants the extra cost.
A session also costs agent inference on top of training, so T=1 h yields *fewer* than 7
experiments whenever the executor is slower than the GPU.

### 7.1 Cells

```
Factor 1  program.md variant   V = 13   (v00 stock + v01..v12)
Factor 2  session base seed    S       (requires gate G1); staged 1 -> 3 -> 5
Executor  fixed, declared by name (D1)
Design    V x S, one session per cell
```

### 7.2 The DV for a session

Primary: **`best_val_bpb@300s`** — the lowest `val_bpb` reached in the session, with `num_steps`
as covariate (D3), and the improvement over the session's own stage-0 baseline run.

Secondary, and reported because they distinguish *searching* from *guessing*: `n_experiments`,
`n_kept` (the keep rate), the reason-code histogram from the trajectory records (FLAW 8), and
the improvement trajectory itself.

Rationale: the brief asks whether the template determines the **yield of the search**. Yield is
a property of the trajectory, which is why the unit cannot be a single job.

### 7.3 Analysis (deferred to `variance_design.md` for the algebra)

- **HPFE (C-n)**: the between-variant spread of seed-averaged `val_bpb@300s`, with `num_steps`
  as covariate, tested **both** ways — difference (is the spread larger than the independence
  null?) **and equivalence** (is the spread's *upper* bound below the re-derived margin?).
  FLAW 1: a difference test alone can never establish equivalence.
- **Verdict vocabulary is fixed in advance**: `SUPPORTED`, `NOT SUPPORTED`, or
  **`UNDERPOWERED TO DECIDE`**. The third is a legitimate outcome and must never be reported as
  a null (FLAW 10).
- **`v00` position in the class distribution** is the headline: a single template's percentile
  among its equally-defensible peers *is* HPFE.

### 7.2 Falsification

C-n is **refuted** if the seed-averaged spread of `val_bpb@300s` across variants, with `num_steps`
held, is not distinguishable from the intrinsic residual — i.e. the template is *irrelevant*, and
the single sample point is as good as any draw from the class.

---

## §8 Limitations (to be stated in any write-up, not hidden)

1. **One dataset.** `prepare.py:46  DATASET_CHOICES = ("tinystories",)` — a single synthetic,
   simplified corpus. No dataset gradient exists, and adding one would require editing the
   read-only `prepare.py`. Claims are about *this* corpus.
2. **One model family.** Model size is not a CLI knob and `train.py` is inside the agent's edit
   surface, so size is part of the treatment space rather than a controlled factor. There is no
   independent capability gradient. Results are an **existence proof**, not a law.
3. **One executor.** Deliberate (D1), but it means the result is conditional on that executor;
   transfer to another agent is untested and is exactly what FLAW 4 warns about.
4. **One hardware platform.** `val_bpb@300s` is compute-normalised; cross-hardware comparison
   is void.

---

## §9 Provenance

- Variants frozen and SHA-256-hashed in `configs/program_variants/manifest.json` before the first
  run; the manifest is committed.
- Each run records the assembled `program.md` hash, the variant id, the seed, `num_steps`,
  `val_bpb@300s`, and the load regime (D4).
- No LLM judge anywhere in the measurement path.
