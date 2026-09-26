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

## §5 Measured instrument characteristics

From the noise-floor block of 2026-09-26 (identical code, commit `a4123c6`,
`TIME_BUDGET=300`, grad-accum 64, seed 42):

| run | `num_steps` | `val_bpb@300s` | host | max dt | min tok/s |
|---|---|---|---|---|---|
| baseline | 44 | 0.863803 | idle | 10,544 ms | 49,723 |
| replicate 3 | 44 | 0.866649 | idle | — | — |
| replicate 1 | 42 | 0.889241 | subagents | 14,972 ms | 35,017 |
| replicate 2 | 42 | 0.888302 | subagents | 11,191 ms | 46,848 |

Two clean runs at 44 steps differ by **0.002846 bpb**. The 44 → 42 step gap is **≈0.0254 bpb**,
i.e. **≈9× larger**. Therefore:

- **intrinsic residual** σ_resid ≈ 0.002 bpb at fixed step count;
- **load-induced variation** enters only through `num_steps`, at **0.0127 bpb/step**;
- an effect is only meaningful if it survives **both** the residual and the covariate.

The equivalence margin is therefore **not ±0.001** (that number, from `variance_design.md`, is
unreachable and predates this measurement). It must be re-derived from the full replicate set
before any variant is run, using the *upper* bound of σ (a 5-replicate σ has a 95% CI spanning
[0.599, 2.874]× the point estimate — a 23× span — so sizing on the point estimate is forbidden).

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

```
Factor 1  program.md variant   V = 13   (v00 stock + v01..v12)
Factor 2  training seed        S = 5    (requires G1)
Executor  fixed (D1)
Design    V x S crossed, R = 1
DV        val_bpb@300s  +  num_steps covariate
Runs      13 x 5 = 65
```

At the measured **8.6 min/run** and **140 runs/day** (FLAW-corrected, idle-host regime), 65 runs ≈
**9.3 hours** — comfortably inside one unattended block, leaving ~75 runs of margin for the C-o
search phase and for replicates.

### 7.1 Analysis (deferred to `variance_design.md` for the algebra)

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
