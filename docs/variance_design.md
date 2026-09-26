# Variance design — separating training-seed, agent-sampling and `program.md` effects on `val_bpb@300s`

**Status:** pre-registration. Written 2026-09-26 before any confirmatory nanochat run.
**Supersedes:** nothing. **Implements:** `docs/critical_review_2026-09-26.md` FLAW 7, FLAW 9, FLAW 10
(corrections #7 and #9).
**Object of study:** `program.md` — the file the human edits in
`D:/2026-AI4S/nanochat-autoresearch/`, which an autonomous agent iterates on.
**DV:** **`val_bpb@300s`** — validation bits per byte at a fixed **300 s wall-clock training
budget** (FLAW 6: the DV is compute-normalised; cross-hardware and fixed-step comparisons are
**void**).
**Nothing in this document was produced by running a GPU job.** All empirical inputs below
were read from existing logs and configuration files.

---

## §0 The claim under test, stated so it can lose

> **C-v.** A `program.md` variant's effect on `val_bpb@300s` is **real and generalises** —
> it survives re-drawing the agent that executes it and re-drawing the training seed —
> rather than being a single lucky draw.

Three claims are nested inside C-v, and they are **not** the same claim. Each has its own
error term, its own degrees of freedom, and its own run cost:

| id | claim | generalises over | error term that governs it |
|---|---|---|---|
| **C-v·art** | variant *v* beats the baseline **for this agent realisation at this seed** | nothing | `ε` (residual / re-run) |
| **C-v·agent** | variant *v* beats the baseline **averaged over agent realisations** | agent sampling | `a_{r(v)}` — agent replicate, nested in variant |
| **C-v·seed** | variant *v* beats the baseline **averaged over training seeds** | init + data order | `c_{vs}` — variant × seed interaction |

C-v claims the conjunction of all three. A design that establishes C-v·art only has
established that *this artefact* scored better once — which is precisely what the existing
N = 1 workflow already asserts, and precisely what is unfalsifiable.

**Falsification conditions are in §8.** They are the load-bearing part of this document.

---

## §1 The three variance sources, and what is already known about each

### 1.1 Measured cost of one experiment

| quantity | value | source |
|---|---|---|
| training seconds | 301.8 s | `.logs/baseline.log`, run `a4123c6` |
| eval + startup + autotune overhead | 211.9 s | 513.7 − 301.8 |
| **total per experiment** | **513.7 s = 8.5617 min** | `total_seconds:` field |
| throughput | **7.008 runs/hour → 168.2 runs/24 h** | 3600 / 513.7 |
| overhead share | **41.2 %** of wall clock is *not* training | — |
| baseline `val_bpb` | **0.863803** | `results.tsv` |
| baseline `num_steps` | **44** optimizer steps (indices 0…43) | `.logs/baseline.log` |

### 1.2 Source 1 — training seed (init + data order), **intrinsic**

`train.py` contains `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` at lines 894–895
(autotune probe) and 1055–1056 (the real run), and `main()` exposes only `--smoke-test` and
`--dataset` (lines 1218–1221). There is **no `--seed` flag**.

*Everything the seed controls is constant across every run the project has ever done.*
Consequently:

- `Var_seed` contributes **exactly zero** to every observed difference to date.
- `Var_seed` is therefore **entirely unmeasured**, not "measured as small".
- The seed lives **inside the agent's editable file**, so an agent that touches RNG handling
  silently redefines the noise model (§10, gate G1).

### 1.3 Source 2 — agent sampling (temperature + prompt realisation), **extrinsic**

The agent's output is a stochastic function of its decoding state: the artefact it commits
for `variant v` on replicate `r` is one draw. Two different draws are different programs, with
different hyperparameters, and `val_bpb` between them differs by far more than any
re-run difference — this is the channel the nanochat loop *exists* to explore. It is
**unmeasured** in this project: `results.tsv` has exactly one row.

The MCQ pilot measured the analogous quantity for its own instrument: **SD = 0.0224**
(5 replicates × 20 items × 2 models at T = 0.7). That number is in *accuracy points* on a
4-way benchmark and is **not transferable** to `val_bpb`. It is used here only as a template
for the replicate structure and as a warning about §7.4 (`n = 5` estimates σ to within a
factor of 2.9, no better).

### 1.4 Source 3 — `program.md` variant, **the treatment**

Fixed factor. Its effect is what C-v is about. Note that because the DV is
`val_bpb@300s` (a *wall-clock-normalised* budget), a variant that makes training faster
reaches more optimizer steps and therefore scores better **without any improvement in the
model**. That is not noise — it is part of the treatment, and §3.4 handles it as a mediator.

### 1.5 A fourth channel nobody has named: **budget quantisation**

The training loop breaks on accumulated wall clock, not on a step count:

```python
if step > 10 and total_training_time >= target_training_seconds: break   # train.py:1184
if step > 10: total_training_time += dt                                  # train.py:1152-1153
```

Steps 0–10 are *not* counted. So the number of optimizer steps is a deterministic function of
the mean step time `dt̄` over the counted steps:

| total steps | requires `dt̄` ∈ | bucket width |
|---|---|---|
| 43 | (9.3750, 9.6774] s | 0.302 s |
| **44 (observed)** | **(9.0909, 9.3750] s** | **0.284 s** |
| 45 | (8.8235, 9.0909] s | 0.267 s |

Observed `dt̄` = 301.8 / 33 = **9.1455 s**, i.e. 0.60 % above the bucket floor. The per-step
SD of `dt` is 240 ms (2.6 %), but its **standard error over the 33 counted steps is only
41.8 ms = 0.457 %** — so the bucket is ±3.4 SE wide and, on a *stationary* clock, 44 steps is
the repeatable answer. A **sustained ≥ 0.6 % shift** in step time flips the run to 45 steps; a
**≥ 2.5 % shift** flips it to 43.

That matters because the loss slope near the end of training is steep. From the logged
trajectory (step 30 loss 3.462902 → step 43 loss 2.823026 over 13 steps):

```
Δloss/step = −0.049222 nats  = −1.74 % relative per step
```

Scaled to the DV that is **≈ 0.015 bpb per optimizer step** — an *order-of-magnitude proxy*
(derived from the smoothed train loss, not from a measured val_bpb step-response; §10, task
T4 measures the real number). Its consequence is already decisive: the effect sizes this
program treats as material are 0.001–0.005 bpb
(`program.md`: *"A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not
worth it."*), so **one step of budget quantisation is 3–15× the size of the effect being
claimed.** Budget quantisation is not a rounding error in this design; it is the largest
identified nuisance channel.

**Prediction to check (registered):** five back-to-back identical-config replicates in one
thermal window will return either (a) bit-identical `val_bpb` at the printed precision of
1e-6, or (b) two clusters separated by exactly one step, i.e. ≈ 0.015 bpb. A smooth
intermediate spread is the least likely outcome. The analysis branches accordingly (§6.5).

### 1.6 Numerical nondeterminism, measured directly

Matched-step comparison of the baseline run and replicate 1 (identical config, identical seed
42, different wall-clock session):

| step | baseline loss | repl-1 loss | Δ |
|---|---|---|---|
| 1 | 8.648036 | 8.648036 | **0** |
| 2 | 7.830703 | 7.830706 | 3e-6 |
| 3 | 7.022687 | 7.022694 | 7e-6 |

So the fixed-seed pipeline is **not bit-exact**: it drifts at ~1e-6 relative per step,
consistent with non-deterministic reduction order in the CUDA kernels. Projected onto the DV
this is ~1e-6 bpb — three orders of magnitude below the 0.001–0.005 bpb effects of interest.
**This is a floor on the residual SD, not the residual SD.**

---

## §2 Design

### 2.1 Factors

| factor | symbol | type | levels | note |
|---|---|---|---|---|
| `program.md` variant | `V` | **fixed** | `v₀` stock baseline (SHA-pinned), `v₁`, `v₂` | the treatment |
| agent replicate | `A` | **random** | `R` independent agent sessions per variant | **nested in `V`** — see 2.2 |
| training seed | `S` | **random** | `S` values of `--seed` | **crossed** with everything; gate G1 |
| session block | `B` | **random** | run order within the block | nuisance: thermal state, co-tenant load |

### 2.2 The nesting that is easy to get wrong

Agent replicates are **nested within variant**, not crossed. Replicate `r` of variant `v`
produces an artefact that belongs to `v` and can never be used as replicate `r` of variant
`w`. Therefore:

- `σ²_A` enters **every** contrast at the rate `1/R`, and it does not cancel between variants —
  there is no "same agent draw" that both variants share.
- `σ²_S` and `σ²_C` (the variant × seed interaction) can only be estimated by re-running
  **the same artefact** at different seeds. That is the only manipulation that isolates the
  seed, and it requires gate G1.

### 2.3 Structure: randomised complete blocks, with a crossed sub-structure

One **block** = one seed value × one contiguous time window, containing **exactly one run of
each variant**. Cost per block: `V · 513.7 s` = **25.7 min for V = 3**; **56 complete blocks
fit in 24 h**.

Within-block contrasts remove the seed main effect `b_s` entirely and distribute clock drift
across variants. Between blocks, the seed value and the artefact both change, so:

> **Identifiability trap (registered).** If every block uses a new seed *and* a new agent
> replicate, then `σ²_A` and `σ²_{S}` / `σ²_{VS}` are **completely aliased**. The variant
> main effect is still estimable (it is a within-block contrast), but "does it generalise over
> seeds?" is not separable from "does it generalise over agent draws?".
> **Breaking the aliasing requires the crossed sub-structure:** at least two variants must
> have at least one *fixed artefact* run at ≥ 2 seeds. That is what buys `df_S = S − 1 > 0`.

### 2.4 The model

```
y_vrs = μ + α_v + a_{r(v)} + b_s + c_vs + ε_vrs

  α_v     fixed        variant effect                 (the treatment)
  a_{r(v)} ~ N(0, σ²_A)  agent replicate, nested in V
  b_s     ~ N(0, σ²_S)   training seed (init + data order)
  c_vs    ~ N(0, σ²_C)   variant × seed interaction
  ε_vrs   ~ N(0, σ²_e)   residual: numerical drift + budget quantisation + session jitter
```

Single-observation variance: `Var(y) = σ²_A + σ²_S + σ²_C + σ²_e ≡ σ²₀`.

**`σ₀` is the single number this whole design turns on, and it is not yet measured.**
The 5-replicate study now in flight estimates `σ²_e + Var(budget quantisation)` at seed 42 —
call it `σ²_resid`. It does **not** estimate `σ²_S`, because the seed is a constant in that
study (`df_S = S − 1 = 0`). Reading `SD ≈ 0` from it as "training is deterministic" is the
exact error FLAW 7 warns about.

### 2.5 The treatment contrast, and the parts that cancel

For variants `v`, `w` measured on the **same seeds**:

```
Δ̂ = ȳ_v·· − ȳ_w··
Var(Δ̂ | blocked on seed) = 2[ σ²_A/R + σ²_C/S + σ²_e/(R·S) ]          (b_s cancels)
Var(Δ̂ | unblocked)       = 2[ σ²_A/R + σ²_S/S + σ²_C/S + σ²_e/(R·S) ]  (+σ²_S/S)
```

**`R` and `S` are not interchangeable.** `σ²_A` is divided by `R`, `σ²_C` by `S`, `σ²_e` by
`RS`. An extra agent replicate buys something different from an extra seed, and which is
worth more depends on the unknown composition of `σ₀²` (§7.3).

**The single cheapest error to avoid:** failing to block on seed inflates the critical value
by `√(1 + σ²_S/(σ²_A + σ²_C + σ²_e))`. Under the three composition scenarios of §7.3 that
factor is **1.026× / 1.581× / 1.155×**, i.e. **1.05× / 2.50× / 1.33× in run cost** — bought
for nothing by giving every variant the same seeds.

---

## §3 Estimator

### 3.1 Primary: variance components by REML, treatment test by Satterthwaite / Kenward–Roger

Fit the §2.4 model by REML with `variant` as the only fixed effect (plus an intercept, plus
`num_steps` in the sensitivity model of §3.4) and crossed random effects
`(1|variant:agent)`, `(1|seed)`, `(1|variant:seed)`, `(1|obs)`. Report:

- `σ̂²_A, σ̂²_S, σ̂²_C, σ̂²_e` with **profile-likelihood CIs** — profile CIs rather than Wald,
  because every one of these components is bounded at 0 and the boundary case is not
  hypothetical (if training really is deterministic given the seed, `σ̂²_C` and `σ̂²_e` sit at
  the boundary and REML asymptotics do not apply).
- The **proportion of variance** table: `σ̂²_x / σ̂²₀` for each `x`, plus
  `η²_V = SS_V / SS_total` and the partial `η²_{V|A,S} = SS_V / (SS_V + SS_error)`
  (§6.2 explains why the raw form is not the effect size to quote).
- Negative variance estimates reported as **0 with the raw negative value shown**. A truncated
  zero is not evidence that the component *is* zero; say "not distinguishable from 0 at this
  design" instead.

### 3.2 Cross-check: closed-form EMS estimators

Because the design is balanced, the ANOVA expected-mean-square estimators are exact and make
no distributional assumption beyond independence:

| source | df | expected MS | closed-form estimator |
|---|---|---|---|
| variant `V` | `V − 1` | `σ²_e + S·σ²_A + R·σ²_C + R·S·ψ²` | `F = MS_V / MS_{V×S}` |
| agent `A` within `V` | `V(R − 1)` | `σ²_e + S·σ²_A` | `σ̂²_A = (MS_A − MS_e)/S` |
| seed `S` | `S − 1` | `σ²_e + V·R·σ²_S + R·σ²_C` | `σ̂²_S = (MS_S − MS_{V×S})/(V·R)` |
| variant × seed `V×S` | `(V − 1)(S − 1)` | `σ²_e + R·σ²_C` | `σ̂²_C = (MS_{V×S} − MS_e)/R` |
| residual | `V(R − 1)(S − 1)` | `σ²_e` | `σ̂²_e = MS_e` |

with `ψ² = Σα_v²/(V − 1)`. Both estimators are reported; disagreement beyond the CIs is a
finding about non-normality or unbalanced data, not something to average away.

> **The denominator matters.** The error term for the treatment F-test is **`MS_{V×S}`, not
> `MS_residual`**. Testing `MS_V` against the residual asserts that the variant effect is
> common to all seeds — i.e. it assumes away the generalisation question. Under the restricted
> mixed model, `MS_{V×S}` is the correct denominator; when `σ̂²_C = 0` it is conservative
> (it throws away `df`), which is why Satterthwaite adjustment is the primary method and the
> EMS F is the cross-check.

### 3.3 Degrees-of-freedom bookkeeping

`df` must sum to `V·R·S − 1` after allocating 1 to the mean. Verified for every candidate
design:

| `(V, R, S)` | `N` | `df_V` | `df_A` | `df_S` | `df_{V×S}` | `df_e` | check |
|---|---|---|---|---|---|---|---|
| (2, 1, 1) | 2 | 1 | **0** | **0** | **0** | **0** | 2 = 2 |
| (3, 1, 1) | 3 | 2 | **0** | **0** | **0** | **0** | 3 = 3 |
| (2, 2, 2) | 8 | 1 | 2 | 1 | 1 | 2 | 8 = 8 |
| (3, 2, 2) | 12 | 2 | 3 | 1 | 2 | 3 | 12 = 12 |
| **(3, 3, 3)** | **27** | **2** | **6** | **2** | **4** | **12** | 27 = 27 |
| (3, 4, 3) | 36 | 2 | 9 | 2 | 4 | 18 | 36 = 36 |
| (3, 6, 5) | 90 | 2 | 15 | 4 | 8 | 60 | 90 = 90 |
| (3, 8, 6) | 144 | 2 | 21 | 5 | 10 | 105 | 144 = 144 |
| (2, 8, 8) | 128 | 1 | 14 | 7 | 7 | 98 | 128 = 128 |

The variant contrast uses the conservative error df `min(df_A, df_{V×S})`:

| design | `df_A` | `df_{V×S}` | conservative df | `t_{.975} + t_{.80}` |
|---|---|---|---|---|
| (2, 1, 1) | 0 | 0 | **none — test undefined** | — |
| (3, 1, 1) | 0 | 0 | **none — test undefined** | — |
| (3, 2, 2) | 3 | 2 | 2 | 5.3633 |
| **(3, 3, 3)** | 6 | 4 | **4** | **3.7174** |
| (3, 6, 5) | 15 | 8 | 8 | 3.1949 |
| (3, 8, 6) | 21 | 10 | 10 | 3.1072 |
| (2, 8, 8) | 14 | 7 | 7 | 3.2607 |

### 3.4 Budget quantisation is a mediator, not a covariate

The DV is the **total** effect on `val_bpb@300s`. `num_steps` is therefore *downstream* of the
treatment (a variant that changes model size or batch shape changes the step count). Adjusting
the primary estimand for `num_steps` would be conditioning on a mediator — over-adjustment
bias. So:

- **Primary estimand:** `Δ_total` on `val_bpb@300s`, unadjusted.
- **Mediation decomposition, reported alongside:**
  `Δ_total = Δ_fixed-steps + (Δ_num_steps) · (∂bpb/∂step)`.
  The second term is calibrated by the T4 budget perturbatation (§10), which measures
  `∂bpb/∂step` directly; the first term is the variant effect *at equal compute*.
- If `|Δ_fixed-steps|` is indistinguishable from 0 while `Δ_total` is not, the honest report is
  **"the variant wins by changing the compute trajectory, not the learning dynamics"**. That
  is a real result, and it must be reported as such rather than as a modelling improvement.
- For the *noise* decomposition (§1.5), only the part of the step count that varies when
  nothing changed is noise. `num_steps` is logged per run so the two parts are separable.

### 3.5 Companion: within-block randomisation test (see §6.3)

Whenever the block structure is complete (every variant present once per block), the variant
effect is additionally tested by permuting variant labels **within each block**, using the
M3 statistic on block-centred data. This is distribution-free and is the direct generalisation
of `oasp/stats.py::permutation_null`. Its resolution floor is `1/(n_perm + 1)` and it is
therefore reported *alongside* the LMM test, never instead of it.

---

## §4 What is identifiable at each run count

| runs | structure | identifiable | **not** identifiable |
|---|---|---|---|
| 1 | single run | `val_bpb` itself | everything. `σ²₀` has 0 df; the "effect" of that configuration is one draw |
| 2–5 | identical config, fixed seed | `σ²_resid` (numerical drift + budget quantisation, **at seed 42 only**) | `σ²_S`, `σ²_A`, `σ²_C` — all 0 df |
| 8 | `2 × 2 × 2` | **all four components get df > 0** (`df_A = 2, df_S = 1, df_C = 1, df_e = 2`) — the *identification floor* | anything with power: the contrast error df is 1, `t` sum = 14.08 |
| 12 | `3 × 2 × 2` | components + a usable (weak) test | `σ²_A` vs `σ²_C` separation is poor (`df_A = 3`, `df_C = 2`) |
| **27** | **`3 × 3 × 3`** | components + a testable treatment effect, `df (2, 4)`; every component `df ≥ 2` — the **inference floor** | effects smaller than ≈ 1.8–2.9 `σ₀` (§7.3) |
| 90 | `3 × 6 × 5` | as above at `df (2, 8)`, `df_A = 15`, `df_e = 60` | effects smaller than ≈ 1.1–1.8 `σ₀` |
| 144 | `3 × 8 × 6` | as above at `df (2, 10)` | effects smaller than ≈ 0.96–1.52 `σ₀` |
| any | **without `--seed`** | `σ²_A`, `σ²_e`, `Δ_total`, `Δ̂` at the frozen seed | **`σ²_S` and `σ²_C` are structurally unidentifiable** — no run count fixes this. Gate G1 |

Also permanently outside this design (stated so they are never claimed):

- transfer to a **different agent model** (a frontier agent is a different treatment, FLAW 4);
- transfer across **hardware** or a different time budget (FLAW 6);
- the agent's internal split of sampling noise into `temperature` vs `prompt realisation` —
  measured only by holding one fixed and varying the other;
- long-horizon drift over days (needs repeated blocks across days, not merely more blocks).

---

## §5 Why a single run per variant cannot separate the sources

Four independent arguments, each with a number.

### 5.1 One equation, four unknowns — the ridge is 3-dimensional

For two variants at one seed, with `R = S = 1`:

```
Δ_observed = (α_v − α_w) + (a_{1(v)} − a_{1(w)}) + (c_{v1} − c_{w1}) + (ε_{v11} − ε_{w11})
                    ↑                ↑                    ↑                   ↑
                 treatment      agent sampling       variant×seed          residual
```

**One scalar observation, four unknown quantities.** The likelihood is flat along a
3-dimensional ridge: absolutely any split of `Δ_observed` among the four terms fits the data
exactly as well. The comparison has no power to attribute *any* part of the difference to the
treatment. This is not "low power" — it is **zero identification**, and it is why FLAW 7
called N = 1 *blocking* rather than *underpowered*.

### 5.2 Every variance component has 0 degrees of freedom

From §3.3: for `(V, 1, 1)`, `df_A = V(1−1) = 0`, `df_S = 0`, `df_{V×S} = 0`,
`df_e = V(1−1)(1−1) = 0`. All three sums of squares are `0`, and every variance component
estimator is `0/0`. The treatment F-statistic is `MS_V / MS_{V×S}` where `MS_{V×S}` is
undefined — **the test statistic is not a random variable with a null distribution, so no
p-value exists at all.** For V = 3 the only defined row of the ANOVA table is the `V` row
with 2 df; the remaining `3 − 1 − 2 = 0` df are nothing.

### 5.3 The permutation path is also closed at one block

`oasp/stats.py::permutation_null` uses the `(count + 1)/(B + 1)` convention, so the smallest
attainable p-value per block is `1/(V! + 1)`:

| variants | blocks | permutations | p_min |
|---|---|---|---|
| 2 | **1** | 2 | **0.3333** |
| 3 | **1** | 6 | **0.1429** |
| 4 | **1** | 24 | 0.0400 |
| 3 | 2 | 36 | 0.0270 |
| 3 | 3 | 216 | 0.0046 |
| 2 | 5 | 32 | 0.0303 |

With one run per variant there is exactly one block. For V = 2 or 3, **the randomisation test
cannot return p < 0.05 no matter how large the effect is.** Minimum block counts to reach
α = 0.05: **5 blocks for V = 2, 2 blocks for V = 3** (and 3+ in practice, because 2 blocks
leaves the p-grid at 0.027 and power near zero).

### 5.4 Borrowing an external σ is worse than it looks

The tempting repair — import `σ̂₀` from the 5-replicate study and z-test the difference — has
three quantified defects:

1. **It is less sensitive than replicating.** σ̂ from 5 draws has 4 df, so the test statistic
   is t₄: MDD = `(t_{.975,4} + t_{.80,4})·√2·σ₀` = **5.25 σ₀**. Five runs per arm with an
   internal pooled σ (8 df) gives MDD = **2.02 σ₀**, and more importantly it is a *different* σ.
   **N = 1 per arm is 2.6× less sensitive than 5 runs per arm** — it saves 4 runs and buys a
   2.6× larger blind spot.
2. **σ is not exchangeable across variants.** The external σ is measured on the *baseline
   artefact*. A variant that changes depth, batch size or activation checkpointing changes the
   noise scale. If the true σ is 2× the assumed one, a nominal α = 0.05 test has actual
   α = **0.327**; at 1.5× it is **0.191**; at 1.2× it is **0.102**.
3. **The sign is wrong 1 time in 4.** With a single run per arm, for a true effect of size
   Δ = 1 σ₀ the probability that the observed difference has the **wrong sign** is
   `Φ(−Δ/(σ₀√2))`:

| true effect | P(sign of observed difference is wrong) |
|---|---|
| 0.5 σ₀ | **0.362** |
| 1.0 σ₀ | **0.240** |
| 2.0 σ₀ | 0.079 |
| 3.0 σ₀ | 0.017 |

### 5.5 …and the search wraps all of this in a winner's curse

The loop keeps the best candidate, which means the *reported* improvement is a maximum over
noise. Under pure noise, the expected apparent improvement from `k` single-run candidates is:

| candidates `k` | `E[max] / σ₀` | at `σ₀ = 0.005` bpb | at `σ₀ = 0.001` bpb |
|---|---|---|---|
| 20 | 1.867 | 0.00933 bpb | 0.00187 bpb |
| 32 | 2.071 | 0.01035 bpb | 0.00207 bpb |
| 100 | 2.509 | 0.01254 bpb | 0.00251 bpb |
| 168 (one 24 h block) | 2.689 | 0.01344 bpb | 0.00269 bpb |

With `σ₀ = 0.005 bpb`, a night of 100 single-run candidates manufactures a **0.0125 bpb
apparent improvement from nothing** — 12× the 0.001 threshold `program.md` itself calls the
minimum interesting gain, and 2.5× the 0.005 "material" threshold. **Any C-v claim made from a
single-run-per-variant trajectory is a maximum of noise until this is ruled out**, and the
only way to rule it out is an independent replicate of the *same* artefact.

---

## §6 Does the M3 η²-permutation approach generalise?

Short answer: **the statistic generalises with a correction; the null does not generalise as
written; a correct distribution-free analogue exists but only for the blocked sub-design.**

(Implementation note, stated for accuracy: `oasp/stats.py` has no function literally named
`metric_m3`. The M3 row of `RESEARCH_PROTOCOL.md §4` is implemented as
`permutation_null()` + `_eta2_prompt()`, with `_spread_stat()` as the second statistic.)

### 6.1 What the MCQ version actually exploits

`permutation_null` shuffles **variant labels within each item**. It is valid because:

- the design is **paired** — every variant is observed on every item;
- items are the exchangeable unit: a 0/1 outcome vector within an item is, under H0, an
  exchangeable multiset;
- item difficulty is preserved by construction (the shuffle never crosses items).

The structural fact doing the work is **"each variant is observed once inside each level of
the nuisance factor, and the nuisance factor is common to all variants."**

### 6.2 The statistic: needs to become partial

`_eta2_prompt` computes `η² = SS_V / SS_total` over the **whole grid**. In the MCQ matrix the
denominator is inflated only by item difficulty. In the nanochat design the denominator would
be inflated by `σ²_A + σ²_S + σ²_C + σ²_e`, of which `σ²_A` alone may be 80 % of the total
(scenario A, §7.3). Quoting that number would report a *dilution*, not an effect.

**Correct generalisation of the statistic:**

```
η²_raw     = SS_V / SS_total                                  (comparable to the MCQ number)
η²_partial = SS_V / (SS_V + SS_error)                         ← use this one
             SS_error = SS_A + SS_S + SS_{V×S} + SS_e
```

Both are reported, with `η²_partial` as the headline and `η²_raw` explicitly labelled
"diluted by design variance; not comparable across designs".

### 6.3 The null: within-**block** shuffle, not global shuffle

The exchangeable unit in the nanochat design is the **block** (§2.3), not the run. The
generalised null is:

> **Permute variant labels within each block** (equivalently: recompute `SS_V` on
> block-centred data after permuting the labels of the `V` runs in each block), and compare
> the observed statistic to that null.

This is valid under H0 *provided* the block is complete (every variant exactly once per
block) and the block is defined by things common to all variants in it (seed, time window).
It is the exact structural analogue of the within-item shuffle.

**A global label shuffle over the whole crossed design is wrong, and the error is
quantifiable.** Under the crossed design, all `R·S` runs of a variant share the same
artefact *within* a replicate, so a run is not exchangeable with a run of a different
replicate. A global shuffle silently assumes a one-way layout with iid errors, which inflates
the null's critical value by

```
√( 1 + σ²_S / (σ²_A + σ²_C + σ²_e) )
```

| composition | inflation of critical value | power-equivalent run cost |
|---|---|---|
| A: agent-dominated | 1.026× | 1.05× |
| B: seed-dominated | **1.581×** | **2.50×** |
| C: even split | 1.155× | 1.33× |

So using the naive global shuffle in the regime the design exists to guard against makes the
test conservative by a factor that costs 2.5× the runs.

### 6.4 The correct alternative when the blocked structure is unavailable

If blocks are incomplete (e.g. the agent crashed on one variant in a block), the within-block
permutation is not available. The replacement is **not** a global shuffle; it is:

1. **Parametric bootstrap under the fitted null.** Fit the §2.4 model, set `α_v = 0` for all
   `v`, resimulate `y*` from the fitted distribution, refit, and recompute `SS_V`. This is the
   permutation substitute for a variance-components null and it is valid at any block
   completeness. Requires the fitted model to be believable — which the EMS cross-check (§3.2)
   makes auditable.
2. **Kenward–Roger / Satterthwaite F-test** (§3.1), reported with the profile-likelihood CIs
   on the components.
3. Report the block-level losses explicitly (`n_blocks_complete`, `n_runs_dropped`) rather
   than silently running the test on an incomplete grid — the same discipline
   `item_outcome_matrix()` already applies to the MCQ matrix, where a hole was found to be
   counted as an IPS flip.

### 6.5 The zero-SD branch

If the in-flight 5-replicate study returns bit-identical `val_bpb` at 1e-6:

| consequence | action |
|---|---|
| `σ̂_resid < 1e-6 bpb` is **resolution-limited**, not zero | report as `< 1e-6`; never as "deterministic" |
| `σ²_e` and `σ²_C` are at the boundary | REML asymptotics void → report profile CIs and the EMS estimator, both truncated at 0 with raw values shown |
| the confirmatory design needs fewer `R`, `S`? | **No** — it needs the same `R` (agent variance is where the variance now provably lives) and it needs `S ≥ 2` *if and only if* gate G1 is open. Without G1, seed generalisation is out of scope |
| freed runs | put them into `R` (agent replicates). §7.3 shows agent variance is the binding constraint under scenarios A and D |
| the channel that this study **cannot** see | budget quantisation under a changed clock state (§1.5). Log `num_steps`; treat a step-count change as a block-level event, not a residual |

---

## §7 Power

### 7.1 The scaling law

For a pairwise contrast with `R` agent replicates and `S` seeds per variant:

```
MDD(R, S) = ( t_{1−α/2, ν} + t_{1−β, ν} ) · σ₀ · √( 2[ a_A/R + a_C/S + a_e/(R·S) ] )
ν         = min( V(R−1), (V−1)(S−1) )        (conservative; Satterthwaite is smaller)
a_x       = σ²_x / σ₀²,  a_A + a_S + a_C + a_e = 1
```

with `α = 0.05` two-sided and power `1 − β = 0.80`. The `a_·` composition is unknown, so the
MDD is quoted across four scenarios: **A** agent-dominated (0.80, 0.05, 0.05, 0.10),
**B** seed-dominated (0.10, 0.60, 0.20, 0.10), **C** even (0.25×4), **D** agent-only
(0.95, 0, 0, 0.05).

**Minimum detectable difference, in units of the single-run SD `σ₀`:**

| design | `N` | A | B | C | D |
|---|---|---|---|---|---|
| (2, 2, 2) | 8 | 13.360 | 8.331 | 11.133 | 13.905 |
| (3, 2, 2) | 12 | 5.088 | 3.173 | 4.240 | 5.296 |
| **(3, 3, 3)** | **27** | **2.853** | **1.752** | **2.318** | **2.984** |
| (3, 4, 3) | 36 | 2.494 | 1.662 | 2.146 | 2.584 |
| (3, 6, 5) | 90 | 1.730 | 1.107 | 1.429 | 1.807 |
| (3, 8, 6) | 144 | 1.460 | 0.962 | 1.228 | 1.521 |
| (2, 8, 8) | 128 | 1.514 | 0.911 | 1.188 | 1.594 |

Read the table as the price of ignorance: at 27 runs the design can only see an effect of
~2–3 `σ₀`; at 144 runs it can see ~1–1.5 `σ₀`; and **no affordable design sees below ~0.9 σ₀**.
Halving the MDD costs 4× the runs.

### 7.2 No replicating at all: the two-sample reference

For comparison, a plain two-sample test with `n` runs per arm and an **internal** σ:

| `n` per arm | df | MDD / σ₀ |
|---|---|---|
| 1 (with an **external** σ from 5 draws, t₄) | 4 | **5.257** |
| 1 (with σ treated as known, z) | ∞ | 3.962 |
| 2 | 2 | **5.363** ← worse than the z-case: the df penalty exceeds the variance gain |
| 3 | 4 | 3.035 |
| 5 | 8 | 2.021 |
| 10 | 18 | 1.325 |
| 20 | 38 | 0.909 |
| 50 | 98 | 0.566 |

Note `n = 2` being *worse* than `n = 1` with a known σ. This is the quantitative reason the
protocol must pre-commit to which σ it is testing against, and why "just add one replicate" is
not a strategy.

### 7.3 Required runs as a function of the noise that has not been measured yet

Target effects, pre-specified from the loop's own materiality language:

- `Δ* = 0.005 bpb` — the **material** improvement (the magnitude of the changes `program.md`'s
  own worked example keeps);
- `Δ_min = 0.001 bpb` — the **minimum interesting** improvement;
- equivalence margin `δ = 0.001 bpb` (mirroring `DESIGN["equivalence_margin"] = 0.05` on the
  MCQ side): a variant whose 90 % CI on `Δ` lies inside ±0.001 is declared **inert**, and that
  must be reportable as a positive finding rather than as a failed test (FLAW 1, FLAW 10).

Required `R` (at worst-case scenario A, `V = 3`):

| `σ₀` (bpb) | for `Δ* = 0.005`, `S = 5` | `N` | wall | for `Δ_min = 0.001`, `S = 5` | `N` | wall |
|---|---|---|---|---|---|---|
| 1e-5 | 1 → use **2** | 30 | 4.3 h | 2 | 30 | 4.3 h |
| 1e-4 | 2 | 30 | 4.3 h | 2 | 30 | 4.3 h |
| 5e-4 | 2 | 30 | 4.3 h | **6** | 90 | 12.8 h |
| 1e-3 | 2 | 30 | 4.3 h | **27** | 405 | 57.8 h ✗ |
| 2e-3 | **4** | 60 | 8.6 h | **> 200** | > 3000 | ✗ |
| 5e-3 | **27** | 405 | 57.8 h ✗ | ✗ | ✗ | ✗ |
| 1e-2 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

> **Feasibility threshold (registered).** One 24 h block supports `Δ* = 0.005 bpb` **iff
> `σ₀ ≤ 0.002 bpb`** — that is 0.23 % of the baseline `val_bpb` of 0.863803. If `σ₀` comes
> back above 0.002, the honest options are (i) longer than a day, (ii) a larger `Δ*`, or
> (iii) reporting "underpowered to decide" — which, per FLAW 10, is **not** the same as
> reporting a null.

Cross-check against the two channels already sized in §1:

- numerical nondeterminism alone: `σ₀ ≈ 1e-6` → **trivially affordable**;
- budget quantisation alone, if the step count flips with probability `p`:
  `σ₀ ≈ 0.015 √(p(1−p))` → `p = 0.5` gives `σ₀ ≈ 0.0075` → **infeasible in one day**;
  `p = 0.1` gives `σ₀ ≈ 0.0045` → still infeasible for `Δ*`.
- agent sampling alone: **unmeasured**, and by construction the largest channel
  (a different artefact is a different program).

**So the empirical question that decides whether this program is feasible at all is
"how stable is the step count and how large is the agent's contribution?" — and it is
answerable in < 3 h of runs (tasks T1–T3, §10).**

### 7.4 Sizing the confirmatory design on a 5-replicate pilot is not allowed

With `n` replicates the 95 % CI on σ is `[√((n−1)/χ²_{.975}), √((n−1)/χ²_{.025})] × σ̂`:

| `n` | 95 % CI on σ | implied range in required `R` (∝ σ²) |
|---|---|---|
| **5** | **[0.599, 2.874] × σ̂** | **[0.36×, 8.26×]** — a **23×** span |
| 10 | [0.688, 1.826] × σ̂ | [0.47×, 3.33×] — 7× |
| 20 | [0.760, 1.461] × σ̂ | [0.58×, 2.13×] — 3.7× |

**A 5-replicate σ̂ pins the required run count only to within a factor of 23.** The
pre-registered consequence: the pilot's σ̂ must be inflated to its **upper 95 % bound** before
it is used to size anything, and the design must be **staged with an interim decision point**
(§9, gates G2/G3) rather than sized once from the pilot.

### 7.5 Multiplicity

`V = 3` gives 2 confirmatory contrasts against the baseline. With `σ₀` as the yardstick and
`Δ* = 0.005`, the horizon is small (`k = 2`), so **Benjamini–Hochberg at q = 0.05** on the
contrast p-values, applied to the confirmatory set only. Any additional variant is
**exploratory** and must be labelled so; adding variants does not improve the precision of the
existing contrasts and only costs runs and multiplicity.

---

## §8 Decision rule and falsification conditions

### 8.1 Decision rule (pre-registered)

For each variant `v` versus the stock baseline `v₀`:

1. **Estimate** `Δ̂_v` with the Satterthwaite / KR SE from the §2.4 model; report `Δ̂`, its
   95 % CI, `num_steps` per cell, and the full variance-component table.
2. **Existence** — *requires a non-degenerate error df.* `Δ̂_v`'s 95 % CI excludes 0.
3. **Agent generalisation** — the CI computed with `σ̂²_A` in the variance (i.e. the
   "over agent draws" CI) excludes 0.
4. **Seed generalisation** — the CI computed with `σ̂²_C` (variant × seed) in the variance
   excludes 0. **Only reachable if gate G1 has been passed.**
5. **Materiality / equivalence** — BH-adjusted `q < 0.05` **and** `|Δ̂_v| ≥ Δ* = 0.005 bpb`.
   If instead the **90 % CI ⊂ (−0.001, +0.001)**, declare the variant **inert**; this is a
   reportable positive result (the M3/equivalence lesson: a difference test can never
   establish equivalence).
6. **Mediation** — report `Δ_total`, `Δ_fixed-steps`, and `Δ_num_steps · ∂bpb/∂step`
   (§3.4). A claim of *modelling* improvement requires `Δ_fixed-steps` to survive step 2.

**C-v is supported** iff steps 2, 3, 4 and 5 all hold for at least one variant. Otherwise the
verdict is one of: **refuted** (§8.2), **inert** (equivalence established), or
**underpowered to decide** (neither — and that phrase must be used, not "not supported").

### 8.2 Falsification conditions — what result would REFUTE C-v

| id | observation | what it refutes |
|---|---|---|
| **F1** | `σ̂²_C`'s upper 95 % bound puts the seed-generalised CI on `Δ̂` across 0 | C-v·seed. The variant's advantage is seed-specific → not a property of the variant |
| **F2** | `σ̂²_A`'s upper bound puts the agent-generalised CI on `Δ̂` across 0, or `σ̂_A ≥ Δ̂√(R/2)/1.96` | C-v·agent. Concretely at `R = 6`, `Δ̂ = 0.005`: if the agent-draw SD exceeds **0.00442 bpb**, six replicates cannot confirm a 0.005 bpb claim |
| **F3** | `Δ_total` significant but `Δ_fixed-steps`'s CI covers 0, with `Δ_num_steps ≠ 0` | any claim that the variant improves *learning*. It improves speed. Report the mediation and drop the modelling claim |
| **F4** | 90 % CI on `Δ̂` ⊂ (−0.001, +0.001) | any material effect of that variant. Registered as **inert**, and this outcome is publishable as a positive finding |
| **F5** | `σ̂₀ > 0.002 bpb` and the block rejects the run-count escalation | nothing about the variant — the **design**. Verdict: "underpowered to decide at affordable cost". Never a null |
| **F6** | the block's best-of-`k` improvement is `≤ E[max]·σ₀` for that `k` (2.509 σ₀ at `k = 100`) | any claim that the *search trajectory* found something rather than sampling noise |
| **F7** | gate G1 not passed (no `--seed` outside the agent's edit surface) | **any use of the word "generalises" over training seeds.** The claim degrades to "at seed 42", and must be written that way |
| **F8** | the 5-replicate study returns `SD = 0` **and** a later block shows a step-count change with a `val_bpb` shift of ~0.015 | "training is deterministic given the seed". Determinism at a frozen clock state is not determinism |

### 8.3 What would NOT be a refutation

- A variant that fails on a *single* run, however large the deficit (F5.1: one draw).
- A variant whose advantage appears only in the `keep` column of `results.tsv` — that column is
  a maximum over noise (§5.5).
- A spread smaller than the noise floor: that is the *equivalence* direction and must be tested
  as such (FLAW 1).

---

## §9 What is affordable in one 24 h unattended block

**Capacity.** 24 × 60 / 8.5617 = **168.2 runs** with zero non-training overhead. Non-training
time beyond the measured 211.9 s is real: the agent must read the log and write the next
artefact. At a 20 % allowance the planning capacity is **140 runs/24 h**; the hard ceiling
stays 168.

| plan | composition | `N` | wall at 7.008 runs/h | % of 24 h |
|---|---|---|---|---|
| **P0 — pilots only** | T1 (5) + T2 (5) + T3 (5) + T4 (2) | **17** | **2.4 h** | 10.1 % |
| **P1 — inference floor** | P0 + `(3, 3, 3)` | **44** | **6.3 h** | 26.2 % |
| **P2 — recommended** | P0 + `(3, 6, 5)` | **107** | **15.3 h** | 63.6 % |
| **P3 — ceiling-limited** | P0 + `(3, 8, 6)` | **161** | **23.0 h** | 95.7 % |

### Worked detail for P2 (the recommended block)

```
0.0 h  T1  rerun floor        5 runs   same artefact, same seed 42, same thermal window
0.7 h  T2  seed sweep         5 runs   same artefact, seeds 42*k,  k=1..5      [gate G1]
1.4 h  T3  agent sweep        5 runs   stock program.md, 5 fresh agent sessions, seed 42
2.2 h  T4  budget calibration 2 runs   same artefact at 295 s and 305 s -> ∂bpb/∂step
2.4 h  --- interim gate G2/G3 (below) ---
2.4 h  C   crossed design    90 runs   V=3 x R=6 x S=5, blocked as 30 blocks of 3 runs
15.3 h spare 5 runs                                   (crashes, 1 exploratory variant)
```

Blocks are 3 runs = **25.7 min**, so the block has room for **56 complete blocks** — far more
than the 30 required. Run order is randomised within each block.

### The two gates, and why they exist

- **G1 (hard prerequisite, before T2).** The seed must be settable from **outside the agent's
  edit surface**: a `--seed` flag in the frozen harness, a static check that `train.py` contains
  no `torch.manual_seed` / `random.seed` / `np.random.seed` that the agent could alter, and a
  runtime assertion that the seed actually took. **Without G1, T2, `σ²_S`, `σ²_C` and C-v·seed
  do not exist** (§4, F7).
- **G2 (interim, at 2.4 h).** Compute `σ̂₀` from T1/T2/T3, inflated to its **upper 95 % bound**
  (§7.4). Then:
  - `σ₀_upper ≤ 0.001` → run P2 with `R = 4` and spend the surplus on `S = 6`;
  - `0.001 < σ₀_upper ≤ 0.002` → run P2 as specified;
  - `σ₀_upper > 0.002` → **do not run the crossed design**. Escalate to a multi-day campaign or
    re-scope `Δ*` upward; record F5.
- **G3 (interim, at 2.4 h).** Compute the `σ̂²_A`-driven expected MDD from T3. If the agent
  channel alone puts MDD above `Δ*`, redirect the surplus into `R` before starting.
  **The agent channel is the one most likely to be binding, and it is the one nobody has
  measured.**

### What one day does *not* buy

- Any effect below **≈ 0.9 σ₀** (the `(3, 8, 6)` floor) — with `σ₀ = 0.002` that is 0.0018 bpb;
- any statement about a variant's behaviour under a **different agent model**, different
  hardware, or a different time budget;
- long-horizon drift, which needs repeated blocks on different days, not more blocks in one day.

---

## §10 Harness tasks this design requires (no GPU work in this document)

| id | task | why | cost |
|---|---|---|---|
| **T1** | finish the in-flight 5-replicate identical-config run | `σ̂_resid` + the §1.5 prediction; already running | 0 (in flight) |
| **T2** | same artefact × 5 seeds | `σ²_S`; the only way to see the seed channel | 5 runs, 0.7 h |
| **T3** | stock `program.md` × 5 fresh agent sessions × seed 42 | `σ²_A` — **the largest unmeasured channel** | 5 runs, 0.7 h |
| **T4** | same artefact at `TIME_BUDGET` 295 s and 305 s | the real `∂bpb/∂step`; replaces the §1.5 proxy | 2 runs, 0.3 h |
| **T5** | log `num_steps`, `dt` mean/SD, `seed`, `artifact_sha256`, `program_md_sha256`, `block_id`, `variant`, `agent_replicate` | makes §3.4 mediation and §6.3 blocking computable at all | 0 |
| **T6** | implement the §2.4 fit + §3.2 EMS cross-check in `oasp/stats.py`, with `η²_partial` and the within-block permutation | the estimator must exist before the block runs, and must be frozen before data | ~1 day of CPU |
| **T7** | run the T6 estimator on the **existing** `pilot_*` MCQ data as a regression test | proves the code path before it is load-bearing | 0 |

**Estimator freeze.** T6's code must be committed and SHA-pinned **before** the block starts.
Any change after that invalidates the confirmatory label — the same discipline
`runs/<id>/manifest.json` already applies to benchmark hashes on the MCQ side.

---

## §11 Summary — identified vs not

| quantity | identified? | by what |
|---|---|---|
| `val_bpb@300s` for one configuration | yes | 1 run |
| `σ²_resid` (numerical drift + budget quantisation, **at seed 42**) | yes, given ≥ 2 identical runs | T1 |
| `σ²_A` (agent sampling) | yes, given ≥ 2 agent replicates per variant | T3 |
| `σ²_S` (training seed) | **only** with gate G1, ≥ 2 seeds on a common artefact | T2 |
| `σ²_C` (variant × seed interaction) | yes, given ≥ 2 seeds on a common variant × replicate | crossed design |
| `Δ_total` (variant effect on `val_bpb@300s`) | yes, given the crossed design and gate G2 | P1/P2/P3 |
| `Δ_fixed-steps` (variant effect at equal compute) | yes, given T4 | T4 + crossed |
| the equivalence verdict ("this variant is inert") | yes, as a TOST direction | §8.1 step 5 |
| allocation of a single observed difference among the three sources at `N = 1` | **no — 3-dimensional ridge** | §5.1 |
| any p-value at `N = 1` per variant | **no — the error MS is `0/0`** | §5.2 |
| seed generalisation without `--seed` | **no — structurally** | §4, F7 |
| agent-model generalisation | **no** | single-agent design |
| cross-hardware or cross-budget generalisation | **no** | FLAW 6 |
| the agent's temperature-vs-prompt split | **no** | needs a factorial on the agent |
| the true `σ₀` at the precision needed to size a design | **not from 5 replicates** (23× span) | §7.4 |

---

## Appendix A — every number in this document, and where it came from

| number | value | source |
|---|---|---|
| per-experiment wall | 513.7 s | `.logs/baseline.log` `total_seconds` |
| training seconds | 301.8 s | same, `training_seconds` |
| baseline `val_bpb` / `num_steps` | 0.863803 / 44 | same + `results.tsv` |
| step-time mean / SD | 9.1455 s / 240.1 ms | parsed from the 44 logged steps |
| SE of mean step time | 41.8 ms (0.457 %) | 240.1/√33 |
| 44-step bucket | `dt̄ ∈ (9.0909, 9.3750]` s | 300/33 and 300/32 |
| late-training loss slope | 0.049222 nats/step (−1.74 %) | loss 3.462902@30 → 2.823026@43 |
| implied `∂bpb/∂step` proxy | ≈ 0.015 bpb | 1.74 % × 0.863803 |
| fixed-seed numerical drift | ≈ 1e-6 relative/step | baseline vs repl-1 at matched steps |
| MCQ noise floor (other DV) | SD 0.0224 | 5 reps × 20 items × 2 models, T = 0.7 |
| `t`-quantile sums | 3.7174 (df 4), 3.1949 (df 8), 5.3633 (df 2), 14.0826 (df 1) | `scipy.stats.t` |
| expected maxima | 1.867 (k=20), 2.509 (k=100), 2.689 (k=168) | 4×10⁵ Monte-Carlo draws |
| σ CI from `n` replicates | [0.599, 2.874]× at n=5 | χ² df = n−1 |
| permutation floors | 1/7 (V=3,1 block), 1/33 (V=2,5 blocks) | `1/(k!^B + 1)` |
