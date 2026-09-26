# OASP — Optimization-as-Search Program

**A pre-registered research protocol on the landing form of LLM optimization strategies**

| | |
|---|---|
| **Program** | OASP — *Optimization-as-Search Program* |
| **Object of study** | the **landing form (落地形态)** of reported LLM optimization strategies, not the benchmarks they are measured on |
| **Primary brief** | `archive/nhb_2026/Problem.docx` (md5 `c49956fd887fb16ec1da1b4012e4512d`) |
| **Status** | pre-registered before data collection |
| **Pilot run** | `runs/pilot_v1/` — labelled **exploratory**; cannot support confirmatory claims (see §6) |
| **Supersedes** | the NHB LLM-mistranslation workspace, archived at `archive/nhb_2026/` |

---

## §0 Why the object is the landing form

The brief is explicit that the target of criticism is **not** the benchmark and **not** prompting in general:

> 你的批评对象不是 benchmark 本身，而是"所谓 LLM 优化策略"的落地形态：CoT、self-consistency、ReAct、ToT 等领域不断提出新策略，但最终往往都固化为一个启发式的静态 prompt 模板，再在静态 benchmark 上测一次。

So the object is a **degeneration**: `strategy S` → `one hand-authored static template p₀` → `one benchmark score Â = A(p₀, m)`.

Two distinct things must be named, and they must **not share a word root** (brief, ¶7):

| Kind | Requirement | Name chosen | Alternative (retained) |
|---|---|---|---|
| **Effect** — observable, quantifiable, describes the phenomenon | 策略最终坍缩/固化为静态提示 | **HPFE — Heuristic Prompt Fixation Effect** | PBCE — Prompt–Benchmark Coupling Effect |
| **Insight** — falsifiable, explains a mechanism, guides method | 优化本质是提示空间中的搜索 | **OASP — Optimization-as-Search Principle** | — |

**Why HPFE over PBCE.** PBCE's virtue is that it foregrounds the *interaction* (score = f(prompt, benchmark, model)). Its defects are decisive for our purpose: "coupling effect" already carries meanings in physics and statistics, and it does **not** encode either *search* or *non-optimality* — the two things the brief actually wants to assert. HPFE names the mechanism we can measure: strategies do not fail to search; they **fixate** on a hand-sampled point. "Heuristic" is also the correct modifier — a static prompt is not random, it is a *human heuristic choice*, which is precisely why it can be arbitrarily far from optimal while looking principled.

**Roots are disjoint**: HPFE (fixation) vs OASP (search). No paper-level confusion.

---

## §1 Formalization

Let

- `X = {x₁ … x_K}` — a **frozen** item set, with gold labels;
- `s : 𝒴 × 𝒳 → {0,1}` — a **mechanical** scorer (exact match on a constrained answer token);
- `𝒫` — the set of instruction strings that are *semantically equivalent* to the task specification (all ways of asking the same thing);
- `f_m` — target model `m`;
- `A(p, m) = (1/K) Σᵢ s( f_m(xᵢ | p), xᵢ )` — accuracy.

Reported practice computes a point estimate from a single element `p₀ ∈ 𝒫`:

```
Â = A(p₀, m)                      ← one sample, treated as the model's property
```

**OASP claim.** `Â` is a *single draw* from the distribution of `A(·, m)` over `𝒫`. The reported **ranking** of models is likewise a single draw of a ranking-valued random variable.

**HPFE claim.** The distinguishing move of the strategies under study is not that they search `𝒫` and stop early — it is that they never search it at all: `S ↦ p₀` is a *fixation*, `p₀` is a heuristic point, and `Â` carries no search-derived guarantee.

### Hypotheses (fixed before data collection)

| ID | Claim | H₀ |
|---|---|---|
| **H1** *Sampling* | `Range_{p∈𝒫_samp} A(p,m) > 2 × noise-floor` for ≥2 models, **and** the induced model ranking is not invariant (mean pairwise Kendall τ < 0.95) | prompt-induced spread ≤ noise floor; τ = 1 |
| **H2** *Non-optimality* | a search loop over `𝒫` attains `A(p*, m) > max_{p∈𝒫_samp} A(p,m)` on a **held-out item split**, beyond the noise floor | no held-out gain beyond the sampled upper tail |
| **H3** *Death valleys* | reason-coded discards are not uniformly bad: they separate into discrete failure modes with distinct score signatures | discard classes are score-indistinguishable |

### Falsifiers (each hypothesis must be able to lose)

- **H1 fails** if all sampled variants land within the noise floor and model rankings are stable (τ ≈ 1). This would *falsify the practical importance of HPFE*, even though strategies still fixate — i.e. it would show the fixation is benign.
- **H2 fails** if search over `𝒫` yields no held-out gain over the sampled upper tail. This falsifies OASP **as a claim about measurable benefit**; it would reduce OASP to a philosophical restatement ("all optimization is search" — the brief's own stated weakness: 太宽泛, 可能被批评为同义反复).
- **H3 fails** if discard classes are score-indistinguishable, in which case "death valleys" is decorative language and must be dropped.

---

## §2 The five design constraints from the brief, each mapped to an implementation

| # | Brief constraint (verbatim intent) | Implementation |
|---|---|---|
| 1 | 明确"随机解"的操作化定义 — generate N semantically equivalent prompts, observe the score distribution | `oasp/prompt_space.py` (N=32 variants from an auditable surface-feature factorial **and** an LLM-optimizer condition); `oasp/stats.py` M1/M2 |
| 2 | 把 fallback 机制量化 — record each discard's reason; failure modes are the analysis material ("死亡谷") | `oasp/landscape.py` reason-coded discard ledger (codes assigned **blind to score**); `oasp/stats.py` M8 |
| 3 | 优化器模型与目标模型解耦 — the optimizing agent must not self-evaluate | `oasp/config.py::assert_decoupled()` — hard failure if `optimizer_model ∈ target_models` |
| 4 | 警惕局部最优 — periodic stochastic cold-start, population diversity, forced large mutation at plateau | `oasp/landscape.py` — three named mechanisms, each logged per generation |
| 5 | LANDSCAPE — hill-climb curve (survivors) **plus** fallback sampling points | `oasp/landscape.py` emits both streams; `oasp/report.py` renders both |

**Constraint 3 is enforced in code, not prose.** Decoupling is the single easiest requirement to violate silently (an agent asked to improve a prompt and also to score it will converge on prompts it happens to score well), so it is a load-time assertion rather than a documented guideline.

---

## §3 Design

**Crossed factorial.** Factors: **prompt** (N variants, random), **model** (M targets, random), **item** (K, random), **split** (dev / test, fixed).

Every `(prompt, model, item)` cell is observed once, at `temperature = 0`, `seed = 42`, `num_predict` fixed.

**Mechanical scoring only.** No LLM judge anywhere in the scoring path — the brief forbids subjective scoring (机械化的数值评测, 不允许主观评分). The scorer is a regex on a constrained answer token, exact-matched against a frozen gold label. This is a **hard** constraint: it means the scored object is *answer extraction*, and prompt variants that fail to produce a parseable answer score 0. That is intentional — it makes format-following part of the measured effect rather than a hidden covariate.

**Noise floor.** A separate sub-study re-runs one fixed prompt at `temperature > 0` with `R = 5` seeds on a random item subset, giving the irreducible stochastic SD. H1's decision rule is a *ratio* against this floor, so a large floor can only make H1 harder to support.

**Paired item design.** Because all variants see the *same* items, the item-level flip statistic (M2) and the paired held-out test (M7) use within-item comparisons, which removes item-difficulty variance.

**Capability-floor screen.** Before any inference, each target model must clear a floor: `A(p₀, m) > chance` with the lower bound of its CI above chance. Models below the floor are reported but excluded from H1/H2 — otherwise accuracy compression at the floor would masquerade as prompt-insensitivity. (This is the design law carried over from the archived program: *separate the capability floor*.)

---

## §4 Metrics — fixed before data collection

| ID | Metric | Definition | Inference |
|---|---|---|---|
| **M1** | Prompt spread | Range and CV of `A(p, m)` across the N variants, per model | item-level bootstrap CI (B=2000) |
| **M2** | **IPS** — Item-level Prompt Sensitivity | fraction of items whose correctness is **not** constant across the N variants (modal-answer disagreement) | point estimate + item bootstrap CI |
| **M3** | η² decomposition | η²_prompt, η²_model, η²_interaction from the crossed design | permutation (prompt-label shuffle within model) + bootstrap |
| **M4** | τ̄ | mean pairwise Kendall τ between model rankings induced by different variants | bootstrap CI |
| **M5** | Flip rate | fraction of (variant-pair × model-pair) with an ordering reversal | exact count + CI |
| **M6** | Noise floor | SD of `A` across seeds at temp > 0 on a fixed prompt | direct estimate |
| **M7** | Held-out gain | `A(p*, m) − A(p₀, m)` on the **test** split | McNemar exact + paired bootstrap |
| **M8** | Death-valley separation | score separation between blind reason-coded discard classes | Kruskal–Wallis + ε² |

**M2 is the primary operationalization of "静态 prompt 只是随机采样点".** Under exact prompt-invariance every item's correctness is *identical* across all variants, so M2 has a **degenerate null at exactly 0**. Any non-zero M2 is therefore direct, assumption-light evidence of prompt-dependence — it does not need a null distribution, only a CI. This is why M2 leads and M1 supports.

> **DV note for the construct-B study (added 2026-09-26; FLAW 6 of [`docs/critical_review_2026-09-26.md`](docs/critical_review_2026-09-26.md)).**
> The project's primary object is `nanochat`'s `program.md`; its dependent variable is written
> **`val_bpb@300s`** — validation bits per byte read at a **fixed 300-second wall-clock training
> budget**, hence compute-normalised. Cross-hardware comparisons and comparisons against
> fixed-step runs are **void**.

---

## §5 Decision rules (pre-registered)

- **H1 supported** iff, for ≥2 of M target models: `M1_range > 2 × M6` **and** `M4 < 0.95`.
- **H2 supported** iff `M7 > M6` **and** McNemar exact `p < 0.05` on the held-out split.
- **H3 supported** iff M8 Kruskal–Wallis `p < 0.05` **and** ε² > 0.06.
- Multiplicity across models × benchmarks is controlled with Benjamini–Hochberg (q = 0.05).
- Any hypothesis that fails is **reported as failed**, and the corresponding claim is dropped from the manuscript — HPFE must survive on H1 alone if H2/H3 fail.

---

## §6 Power and sizing — stated honestly

| | Pilot (`runs/pilot_v1`) | Confirmatory run |
|---|---|---|
| Items (dev/test) | 60 / 60 | 60 / **≥150** |
| Prompt variants N | 32 | 32 (+ LLM condition) |
| Target models M | 3 | ≥6 |
| Seeds for noise floor R | 5 | 5 |
| Label | **exploratory** | confirmatory |

**The pilot cannot support confirmatory claims.** H2 needs enough *held-out* items to detect a ~10-point paired accuracy gain; with paired binary outcomes and ~20 % discordance, 80 % power at α = 0.05 requires roughly `K_holdout ≈ 150`. The pilot runs `K_holdout = 60` and is therefore **exploratory for H2** and hypothesis-generating for H3. H1 is a variance-ratio comparison, not a small-effect test, and is estimated with a bootstrap CI in the pilot — it is reported as *provisional* until the confirmatory run.

Directory names must keep this distinction: **never** cite a `pilot_*` number in a manuscript as confirmatory.

---

## §7 Threats to validity — to be carried into the paper verbatim

1. **`𝒫` is not enumerable.** The N variants sample *a generator's* distribution over wordings, not `𝒫` itself. Sampled spread is therefore a **lower bound** on the true spread — good for H1's direction, bad for any claim about the *magnitude* of the optimum gap.
2. **Gold-label noise.** ARC-family benchmarks contain known label errors. Label noise inflates both the noise floor (M6) and the measured spread (M1), so it is *conservative* for a range-vs-floor comparison and must be acknowledged as a ceiling on achievable accuracy.
3. **`temperature = 0` is not guaranteed deterministic.** Batching, hardware, and kernel non-determinism can perturb greedy decoding. A **determinism check** (the same prompt, run twice, on a 20-item subset, must give byte-identical answers) is a prerequisite for interpreting the whole matrix; any mismatch is reported and the discrepancy is added to the noise floor.
4. **Capability floor.** Small target models may sit at chance on hard benchmarks, compressing accuracy and masking prompt effects. Mitigated by running an easier and a harder sibling benchmark and by the §3 floor screen — not by deleting the hard one.
5. **The optimizer is a confound.** The prompt-generating model was itself trained on literature that uses static templates, so its prior over `𝒫` is not neutral. The deterministic surface-feature generator is therefore the *primary* condition and the LLM generator a secondary one; divergence between them is reported, not averaged away.
6. **"Semantically equivalent" is an assumption.** Variants are equivalent by *construction* (they differ only in surface features we enumerate), not by proof. The surface-feature factorial makes the assumption auditable: each variant is a point in an explicit feature space, so a reader can see exactly what was varied.

---

## §8 Deliverables

1. `RESEARCH_PROTOCOL.md` — this document (pre-registration).
2. `oasp/` — the harness (prompt space → runner → mechanical evaluator → statistics → landscape).
3. `runs/pilot_v1/` — raw JSONL + manifest + hashes.
4. `analysis/pilot_v1/` — M1–M8 with CIs, tables, figures.
5. Confirmatory run plan and manuscript skeleton (title/abstract frozen to the claims that survive).

---

## §9 Reproducibility contract

- Every run writes `manifest.json`: git commit, python version, model digests, benchmark SHA-256, N/M/K, temperature, seed, prompt-variant feature vectors.
- Raw model outputs are append-only JSONL keyed by a stable cell id `(split, model, variant, item)`; re-running resumes rather than overwrites.
- Benchmark items are frozen to `benchmarks/<key>/items.jsonl` once, with the source parquet's SHA-256 recorded — later HF revisions cannot silently change the stimulus set.
- All thresholds in §5 are copied into `configs/oasp.yaml` at run start and hashed into the manifest, so a post-hoc threshold change is detectable.
