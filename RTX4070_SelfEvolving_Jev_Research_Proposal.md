# Research Proposal — Full Edition (Harness-First, Ecosystem-Aligned)

## RTX 4070 Single-GPU 24/7 Self-Evolving Jev Decision Model Optimization Framework

> A fully local, zero-API, multi-fidelity autonomous research framework  
> with statistically rigorous evaluation and domain-native scientific governance rules  
> **Technical posture: harness-first; parameter updates secondary; multi-objective archive**

| Field | Value |
|-------|-------|
| **Hardware Target** | Single NVIDIA RTX 4070 (12 GB GDDR6X), power-capped at 150 W |
| **Seed Model** | AgentJev-0.6B (Qwen3-0.6B backbone) + modular harness (AnyJev-class levels) |
| **Primary technical route** | **Harness evolution** (calibration, scoring pipeline, inference backend, robustness) — seconds per cycle |
| **Secondary technical route** | QLoRA / lightweight adapters — minutes per cycle, only after harness saturation |
| **Runtime Paradigm** | Fully local multi-fidelity continuous loop — zero external API calls |
| **Primary Deliverable** | Quantifiable progress curves + multi-objective archive + reproducible artifact package |
| **Duration** | 6 weeks (1 week setup + 4 weeks continuous run + 1 week validation & review) |
| **External Cost** | Electricity only (~200–350 RMB); no cloud GPU, no LLM API, no paid service |
| **Governance** | Six domain-native scientific rules; dual independent review before success claim |

---

## Abstract

This proposal designs an autonomous evolution framework for the Jev decision model under two hard constraints: a single RTX 4070 (12 GB VRAM) and zero external API dependency. The **primary technical route is harness evolution**—post-hoc calibration (AnyJev-style L0/L1/L2), scoring-pipeline modules, inference backends, and robustness transforms—rather than weight updates. For a 0.6 B scoring model that emits no text, most early and mid-horizon gains are expected from the layer around the frozen backbone. QLoRA remains a **gated secondary lever** after harness saturation.

Search is performed by a fully local multi-fidelity controller (Select → Mutate → Evaluate → Archive → Persist) that never calls a remote language model. Evaluation uses day-1 locked partitions (proxy / medium / frozen V), with medium-set keep decisions strengthened by a **regression gate**, optional **repeated measurement** for stability, and **failure-conditioned proposal bias**. The archive is **multi-objective** (Intelligence, calibration error, flip rate, latency), not scalar-only. Secondary reportable metrics include **selective coverage at fixed risk**. Six domain-native scientific governance rules bind the experiment. Success is a statistically detectable improvement on frozen V within six weeks, documented by an auditable archive and a complete reproducible package. A plateau that fails the held-out criterion is a valid negative result under the same standards.

---

## 0. Scientific Reflection and Ecosystem Alignment

### 0.1 Why harness-first, not fine-tune-first

On RTX 4070, a typical QLoRA pass over ~1,000 samples costs **minutes** and several GB VRAM. A harness mutation (AnyJev L0/L1/L2, isotonic/Platt, CUDA Graphs, option-order stabilizations) costs **seconds** and usually < 1 GB. In a 24/7 loop the throughput gap is decisive: harness-dominated schedules evaluate far more candidates under the same wall-clock and energy budget.

2025–2026 evidence supports this posture:

- Harness optimization has moved open-weight agents toward frontier-comparable pass rates at a small fraction of frontier inference cost **without changing weights**.
- For small models, adapted harnesses recover a large fraction of larger-model performance at much lower cost.
- When harness evolution and LoRA are combined, **harness-first** is safer; naive fine-tuning after harness evolution can degrade task success.
- For Jev-class models, calibration (Platt, isotonic, AnyJev-family post-hoc methods) is a central driver and does not require gradient steps through the backbone.

### 0.2 Fit to AgentJev-0.6B and the open System One ecosystem

AgentJev is a **scoring model**: one forward pass, distribution over supplied options, zero output-token decoding. The natural harness is:

1. **Calibration** — L0 (zero-label), L1 (temperature / vector), L2 (closed-form head, label-gated)  
2. **Scoring pipeline** — option ordering, prefix sharing, numerical stabilizations  
3. **Inference backend** — CUDA Graphs, kernel/precision policy  
4. **Robustness transforms** — order-invariance, light state-perturbation resistance  
5. **Optional lightweight adapters** — only when the above plateau  

As of late September 2026, open models (Plumb-4B, decider-4b v2, Cygnet, Open-Jev family, etc.) compete with or lead composite boards, while commercial Jev often still leads calibration. The plan therefore:

- Keeps AgentJev-0.6B as the primary seed under 12 GB  
- Optionally allows rare swaps to other ≤2B open System One checkpoints that fit the same VRAM envelope  
- Requires Rule-4 comparisons against current open leaders with harness vs parameter attribution  

### 0.3 Ideas adopted from concurrent work

| Idea | Source theme | Role in this plan |
|------|----------------|-------------------|
| Multi-objective archive | MoMHa-style joint objectives | Archive elites on Intelligence × ECE (or flip); report Pareto, not only scalar best |
| Regression gate on keep | Growing Harness / Harness-R1 | Keep only if medium fitness rises **and** regression probe does not drop by more than δ |
| Failure-conditioned proposals | Trace-local / failure-guided harness edit | Bias next mutation family from recent failure diagnoses and bottleneck axis |
| Coverage @ fixed risk | Selective prediction; AnyJev auto-decidable share | Secondary metric: coverage at risk ≤ 5% (or 1%) |
| Keep stability | Coverage vs specialization diagnostics | Optional 2–3× medium re-eval before keep |
| Compositional probe | Compositional harness safety failures | After keep, smoke-test combination with last K kept modules |
| Branch complementarity | Mixture of self-improving branches | Soft: MAP-Elites cells + optional parallel cool-down pools; not required for Rule-1 claim |

### 0.4 Technical posture summary

| Priority | Route | Typical cycle cost | Role |
|----------|--------|--------------------|------|
| **P0** | Harness: calib, pipeline, backend, robustness | Seconds | Vast majority of cycles |
| **P1** | QLoRA / rank–target–LR sweeps within VRAM | Minutes | After harness saturation on bottleneck axis |
| **Excluded** | Full FT, speculative decoding, multi-model committees, external APIs | — | Outside 12 GB / zero-API envelope |

---

## 1. Scientific Governance Rules

These six rules define what counts as a valid experimental outcome. Every automated cycle and human decision is bound by them. A success claim that violates any rule is void.

### Rule 1 — Pre-registered Held-Out Criterion

Before the continuous loop starts, public evaluation tasks are permanently split into a working set (proxy + medium) and a frozen held-out set **V**. Task IDs of V are written to the laboratory notebook and never used for keep/revert decisions. Success is defined solely as: the final elite beats the L1-calibrated seed on V, with the lower bound of a bootstrap 95% CI (over ≥ 3 independent seeds) strictly positive. No other metric or split may be substituted after the fact.

### Rule 2 — Fixed Local Resource Envelope

All training, evaluation, monitoring and decision logic must run on one RTX 4070 (≤ 12 GB VRAM, power limit ≤ 150 W) with no external model API, no cloud GPU, and no paid inference service. The wall-clock budget is six weeks. Gains that require relaxing these constraints are outside the claim.

### Rule 3 — End-to-End Reproducibility from Public Artifacts

The released package must allow a third party, given only an RTX 4070-class machine and the public repository, to: (1) recreate the environment from a lockfile; (2) rebuild progress curves and the multi-objective archive from the raw JSONL log alone; (3) re-evaluate the elite on frozen V and obtain a CI consistent with the reported claim. Missing seeds, silent weight patches, or undocumented manual steps invalidate the package.

### Rule 4 — Honest Placement against Concurrent Work

The final report must include a comparison table against the strongest relevant baselines at the time of writing (JevBench / open System One leaders such as Plumb-4B or decider-4b-class, commercial Jev 1.13 if comparable, recent calibration methods, ≤ 2 B PEFT under similar VRAM, concurrent single-GPU autonomous systems). Each row must note compute parity. Superiority may be claimed only on metrics the baseline reported. Gains must be **disaggregated** into harness-attributed vs parameter-attributed. Report flip rate and selective coverage@risk where measurable.

### Rule 5 — First-Class Negative Results

Every attempted mutation—kept or discarded—is appended to `autoresearch.jsonl` with scores, action, diagnostic, VRAM peak, wall time, and route. A maintained `FAILURES.md` groups dead-ends by module family (including `harness.anyjev.L2`, `robustness.flip`, `param.qlora`, `compositional:A+B`). A run that plateaus early must still ship the full failure archive.

### Rule 6 — Two Independent Checks before Any Success Claim

No success announcement until both pass and are committed:

- **(A) Protocol check** — V never used for keep/revert; CI matches pre-registration; artifact complete; comparison table with harness/param split; failure log intact  
- **(B) Replication check** — clean environment from lockfile only; `verify_elite.py` still yields positive 95% CI lower bound on V  

Both produce signed checklist files in the repository.

---

## 2. Hardware Constraint Analysis

### 2.1 Capability Envelope

RTX 4070: 12 GB GDDR6X, FP16/INT8, ~200 W TDP, sustainable core temperature typically < 68 °C under controlled load.

- Full-parameter fine-tuning of ≥ 7 B models is infeasible (~24 GB+).  
- QLoRA is viable for sub-2 B models but **expensive in wall-clock** relative to harness mutations.  
- 0.6 B models leave substantial headroom; pure harness evaluation is far cheaper still.

### 2.2 VRAM Budget

| Allocation | Budget | Rationale |
|------------|--------|-----------|
| QLoRA fine-tuning (when invoked) | ≤ 5 GB | 4-bit base + LoRA + optimizer states |
| Inference / calibration / harness eval | ≤ 4 GB | 0.6 B FP16 + calibration artifacts |
| Evaluation buffers | ≤ 3 GB | Proxy or medium task batches |
| System reserve | ≥ 2 GB | CUDA context, driver, monitoring |

*Table 1. Most cycles never enter the QLoRA row.*

### 2.3 Power and Thermal Policy

Default: **`nvidia-smi -pl 150`**. Temperature every 5 minutes via local `nvidia-smi`. Batch-size reduction above 78 °C; loop suspension above 82 °C until ≤ 70 °C. Optional night derating (23:00–07:00) to 100 W. All monitoring is local.

---

## 3. Seed Model and Baseline Harness

### 3.1 Why AgentJev-0.6B

Under 12 GB, Open-Jev-27B-class weights are impossible on-card. AgentJev-0.6B is the practical seed:

- Peak QLoRA VRAM ~3.9 GB when rarely needed  
- Community composite in the mid-60s class for kev/AgentJev-scale models; ECE on the order of ~0.07 in published small-model cards  
- ~50 ms forward; zero output-token decoding → ideal for high-throughput harness search  

Optional rare swaps: other ≤2B open System One checkpoints that load under the same VRAM budget. Backbone identity is a slow mutation, not a per-cycle default.

### 3.2 Seed Configuration

```
AgentJev-0.6B (Qwen3-0.6B backbone, permutation-equivariant scoring head)
+ Baseline harness:
    - AnyJev-style L1 post-hoc calibration (per-(model, question) temperature scaling)
    - Default scoring pipeline (shared prefix / option scoring)
    - Default inference backend (eager or simple graph)
+ Optional trainable QLoRA adapters (rank = 16, all-linear) — secondary lever only
```

### 3.3 QLoRA Recipe (Secondary Only)

When the controller opens parameter-level search:

- `seq_len = 2048`; batch = 1 × grad_accum = 16; max_samples = 1000; epochs = 1  
- `memory_budget_gb = 10.5`; Liger fused CE + activation checkpointing; APOLLO-Mini optimizer  

Do not schedule QLoRA until harness families on the bottleneck axis are flat for a cool-down window.

---

## 4. Fully Local Multi-Fidelity Research Loop

### 4.1 Design Invariants

- Zero external API  
- Deterministic, reproducible controller  
- Harness-first schedule (vast majority of cycles sub-minute)  
- Multi-fidelity evaluation (proxy → medium; frozen V only at checkpoints)  
- **Multi-objective MAP-Elites-style archive**  
- **Regression gate + optional keep stability re-eval**  
- **Failure-conditioned proposal bias**  
- Electricity-only cost (~200–350 RMB / 6 weeks)

### 4.2 Data Partitions (Locked Day 1)

| Partition | Size (approx.) | Role | When used |
|-----------|----------------|------|-----------|
| Proxy set P | ~30 tasks | Ultra-cheap filter | Every candidate |
| Medium set M | ~80 tasks | Keep/revert, effect-size, regression probe, flip, selective | After proxy pass |
| Frozen val V | ~60 tasks | Sole basis for Rule-1 success claim | Weekly checkpoints + final |
| Sealed (if any) | — | External generalization probe | Phase 3 only |

*Table 2. V never enters keep/revert logic (Rule 1).*

A fixed **regression probe** subset of M (or a frozen slice of M) is used only to test non-regression on keep—not as a free optimization target.

### 4.3 Cycle Phases

#### Phase 1 — Select (failure-conditioned)

Read archive statistics: best fitness, axis slopes, cool-downs, archive occupancy, recent harness vs param fraction, recent failure diagnoses.

Priority rules:

1. Untried **P0 harness** ideas first  
2. Ideas targeting the **current bottleneck axis** (from rolling slopes of Intelligence / Calibration / Speed / flip)  
3. **Failure-conditioned bias**: if last N failures are high flip → prefer order-invariance / L0-perm modules; if high ECE → prefer L1/L2 calibration; if high latency → prefer backend  
4. Cool-down after *k* consecutive failures of a module family  
5. **QLoRA (P1) only if** harness families on the bottleneck axis are in cool-down / saturated  

Always query `FAILURES.md` before proposing (Rule 5). Optional: rank a shortlist with one local AgentJev forward.

#### Phase 2 — Mutate

Edit harness config, calibration module, robustness transform, or (rarely) QLoRA config; git commit with parent hash and tags (`harness.*` vs `param.*`).

#### Phase 3 — Train or skip

- Harness-only: **skip training**, evaluate immediately  
- Parameter: run QLoRA under VRAM budget (typically ≤ 5 min)

#### Phase 4 — Multi-Fidelity Evaluate

1. **Proxy P** (seconds). Discard if Δ < ε_proxy.  
2. **Medium M**:  
   - Primary: fitness / effect-size test (ε_M + bootstrap 90% CI or sign test)  
   - **Regression gate**: regression-probe metric must not drop by more than δ  
   - Optional **2–3× re-eval** for keep stability; require stable sign of Δ  
   - Record flip rate and selective coverage@risk (secondary)  
3. Optional **compositional probe**: re-test combination with last K kept modules on a small probe; log failures as `compositional:A+B`  
4. Frozen V only at pre-registered checkpoints  

#### Phase 5 — Archive and Persist

- Insert into multi-objective archive (e.g. discretized **Intelligence × ECE**, or Intelligence × flip); replace cell only if dominates or improves the cell’s primary fitness under archive rules  
- Append full JSON to `autoresearch.jsonl` (`route`, wall time, VRAM, flip, coverage@risk, regression delta)  
- TensorBoard scalars; every 12 h rebuild curves, Pareto snapshot, and evolution DAG from JSONL alone  

### 4.4 State Persistence

Control state rebuilds from `autoresearch.jsonl` after any crash. No conversational context.

### 4.5 Power / Thermal

As in §2.3. Harness-heavy schedules reduce average draw relative to continuous QLoRA.

---

## 5. Progress Instrumentation

### 5.1 Objectives

1. Is fitness still rising on M? On V at checkpoints?  
2. Which axis is the bottleneck (Intelligence, Calibration, Speed, flip)?  
3. What fraction of gains came from harness vs parameter routes?  
4. What does the multi-objective archive look like (Pareto / coverage)?  
5. What is the elite lineage?

### 5.2 Components

- Primary curves: Best-on-M, Archive mean / coverage, Best lineage, Best-on-V (checkpoints)  
- Four-axis (+ flip) panel → bottleneck detector  
- Evolution DAG (grey = discard, green = keep, gold = elite); color by `route`  
- **Route attribution**: cumulative Δfitness from harness vs param  
- **Pareto / archive heatmap**: Intelligence vs ECE (or flip)  
- Secondary: selective coverage@risk over time  

### 5.3 Logging Schema (excerpt)

```json
{
  "iter": 247,
  "ts": "2026-10-15T03:42:17Z",
  "commit": "a3f8c2d",
  "parent": "7b1e4a9",
  "route": "harness",
  "modules": {"cal": "L1_vector", "robust": "order_invar"},
  "proxy": 68.1,
  "medium": 71.4,
  "fitness_M": 71.4,
  "ece": 0.068,
  "flip_rate": 0.04,
  "coverage_at_5pct_risk": 0.52,
  "regression_delta": 0.0,
  "action": "keep",
  "ci90_lo": 0.3,
  "wall_s": 12.4,
  "peak_vram_mb": 890,
  "diagnosis": "vector temperature improved ECE; flip stable; regression gate ok"
}
```

Local TensorBoard only; matplotlib rebuild from JSONL every 12 h. No remote telemetry.

---

## 6. Modular Candidate-Idea Pool (Harness-First)

### 6.1 Priority Ranking

| Pri. | Candidate Idea | Type | Cycle cost | VRAM |
|------|----------------|------|------------|------|
| **P0** | AnyJev-style L0 / L1 / vector temperature | Harness · Calib. | Seconds | < 1 GB |
| **P0** | L2 closed-form head (gated by ≥ 100–500 labels) | Harness · Calib. | Seconds–minutes fit | < 1 GB |
| **P0** | Isotonic / Platt post-hoc maps (label-gated) | Harness · Calib. | Seconds | < 1 GB |
| **P0** | Scoring-pipeline variants (prefix share, option order, stabilizations) | Harness · Pipeline | Seconds | < 1 GB |
| **P0** | Order-invariance / flip-reduction transforms | Harness · Robust | Seconds | < 1 GB |
| **P0** | CUDA Graphs / inference-backend swaps | Harness · Infer. | ~5 min wall | < 1 GB |
| **P1** | QLoRA adapter train + merge (task families) | Param. | ≤ 30 min | ~4 GB |
| **P1** | Rank / target-module / LR sweeps within VRAM | Param. | ≤ 15 min | ≤ 5 GB |
| **P2** | Prefix-tree / advanced batching | Harness · Infer. | ~5 min | < 1 GB |
| **P3** | Speculative decoding / multi-model committees | — | — | **Excluded** |

*Table 3. Harness mutations dominate; QLoRA is gated.*

### 6.2 Harness Modules (Primary)

- **Calibration:** L0, L1, vector temperature, L2 heads, isotonic/Platt with sample-size gates  
- **Pipeline:** shared-prefix scoring, numerical stability, option-permutation robustness  
- **Backend:** CUDA Graphs, precision policy, kernel selection  
- **Robustness:** order/phrasing invariance; optional light state-perturbation probes  

### 6.3 QLoRA (Secondary)

Only when harness slopes on the bottleneck axis are flat for a cool-down window. Procedure: train → merge → proxy → medium keep (with regression gate) → else `git reset`. Log under `param.qlora.*`.

### 6.4 Diversity and Safety Safeguards

- Multi-objective MAP-Elites grid (e.g. 5×5 on Intelligence × ECE)  
- Module-family cool-downs  
- Offline idea-injection file  
- Forced exploration of under-filled archive cells  
- Mandatory `FAILURES.md` query  
- Compositional smoke after keep  
- Optional soft branching via parallel cool-down pools (calib vs robust vs backend)  

---

## 7. Experimental Protocol

### 7.1 Phase 1 — Baseline and Feasibility (Week 1)

**Objective.** Baselines, lock V, measure harness vs QLoRA cycle costs, bootstrap robustness metrics.

**Method.** Evaluate: raw logits, L0, L1, selected L2/harness variants, one QLoRA smoke run on P/M/V. Record wall time, peak VRAM, ECE, flip rate, coverage@risk. Lock V task IDs. Define regression probe. Bootstrap `FAILURES.md`. Confirm JSONL recovery after kill + reboot.

**Success gate.** L1 (or better harness calib) reduces ECE; harness cycle ≪ QLoRA cycle; recovery works; V locked.

### 7.2 Phase 2 — Continuous Local Loop (Weeks 2–5)

**Objective.** Maximize evaluated candidates under harness-first schedule; collect multi-objective progress data.

**Method.** ~672 h at ≤ 150 W. Soft target: high cycle count via harness dominance. Weekly frozen-V checkpoints only. Track route mix, archive coverage, flip, coverage@risk, regression-gate reject rate.

**Null expectation.** Early gains mostly Calibration / flip / selective coverage from harness; mid-run still mostly harness; late plateau may open sparse QLoRA. Plateau is informative (Rule 5).

### 7.3 Phase 3 — Generalization, Artifact, Dual Review (Week 6)

**Objective.** Final CI on V, attributed comparison table, artifact, Reviews A & B.

**Method.** Elite + top archive cells on V with ≥ 3 seeds; bootstrap 95% CI. Sealed half if available. Transfer on BANKING77 / Typed Decisions (and optional MASSIVE-style intent if local data available). **Rule-4 table** includes open leaders + harness/param split + flip + coverage@risk where comparable. Assemble artifact. Pass Review A and Review B before any success claim.

**Pre-registered success (Rule 1).** CI lower bound of (elite − L1 seed) on V > 0; public–sealed gap < 5 points if measurable.

**Secondary reportables (not substitutes for Rule 1):** flip rate, coverage@risk, route attribution, archive coverage.

---

## 8. Artifact Package and Review Gates

### 8.1 Required Contents (Rule 3)

- Environment lockfile  
- Harness configs, calibration modules, optional QLoRA scripts, controller  
- `seeds.txt`  
- Full `autoresearch.jsonl` (with `route`, flip, coverage@risk, regression_delta)  
- `FAILURES.md` by module family  
- `scripts/rebuild_curves.py`, `rebuild_archive.py`, `verify_elite.py`  
- Elite checkpoint(s) + top multi-objective archive cells  
- Comparison table (CSV + markdown): open leaders, harness/param split, compute-parity notes  
- Signed `REVIEW_A.md`, `REVIEW_B.md`

### 8.2 Review A — Protocol Integrity

- [ ] V IDs match day-1 lock; no V in keep/revert history  
- [ ] CI matches pre-registration; ≥ 3 seeds  
- [ ] ≤ 150 W policy; no API/network in controller path  
- [ ] JSONL has keeps + discards; `FAILURES.md` complete  
- [ ] Comparison table present with **harness vs param attribution**  
- [ ] Regression gate and archive rebuild documented  
- [ ] `verify_elite.py` runs from clean install  

### 8.3 Review B — Clean Replication

- [ ] Environment from lockfile only  
- [ ] `verify_elite.py` yields positive 95% CI lower bound on V  
- [ ] Curves and archive rebuild from JSONL alone  
- [ ] No undocumented dependency or weight patch  
- [ ] Checklist signed and committed before success announcement  

---

## 9. Resource and Cost Estimates

### 9.1 Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Phase 1 | Baselines, lock V, harness vs QLoRA cost, flip/coverage baselines, recovery | 1 week |
| Phase 2 | 24/7 harness-first loop + weekly V checkpoints | 4 weeks |
| Phase 3 | Multi-seed final, attributed comparison table, artifact, dual review | 1 week |
| **Total** | Electricity only · no API · no cloud GPU | **6 weeks** |

*Table 4. Timeline.*

### 9.2 Cost Model

Sole operating cost: electricity under 150 W ≈ **200–350 RMB** ($28–$49) for six weeks. Harness-heavy schedules reduce average draw. Night derating to 100 W optional. Daily power logged locally.

---

## 10. Risks and Mitigations

| Risk | P | I | Mitigation |
|------|---|---|------------|
| No detectable gain on frozen V | M | H | Publish negative result + full failure archive (Rule 5) |
| Overfit to proxy/medium | H | H | V locked; multi-seed CI; Review B |
| False keep from noisy medium scores | M | M | Effect-size + optional 2–3× re-eval; regression gate |
| Harness search saturates early | H | M | Then open P1 QLoRA; log saturation in FAILURES.md |
| Module A+B compositional failure | M | M | Compositional probe after keep; FAILURES.md tags |
| QLoRA burns budget with no gain | M | M | Gate on harness cool-down; cap param-cycle fraction |
| Incomplete artifact | M | H | Review A blocks claim |
| Unfair / missing baseline table | M | M | Rule 4 + open leaders + harness/param split |
| OOM on rare QLoRA | M | M | 10.5 GB budget; Liger; CKPT; auto batch reduce |
| Thermal throttling | M | M | 150 W; derate > 78 °C; suspend > 82 °C |
| Controller crash | L | M | State rebuild from JSONL |
| Scientific overclaim | M | H | Pre-registered criterion; separate harness vs param reporting |

*Table 5. Risk register.*

---

## 11. Conclusion and Claim Boundary

This proposal adopts a **harness-first** technical posture because cycle-cost asymmetry and 2025–2026 evidence both favor evolving the layer around a frozen 0.6 B scoring model before spending scarce GPU hours on QLoRA. Concurrent ideas—multi-objective archives, regression gates, failure-conditioned search, selective coverage@risk, and keep stability—are folded into the loop without relaxing the single-GPU or zero-API constraints.

The only success claim permitted is:

> **Under a fixed 12 GB / 150 W local budget, zero external API cost, and a six-week horizon, a multi-fidelity deterministic controller—predominantly evolving harness modules (calibration, scoring pipeline, inference backend, robustness), with optional sparse QLoRA—produced a configuration whose frozen-validation fitness exceeds the L1-calibrated seed by a statistically detectable margin, supported by a complete reproducible artifact package, an explicit comparison with concurrent baselines that attributes gains to harness vs parameters, a multi-objective archive and failure log, and two independent review checklists.**

Open-ended capability discovery is not claimed. Calibration, robustness, and systems-level harness gains are expected to dominate; parameter-level gains, if any, are reported separately. Secondary metrics (flip rate, coverage@risk, archive coverage) enrich the report but do not replace Rule 1.

If the criterion is met, the result shows that continuous, measurable optimization of decision models is accessible on a desk-class GPU without cloud or paid APIs, and that **harness search is the efficient default route** under that constraint. If not, the archive will show where harness families and parameter families each plateaued—itself a useful community result.
