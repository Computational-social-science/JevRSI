# Research Proposal — Full Edition (Harness-First Revision)

## RTX 4070 Single-GPU 24/7 Self-Evolving Jev Decision Model Optimization Framework

> A fully local, zero-API, multi-fidelity autonomous research framework with statistically rigorous evaluation and domain-native scientific governance rules  
> **Technical posture: harness-first; parameter updates secondary**

| Field | Value |
|-------|-------|
| **Hardware Target** | Single NVIDIA RTX 4070 (12 GB GDDR6X), power-capped at 150 W |
| **Seed Model** | AgentJev-0.6B (Qwen3-0.6B backbone) + AnyJev L1 + modular harness |
| **Primary technical route** | **Harness evolution** (calibration, scoring pipeline, inference backend) — seconds per cycle |
| **Secondary technical route** | QLoRA / lightweight adapters — minutes per cycle, only after harness saturation |
| **Runtime Paradigm** | Fully local multi-fidelity continuous loop — zero external API calls |
| **Primary Deliverable** | Quantifiable progress curves + reproducible artifact package |
| **Duration** | 6 weeks (1 week setup + 4 weeks continuous run + 1 week validation & review) |
| **External Cost** | Electricity only (~200–350 RMB); no cloud GPU, no LLM API, no paid service |
| **Governance** | Six domain-native scientific rules; dual independent review before success claim |

---

## Abstract

This proposal designs an autonomous evolution framework for the Jev decision model under two hard constraints: a single RTX 4070 (12 GB VRAM) and zero external API dependency. After scientific reflection on cycle cost and recent evidence, the **primary technical route is harness evolution**—post-hoc calibration, scoring-pipeline modules, and inference-backend engineering—rather than weight updates. For a 0.6 B scoring model that emits no text, most early and mid-horizon gains are expected from the layer *around* the frozen backbone. QLoRA remains available as a **secondary, lower-frequency lever** once harness search saturates. Search is performed by a fully local multi-fidelity controller (Select → Mutate → Evaluate → Archive → Persist) that never calls a remote language model. Evaluation uses a three-way partition locked on day 1: proxy, medium (keep/revert), and frozen held-out V (claims only). A MAP-Elites archive and exhaustive failure logging maintain diversity. Six domain-native scientific governance rules bind the experiment. The objective is a statistically detectable improvement on frozen V within six weeks, documented by an auditable archive and a complete reproducible package. A plateau that fails the held-out criterion is a valid negative result under the same standards.

---

## 0. Scientific Reflection: Why Harness-First, Not Fine-Tune-First

### 0.1 The cost asymmetry

On RTX 4070, a typical QLoRA pass over ~1,000 samples costs on the order of **minutes** and occupies several GB of VRAM for the duration of training. A harness mutation—AnyJev L0/L1 temperature scaling, isotonic/Platt post-processing, CUDA Graphs capture, batching policy, scoring-head post-transform—costs **seconds** and usually < 1 GB. In a 24/7 loop the throughput difference is decisive: a harness-dominated schedule can complete **orders of magnitude more evaluated candidates** in the same wall-clock and energy budget.

### 0.2 What 2025–2026 evidence suggests

Recent work on agent systems consistently finds that the **harness** (orchestration, prompts/tools, context management, post-processing, inference scaffolding) is a first-class lever, often larger than swapping models of similar class:

- Harness optimization has moved open-weight agents from low teens to frontier-comparable pass rates at a few percent of frontier inference cost **without changing weights**.
- For small models, adapted harnesses can recover a large fraction of larger-model performance at ~90% lower cost.
- When harness evolution and LoRA-style fine-tuning are combined, **harness-first** is the safer order; naive fine-tuning on expert trajectories *after* harness evolution can *degrade* task success, whereas on-policy correction under the evolved harness can still help.
- For Jev-class models specifically, calibration (Platt, isotonic, AnyJev-family post-hoc methods) is a central performance driver and does not require gradient steps through the backbone.

### 0.3 Fit to AgentJev-0.6B

AgentJev is a **scoring model**: one forward pass, probability distribution over supplied options, zero output-token decoding. The natural “harness” is therefore:

1. **Calibration layer** — L0/L1 temperature scaling, vector temperature, isotonic/Platt maps, per-(model, question) transforms  
2. **Scoring pipeline** — option ordering, prefix sharing, batching, numerical stabilizations  
3. **Inference backend** — CUDA Graphs, kernel choices, precision policy  
4. **Optional lightweight adapters** — only when the above plateau

Treating QLoRA as the *default* mutation would burn most of the six-week budget on low-throughput cycles and under-explore the high-throughput harness space.

### 0.4 Revised technical posture

| Priority | Route | Typical cycle cost | Role in the loop |
|----------|--------|--------------------|------------------|
| **P0 (primary)** | Harness: calibration, pipeline, inference backend | Seconds | Vast majority of cycles |
| **P1 (secondary)** | QLoRA / rank-and-target sweeps within VRAM | Minutes | After harness saturation or for targeted Intelligence gains |
| **Excluded** | Full FT, speculative decoding, multi-model committees | — | Incompatible with 12 GB / single-card envelope |

This posture maximizes evaluated candidates under Rule 2 (fixed local resource envelope) and aligns with multi-fidelity evaluation: cheap harness mutations are filtered on proxy P; expensive QLoRA is invoked only when the controller has evidence that harness search has stalled on the relevant axis.

---

## 1. Scientific Governance Rules

These six rules define what counts as a valid experimental outcome. They are written for an autonomous, single-GPU, offline ML optimization experiment. Every automated cycle and every human decision is bound by them. A success claim that violates any rule is void.

### Rule 1 — Pre-registered Held-Out Criterion

Before the continuous loop starts, the public evaluation tasks are permanently split into a working set (proxy + medium) and a frozen held-out set **V**. Task IDs of V are written to the laboratory notebook and never used for keep/revert decisions. Success is defined solely as: the final elite beats the L1-calibrated seed on V, with the lower bound of a bootstrap 95% CI (over ≥ 3 independent seeds) strictly positive. No other metric or split may be substituted after the fact.

### Rule 2 — Fixed Local Resource Envelope

All training, evaluation, monitoring and decision logic must run on one RTX 4070 (≤ 12 GB VRAM, power limit ≤ 150 W) with no external model API, no cloud GPU, and no paid inference service. The wall-clock budget is six weeks. Any improvement that depends on relaxing these constraints is outside the scope of the claim.

### Rule 3 — End-to-End Reproducibility from Public Artifacts

The released package must allow a third party, given only an RTX 4070-class machine and the public repository, to: (1) recreate the environment from a lockfile; (2) rebuild progress curves and the MAP-Elites archive from the raw JSONL log alone; (3) re-evaluate the elite checkpoint on frozen V and obtain a CI consistent with the reported claim. Missing seeds, silent weight patches, or undocumented manual steps invalidate the package.

### Rule 4 — Honest Placement against Concurrent Work

The final report must include a comparison table against the strongest relevant published baselines available at the time of writing (JevBench leaders, recent calibration methods, harness-optimization results on small models, ≤ 2 B PEFT under similar VRAM, and any known single-GPU autonomous experiment systems). Each row must note whether compute was matched. Superiority may be claimed only on metrics the baseline actually reported. Gains must be **disaggregated** into harness-attributed vs parameter-attributed.

### Rule 5 — First-Class Negative Results

Every attempted mutation—kept or discarded—is appended to `autoresearch.jsonl` with scores, action, diagnostic, VRAM peak and wall time. A maintained `FAILURES.md` groups dead-ends by module family (including “QLoRA: no Intelligence gain after N trials”) so that the controller and future operators can avoid repeating them. A run that plateaus early is still required to ship the full failure archive; suppression of negative outcomes is not permitted.

### Rule 6 — Two Independent Checks before Any Success Claim

No success announcement is allowed until both of the following have passed and been committed:

- **(A) Protocol check** — frozen V was never used for keep/revert; CI formula matches pre-registration; artifact complete; comparison table present with harness/parameter split; failure log intact  
- **(B) Replication check** — from a clean environment built only from the lockfile, re-running the elite evaluation on V still yields a positive CI lower bound  

Both checks produce signed checklist files in the repository.

> These rules are enforceable by the local controller and by the final verification scripts. They exist to make positive and negative outcomes equally publishable and checkable.

---

## 2. Hardware Constraint Analysis and Optimization Strategy

### 2.1 Capability Envelope of the RTX 4070

The RTX 4070 provides 12 GB GDDR6X, full CUDA core complement, native FP16/INT8 support, approximately 200 W TDP, and sustained core temperatures controllable below 68 °C under load.

- **Full-parameter fine-tuning is infeasible** for ≥ 7 B models (~24 GB+).
- **QLoRA remains viable** for sub-2 B models but is *expensive in wall-clock* relative to harness mutations.
- **0.6 B models leave substantial headroom**: community measurements show Qwen2.5-1.5B LoRA rank-32 training peaking at ~3.9 GB / ~98 s for 123 examples. AgentJev-0.6B under QLoRA leaves margin for concurrent evaluation; pure harness evaluation is far cheaper still.

### 2.2 Hierarchical VRAM Budget

| Allocation | Budget | Rationale |
|------------|--------|-----------|
| QLoRA fine-tuning (when invoked) | ≤ 5 GB | 4-bit base + LoRA + optimizer states |
| Inference / calibration / harness eval | ≤ 4 GB | 0.6 B FP16 + calibration artifacts |
| Evaluation buffers | ≤ 3 GB | Proxy or medium task batches |
| System reserve | ≥ 2 GB | CUDA context, driver, monitoring |

*Table 1. Concurrent VRAM budget. Most cycles never enter the QLoRA row.*

### 2.3 Power and Thermal Policy

Default: **`nvidia-smi -pl 150`**. Temperature sampled every 5 minutes via local `nvidia-smi`. Batch-size reduction above 78 °C; loop suspension above 82 °C until ≤ 70 °C. Optional night derating (23:00–07:00) to 100 W. All monitoring is local; no network service.

---

## 3. Seed Model and Baseline Harness

### 3.1 Why AgentJev-0.6B

Under 12 GB, Open-Jev-27B is impossible (~54 GB FP16 weights alone). AgentJev-0.6B is the practical seed:

- Peak QLoRA VRAM ~3.9 GB when rarely needed  
- Community composite ≈ 66.4% (kev-0.6B class); ECE ≈ 0.0712  
- ~50 ms forward; zero output-token decoding → ideal for high-throughput harness search  

### 3.2 Seed Configuration

```
AgentJev-0.6B (Qwen3-0.6B backbone, permutation-equivariant scoring head)
+ Baseline harness:
    - AnyJev L1 post-hoc calibration (per-(model, question) temperature scaling)
    - Default scoring pipeline (shared prefix / option scoring)
    - Default inference backend (eager or simple graph)
+ Optional trainable QLoRA adapters (rank = 16, all-linear) — secondary lever only
```

### 3.3 QLoRA Recipe (Secondary Lever Only)

When the controller decides to invoke parameter-level search:

- `seq_len = 2048`; batch = 1 × grad_accum = 16; max_samples = 1000; epochs = 1  
- `memory_budget_gb = 10.5`; Liger fused CE + activation checkpointing; APOLLO-Mini optimizer  

Mutations may alter rank, target modules, sample count or LR only within the VRAM envelope. The controller should not schedule QLoRA until harness families show flat slopes on the relevant axis for a cool-down window.

---

## 4. Fully Local Multi-Fidelity Research Loop

### 4.1 Design Invariants

- **Zero external API**  
- **Deterministic controller** (reproducible selection and keep/revert)  
- **Harness-first schedule** (vast majority of cycles are sub-minute)  
- **Multi-fidelity evaluation** (proxy → medium; frozen V only at checkpoints)  
- **MAP-Elites archive** over (Intelligence, Calibration) bins  
- **Electricity-only cost** (~200–350 RMB / 6 weeks)

### 4.2 Data Partitions (Locked Day 1)

| Partition | Size (approx.) | Role | When used |
|-----------|----------------|------|-----------|
| Proxy set P | ~30 tasks | Ultra-cheap fitness estimate | Every candidate |
| Medium set M | ~80 tasks | Keep/revert + effect-size test | After proxy pass |
| Frozen val V | ~60 tasks | Sole basis for success claim | Weekly checkpoints + final |
| Sealed (if any) | — | External generalization probe | Phase 3 only |

*Table 2. V never enters keep/revert logic (Rule 1).*

### 4.3 Cycle Phases

#### Phase 1 — Select

Read archive statistics (best fitness, axis slopes, cool-downs, archive occupancy, fraction of recent cycles that were harness vs QLoRA). Priority rules:

1. Untried **P0 harness** ideas first  
2. Ideas targeting the current bottleneck axis  
3. Cool-down after *k* consecutive failures of a module family  
4. **QLoRA (P1) only if** harness families on the bottleneck axis are in cool-down / saturated  

Optional: rank a shortlist with one local AgentJev forward. Always query `FAILURES.md` before proposing (Rule 5).

#### Phase 2 — Mutate

Edit harness config, calibration module, or (rarely) QLoRA training config; git commit with parent hash and module tags (`harness.*` vs `param.*`).

#### Phase 3 — Train or skip

- Harness-only mutations: **skip training** (evaluate immediately)  
- Parameter mutations: run QLoRA under VRAM budget (typically ≤ 5 min)

#### Phase 4 — Multi-Fidelity Evaluate

1. Proxy P (seconds). Discard if Δ < ε_proxy  
2. Medium M with ε_M + bootstrap 90% CI / sign test for keep  
3. Frozen V only at pre-registered checkpoints  

#### Phase 5 — Archive and Persist

MAP-Elites update on discretized (Intelligence, Calibration). Append full JSON to `autoresearch.jsonl` (including `route: harness|param`). TensorBoard scalars; every 12 h rebuild curves and DAG from JSONL alone.

### 4.4 State Persistence

Control state rebuilds from `autoresearch.jsonl` after any crash. No conversational context.

### 4.5 Power / Thermal

As in §2.3. Harness-heavy schedules also reduce average power draw relative to continuous QLoRA.

---

## 5. Progress-Curve Visualization

### 5.1 Objectives

1. Is fitness still rising on M? On V at checkpoints?  
2. Which axis is the bottleneck?  
3. What fraction of gains came from harness vs parameter routes?  
4. What is the elite lineage?

### 5.2 Components

- Primary curves: Best-on-M, Archive mean, Best lineage, Best-on-V (checkpoints)  
- Four-axis panel → bottleneck detector  
- Evolution DAG (grey = discard, green = keep, gold = elite); optional color by `route`  
- **Route attribution plot**: cumulative Δfitness attributed to harness vs param mutations  

### 5.3 Logging Schema (excerpt)

```json
{
  "iter": 247,
  "ts": "2026-10-15T03:42:17Z",
  "commit": "a3f8c2d",
  "parent": "7b1e4a9",
  "route": "harness",
  "modules": {"cal": "L1_vector", "backend": "cuda_graphs"},
  "proxy": 68.1,
  "medium": 71.4,
  "fitness_M": 71.4,
  "action": "keep",
  "ci90_lo": 0.3,
  "wall_s": 12.4,
  "peak_vram_mb": 890,
  "diagnosis": "vector temperature improved ECE 0.081→0.068"
}
```

TensorBoard local only; matplotlib rebuild from JSONL every 12 h. No remote telemetry.

---

## 6. Modular Candidate-Idea Pool (Harness-First)

### 6.1 Priority Ranking

| Pri. | Candidate Idea | Type | Cycle cost | VRAM |
|------|----------------|------|------------|------|
| **P0** | AnyJev L0 / L1 / vector temperature scaling | Harness · Calib. | Seconds | < 1 GB |
| **P0** | Isotonic / Platt post-hoc maps (gated by ≥ 500 labels) | Harness · Calib. | Seconds | < 1 GB |
| **P0** | Scoring-pipeline variants (prefix share, option order, stabilizations) | Harness · Pipeline | Seconds | < 1 GB |
| **P0** | CUDA Graphs / inference-backend swaps | Harness · Infer. | ~5 min wall | < 1 GB |
| **P1** | QLoRA adapter train + merge (task families) | Param. | ≤ 30 min | ~4 GB |
| **P1** | Rank / target-module / LR sweeps within VRAM | Param. | ≤ 15 min | ≤ 5 GB |
| **P2** | Prefix-tree / advanced batching | Harness · Infer. | ~5 min | < 1 GB |
| **P3** | Speculative decoding / multi-model committees | — | — | **Excluded** |

*Table 3. Harness mutations dominate the schedule; QLoRA is gated.*

### 6.2 Harness Modules (Primary)

- **Calibration:** L0, L1, vector temperature, isotonic/Platt with sample-size gate  
- **Pipeline:** shared-prefix scoring, numerical stability, option-permutation robustness  
- **Backend:** CUDA Graphs capture, precision policy, kernel selection  

### 6.3 QLoRA (Secondary)

Only when harness slopes on the bottleneck axis are flat for a cool-down window. Procedure: train on task family → merge → proxy → medium keep test → else `git reset`. Log all failures into `FAILURES.md` under `param.qlora.*`.

### 6.4 Diversity Safeguards

- MAP-Elites 5×5 on (Intelligence, Calibration)  
- Module-family cool-downs  
- Offline idea-injection file  
- Forced exploration of under-filled archive cells  
- Mandatory `FAILURES.md` query before new proposals  

---

## 7. Experimental Protocol

### 7.1 Phase 1 — Baseline and Feasibility (Week 1)

**Objective.** Baselines, lock V, verify recovery, measure harness vs QLoRA cycle costs.

**Method.** Evaluate: raw logits, L0, L1, L1+selected harness variants, and one QLoRA smoke run on P/M/V. Record wall time and peak VRAM per route. Lock V task IDs. Bootstrap `FAILURES.md`. Confirm JSONL recovery after kill + reboot.

**Success gate.** L1 (or better harness calib) reduces ECE; harness cycle ≪ QLoRA cycle; recovery works; V locked.

### 7.2 Phase 2 — Continuous Local Loop (Weeks 2–5)

**Objective.** Maximize evaluated candidates under harness-first schedule; collect progress curves.

**Method.** ~672 h at ≤ 150 W. Target high cycle count via harness dominance (≥ 2,000 accepted after proxy filter is a soft goal; quality over forced QLoRA). Weekly frozen-V checkpoints only. Track route mix (harness % of cycles and of cumulative Δfitness).

**Null expectation.** Early gains mostly Calibration/Speed from harness; mid-run may still be harness; late plateau may trigger sparse QLoRA. Plateau is informative (Rule 5).

### 7.3 Phase 3 — Generalization, Artifact, Dual Review (Week 6)

**Objective.** Final CI on V, route-attributed comparison table, artifact, Reviews A & B.

**Method.** Elite + top-3 archive cells on V with ≥ 3 seeds; bootstrap 95% CI. Sealed half if available. Transfer on BANKING77 / Typed Decisions. **Rule-4 table must split harness-attributed vs parameter-attributed gains.** Assemble artifact. Pass Review A and Review B before any success claim.

**Pre-registered success (Rule 1).** CI lower bound of (elite − L1 seed) on V > 0; public–sealed gap < 5 points if measurable.

---

## 8. Artifact Package and Review Gates

### 8.1 Required Contents (Rule 3)

- Environment lockfile  
- Harness configs, calibration modules, optional QLoRA scripts, controller  
- `seeds.txt`  
- Full `autoresearch.jsonl` (with `route` field)  
- `FAILURES.md` by module family  
- `scripts/rebuild_curves.py`, `rebuild_archive.py`, `verify_elite.py`  
- Elite checkpoint(s) + top MAP-Elites cells  
- Comparison table (CSV + markdown) with harness/param split and compute-parity notes  
- Signed `REVIEW_A.md`, `REVIEW_B.md`

### 8.2 Review A — Protocol Integrity

- [ ] V IDs match day-1 lock; no V in keep/revert history  
- [ ] CI matches pre-registration; ≥ 3 seeds  
- [ ] ≤ 150 W policy; no API/network in controller path  
- [ ] JSONL has keeps + discards; `FAILURES.md` complete  
- [ ] Comparison table present with **harness vs param attribution**  
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
| Phase 1 | Baselines, lock V, harness vs QLoRA cost measurement, recovery | 1 week |
| Phase 2 | 24/7 harness-first loop + weekly V checkpoints | 4 weeks |
| Phase 3 | Multi-seed final, attributed comparison table, artifact, dual review | 1 week |
| **Total** | Electricity only · no API · no cloud GPU | **6 weeks** |

*Table 4. Timeline.*

### 9.2 Cost Model

Sole operating cost: electricity under 150 W ≈ **200–350 RMB** ($28–$49) for six weeks. Harness-heavy schedules reduce average draw relative to continuous training. Night derating to 100 W optional. Daily power logged locally.

---

## 10. Risks and Mitigations

| Risk | P | I | Mitigation |
|------|---|---|------------|
| No detectable gain on frozen V | M | H | Publish negative result + full failure archive (Rule 5) |
| Overfit to proxy/medium | H | H | V locked; multi-seed CI; Review B |
| Harness search saturates early | H | M | Then and only then open P1 QLoRA; log saturation in FAILURES.md |
| QLoRA burns budget with no gain | M | M | Gate QLoRA on harness cool-down; hard cap on param-cycle fraction |
| Incomplete artifact | M | H | Review A blocks claim |
| Unfair / missing baseline table | M | M | Rule 4 + mandatory harness/param split |
| OOM on rare QLoRA | M | M | 10.5 GB budget; Liger; CKPT; auto batch reduce |
| Thermal throttling | M | M | 150 W; derate > 78 °C; suspend > 82 °C |
| Controller crash | L | M | State rebuild from JSONL |
| Scientific overclaim | M | H | Pre-registered criterion; separate harness vs param reporting |

*Table 5. Risk register under harness-first posture.*

---

## 11. Conclusion and Claim Boundary

This proposal adopts a **harness-first** technical posture because cycle-cost asymmetry and 2025–2026 evidence both favor evolving the layer around a frozen 0.6 B scoring model before spending scarce GPU hours on QLoRA. Parameter updates remain available as a gated secondary lever.

The only success claim permitted is:

> **Under a fixed 12 GB / 150 W local budget, zero external API cost, and a six-week horizon, a multi-fidelity deterministic controller—predominantly evolving harness modules (calibration, scoring pipeline, inference backend), with optional sparse QLoRA—produced a configuration whose frozen-validation fitness exceeds the L1-calibrated seed by a statistically detectable margin, supported by a complete reproducible artifact package, an explicit comparison with concurrent baselines that attributes gains to harness vs parameters, a full negative-result archive, and two independent review checklists.**

Open-ended capability discovery is not claimed. Calibration and systems-level harness gains are expected to dominate; parameter-level gains, if any, are reported separately. A run that fails the held-out criterion still releases the same quality of artifact and failure log—negative results are first-class outcomes under these rules.

If the criterion is met, the result shows that continuous, measurable optimization of decision models is accessible on a desk-class GPU without cloud or paid APIs, and that **harness search is the efficient default route** under that constraint. If not, the archive will show where harness families and parameter families each plateaued—itself a useful community result.
