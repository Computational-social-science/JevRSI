# Research Proposal — Full Edition

## RTX 4070 Single-GPU 24/7 Self-Evolving Jev Decision Model Optimization Framework

> A fully local, zero-API, multi-fidelity autonomous research framework with statistically rigorous evaluation and domain-native scientific governance rules

| Field | Value |
|-------|-------|
| **Hardware Target** | Single NVIDIA RTX 4070 (12 GB GDDR6X), power-capped at 150 W |
| **Seed Model** | AgentJev-0.6B (Qwen3-0.6B backbone) + QLoRA rank-16 + AnyJev L1 |
| **Runtime Paradigm** | Fully local multi-fidelity continuous loop — zero external API calls |
| **Primary Deliverable** | Quantifiable self-evolution progress curves + reproducible artifact package |
| **Duration** | 6 weeks (1 week setup + 4 weeks continuous run + 1 week validation & review) |
| **External Cost** | Electricity only (~200–350 RMB); no cloud GPU, no LLM API, no paid service |
| **Governance** | Six domain-native scientific rules; dual independent review before success claim |

---

## Abstract

This proposal designs an autonomous evolution framework for the Jev decision model under two hard constraints: a single RTX 4070 (12 GB VRAM) and zero external API dependency. The seed is AgentJev-0.6B with QLoRA fine-tuning and AnyJev L1 calibration—the only parameter-efficient path compatible with 12 GB. Search is performed by a fully local multi-fidelity controller (Select → Mutate → Train → Evaluate → Archive → Persist) that never calls a remote language model. Evaluation uses a three-way partition locked on day 1: a cheap proxy set, a medium set for keep/revert decisions, and a frozen held-out set used only for pre-registered success claims. A MAP-Elites archive and exhaustive failure logging maintain search diversity and prevent repeated dead-ends. Six domain-native scientific governance rules bind the experiment: pre-registered held-out criterion, fixed local resource envelope, end-to-end reproducibility, honest comparison with concurrent work, first-class negative results, and dual independent review before any success declaration. The objective is a statistically detectable improvement on the frozen set within six weeks, documented by an auditable progress archive and a complete reproducible artifact package. A plateau that fails the held-out criterion is treated as a valid negative result and is released under the same standards.

---

## 0. Scientific Governance Rules

These six rules define what counts as a valid experimental outcome. They are written for an autonomous, single-GPU, offline ML optimization experiment. Every automated cycle and every human decision is bound by them. A success claim that violates any rule is void.

### Rule 1 — Pre-registered Held-Out Criterion

Before the continuous loop starts, the public evaluation tasks are permanently split into a working set (proxy + medium) and a frozen held-out set **V**. Task IDs of V are written to the laboratory notebook and never used for keep/revert decisions. Success is defined solely as: the final elite beats the L1-calibrated seed on V, with the lower bound of a bootstrap 95% CI (over ≥ 3 independent seeds) strictly positive. No other metric or split may be substituted after the fact.

### Rule 2 — Fixed Local Resource Envelope

All training, evaluation, monitoring and decision logic must run on one RTX 4070 (≤ 12 GB VRAM, power limit ≤ 150 W) with no external model API, no cloud GPU, and no paid inference service. The wall-clock budget is six weeks. Any improvement that depends on relaxing these constraints is outside the scope of the claim.

### Rule 3 — End-to-End Reproducibility from Public Artifacts

The released package must allow a third party, given only an RTX 4070-class machine and the public repository, to: (1) recreate the environment from a lockfile; (2) rebuild progress curves and the MAP-Elites archive from the raw JSONL log alone; (3) re-evaluate the elite checkpoint on frozen V and obtain a CI consistent with the reported claim. Missing seeds, silent weight patches, or undocumented manual steps invalidate the package.

### Rule 4 — Honest Placement against Concurrent Work

The final report must include a comparison table against the strongest relevant published baselines available at the time of writing (JevBench leaders, recent calibration methods, ≤ 2 B PEFT results under similar VRAM, and any known single-GPU autonomous experiment systems). Each row must note whether compute was matched. Superiority may be claimed only on metrics the baseline actually reported.

### Rule 5 — First-Class Negative Results

Every attempted mutation—kept or discarded—is appended to `autoresearch.jsonl` with scores, action, diagnostic, VRAM peak and wall time. A maintained `FAILURES.md` groups dead-ends by module family so that the controller and future operators can avoid repeating them. A run that plateaus early is still required to ship the full failure archive; suppression of negative outcomes is not permitted.

### Rule 6 — Two Independent Checks before Any Success Claim

No success announcement is allowed until both of the following have passed and been committed:

- **(A) Protocol check** — frozen V was never used for keep/revert, CI formula matches the pre-registration, artifact is complete, comparison table is present, failure log is intact
- **(B) Replication check** — from a clean environment built only from the lockfile, re-running the elite evaluation on V still yields a positive CI lower bound

Both checks produce signed checklist files in the repository.

> These rules are enforceable by the local controller and by the final verification scripts. They exist to make positive and negative outcomes equally publishable and checkable.

---

## 1. Hardware Constraint Analysis and Optimization Strategy

### 1.1 Capability Envelope of the RTX 4070

The RTX 4070 provides 12 GB GDDR6X, full CUDA core complement, native FP16/INT8 support, approximately 200 W TDP, and sustained core temperatures controllable below 68 °C under load. In the LLM fine-tuning regime the practical constraints are:

- **Full-parameter fine-tuning is infeasible.** A 7 B model already requires ~24 GB; larger models are impossible.
- **LoRA / QLoRA remains viable.** QLoRA (4-bit quantized base + 16-bit LoRA adapters) compresses 7 B fine-tuning into the 6–10 GB range; for sub-2 B models the footprint is far lower.
- **0.6 B–2 B models occupy a comfortable operating zone.** Community measurements on the same GPU show Qwen2.5-1.5B-Instruct trained with LoRA rank 32 on 123 Q&A pairs for three epochs peaking at only ~3.9 GB and completing in 98 seconds. Consequently, AgentJev-0.6B under QLoRA leaves substantial headroom (~8 GB) for concurrent evaluation and calibration.

### 1.2 Hierarchical VRAM Budget

A hard upper bound of 12 GB is partitioned so that training, evaluation and monitoring can coexist without continual model swapping:

| Allocation | Budget | Rationale |
|------------|--------|-----------|
| QLoRA fine-tuning (train mode) | ≤ 5 GB | 4-bit base + LoRA adapters + optimizer states |
| Inference / calibration | ≤ 4 GB | 0.6 B FP16 forward + AnyJev calibration artifacts |
| Evaluation buffers | ≤ 3 GB | Proxy or medium-fidelity task batches |
| System reserve | ≥ 2 GB | CUDA context, driver, monitoring overhead |

*Table 1. Concurrent VRAM budget for train–eval–monitor operation.*

### 1.3 Power and Thermal Policy for Continuous Operation

Empirical 90-day continuous-run data for the RTX 4070 show strong ambient-temperature dependence: summer (30–40 °C) power draw is approximately 1.99× winter (0–10 °C) draw, with monthly electricity cost fluctuating between ¥3,254 and ¥6,486. The framework therefore embeds an explicit power-limit policy: **`nvidia-smi -pl 150`** (well below the 200 W TDP) is the default. Monitoring scripts log GPU temperature and power every five minutes via local `nvidia-smi`. Sustained temperature above 75 °C triggers automatic batch-size reduction; temperatures above 82 °C suspend the training loop until the device cools below 70 °C. Optional night-time derating (23:00–07:00 local) to 100 W further reduces cost and noise. All monitoring is local system utilities; no network service is involved.

---

## 2. Seed Model and Training Recipe

### 2.1 Why AgentJev-0.6B Is the Only Feasible Seed

Under a 12 GB ceiling, Open-Jev-27B is ruled out (FP16 weights alone occupy ~54 GB). Even 4 B-class models push QLoRA footprints into the 6–8 GB range; adding evaluation buffers frequently exceeds capacity. AgentJev-0.6B is the unique practical seed because:

- Peak QLoRA VRAM of only ~3.9 GB leaves ample margin for calibration and evaluation.
- Demonstrated JevBench compatibility: community baselines report a composite score of 66.4% for kev-0.6B, with particularly strong Cost-axis performance.
- Adequate calibration quality (ECE = 0.0712) that remains well-behaved in the high-confidence regime.
- Extremely low inference latency (~50 ms per forward pass), enabling high experimental throughput and optional local ranking of candidate shortlists without external services.

### 2.2 Seed Configuration

```
AgentJev-0.6B (Qwen3-0.6B backbone with permutation-equivariant scoring head)
+ AnyJev L1 post-hoc calibration (per-(model, question) temperature scaling)
+ Trainable QLoRA adapters (rank = 16, target_modules = all-linear)
```

### 2.3 QLoRA Memory-Optimized Training Recipe

Drawing on low-VRAM configurations validated on 12 GB consumer GPUs:

- `seq_len = 2048` (instead of the customary 4096/8192)
- `per_device_batch_size = 1`, `gradient_accumulation_steps = 16` (effective batch size 16)
- `max_samples = 1000`, `epochs = 1` (single-pass fine-tuning)
- `memory_budget_gb = 10.5` (1.5 GB system reserve)
- Liger fused chunked cross-entropy + activation checkpointing to suppress peak memory
- APOLLO-Mini optimizer (approximately 50% reduction in optimizer-state footprint relative to AdamW)

These settings are treated as a fixed baseline; mutations may alter rank, target modules, sample count or learning rate only within the VRAM envelope.

---

## 3. Fully Local Multi-Fidelity Research Loop

### 3.1 Design Invariants

- **Zero external API.** No cloud LLM, no paid inference, no remote logging.
- **Deterministic controller.** Selection, keep/revert and cool-downs are rule-based and reproducible.
- **Multi-fidelity evaluation.** Cheap proxies filter candidates before medium evaluation.
- **Frozen validation set.** Locked on day 1; touched only at pre-registered checkpoints (Rule 1).
- **Quality-diversity archive.** MAP-Elites grid over axis scores preserves diversity.
- **Electricity-only cost.** Estimated 200–350 RMB over six weeks under the 150 W cap.

### 3.2 Data Partitions (Fixed on Day 1)

| Partition | Size (approx.) | Role | When used |
|-----------|----------------|------|-----------|
| Proxy set P | 30 tasks | Ultra-cheap fitness estimate | Every candidate, every cycle |
| Medium set M | 80 tasks | Keep/revert decision + effect-size test | Candidates that pass proxy |
| Frozen val V | 60 tasks | Progress claims and final report only | Checkpoints (pre-registered) |
| Sealed (if available) | — | External generalization | End of Phase 3 only |

*Table 2. Evaluation partitions. V is never used for keep/revert decisions during the loop (Rule 1).*

### 3.3 Cycle Phases

#### Phase 1 — Select

Read archive statistics (best fitness, recent axis slopes, cool-down timers, archive occupancy). Apply priority rules: (a) untried high-priority ideas first; (b) ideas targeting the current bottleneck axis (detected from rolling slope of the four-axis curves); (c) after *k* consecutive failures of a module family, apply cool-down. Optionally score a shortlist with one local forward pass of AgentJev-0.6B. Emit a single concrete mutation descriptor. Before proposing, query `FAILURES.md` to avoid known dead-ends (Rule 5).

#### Phase 2 — Mutate

Apply the selected mutation by editing configuration or training scripts; create a git commit whose message encodes parent hash and module tags.

#### Phase 3 — Train (or skip)

If the mutation is parameter-level, run QLoRA under the VRAM budget (typically ≤ 5 minutes for 1,000 samples). Calibration-only mutations skip training entirely.

#### Phase 4 — Multi-Fidelity Evaluate

1. Evaluate on proxy set P (seconds). If improvement over parent is below ε_proxy, discard immediately.
2. Otherwise evaluate on medium set M. Keep only if Δfitness ≥ ε_M and the lower bound of a bootstrap 90% CI is positive (or a paired sign test succeeds).
3. Full public-half and frozen-V evaluation are performed only at scheduled checkpoints, never every cycle.

#### Phase 5 — Archive Update and Persist

If kept, insert into the MAP-Elites archive cell corresponding to discretized (Intelligence, Calibration) scores; replace the cell occupant only if fitness is higher. Append a complete JSON record to `autoresearch.jsonl`. Update TensorBoard scalars. Every 12 hours regenerate static progress curves and the evolution DAG from the JSONL file alone.

### 3.4 State Persistence and Crash Recovery

All experimental state is persisted to `autoresearch.jsonl`, which survives process kills and host reboots. Each record stores commit hash, metrics, peak VRAM, run status and a short diagnostic. Control state (eligible candidates, cool-down timers, current best) is recomputed from the JSONL file on every restart; the cycle resumes from the last interrupted point with no reliance on in-memory context.

### 3.5 Automated Power and Thermal Management

- Default power limit: `nvidia-smi -pl 150`
- Temperature telemetry every 5 minutes via local `nvidia-smi`; batch-size reduction above 78 °C; loop suspension above 82 °C until temperature falls below 70 °C
- Optional night-time derating (23:00–07:00) to 100 W
- All thermal logic is pure local scripting—no external monitoring service

---

## 4. Progress-Curve Visualization System

### 4.1 Visualization Objectives

The primary scientific product of a multi-week autonomous run is a self-evolution progress curve. The visualization suite answers three questions:

1. Is composite fitness still rising on the medium set, and is the rate accelerating or decelerating?
2. How do the four axes (Intelligence, Calibration, Speed, Cost) individually evolve, and which is the current bottleneck?
3. Which module combinations were retained, and what is the ancestral lineage of the current best individual?

Frozen-V scores appear only at pre-registered checkpoints.

### 4.2 Core Visualization Components

**(1) Primary Progress Curves.** Best-so-far on medium set M (every keep), Archive mean, Best lineage, and Best-so-far on frozen V (checkpoints only). At 5–10 minutes per cycle a 24-hour period yields 150–250 data points; a one-week run produces 1,000–1,700 points.

**(2) Four-Axis Decomposition Panel.** Synchronized sub-plots for Intelligence, Calibration, Speed and Cost feed the bottleneck detector used in Phase 1 selection.

**(3) Evolution Tree (DAG).** Nodes are commit hashes; edges encode parent→child inheritance. Discarded configurations appear in grey, retained ones in green, elite lineage in gold.

### 4.3 Fully Local Implementation

**Logging layer.** After every cycle a JSON record is appended to `autoresearch.jsonl`:

```json
{
  "iter": 247,
  "ts": "2026-10-15T03:42:17Z",
  "commit": "a3f8c2d",
  "parent": "7b1e4a9",
  "modules": {"cal": "L1", "lora": "banking77_r16"},
  "proxy": 68.1,
  "medium": 71.4,
  "fitness_M": 71.4,
  "action": "keep",
  "ci90_lo": 0.3,
  "diagnosis": "L1 ECE 0.081→0.068; Intelligence unchanged"
}
```

**Visualization layer.** TensorBoard runs as a local process (`tensorboard --logdir ./logs/jev-evolve --port 6006`). For archival analysis, Python + matplotlib scripts regenerate full progress plots and evolution trees every 12 hours from JSONL alone. No cloud dashboard, no remote telemetry, no third-party analytics service is used.

---

## 5. Modular Candidate-Idea Pool (RTX 4070 Adaptation)

### 5.1 Priority Ranking under 12 GB Constraints

| Pri. | Candidate Idea | Type | Cost | VRAM |
|------|----------------|------|------|------|
| P0 | AnyJev L0 / L1 calibration strategies | Calib. | Seconds | < 1 GB |
| P0 | QLoRA adapter train + merge (task families) | Param. | ≤ 30 min | ~4 GB |
| P1 | Vectorized temperature scaling (≥ 500 labels) | Calib. | Seconds | < 1 GB |
| P1 | CUDA Graphs / inference-backend swaps | Infer. | 5 min | < 1 GB |
| P2 | Prefix-tree / batching variants | Infer. | 5 min | < 1 GB |
| P2 | Learning-rate / rank / target-module sweeps | Param. | ≤ 15 min | ≤ 5 GB |
| P3 | Speculative decoding / multi-model committees | — | — | Excluded |

*Table 3. Prioritized local candidates. Speculative decoding and committees are excluded (VRAM).*

### 5.2 QLoRA Adapter Merging — the Sole Parameter-Level Lever

On the RTX 4070 the only practical parameter-level optimization is QLoRA adapter merging:

1. Train a LoRA adapter on a task family (e.g., BANKING77) atop a 4-bit AgentJev-0.6B base. Community measurements: 123 examples, rank 32, 3 epochs finish in 98 s at ~3.9 GB peak.
2. Merge the adapter back into the base weights, producing a new checkpoint.
3. Evaluate on proxy then medium set against the unmerged baseline (local GPU only).
4. Retain if Intelligence or Calibration improves under the keep rule; otherwise `git reset`.

### 5.3 Diversity Safeguards and Explicit Exclusions

**Diversity mechanisms:**

- MAP-Elites grid over quantized Intelligence × Calibration bins (e.g., 5 × 5)
- Cool-down after *k* consecutive failures of a module family
- Offline idea-injection file (human-droppable without network access)
- Periodic forced exploration of under-populated archive cells
- Mandatory `FAILURES.md` query before new candidate proposal (Rule 5)

**Excluded a priori:**

- Full-parameter fine-tuning (optimizer/gradient overhead)
- Speculative decoding (draft + target doubles VRAM)
- Multi-model committees (linear memory growth)
- Any mutation requiring an external LLM or paid API

---

## 6. Experimental Protocol

### 6.1 Phase 1 — Baseline and Feasibility (Week 1)

**Objective.** Establish baselines, lock frozen V, verify recovery and QLoRA feasibility.

**Method.** Measure four configurations (raw logits, L0, L1, L1+QLoRA) on partitions P, M and V. Record peak VRAM, wall time and axis scores. Confirm `autoresearch.jsonl` restores state after intentional process kill and host reboot. Lock V task IDs in the laboratory notebook. Bootstrap `FAILURES.md` with any setup-phase dead-ends.

**Success gate.** L1 reduces ECE; QLoRA finishes < 5 min at ≤ 5 GB; recovery works; V is locked.

### 6.2 Phase 2 — Continuous Local Loop (Weeks 2–5)

**Objective.** Run the multi-fidelity controller and collect progress-curve data under Rule 2.

**Method.** Four weeks (~672 h) under the 150 W power limit. Target ≥ 2,000 accepted mutations after proxy filtering (far fewer full medium evaluations). Checkpoints on frozen V at the end of each week only (pre-registered). Primary observables: best-on-M trajectory, archive occupancy, axis slopes, GPU health counters, best-on-V at checkpoints.

**Null expectation.** After the first ~200–400 cycles most remaining gains will be small engineering improvements; a plateau is scientifically informative, not a failure (Rule 5).

### 6.3 Phase 3 — Generalization, Artifact and Dual Review (Week 6)

**Objective.** Final statistical evaluation, comparison table, artifact package, dual review.

**Method.** Re-evaluate the elite (and the three best archive cells) on frozen V with ≥ 3 random seeds; report mean ± bootstrap 95% CI. If a sealed half is available, evaluate once. Transfer tests on BANKING77 and Typed Decisions official splits. Compile the Rule-4 comparison table. Assemble the full artifact package. Execute Review A (protocol integrity) and Review B (clean-env replication).

**Pre-registered success criterion (Rule 1).** Lower bound of the 95% CI of (elite − L1 baseline) on V is > 0, and the public–sealed gap (if measurable) is < 5 points. Both review checklists must pass before any success declaration (Rule 6).

---

## 7. Artifact Package and Review Gates

### 7.1 Required Artifact Contents (Rule 3)

- Environment lockfile (`pip freeze` or `conda export`)
- Training, calibration, evaluation and controller scripts with pinned configs
- `seeds.txt` (all random seeds used for training and final evaluation)
- Complete `autoresearch.jsonl` (successes and failures)
- `FAILURES.md` grouped by module family
- `scripts/rebuild_curves.py`, `scripts/rebuild_archive.py`, `scripts/verify_elite.py`
- Final elite checkpoint(s) and the three best MAP-Elites cells
- SOTA comparison table as machine-readable CSV + rendered markdown (Rule 4)
- Signed `REVIEW_A.md` and `REVIEW_B.md` checklists (Rule 6)

### 7.2 Review A — Protocol Integrity Checklist

- [ ] V task IDs match day-1 lock file; no V evaluation appears in loop keep/revert records
- [ ] Bootstrap CI script matches the pre-registered formula; ≥ 3 seeds used
- [ ] Power log shows ≤ 150 W policy; no external API keys or network calls in controller path
- [ ] JSONL contains both keeps and discards; `FAILURES.md` covers every discarded module family
- [ ] Comparison table cites concurrent baselines with compute-parity notes
- [ ] Artifact package runs `verify_elite.py` to completion on a clean environment

### 7.3 Review B — Clean Replication Checklist

- [ ] Clean environment created from `environment.lock` only
- [ ] `verify_elite.py` reproduces a positive 95% CI lower bound on frozen V
- [ ] Progress curves and MAP-Elites archive rebuild from JSONL alone
- [ ] No undocumented dependency or manual weight patch required
- [ ] Checklist signed and committed before any success announcement

---

## 8. Resource and Cost Estimates

### 8.1 Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Phase 1 | Baselines, lock V, recovery tests, `FAILURES.md` bootstrap | 1 week |
| Phase 2 | 24/7 multi-fidelity local loop + weekly V checkpoints | 4 weeks |
| Phase 3 | Multi-seed final, comparison table, artifact, dual review | 1 week |
| **Total** | Electricity only · no API · no cloud GPU | **6 weeks** |

*Table 4. Project timeline.*

### 8.2 Cost Model — Electricity Only

Because the framework is fully local and issues no external API calls, the sole operating cost is electricity. Under the 150 W power-limit policy, six weeks of continuous operation is estimated at **200–350 RMB** (approximately $28–$49 USD), depending on ambient temperature. There is no cloud-GPU rental, no LLM API subscription, and no third-party monitoring fee. Optional night-time derating to 100 W further reduces the upper bound. Daily power draw is logged locally.

---

## 9. Risks and Mitigations

| Risk | P | I | Mitigation |
|------|---|---|------------|
| No detectable gain on frozen V | M | H | Publish as negative result with full failure archive (Rule 5) |
| Overfitting to proxy / medium sets | H | H | V locked day 1; multi-seed final CI; Review B |
| Early search plateau | H | M | MAP-Elites; cool-downs; offline idea injection; `FAILURES.md` |
| Incomplete artifact package | M | H | Review A blocks any success claim (Rule 6) |
| Missing or unfair baseline table | M | M | Rule 4 required section; compute-parity column |
| QLoRA triggers OOM | M | M | 10.5 GB budget; Liger CE; activation CKPT; auto batch reduction |
| Long-term thermal throttling | M | M | 150 W cap; derate > 78 °C; suspend > 82 °C |
| Controller crash / state loss | L | M | Full control state rebuilt from JSONL on restart |
| Electricity cost overrun | L | L | Night 100 W derating; daily local power log |
| Scientific overclaim | M | H | Pre-registered criterion; separate calib. vs param. gains |

*Table 5. Risk register. All mitigations preserve the six governance rules.*

---

## 10. Conclusion and Claim Boundary

This proposal reconstitutes autonomous decision-model optimization around two premises: (a) a single consumer-grade GPU is sufficient, and (b) the research loop must run with zero external API dependency. Six domain-native scientific governance rules make both positive and negative outcomes falsifiable, reproducible and publishable.

The only success claim permitted is:

> **Under a fixed 12 GB / 150 W local budget, zero external API cost, and a six-week horizon, a multi-fidelity deterministic controller produced a configuration whose frozen-validation fitness exceeds the L1-calibrated seed by a statistically detectable margin, supported by a complete reproducible artifact package, an explicit comparison with concurrent baselines, a full negative-result archive, and two independent review checklists.**

We do not claim open-ended capability discovery, nor that 0.6 B QLoRA is a universal substitute for larger-scale training. Calibration and systems gains are expected to dominate; parameter-level gains, if they appear, are reported separately. A run that fails the held-out criterion is still required to release the same quality of artifact and failure log—negative results are first-class outcomes under these rules.

If the pre-registered criterion is met, the result demonstrates that continuous, measurable optimization of decision models is accessible on hardware already present on a typical research desk—without cloud GPUs and without paid language-model APIs. If the criterion is missed, the same instrumentation will show where the plateau occurred and which module families failed, which is itself a useful contribution to the community.
