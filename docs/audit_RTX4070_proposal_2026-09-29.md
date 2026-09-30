# Proposal Audit: RTX4070 Self-Evolving Jev Framework v6

<!-- RETIRED OBJECT DECLARATION -->
The laya-multilingual 322M substrate is a RETIRED research object (retired 2026-09-30).
It is named here only as the record of a measurement that must NOT be inherited: its
Floor B (0.40 pp) was carried into INSTRUMENT_CALIBRATION.json and thence into the live
seed's accept threshold. That threshold is revoked. Do not work on this object.


**Audit target** `RTX4070_SelfEvolving_Jev_Research_Proposal.md` (381 lines, 10 sections, 6 governance rules)
**Audit date** 2026-09-29
**Audit method** Pass 2c — do not assert from the proposal's self-description; first obtain primary evidence (HF Hub API, `temperatures.json` inside the checkpoint), then place it side by side with artifacts this project has already measured
**Verdict** **Not adopted as a research goal for JevRSI.** 4 elements should be absorbed; 5 assertions are refuted or unverifiable upon checking.

---

## 0. One-sentence determination

The proposal's **engineering skeleton** (multi-fidelity, JSONL state machine, failure archive, dual review) is consistent with ours and more complete;
its **three cited numbers have no source** (66.4% / ECE=0.0712 / ~50ms, independently verified);
and its **core experimental premise directly conflicts with quantities we have already measured**: it assumes "longer training = more gain",
whereas our learning curve has already measured 1499→8345 steps (5.6× compute) yielding only 2.17 pp,
**below the decision threshold τ = 4.95 pp that we measured ourselves**. Searching the same axis for 6 weeks,
at the already-measured effect size, the expected number of detectable successes is **0**.

> **The first draft of this section cited "AnyJev is a nonexistent fictional baseline" as the primary reason for rejection. That assertion has been withdrawn; see the §0 Correction section.**

---

## 0. Correction section (2026-09-29, must read first)

**In §1.2 I asserted that the `AnyJev` L1 calibrator does not exist, and on that basis wrote "the baseline does not exist → success is undecidable" as the primary reason for rejecting the proposal. Both of these were wrong.**

**Nature of the error**: I searched only the HuggingFace model hub (`anyjev` / `any-jev` → n=0), then took "absent from one hub" to mean "does not exist".
AnyJev is a **GitHub repository**: `nokia-applied-research/AnyJev` (Nokia Applied Research, **869★ / 110 forks**,
Apache-2.0, v0.2.0 released 2026-09-28). **A negative conclusion was drawn without cross-hub search.**

**Content hereby voided**:
- The "AnyJev does not exist" row in the §1.2 table → marked RETRACTED
- "the success criterion is anchored to a fictional baseline" in the §0 summary → **void**
- R5 in §4 → **still stands** (66.4% indeed has no source), but **the reason is unrelated to AnyJev**

**What still stands** (independent of AnyJev, independently verified):
- The three numbers 66.4% composite, ECE=0.0712, ~50ms are **all without a source** in the AgentJev-0.6B model card and in `temperatures.json`
- `temperatures.json` is in fact **3 scalars of per-primitive-type granularity**, whereas the proposal claims per-(model, question) — **a granularity gap of 2 orders of magnitude**
- The conflict between the proposal's core experimental premise and our already-measured quantities (training axis +2.17 pp < τ = 4.95 pp) **holds in full**

**More serious consequence**: while verifying AnyJev, we found `docs/results_typed.md` — it measured **laya-multilingual = 0.340** on **exactly the same
400 cases / 2,000 decisions test set as ours** (we measured 0.3500, consistent across machines),
and reported **0.768 / 0.786 / 0.799**. **This overturns this project's prior setting of 0.704 / 0.735 as the capability ceiling.**
The research goal has been reset accordingly → `docs/RESEARCH_GOAL_v2_2026-09-29.md`.

**Lesson (mechanized)**: `scripts/check_citation_provenance.py` requires every external number to give
HF model-id + file path / arXiv id / GitHub repo + path + commit. **"Does not exist" is the highest-risk form of assertion**,
because it disguises the insufficiency of the search surface as a property of the world.

---

## 1. Load-bearing fact check (primary evidence)

### 1.1 What holds

| Proposal assertion | Evidence | Verdict |
|---|---|---|
| AgentJev-0.6B is a Qwen3-0.6B base + permutation-equivariant scoring head | `huggingface.co/api/models/aimeigaoshou/agent-jev`: `base_model: Qwen/Qwen3-0.6B`, `architectures: ["AgentJevModel"]`, `hidden_size 1024`, `vocab 151936`, 821 downloads / 35 likes, Apache-2.0 | ✅ **holds** |
| Open-Jev-27B exists (hence excluded by 12 GB) | `ZefanCai/Open-Jev-27B-v1.1` is on the hub | ✅ **holds** |

### 1.2 What does not hold or is unverifiable

| Proposal assertion | Measured result | Verdict |
|---|---|---|
| **`AnyJev` L1 calibrator** | **RETRACTED — see the §0 Correction section. AnyJev genuinely exists** (`github.com/nokia-applied-research/AnyJev`, Nokia, 869★) | ~~❌ does not exist~~ → ✅ **exists** |
| L1 is "per-(model, question) temperature scaling" | AgentJev-0.6B actually ships `temperatures.json`, **only 3 scalars, per-primitive-type**: `boolean` T=1.0718 (n=180), `choice` T=1.0353 (n=180), `score` T=1.0718 (n=240), covering 600 calibration samples | ❌ **granularity gap of 2 orders of magnitude** (3 vs 1200 parameters) |
| "community baselines report a composite score of **66.4%** for kev-0.6B" | The model card `model-index` has **no 66.4%, no composite, no JevBench**. What is actually recorded is: Coding completion accuracy **57.8%**, Invoice processing teacher-agreement **87.2%**, both labeled `verified: false` | ❌ **the number has no source, and the stated metric does not exist** |
| "adequate calibration quality (**ECE = 0.0712**)" | `temperatures.json` contains only temperatures and soft CE, **no confidence or correctness** → ECE cannot be derived from it; the checkpoint does not provide the data needed for verification | ⚠️ **unverifiable** |
| "inference latency **~50 ms** per forward pass" | the checkpoint contains no latency benchmark; no reproducible third-party source | ⚠️ **unverifiable** |
| "Qwen2.5-1.5B + LoRA r32, 123 Q&A, 3 epochs → 3.9 GB / 98 s" | an **external measurement unrelated to this project**, used as a local budget basis but with no citation | ⚠️ **unverifiable** (and Qwen2.5-1.5B ≠ Qwen3-0.6B) |

**Note that the `temperatures.json` finding is itself a scientific result**: AgentJev-0.6B's "per-question calibration" is in fact
**3 temperatures, one per primitive type**. This is isomorphic to our own earlier independent finding — on the laya side
`temperature` is likewise a buffer rather than a learnable parameter. **The granularity of calibration is coarse in both independent implementations**,
which is itself worth writing into the manuscript, rather than being cited by the proposal as a strong "per-(model,question)" capability.

---

## 2. Conflicts with quantities this project has already measured

| Quantity | This project's measurement | Source | Implication for the proposal |
|---|---|---|---|
| Training-axis gain | 1499 steps 0.5550 → 8345 steps 0.5567, **+2.17 pp** (5.6× compute) | `learning_curve.json` | The proposal's §6.2 expects "weekly checkpoints continuously rising" |
| Decision threshold τ(dev) | **4.95 pp** (H=1000) / 2.39 pp (H=1), Bonferroni | `learning_curve.json`, `tau_calibration.py` | **Total training-axis gain < τ** |
| Interpretability floor | **1.73 pp** | `learning_curve.json` | The dev change of a single run may lie entirely within noise |
| Floor B (run-to-run SD) | **0.40 pp** (5 seed, df=4) | `floor_b_laya.json` | Irreproducibility across repeated runs |
| Empirical false-positive rate of the acceptance rule | Naive rule **5/10 = 50%**; τ rule **0/10** | `accept_rule_audit.py` | The proposal's `Δfitness ≥ ε_M` does not define the source of ε |
| Per-replicate test accuracy | 0.5300 / 0.5340 / 0.5290 / 0.5360 / 0.5260 (SD 0.0040) | `floor_b_laya.json` | The proposal's "≥2000 accepted mutations" is incompatible with an SD of 0.4 pp |
| Fitness resolution | acc SE 2.14 pp (dev, 120 cases) | `fitness_resolution.json` | The proposal uses a 30-task proxy for per-round screening, **the SE will be larger** |

**Core conflict**: the proposal's §3.2 sets proxy 30 tasks / medium 80 / frozen V 60,
whereas we have already measured an acc SE of 2.14 pp on the 120-case dev set alone.
**Using 30 items for per-round keep/revert decisions means noise exceeds signal.** The proposal does not define the source of ε_proxy and ε_M —
and that is precisely what this project had to build at the greatest cost (`tau_calibration.py`).

---

## 3. The 4 items that should be absorbed (which we genuinely lack)

| # | Proposal element | Our status | How to absorb it |
|---|---|---|---|
| **A1** | **Failure archive grouped by module family** (`FAILURES.md`), and **mandatory lookup before proposing** | We have the retention policy in `prune_checkpoints.py`, but **no family-grouped index of dead ends** | Add `FAILURES.md` + make the proposer read it mandatorily; place it alongside the existing contamination guard |
| **A2** | **Dual review**: A = protocol completeness, B = clean-environment reproduction | We have **seven gates** (contamination/audit/numeric/boundary/weight/split/graph QC), but **no "reproduce elites from lockfile in a clean environment"** | Add a Review B equivalent: rebuild from `pip freeze` and re-evaluate |
| **A3** | **Explicit power/thermal policy** (`-pl 150`, >78 °C reduce batch, >82 °C suspend) | We have a watchdog (3 times NOT_VIABLE → stop + `host_degraded.json`), **but no temperature/power dimension** | Add temperature and power telemetry to the existing watchdog |
| **A4** | **JSONL as the single source of state; rebuild all control state from the log after restart** | We have a record-then-reset trajectory protocol and checkpoint resume, **but control state is not fully rebuilt from the log** | Add and test a function that "rebuilds the current best + cooldown timing from JSONL" |

**Note that A1–A4 are all engineering/governance elements; not one of them requires us to change the research object.** That is exactly the part that can be absorbed.

---

## 4. The 5 items explicitly refused for absorption

| # | Proposal element | Reason for rejection |
|---|---|---|
| **R1** | Switch the base model to **AgentJev-0.6B + QLoRA** | We have already measured that laya-multilingual(322M) is fully trainable with all parameters (0.22 s/step, 11.5 GiB peak). Switching to QLoRA introduces 4-bit quantization noise and **discards the completed base-model rebuild and three-way verification**. The user has already stated that "AgentJev-0.6B is only one of the benchmark works we need to surpass". |
| **R2** | Search scale of **6 weeks / ≥2000 accepted mutations** | Incompatible with the measured 0.40 pp Floor B and 4.95 pp τ. Expected number of detectable successes = 0. |
| **R3** | Split of **30/80/60** | 30 items for per-round screening means noise of 2.14 pp > any plausible effect. **And it conflicts with the protocol-locked 960/120/120+400** — the latter has already been verified case by case via `verify_split_against_protocol.py`. |
| **R4** | **MAP-Elites four axes (Intelligence/Calibration/Speed/Cost)** | Among the four axes, Speed/Cost are **system properties**, unrelated to this project's decision-quality construct. They would dilute the construct into an "engineering score". |
| **R5** | Using **66.4% composite** as the benchmark | That number does not exist. Rule 4 (honest benchmarking) is precisely what requires us to **cite only metrics the baseline actually reported** — the proposal violates its own Rule 4. |

---

## 5. One observation worth recording

The proposal's §2.1 and §5.2 both cite "3.9 GB / 98 s" as evidence that is **verified on this machine**,
but that is a measurement taken on **Qwen2.5-1.5B**, applied to **Qwen3-0.6B + all-linear + rank 16**.
When `target_modules = all-linear` acts on 0.6B, the relationship between trainable parameters and rank differs from 1.5B,
so 98 s does not transfer.

This is isomorphic to a class of error this project has already documented: **treating a number from another machine / another model as a local budget basis**.
Our `JevRSI_global_principles_v1.md` §1 has already drawn this boundary — **borrowing a framework is allowed, borrowing a number is not**.
The proposal is stepping exactly on this line.

---

## 6. Impact on the current goal

**The current goal is unchanged**: on laya-multilingual(322M), with a fixed wall-clock generational budget + pre-registered τ + frozen test set,
test whether "intrinsically driven generational iteration can produce a detectable surpassing". The user has ruled that the theoretical ceiling is 1.

**Three corrections brought by the proposal**:

1. **D3 (loop form) can now be answered**. The proposal's six governance rules + multi-fidelity + JSONL state machine
   show that what the user wants is **a program that can run 24/7**, not a 5-item checklist.
   But **the program must run on our τ, not the proposal's ε**.
2. **A1–A4 are net gains**; they go straight into the backlog.
3. **One newly added pre-registered constraint**: every cited external number must give a verifiable source
   (HF model-id + file path, or arXiv id). **No source = it must not enter the manuscript.**
   This follows directly from the proposal's failure mode.

---

## 7. Awaiting user adjudication

| id | Item | My judgment |
|---|---|---|
| **Q1** | Whether to write A1–A4 into the pre-registration as a governance extension | **Yes** — they do not change the research object, they only strengthen auditability |
| **Q2** | Whether "external numbers must be verifiable" becomes a hard gate (added to the audit scripts) | **Yes** — the three unsourced numbers in the proposal, 66.4%/ECE/50ms, are precisely why it failed |
| **Q3** | Fixing the loop form as "a program that can run 24/7" (per D3) | **Yes**, but τ uses our already-measured 4.95 pp |