# CURRENT OBJECT — read this first

> **The full research objective is
> [`RTX4070_SelfEvolving_Jev_Research_Proposal.md`](RTX4070_SelfEvolving_Jev_Research_Proposal.md)** —
> the six-week, single-RTX-4070, zero-API self-evolving optimisation framework. This page indexes what is
> measured and what is retired; the proposal is the objective itself.

**Last revised 2026-09-27.** If any other file in this repository appears to disagree with this page,
**this page wins** — or the disagreeing file is a retired artifact that should have been archived.

> **Renamed 2026-09-27 — the project code is now `JevRSI`.** The brief (`project-2.docx`) called it
> *Jevolution* / *JEVO*; that name is **retired** and must not be used. It survives only in `archive/`
> (history, deliberately unmodified) and in this notice. `scripts/check_object_purity.py` asserts its
> **absence** from the live tree, so a stray *Jevolution* reference now fails the drift check.

---

## What we are working on

> **JevRSI** — a 24/7 autoresearch retraining loop for **`AgentJev-0.6B`**, a
> **non-generative** typed-decision model (Qwen3-0.6B backbone + permutation-equivariant candidate head),
> where the optimised artefact is the **training script**, not a prompt.

| | |
|---|---|
| **Subject repository** | `E:/2026-AI4S/agent-jev` — clone of `github.com/malevrigns/agent-jev`, `main @ a965ca8f` |
| **Loop harness (substrate)** | `D:/2026-AI4S/nanochat-autoresearch` — **`jsegov/autoresearch-win-rtx`**, a Windows-native fork of karpathy/autoresearch (RTX 4070 12 GB in its supported matrix; official PyTorch CUDA wheels + SDPA, no Triton-on-Windows). Ships the loop artefacts JevRSI lacks: a 213-line `program.md` constitution (incl. the record-before-reset rejection protocol), a `results.tsv` ledger, and a read-only `prepare.py` instrument (`TIME_BUDGET`=300, `EVAL_TOKENS`=40×524288, `evaluate_bpb`). **`nanochat` / `val_bpb` are LIVE here as the instrument** and retired only as a *research object* — see § Retired |
| **Benchmark** | `LocalLLaMA/typed-decisions`, official test split — **400 cases / 2,000 questions**, 5 questions per case |
| **Baseline to beat** | **79.25%** Top-1 (1585/2000) raw — but **0.7036** chance-corrected (§3.2); `phase4/final.pt` → `agentjev_v1` |
| **Metric** | Top-1, plus soft CE / Brier / ECE / Score MAE — reported pooled, per primitive, and per workflow |
| **Primitives** | Boolean · Choice · Score. API accepts 2–255 / 2–10 options, but the **benchmark's realised `k` is `[2]` / `[4, 5]` / `[4, 5]`** (measured — do not quote the API range as the benchmark's) |
| **Measured noise floor** | case-level bootstrap SE **1.0668 pp**; **paired**-difference SE **0.8431 pp**; design effect **1.384**; accept threshold τ = `1.96·sqrt(SE_paired² + 2·Floor_B²)` — a **curve** in the unmeasured Floor B, whose breaking point vs the published +2.25 pp margin is **Floor B = 0.551 pp**. From `measurement/measure_noise_floor.py` + `measure_floor_b.py`, 2026-09-27. **Still a curve for the AgentJev arm (hardware-blocked); see the next row for the arm that is actually measured.** |
| **C1 arm noise floor — MEASURED 2026-09-27** | **Floor B = 0.40 pp** (SD across **5 pre-registered replicates**, **df = 4**; accuracies 0.5260–0.5360). Paired Δ vs the untrained arm **+18.00 pp**, SE_paired **1.484 pp**, 95% CI **[+15.10, +20.90]**, p ≈ 0 → **τ = 1.96·sqrt(SE_paired² + 2·Floor_B²) = 3.11 pp**. Decomposition: **evaluation variance 87.3%, run-to-run 12.7%**. The df = 1 pilot overstated Floor B by **3.53×** (1.41 → 0.40 pp), which is why it was never quoted. From `measurement/run_floor_b_laya.py` → `measurement/floor_b_laya_runs/floor_b_laya.json` |
| **What τ = 3.11 pp implies** | The C1 arm holds **17.26 pp** of legitimate headroom vs the 0.7036 ceiling (D06) — **5.5× τ** — so the accept rule is usable rather than vacuous. **Conversely, the published +2.25 pp Laya margin does not clear τ = 3.11 pp**: that claim cannot be made on this instrument. τ = 3.11 pp replaces the Floor-B = 0 lower bound of 2.96 pp, adding only 0.15 pp. |
| **Accept-rule audit — false-accept rate under a TRUE NULL (the empirical case for the hypothesis)** | The 5 pre-registered replicates differ **only by seed**, so all **10 unordered pairs** are trials under a true null. Measured: **max\|Δ\| = 1.00 pp**, **SD(Δ) = 0.58 pp**, vs **τ = 3.11 pp**. The brief's original rule — *frozen evaluator + unbounded selection + accept-if-improved* — **fires on 6/10 = 60%** of pure-noise comparisons (95% CI [26.2, 87.8]%). **The τ rule fires on 0/10** (95% CI [0.0, 30.8]%). Projected at the measured SD: naive **50.1%**, τ **≈4.6e-08** (z = 5.34; analytic, assumes normality — the empirical 0/10 is the hard evidence, the projection is an extrapolation). From `measurement/accept_rule_audit.py` → `measurement/accept_rule_audit.json` |
| **Same-seed control — the trainer is NOT deterministic given a seed (final, 2026-09-27)** | Five runs of seed 20260921 with identical flags: **0.5300, 0.5280, 0.5285, 0.5350, 0.5310** — spread **0.70 pp**. `SD_within` = **0.278 pp** (df=4) vs `SD_between` = **0.400 pp** (df=4), ratio **0.696** with **95% CI [0.225, 2.157]**. Firm finding, independent of sample size: **the trainer does not reproduce given a seed** (step 1 is bit-identical every time, so divergence arises later — CUDA reductions). **Ratio status: INDETERMINATE** — 0.696 sits on the pre-registered 0.70 boundary and the df(4,4) interval spans it, so it is recorded as undetermined rather than rounded to a verdict. An earlier df=1 version of this control was correctly rejected by the project's own standard (the error that overstated the pilot's Floor B by 3.53×); AMENDMENT 3 raised it to df=4. `SD_within` is **not** added to τ. `measurement/sameseed_control_analysis.py` → `measurement/sameseed_control.json` |
| **Disk governance — retention policy + measured budget (2026-09-27)** | One C1-arm run costs **1.20 GiB**, of which **1,287,653,720 B is `model.safetensors`** (321.91M params × 4 B, fp32) while the artifacts the analysis actually reads — `predictions.json` (501 KB) + `metrics.json` (371 B) — are **0.04%** of it. 11 checkpoints = **13.19 GiB**. `measurement/prune_checkpoints.py` classifies by group (ACTIVE never pruned / pre-registered KEEP / number-already-frozen PRUNE), **defaults to dry-run**, and refuses to delete a weight with no analysis artifact beside it. Applied: **7.20 GiB reclaimed (14G → 6.1G)**, 6 artifacts kept per run, and the same-seed analysis plus the Floor B artifact re-verified byte-identical afterwards. Weights are derived (seed + config + ~7 min GPU); the numbers are the evidence. |
| **The hill-climb — one dataset, two rules (MEASURED 2026-09-28)** | 5 pre-registered proposals, selection on **dev**, test touched **once**. **τ_dev = 6.05 pp** (dev SE_paired 3.04 pp) vs **τ_test = 3.11 pp** — a test-derived τ used for a dev decision would accept **94% more noise** than the split allows. Verdicts: it1 `steps=3000` dev **0.5567, Δ +2.83 pp → REJECT**; it2/it3/it4 all **0.5217, Δ −0.67 pp → reject**. **0 accepted.** The **naive running-max chain rises 0.5283 → 0.5567 (+2.83 pp), entirely noise-chasing**, while the **τ chain stays flat at 0.5283**. Final **test 0.5290** — **0.20 pp = 0.50 SD from the pre-registered protocol mean 0.5310**, i.e. the loop correctly changed nothing. Figure `figures/out/Fig_hillclimb.png` (QC PASS: 28 text artists, 0 clipping, 0 overlap). **DEVIATION 1:** proposal 5 (`batch_cases=8`) was stopped mid-run on cost grounds (allocator slow path, below) and is recorded as **failed**; the result is stated as **0/5 with 4 completed**, never as a full 5-proposal schedule. |
| **A discrete allocator slow-path costs up to 10× step time (diagnosed 2026-09-28)** | Step-time trajectories are **step changes, not drift**: it4 holds **0.199–0.203 s/step through step 801, then jumps to 0.688 s/step** and never recovers; it5 (`batch_cases=8`) runs **~2.0 s/step from step 1**. Ruled out: thermal/power (55 °C, all HW and SW throttle reasons **Not Active**, 62 W of 200 W), host pressure (89 GiB RAM free), and sequence length (`max_len_seen = 666` identical in every iteration). With `peak_reserved = 10.395 GiB` of ~11.9 GiB available, this is the allocator's slow path near the ceiling — the failure mode already recorded in `llm-training-instrumentation`. **Consequence:** ~70 min instead of ~40. It does **not** invalidate the result: the protocol is **step-bounded**, so wall-clock cannot change what was computed. |
| **Disk: prune by OUTCOME, not by directory name** | The climb re-created the 6 GiB that had just been reclaimed — every iteration writes a **1.20 GiB** checkpoint while only the **surviving endpoint** is ever re-read. `prune_checkpoints.py` now reads `hillclimb.json` and retains only the accepted arm (or the baseline when none was accepted), pruning rejected/failed iterations because their dev number is already frozen in `metrics_*.json`. **Reclaimed 4.80 GiB (13G → 7.3G)**, 5 artifacts kept per iteration. |
| **Pipeline audit — 12 PASS / 0 FAIL (2026-09-28)** | Code-verified structural review, run by `measurement/audit_pipeline.py` (exits non-zero while a defect stands) and now a **preflight gate inside the driver** alongside the purity guard. **Found and fixed: (1) results were not bound to their artifacts** — `train()` reused any checkpoint on file existence and `evaluate()` returned any metrics file without checking which weights produced it, so a changed config under a stable name scored the OLD arm; now `run_stamp.json` (config) + a zlib weights fingerprint are required on both paths, verified byte-identical (`0.528333` dev, `0.529000` test) after rebinding. **(2) τ was built from the wrong variance** — `SE_paired` of an untrained→trained jump (≈18 pp apart) instead of the decision's own noise (two comparably trained arms); **over-inflated 2.5–2.7×**. Dev decision SD **MEASURED** at **1.22 pp** (5 retained replicate checkpoints scored on dev, 1.06 pp predicted by scaling) → τ_dev 6.05 → **2.39 pp**. **(3) no multiplicity control** — α=0.05 per decision reaches FWER **1.0000 over 1000 decisions**; τ_dev becomes **4.95 pp**. **(4) the loop was nearly powerless**: the original τ needed a TRUE gain of **7.08 pp** for 80% power against 17.26 pp of headroom — it could never close its own loop. **Verdict on the climb is robust**: it1's +2.83 pp is rejected at 6.05 pp and at 4.95 pp, but **would be accepted at 2.39 pp** — so the multiplicity term is load-bearing, not decorative. **(5) `hashlib → torch → pyarrow` SEGFAULTS** (8/8, no traceback) — existing scripts were safe only by import-order luck; now checked structurally and `provenance.py` uses zlib. Full record: `docs/JevRSI_pipeline_audit_2026-09-28.md` |
| **Numerics floor + a published-artifact inconsistency** | **Split by measurement.** *Numerics:* a dtype change (bf16↔fp32) moves the logits by only **0.0075 mean abs Δlogit ≈ 0.15 pp Top-1** on a fixed machine, and `transformers` 4.51.0 ≡ 5.3.0 **to 15 digits** — so the per-run numerical cost is **small**. *Artifact:* re-evaluating the **published weights** gives **1617/2000 = 0.8085** (bf16) / **1614/2000 = 0.8070** (fp32) vs the published **1585/2000 = 0.7925**, with `D_ours_pub` = **0.876** — **~116× the precision term**, so it is **not** floating-point noise. The repo's stored predictions are therefore **irreproducible from its own released checkpoint**, which means **they cannot serve as ground truth** and any comparison against them (incl. the +2.25 pp Laya margin) inherits the 1.60 pp. See `docs/JevRSI_retraining_readiness.md` §4 |
| **Programme document** | `docs/JevRSI_program.md` |
| **Pre-registration** | `docs/JevRSI_preregistration.md` — the pre-committed accept rule, optimand, and the τ(Floor B) curve |
| **Measurement** | `measurement/measure_noise_floor.py` (Floor A, paired SE, endpoint) + `measurement/measure_floor_b.py` (AgentJev Floor B, τ) + `measurement/run_floor_b_replicates.py` (**the producer** — trains `--n` seed replicates and writes the `floor_b_runs/*.json` contract that `measure_floor_b.py` reads; refuses to run on an impure tree or below a VRAM headroom floor) + `measurement/measure_floor_b_nanochat.py` (harness Floor B, η² on `num_steps`) |
| **C1 arm measurement** | `measurement/floor_a_cluster_bootstrap.py` (Floor A for the C1 pair, cluster-aware) + `measurement/run_floor_b_laya.py` (**C1 Floor B producer**) + `measurement/train_laya_arm.py` (optimiser loop only — network, sequence builder, collation and objective are all imported from `laya`) + `measurement/eval_laya_checkpoint.py` (evaluates in a **separate process**; see `floor_b_laya_prereg.md` AMENDMENT 1) + `measurement/verify_laya_forward_equivalence.py` (training forward vs served probabilities) |
| **Evidence annex** | `docs/EVIDENCE_prior_art_and_feasibility.md` |
| **Document index** | `docs/DOCUMENT_MAP.md` |

**The scientific gap (G1, primary):** the brief specifies a *frozen evaluator + unbounded selection loop
+ accept-if-improved* — a selection operator on a noisy statistic. The completed v1 run already
implements the correct discipline (dev split for checkpoint selection, a separate calibration split for
temperatures, `selection_completed_before_test: true`, revision pinned by sha256). **The proposed 24/7
loop discards every one of those protections.** Read `docs/JevRSI_program.md` §3.1 before designing any
experiment.

---

## The C1 arm — `laya-multilingual` (322M) as a LOW-START subject

**Registered 2026-09-27.** Scientific form **(C1)**: pair the existing high-start subject with a
**low-start** one, so the selection operator is studied in a regime that genuinely has headroom,
without altering the pre-registered claim.

| | |
|---|---|
| **Checkpoint** | `convaiinnovations/laya-multilingual` — 322M; encoder `jhu-clsp/mmBERT-base` (ModernBERT 768 / 22L / vocab 256k), head 2 transformer layers (ffn 3072, nhead 12, dropout 0.1). **321.909 M params, 170 tensors** |
| **Why low-start** | measured Top-1 **0.3500** on the frozen test split, vs the **0.7036** chance-corrected ceiling (D06). AgentJev sits at **0.7925** — *above* the 0.75 saturation point — so part of its apparent headroom was illegitimate. This arm's is not. |
| **Publisher ships the training toolkit** | `laya.common` exposes `DecisionModel`, `build_model`, `build_sequence`, `collate_items`, `proper_reward`, `td_lambda_targets`; **only the optimiser loop is ours**. Attribution: network ✓ publisher · baseline ✓ publisher · objective ✓ publisher · trainer ours. |
| **Floor A (cluster bootstrap)** | SE_cluster 0.01057 (deff 1.24) / 0.01130 (deff 1.11); paired Δ = **−0.4190**, SE 0.01509, 95% CI [−0.4485, −0.3895] |
| **Floor B — MEASURED under the pre-registered protocol** | 5 seeds × 2000 steps, batch-cases 4, LR 1e-5 / head 1e-4 → accuracies **0.5300 / 0.5340 / 0.5290 / 0.5360 / 0.5260** (mean **0.5310**), **Floor B = 0.40 pp, df = 4**. Paired Δ vs the untrained arm **+18.00 pp**, SE **1.484 pp**, 95% CI **[+15.10, +20.90]**, p ≈ 0 → **τ = 3.11 pp**. Pre-registration `measurement/floor_b_laya_prereg.md`; artifact `measurement/floor_b_laya_runs/floor_b_laya.json` |
| **Pilot — 400 steps, 2 seeds, exploratory, NOT the protocol** | base 0.3500 → **0.5050 / 0.4850**; paired Δ **+0.1550**, SE 0.01656, 95% CI [+0.1230, +0.1875], p < 1e-4. Kept under `measurement/floor_b_laya_pilot400/`. **Its Floor B (1.41 pp, df = 1) overstated the pre-registered value by 3.53×** — retained only as an illustration of why a 2-sample SD must not be quoted |
| **Verification** | three-way: publisher class ≡ checkpoint ≡ our independent reconstruction (170/170 keys, 0 shape mismatches, 321.909 M both). Training forward ≡ served probabilities to max\|Δ\| **4.9e-05**, argmax **15/15** — see `measurement/laya_reconstruction_feasibility.md` |
| **Wedge, measured and fixed** | Floor B first stalled on step 2 of *every* replicate — a driver holding a GPU-resident Agent while spawning the training child. The same trainer **standalone** ran 400 steps at 0.27 s/step. Fix: evaluate in a subprocess that exits first. **No scientific parameter changed** (AMENDMENT 1 in the prereg). |

**Footprint — measured, not inferred.** Peak reserved is **~11.5 GiB** on this 12 GiB card, not the
4.80 GiB a static params+grads+optimiser sum predicts. A 3-step probe reported only 6.3 GiB because its
sequences were 167–187 tokens while the builder allows 1024 and attention is O(L²). Every run now writes
`peak_reserved_gib` and `max_len_seen` into its `meta.json`.

---

## ⛔ Retired — do NOT work on these

These were real work by this project and are preserved under `archive/`. **None of them is the current
object.** If you find yourself reading, extending, or citing them as live, stop.

| retired object | what it was | why it is not the object | where it lives now |
|---|---|---|---|
| **prompt-multiplier / HPFE / OASP / κ** | optimise the *prompt*; "Heuristic Prompt Fixation Effect"; prompt-space search | **AgentJev is not prompted** — it consumes structured state and is not a generative model. The names cannot be reused without equivocation. | `archive/retired_object_prompt_multiplier_2026-09-27/` |
| **Gate B / recursive self-training collapse** | nanochat `val_bpb@300s` under synthetic-corpus generations | a different topic: **data-policy** sensitivity, not training-code search | `archive/orthogonal_gate_b/` |
| **Zarankiewicz / cap-set benchmark selection** | extremal-construction topic selection | a wrong turn: an *example* in the brief, adopted as *the* topic | `archive/wrong_turns/` |
| **MentalBench / MentalHealthBench** | a candidate downstream benchmark | superseded — the new object's benchmark is `LocalLLaMA/typed-decisions` | `archive/superseded_2026-09-27/` |

### The trap this page exists to prevent

`archive/orthogonal_gate_b/code/configs/program_variants/assembled/v00–v12.md` are **thirteen
12–13 KB files that open exactly like the `program.md` the current brief asks for**:

```
# autoresearch

This is an experiment to have the LLM do its own research.
```

They are the **old** object's experimental arms (each mentions `nanochat` / `val_bpb`) — the vocabulary
overlaps the harness's, but these are *data-policy* experiments, not JevRSI constitutions. **JevRSI's own
`program.md` does not exist yet and must be written**; the task is a *transplant* of the harness's
protocol (`docs/JevRSI_program.md` §2.6), not a design from scratch. Note the harness's rejection
protocol is currently **specified but unexercised** (0 `trajectory/` directories).

---

## How to tell you are off-target

Any one of these means you have picked up a retired object:

- you are working on **acronym expansion**, **LLM self-mistranslation**, **"Master of Laws"**, or
  **FDLH / Frequency-Dependent Learning** — that is **retired object #1** (the NHB
  LLM-mistranslation workspace), whose MLE objective was audited as tautological. It is a *different
  project*; none of its findings transfer to a non-generative typed-decision model. `NHB-mistranslation`
  in `RETIRED_CONCEPTS` now fails the build on any such reference
- you are editing or referencing a **prompt**, a prompt template, or a "prompt space"
- you are using **HPFE**, **OASP**, or **κ** as if they named the current object
- you are working on **`val_bpb`**, **nanochat**, **TinyStories**, or a synthesized corpus **as the
  object of study** — i.e. treating data-policy sensitivity as the topic. Naming them while *running*
  the `autoresearch-win-rtx` harness is on-target: there the harness is the **instrument**, and the
  object is still `AgentJev-0.6B`'s training code. The guard enforces exactly this distinction
  (`HARNESS_CONCEPTS` + CHECK D), so a file may name them only if it declares the harness.
- you are editing anything in `archive/`
- you are looking for a file that is now at `archive/...` and not finding it live

Mechanical check — run this after any change to the tree:

```bash
python scripts/check_object_purity.py
```

It asserts the **absence** of retired-object vocabulary from every live entry point, and the **absence**
of retired artifacts from the live tree. It exits non-zero on violation. A validator that only asserts
the *presence* of correct values cannot catch a silent revert; this one asserts absence by design.

Five checks, all fatal:

| check | asserts |
|---|---|
| **A** | no retired artifact path exists in this repository |
| **B** | no live file carries an *actionable* retired instruction (a command/import that would run the old object) |
| **C** | a live file may name a retired concept only if that same file declares the retirement (prior-art annexes under `docs/evidence/` are exempt from this one rule — they must be able to cite the published literature by name) |
| **D** | `CURRENT_OBJECT.md` really does declare the harness, so the `HARNESS_CONCEPTS` allowance cannot outlive its anchor |
| **E** | the **subject repository** (`E:/2026-AI4S/agent-jev`, where replicates actually execute) carries the same vocabulary ban — override with `JEVRSI_SUBJECT` |

> **Object #1 is now enforced.** The guard's own history lists *the NHB LLM-mistranslation workspace* as
> retired object #1, but no pattern for it existed, so an FDLH / acronym-expansion / self-referential
> mistranslation reference would have drifted the work back to a **falsified** programme while the guard
> still reported PASS. That blind spot is closed as of 2026-09-27: `NHB-mistranslation` is a
> `RETIRED_CONCEPTS` entry, and **E** extends the ban to the subject repository, which the guard
> previously did not read at all.

`measurement/run_floor_b_replicates.py` calls this guard **before spending GPU time** and again before
**every replicate**, so an experiment cannot be produced on a drifted tree. `--skip-purity-check` exists
only for reproducing archived results and marks every resulting row provisional.

---

## The brief

**`project-2.docx`** (md5 `ababd76a34adc2376bcb2d81da390c9c`, 24,679 B) — the JevRSI re-location.
The four `Problem*.docx` files are the **previous** object; `Problem.docx` at the repo root is
byte-identical to `Problem-4.docx` and is **superseded**.

Lineage: `c49956fd…` → `d01242eb…` → `1fb0ab71…` → `8be65ee4…` → **`ababd76a…`** (current).
