# Retraining readiness — AgentJev-0.6B

**Question asked:** do we *completely* have the conditions to retrain `AgentJev-0.6B`?
**Answer: yes for evaluation and loop development; no for exact reproduction of the published training
run — and that second gap is a *publication* gap, not a local capability gap.** Everything below is
verified on disk on this machine (RTX 4070, 12 GB), not inferred.

Sandbox: `E:/2026-AI4S/jev_repro/` (subject repo `E:/2026-AI4S/agent-jev` is read-only throughout).

---

## 1. Prerequisite checklist

| # | prerequisite | status | evidence |
|---|---|---|---|
| 1 | Benchmark data (`LocalLLaMA/typed-decisions` @ `ea9306458d6e9563628369a3d1e72e362fb381d2`) | **HAVE** | `typed_decisions/data/all/{train,test}-00000-of-00001.parquet`, 36 MB |
| 2 | Split reproduction | **EXACT** | train 960 / dev 120 / calib 120 / test 400 cases (2,000 questions); **all four case-id sets set-equal** to the published `split_manifest.json`, seed 20260921 |
| 3 | Training JSONL in `data.py`'s schema | **HAVE, 0 rejections** | `train_jsonl_stats.json`: train 4,800 q / dev 600 / calib 600, `_validate_gold` rejects 0 |
| 4 | Base model | **HAVE** | `models/Qwen3-0.6B` — `model.safetensors` 1,503,300,328 B; qwen3, hidden 1024, 28 layers, vocab 151,936 |
| 5 | Published trained weights | **HAVE** | `models/agent-jev/model.safetensors` (the 79.25% endpoint) |
| 6 | Weights load into `AgentJevModel` | **EXACT** | **343/343 keys, missing = 0, unexpected = 0, `strict=True` load succeeds** |
| 7 | Faithful code path importable | **HAVE** | `jev_service.contract.encode_paths` ✓, `agentjev.model.AgentJevModel` ✓ |
| 8 | Runtime | **HAVE** | torch 2.13.0+cu126 (`cuda.is_available()` True), transformers 5.3.0, Py 3.14 |
| 9 | Hardware / disk | **HAVE** | RTX 4070 12 GB (~11.3 GB free), E: 3.0 TB free |
| 10 | **Initial checkpoint of the published SFT run** | **MISSING** | `experiment.py:20` `START='/root/agentjev/runs/phase4/final.pt'` — never released |
| 11 | A completed training run on this machine | **YES — but not viable** | 2-step smoke ran to exit 0 and saved a 2.39 GB checkpoint. **OOM cascade, ~470–1350 s/step vs the published 1.4 s/step. See §4b.** |
| 12 | Acid test (reproduce 1585/2000) | **DONE — does NOT reproduce** | **1617/2000 = 0.8085** vs published 1585 = 0.7925. See §4. |
| 13 | 24/7 loop automation | **ABSENT** | no cron in either repo; harness `.github/` holds only `FUNDING.yml` |
| 14 | Seed control (Gate G1) | **ABSENT** | neither `AgentJev/train.py` nor the harness `train.py` exposes `--seed` |

**9 of 14 fully satisfied.** The failures split cleanly into *publication gaps* (10, 13, 14 — nothing local
can close them) and *execution pending* (11, 12 — ours to close, running now).

---

## 2. Blocker A (hard): the published run cannot be re-run

`typed_decisions/experiment.py` is the driver that produced 79.25%. Its initial checkpoint is
`/root/agentjev/runs/phase4/final.pt` — an **absolute path from the authors' machine**, referenced only
in `protocol.json`, and **absent from both the repo and the Hub**. The published weights are the *final*
state of that run (38.70% → 79.25% by 600-step SFT); the *entry* state was never released.

**Consequence, stated plainly:** the published training trajectory is **not reproducible by anyone**,
on any hardware. This is not a limitation of our setup. It belongs in the manuscript as a stated
limitation of the baseline, not as a work item.

## 3. Blocker B (real): the reconstruction is a *different experiment*

`experiment.py` is **self-contained** — it imports only `agentjev.model` and `jev_service.contract`, never
`agentjev/train.py` or `agentjev/data.py`. So the reconstructed config, however carefully derived, drives
a different program. Five measured divergences:

| # | published `experiment.py` | reconstructed `train.py` + `data.py` | impact |
|---|---|---|---|
| 1 | `encode_paths(max_tokens=2048)`; line 101 writes `'truncated':0` as an assertion | `max_state_tokens=256` truncates **252/960 train states (26.2%)** | inputs differ |
| 2 | `microbatch_questions` = 4 **questions**/microbatch | `batch_states`=4 collates whole **states** ≈ 20 questions/microbatch | ~**5×** effective batch |
| 3 | **2** optimizer groups (backbone, head) | **4** layer-wise groups | different LR application |
| 4 | `warmup=20`, a **literal** | `warmup_ratio: 0.02` → `max(1,int(600·0.02))` = **12** | different schedule |
| 5 | `(s+1)/warmup` → LR never 0 | `step/warmup` → **LR = 0 at step 0** (measured: `after scheduler init: [0.0, 0.0]`) | one no-op update |

Divergence 5 is a genuine **bug** — but it exists only on the reconstructed path, so it must be reported as
such rather than as a defect in the published result. Also note `experiment.py:19` uses
`Qwen3-0.6B-Base` while we hold `Qwen3-0.6B`; minor, because it supplies only the tokenizer and the
architecture config, and init weights are overridden anyway.

## 4. What *is* proven — and the finding the acid test produced

The **evaluation half** is verified: split exact, weights load strictly (343/343, `strict=True`,
`missing=0`), faithful path importable, data valid with 0 rejections.

**The acid test ran and it does NOT reproduce.** Evaluating the published weights, on the published
2,000-question test split, through `experiment.py`'s own functions:

| | correct | accuracy |
|---|---|---|
| **our re-evaluation** (RTX 4070, torch 2.13.0+cu126, transformers 5.3.0) | **1617** / 2000 | **0.8085** |
| published headline (README) and the repo's own `test_calibrated_predictions.json` | 1585 / 2000 | 0.7925 |
| **difference** | **+32 items** | **+1.60 pp** |

Two independent implementations agree on 1617, and the published per-item predictions reproduce 1585
under **both** the raw-logit and calibrated conventions — so the published side is self-consistent and
the discrepancy is real.

**Localisation (the informative part):** comparing per item against
`test_calibrated_predictions.json`, the two agree on **1875/2000 (93.75%)** of items, while
`mean |Δlogit| = 0.866` and `max |Δlogit| = 1.984`. **The logits themselves differ**, so the cause is in
the forward pass, not in the split, the metric, or the weight loading. 125 items flip.

**Candidate causes (NOT yet isolated — attribution is an open question, deliberately not guessed):**
`evaluate()` runs under `torch.autocast('cuda', dtype=torch.bfloat16)`; the original ran on a different
Linux GPU; and PyTorch's `TransformerEncoder` emitted `enable_nested_tensor is True, but
self.use_nested_tensor is False because encoder_layer.norm_first was True` — a version-dependent
nested-tensor path. **The strongest lead is now concrete: both published configs declare
`transformers_version: "4.51.0"`, while this machine runs `5.3.0`** — a major-version jump (4.x → 5.x),
which is exactly the kind of change that moves numerics and changes which encoder path is taken. Two
supporting measurements: **the logit multisets match for only 4/2000 items**, so this is *not* a
candidate-permutation or ordering effect (mean per-item logit correlation 0.967); and the checkpoint's
dev-split uncalibrated soft CE is **0.828846** against the run's own `selection.json` value of
**0.814509** (Δ 0.0143, 1.8% relative) — the discrepancy is present on dev too, not only on test.

**Decisive experiment — RUN, and it REFUTES the library-version hypothesis.** Re-running the acid test
under the pinned `transformers==4.51.0` (isolated in `_tf451/`; ships `tokenizers 0.21.4`) gives
**exactly the same numbers as 5.3.0** — accuracy **0.8085 (1617/2000)**, soft CE
**0.8671486395529757**, ECE **0.13129953988795498**, identical to 15 significant digits, with the same
343/343 `strict=True` load. So the 4.x→5.x jump is **not** the cause and the model path is
version-stable. That eliminates the only *controllable* candidate and leaves two:

- **(a) hardware/kernel numerics** — the published run was on a different (Linux) GPU, and a mean
  |Δlogit| of 0.866 is well within what bf16 accumulation across 28 layers plus a `TransformerEncoder`
  can produce when the kernel implementations differ;
- **(b) the Hub checkpoint is not bit-identical to the `best.pt` that produced the stored predictions**
  — the child's alternative, and consistent with the dev-split mismatch (our 0.828846 vs the run's own
  `selection.json` 0.814509) appearing on dev as well as test.

**Decisive test — RUN, and the result is the OPPOSITE of "irreducible noise".** Re-running identically
with **autocast disabled (fp32)** gives **1614/2000 = 0.8070** (+1.45 pp) against the bf16 run's
1617/2000 (+1.60 pp), and decomposes the logit distances:

| distance | value | what it isolates |
|---|---|---|
| `D_bf16_fp32` — our fp32 vs **our own** bf16 logits | **0.0075** | the pure precision term, same weights, same machine |
| `D_ours_pub` — our logits vs the **published** ones | **0.8758** | the gap to be explained |

**The precision term is ~116× smaller than the publication gap.** So bf16 is *not* the cause, and neither
was the library version — **and critically, the 1.60 pp is NOT floating-point noise.** It cannot be: if
merely changing dtype moves the logits by 0.0075, no numerical mechanism accounts for 0.876. A genuine
numerics explanation is therefore excluded, leaving a **checkpoint-identity or rendering difference
between the artifact the repo published and the artifact the repo's own Hub checkpoint produces.**

Precision does have a small real effect on Top-1 — 3/2000 = **0.15 pp** (1617 bf16 vs 1614 fp32) — but
that is ~10× *smaller* than the 1.45–1.60 pp gap. Both runs use `experiment.py`'s own
`tokenize()`/`batch()`/`evaluate()`, so the rendering is theirs, not ours.

### Consequence: the loop's noise budget is much better than feared

I previously recorded Floor C as "≥1.60 pp, environment/numerics". **That conflated two different
things** and must be split:

- **True numerics floor (measured): ~0.0075 mean |Δlogit| ≈ 0.15 pp on Top-1**, for a dtype change on a
  fixed machine. This is the part a loop actually pays per run, and it is **small**.
- **The 1.60 pp is a published-artifact inconsistency**, not a per-run noise floor: the repo's stored
  `test_calibrated_predictions.json` is not reproducible from the repo's own released checkpoint. It is a
  one-off integrity finding about the subject repository, and it must be **excluded from the noise
  budget** — while also meaning the published per-item predictions **cannot be used as ground truth**.

**Net effect: the accept rule is numerically viable on this hardware.** A single consecutive-run
comparison pays ~0.15 pp of precision noise, not 1.6 pp, provided the environment is held fixed and the
published per-item predictions are not treated as the reference. Caveat, stated honestly: we measured the
**dtype** component; cross-GPU (Ada vs the authors' hardware) is untested — but it is now the only
surviving numerics candidate, and a 0.876 gap is implausible for a term that is 0.0075 within a machine.

### Why this matters more than the acid test was supposed to

The programme's accept rule is `τ = 1.96·sqrt(SE_paired² + 2·Floor_B²)` with a paired SE of **0.8431 pp**.
A **re-implementation of the same weights** shifts the metric by **1.60 pp** — roughly **twice** the paired
SE, and above the 0.551 pp breaking point at which the published +2.25 pp Laya margin stops being
claimable. The follow-up experiments then **split that 1.60 pp into two very different things:**

- Floor A — measurement/eval sampling: **1.0668 pp** SE
- Floor B — training-seed reproducibility: *still unmeasured* for AgentJev (no `--seed` exists — item 14)
- **Numerics floor (measured): ~0.0075 mean |Δlogit| ≈ 0.15 pp** Top-1 for a dtype change on a fixed
  machine. **This is the per-run cost, and it is small.** Library version contributes nothing
  (`transformers` 4.51.0 ≡ 5.3.0 to 15 digits).
- **NOT a noise floor — a published-artifact inconsistency: 1.60 pp.** The repo's stored
  `test_calibrated_predictions.json` is not reproducible from the repo's own released checkpoint
  (0.876 mean |Δlogit|, ~116× the precision term). It belongs in the integrity findings, **not** in the
  noise budget — and it means the published per-item predictions **cannot serve as ground truth**.

**This sharpens G1 rather than weakening it.** The loop's per-run numerical cost is only ~0.15 pp, so a
sub-1.6 pp accept threshold is *not* automatically meaningless — but the subject repository's published
per-item predictions are **not a valid reference**, being irreproducible from the repository's own
released checkpoint. Any comparison against the published artifact — **including the +2.25 pp Laya
margin** — inherits that 1.60 pp inconsistency. The pre-registered baseline must therefore be
**re-measured locally from the weights**, never taken from the published prediction files.

## 4b. The completed smoke run — plumbing proven, throughput fatal

The last outstanding execution item is now closed: a 2-step run (the corrected `PYTHONPATH`, see
below) **ran to exit 0 and saved a checkpoint**. What it proves and what it disproves are both useful.

**Proven (the plumbing works):**
- published weights initialize into the model — `params: 598.4M`, `aggregate 343/343 strict`
- data loads and mixes — `sources` breakdown + `source_mix: {typed_decisions_train: 16}`
- forward + backward + `opt.step()` execute; `grad_norm` finite (1.65, 6.74)
- **checkpoint saves**: `runs/jev_v1_smoke2/final.pt` = **2,393,802,903 B (2.39 GB)** = 598.4M params × 4 B, self-consistent
- `skipped_batches=0 dropped_questions=0`; batches are `4 states × 4 grad_accum = 16 states = 80 questions` per optimizer step — which independently confirms divergence #2 (**80 questions/update vs the published 16**, ≈**5×**)

**Disproven (this configuration cannot train):**
- **OOM cascade**: dozens of `memory allocation failed with OOM on device 0 (free: 0, total: 12878086144)` — free space reached **0** and PyTorch entered an allocator-retry storm
- **~470–1350 s per step** vs the published **1.4 s/step** (600 steps in 841 s). Extrapolating: **~78 hours** for the published schedule, ≈**600× slower**
- **126 tok/s** peak (step 2), i.e. a fraction of one percent of expected 0.6B throughput
- **LR = 0 on both steps**: at construction `LambdaLR.__init__` calls `step()` → lambda = `step/warmup` = 0, so step 1's `opt.step()` is a **silent no-op**; and because this config set `max_steps: 2`, the cosine schedule had already decayed to 0 by step 2. The loss change (0.8198 → 0.8148) is therefore **different batches, not learning**

**CORRECTION (mine, retracted).** I first called the VRAM figures *impossible → instrumentation bug*.
**They are real, and the mechanism is worse than a bug.** This RTX 4070 runs the Windows CUDA
**sysmem-fallback** policy: instead of raising OOM it silently spills to host memory, so
`max_memory_allocated()` legitimately exceeds physical VRAM — 21.5 GB (`batch_states` 1), 33.85 GB
(`batch_states` 2), and **62.343 GB allocated / 79.822 GB reserved** here on a 12.878 GB card. The
`free: 0` OOM lines are the allocator thrashing *inside* that spill. Fixed fp32 AdamW alone needs
**9.574 GB** (2.394 GB each for params, grads, `exp_avg`, `exp_avg_sq`) against 10.537 GB free, so there
is almost no headroom and every extra allocation spills.

**This is the actual cause of the ~900 s/step — not Python overhead.** Fwd+bwd runs at 46.7–126.4 padded
tok/s against **7,079–16,583 tok/s forward-only** (~56–95×), and spilling explains the gap. I had also
called this a *clean* measurement; it was not — the child had launched a **duplicate of the same config
~26 s apart** and killed its copy at 13:59, so step 1 (1350.27 s) was contended and step 2 (474.23 s) was
not. **The honest figure is ~474 s/step at best — still ≈340× the published 1.4 s/step, i.e. ~79 h** for
the published schedule. A 600-step run is **not viable on this card** at these settings.

- **`trunc_states: 5` then `4`** per step — divergence #1 confirmed at runtime. The published run
  asserts `'truncated': 0`; this path drops ~28% of states per step, matching the 26.2% audit.

**Shell note:** the run only started after quoting `PYTHONPATH`. Unquoted, git-bash treats `;` as a
command separator and tries to *execute* the second path — `Is a directory`, exit **126**, no Python
process at all. That is why the first attempt produced nothing.

## 4c. The subject repository's own artifacts, audited

40 quantitative claims re-derived from the shipped per-decision files with **verbatim copies of the
repo's own `experiment.py` metrics(), `summarize.py` bootstrap() and `evaluate_laya.py` ingestion**:
**20 AGREE / 8 DISAGREE / 12 UNVERIFIABLE**
(`measurement/agentjev_typed_decisions_artifact_consistency_audit.json`).

**The 20 agreements matter, because they relocate the critique.** The headline `1585/2000 = 0.7925`,
the per-primitive split (0.8883 / 0.7533 / 0.7500), the pooled-vs-per-primitive reconciliation, all Laya
and phase-4 metrics, the **+2.25 pp gap and its case-level bootstrap CI [+0.65, +3.90] bit-exactly**, the
serving error `1.0652881710093709e-07` with 20/20 argmax matches, and Laya's P50/P90 = 41.532/47.142 ms
are **all internally consistent and reproducible**. So the disagreements below are **not arithmetic
errors — they are selection and framing choices**, which is the harder criticism to dismiss.

**D06 (HIGH) — the headline exceeds the benchmark's own stated ceiling, and the repo never says so.**
The pinned dataset card states *"Teacher self-agreement | 0.735"*, *"Around 0.75 is saturation"*, and
*"A score much above 0.75 means a model has learned the teacher's quirks rather than the task"*, with a
factor ceiling of **0.704**. The published headline is **0.7925** and Laya's **0.7700** — i.e. **+5.75 pp
above the ceiling and +4.25 pp above the saturation point**, **on the benchmark the card describes**.
**No artifact in the repo** (`README.md`, `README_zh.md`, `REPORT_zh.md`, `protocol.json`,
`comparison.json`) mentions 0.704 / 0.735 / 0.520 or the saturation wording — yet the repo *does* copy
the card's **low** baselines (Prior 47.0%, Uniform 30.8%) into the same results table. Caveat, stated
honestly: the card's ceilings are given on the 1600-case set while the repo scores the 400-case test
split — so this is a strong flag, not a proven contradiction. It is nonetheless the sharpest available
form of the G6 concern: the benchmark's own documentation says a score above 0.75 indicates learning the
**teacher's quirks**, and the programme's baseline to beat sits above it.

**D01** — the README's `Uniform 30.8%` row contradicts the realised candidate counts (boolean `[2]`,
choice `[4,5]`, score `[4,5]`): the chance level is **0.3175** as mean(1/k) or **0.2775** as
argmax-of-uniform, and the published figure is reproducible under **neither**. The same row's
Brier 0.238 / KL 0.444 / TV 0.381 *do* reproduce exactly, so the row was computed on this split while its
accuracy figure was not.

**D02** — the published `2–255 options` range and the 64-option wide load are up to **51×** the realised
maximum of **5**; consequently **3 of 6 Laya temperature buckets** (choice:2, choice:6–10, choice:11+)
never fire on this split, and every wide-set result (2× speedup, 92.4% token reduction) is measured
*outside* the benchmark.

**D03 / D04** — the wide-load latency row attributes *"~500–600 ms"* to **Laya**, but the only such
artifact is **AgentJev's own** unshared `path` mode (609.65 ms); and the README's *"shared-prefix path
298.91 ms"* is the artifact's `auto` mode, while the artifact's actual `shared` mode (**348.56 ms**, 17%
slower) is never reported.

**D05** — one model, two published metric sets: temperature scaling makes the calibration metric
**worse** (ECE 0.1573 → 0.1687; Brier 0.0454 → 0.0448), and the README headline shows only the
**worse-ECE** calibrated set under the label *"this run"*.

**D07 / D08** — Laya's reported evaluation seconds (19.1831) exceed the sum of its own per-case traces
(18.9099, +1.4%, no artifact explains the gap); `data_metadata.json` declares `size_categories` as both
`1K<n<10K` and `n<1K`.

## 5. Verdict and recommended path

**Retraining from scratch to reproduce 38.70% → 79.25% is impossible** (Blocker A, permanently), and
**reconstructing it through `train.py` yields a different experiment** (Blocker B, by construction).

**But the JevRSI programme does not need either.** Its object is *search over training code against a frozen
evaluator*, and its baseline is the published endpoint. So:

1. **Adopt the published 79.25% endpoint as the frozen baseline**, not the unpublished phase-4 state.
   It is verifiable — the acid test is exactly that verification — and it is the number the README
   headline reports.
2. **Drive the loop with `experiment.py`'s own functions**, not `train.py`, so training and evaluation
   share one code path and the divergences above cannot silently enter.
3. **Record Blocker A as a stated limitation** of the baseline: the reference run's entry checkpoint is
   unreleased, so its training trajectory is unreproducible by any party.
4. **Close the two execution items** (11, 12) — a completed smoke run and the acid test number.
5. **Add `--seed`** outside the agent-editable surface (Gate G1); it is the precondition for Floor B.

**Honest summary:** we hold every artefact needed to *evaluate* this model exactly and to *develop and run*
the loop on it. We do not hold — and cannot obtain — the artefact needed to *re-run the published
training step by step*. That distinction should be stated in the manuscript rather than papered over.
