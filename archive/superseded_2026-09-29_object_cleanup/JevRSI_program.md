# JevRSI (JevRSI) — the re-located research object

**Briefs:** `project-2.docx` (= `project.docx`, md5 `ababd76a34adc2376bcb2d81da390c9c`, 24,679 B) —
a **change of object**, not a fifth revision of `Problem*.docx`.
**Subject repository:** [`github.com/malevrigns/agent-jev`](https://github.com/malevrigns/agent-jev),
cloned and read at `main @ a965ca8f` (2026-09-23), placed at `E:/2026-AI4S/agent-jev` (79 files, 5.8 MB).
**Status:** programme definition, **revised against the actual source** · date 2026-09-27

> **This revision exists because reading the repository refuted three claims I had made from the brief
> alone.** §2 records every retraction, including one that inverted what the project actually does. A gap
> analysis written without the source is a *hypothesis about the source*, and this one was wrong in three
> places. Where the two conflict, the source wins.

---

## 0. The object

**AgentJev-0.6B** — a *System One* decision model. Qwen3-0.6B backbone with the LM head removed, a
**permutation-equivariant candidate head** (`[C, 1024]` candidate vectors → `Linear 1024→256` →
residual → RMSNorm → SiLU → scalar). **Nothing is decoded**; one forward pass returns a distribution over
supplied options. Three primitives:

| primitive | option space | returns |
|---|---|---|
| Boolean | a proposition (+ optional true/false criteria) | `value`, P(true), both masses |
| Choice | **2–255** options | `value`, `top_probability`, `margin`, full map |
| Score | **2–10** ordered levels | `level` (argmax), `score` = Σ i·Pᵢ |

Hard context **2,048** tokens, and **over-length input is refused, not truncated** — *"a diff or a trace
that does not fit is an error, not a silent crop"* (`--max-tokens` default 2048,
`jev_service/server.py:44`). Serving binds **127.0.0.1 only**; there is an HTTP contract
(`agentjev.decision.v1`), a stdlib-only Python client, a workbench, and a Claude Code `PreToolUse` hook
gating Bash/Write/Edit. Weights are published (`aimeigaoshou/agent-jev`, Apache-2.0).

Local substrate verified 2026-09-27: `Qwen/Qwen3-0.6B` → 200, `LocalLLaMA/typed-decisions` → 200, target
org `github.com/Computational-social-science` → 200.

---

## 1. What the repository already establishes

This is **not** a greenfield project. The v1 result is finished, documented, and version-pinned:

| quantity | value |
|---|---|
| headline | **79.25%** Top-1 · 1585/2000 on the official 400-case / 2,000-question test split |
| reporting | per model: Top-1, **soft CE, Brier, ECE, Score MAE** — and **per primitive** and **per workflow** |
| vs Laya (published specialist encoder) | **+2.25 pp**, case-level bootstrap 95% CI **[+0.65, +3.90]** |
| vs its own phase-4 start | **+40.55 pp**, CI **[+37.35, +43.50]** |
| loss | *"soft cross entropy + 0.1 × sum-candidate Brier"* |
| calibration | **one positive scalar temperature per primitive, fitted only on separate calibration cases** |
| selection | step 600 by **dev soft CE**; `selection_completed_before_test: **true**` (`selection.json`) |
| training | 600 steps, **841 s ≈ 14 min** — inside the brief's ≤30 min budget |
| seed | **20260921** (single run); dataset revision `ea930645…` and Laya revision `f9ab0b22…` pinned |

**The split is documented and machine-readable** (`split_manifest.json`, sha256-pinned to the source
parquet):

| split | cases | questions | per workflow | role |
|---|---|---|---|---|
| train | 960 | 4,800 | 240 | fitting |
| dev | 120 | 600 | 30 | checkpoint selection |
| calibration | 120 | 600 | 30 | temperature fitting |
| **test** | **400** | **2,000** | 100 | `test_usage: "final evaluation only"` |

**5 questions per case** in every split. Splits are by **case id** in disjoint namespaces
(`tr_<workflow>_NNNNNN` vs `<workflow>_NNNNNN`), so no case straddles a split. The README states the
discipline plainly: *"The test split was not used to pick the checkpoint or the temperatures."*

**This is good methodology, and it is the single most important fact for the gap analysis below.**

---

## 2. Corrections — five claims of mine that the source refuted

### 2.1 RETRACTED: "calibration is measured but not acted on"

I claimed the setup *"already measures the divergence and has no mechanism to act on or even report it."*
**False on both counts.** Brier is a term **in the training loss** (`+ 0.1 × sum-candidate Brier`);
temperature is **fitted per primitive on a held-out calibration split**; and Brier/ECE are **published per
model and per primitive**. Calibration is optimised, fitted, and reported — not ignored. **The claim is
withdrawn entirely.** It is exactly the class of plausible-from-the-brief assertion that the source
exists to kill.

### 2.2 DOWNGRADED: "the sole metric discards the type information"

The per-primitive breakdown **is** published: Boolean 600 q → **88.83%**, Choice 600 q → **75.33%**,
Score 800 q → **75.00%**. So *"measures and ignores"* is wrong. What survives is narrower and sharper —
see §3.2.

### 2.3 NARROWED: "the optimisation toolkit is unvalidated for non-generative models"

Overstated. The repo contains a comparative evaluation against Laya, ModernBERT-base (149M), MiniLM-L6
(22M) and TypeSafe Jev 1.13.0 zero-shot, with a **sharp and honest** specialist-vs-generalist distinction
(*"Specialist and generalist are not the same measurement"*), plus a published latency study
(shared-prefix KV reuse: **33,547 → 2,551** path tokens, 92.4% reduction; 609.65 ms → 298.91 ms; max
probability difference **5.08e-4**, argmax unchanged). What is genuinely unvalidated is **agent-driven
training-code search on this object** — not the object itself.

### 2.4 CONFIRMED: the mandated loop does not exist

`program.md`, `results.tsv`, and `.github/workflows/` are **absent from the repository** — searched for
and not found. The brief specifies a system that **has not been built**. So the headline 79.25% is a
**supervised fine-tuning result on the benchmark's own training split**, not an autoresearch outcome.

### 2.5 RETRACTED: the per-item chance range (my §3.2 overstated it by 100×)

I wrote that `Choice` "accepts **2–255 options**, so its per-item chance floor spans **0.50 down to
0.004** — a **125× range**". **That is the API's *acceptance* range, not what the benchmark contains.**
Measuring the 2,000 published predictions gives the realised option counts:

| primitive | realised `k` | per-item chance | spread |
|---|---|---|---|
| Boolean | `[2]` | 0.5000 | — |
| Choice | **`[4, 5]`** | 0.2000 – 0.2500 | **1.25×** |
| Score | **`[4, 5]`** | 0.2000 – 0.2500 | **1.25×** |

The realised spread is **1.25×, not 125×** — I was out by a factor of 100. The claim is **withdrawn**, and
§3.2 is rewritten around the measurement. The underlying point survives in a *sharper* and now quantified
form (§3.2): the raw pooled Top-1 overstates chance-corrected skill by **8.89 points**, and the driver is
the **Boolean 0.50 floor**, not variation within Choice. Measurement replaced assertion, and the assertion
had picked the wrong mechanism.

### 2.6 CORRECTED: the loop's negative-result loss is already solved in the harness

I wrote (§3.3) that the loop "destroys its own negative results" because rejection is a `git reset`.
That describes **the brief's sketch** (`if accuracy improved → commit`, no record). It is wrong about the
harness. `jsegov/autoresearch-win-rtx`'s `program.md` carries a full protocol section —
*"Rejection: the attempt is persisted before the reset"* — specifying, for every rejected attempt:

- a `runs/<run_id>/trajectory/<NNNN>.diff` (the full diff against the base commit) **and**
  `<NNNN>.json` (attempt index, UTC timestamp, reason code, stated reason, diff base, discarded HEAD,
  diff bytes, diff sha256);
- reason codes `WORSE | TIED | CRASH | TIMEOUT | BREAKS_CONSTRAINT | OTHER`, with a required one-sentence
  `REASON`;
- enforcement of the ordering **record → reset**, under `set -e`, so the reset is unreachable if the
  record could not be written ("A rejected attempt without a record is an invalid experiment");
- a monotonic attempt counter that reads the directory back, so a restart cannot reuse an index.

That section was **added by the local commit `a504be2`** in that repo, as the fix for *FLAW 8* of
`archive/superseded_2026-09-27/critical_review_2026-09-26.md` in the sibling project — i.e. it is the
programme's own prior work, not upstream stock (verified: `program.md` differs from the pre-commit
revision by 327 lines, then committed alongside a `program.md.stock_backup` copy).

**Consequence for this document.** G3 is not an open gap in the harness; it is a gap in
**AgentJev's** repository, which ships `program.md`, `results.tsv` and `.github/workflows/` **not at all**.
The task is therefore *transplant* the protocol, not *design* it. §3.3 and §7 are re-scoped accordingly,
and the pre-registration's §7 is now a statement of an existing, tested convention rather than a
proposal. Note the mechanism is **written but unexercised** here: `runs/` contains one probe record and
**zero** `trajectory/` directories, so the first rejected attempt is still a live test of it.

---

## 3. The scientific gap

### 3.1 G1 (primary) — the specification omits the measurement discipline the project already implements

This is the gap, and the source makes it precise and near-paradoxical.

The completed v1 run **implements the correct discipline**: a dev split for checkpoint selection, a
separate calibration split for temperatures, `selection_completed_before_test: true`, the test split held
at 400 cases with `test_usage: "final evaluation only"`, and the dataset revision pinned by sha256. The
measured training time (841 s) sits inside the brief's own ≤30 min budget.

The **proposed 24/7 loop then discards every one of those protections.** `evaluate.py` is frozen and the
test set is fixed, while a 6-hourly cron queries that same instrument without bound, and the accept rule
is *"if accuracy improved → commit"* — **no threshold, no noise model, no dev/test separation for the
decision.** The rule is a selection operator on a noisy statistic. Two arithmetic consequences follow.

**(a) The accept rule is noise-blind — measured, not assumed.**

I previously estimated the floor as a binomial SD of **0.907 pp**. That was wrong, and the correction is
now measured from the subject repository's own 2,000 published predictions
(`measurement/measure_noise_floor.py`; a case-level bootstrap over the 400 test cases, 20,000 replicates,
all three headline accuracies reproduced exactly — 1585/2000, 774/2000, 1540/2000):

| quantity | value |
|---|---|
| case-level bootstrap SE (one arm) | **1.0668 pp** |
| **SE of the paired difference** (two arms, same test set) | **0.8431 pp** |
| binomial (independence) SE | 0.9068 pp |
| **design effect** | **1.384** |
| one-arm margin reaching p < 0.05 (1.96 × 1.0668) | 2.091 pp — **NOT the accept rule's threshold** |

Clustering is real but modest: the case-level SE is **1.38×** the binomial, not the 2.2× my ICC=0.3
illustration guessed.

**But the accept rule is a difference test, not a test against a known number.** It compares two trained
runs, and the correct scale is `SD(Δ) = sqrt(SE_paired² + 2·Floor_B²)`. At the most favourable case
(Floor B = 0, training perfectly reproducible) that is **0.8431 pp**, so the brief's own illustrative
`results.tsv` accepts:

| step | Δ | in SD(Δ) units | p (one-sided) | detectable? |
|---|---|---|---|---|
| + LoRA rank 32 | **+0.85 pp** | **1.01 σ** | 0.157 | **no** |
| + data augmentation | **+0.35 pp** | **0.42 σ** | 0.339 | **no** |

Both are indistinguishable from noise — and under `git reset`-on-rejection they become **permanent**,
since the next attempt must beat the *incremented* baseline.

**The corrected threshold is `τ = 1.96·sqrt(SE_paired² + 2·Floor_B²)`** — with Floor B unmeasured, a
*curve* rather than a number (`measurement/measure_floor_b.py`):

| Floor B (pp) | τ (pp) | published +2.25 pp Laya margin |
|---|---|---|
| 0.00 | 1.652 | claimable |
| 0.50 | 2.157 | claimable |
| **0.75** | **2.656** | **NOT claimable** |
| 2.00 | 5.785 | NOT claimable |

**Breaking point: the published +2.250 pp margin stops being distinguishable from training noise as soon
as Floor B exceeds 0.551 pp.** And even at the corrected τ, a 100-attempt loop carries a **92%
family-wise** false-accept probability — so τ alone never makes a 24/7 loop evidential.

**(b) The curve must rise under a true null** — quantified at the measured floor. Monte Carlo over
Gaussian draws at the measured SE, clipped to [0,1], 4,000 trials per k:

| attempts k | expected max gain | P(≥ one +2 pp jump) |
|---|---|---|
| 10 | 1.644 pp | 26.8% |
| **100** | **2.684 pp** | **95.8%** |
| 500 | 3.237 pp | 100.0% |
| **1460** (one per 6 h for a year) | **3.577 pp** | 100.0% |

**Pure noise alone manufactures 2.68 pp of apparent gain within 100 attempts, with 95.8% probability of
at least one +2 pp "improvement"** — and **3.58 pp over a year**. Note what that compares against: the
project's own published margin over Laya is **+2.25 pp**. **A pure null would generate a larger apparent
gain than the project's headline competitive result.** The hill-climb curve, which the brief names as the
headline deliverable (爬山曲线是展示 autoresearch 效果最直观的方式), is therefore **not evidence of
anything**, and the trajectory is an optimistically biased estimate of generalisation **by
construction**.

**The magnitude check that makes it acute.** The loop does not start from scratch: phase-4 scored 38.70%
and v1 reached 79.25% by *supervised fine-tuning on the training split itself*. The loop therefore starts
from an **already benchmark-adapted checkpoint**, so the gains realistically available to it are small —
while the instrument is blind below ~2 pp. **The loop's plausible effect size and its noise floor
overlap.** That is the sharpest form of the gap, and it is visible only from the source. Compounding it:
the loss already carries a tuned Brier coefficient (0.1) and three per-primitive temperatures, so several
of the obvious knobs are already set.

> **Reliability caveat — now partly resolved, with a second floor opened.** The 0.907 pp figure I first
> assumed is replaced above by the measured **1.0668 pp**. What the measurement *cannot* supply is the
> second noise source: **training-run reproducibility (Floor B)**. The subject repository publishes a
> **single** run (`seed 20260921`); `phase4` vs `agentjev_v1` differ by 600 training steps, not by seed,
> so their difference is an *effect*, not a replicate spread. **The accept rule compares consecutive runs
> that differ by both the code change and training randomness — and the size of that second source is
> unknown.** Floor A is therefore only a **lower bound** on the loop's noise, and it is the *favourable*
> bound. See §7.1a.

### 3.2 G2 (retained, sharpened by measurement) — the raw pooled Top-1 overstates skill by 8.89 points

The typed structure *is* reported, but the **pooled 79.25% is what the loop optimises and what the Laya
comparison is stated on**. Measuring per-item chance floors from the 2,000 predictions gives the first
chance-corrected figures for this benchmark:

| primitive | n | raw Top-1 | realised `k` | chance floor | **chance-corrected skill** |
|---|---|---|---|---|---|
| Boolean | 600 | 0.8883 | `[2]` | **0.5000** | **0.7767** |
| Choice | 600 | 0.7533 | `[4, 5]` | 0.2333 | **0.6783** |
| Score | 800 | 0.7500 | `[4, 5]` | 0.2437 | **0.6694** |
| **pooled (per-item)** | 2,000 | **0.7925** | — | — | **0.7036** |

Two measured consequences:

- **The raw headline overstates skill by 8.89 points.** Every one of the 600 Boolean items has a **0.50**
  chance floor, so half of a Boolean item's credit is free. The pooled Top-1 therefore **rewards the item
  mix** — a 600/600/800 design — rather than capability. The honest headline for this checkpoint is
  **0.7036**, and the gap between the two numbers is an artefact of *which primitives are in the test
  set*, not of model quality.
- **The pooling is mix-sensitive in a second way.** Raw Top-1 spans 0.8883–0.7500 across primitives; on
  chance-corrected skill the spread narrows to 0.7767–0.6694, and the primitive *ranking* is preserved
  (Boolean > Choice > Score). So the aggregate is not merely a mixture — it is a mixture with a
  **systematically weighted** contribution from the easiest primitive, which has the *highest* floor.
- **My earlier "125× choice range" claim is withdrawn** (§2.5): the realised spread is **1.25×**. The
  per-item correction is still worth making — but its justification is the Boolean floor and the mix, not
  variation inside Choice.

The cleanest available measurement improvement is therefore a **per-item chance-corrected skill score**
reported **alongside** the pooled number — the substrate supports it because `k` is known per item, and
the figures above are computable from the published predictions today.

### 3.3 G3 (re-scoped) — the negative-result loss is a property of the BRIEF and of AgentJev's repo, not of the harness

`git reset` on rejection discards the rejected experiment, so a surviving history is a **running
maximum, not a sample path**: later attempts cannot see where the loop already searched and failed, and
the loop's true effect is unestimable because the denominators are gone. This is exactly the evidence G1
needs, so a design that reset without recording would make the primary claim **untestable**.

**But the harness already solves it** (§2.6): `jsegov/autoresearch-win-rtx`'s `program.md` specifies
record-before-reset with a per-attempt `runs/<run_id>/trajectory/` diff+json pair, reason codes, and a
monotonic attempt index, enforced under `set -e`. So the gap is located precisely:

| artefact | `autoresearch-win-rtx` | AgentJev repo |
|---|---|---|
| `program.md` (loop constitution) | **present** (213 lines) | **absent** |
| `results.tsv` (ledger with `keep`/`discard`/`crash`) | **present** (1 baseline row) | **absent** |
| `.github/workflows/` (cron) | **absent** (only `FUNDING.yml`) | **absent** |
| rejection-trajectory protocol | **specified but unexercised** (0 `trajectory/` dirs) | **absent** |

The work is therefore **transplant + first exercise**, not design. AgentJev ships only
`typed_decisions/verify_eval.py` and `evaluate_laya.py` as evaluation guards — the *discipline* exists
there; the *loop* does not.

### 3.4 G4 (narrowed) — agent-driven training-code search is unvalidated here

Fit-on-benchmark specialists are well covered (§2.3). Untested is whether *search over training code*
works on a **non-generative, permutation-equivariant, typed-primitive** object, and the brief's own
constraint (*prefer data augmentation and loss adjustment before touching architecture*) encodes an
unchecked prior.

### 3.5 G5 (NEW, from the source) — the Laya comparison is confounded by a 2× context ceiling

The README's own latency table records it: **AgentJev context 2,048; Laya context 1,024** — *"Laya
truncates or refuses beyond 1,024; AgentJev retains twice the state."* The **+2.25 pp** accuracy gap is
therefore **not a like-for-like modelling comparison**: on any case whose serialized state exceeds 1,024
tokens, Laya is information-starved and AgentJev is not. The confound is *disclosed* in the throughput
section but **not controlled in the accuracy comparison**. A like-for-like claim needs either a
length-matched (≤1,024-token) subset re-scored, or an explicit AgentJev arm truncated to 1,024. This is
the most reviewable weakness in the headline result, and it is cheap to fix.

### 3.6 G6 (NEW, from the source) — the benchmark measures agreement with a teacher, not task success

The protocol's own first limitation: *"Targets are public teacher distributions, not empirical outcome
probabilities."* The README is equally direct: *"Accuracy is agreement with the public teacher argmax. It
is not a measured coding-agent success rate."* So the optimised quantity is **distillation fidelity**,
whose ceiling is the **teacher's own accuracy**, not 100% — and an unbounded loop climbs toward that
teacher, not toward correctness. Every accuracy number here must carry that interpretation, and the
loop's optimand should be stated as *agreement with the teacher* rather than *decision quality*.

---

## 4. The invariant, and why this object is the better place to prove it

> **Previous object:** a frozen benchmark + one static template cannot distinguish *searching* from *sampling*.
> **This object:** a frozen evaluator + an unbounded selection loop cannot distinguish *searching* from *fitting*.

The same structural defect — **an instrument queried adaptively stops being a measurement and becomes a
target** — on a different substrate. The object moved from prompt space to training-code space; the gap
did not. And this object is the **better** place to prove it, for a reason the source made concrete: the
correct discipline is **already implemented in v1 and already machine-recorded** (`selection.json`,
`split_manifest.json`, `test_usage: "final evaluation only"`). The comparison is therefore not
hypothetical — it is **the project's own documented standard versus its own proposed specification**.

---

## 5. Scientific significance

1. **A general result about autoresearch loops.** The same structure governs every *frozen evaluator +
   adaptive query + keep-the-best* loop — FunSearch, AlphaEvolve, the Karpathy-style single-file loop
   this descends from, and any pipeline whose validation set also serves as its acceptance signal.
2. **It converts a one-line convention into a falsifiable prediction.** *"Commit if accuracy improved"*
   becomes a selection procedure with a derived null (expected max gain = SD·√(2 ln k)) and a sharp
   quantitative prediction: **the curve must rise ~2.8 pp within 100 attempts under a null.**
3. **It imports mature theory where practice has outrun it** — selection bias, the garden of forking
   paths, and *adaptive data analysis* (reusable holdout; a privacy-like budget on queries). Agent-loop
   work adopted the practice without the theory.
4. **It is constructive and cheap.** The fix is protocol, not modelling: separate the **decision** signal
   (dev) from an **audit** signal (a shadow split never used for a decision); **log every attempt**
   including rejections; pre-register an accept margin **at or above the bootstrap-derived minimum
   detectable effect**; report **per-item chance-corrected, per-primitive** skill; and keep the
   calibration reporting that already exists.
5. **The typed substrate is a genuine measurement upgrade.** Because every item's option count is known at
   request time, a **per-item chance-corrected score** is computable — including a fix for Choice's
   0.50 → 0.004 floor. Text benchmarks can only offer a uniform nominal chance level. This is a positive
   opportunity created by the object, not a criticism of it.

---

## 6. What must be built, and what must be retired

**Must be built** — all three absent: `program.md` (the constitution the brief specifies), the
`.github/workflows/` cron driver, and `results.tsv` — plus, per G1, the **shadow split** and
**all-attempts logging** the brief does not mention.

| item | disposition |
|---|---|
| prompt-multiplier / HPFE / OASP / κ | **retired as an object.** AgentJev is not prompted; the names cannot be reused without equivocation. |
| Lean artifact (29 theorems, `E:/2026-AI4S/lean-prompt-multiplier/`) | **carries over subject to re-derivation.** The sampling/gap theorems concern any monotonically scored finite candidate set and already describe this loop; the moderation core does not transfer and must be re-derived, not relabelled. |
| `docs/EVIDENCE_prior_art_and_feasibility.md` | **carries over** — contamination tests T1–T9 and the certified-gap headroom definition apply directly. |
| `docs/research_program.md`, `RESEARCH_PROTOCOL.md`, `oasp/`, pilot runs | **previous object**; **moved out of the live tree** by the de-interference sweep (`archive/retired_object_prompt_multiplier_2026-09-27/`, manifest `archive/MANIFEST_deinterference_2026-09-27.json`). Retained there as methodological precedent and still runnable from the archive. The 3×-case and power reasoning is the same reasoning G1 needs, and is restated inline in §3.1. |
| the v1 repository itself | **the baseline to beat and the standard to match.** Not a thing to rebuild. |

---

## 7. Open items

**1. ~~Bootstrap the noise floor~~ — DONE (measured).** `measurement/measure_noise_floor.py` reads the
subject repository's own 2,000 published predictions and reports: case-level bootstrap SE **1.0668 pp**,
design effect **1.384**, **paired**-difference SE **0.8431 pp**, and the corrected accept threshold
`τ = 1.96·sqrt(SE_paired² + 2·Floor_B²)` — a curve in the unmeasured Floor B with a breaking point at
**0.551 pp**.
All three headline accuracies reproduce exactly, so the artifacts are the right ones. Machine-readable
output: `measurement/noise_floor.json`.

**1a. Floor B — training-run reproducibility — is now the blocking item.** The floor above is the
*test-set* floor (Floor A). The accept rule's actual noise also includes **run-to-run training
randomness**, and that is **not measurable from anything published**: the subject repository ships a
single run (`seed 20260921`). **Blocking: the accept margin cannot be set until the baseline's
seed-to-seed SD is known** — and Floor A is the *favourable* bound, so the true margin is larger.
Requires seed replicates of a fixed configuration. *This is the same defect as the earlier object's
`train.py` having no `--seed` flag: a protocol that cannot express replication cannot measure its own
noise.*

**2. Control the context confound (G5)** — re-score Laya, or AgentJev truncated to 1,024, on the
length-matched subset. *Blocking: the +2.25 pp headline claim.* *(Partially de-risked: the reported CI
now reproduces independently as **[+0.600, +3.900]**, against the published [+0.650, +3.900] — a Monte
Carlo difference, not a discrepancy. But reproducing an interval does not remove the confound.)*

**3. ~~Publish per-item chance-corrected skill~~ — DONE (measured).** Realised option counts are
`[2]` / `[4, 5]` / `[4, 5]`; chance-corrected skill is **0.7767 / 0.6783 / 0.6694**, pooled **0.7036**
against a raw **0.7925**. These are the first chance-corrected figures for this benchmark. My earlier
"Choice 2–255, a 125× range" claim is **withdrawn** (see §2.5) — the realised spread is 1.25×.

**4. State the optimand explicitly** — pooled Top-1 or per-item chance-corrected skill, and in both cases
as *teacher agreement* (G6). *Blocking: pre-registration.* With §3.2 measured, the recommendation is now
concrete: report **both**, and treat the chance-corrected figure as the honest headline.
5. **Runner security** — a self-hosted Actions runner executing agent-written Python with
   `AGENT_API_KEY` in the environment. Not a scientific gap; a project-stopping risk.
6. **The retired object has been removed from the live tree** — its harness, drivers, tests, configs,
   pilot outputs, and the two documents that read as authoritative now live under `archive/` with
   `archive/MANIFEST_deinterference_2026-09-27.json` (140 files, 35.9 MB, reversible). Entry points
   (`CURRENT_OBJECT.md`, `README.md`) describe only the current object, and
   `scripts/check_object_purity.py` asserts that mechanically. Any remaining reference to the retired
   object in this document sits inside a retirement notice by design.
7. **`DOCUMENT_MAP.md`'s claim about `Problem.docx` was false** — it asserts the working copy is
   byte-identical to `Problem-4.docx` (`8be65ee4…`, 24,686 B). Measured: the working copy is
   `dd381c41a245`, **24,711 B** — a *near*-copy, not a byte-identical one. Corrected in that file; noted
   here because the same class of assertion ("X is identical to Y") should be verified, not assumed.

---

## 8. What this document is not

- It is **not** a claim that the v1 result is wrong. On the evidence in the repository the v1 run is
  **well executed**: pre-registered selection, a held-out calibration split, a case-level bootstrap,
  pinned revisions, honest specialist/generalist labelling, and a self-declared ceiling (*"agreement with
  a teacher"*). The critique targets the **proposed loop**, not the completed run — and §3.1 is precisely
  the observation that the two are in tension.
- The noise figures are **arithmetic from the brief's parameters**, not measurement; §7.1 replaces them
  with the repository's own instrument, and clustering can only make them worse.
- G5 and G6 are **newly found in the source** and are stated as confounds to control, not as authorial
  errors — the context difference is disclosed in the README's own latency table, and the
  teacher-agreement caveat is disclosed in both the README and `protocol.json`.
- §2's retractions are mine, not the project's. Three claims I made from the brief alone did not survive
  contact with the source; that is a fact about the method, and it is recorded rather than quietly
  dropped.
