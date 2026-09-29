# Evidence annex — prior art, contamination, headroom, feasibility

**Source:** `docs/evidence/prior_art_and_feasibility.json` (machine-readable, 44 KB; promoted from a
root-level stray `final.json` on 2026-09-27 — it was **research output**, not a distractor).
Five keys: `precedents` (20 papers), `headroom_definition`, `contamination`, `budget_table`, `feasibility`.

---

## 0. Scope — which object this annex serves

This annex serves the **current object, JevRSI (JevRSI)** — a 24/7 autoresearch retraining loop over
the *training code* of `AgentJev-0.6B`, a non-generative typed-decision model. See `CURRENT_OBJECT.md`.

It was **compiled for a previous object** (prompt-multiplier / **HPFE** / **OASP** / κ), which is
**retired** and archived at `archive/retired_object_prompt_multiplier_2026-09-27/`. The material below
has been **re-scoped rather than rewritten**, because most of it is object-independent. Where a section
still carries the old framing, it is marked as such inline.

| section | transferability to JevRSI |
|---|---|
| §1 prior art | **direct** — the two closest precedents bear on JevRSI's G1 and G4; §1's framing was the old object's and has been re-stated |
| §2 headroom / certified gap | **translatable** — the certified-gap idea becomes the *noise floor* (minimum detectable effect); §2.1 gives the translation |
| §3 contamination T1–T9 | **direct** — applies unchanged to any evaluator whose test set also drives decisions |
| §4 host throughput | **partial** — measured on this host and reproducible, but the *candidate-budget arithmetic* was computed for a generation-based proposer and must be redone for JevRSI's training-loop economics (§4.1) |

---

## 1. The precedents that bear on this object

All **2026** entries; both closest precedents cut **toward** the proposition rather than pre-empting it.

| # | paper | finding | why it matters for JevRSI |
|---|---|---|---|
| **20** | Erdos minimum-overlap problem | `gpt-oss-120b`, **prompt-only, no search** (Best-of-25600) — **the prompt-only baseline ALSO improved the record** | The closest existing evidence that a **baseline which is not an optimisation loop can match what "optimisation" claims**. For JevRSI this is the precedent for **G1**: a non-searching procedure reproducing a search result is exactly what a noise-blind accept rule will manufacture. |
| **19** | methodological study: 30 budget-matched discovery harnesses × 12 model–problem cells | meta-result: **no fixed harness is reliably superior** | Undercuts "strategy choice is what matters". For JevRSI this bears on **G4**: if no harness is reliably superior, then "which training-code changes work" is not answerable from a single unbounded trajectory. |

**Action:** cite both in related work as the closest prior art. Neither invalidates the proposition.

**On #13** — `Zarankiewicz numbers Z(m,n,3,3)`, OpenEvolve + Claude, 2026: *first exact determination of
Z(11,21,3,3)=116*. Recorded here as **literature only**. Selecting an extremal-combinatorics surface as
the research topic was a **wrong turn**, archived at `archive/wrong_turns/`; the topic is not the object.
The precedent remains useful as evidence of the field's pace, not as a target.

### 1.1 The other 17 precedents (summary)

FunSearch (#1, cap set 512 in F₃⁸), AlphaEvolve (#2 4×4 complex matrix rank 48; #3 67 problems;
#11 inapproximability gadgets MAX-4-CUT 0.987), TTT-Discover (#4), ThetaEvolve (#5), ShinkaEvolve (#6),
OpenEvolve-family (#7, #13, #15), heuristic-design work (#8 EoH, #9, #10), Ramsey lower bounds (#12,
nine improved), adversarial instances (#14), bijection discovery (#15 — a **negative result**: no run
yielded a new bijection), non-LLM transformer search (#16), reproducibility (#17), ~200-task
meta-optimization (#18).

**#15 deserves note as the honest counterweight:** a headline negative result. Cite it alongside the
successes so the literature review is not a wins-only selection.

---

## 2. Headroom — the certified gap, and its translation

**Original form (previous object).** The brief's `easy-to-evaluate / hard-to-solve` principle had no
operationalization; this supplied one, computable **without running the LLM loop**, which is what makes
it pre-registerable rather than post-hoc.

Fix a family `F = (G, s, check, M)`: `G(seed)` a deterministic instance generator, `s` a scoring
function, `check` a feasibility/purity verifier, `M` a monotone tuple of solution-quality statistics.
For an instance `I ~ G(seed)`:

- `VFeas(I)` — best feasible objective value published in the literature, or produced by any
  deterministic procedure in the baseline ensemble;
- `Bnd(I)` — a **certified** bound on the optimum, with a verifiable certificate.

Then the instance-level **certified gap** is

```
Delta(I) = Bnd(I) - VFeas(I)  >= 0          rho(I) = Delta(I) / max(1, |Bnd(I)|)
```

`Delta(I)` is the **informational headroom** — room that *provably* exists for a better construction.

| level | claim | falsified by |
|---|---|---|
| **H1 informational** (deterministic) | `Delta(I) > 0` for the pre-registered instances | `Delta(I) = 0`, or an integer objective where `Delta(I) < 1` scoring unit |
| **H2 algorithmic** (the hypothesis under test) | ∃ program `p` reachable by the loop `L` with `s(evaluate(p,I)) >= VFeas(I) + delta` on ≥ k of N fresh instances | failing the pre-registered `k/N` |
| **H3 detectable** (the statistical claim) | the H2 gain is separable from run-to-run randomness and from contamination | failure of separation |

### 2.1 Translation to JevRSI — the certified gap becomes the noise floor

The H1/H2/H3 structure **transfers**, with one substitution that is the heart of the current programme:
JevRSI has no certified bound `Bnd(I)`, because its objective is *agreement with a teacher*, not an
extremal optimum. What plays the role of "room that provably exists" is instead **the smallest
difference the instrument can resolve**:

| certified-gap concept | JevRSI equivalent |
|---|---|
| `Bnd(I)`, a certified bound | *not available* — the ceiling is the teacher's own accuracy, not 100% |
| `Delta(I) = Bnd(I) - VFeas(I)`, informational headroom | **the minimum detectable effect** at the given sample size — **measured**: SE of the **paired** difference **0.8431 pp**, so the two-run accept threshold is `τ = 1.96·sqrt(0.8431² + 2·Floor_B²)`, a *curve* in the unmeasured Floor B (breaking point vs the published +2.25 pp margin: **Floor B = 0.551 pp**). The single-arm case-level SE is 1.0668 pp (design effect 1.384) and its 1.96× value 2.091 pp is **not** the rule's threshold. |
| `Delta(I) = 0` disqualifies an instance *by construction* | **an improvement below the noise floor is disqualified by construction** — it cannot be evidence |
| H1 / H2 / H3 | *unchanged in form*: (H1) the headroom exists; (H2) the loop finds it; (H3) the gain is separable from randomness and contamination |

**Consequence — the single most important design rule for JevRSI.** Before any loop is run, the
pre-registration must state the **accept threshold**. It is now **measured, not assumed**, and it is not
a single number. `measurement/measure_noise_floor.py` measures the SE of the **paired** difference between
two arms on the same frozen test set (**0.8431 pp**) and `measurement/measure_floor_b.py` combines it with
the unknown training floor:

```
τ = 1.96 · sqrt( SE_paired² + 2 · Floor_B² )
```

which at Floor B = 0 is **1.652 pp**, and rises past the published +2.25 pp margin as soon as **Floor B
exceeds 0.551 pp**. Any accept margin below τ is void, because the accept rule would then be selecting
noise. This is the operational form of `Delta(I) = 0 disqualifies the instance`.

Note the correction: an earlier draft used `τ = 1.96 × Floor A` = **2.091 pp**, where Floor A (1.0668 pp)
is the SE of a *single* arm. That is the threshold for testing one measurement against a **known** number;
the accept rule compares **two noisy runs**, so it is a difference test and the paired SE applies.

**And a second, unmeasured floor.** The paired SE above is still the *test-set* term. The accept rule's
real noise also includes **run-to-run training randomness**, which nothing published can supply (the
subject repository ships one seed). Hence τ is a *curve* in Floor B, and every value quoted at Floor B = 0
is the favourable end — see `docs/JevRSI_program.md` §7.1/§7.1a.

---

## 3. Contamination — nine cheap tests

The briefs require benchmarks released in 2026 so data cannot have leaked into pretraining. This
supplies the detection machinery and separates two mechanisms that must never be conflated:

- **(a) LLM proposes a program, search improves it** — contamination lives in (i) the *program space*
  (the proposer may emit a known strong algorithm from memory) and (ii) the *instance-level record*;
- **(b) LLM directly outputs a solution from memory** — contamination shows up as reproduction of a
  published value with **no search**.

For **JevRSI the relevant instance of (b)** is direct: the model's targets are *public teacher
distributions*, and the benchmark is a public HF dataset pinned at revision
`ea9306458d6e9563628369a3d1e72e362fb381d2`. The proposer agent may also have memorised published
results for `AgentJev-0.6B` itself.

| test | what it does |
|---|---|
| **T1** | **isomorphism / relabeling invariance** (strongest single test) — run the identical loop on a structure-preserving relabeled twin; retrieval shows as a large quality drop on the twin, genuine search is approximately invariant. Report twin/original ratio with a paired CI over ≥16 pairs. |
| **T2** | **post-cutoff instance test** (design-level, cost 0) — make instance parameters post-date the proposer's cutoff and verify best-known values are unpublished. **The only test that removes the risk rather than detecting it.** |
| **T3** | stale-solution test |
| **T4** | name-stripping |
| **T5** | likelihood test on the published construction |
| **T6** | verbal recall probe |
| **T7** | canonical exact-match on search output |
| **T8** | **in-context-SOTA ablation** — grants the baseline the same memory the proposer has |
| **T9** | fresh-family replication |

**Design rule.** Choose a setting where **freshness is a knob**, pre-register the split (famous instances
= **controls**; fresh post-cutoff instances = **endpoints**), and state in advance the exact statistics
reported on each. Treat a memorised baseline as a **measured quantity, not an assumption**.

**Memorisation-prone targets to avoid** (one famous answer = the whole target): specific Ramsey numbers
with published constructions, the 4×4 matrix-multiplication rank (48), kissing number in dimension 11
(593), FunSearch's cap set of 512, the 26-circles packing, QAPLIB best-known permutations, TSPLIB optimal
tours, Taillard job-shop optima. **A famous instance that is also solved has zero headroom and is doubly
disqualified** — its honest use is as a **recovery control**, never as the discovery target.

> **JevRSI-specific caveat.** Test **T2 cannot be used here**: the benchmark is fixed and public by design,
> and the test split is the decision surface. That is precisely why the decision/audit split separation
> in the programme document (`docs/JevRSI_program.md` §3.1, §5.4) is not optional — it is the only
> remaining instrument once T2 is unavailable.

---

## 4. Feasibility — measured on this host

**All numbers measured on the actual target host**, not estimated: Windows 11, RTX 4070 12 GB
(12,282 MiB; ~10.4 GiB usable, 876 MiB idle desktop), i9-13900K, Ollama 0.34.4, generation-only figures
from the API's own `eval_count`/`eval_duration`, `num_ctx = 4096`:

| model | quant / residency | tok/s |
|---|---|---|
| `mistral:7b` | Q4, ~4.95 GB, 100% GPU | **98.3** |
| `gemma3:12b` | Q4, ~8.04 GB, 100% GPU | **56.2** |
| `deepseek-r1-qwen-14b` | Q4_K_M, ~9.47 GB, 100% GPU | **50.3** (same at `num_ctx=8192`, 10.28 GB resident) |
| `deepseek-r1-qwen-7b` | Q4_K_M, ~4.7 GB | 41.7 (model-specific; `mistral:7b` is the better 7B datapoint) |
| `qwen2.5:32b` | Q4_K_M, 21.2 GB total | **4.1** — only ~10.5 GB resident, ~50% of layers on CPU → **NOT viable for a 24/7 loop** |

**⚠ Configuration trap (silent, and it will ruin a long run):** with Ollama's *default* context — these
GGUFs advertise up to 131,072 tokens — a 14B model claims 35.9 GB and offloads only 24 of 49 layers,
collapsing to **7.0 tok/s**. Forcing `num_ctx = 4096–8192` restores 100% GPU residency and 50 tok/s.
**Uncapped context silently destroys throughput.** Any 24/7 launcher must pin `num_ctx` explicitly
rather than trusting the model's advertised maximum.

### 4.1 Which parts of §4 transfer to JevRSI, and which do not

The table above measures **local LLM generation** — it was taken for a generation-based proposer. JevRSI's
loop is a **training** loop, so:

- **Still valid:** every figure above, and the `num_ctx` trap, apply to any local LLM the loop calls
  (the agent that writes training-code edits, or any local grader).
- **Must be recomputed:** the candidate-budget arithmetic. JevRSI's unit of work is one *training run*,
  not one generated candidate. The subject repository measures a run at **841 s** (600 steps) — so the
  loop's budget is roughly **4 runs/hour serial on this card**, against a cron period of 6 h. The old
  "6,000–21,000 evaluated candidates/day" figure is **not** the right denominator for JevRSI and must not
  be quoted for it.

Two model tiers remain practical for any local-LLM role: a fast **7–8B** (~98 tok/s) and a stronger
**12–14B** (~50–56 tok/s). A sparse/MoE 20–30B-total with 2–4B active was **not measured** and is flagged
**unverified** — the precedents that used such models (`gpt-oss-20b` in #4, `Qwen3-30B-A3B` in #10) were
not run on a 12 GB consumer card.

---

## 5. Status

| item | status |
|---|---|
| prior-art check on the novelty claim | **substantially closed** — 20 precedents tabulated; #19 and #20 are the closest prior art and must be cited |
| operationalizing "hard to solve" | **closed for the previous object**; **translated** for JevRSI in §2.1 — the certified gap becomes the accept threshold `τ = 1.96·sqrt(SE_paired² + 2·Floor_B²)`, computed from the subject repository's own predictions (paired SE **0.8431 pp**) plus seed replicates |
| leakage / contamination control | **closed** — §3 gives 9 tests and a design rule; note **T2 is unavailable for JevRSI** (§3 caveat), which is why decision/audit split separation is mandatory |
| 24/7 free-local feasibility | **partially closed** — §4 gives measured throughput and the model ceiling; §4.1 states which arithmetic must be recomputed for JevRSI's training-loop economics |
| noise floor / accept threshold | **closed for the measurement term, open for Floor B** — the paired-difference SE is **measured** at **0.8431 pp** (single-arm case-level SE 1.0668 pp, design effect 1.384); Floor B (seed-to-seed training variance) is **not measurable** from anything published, so τ is a curve with a breaking point at Floor B = 0.551 pp. See `docs/JevRSI_program.md` §7.1/§7.1a and `docs/JevRSI_preregistration.md` §3/§5 |

**Caveat on provenance.** This annex summarises a research dump whose citations were compiled by a
subagent and were **not** individually verified against each paper. Treat every arXiv id, year and
"beat best known" verdict in §1 as **needing independent confirmation before entering a manuscript** —
the FunSearch/cap-set and AlphaEvolve/matrix-rank claims are well known and low-risk, but the 2025–2026
entries are not, and the tabulated verdicts are one-line summaries, not quotes. The **§4 measurements**
are different in kind: they were taken on this host and are reproducible locally.
