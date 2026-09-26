# OASP — Implementation Plan (实施方案)

**Program:** Optimization-as-Search Program · **Object:** the landing form (落地形态) of reported
LLM optimization strategies · **Protocol:** [`RESEARCH_PROTOCOL.md`](../RESEARCH_PROTOCOL.md)

**Added constraint (Problem-2.docx ¶14):** 全程在本机 24/7 部署，并全程免费
— the entire pipeline must run locally, continuously, at zero marginal cost.

> **Scope note (added 2026-09-26; FLAW 2 of [`critical_review_2026-09-26.md`](critical_review_2026-09-26.md)).**
> The ARC/MCQ work planned and reported below — the surface-variant matrix, M1–M8, the negative
> pilot result — is the **pilot**, and it measures **construct A** (measurement-instrument method
> variance: reproducibility of a reported score under evaluator-authored surface variation). It is
> **not** the project's object. The project's object is **construct B** (the landing form of an
> optimization strategy: whether the researcher's hand-authored template determines the result),
> carried by `nanochat`'s `program.md`, which is the primary object of study. Construct B has not
> been tested yet. No content is deleted; claims below about "the project" are claims about the pilot.
>
> **DV note (added 2026-09-26; FLAW 6).** The construct-B dependent variable is written
> **`val_bpb@300s`** — validation bits per byte read at a **fixed 300-second wall-clock training
> budget**, hence compute-normalised. Cross-hardware comparisons and comparisons against
> fixed-step runs are **void**.

---

## 0. What the first pilot actually found (and why it forces a design change)

Pilot `pilot_v1_dev` — ARC-Easy, 60 items × 32 surface variants, gemma3:4b complete
(1,920 cells), temperature 0, seed 42. Full numbers: `analysis/pilot_v1_dev/report.md`.

| metric | value | reading |
|---|---|---|
| M1 range | **0.0833** (0.850 → 0.933) | 8.3-point spread across 32 equivalent prompts |
| M2 IPS | **0.1833** [0.0996, 0.2833] | 11 of 60 items flip correctness |
| M2 independent reference | 0.153 | observed only **+0.030** above the independent-outcome baseline |
| **M3 permutation (spread)** | obs 0.0833 vs null 0.0707, **p = 0.3073** | **spread is NOT distinguishable from within-item label shuffling** |
| M3 permutation (η²_prompt) | 0.0076 vs null 0.0041, p = 0.0005 | detectable but substantively **negligible** |
| capability floor | 0.9057, Wilson [0.8918, 0.918] vs chance 0.2502 | clears the floor comfortably |

**Verdict on H1 as operationalized: NOT SUPPORTED.** qwen3_4b is excluded outright —
0.2436 accuracy against a 0.2502 chance level, i.e. **at or below chance**; a model with no
signal cannot exhibit a prompt effect, and its 234-row grid is incomplete.

**This is the most valuable result so far, and it is the brief's own cell B**
(`docs/theory.md`): a static prompt with equivalent variants *is* a fixation, but on this
model+benchmark the fixation is **benign**. Two properties of the design, not of the theory,
made H1 unfalsifiable-in-the-right-direction here:

1. **Ceiling compression.** gemma3:4b scores 0.906 on ARC-Easy. With only 9.4 points of
   headroom, accuracy variance is bounded from above regardless of wording. A ceiling-bound
   model *cannot* demonstrate sensitivity, so failing to find it is uninformative.
2. **Only surface features were varied.** The 180-point space varies verbs, clause order,
   label style, format clauses — deliberately conservative (auditable equivalence), but it
   excludes the *content* of the instruction, which is where the literature's large
   prompt-induced swings originate.

**Rule adopted from this:** a target must be **interior** — accuracy strictly inside
`[0.35, 0.85]` with a Wilson CI excluding chance — or its H1 test is not reported at all.
Ceiling and floor models are reported in a separate "bounds" table and never pooled.

> **Guards against circularity.** Targets are selected on their **baseline accuracy band**
> — a property measured on a *fixed canonical prompt* — **never** on their prompt-variance.
> Selecting models for showing the effect would make H1 tautological. The band is a property of
> the model+benchmark pair, not of the outcome.
>
> **Provenance of the interior-band rule (corrected 2026-09-26; FLAW 3 of
> [`critical_review_2026-09-26.md`](critical_review_2026-09-26.md)).** The rule was **introduced
> following pilot 1**, when ceiling compression on ARC-Easy became visible; it is **defined on
> baseline accuracy alone**, and it was **applied unchanged to pilot 2**. It is a **post-pilot-1
> criterion, NOT pre-registered** — "pre-registered" must never be used to describe it.

---

## 1. Research goal, restated precisely

**Not** "benchmarks are biased" and **not** "prompts matter". The object is the degeneration

```
strategy S  ──►  one hand-authored static template p₀  ──►  one benchmark score Â = A(p₀, m)
```

claimed in the literature to represent the strategy. Two claims, two names, disjoint roots:

- **Effect — HPFE (Heuristic Prompt Fixation Effect).** Strategies fixate on a heuristic
  point instead of searching prompt space. *Measured by:* M1 spread, M2 IPS, M4/M5 rank
  stability.
- **Insight — OASP (Optimization-as-Search Principle).** Optimization should be modelled as a
  search loop, not a single-point score. *Measured by:* M7 held-out gain from search.

**The load-bearing hypothesis is H2, not H1.** "Optimization is search" is analytically true;
OASP is only worth asserting if searching measurably beats the sampled upper tail **on data
the search never saw**. H1 failing (as in this pilot) does not rescue OASP — it *narrows* it.

---

## 2. Revised design (v2) — changes from v1 and why

| change | from | to | why |
|---|---|---|---|
| benchmark | ARC-Easy only | ARC-Challenge (primary) + ARC-Easy (control) | ARC-Easy is ceiling-bound for 4B–7B models |
| target selection | all local models | **interior band [0.35, 0.85]**, Wilson CI excludes chance | ceiling/floor models cannot test H1 |
| prompt space | 180-pt surface factorial | surface factorial **+ semantic condition** | surface-only is conservative; the brief's own wording (语义等价但措辞不同) covers content |
| optimizer | OpenRouter (paid) | **local Qwen3.8-27B via Ollama** | new 24/7 + free constraint; also keeps decoupling (optimizer ≠ target) |
| M6 noise floor | not run | required before any H1 verdict | the decision rule compares spread to `2 × floor` |
| held-out items | 60 | **≥150** for H2 | paired-binary power (protocol §6) |

**Two prompt conditions, never pooled:**

- **Condition A (primary, auditable):** the 180-point surface-feature factorial. Equivalence
  is *constructed* — each variant is a point in an explicit space, so a reader can see exactly
  what varied. This is the conservative test.
- **Condition B (secondary, realistic):** semantically-equivalent paraphrases produced by the
  local optimizer model, plus a few-shot exemplar manipulation. This is the test with more
  power, and it is the one that can conflict with A — divergence is reported, never averaged.

---

## 3. Model plan — local, free, continuous

**Resource envelope.** RTX 4070 (12 GB), Ollama on `127.0.0.1:11434`. Measured: parallel
requests to one model gave only **1.22×** (Ollama already saturates), so the runner uses
`--concurrency 1` and holds one model resident at a time via `keep_alive`.

| role | model | why |
|---|---|---|
| **targets** | to be chosen by the ARC-Challenge screen, from `gemma3:4b`, `gemma3:12b`, `mistral:7b`, `qwen2.5:7b`, `qwen3:4b`, `deepseek-r1-qwen-7b` | must land in the interior band; mixed families and sizes |
| **optimizer** | `qwen3.8:27b` (Ollama, local) | free, capable, and **disjoint from every target** — enforced by `config.assert_decoupled()` |
| **excluded** | `deepseek-r1-qwen-7b` (parse rate 0.00–0.05), `qwen3:4b` on ARC-Easy (at chance) | format non-adherence / capability floor |

**Cost:** zero. No API key is required for any stage; `chat_openrouter` exists but is unused
unless `OPENROUTER_API_KEY` is set and a stage explicitly opts in.

**Throughput measured (RTX 4070, temp 0, `num_predict=192`):** gemma3:4b ≈ 4.7 cells/s;
mistral:7b ≈ 5.9 cells/s; qwen3:4b ≈ 0.7 cells/s (starved by its own reasoning tokens);
gemma3:12b ≈ 2 cells/s. Plan on ~2 cells/s sustained.

---

## 4. 24/7 operation

**Scheduling.** One long-running sequential worker driven by cron, not by interactive turns:
the unit of work is a *batch*, and every batch is resumable.

```
cron: nightly 01:00  ->  scripts/nightly_campaign.sh
                          ├─ preflight (Ollama up? models present? disk? VRAM?)
                          ├─ oasp.runner  --run-id <campaign>_<split>   (resumes)
                          ├─ run_noise_floor.py                          (once per campaign)
                          └─ oasp.analyze                                 (report + figures)
```

**Why this is safe to run unattended:**

| hazard | mitigation |
|---|---|
| crash / reboot mid-run | append-only JSONL + `load_done_cells()` resume — proven: the pilot was killed at 2,154 cells and resumed without loss |
| disk exhaustion | D: has **186 GB free (98 % used)**. Result volume is ≈ 0.7 KB/cell, so a 100 k-cell campaign ≈ 70 MB. Preflight fails the batch below a 5 GB floor |
| VRAM thrash from model switching | `--concurrency 1`, one resident model, explicit `keep_alive=10m` |
| silent idle loop ("it ran" ≠ progress) | the search reports `stalled` and exits non-zero when nothing improved (anti-spin) |
| model silently upgraded upstream | benchmark items frozen with source SHA-256; model digest recorded in the manifest |
| threshold drift after seeing data | design constants live in `configs/oasp.yaml`, hashed into the manifest |

**Free-tier fallback.** If a stage ever needs an LLM bigger than the GPU can hold, the
llama.cpp server already available locally (Hermes-managed, port 18434) serves
`Qwen3.6-35B-A3B` — still free, still local.

---

## 5. Phases, with stop/go gates

| phase | work | gate to proceed |
|---|---|---|
| **P0 — recover & harden** ✅ done | tree restored from the Recycle Bin; `runs/pilot_v1_dev` intact (2,154 cells); harness re-verified (27/27 evaluator tests); three NaN-contamination defects fixed | all tests pass on the recovered tree |
| **P1 — calibrate** | screen candidates on ARC-Challenge; pick the interior set; run the **noise floor** (M6) | ≥3 targets in `[0.35, 0.85]`; M6 measured |
| **P2 — re-run H1 (condition A)** | 32 surface variants × 60 dev items × interior targets | H1 verdict computable against `2 × M6` |
| **P3 — condition B** | semantic paraphrases + few-shot manipulation from the local optimizer | divergence A vs B reported |
| **P4 — landscape (H2/H3)** | 40-generation search, dev only; reason-coded discards | held-out gain vs seed, McNemar |
| **P5 — confirmatory** | ≥150 held-out items, ≥6 targets, Benjamini–Hochberg | only now are numbers citable as confirmatory |

**Nothing from P2–P4 may be cited as confirmatory.** Run ids containing `pilot` are labelled
`exploratory` in their manifest; the analysis prints the label at the top of every report.

---

## 6. Immediate next actions

1. Finish the ARC-Challenge screen; select the interior target set (data-driven, rule fixed).
2. Run the noise floor `noise_v1` — H1 is undecidable without it.
3. Re-run the condition-A matrix on the interior set (`pilot_v2_dev`).
4. Run the landscape (`search_v1`) with the local optimizer.
5. Only then: widen to ≥150 held-out items and ≥6 targets.

---

## 7. Honest risk register

| risk | status |
|---|---|
| H1 may be false (spread is noise) | **live** — the pilot says so for a ceiling-bound model; P1–P2 either overturn it on interior models or confirm it, and either outcome is publishable |
| H2 may fail (search gives no held-out gain) | **live** — this is the outcome that reduces OASP to a restatement; the protocol commits to reporting it |
| ARC label noise inflates every spread | known; it is conservative for a range-vs-floor comparison but caps achievable accuracy |
| surface-only variation is a weak treatment | addressed by condition B; A alone cannot establish the magnitude of the optimum gap |
| optimizer prior is not neutral | the deterministic feature-space generator is primary for exactly this reason; an LLM-written template is not treated as neutral |
| single-machine, single-GPU throughput | ~2 cells/s sustained; the confirmatory run is a multi-day campaign, which is what the 24/7 requirement is for |
