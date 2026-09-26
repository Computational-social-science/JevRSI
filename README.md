# OASP — Optimization-as-Search Program

> **Object of study:** the **landing form (落地形态)** of reported LLM optimization
> strategies — not the benchmarks they are measured on, and not prompting in general.
>
> Reported practice: a strategy `S` (CoT, self-consistency, ReAct, ToT, …) is
> published, but its evaluation collapses to **one hand-authored static prompt
> template** evaluated **once** on **one static benchmark**. We measure what that
> collapse costs and how much of the reported effect is an artifact of the sampled point.

| | |
|---|---|
| **Effect** | **HPFE — Heuristic Prompt Fixation Effect**: strategies fixate on a hand-sampled point rather than searching prompt space |
| **Insight** | **OASP — Optimization-as-Search Principle**: prompt-space search, not single-point scoring, is the right model of LLM optimization |
| **Protocol** | [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) — pre-registered before data collection |
| **Brief** | [`archive/nhb_2026/Problem.docx`](archive/nhb_2026/Problem.docx) |
| **Supersedes** | the NHB LLM-mistranslation workspace, archived under `archive/nhb_2026/` |

> **Scope note (added 2026-09-26; FLAW 2 of [`docs/critical_review_2026-09-26.md`](docs/critical_review_2026-09-26.md)).**
> The measurement work this README documents — the MCQ/ARC variant matrix, its metrics, and its
> negative pilot result — is the **pilot**, and it measures **construct A** (measurement-instrument
> method variance: reproducibility of a reported score under evaluator-authored surface variation).
> It is **not** the project's object. The project's object is **construct B** (the landing form of an
> optimization strategy: whether the researcher's hand-authored template determines the result),
> carried by `nanochat`'s `program.md` — the file the human, not the agent, edits — and construct B
> is the primary object of study; it has not been tested yet. Nothing here is deleted; existing
> claims about "the project" are to be read as claims about the pilot.
>
> **DV note (added 2026-09-26; FLAW 6).** The construct-B dependent variable is written
> **`val_bpb@300s`** — validation bits per byte read at a **fixed 300-second wall-clock training
> budget**, hence compute-normalised. Cross-hardware comparisons and comparisons against fixed-step
> runs are **void**.

---

## The two names are deliberately distinct

The brief requires the **effect** (observable, quantifiable) and the **insight**
(falsifiable, mechanism-explaining) to be named with *different roots* so they can
never be confused in a manuscript. HPFE vs OASP satisfies that; each carries its own
falsifier, listed in [`docs/theory.md`](docs/theory.md).

---

## What makes this a test rather than an assertion

| Naive version | What this program does instead |
|---|---|
| "prompts matter" | **IPS (M2)**: fraction of items whose correctness *flips* across 32 semantically equivalent prompts. Under exact invariance its null is **exactly 0** — so any non-zero value is evidence needing no null distribution, only a CI. |
| "our prompt is better" | search on a **dev** split, gain measured on a **held-out** split (M7), against the noise floor (M6) |
| "the failure modes" | reason codes assigned **blind to score**, then tested for score separation (M8 / Fig 4) |
| "the optimizer improved it" | `config.assert_decoupled()` **hard-fails** if the optimizer is also a target |
| "it ran" | the search reports `stalled` loudly when nothing improved (anti-spin) |

---

## Layout

```
autoresearch/
├── RESEARCH_PROTOCOL.md      pre-registration: hypotheses, metrics, decision rules
├── docs/theory.md            effect/insight definitions + falsifiers
├── oasp/                     the harness
│   ├── config.py             single source of truth: paths, models, benchmarks, thresholds
│   ├── prompt_space.py       the 180-point surface-feature prompt space + sampling
│   ├── benchmark.py          frozen stimulus sets with source SHA-256
│   ├── evaluator.py          mechanical scorer (no LLM judge) + regression tests
│   ├── llm.py                Ollama / OpenRouter client with retries
│   ├── runner.py             crossed design with resume
│   ├── landscape.py          hill-climb + fallback ledger + anti-local-optimum mechanisms
│   ├── git_memory.py         accept = commit, reject = reset
│   ├── stats.py              M1–M8
│   ├── figures.py            publication figures
│   └── analyze.py            report CLI
├── benchmarks/               frozen items + provenance
├── runs/                     raw JSONL + manifests (append-only, resumable)
├── analysis/                 reports + figures
├── scripts/                  screen_models.py, run_noise_floor.py, run_landscape.py
└── archive/nhb_2026/         prior project, preserved
```

---

## Run

```bash
# 1. Screen targets for capability floor and format adherence (choose targets from data)
python scripts/screen_models.py --items 10 --budgets 64 192

# 2. The sensitivity matrix: 32 variants x 60 items x N models
python -m oasp.runner --run-id pilot_v1_dev --benchmark arc_easy \
    --models gemma3_4b qwen3_4b mistral_7b gemma3_12b \
    --variants 32 --items-dev 60 --items-test 60 --splits dev \
    --num-predict 192 --concurrency 1

# 3. Noise floor (M6)
python scripts/run_noise_floor.py --run-id noise_v1 --models gemma3_4b mistral_7b

# 4. Analysis -> report.md + figures
python -m oasp.analyze --run-id pilot_v1_dev --split dev --noise-run-id noise_v1

# 5. The search landscape (H2/H3)
python scripts/run_landscape.py --run-id search_v1 --generations 40 --items 30
python -m oasp.analyze --run-id search_v1 --landscape
```

No install step: the harness uses only `requests`, `numpy`, `pandas`, `scipy`,
`matplotlib`, `pyarrow` and stdlib `subprocess`-driven git.

---

## Two rules that keep the results honest

1. **A pilot number is never a confirmatory number.** Runs whose id contains
   `pilot` are labelled `exploratory` in their manifest; the confirmatory H2 test
   needs ≥150 held-out items, not 60 (`RESEARCH_PROTOCOL.md` §6).
2. **Nothing in the scoring path calls a model.** `oasp/evaluator.py` is the only
   scorer, its regression tests live in `tests/test_evaluator.py`, and a false
   extraction there would appear as a *finding* about prompt sensitivity rather
   than as a bug — which is why it is tested in both directions.
