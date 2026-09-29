# JevRSI — read this first

> **The live research object is [`docs/RESEARCH_GOAL_v2_2026-09-29.md`](docs/RESEARCH_GOAL_v2_2026-09-29.md).**
> Everything else on this page is orientation. If the two disagree, the goal document wins.

---

## The object in one paragraph

A self-evolving research loop on a **Jev-style decision model** — given a state and a typed question
(choice / boolean / score), it returns a *distribution over answers* rather than generating text. The loop runs
entirely on one consumer GPU, decides keep/revert with a **pre-registered threshold** derived from two
independently measured noise floors, and opens the frozen test split exactly once.

| | |
|---|---|
| **Subject repository** | `E:/2026-AI4S/agent-jev` — clone of `github.com/malevrigns/agent-jev`, `main @ a965ca8f` |
| **Benchmark** | `LocalLLaMA/typed-decisions`, official test split — 400 cases / 2,000 questions |
| **Substrate** | `convaiinnovations/laya-multilingual` (322M) — trained full-parameter here, 0.22 s/step |
| **Optimised artefact** | the **readout**, then the training code — *not* a prompt |
| **Current level** | 0.5310 (5 pre-registered seeds), from a 0.3500 untrained start |
| **Figure to beat** | **0.799** (AnyJev L2, Qwen3-30B-A3B) |

## The current question, and why the axis moved

The obvious move — tune the training hyperparameters of a fixed substrate — has been **measured to
saturate**:

| budget | steps | dev Top-1 |
|---|---|---|
| 300 s | 1,499 | 0.5550 |
| 600 s | 2,968 | 0.5567 |
| 1,200 s | 2,850 | 0.5750 |
| 1,800 s | 8,345 | 0.5767 |

**5.6× the compute bought +2.17 pp — below the τ = 4.95 pp** measured for that split. Continuing to search
that axis has an expected detectable-success count of zero.

A **16.9 pp** gap sits on the readout side, and it is reachable with **no labels and no gradients**. So:

| | question | criterion |
|---|---|---|
| **Q1** | Does cyclic-shift marginalisation of the option list transfer to this architecture? | Δ > τ on dev |
| **Q2** | Where does the gap to `laya-typed-decisions` (0.768) come from — capacity, training volume, labels, or readout? | four-term decomposition |
| **Q3** | Is there any increment left for gradient training *after* the readout is fixed? | Δ > τ |

**Q1 is running**: zero labels, one forward pass per rotation, decision rule fixed before the numbers exist.

## Why the old ceilings were retired

The dataset card labels 0.704 a *factor ceiling* and 0.735 a *teacher self-agreement* ceiling. Earlier
versions of this work treated them as upper bounds. **They are not.** On the same 400-case /
2,000-decision split:

- `laya-typed-decisions` scores **0.768**
- a training-free readout (AnyJev L2, Nokia Applied Research, 869★) scores **0.786** (4B) and **0.799** (30B-A3B)

So 0.735 bounds only systems that reproduce *the teacher's own samples*; it does not bound a system that
predicts the scenario. The constants are reclassified: **0.470 is a floor**, 0.735 and 0.799 are reference
lines, and **0.799 is the figure to beat**.

## Where to look

| want | read |
|---|---|
| **the live objective** | [`docs/RESEARCH_GOAL_v2_2026-09-29.md`](docs/RESEARCH_GOAL_v2_2026-09-29.md) |
| the paper | [`manuscript.html`](manuscript.html) |
| all documents, rendered | [`html/index.html`](html/index.html) |
| the frozen pre-registration | [`docs/JevRSI_preregistration.md`](docs/JevRSI_preregistration.md) — **immutable after execution; amend, never rewrite** |
| what was retired and why | [`archive/superseded_2026-09-29_object_cleanup/MANIFEST.json`](archive/superseded_2026-09-29_object_cleanup/MANIFEST.json) |

### Gates — all must pass before any claim

```bash
python scripts/check_object_purity.py            # retired-object absence
python measurement/audit_pipeline.py             # structural audit of the live pipeline
python measurement/audit_manuscript_numbers.py   # every quantitative claim has a source
python measurement/test_meaning_boundary.py      # the meaning guardrail behaves as specified
python measurement/test_release_weights.py       # survive/fallback storage rule holds
python measurement/verify_split_against_protocol.py  # splits match the protocol manifest
python scripts/check_citation_provenance.py      # no unanchored external figure
python figures/fig_qc.py                         # figure layout audit
```

## Rules that bind the work

1. **No unverified numbers.** Every external figure carries a resolvable source (HF model-id + file path,
   arXiv id, or GitHub repo + path + commit). Enforced by `check_citation_provenance.py`.
2. **The frozen split is opened once.** Selection happens on dev; test is touched at the end.
3. **Negative results are first-class.** A plateau that fails the criterion ships with the same rigour as a
   win, including the full failure archive.
4. **Retire by moving, not by annotating.** Superseded documents live in `archive/<reason>_<date>/` with a
   hash manifest. A banner inside a file readers still open is not a retirement.
5. **The accept rule is audited, not asserted.** τ comes from two measured noise sources; a true-null audit
   records the false-accept rate of both the naive rule and the τ rule (measured: 5/10 vs 0/10).

## Layout

```
autoresearch/
├── README.md                     this file
├── manuscript.html               the paper
├── CURRENT_OBJECT.md             object + retirements (predates v2; see the goal document)
├── docs/RESEARCH_GOAL_v2_*.md    THE LIVE OBJECTIVE
├── docs/                         analyses, pre-registration, audits
├── measurement/                  every measurement script and its artifacts
├── figures/                      figure generators + programmatic layout audit
├── html/                         rendered document set (generated; markdown is the source)
├── lean/                         Lean 4 formalisation of the accept rule (NOT yet kernel-checked)
└── archive/                      everything retired, with hash manifests
```

**There is no live harness in this repository — the harness lives elsewhere, and it *is* live.**
`jsegov/autoresearch-win-rtx` (cloned at `D:/2026-AI4S/nanochat-autoresearch`) is the sanctioned loop
substrate: it carries `program.md`, a `results.tsv` ledger, and a read-only `prepare.py`. So `nanochat` /
`val_bpb` are **retired as a research object but live as the instrument** — a distinction
`scripts/check_object_purity.py` enforces mechanically. We borrow its framework; we do not import its
numbers ("借框架可以，借数字不行").

## Retired — do not work on these

**prompt-multiplier / HPFE / OASP / κ** · **Gate B / `val_bpb` / recursive self-training collapse** ·
**Zarankiewicz topic selection** · **MentalBench** · **AgentJev-0.6B as the retraining substrate** ·
**0.704 / 0.735 as attainable-accuracy ceilings**.

All preserved under `archive/`. See `CURRENT_OBJECT.md` for what each was and why it is not the object —
including the trap of the thirteen `program_variants/assembled/v00–v12.md` files, which are shaped exactly
like the `program.md` a brief might ask for but belong to the old object.
