# TERMINOLOGY — autoresearch is not ours to redefine

**Recorded 2026-09-29, immediately after an agent introduced a compound term that should not exist.**

## The ruling

**"autoresearch" means [karpathy/autoresearch](https://github.com/karpathy/autoresearch).** Nothing else.

Our Windows-local instrument is **`jsegov/autoresearch-win-rtx`**, a fork of it. When this project says
"the autoresearch harness", it means that repository and its design. We borrow its **ideas and
iteration framework**; we do not reproduce its work, and retraining that harness's own substrate is
not a goal of this project.

## What is forbidden

**`bilevel autoresearch` is not a thing.** An agent introduced it on 2026-09-29 while naming the
pipeline skeleton, and it has been removed from every file.

Two unrelated concepts were being welded into one name:

- **karpathy/autoresearch** — a concrete, existing repository with a concrete design: an agent edits
  ONE file (`train.py`), trains for a **fixed 5-minute budget**, checks whether the metric improved,
  keeps or discards, appends to a log, repeats. ~12 experiments/hour, ~100 overnight. The
  `program.md` file is the constitution the agents read.
- **bilevel optimization** — a standard term from the optimization literature, where an upper level
  optimizes a meta-objective whose evaluation requires solving a lower-level subproblem. It has
  nothing to do with the repository above, and this project uses no bilevel optimizer.

Calling our scheduler a "bilevel autoresearch" therefore does two harmful things at once: it implies a
lineage from karpathy/autoresearch that does not exist, and it makes an unrelated optimization term
look like a property of the borrowed framework. Both are exactly the kind of drift this repository's
retired-object guard exists to prevent — the vocabulary of a retired or foreign idea surviving into the
live tree and being read as current.

## The naming this project actually uses

| concept | name | what it is |
|---|---|---|
| the borrowed instrument | **the autoresearch harness** (`jsegov/autoresearch-win-rtx`) | karpathy/autoresearch, forked for Windows + consumer RTX |
| our loop skeleton | **the pipeline** (`pipeline/`) | Select → Mutate → Execute → Judge → Archive → Persist, on `autoresearch.jsonl` |
| our two-level schedule | **the outer level / the inner level** | harness routes (seconds) and parameter routes (minutes, gated) |
| the prior-art tier | **open-jev-fast** | adjudicates whether an external idea is transferable here |
| the first generation | **AgentJev-0.6B** | the seed the pipeline improves on |

"Outer level" and "inner level" are structural descriptions of our own schedule. They are not a claim
about karpathy/autoresearch's design, and they do not invoke bilevel optimization.

## What we did take from the harness, and what we deliberately did not

Taken — these are the framework, and the reason this project uses it at all:
- **one mutable surface per cycle.** The harness lets the agent touch only `train.py`. Ours is
  `pipeline/routes.py`: every candidate must name a declared route, so a cycle's cost, axis and gate
  status are known before it runs.
- **an append-only log that rebuilds all control state.** `autoresearch.jsonl` + `results.tsv`.
- **a cool-down after consecutive failures of one module family**, and a failure log the selector must
  consult before proposing anything.
- **persist the attempt before the reset** — the rejection protocol, which the harness specifies and
  which this project has not yet exercised.

Not taken:
- **the fixed 5-minute budget as a number.** The principle — a fixed budget makes experiments mutually
  comparable — is kept, but 5 minutes is not this project's budget. Measured here: 12.6 s/step, so a
  600-step parameter cycle is **2.11 h**, and a harness cycle is **~10 ms**. That ratio is why the
  harness is the primary route and the parameter route is gated behind a measured saturation
  condition rather than a schedule.
- **the harness's own substrate, its perplexity benchmark, and its synthetic-corpus training set.**
  Those belong to the borrowed instrument's lineage. Here the benchmark is `typed-decisions` and the
  model is a non-generative scoring model. Do not import numbers from that substrate.
- **the Linux/H100 path**, removed in the fork and not supported.

## The guard

`scripts/check_terminology.py` fails the build on the compound term and on a few adjacent
misattributions. It exists because the failure mode is silent: a wrong name reads naturally, nothing
errors, and the next session inherits it as vocabulary.

```bash
python scripts/check_terminology.py
```
