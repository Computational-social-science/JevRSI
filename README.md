# JevRSI

> **The research objective is [`RTX4070_SelfEvolving_Jev_Research_Proposal.md`](RTX4070_SelfEvolving_Jev_Research_Proposal.md).**
> **The single source of truth is [`CURRENT_OBJECT.md`](CURRENT_OBJECT.md).** If any file disagrees with
> those two, those two win.

## The objective, in one paragraph

A 24/7 self-evolving optimisation loop for a Jev-style decision model, under two hard constraints: **one
RTX 4070 (12 GB, 150 W cap)** and **zero external API**. Seed: `AgentJev-0.6B` (Qwen3-0.6B backbone,
permutation-equivariant scoring head) + QLoRA rank-16 + AnyJev L1 calibration. A deterministic
multi-fidelity controller (Select → Mutate → Train → Evaluate → Archive → Persist) searches for a
configuration that beats the L1-calibrated seed on a frozen held-out set, with every attempt logged and
every failure shipped.

## What is already built, and what is not

| | status |
|---|---|
| **The instrument** — two measured noise floors, a pre-registered threshold, and a true-null audit of the accept rule | ✅ built, and calibrated on this benchmark |
| **The objective** — the six-week single-GPU plan | ⬜ not started |
| **The loop** — controller, MAP-Elites archive, JSONL state machine | ⬜ not built |
| **Checkpoints for the seed** | ⬜ none; the 11 deleted checkpoints belonged to a superseded substrate |

## The one thing to settle before running anything

The proposal's success criterion is *"the lower bound of a bootstrap 95% CI (elite − L1 seed) on V is
strictly positive."* Measured on this benchmark, **that rule fires on 5 of 10 pure-noise comparisons**
(`measurement/accept_rule_audit.py` — five replicates differing only by seed give ten pairs, every one a
trial under a true null). A rule that accepts half of pure noise cannot support a success claim.

The replacement is already calibrated and in `measurement/INSTRUMENT_CALIBRATION.json`:

| quantity | value | scope |
|---|---|---|
| **Floor A** — measurement sampling | SE **1.07 pp**, design effect 1.384 | a property of the 400-case split; **transfers to any model** |
| **Floor B** — run-to-run | SD **0.40 pp**, df = 4 | a property of *this training path*; **re-derive for the QLoRA seed** |
| **τ (test)** | **3.11 pp** | decisions on the frozen 400-case split |
| **τ (dev)** | **4.95 pp** | selection on the 120-case split, Bonferroni over 1000 decisions |
| naive rule | 5/10 fire | **replace with Δ > τ**, which fires 0/10 |

This is not an objection to the objective. It is the instrument the objective needs in order to be
decidable — and it is already paid for.

## Layout

```
autoresearch/
├── CURRENT_OBJECT.md                          SSOT: the object, the retirements, the drift checks
├── RTX4070_SelfEvolving_Jev_Research_Proposal.md   THE OBJECTIVE
├── measurement/
│   ├── INSTRUMENT_CALIBRATION.json            the two floors, τ, the accept-rule audit — self-contained
│   ├── tau_calibration.py                     derive τ for any split, with multiplicity control
│   ├── floor_a_cluster_bootstrap.py           cluster bootstrap (questions nest inside cases)
│   ├── accept_rule_audit.py                   empirical false-accept rate under a true null
│   ├── provenance.py                          zlib fingerprint; hashlib segfaults with torch+pyarrow
│   ├── audit_pipeline.py                      structural audit; runs as a preflight gate
│   └── test_*.py, verify_split_against_protocol.py, eval_laya_split.py
├── figures/                                   generators + programmatic layout audit (fig_qc.py)
├── scripts/check_object_purity.py             asserts ABSENCE of retired objects
├── scripts/check_citation_provenance.py       every external number carries a resolvable source
└── archive/                                   everything superseded, with hash manifests
```

## Gates — all must pass before any claim

```bash
python scripts/check_object_purity.py
python measurement/audit_pipeline.py
python measurement/audit_manuscript_numbers.py
python measurement/test_meaning_boundary.py
python measurement/test_release_weights.py
python measurement/verify_split_against_protocol.py
python scripts/check_citation_provenance.py
python figures/fig_qc.py
```

## Rules

1. **No unverified numbers.** Every external figure carries a resolvable source (HF model-id + file path,
   arXiv id, or GitHub repo + path + commit).
2. **The frozen split is opened once.** Selection on dev/medium; the held-out set at pre-registered checkpoints.
3. **Negative results are first-class**, with their mechanism and their power.
4. **Retire by moving, not by annotating** — and always leave an SSOT behind.
5. **State the df beside every SD.** A pilot once overstated the run-to-run floor by 3.53× because it was
   measured at df = 1.
6. **The agent does not retire the owner's objective.** An audit may report that a cited figure has no
   source; whether a stated objective is still pursued is the owner's decision. (Learned 2026-09-29, the
   hard way — see `archive/superseded_2026-09-29_object_cleanup/MANIFEST.json`.)

## Retired

`prompt-multiplier` / HPFE / OASP / κ · Gate B / `val_bpb` / recursive self-training collapse ·
Zarankiewicz topic selection · MentalBench · **the laya-multilingual substrate as the optimisation
target** (its 11 checkpoints and manuscript are archived; its measured floors survive in
`INSTRUMENT_CALIBRATION.json`) · **0.704 / 0.735 as attainable-accuracy ceilings** (public measurements on
the same split reach 0.768, 0.786, 0.799).

`nanochat` / `val_bpb` remain **retired as an object but live as the instrument** — `autoresearch-win-rtx`
is the sanctioned loop substrate. Borrow the framework; do not import its numbers.
