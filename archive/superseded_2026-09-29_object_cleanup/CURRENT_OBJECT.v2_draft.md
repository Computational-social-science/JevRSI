# CURRENT OBJECT — read this first

> **The live research objective is [`docs/RESEARCH_GOAL_v2_2026-09-29.md`](docs/RESEARCH_GOAL_v2_2026-09-29.md).**
> This file is the single source of truth for *what the object is* and *what is retired*. If any file
> disagrees with this page, that file is a retired artifact and should be archived.
>
> **Revised 2026-09-29.** The previous revision of this file defined the object as a retraining loop over
> `AgentJev-0.6B` with a 79.25% baseline. That object is **retired** and the file that stated it has been
> archived. The reason is recorded below, because the failure mode is worth remembering: a *plausible,
> internally consistent, authoritative-looking* document is more dangerous than an obviously broken one,
> and rewriting the README does not fix it — a fresh session loads the SSOT, not the README.

---

## 1. What we are working on

**JevRSI** — a self-evolving research loop on a **Jev-style decision model**: given a state and a typed
question (choice / boolean / score), it returns a *distribution over answers* rather than generating text.

| | |
|---|---|
| **Live objective** | [`docs/RESEARCH_GOAL_v2_2026-09-29.md`](docs/RESEARCH_GOAL_v2_2026-09-29.md) — Q1/Q2/Q3, criteria pre-registered |
| **Subject repository** | `E:/2026-AI4S/agent-jev` — clone of `github.com/malevrigns/agent-jev`, `main @ a965ca8f` |
| **Benchmark** | `LocalLLaMA/typed-decisions` — train 960 / dev 120 / calibration 120 / **test 400** (2,000 questions), verified case-by-case against the protocol manifest |
| **Substrate** | `convaiinnovations/laya-multilingual` (322M) — trained full-parameter on this machine, 0.22 s/step, 11.5 GiB peak |
| **Optimised artefact** | the **readout**, then the training code — *not* a prompt |
| **Loop harness** | `D:/2026-AI4S/nanochat-autoresearch` — `jsegov/autoresearch-win-rtx`. Borrow its *framework*; do not import its *numbers*. |
| **Metric** | Top-1 against the teacher distribution, with soft CE / Brier / ECE / Score MAE pooled and per type |
| **Current level** | **0.5310** (5 pre-registered seeds, mean), from a **0.3500** untrained start |
| **Figure to beat** | **0.799** — AnyJev L2 on Qwen3-30B-A3B, same test split |

## 2. The three live questions

| | question | pre-registered criterion | status |
|---|---|---|---|
| **Q1** | Does cyclic-shift marginalisation of the option list transfer to this architecture? | paired Δ > τ = 4.95 pp on dev | **DONE — NOT SUPPORTED.** Δ = −1.00 pp, CI [−4.20, +2.20], McNemar p = 0.61. Mechanism engaged (169/180 choice argmax changed) yet the effect is net negative. |
| **Q2** | Where does the gap to `laya-typed-decisions` (0.768) come from — substrate capacity, training volume, label count, or readout form? | four-term decomposition | **next** |
| **Q3** | Is there any increment left for gradient training *after* the readout is fixed? | paired Δ > τ | pending Q2 |

**Q1's negative result is a finding, not a failure.** It excludes option-order bias as an explanation for
the gap to public results, which narrows Q2.

## 3. What is retired, and why

| retired | why | where it went |
|---|---|---|
| `prompt-multiplier` / HPFE / OASP / κ | the earlier object; its MLE objective was audited as tautological | `archive/retired_object_prompt_multiplier_2026-09-27/` |
| Gate B / `val_bpb` / recursive self-training collapse | data-policy sensitivity. **Retired as an object, LIVE as the instrument** — `autoresearch-win-rtx` is the sanctioned harness | — (distinction enforced by `HARNESS_CONCEPTS`, CHECK D) |
| Zarankiewicz / cap-set topic selection | a wrong turn | `archive/wrong_turns/` |
| MentalBench | not the object | `archive/` |
| **The 6-week / ≥2000-mutation plan** | its search axis is measured to saturate (+2.17 pp over 5.6× compute, **below** τ = 4.95 pp), so its expected detectable-success count is **0**; its three cited figures (66.4%, ECE 0.0712, ~50 ms) have no source; its 30/80/60 split puts the per-cycle decision below the measurement floor | `archive/superseded_2026-09-29_object_cleanup/` |
| **`AgentJev-0.6B` as the retraining substrate** | superseded by the readout-geometry axis; adopting it would discard a substrate already trained and three-way-verified here | `archive/superseded_2026-09-29_object_cleanup/SSOT/` |
| **0.704 / 0.735 as attainable-accuracy ceilings** | public measurements on the *same* 400-case split reach 0.768, 0.786 and **0.799**. 0.735 bounds only systems that reproduce *the teacher's own samples* | reclassified in `measurement/run_hillclimb.py`; test in `measurement/test_meaning_boundary.py` |
| **The previous SSOT** (`CURRENT_OBJECT.md` @ 2026-09-27) | it declared the retired object *and* declared itself authoritative | `archive/superseded_2026-09-29_object_cleanup/SSOT/` |

**The pattern worth naming**: each retirement was a document that read as current. The guard that catches
this is not a keyword list — it is `CHECK F`, which asserts a retired document is **absent from the live
tree**, and which also asserts the archive copy is **present**, because a retirement that lost its record
is indistinguishable from a deletion.

## 4. Rules that bind the work

1. **No unverified numbers.** Every external figure carries a resolvable source (HF model-id + file path,
   arXiv id, or GitHub repo + path + commit). Enforced by `scripts/check_citation_provenance.py`.
2. **The frozen split is opened once.** Selection on dev; test at the end.
3. **Negative results are first-class**, including their mechanism and their power.
4. **Retire by moving, not by annotating** — and always leave an SSOT behind.
5. **The accept rule is audited, not asserted**: τ from two measured noise sources; true-null false-accept
   measured at 5/10 for the naive rule and 0/10 for the τ rule.

## 5. Drift check

```bash
python scripts/check_object_purity.py
```

Asserts absence: retired artifacts, actionable retired instructions, retired concepts named without a
retirement declaration, **and (CHECK F) retired documents back in the live tree**.
