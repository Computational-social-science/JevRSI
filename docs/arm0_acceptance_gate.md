# ARM 0 — the acceptance gate, declared before the result exists

Written 2026-10-03 06:58 while arm 0 was still training. **The gate is written now, not after the
result, because `acceptance-gate` step 0 requires it:** *"Before any fit, search, or model choice,
write down: which evaluation points are reserved, what the independent route will be, what digit
count will count as passing, and what the controls are. A gate written after the candidate exists
inherits the candidate's blind spots."*

This file is that, for the first real arm. It is deliberately written in an order that cannot be
adjusted later: pass/fail thresholds first, then the checks, then what the checks cannot do.

---

## The verdict vocabulary

Three outcomes, and nothing in between. `acceptance-gate`: *"CLOSED means every part above passed
and the counts are written down. Anything less is OPEN, reported with what was established and what
remains. An honest OPEN is a result; a false CLOSED is damage."*

| verdict | meaning |
|---|---|
| **CLOSED** | every gate below passed, with the numbers written down |
| **OPEN** | something did not pass, or could not be run — stated with what remains |
| **INVALID** | a control fired, or an invariant broke; nothing downstream may be built on this arm |

There is no "close enough". `acceptance-gate`: *"Adjectives in place of counts"* is a named failure.

---

## Gate A — the run is the one we think it is

| check | pass | fail | why it can fail |
|---|---|---|---|
| `n_train_cases` | exactly **6277** | any other value | the split or the `sources` filter moved; their published number |
| `tower_keys` | **309** | another count | a different backbone or a frozen tower |
| `tapped_layer_type` | `full_attention` | linear/other | the readout tap landed on the wrong attention class |
| `tapped_layer` | **27** | another layer | `readout_layer: -1` resolved differently |
| `code_version` / module hashes | match the recorded `LAUNCH.json` fingerprint | mismatch | the run used a different code revision |
| `spec` in `meta.json` | field-for-field the published spec, modulo the one recorded deviation | any drift | the spec was edited between launch and run |
| tower moved | `max |dw| > 0` (their guard fires) | guard silent | gradients never reached the tower |

**These can fail and are not decoration.** Two of them already have: the spec drift check caught
`lr_head` and the missing `option_order` before launch, and the negative controls for it are recorded.

---

## Gate B — the evaluation is the one we think it is

| check | pass |
|---|---|
| records count | **12** = 2 roles × 2 option orders × 3 targets |
| items count | **21792** |
| both option orders present | `canonical` **and** `reversed` for every target and role |
| `pooled_top1` control (zero-shot) | reproduces **0.4025** on typed_decisions / canonical |

**The control row is the calibration.** If the zero-shot control does not reproduce the number
measured before arm 0 launched, the evaluation itself has moved, and no candidate number from this
arm means anything. **That is a check that can fail and has been watched to fail** — the zero-shot
row differed between the two order values in the probe, so the machinery is live.

---

## Gate C — the result, against a number that could not have known it

This is the gate that decides the arm, and it is written in two parts because the reference and the
anchor answer different questions.

**C1 — the anchor.** The arm passes C1 when

```
pooled_top1(typed_decisions, candidate, canonical)  >  0.4025
```

where **0.4025 is the zero-shot control's own score**, measured before this arm existed. It is a
legitimate independent reference: it is produced by a different readout (`LogprobReadout`, no
training) on the same split, and it entered no part of this arm's fit.

**C1 can fail and already has.** The 1-step probe scored **0.3845**, below the anchor. So this is not
a gate that only knows how to pass.

**C2 — the reference.** Their v1.0 scores **0.6525**. This is *not* a pass threshold. Their backbone
is a 2B model more than three times larger; ours is the deliberate independent variable. C2 is
reported as **headroom consumed**: `(candidate − 0.4025) / (0.6525 − 0.4025)`.

**Stating explicitly what would count as failure, so the number cannot be rationalised afterwards:**

| outcome | verdict |
|---|---|
| `candidate ≤ 0.4025` | **OPEN** — the arm did not beat its own zero-shot control. The reproduction did not work. |
| `0.4025 < candidate < 0.50` | **OPEN** — beats the anchor by an amount a 0.6B might produce from 1500 steps, but the loop's progress is not established. Recorded, not celebrated. |
| `candidate ≥ 0.50` | **CLOSED on C1**, with headroom consumed reported |

---

## Gate D — the hypothesis is testable and could be wrong

The order-gap reading — that a model selecting option *content* scores the same under either
presentation, while one tracking *position* does not — becomes a **testable** claim here:

- the zero-shot control shows a **3.1 pp** gap,
- their v1.0 shows **0.0 pp**,
- the reading predicts arm 0's gap is **smaller than 3.1 pp**.

**Declared now, before the number exists.** If arm 0's gap is ≥ 3.1 pp the reading is falsified for
this backbone and the doc that proposed it says so.

---

## What this gate cannot do — the honest limits

These are not caveats; they are the reasons a CLOSED verdict here is weaker than it looks, and
`independence-bookkeeping` requires them written down before the comparison runs.

1. **There is no independent route to the score.** `pooled_top1` is computed from the harness's own
   `pred` field. `report_from_items.py` does not re-derive a single prediction — it counts
   agreements between two values the harness produced. **A bug in the harness's prediction path
   would be invisible to this gate**, because the check shares the whole production path with the
   thing it checks. This is `acceptance-gate`'s *"False independence through a shared
   representation"*, and it is the deepest weakness here.

2. **Determinism is not correctness.** Two runs agreeing to full float precision proves the machinery
   is reproducible. `planted-truth`: *"a pipeline that is consistently wrong is still consistent.
   Internal agreement, stability under iterations, and smooth residuals are properties of the
   machinery, not of the answer."* An earlier write-up in this repository offered that agreement as
   evidence; it is not, and the claim is corrected alongside this file.

3. **No planted truth has ever been run through this pipeline.** The harness has never been fed data
   whose correct answer was written down first. Every number it has produced is real-data output,
   and real data cannot grade a pipeline.

4. **Thresholds here are fixed and will not move.** If a value lands just outside a band, the verdict
   is OPEN and says so. `acceptance-gate`: *"Thresholds are fixed at step 0 and never touched after
   the candidate exists; a candidate that needs the threshold moved has failed."*

---

## Controls still owed, and what they would settle

| control | what it would establish | status |
|---|---|---|
| **Planted-truth battery** — synthetic corpus with known gold labels through the full harness, demanding recovery | that the harness's scoring path computes what it claims | **not built** — the single largest gap |
| **Corrupted-gold twin** — flip a known fraction of gold labels, demand the score drops by the expected amount | that the scorer responds to correctness at all | **not built** |
| **Independent prediction route** — re-derive a handful of predictions outside the harness from raw model outputs | that `pred` means what we think | **not built** |
| **Order-gap falsifier** | Gate D above | declared, runs with the arm |

The first two are the `planted-truth` procedure applied to this pipeline. They are designable now and
runnable the moment the GPU frees; they must run **before any arm's number is used in a claim**, not
after a doubt arises.
