# Day-1 Baseline and Feasibility Adjudication — 2026-09-29

**Target** [`RTX4070_SelfEvolving_Jev_Research_Proposal.md`](../RTX4070_SelfEvolving_Jev_Research_Proposal.md) §6.1 Phase 1
**Instrument** `measurement/INSTRUMENT_CALIBRATION.json` (verified working in this session)
**Status** Baseline measured; **one feasibility conflict awaits adjudication by the project owner**

---

## 1. Day-1 baseline (L1-calibrated seed, dev split)

Command (frozen evaluator, not a single line changed):
```
python scripts/evaluate_split.py --checkpoint checkpoints/autoresearch/final.safetensors --split dev
```

| Metric | Value |
|---|---|
| **Accuracy** | **0.8150** (489/600) |
| **Chance-corrected skill** | **0.7326** |
| Soft cross-entropy | 0.8243 |
| Brier | 0.0476 |
| **ECE** | **0.1481** |
| Score MAE | 0.1849 |
| Wall clock | **38.9 s** / 600 items |

Breakdown by question type: boolean 0.8833 · choice 0.7833 · score 0.7875
Breakdown by workflow: invoice 0.8400 · agent_trace 0.8133 · customer_service 0.8067 · security 0.8000

**test split untouched** (Rule 1).

### 1.1 One citation mismatch, confirmed by measurement

The proposal §2.1 claims this seed has "adequate calibration quality (ECE = 0.0712)".
**Measured ECE = 0.1481, which is 2.08× the proposal's value.**

Independently verified earlier: `temperatures.json` contains only **3 per-primitive temperature scalars**
(boolean 1.0718 / choice 1.0353 / score 1.0718) and no confidence or correctness data, so
**ECE cannot be derived from the checkpoint** — that number has no source in the published material.

**This does not change the executability of the proposal** (the baseline runs, 38.9 s, all 6 metrics present), but Rule 4
requires claiming superiority "only on metrics the baseline actually reported", so **ECE 0.0712 cannot be used as the comparison value**.
The actual starting point is **0.1481**, which in fact leaves room for calibration-type variants.

---

## 2. The proposal's `ε_proxy` / `ε_M` are undefined — derivation now supplied

The proposal §3.3 Phase 4 uses `ε_proxy` and `ε_M` for keep/revert decisions; **both are named, neither is defined**.
They are not free parameters: each one is a decision threshold, and setting them by feel means making choices on noise.

`loop/epsilon.py` derives them from two already-measured quantities:

```
ε = z_bonferroni · sqrt( SE_paired² · (1 + replicas) + floor_B² )
```

All three details are motivated:
- **SE scales with the number of CASEs, not the number of items.** Empirically verified: test 400 cases SE=1.067 pp, dev 120 cases SE=2.140 pp; the 1/√cases prediction for dev = 1.948 pp (the measured value is 1.10× that). **The independent unit is the case**, because the 5 items of the same case are not independent.
- **The `(1 + replicas)` term** is the parent generation's own variance: a candidate is compared against a sampled parent.
- **Bonferroni** pays for **every** decision the loop will make, not just the current one.

### 2.1 Derivation result for the dev split

| Quantity | Value |
|---|---|
| paired SE (measured) | 2.14 pp |
| Floor B (run-to-run) | 0.40 pp |
| SE including the replica term | 3.05 pp |
| Bonferroni per-comparison (95% family, H=1000) | 5.0e-05 |
| **ε** | **12.38 pp** |

**Floor B must be supplied by default**: it is a property of the **training path**, not of the split.
QLoRA has a different numerical noise structure on different base models, so the previous run's 0.40 pp cannot simply be inherited.
`loop/epsilon.py` **refuses to emit a threshold** when Floor B is missing (exit 2), unless `--allow-missing-floor-b` is passed explicitly.

---

## 3. ⚠ Feasibility conflict: the proposal's 30/80/60 split

The proposal §3.2 specifies proxy 30 items / medium 80 items / frozen V 60 items.
Under the measured scaling law (5 items/case, the independent unit is the case):

| Split | Items | ≈cases | SE | Raw 95% half-width | Bonferroni H=1000 |
|---|---|---|---|---|---|
| proxy P | 30 | 6 | 8.71 pp | ±17.07 pp | **50.35 pp** *(with the replica term; was 35.33 pp without it)* |
| medium M | 80 | 16 | 5.33 pp | ±10.45 pp | **30.81 pp** *(with the replica term; was 21.63 pp without it)* |
| frozen V | 60 | 12 | 6.16 pp | ±12.07 pp | **35.61 pp** *(with the replica term; was 24.98 pp without it)* |
| *(control) dev 120 cases* | *600* | *120* | *2.14 pp* | *±4.20 pp* | *12.38 pp* |

**The usable effect space is bounded at roughly 20 pp** (starting point 0.815, theoretical ceiling 1).

> **Under the proposal's split, medium M's decision threshold of 30.81 pp is larger than the entire usable space.**
> That is: even if the loop finds a real improvement, in a loop with H≈1000 it is **undetectable**.
> And a scale of 2000 mutations implies H far greater than 1000, so the threshold only gets higher.

**This is not "the effect may be small"; the decision mechanism is arithmetically incapable of distinguishing a real improvement from noise.**

### 3.1 Three ways out (please adjudicate)

| Option | What it does | Cost |
|---|---|---|
| **A. Keep the official split** (recommended) | train 960 / dev 120 / calibration 120 / test 400, using the already-verified 960/120/120+400 | Abandons the proposal's three-way partition; **but this is the only option that is arithmetically decidable** |
| **B. Keep the three-way split but lower H** | 30/80/60, H capped at 50 → threshold drops to about 6.5 pp | The mutation scale drops from 2000 to a few dozen, **voiding the proposal's core selling point (24/7 continuous search)** |
| **C. Three-way split + pooled repetitions** | Repeat each of 30/80/60 multiple times and take the mean | Cost per decision ×N, **and the SE falls only as √N**, still far above the effect space |

**My judgment is A**, on the grounds that it has already been verified case-by-case on this machine (`verify_split_against_protocol.py`),
and that ε=12.38 pp for dev 120 cases, though large, is **smaller than** the 20 pp usable space — it is **decidable**.

But this is your proposal, **you decide**.

---

## 4. Components already in place

| Component | Status |
|---|---|
| Day-1 baseline (raw/L0/L1 path) | ✅ 38.9 s/600 items, frozen evaluator |
| Decision-threshold derivation | ✅ `loop/epsilon.py`, with mandatory Floor B validation |
| Noise floor | ✅ `measurement/INSTRUMENT_CALIBRATION.json` |
| Case-by-case split verification | ✅ `verify_split_against_protocol.py` |
| Contamination / citation / split gates | ✅ 8/8 PASS |
| Controller / MAP-Elites / JSONL state machine | ⬜ not built |
| QLoRA training loop | ⬜ feasibility not verified (one of the success gates in proposal §6.1) |
| Floor B on QLoRA seeds | ⬜ **not measured**, and `epsilon.py` will refuse to emit a threshold before that |

> **Correction, 2026-09-29 (Rule 1 audit).** The Bonferroni column in the table above originally
> reported `z x SE` only. Section 2.1's derivation for the official dev split includes the parent's own
> sampling variance, `SE x sqrt(1 + replicas)`, with `1 + replicas = 2.0313` implied by the document's own
> `3.05 / 2.14`. The three thresholds are therefore restated on the same footing as Sec 2.1. **The
> conclusion is unchanged and in fact strengthened**: every threshold rises, so the proposal's split
> remains even less able to separate a real improvement from noise, and decision D1 -- keep the
> official 960/120/120+400 split -- stands. The z used is the document's own 4.0556.

> **Correction, 2026-09-29 (Rule 1 audit).** The accuracy row read `0.8150 (488/600)`. Recomputing with
> the frozen evaluator (`typed_decisions.experiment.metrics` over `measurement/day1_raw_dev.json`) gives
> **0.8150 = 489/600**; 488/600 would be 0.8133. The count was 489. The metric value was right and the
> count in the prose was wrong; only the count is corrected here.
