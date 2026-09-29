# Day-1 Progress Log — 2026-09-29 (living document, appended as work proceeds)

**Goal** [`RTX4070_SelfEvolving_Jev_Research_Proposal.md`](../RTX4070_SelfEvolving_Jev_Research_Proposal.md)
**Principle** Criteria are fixed before any numbers are seen; negative results are recorded with the same weight as positive ones.

---

## Established

### 1. Day-1 baseline (L1-calibrated seed, dev 600 items, 38.9 s, test untouched)

| Metric | Value |
|---|---|
| **Accuracy** | **0.8150** (488/600) |
| **Chance-corrected skill** | **0.7326** |
| Soft CE | 0.8243 |
| Brier | 0.0476 |
| **ECE** | **0.1481** |
| Score MAE | 0.1849 |

By archetype: boolean 0.8833 · choice 0.7833 · score 0.7875
By workflow: invoice 0.8400 · agent_trace 0.8133 · customer_service 0.8067 · security 0.8000

### 2. The ECE cited by the proposal has no source — confirmed by measurement

The proposal §2.1 claims "adequate calibration quality (ECE = 0.0712)".
**Measured 0.1481, i.e. 2.08× the proposal value.**

Measured content of `checkpoints/autoresearch/temperatures.json`:

```json
{"boolean": {"temperature": 1.0717734625362931, "calibration_n": 180, "soft_cross_entropy": 0.5016},
 "choice":   {"temperature": 1.0352649238413776, "calibration_n": 180, "soft_cross_entropy": 0.9666},
 "score":    {"temperature": 1.0717734625362931, "calibration_n": 240, "soft_cross_entropy": 1.0155}}
```

**3 per-primitive scalars**, not the "per-(model, question)" claimed in proposal §2.2.
The file contains no confidence or correctness data, so **ECE cannot be derived from the checkpoint**.

**Consequence**: under the proposal's own Rule 4 ("claim superiority only on metrics actually reported at baseline"),
**ECE 0.0712 cannot serve as the comparison value**. The real starting point is 0.1481 — which instead leaves room for calibration-type mutation.

### 3. A mathematical fact that must be stated first: temperature does not change Top-1

I at one point treated "raw and L1 accuracy are exactly identical" as a bug; after measuring, I confirmed that **this is definitional**:

> For fixed logits, `softmax(z/T)` and `softmax(z)` have **identical argmax** (the monotone transform with T > 0 preserves order).

Verified item by item (same logits, three temperature settings):

| T | argmax | probs |
|---|---|---|
| 1.0000 | 1 | [0.0067, **0.6673**, 0.0548, 0.2712] |
| 1.0353 | 1 | [0.0077, **0.6578**, 0.0588, 0.2757] |
| 1.0718 | 1 | [0.0089, **0.6483**, 0.0629, 0.2799] |

**Corollary, directly affecting the proposal's search-space design**:

- The proposal's Table 3 lists "AnyJev L0/L1 calibration strategies" as a **P0** candidate
- But **temperature-type calibration has an identically zero effect on Top-1**, showing up only in ECE / Brier / soft-CE
- Therefore fitness **must include calibration metrics**, otherwise the entire P0 family is invisible in the selection signal

**This is not an objection to the proposal; its fitness definition needs to explicitly include calibration terms.**
The instrument in this project happens to be able to measure it: ECE 0.1481 is currently the worst single metric.

### 2b. Published L1 calibration gets worse on ECE — and this is a metric conflict, not a defect

Same checkpoint, same logits, only the temperature changes (T=1.0 raw vs the published L1 with T=per-primitive), dev 600 items:

| Metric | raw (T=1.0) | L1 (published) | Δ | Direction |
|---|---|---|---|---|
| Accuracy | 0.8150 | 0.8150 | +0.0000 | **identical (definitional)** |
| Chance-corrected | 0.7326 | 0.7326 | +0.0000 | identical (definitional) |
| Soft CE | 0.8286 | 0.8243 | −0.0043 | **L1 better** |
| Brier | 0.0491 | 0.0476 | −0.0015 | **L1 better** |
| **ECE** | **0.1368** | **0.1481** | **+0.0113** | **L1 worse** |
| Score MAE | 0.1862 | 0.1849 | −0.0013 | L1 better |

Per-archetype ECE — **all three get worse**: boolean +0.0087 · choice +0.0073 · score +0.0156.

**The self-implemented ECE agrees exactly with the frozen evaluator (0.1368)**, so the sweep results are trustworthy.

#### An intermediate conclusion that was refuted (recorded here because it nearly became a wrong conclusion)

The temperature sweep shows ECE **dropping to 0.0113 at T=0.5** (0.125 below raw), which looks like a huge calibration improvement.
**This is an artifact of over-smoothing**, and three metrics expose it at once:

| T | ECE | soft CE | meanConf | Brier |
|---|---|---|---|---|
| 0.50 | **0.0113** | 1.0500 ⬅ worse | 0.8060 | 0.1017 ⬅ worse |
| 1.00 | 0.1368 | 0.8286 | 0.6782 | 0.0491 |
| **1.0718 (published)** | 0.1505 | **0.8231** ⬅ best | 0.6645 | **0.0472** ⬅ best |

T→0 pushes the distribution toward uniform and confidence collapses toward 1/K, so ECE falls spuriously while the proper scoring rules deteriorate in lockstep.

#### Conclusions

1. **The published L1 temperature is correct**: it simultaneously minimizes soft CE and Brier at T≈1.07.
2. **ECE and soft CE run in opposite directions on this checkpoint**. Calibration correctly optimized soft CE/Brier, at the cost of a rising ECE.
3. **This is a metric conflict, not a defect** — but it exposes a real gap in the proposal:

> **Rule 1 only specifies "beat the L1 seed"; it does not specify which metric decides the winner.**
> And per the measurements above, "beating L1" is feasible on soft CE and infeasible on ECE (the published L1 is itself the ECE-worse one).
> **Putting accuracy + ECE into fitness together yields opposite-direction signals**; and temperature-type calibration has an identically zero effect on Top-1.

**Therefore fitness must explicitly declare a primary metric, and that metric must be a proper scoring rule (soft CE or Brier);
ECE may only be reported as a diagnostic quantity and must not enter the selection signal.** Otherwise the P0 family (calibration strategies) is
either identically zero in the selection signal or in conflict with the primary objective.

---

## Built and tested

### 4. `loop/epsilon.py` — supplies the `ε_proxy` / `ε_M` that the proposal leaves undefined

Proposal §3.3 uses these two thresholds for keep/revert, **naming them but never defining them**.
This implementation derives them from two already-measured quantities:

```
ε = z_bonferroni · sqrt( SE_paired² · (1 + replicas) + floor_B² )
```

- **SE scales with the number of CASEs, not the number of items**. Empirically verified: test 400 cases SE=1.067 pp, dev 120 cases SE=2.140 pp;
  the 1/√cases prediction for dev is 1.948 pp (measured is 1.10× that). The independent unit is the case.
- **`(1 + replicas)`** is the parent generation's own variance.
- **Bonferroni** pays in full for every decision the loop will make.
- **Floor B is enforced by default**: it is a property of the **training path**; QLoRA has different noise on a new base model,
  so the previous arm's 0.40 pp cannot be inherited. When it is missing, the implementation **refuses to emit a threshold** (exit 2, verified by negative control).

**dev (H=1000, 95% family): ε = 12.38 pp**

### 5. `loop/state.py` + `loop/test_state.py` — the claim in proposal §3.4 is proven

Proposal §3.4 claims "state is stored in autoresearch.jsonl, and all control state is rebuilt from the log after a crash".
**This is the claim most likely to fail silently** (a state file that is written but never read is exactly like a usable state file until the crash).
So 8 test groups were written, **all passing**:

| | Test | Result |
|---|---|---|
| T1 | Rebuild determinism | ✅ two rebuilds byte-identical |
| T2 | Process killed → **INCOMPLETE, not FAILURE** | ✅ unjudged candidates do not enter the failure archive |
| T3 | Truncation mid-append | ✅ complete records survive, the truncated line takes no number |
| T4 | **Numbering is monotonic after a real restart** | ✅ an independent process read 4 |
| T5 | MAP-Elites is order-independent | ✅ replaying in reverse yields the same archive |
| T6 | Cooldown counted per family, reset on keep | ✅ |
| T7 | Empty log / missing log | ✅ both yield a usable empty state |
| T8 | **Records survive SIGKILL** | ✅ **25 records intact, next_iter=26** |

**The tests caught 3 real bugs** (not environment issues):
1. `next_iter` failed to count RUNNING records → numbering would be reused after a restart
2. `next_iter` was zero-based → the first loop is `iter=0`, which reads as "not run" in every downstream summary
3. `@dataclass` + `exec_module` requires registering `sys.modules` first (otherwise it reports `'NoneType' has no '__dict__'`)

T8 is the critical one: **fsync works**, and six weeks of unattended operation will inevitably hit at least one unclean shutdown.

---

## Pending adjudication (blocks what follows)

### 6. ⚠ The arithmetic of the proposal's 30/80/60 split is infeasible

Under the measured scaling law (5 items/case, the independent unit is the case):

| Split | # items | ≈cases | SE | Raw 95% half-width | Bonferroni H=1000 |
|---|---|---|---|---|---|
| proxy P | 30 | 6 | 8.71 pp | ±17.07 pp | **35.33 pp** |
| **medium M** | 80 | 16 | 5.33 pp | ±10.45 pp | **21.63 pp** |
| frozen V | 60 | 12 | 6.16 pp | ±12.07 pp | 24.98 pp |
| *(control) dev 120 cases* | *600* | *120* | *2.14 pp* | *±4.20 pp* | *12.38 pp* |

**The usable effect space has an upper bound of about 20 pp** (starting point 0.815, ceiling 1).

> The decision threshold for medium M is **21.63 pp > the entire usable space**.
> That is, even if the loop finds a real improvement, at H≈1000 it is **undetectable**; 2000 mutations implies an even larger H and therefore a higher threshold.

| Option | Approach | Cost |
|---|---|---|
| **A (recommended)** | Keep the official 960/120/120+400 | Give up the three-way split; **the only arithmetically adjudicable option** |
| B | Three-way split + cap H at 50 | Threshold drops to ~6.5 pp, but the core selling point of 24/7 continuous search is voided |
| C | Three-way split + pooled replication | Cost ×N, SE drops only by √N, still insufficient |

**Recommend A**, on the grounds that: it has been verified case-by-case on this machine (`verify_split_against_protocol.py`),
and dev's ε=12.38 pp is **smaller** than the 20 pp usable space — it is **adjudicable**.

---

## Progress

| Component | Status |
|---|---|
| Day-1 baseline (L1) | ✅ 0.8150 |
| Day-1 baseline (raw) | ✅ identical by Top-1 definition; calibration metrics pending |
| ε derivation | ✅ `loop/epsilon.py` |
| JSONL state machine | ✅ 8/8 tests (including SIGKILL) |
| Split case-by-case verification | ✅ |
| Gates | ✅ 8/8 PASS |
| QLoRA training loop | ⬜ unverified (proposal §6.1 gate) |
| Floor B on a QLoRA seed | ⬜ **untested**; `epsilon.py` refuses to emit a threshold before this |
| Controller / MAP-Elites | ⬜ state machine ready, controller not yet written |
