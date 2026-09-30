# Pre-registration — accept threshold for the OUTER (harness) level

**Frozen 2026-09-30, before any P0 candidate was evaluated under this rule.** The isotonic
feasibility probe described in §3 was run to verify the *machinery*; its result is recorded here and
no threshold was chosen after seeing it.

---

## 1. What this pre-registers, and what it deliberately does not

It pre-registers **one number**: the accept threshold τ used by the outer (harness) level of the
loop, the level the proposal makes primary in §0.4 and §6.1.

It does **not** pre-register a threshold for the inner (QLoRA) level. That one cannot be written
yet, and §5 says why. The two levels have different noise structures and conflating them was the
error that made the previous threshold unusable.

## 2. The accept rule, taken from the ecosystem rather than invented

The controller is `agent-jev/scripts/autoresearch_agent.py`. Its rule, at lines 391–448, is:

```
delta = decision_value - baseline
keep  if delta >= tau
tied  if |delta| < 1e-4      -> reject, "change not simpler"
worse otherwise              -> reject
```

**Single declared decision metric, single threshold, everything else recorded but not gating.** This
pre-registration adopts that rule unchanged. Two reasons, in order of weight:

1. **It is the rule the ecosystem actually runs.** A second rule would mean two accept decisions
   from one loop, and the log would not say which produced which.
2. **It is the rule the data supports.** A feasibility probe on 2026-09-30 (§3) found a candidate
   that improved ECE by 67% and soft cross-entropy by 9.9% while moving accuracy by exactly 0.00 pp
   and worsening Brier by 59%. Under a per-axis rule that candidate looks like a large win on two
   axes; under the ecosystem's rule it is correctly **tied and rejected**, because it did not move the
   metric the rule tests. Inventing a multi-axis gate here would have accepted a configuration whose
   decision metric did not move.

The proposal's multi-objective archive (§4.1, §5.2) is therefore a **reporting** structure, not a
gating one. Pareto cells, per-axis values, flip rate and coverage@risk are recorded for every cycle
and reported at the end; they do not participate in keep/revert. That is how the ecosystem uses them
too — `results.tsv` carries all of them and the accept branch reads one.

## 3. The feasibility probe, recorded so the choice of τ cannot be accused of fitting it

Before any threshold was fixed, one candidate was evaluated end to end on the decision split.

| | identity (reference) | isotonic top-1 | Δ |
|---|---|---|---|
| **accuracy** (decision metric) | 0.7955 | 0.7955 | **+0.00 pp** |
| chance-corrected skill | 0.7042 | 0.7042 | +0.00 pp |
| soft cross-entropy | 0.8421 | 0.7589 | −0.0833 |
| ECE (10 bins) | 0.1240 | 0.0408 | −0.0832 |
| Brier | 0.0522 | 0.0828 | +0.0306 |
| score expectation MAE | 0.1955 | 0.3329 | +0.1374 |

Split: 440 questions over 88 cases (the medium set, carved from dev by whole case). Calibration on
the 600-question calibration split; the decision split is never fitted on.

Three facts follow, and all three are used below.

- **Accuracy is unchanged, exactly.** Isotonic regression is a monotone map on the top-1
  confidence, and a monotone map cannot change an argmax. This is a property of the family, not an
  accident of the fit, and it is why the ecosystem's rule rejects it.
- **A large calibration gain exists and is reachable.** ECE 0.1240 → 0.0408 is a 67% reduction from a
  stage that costs milliseconds. The proposal's claim that calibration is "a central driver" for
  Jev-class models is supported on this seed — on the *calibration axis*. Whether that can be turned
  into decision-metric movement is a separate question, and §4 says what would settle it.
- **Two metrics move the wrong way.** Brier +59%, score MAE +71%. Recorded, reported, not gating.
  A future configuration that wants the ECE gain without those costs is a legitimate candidate; this
  pre-registration neither endorses nor forbids it.

## 4. The measurement τ is built from

`measurement/floor_a_dev_bootstrap.py`, 10 000 replicates, seed 0, resampling **whole cases**.

| quantity | value | what it is |
|---|---|---|
| cases | 88 | the unit of dependence; 5 questions share a case's context |
| questions | 440 | 0 excluded for having <2 candidates |
| observed accuracy | 0.7955 | identity stage on the decision split |
| **cluster SE (absolute)** | **2.1544 pp** | spread of the score when the *cases* change |
| binomial SE | 1.9230 pp | the wrong estimator here; understates by the design effect |
| design effect | 1.2552 | 1.00 would mean questions were independent, which they are not |
| **paired SE (difference)** | **2.0640 pp** | spread of a *difference* between two configurations on the same cases |

**The paired SE sets τ, not the absolute SE.** The accept rule tests a difference between two
configurations evaluated on the same 88 cases; the case-to-case difficulty is common to both and
cancels in the difference. Using the absolute SE would set a threshold for a question nobody asks.

The paired SE was measured against a fixed random per-dimension **vector** temperature (seed 12345,
scale drawn per question in [0.6, 1.8]). The arm matters: a *scalar* temperature is a monotone map
and reports a paired SE of exactly 0.0000, which is a true statement about that arm and a useless
measurement, because a real candidate is not monotone. The vector arm moved the argmax on 53 of 440
questions (12.0%), so the spread it produces is real spread in a difference.

This reproduces the 2.14 pp previously recorded as `dev_split_resolution.acc_se_pp`, to within
rounding, and unlike that number it can be re-run: its source file was absent from the live tree.

## 5. The number

Bonferroni–Horn, over a horizon of **H = 1000 selection decisions** on the decision split:

```
c      = z + sqrt(2 ln H) - 0.5 = 1.95996 + 3.71690 - 0.5 = 5.17686
tau_pp = c x paired SE          = 5.17686 x 2.0640 pp = 10.685 pp
```

| | value |
|---|---|
| **τ (harness / outer level, accuracy, decision split)** | **10.685 pp = 0.10685** |
| usable headroom above the 0.7955 baseline | ≈ 20 pp |
| inner (QLoRA) level | **NOT PRE-REGISTERED** — see below |

**H = 1000** is the proposal's own planning horizon for selection decisions (§4.2 sizes the working
splits for a six-week run). The correction is what makes the threshold survive repeated selection:
without it, 1000 decisions at a nominal 5% level produce roughly 50 false keeps. The previous
threshold in this repository was 4.949 pp, derived with a Floor B that has since been revoked; it is
not comparable and is not used.

**Usable headroom is ≈ 20 pp and τ is 10.685 pp, so roughly half the headroom must be cleared in a
single cycle.** That is strict, and deliberately so: a 0.6 B scoring model with a frozen backbone
has a small ceiling on what post-processing can reach, and a permissive threshold on that ceiling
buys false keeps rather than gains. If the outer level's real candidates turn out to be uniformly
smaller than τ, that is a **finding about the outer level** — it means harness search is exhausted on
this seed and the inner level's floor must be measured — not a reason to lower τ after the fact.

**τ may be revised only by a new pre-registration that states what was measured and why the
measurement changed.** It may not be revised after seeing which candidates pass.

## 6. Why the inner level has no threshold yet

The inner level trains weights. Its noise is not measurement noise; it is **run-to-run dispersion of
a stochastic training procedure**, and that is a property of the training path on this seed.

The 0.40 pp previously in circulation was measured on a retired 322M full-parameter arm with
replicate accuracies of 0.526–0.536, against this seed's dev baseline of 0.7955. Different model,
different scale, different procedure. `measurement/INSTRUMENT_CALIBRATION.json` carries
`_REVOKED_2026-09-30`, and `adaptation/gate_tau.py` refuses to arm any threshold derived from it.

**The outer level needs no such number, and that is the load-bearing observation.** A harness
candidate is a deterministic function of cached logits — measured on 2026-09-30: three independent
evaluations of three different stages produced bit-identical metric fingerprints. Its run-to-run
dispersion is zero by construction, which is exactly why §4 needs only the measurement floor. The
inner level's is not zero, and cannot be assumed to be.

Until `measurement/floor_b_lora_runs/` completes 5 seeds on this seed differing only by seed, the
inner level has no threshold and the loop must not run a QLoRA cycle. The proposal gates that level
behind measured harness saturation anyway (§4.3 priority rule 5), so nothing is lost by waiting.

## 7. What this pre-registration does not claim

- It does not claim any harness stage improves accuracy. The one stage measured here did not.
- It does not claim the calibration gain is available to the decision metric. It is not, for this
  family, and §3 records that as a property rather than a defect.
- It says nothing about the frozen test split. V is not read by any measurement in this document, and
  Rule 1's criterion is untouched by it.
- It does not license any QLoRA cycle. See §6.
