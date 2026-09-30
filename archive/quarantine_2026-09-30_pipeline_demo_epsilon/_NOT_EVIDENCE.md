# NOT EVIDENCE -- demo-epsilon run, 2026-09-30

Quarantined, not deleted, and must not be cited.

## Why

1. **The threshold was a demo value, not a pre-registration.** The runner used `epsilon = 0.005`
   (0.50 pp) and printed so itself: `DEMO VALUE, not a derived noise threshold`. The pre-registered
   value was 4.949 pp, and is now REVOKED anyway -- its Floor B was measured on the retired
   laya-multilingual 322M substrate, not on this seed. See
   measurement/INSTRUMENT_CALIBRATION.json -> _REVOKED_2026-09-30.

2. **The single `keep` had no comparison.** cycle_cal_isotonic_top1 was kept with `delta = null`,
   reason "first candidate; nothing to compare against". The first candidate is accepted by
   construction, so that row records that the loop started, not that the mutation is good.

3. **The controller is superseded.** pipeline/ predates the decision to drive agent-jev's
   scripts/autoresearch_agent.py, which has a pre-registered threshold, a dev/shadow two-signal
   scheme, crash classification with a circuit breaker, and trajectory persistence.

## What remains usable

The per-cycle JSONs are real numbers on the case-disjoint proxy/medium split, and they are the
evidence that head.probe_refit runs end to end in ~18 s against a 7596 s LoRA replicate -- a COST
measurement, not a result. The accuracies (medium 0.7886 refit vs 0.7955 shipped) are single-draw
with no seed dispersion, and the refit head was fit on train and scored on dev, so the comparison
against the shipped head's calibration-set number is not like-for-like.
