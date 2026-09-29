"""harness.py -- the P0 route: evolve the layer around a frozen backbone.

WHY THIS MODULE EXISTS

The objective's technical posture is harness-first: for a 0.6 B scoring model that emits no text, most
early and mid-horizon gains live in calibration, the scoring pipeline and the inference backend -- not
in weights. This module is the calibration family of that route. Every stage here is a deterministic
function of the model's own output, so a cycle costs seconds instead of 2.11 hours, and Floor B is zero
by construction because nothing is trained.

THE MEASURED PROBLEM THIS ATTACKS

On this seed, the one calibration artefact in the repository makes the Calibration axis WORSE:

    ECE   raw 0.1368  ->  shipped L1 0.1481

The shipped temperatures were fitted to soft cross-entropy. Soft CE is a proper scoring rule and it is
already near its ceiling here (the oracle scalar temperature sits within 0.3 of the shipped value, worth
about -0.004 CE). But ECE is a different objective, and optimising one does not optimise the other --
which is exactly what the seed's own numbers show. So the first P0 family fits monotone maps directly
against the calibration objectives, and reports every axis so a regression on one cannot hide behind a
gain on another.

WHAT CAN AND CANNOT MOVE THE INTELLIGENCE AXIS

The evaluator takes `probs.argmax()`. Any monotone per-candidate transform preserves argmax, so the
monotone calibration family cannot move accuracy -- and that is a property of the metric, not a defect
of the implementation. Measured on the seed: 87.4% of errors sit within margin 1.0 of the boundary
(wrong-item median 0.38 against 1.06 for correct items), so those near-ties are reachable -- but only by
a NON-monotone transform, or by something that acts on the features rather than the probabilities.
Those are separate families (`NonMonotoneWindow`, and the feature-space probe in
`measurement/fit_head_offline.py`), and the sweep reports each family separately so the Intelligence
result is never silently attributed to the Calibration work.

HELD-OUT DISCIPLINE

Every stage is fitted on the calibration split and scored on dev. Fitting on dev would make every dev
number a training number. The frozen test split is never read by this module.
"""
from __future__ import annotations

import os
import json
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
TYPES = ("boolean", "choice", "score")


# --------------------------------------------------------------------------------------
# logit plumbing
# --------------------------------------------------------------------------------------
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def top1_confidence(p: np.ndarray) -> float:
    return float(np.max(p))


# --------------------------------------------------------------------------------------
# stages. Each is fitted on calibration, then applied to any split.
# --------------------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    per_type: dict = field(default_factory=dict)     # type -> params

    def apply(self, probs: list, types: list) -> list:
        out = []
        for p, t in zip(probs, types):
            params = self.per_type.get(t) or self.per_type.get("_all")
            out.append(p if params is None else params(p, t))
        return out

    def describe(self) -> dict:
        d = {}
        for t, fn in self.per_type.items():
            d[t] = getattr(fn, "_params", "opaque")
        return d


def identity_stage() -> Stage:
    return Stage("identity", {"_all": lambda p, t: p})


def scalar_temperature(temps: dict) -> Stage:
    """The shipped baseline family: one temperature per question type, applied to the logits.

    Kept as a comparison arm, not as a candidate. It cannot move accuracy -- argmax is invariant under
    a positive scalar temperature -- so its only possible contribution is on the probabilistic metrics.
    """
    def mk(T):
        def f(z, t):                      # takes LOGITS, unlike the others
            return softmax(z, T)
        f._params = {"kind": "scalar_temperature", "T": round(float(T), 4)}
        return f
    return Stage("scalar_temperature", {t: mk(v) for t, v in temps.items()})


def isotonic_confidence(targets: list, probs: list, types: list) -> Stage:
    """Per-type monotone map on the TOP-1 CONFIDENCE, fitted to make confidence equal accuracy.

    This is the map ECE actually wants. ECE is a statement about the top-1 probability versus the
    top-1 hit rate, so fitting a monotone recalibration `p_top -> P(correct | p_top)` directly targets
    it, instead of the proxy that soft-CE fitting optimises. PAVA (pool adjacent violators) is
    implemented here rather than pulled from sklearn so this file has no dependency beyond numpy and
    stays runnable in the clean environment Rule 3 requires.

    Fit on calibration; applied to dev. Non-monotone is impossible by construction, so accuracy is
    untouched by this family -- the gain, if any, is on ECE and on the proper scores.
    """
    def pava(x: np.ndarray, y: np.ndarray, w: np.ndarray):
        """Weighted isotonic regression. Returns the fitted step function's knot/value arrays."""
        order = np.argsort(x, kind="stable")
        xs, ys, ws = x[order], y[order], w[order]
        # pool into blocks carrying (weight, weighted sum, size, x-range midpoint)
        bw, bs, bn, bx = [], [], [], []
        for xi, yi, wi in zip(xs, ys, ws):
            bw.append(wi); bs.append(wi * yi); bn.append(1); bx.append(xi)
            while len(bw) > 1 and bs[-2] / bn[-2] > bs[-1] / bn[-1]:
                w2, s2, n2, x2 = bw.pop(), bs.pop(), bn.pop(), bx.pop()
                bw.append(bw[-1] + w2); bs.append(bs[-1] + s2)
                bn.append(bn[-1] + n2); bx.append((bx[-1] * 1 + x2) / 2)
        vals = np.array([s / n for s, n in zip(bs, bn)])
        edges = np.array(bx)
        return edges, vals

    per = {}
    for t in TYPES:
        sel = [(p, y) for p, y, tt in zip(probs, targets, types) if tt == t]
        if not sel:
            continue
        conf = np.array([top1_confidence(p) for p, _ in sel])
        hit = np.array([float(int(np.argmax(p)) == int(np.argmax(y))) for p, y in sel])
        edges, vals = pava(conf, hit, np.ones_like(conf))
        # Clamp into (eps, 1-eps): a top-1 confidence of exactly 0 or 1 would make soft-CE infinite.
        eps = 1e-6
        vals = np.clip(vals, eps, 1.0 - eps)

        def mk(edges=edges, vals=vals):
            def f(p, _t, _e=edges, _v=vals):
                q = np.array(p, dtype=np.float64)
                c = q.max()
                new_top = float(np.interp(c, _e, _v))
                scale = new_top / c if c > 0 else 1.0
                return np.clip(q * scale, 0.0, 1.0)
            f._params = {"kind": "isotonic_top1", "knots": int(len(edges)),
                         "top1_range": [round(float(vals.min()), 4), round(float(vals.max()), 4)]}
            return f
        per[t] = mk()
    return Stage("isotonic_top1_confidence", per)


def non_monotone_window(gamma: float) -> Stage:
    """Multiply each candidate's probability by a mid-confidence window.

    f(p) = p * (1-p)^gamma, normalised. This is the cheapest family that is allowed to move the
    argmax, because it is not monotone in p. gamma=0 recovers the identity; larger gamma suppresses
    confident candidates relative to uncertain ones. Included precisely so the sweep can MEASURE
    whether the 87.4% of near-tie errors are actually reachable this way, rather than asserting it.
    """
    def mk(g):
        def f(p, _t):
            q = np.asarray(p, dtype=np.float64)
            w = q * np.power(np.clip(1.0 - q, 1e-12, None), g)
            s = w.sum()
            return w / s if s > 0 else q
        f._params = {"kind": "non_monotone_window", "gamma": round(float(g), 3)}
        return f
    return Stage("non_monotone_window", {"_all": mk(gamma)})


# --------------------------------------------------------------------------------------
# fitness: four axes, using the FROZEN evaluator's own metric code
# --------------------------------------------------------------------------------------
def four_axis_fitness(preds: list, chance_corrected_skill) -> dict:
    """Score a prediction set on all four axes of the objective's composite.

    The metric numbers come from `typed_decisions.experiment.metrics`, imported from the subject
    repository, so this can never drift from the frozen evaluator. The axes are then reported
    separately rather than collapsed into one scalar: the objective requires gains to be attributed
    to harness vs parameter routes, and a composite would hide exactly the regression this module
    was written to catch.
    """
    sys.path.insert(0, str(SUBJECT))
    from typed_decisions.experiment import metrics as compute_metrics
    m = compute_metrics(preds)
    return {
        "intelligence": {
            "accuracy": m["accuracy"],
            "chance_corrected": chance_corrected_skill(preds),
        },
        "calibration": {
            "ece": m["ece_10_bins_vs_gold_argmax"],
            "soft_cross_entropy": m["soft_cross_entropy"],
            "brier_sum": m["brier_sum"],
            "score_expectation_mae": m.get("score_expectation_mae"),
        },
        "speed": {"note": "unchanged by a post-hoc transform family; measured separately"},
        "cost": {"note": "unchanged by a post-hoc transform family; measured separately"},
        "_n": m["n"],
    }


def make_preds(rows: list, probs: list) -> list:
    """Prediction dicts in the shape the frozen evaluator expects."""
    return [{"id": r.get("id", str(i)), "type": r["type"], "target": r["target"], "probs": list(map(float, p))}
            for i, (r, p) in enumerate(zip(rows, probs))]
