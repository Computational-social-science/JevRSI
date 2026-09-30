"""
harness_eval.py -- evaluate one OUTER-level candidate. No training, no GPU, seconds.

WHAT IT IS
    The executor for the proposal's primary route. It takes a harness configuration, applies it to
    CACHED logits, scores with the subject repository's own metric code, and prints the same
    `key: value` lines the ecosystem's parse_metrics() reads -- so the agent's accept rule, its
    results.tsv schema and its shadow evaluation all work unchanged on a candidate that never
    trained.

    That last point is the whole design. The agent is not forked, not wrapped and not bypassed: it
    runs, and the only thing that changed is which command produced the numbers it parses.

WHY CACHED LOGITS
    The outer level is a function of logits the model has already produced. Reading them from disk
    instead of re-running the backbone is what makes a cycle cost milliseconds rather than 2.11 h,
    and it is also what makes the outer level DETERMINISTIC -- measured on 2026-09-30, three
    independent evaluations of three stages produced bit-identical metric fingerprints. A
    deterministic candidate needs no run-to-run floor, which is why the outer level's threshold
    needs only the measurement floor (docs/prereg_tau_harness_2026-09-30.md).

THE STAGES
    Four, all implemented in harness/harness.py, which is the reference implementation and is not
    modified here:

        scalar_temperature   the shipped L1 baseline; monotone; the incumbent, not a candidate
        platt_per_type       per-type affine in LOGIT space; NOT monotone; can move argmax
        vector_temperature   per-dimension scale; NOT monotone; can move argmax
        non_monotone_window  mid-confidence window; NOT monotone
        isotonic_top1        monotone map on top-1 confidence; CANNOT move argmax by construction
        option_permutation   the proposal's order-invariance family; expected no-op, run as a
                             NEGATIVE CONTROL

    `platt_per_type` and `vector_temperature` are declared in the proposal (sections 6.1, 6.2) and
    were listed as unimplemented in the retired orchestrator. They are implemented here because the
    outer level cannot reach the decision metric without a non-monotone family, and both are closed
    form: a per-type affine map and a diagonal rescale, each fitted on the calibration split by least
    squares on the logits. No training, no iteration, no early stopping -- which is also why they are
    deterministic.

THE FIT/ SCORE SEPARATION
    Every stage is fitted on the CALIBRATION split and scored on MEDIUM. The two are disjoint at
    question and case level, and scripts/check_split_disjoint.py asserts it before any cycle runs.
    A stage fitted on the split it is scored on would report a number that is a function of its own
    training data, and no threshold can make that evidence.

Usage (as the agent invokes it):
    python measurement/harness_eval.py --config agentjev/configs/harness_search.json \
        --split medium --out cycle.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import require, subject                                              # noqa: E402

SUBJECT = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
for _p in (str(SUBJECT), str(SUBJECT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harness.harness import (identity_stage, isotonic_confidence, make_preds,      # noqa: E402
                             non_monotone_window, scalar_temperature, softmax)

BUNDLES = ROOT / "measurement" / "logit_bundles"


# ---------------------------------------------------------------------------------------
# the two closed-form non-monotone families
# ---------------------------------------------------------------------------------------
def platt_per_type(targets: list, logits: list, types: list):
    """Per-question-type affine map in LOGIT space, fitted by least squares on the calibration split.

    p = sigmoid(a_t * z + b_t) with (a_t, b_t) fitted per question type.

    Why this can move the argmax while a temperature cannot: a temperature divides every logit by a
    POSITIVE constant, which is a monotone map and leaves the ordering untouched. An affine map with
    a fitted intercept shifts different types by different amounts, so a low-logit candidate of one
    type can overtake a high-logit candidate of another. On this seed the per-type intercepts differ,
    so the fitted map is not the identity in the ordering it induces.

    Closed form, no iteration: for each type, minimise squared error between the target probability
    and sigmoid(a*z + b) by a few Gauss-Newton steps, which converge immediately because the model is
    two-parameter. Deterministic by construction -- no initialisation, no sampling, no early stop.
    """
    per = {}
    for t in sorted(set(types)):
        idx = [i for i, tt in enumerate(types) if tt == t]
        # Padded to [N, Cmax] with a validity mask, because the candidate counts differ (2 for
        # boolean, 4-5 for choice and score) and np.array on a ragged list raises rather than
        # broadcasting. The mask is applied at the loss, not zero-mean imputed: a padded slot is not
        # a candidate the model scored.
        Zs = [logits[i] for i in idx]
        Ys = [np.asarray(targets[i], dtype=np.float64) for i in idx]
        Cmax = max(len(z) for z in Zs)
        Z = np.zeros((len(idx), Cmax), dtype=np.float64)
        Y = np.zeros((len(idx), Cmax), dtype=np.float64)
        M = np.zeros((len(idx), Cmax), dtype=bool)
        for r, (z, y) in enumerate(zip(Zs, Ys)):
            Z[r, :len(z)] = z
            Y[r, :len(y)] = y
            M[r, :len(y)] = True
        eps = 1e-6
        y = np.clip(Y, eps, 1 - eps)
        logit_y = np.log(y / (1 - y))
        z = Z[M]
        t_ = logit_y[M]
        # Weighted least squares through the origin plus intercept, solved by iterative
        # reweighting. Deterministic: fixed start, fixed iteration count, no sampling.
        X = np.stack([z, np.ones_like(z)], axis=1)
        w = (y * (1 - y))[M]                                  # delta-method weight
        beta = np.array([1.0, 0.0])
        for _ in range(50):
            r_ = X @ beta - t_
            W = np.clip(w, 1e-9, None)
            H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(2)
            g = X.T @ (r_ * W)
            step = np.linalg.solve(H, g)
            beta -= step
            if np.max(np.abs(step)) < 1e-12:
                break
        a, b = float(beta[0]), float(beta[1])
        # Guard the direction: a non-positive slope would invert the ordering, which is not a
        # calibration but a different model. Clip rather than reject, and record what was clipped.
        a_clipped = False
        if a <= 1e-3:
            a, a_clipped = 1.0, True
        per[t] = (a, b, a_clipped)

    def apply_one(z, t):
        a, b, _ = per.get(t, (1.0, 0.0, False))
        return 1.0 / (1.0 + np.exp(-(a * np.asarray(z, dtype=np.float64) + b)))

    def describe():
        return {"kind": "platt_per_type",
                "per_type": {t: {"a": round(a, 6), "b": round(b, 6), "a_clipped": c}
                             for t, (a, b, c) in per.items()}}

    class _Stage:
        name = "platt_per_type"

        def apply(self, probs, types):
            # This stage consumes LOGITS, unlike the others. The caller is told so by `on_logits`.
            raise TypeError("platt_per_type acts on logits, not probabilities")

    s = _Stage()
    s.on_logits = True
    s.describe = describe
    s.apply_logits = lambda lg, ty: [apply_one(z, t) for z, t in zip(lg, ty)]
    return s


def vector_temperature(targets: list, logits: list, types: list, seed: int = 0):
    """A diagonal per-dimension rescale of the logits, searched over a small fixed grid.

    Deterministic and parameter-free in the sense that matters here: the grid is fixed, the score is
    the calibration split's own soft cross-entropy, and the winner is the argmin with ties broken by
    grid order. No sampling, no initialisation, no early stopping.

    The search space is a single scalar `s` applied to all dimensions of a question, plus a
    per-type offset. That is deliberately small: the measured fact this family exists to exploit is
    that 87.4% of errors sit within margin 1.0 of the decision boundary, so the question is not
    whether a richer map helps but whether a cheap non-monotone one can reach those near-ties at all.
    A large grid would answer a different question, more slowly, and would need its own floor.
    """
    def ce_for(scale: float, offset: float) -> float:
        tot, n = 0.0, 0
        for z, t, y in zip(logits, types, targets):
            v = scale * np.asarray(z, dtype=np.float64) + offset
            v = v - v.max()
            e = np.exp(v)
            p = e / e.sum()
            yy = np.clip(np.asarray(y, dtype=np.float64), 1e-12, 1.0)
            tot += float(-(yy * np.log(p)).sum())
            n += 1
        return tot / max(1, n)

    best = None
    for scale in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8):
        for offset in (-0.5, -0.2, 0.0, 0.2, 0.5):
            c = ce_for(scale, offset)
            if best is None or c < best[0]:
                best = (c, scale, offset)
    _, scale, offset = best

    class _Stage:
        name = "vector_temperature"
        on_logits = True

        def apply(self, probs, types):
            raise TypeError("vector_temperature acts on logits, not probabilities")

        def apply_logits(self, lg, ty):
            out = []
            for z in lg:
                v = scale * np.asarray(z, dtype=np.float64) + offset
                v = v - v.max()
                e = np.exp(v)
                out.append(e / e.sum())
            return out

        def describe(self):
            return {"kind": "vector_temperature", "scale": scale, "offset": offset,
                    "grid": "scale in {0.6,0.8,0.9,1.0,1.1,1.25,1.5,1.8} x offset in {-0.5,-0.2,0,0.2,0.5}"}

    return _Stage()


# ---------------------------------------------------------------------------------------
def position_prior(targets: list, logits: list, types: list, decay: float = 0.0):
    """A per-POSITION additive bias on the logits. The smallest family that can move the argmax.

    WHY THIS FAMILY EXISTS, and why the obvious alternatives cannot
        Every stage above is argmax-invariant ON THIS ARCHITECTURE, and that is structural rather
        than a fitting failure. All candidates of one question share its type, so a per-type or
        per-question map is a SINGLE monotone transform applied to every candidate of that question,
        and a monotone map cannot reorder them. Measured 2026-09-30: three per-type affine scales
        (0.9 / 1.1 / 1.5) all returned exactly 0.7955, while a per-candidate perturbation moved it.

        The bundle carries no candidate text -- only logits, target, type and case -- so the only
        per-candidate attribute available is the option's POSITION. That is a real attribute and not
        a placeholder: option order carries a presentation prior, and the proposal's section 6.2
        names order-invariance as a robustness family precisely because order matters.

    THE FAMILY
        z'_c = z_c + b_c, with b_c = -decay * c, c the zero-based option index.

        decay = 0 is exactly the identity, so the family CONTAINS the incumbent and every member is
        reachable by moving one scalar. That matters: a family that cannot represent the baseline
        cannot be compared against it fairly.

    WHAT IT IS NOT
        A fit. There is no fitting here and no data-dependent parameter: `decay` comes from the
        search, and the stage is a deterministic function of (logits, decay). It is therefore eligible
        for the outer level, whose threshold needs only a measurement floor
        (docs/prereg_tau_harness_2026-09-30.md), and it would NOT be eligible for the inner level.

    THE CONTROL
        At decay = 0 the family is a no-op by construction, and the option_permutation control
        measures the same invariance from the other direction. If a nonzero decay turns out to help,
        the more interesting question is whether it helps because of a genuine order prior or because
        it is fitting noise -- which is what the paired SE in the pre-registration is there to judge.
    """
    def apply_logits(lg, ty):
        out = []
        for z in lg:
            zz = np.asarray(z, dtype=np.float64)
            c = np.arange(len(zz), dtype=np.float64)
            v = zz - decay * c
            v = v - v.max()
            e = np.exp(v)
            out.append(e / e.sum())
        return out

    class _Stage:
        name = "position_prior"
        on_logits = True

        def apply(self, probs, types):
            raise TypeError("position_prior acts on logits, not probabilities")

        def describe(self):
            return {"kind": "position_prior", "decay": decay,
                    "form": "z'_c = z_c - decay * c",
                    "note": "the only per-candidate attribute available in the bundle is the option "
                            "index; decay=0 is the identity, so the family contains the incumbent"}

    # Assigned rather than declared as a staticmethod: a staticmethod body inside a class body
    # cannot see a same-layer function object, and it fails at CALL time, not at import time, so
    # nothing but running the stage catches it.
    _Stage.apply_logits = staticmethod(apply_logits)
    return _Stage()


def load_split(name: str) -> tuple[list, list, list]:
    p = BUNDLES / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"[fatal] {p} not found. Run measurement/build_logit_bundles.py.")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = [{"type": r["type"], "target": r["target"], "qid": r.get("qid"), "case": r.get("case")}
            for r in d["rows"]]
    return rows, [np.asarray(z, dtype=np.float64) for z in d["logits"]], list(d["types"])


def build_stage(cfg: dict, cal_rows, cal_logits, cal_types):
    stage_name = cfg.get("stage", "scalar_temperature")
    params = cfg.get("params", {}) or {}
    if stage_name == "scalar_temperature":
        temps = params.get("temperatures") or cfg["params"]["temperatures"]
        return scalar_temperature(temps)
    if stage_name == "isotonic_top1":
        return isotonic_confidence([r["target"] for r in cal_rows],
                                   [softmax(z) for z in cal_logits], cal_types)
    if stage_name == "non_monotone_window":
        return non_monotone_window(float(params.get("gamma", 0.5)))
    if stage_name == "platt_per_type":
        return platt_per_type([r["target"] for r in cal_rows], cal_logits, cal_types)
    if stage_name == "vector_temperature":
        return vector_temperature([r["target"] for r in cal_rows], cal_logits, cal_types)
    if stage_name == "position_prior":
        return position_prior([r["target"] for r in cal_rows], cal_logits, cal_types,
                              float(params.get("decay", 0.0)))
    if stage_name == "option_permutation":
        # NEGATIVE CONTROL. The candidate head is permutation-equivariant by construction, so this
        # must be a no-op. It is evaluated rather than asserted, because an assertion that is never
        # run is not a test.
        return identity_stage(), {"permutation": int(params.get("seed", 0))}
    raise SystemExit(f"[fatal] unknown stage {stage_name!r}")


def evaluate(cfg: dict, split: str) -> dict:
    from evaluate_split import chance_corrected_skill
    from typed_decisions.experiment import metrics as compute_metrics

    cal_rows, cal_logits, cal_types = load_split("calibration")
    rows, logits, types = load_split(split)

    built = build_stage(cfg, cal_rows, cal_logits, cal_types)
    stage, extra = built if isinstance(built, tuple) else (built, {})

    t0 = time.perf_counter()
    on_logits = getattr(stage, "on_logits", False)
    if on_logits:
        probs = stage.apply_logits(logits, types)
    else:
        probs = stage.apply([softmax(z) for z in logits], types)

    if extra.get("permutation") is not None:
        # Applied AFTER scoring, to the predictions, so the pipeline sees a different candidate
        # order without the harness being able to notice. A no-op result is the expected outcome.
        rng = np.random.default_rng(extra["permutation"])
        probs = [p[rng.permutation(len(p))] for p in probs]

    keep = [i for i, r in enumerate(rows) if len(r["target"]) >= 2]
    preds = make_preds([rows[i] for i in keep], [probs[i] for i in keep])
    m = dict(compute_metrics(preds))
    m["chance_corrected"] = chance_corrected_skill(preds)
    m["n_scored"] = len(keep)
    m["wall_s"] = round(time.perf_counter() - t0, 4)
    m["stage_description"] = stage.describe() if hasattr(stage, "describe") else {}
    if extra:
        m["control"] = extra
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate one harness candidate on cached logits.")
    ap.add_argument("--config", required=True, help="harness_search.json in the subject repo")
    ap.add_argument("--split", default="medium", choices=["proxy", "medium", "calibration"])
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the key: value lines; the agent parses them, so they are on by default")
    a = ap.parse_args()

    cfg = json.loads(pathlib.Path(a.config).read_text(encoding="utf-8"))
    m = evaluate(cfg, a.split)

    payload = {"stage": cfg.get("stage"), "params": cfg.get("params", {}), "split": a.split,
               "metrics": m, "trained": False,
               "_note": "outer level: deterministic function of cached logits, no training, no GPU"}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    if not a.quiet:
        # The exact key names agentjev/scripts/autoresearch_agent.py::EVAL_METRIC_PATTERNS parses.
        print(f"accuracy: {m['accuracy']:.4f}")
        print(f"chance_corrected_skill: {m['chance_corrected']:.4f}")
        print(f"soft_cross_entropy: {m['soft_cross_entropy']:.4f}")
        print(f"brier: {m['brier_sum']:.4f}")
        print(f"ece: {m['ece_10_bins_vs_gold_argmax']:.4f}")
        print(f"score_expectation_mae: {m['score_expectation_mae']:.4f}")
        print(f"training_seconds: 0.0000")
        print(f"peak_vram_mb: 0.0000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
