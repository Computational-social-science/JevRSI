"""executor.py -- run ONE route against the real model and return real numbers.

WHY A SUBPROCESS, ALWAYS

The orchestrator never loads a model. Every cycle goes through this script, which loads, measures, and
exits. That is not tidiness -- it is a measured requirement on this host. A parent holding a
GPU-resident model while spawning a child squeezes the child into an allocator stall: the child sat at
step 1 for 632 s with the GPU at 100% and a frozen log mtime. Worse, the contention made a healthy
configuration look 9x slower, so the slowdown was recorded as evidence about the configuration. A
cycle that reports a number must be the only thing touching the card.

THE POST-HOC ROUTES COST NO GPU AT ALL

The calibration family operates on logits the model has already produced. So those routes are pure
NumPy against cached logits: no model, no VRAM, no CUDA context, milliseconds per cycle. That is the
harness-first posture made concrete rather than asserted -- and it is why the outer level can afford
hundreds of cycles while a single inner cycle is still 2.11 hours.

Routes that genuinely need the model (pipeline and backend families) load it here, in this process,
and are correspondingly marked `needs_model` in the plan.

THE SINGLE SOURCE OF EVERY NUMBER

All metrics come from `typed_decisions.experiment.metrics`, imported from the subject repository, which
is the same code the frozen evaluator runs. A second implementation of "accuracy" would drift, and a
drifting metric is the one failure this whole project cannot detect from the inside.

Usage (as the orchestrator calls it):
    python pipeline/executor.py --route cal.isotonic_top1 --params '{"gamma":0.5}' \
        --proxy proxy_logits.json --medium medium_logits.json --out metrics.json
"""
from __future__ import annotations

import os
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SUBJECT))
sys.path.insert(0, str(SUBJECT / "scripts"))

from harness.harness import (identity_stage, isotonic_confidence, non_monotone_window,   # noqa: E402
                             make_preds, scalar_temperature, softmax)
from pipeline.routes import ALL_ROUTES                                                    # noqa: E402


def load_logits(path: pathlib.Path):
    """A logits bundle: {rows: [...], types: [...], logits: [[...]]}. Kept as plain JSON so a cycle's
    inputs are inspectable on disk and a rerun is exactly reproducible."""
    d = json.loads(path.read_text(encoding="utf-8"))
    return ([{"type": r["type"], "target": r["target"], "id": r.get("id", str(i))}
             for i, r in enumerate(d["rows"])],
            [np.asarray(z, dtype=np.float64) for z in d["logits"]],
            list(d["types"]))


def build_stage(route: str, params: dict, cal_rows, cal_logits, cal_types):
    """Return the harness stage for a route, fitted on the CALIBRATION bundle only.

    Fitting happens here, inside the cycle, rather than being hoisted into a global: a stage fitted
    once and reused across cycles would silently share information between candidates, and the whole
    point of the accept rule is that each candidate is judged on its own.
    """
    if route == "cal.scalar_temperature":
        temps = params.get("temperatures") or {
            "boolean": 1.0717734625362931, "choice": 1.0352649238413776, "score": 1.0717734625362931}
        return scalar_temperature(temps), False
    if route == "cal.isotonic_top1":
        return isotonic_confidence([r["target"] for r in cal_rows],
                                   [softmax(z) for z in cal_logits], cal_types), False
    if route == "cal.non_monotone_window":
        return non_monotone_window(float(params.get("gamma", 0.5))), False
    if route in ("cal.platt_per_type", "cal.vector_temperature"):
        # Not yet implemented. Returning None makes the cycle record itself honestly rather than
        # silently falling back to something that would produce a number nobody asked for.
        return None, False
    return None, True          # needs the model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--proxy", required=True, help="proxy-set logits bundle (filtering)")
    ap.add_argument("--medium", required=True, help="medium-set logits bundle (the decision signal)")
    ap.add_argument("--calibration", required=True, help="fitting bundle; never scored")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    params = json.loads(a.params)
    if a.route not in ALL_ROUTES:
        print(f"[FAIL] unknown route {a.route}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    cal_rows, cal_logits, cal_types = load_logits(pathlib.Path(a.calibration))
    stage, needs_model = build_stage(a.route, params, cal_rows, cal_logits, cal_types)

    if stage is None and not needs_model:
        out = pathlib.Path(a.out)
        out.write_text(json.dumps({
            "route": a.route, "status": "not_implemented",
            "note": "declared route with no implementation yet. Recorded rather than approximated, "
                    "because a fallback that returns a plausible number would enter the log as "
                    "evidence about an idea nobody actually tested."}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "not_implemented"}))
        return 0

    if needs_model:
        out = pathlib.Path(a.out)
        out.write_text(json.dumps({
            "route": a.route, "status": "needs_model_executor",
            "note": "this route changes the forward pass, so it cannot be evaluated on cached logits. "
                    "It needs its own executor that runs the model under the mutated pipeline. Not "
                    "yet wired."}, indent=2), encoding="utf-8")
        print(json.dumps({"status": "needs_model_executor"}))
        return 0

    from evaluate_split import chance_corrected_skill
    from typed_decisions.experiment import metrics as compute_metrics

    def score(bundle_path):
        rows, logits, types = load_logits(pathlib.Path(bundle_path))
        if a.route == "cal.scalar_temperature":
            # temperature acts on LOGITS, everything else acts on probabilities
            temps = params.get("temperatures") or {
                "boolean": 1.0717734625362931, "choice": 1.0352649238413776, "score": 1.0717734625362931}
            probs = [softmax(z, temps.get(t, 1.0)) for z, t in zip(logits, types)]
        else:
            probs = stage.apply([softmax(z) for z in logits], types)
        preds = make_preds(rows, probs)
        m = dict(compute_metrics(preds))
        m["chance_corrected"] = chance_corrected_skill(preds)
        return m

    proxy = score(a.proxy)
    medium = score(a.medium)
    wall = time.perf_counter() - t0

    payload = {
        "route": a.route, "params": params, "status": "ok",
        "stage": stage.describe(),
        "proxy": proxy, "medium": medium,
        "wall_s": round(wall, 4),
        "needs_model": False,
        "_note": "proxy is the filter, medium is the decision signal. The frozen test split is not "
                 "read by this executor under any route.",
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps({"status": "ok", "proxy_ce": proxy["soft_cross_entropy"],
                      "medium_ce": medium["soft_cross_entropy"],
                      "medium_acc": medium["accuracy"], "wall_s": round(wall, 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
