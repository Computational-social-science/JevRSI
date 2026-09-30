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
# paths.py lives at the project root. ROOT is put on sys.path immediately below, so the import that
# reads the subject repository's location must come after that -- hence the two-step form here rather
# than a single import-and-use line.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "loop"))

# The subject repository is an EXTERNAL clone, so no path to it is portable and none can be derived
# from __file__. config/paths.json declares it and paths.py raises rather than falling back silently
# (Rule 1 of docs/PROJECT_RULES.md).
from paths import require, subject                                              # noqa: E402

SUBJECT = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
sys.path.insert(0, str(SUBJECT))
sys.path.insert(0, str(SUBJECT / "scripts"))

from harness.harness import (identity_stage, isotonic_confidence, non_monotone_window,   # noqa: E402
                             make_preds, scalar_temperature, softmax)
from pipeline.routes import ALL_ROUTES                                                    # noqa: E402


def load_logits(path: pathlib.Path):
    """A logits bundle: {rows: [...], types: [...], logits: [[...]]}. Kept as plain JSON so a cycle's
    inputs are inspectable on disk and a rerun is exactly reproducible.

    `qid` and `case` are carried through, not just `id`. The positional `id` is unique only within a
    bundle, so it cannot locate a row in another artefact; the original question id can, and routes
    that read a feature cache instead of this bundle need exactly that to join on. Dropping the field
    here made every such join fail with `None` rather than with a clear error at the join site.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    return ([{"type": r["type"], "target": r["target"], "id": r.get("id", str(i)),
              "qid": r.get("qid"), "case": r.get("case")}
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
    if route == "head.probe_refit":
        # NOT a needs_model route, despite the fallback below. It runs on cached features and is
        # handled by run_head_refit, which is dispatched from main() before build_stage's result is
        # used. Marking it needs_model here would return a needs_model_executor payload and the
        # branch would never be reached -- so the flag below must stay False.
        return None, False
    return None, True          # needs the model


def run_head_refit(a, params, t0, chance_corrected_skill, compute_metrics, make_preds, softmax) -> int:
    """head.probe_refit: refit the 2.37M head on cached features, score on proxy and medium.

    Scored per split, each with its own refit, because the two bundles are different question sets
    drawn from different official splits. Fitting one head on the proxy rows and scoring medium with
    it would make the medium number a function of the proxy's contents, which is the leak the split
    check exists to prevent -- just moved from the ids into the weights.

    Returns the executor's own exit code. It raises rather than returning a partial payload: a cycle
    that silently scored fewer questions than it reported is worse than a cycle that errored.
    """
    from pipeline.head_route import FEATDIR_DEFAULT, feature_path, refit_head_logits

    out = pathlib.Path(a.out)
    tag = params.get("tag", "final")
    # Resolve the feature directory against THIS PROJECT, not against the working directory. The
    # orchestrator runs the executor with cwd = the SUBJECT repository, so a relative default points
    # at a directory that does not exist there and every cycle reports missing_feature_cache. The
    # route had already been proven to work when run by hand from the project root, which is exactly
    # the shape of bug a single manual test cannot catch.
    featdir = pathlib.Path(params["featdir"]) if params.get("featdir") else (ROOT / FEATDIR_DEFAULT)
    # The proposal's fit set is the one thing a stage may be fitted on. train is the only split large
    # enough to fit 2.37M parameters on and is disjoint from every split scored here.
    fit_split = params.get("fit_split", "train")
    epochs = int(params.get("epochs", 30))
    lr = float(params.get("lr", 3e-4))
    seed = int(params.get("seed", 0))

    missing = [s for s in ("train", "dev")
               if not feature_path(s, tag, featdir).exists()]
    if missing:
        out.write_text(json.dumps({
            "route": a.route, "status": "missing_feature_cache",
            "missing": {s: str(feature_path(s, tag, featdir)) for s in missing},
            "note": "this route reads CACHED features and will not run a backbone pass to make them -- "
                    "that pass is the 596M cost this route exists to avoid. Run "
                    "measurement/cache_candidate_features.py --splits train dev first.",
        }, indent=2), encoding="utf-8")
        print(json.dumps({"status": "missing_feature_cache"}))
        return 0

    # proxy and medium are both carved out of dev, so the FEATURES are dev's -- 600 questions, of
    # which proxy is 160 and medium is 440. The refit is what differs per split, not the feature
    # file, so it is refit once and each split then selects its own rows by ORIGINAL question id.
    # Scoring all 600 for both splits would report the same number twice and call it two results.
    import numpy as _np
    r = refit_head_logits(fit_split=fit_split, score_split="dev", tag=tag, featdir=featdir,
                          epochs=epochs, lr=lr, seed=seed)
    dev_ids = [str(q) for q in np.load(feature_path("dev", tag, featdir),
                                        allow_pickle=True)["ids"]]
    dev_pos = {q: i for i, q in enumerate(dev_ids)}
    logits, mask, targets, types = r["logits"], r["mask"], r["targets"], r["types"]

    results, prov = {}, {}
    for name, bundle_path in (("proxy", a.proxy), ("medium", a.medium)):
        b_rows, _, _ = load_logits(pathlib.Path(bundle_path))
        idx, absent = [], []
        for row in b_rows:
            q = str(row.get("qid"))
            if q in dev_pos:
                idx.append(dev_pos[q])
            else:
                absent.append(q)
        if absent:
            # Silently scoring a subset would report a number on fewer questions than the bundle
            # holds, with nothing in the payload to say so.
            raise RuntimeError(
                f"{len(absent)} of {len(b_rows)} {name} questions are absent from the dev feature "
                f"cache (e.g. {absent[:3]}). Re-run cache_candidate_features.py --splits dev, or the "
                f"{name} number would be computed on a silently smaller set.")
        li, mi, ti = logits[idx], mask[idx], targets[idx]
        # types arrives as a plain list from load_features, and `idx` is a list of positions -- so the
        # selection has to go through a comprehension rather than a slice. Indexing the list directly
        # raises "list indices must be integers", which reads as a type bug and is really a missing
        # selection step.
        tsel = [types[i] for i in idx]
        # Reconstruct the row list so the metric code sees the shape it sees for every other route.
        # y[mask] rather than `if mask`: the mask is a per-slot boolean ARRAY, and using it as a
        # scalar condition makes NumPy compare whole arrays and raise on ambiguity.
        rows = [{"type": t, "target": [float(v) for v in yy[mm]], "id": str(q)}
                for q, (t, yy, mm) in enumerate(zip(tsel, ti, mi))]
        # Softmax over the VALID candidates only. A padded row is [.. k real logits .. 0 .. 0], and
        # softmax over the whole row puts probability mass on the padding -- so the returned vector is
        # length Cmax while the target is length k, and the metric code broadcasts them together and
        # raises. Renormalising inside the mask is also the honest thing: a padded slot is not a
        # candidate the model scored, so it must not receive a share of the probability.
        probs = []
        for lg, mm in zip(li, mi):
            v = lg[mm]
            e = np.exp(v - v.max())
            probs.append(e / e.sum())
        keep = [i for i, k in enumerate(mi.sum(1)) if k >= 2]
        preds = make_preds([rows[i] for i in keep], [probs[i] for i in keep])
        m = dict(compute_metrics(preds))
        m["chance_corrected"] = chance_corrected_skill(preds)
        m["n_scored"] = len(keep)
        m["n_bundle"] = len(b_rows)
        results[name] = m

    prov = {k: v for k, v in r.items() if k not in ("logits", "mask", "targets", "types")}

    wall = time.perf_counter() - t0
    payload = {
        "route": a.route, "params": params, "status": "ok",
        "stage": {"kind": "head_refit", "note": "produces logits rather than transforming them; "
                                               "scored with the same metric code as every other route"},
        "proxy": results["proxy"], "medium": results["medium"],
        "wall_s": round(wall, 4), "needs_model": False,
        "provenance": prov,
        "_note": "proxy and medium are both carved from dev, so both are scored on dev features; "
                 "what differs per split is the refit, not the feature file. The head is fitted on "
                 "train only, which is disjoint from both.",
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "medium_ce": results["medium"]["soft_cross_entropy"],
                      "medium_acc": results["medium"]["accuracy"], "wall_s": round(wall, 4)}))
    return 0


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

    # head.probe_refit is dispatched BEFORE build_stage, because build_stage cannot express it: it
    # produces logits rather than transforming them. Letting build_stage claim it first would end the
    # cycle at the not_implemented branch below, so this ordering is load-bearing rather than stylistic.
    if a.route == "head.probe_refit":
        from evaluate_split import chance_corrected_skill
        from typed_decisions.experiment import metrics as compute_metrics
        return run_head_refit(a, params, t0, chance_corrected_skill, compute_metrics,
                              make_preds, softmax)

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

    # ---- head.probe_refit: a different KIND of route -------------------------------------------------
    # It does not transform logits, it produces them, so it cannot go through build_stage. It also
    # cannot read a logits bundle: the medium bundle's rows carry the seed checkpoint's logits, and
    # this route's whole claim is that different logits score differently. Scoring the bundle would
    # measure the stage that was never applied.
    if a.route == "head.probe_refit":
        return run_head_refit(a, params, t0, chance_corrected_skill, compute_metrics,
                              make_preds, softmax)

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
