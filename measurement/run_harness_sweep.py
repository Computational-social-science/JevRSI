"""run_harness_sweep.py -- first P0 harness batch: fit on calibration, score on dev.

WHAT THIS DOES

The objective's primary route is harness evolution: seconds per cycle instead of 2.11 hours, and no
training variance, so Floor B is zero by construction on this route. This is the first batch. It fits
every stage on the CALIBRATION split and reports the result on DEV, which stays held out throughout.

WHY FIT ON CALIBRATION AND NOT DEV

Fitting on dev and reporting on dev produces a confident wrong number. The calibration split has 600
questions and the dev split has 600; using calibration for fitting leaves dev as a genuine held-out
measurement. The frozen test split is never read.

WHAT IS BEING ATTACKED

The seed's shipped calibration makes the Calibration axis worse -- ECE 0.1368 raw -> 0.1481 after L1 --
because those temperatures were fitted to soft cross-entropy, which is a different objective. The first
stage family here fits monotone maps directly against confidence, which is what ECE actually measures.

THE RESULT IS REPORTED ON FOUR AXES, NEVER COLLAPSED

The objective requires gains to be attributed, and a single composite would hide exactly the regression
this batch is looking for: a stage that improves ECE while quietly wrecking Brier must be visible as
such. So Intelligence and Calibration are printed separately, and Speed/Cost are declared out of scope
for a post-hoc family rather than reported as zero -- a family that cannot affect them must not be
allowed to look like it did.

Run from the agent-jev directory (the evaluator's import paths are relative to it):
    cd ../agent-jev && python measurement/run_harness_sweep.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md. This file sits one level below the
# root, so the root is parents[1]; parents[2] would be the root's parent, and every artifact
# read through it would come from a neighbouring directory.
import os
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

AUTORESEARCH = pathlib.Path(str(ROOT))
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
sys.path.insert(0, str(SUBJECT))
sys.path.insert(0, str(SUBJECT / "scripts"))
sys.path.insert(0, str(AUTORESEARCH))

OUT = AUTORESEARCH / "measurement" / "harness_sweep_01.json"
DEV_LOGITS = AUTORESEARCH / "measurement" / "day1_raw_dev.json"
FEAT = AUTORESEARCH / "measurement" / "features" / "calibration_final.npz"
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
# The base model is an EXTERNAL download, so no path to it is portable. Override with
# JEVRSI_BACKBONE; the default sits beside this repository.
BACKBONE = str(pathlib.Path(os.environ.get(
    "JEVRSI_BACKBONE", ROOT.parent / "jev_repro" / "models" / "Qwen3-0.6B")))
TYPES = ("boolean", "choice", "score")

import torch                                                    # noqa: E402
from harness.harness import (four_axis_fitness, identity_stage, isotonic_confidence,   # noqa: E402
                             make_preds, non_monotone_window, scalar_temperature, softmax)
from evaluate_split import chance_corrected_skill               # noqa: E402


def load_dev():
    """Dev = the held-out decision signal. Logits only; the harness never sees a dev gradient."""
    rows = [json.loads(l) for l in DEV_LOGITS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows, [np.asarray(r["logits"], dtype=np.float64) for r in rows], [r["type"] for r in rows]


def load_calibration():
    """Calibration logits, reconstructed from the CACHED features plus the checkpoint's own head.

    Reading cached features rather than re-running the backbone is the point of the harness route: the
    596M forward is paid once (7.4 min for 5400 questions) and every later stage is seconds. The head
    used here is the checkpoint's own, so the logits are the model's, not a reimplementation's.
    """
    from safetensors.torch import load_file
    from agentjev.model import AgentJevModel
    z = np.load(FEAT, allow_pickle=True)
    feats, off, tg, types = z["features"].astype(np.float32), z["offsets"], z["targets"], z["types"]
    model = AgentJevModel(BACKBONE, dtype=torch.float32)
    model.load_state_dict(load_file(SEED_CKPT, device="cpu"), strict=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()

    logits = []
    with torch.inference_mode():
        for i in range(len(off) - 1):
            k = int(off[i + 1] - off[i])
            x = torch.from_numpy(feats[off[i]:off[i + 1]]).to(dev).unsqueeze(0)
            m = torch.ones(1, k, dtype=torch.bool, device=dev)
            h = model.proj_in(x)
            h = model.set_encoder(h, m)
            lg = model.scorer(x + model.proj_out(h))
            logits.append(lg[0, :k].double().cpu().numpy())
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    rows = [{"type": t, "target": tg[off[i]:off[i + 1]].tolist(), "id": f"cal{i}"}
            for i, t in enumerate(types)]
    return rows, logits, list(types)


def evaluate(name, stage, dev_rows, dev_logits, dev_types, extra=None):
    """Apply a stage and score the four axes. Timing is reported because cycle cost is an axis."""
    t0 = time.perf_counter()
    probs = stage.apply([softmax(z) for z in dev_logits], dev_types)
    preds = make_preds(dev_rows, probs)
    fit = four_axis_fitness(preds, chance_corrected_skill)
    dt = time.perf_counter() - t0
    fit["wall_s"] = round(dt, 4)
    fit["stage"] = stage.describe()
    if extra:
        fit.update(extra)
    print(f"\n--- {name} ---")
    I, C = fit["intelligence"], fit["calibration"]
    print(f"  Intelligence : acc={I['accuracy']:.4f}  skill={I['chance_corrected']:.4f}")
    print(f"  Calibration  : ECE={C['ece']:.4f}  softCE={C['soft_cross_entropy']:.4f}  "
          f"Brier={C['brier_sum']:.4f}  scoreMAE={C['score_expectation_mae']:.4f}")
    print(f"  cycle cost   : {dt*1000:.1f} ms for 600 questions")
    return fit


def main() -> int:
    print("loading dev (held out) ...", flush=True)
    dev_rows, dev_logits, dev_types = load_dev()
    print("loading calibration (fit set) via cached features + the checkpoint's own head ...",
          flush=True)
    cal_rows, cal_logits, cal_types = load_calibration()
    print(f"  dev n={len(dev_rows)}  calibration n={len(cal_rows)}")

    cal_probs = [softmax(z) for z in cal_logits]
    results = {}

    # ---- arm 0: the shipped baseline, as the reference every other arm is read against ----------
    temps = {"boolean": 1.0717734625362931, "choice": 1.0352649238413776, "score": 1.0717734625362931}
    results["shipped_L1"] = evaluate(
        "shipped L1 (reference)", scalar_temperature(temps),
        dev_rows, dev_logits, dev_types,
        extra={"note": "the artefact already in the repository; fitted to soft-CE, not to ECE"})
    results["raw_T1"] = evaluate("raw T=1 (no calibration)", identity_stage(),
                                 dev_rows, dev_logits, dev_types)

    # ---- family 1: monotone maps fitted against CONFIDENCE, which is what ECE measures ---------
    stage_iso = isotonic_confidence([r["target"] for r in cal_rows], cal_probs, cal_types)
    results["isotonic_top1"] = evaluate(
        "isotonic top-1 confidence (fit on calibration)",
        stage_iso, dev_rows, dev_logits, dev_types,
        extra={"fitted_on": "calibration (600 q)", "family": "calibration / monotone"})

    # ---- family 2: the cheapest NON-monotone family, allowed to move argmax -------------------
    # Included to MEASURE whether the near-tie error population is reachable this way, not to assume it.
    for g in (0.25, 0.5, 1.0, 2.0):
        results[f"non_monotone_g{g}"] = evaluate(
            f"non-monotone window gamma={g}", non_monotone_window(g),
            dev_rows, dev_logits, dev_types,
            extra={"family": "intelligence / non-monotone", "gamma": g})

    # ---- summary: the axes are compared separately, on purpose -------------------------------
    base = results["shipped_L1"]
    print(f"\n{'=' * 78}\nSUMMARY vs the shipped L1 baseline (dev, held out)\n{'=' * 78}")
    print(f"  {'arm':26s} {'acc':>8} {'d_acc':>8} {'ECE':>8} {'d_ECE':>8} {'Brier':>8} {'ms':>7}")
    for name, r in results.items():
        I, C = r["intelligence"], r["calibration"]
        dacc = I["accuracy"] - base["intelligence"]["accuracy"]
        dece = C["ece"] - base["calibration"]["ece"]
        print(f"  {name:26s} {I['accuracy']:8.4f} {dacc:+8.4f} {C['ece']:8.4f} {dece:+8.4f} "
              f"{C['brier_sum']:8.4f} {r['wall_s']*1000:7.1f}")

    print("\n  Read the d_ECE column first. The shipped baseline's own ECE is 0.1481 against 0.1368")
    print("  raw, so the repository's only calibration artefact currently makes the Calibration axis")
    print("  worse. A stage that fixes that is a real, cheap, pre-registerable win on a P0 axis.")
    print("  The d_acc column should stay at 0.0000 for every MONOTONE family: argmax is invariant")
    print("  under a monotone per-candidate transform, so any movement there would be a bug, not a gain.")

    OUT.write_text(json.dumps({
        "_what_this_is": "First P0 harness batch. Fitted on calibration, scored on the held-out dev "
                         "split, four axes reported separately and never collapsed.",
        "_why": "The objective's primary route is the harness: seconds per cycle, and no training "
                "variance, so Floor B is zero by construction on this route. The seed's shipped "
                "calibration regresses ECE (0.1368 raw -> 0.1481) because it was fitted to soft-CE, "
                "which is a different objective.",
        "_held_out": "dev was never fitted on. The frozen test split was never read.",
        "_single_source": "all metric numbers come from typed_decisions.experiment.metrics, imported "
                          "from the subject repository, so they cannot drift from the frozen evaluator",
        "recipe": {"fit_split": "calibration", "score_split": "dev",
                   "calibration_logits_source": "cached frozen-backbone features + the checkpoint's own head",
                   "families": ["scalar_temperature (reference)", "isotonic_top1_confidence",
                                "non_monotone_window"]},
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
