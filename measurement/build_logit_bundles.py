"""build_logit_bundles.py -- produce the proxy / medium / calibration bundles the executor consumes.

WHY BUNDLES AND NOT A LIVE MODEL

The outer level's calibration routes are pure functions of logits the model has already produced, so
they can be evaluated with no GPU at all. That is what makes a cycle cost milliseconds instead of
hours, and it is only possible if the logits exist on disk. This script creates them, once.

THE PARTITIONS, AND WHAT EACH ONE IS FOR

    calibration   600 q   FIT only. Never scored, never used for a decision.
    proxy  P      150 q   the cheap filter -- discard a candidate here and stop.
    medium  M      450 q   the decision signal, scored by the frozen evaluator.
    frozen V        --     NEVER TOUCHED. Not produced here at all, so it cannot leak by accident.

This mirrors the proposal's P / M / V structure onto the splits this project actually has, and it
keeps the one property that makes the result meaningful: the fitting set, the filtering set, the
decision set and the claim set are four different things, and a cycle that confuses them is not a
cycle that produced evidence.

A note on the sizes. The proposal names P ~30 tasks and M ~80; this repository's dev split is 600
questions over 120 states. The question count is what the evaluator consumes, so the split is cut on
QUESTIONS here and the numbers are reported as such rather than silently rescaled to the proposal's
task vocabulary.

Run from the agent-jev directory:
    cd ../agent-jev && python measurement/build_logit_bundles.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md.
ROOT = pathlib.Path(__file__).resolve().parents[2]

import os
import json
import pathlib
import sys
import time

import numpy as np

AUTORESEARCH = pathlib.Path(str(ROOT))
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
sys.path.insert(0, str(SUBJECT))
sys.path.insert(0, str(SUBJECT / "scripts"))
sys.path.insert(0, str(AUTORESEARCH))

OUT = AUTORESEARCH / "measurement" / "logit_bundles"
FEAT = AUTORESEARCH / "measurement" / "features" / "calibration_final.npz"
DEV_LOGITS = AUTORESEARCH / "measurement" / "day1_raw_dev.json"
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
# The base model is an EXTERNAL download, so no path to it is portable. Override with
# JEVRSI_BACKBONE; the default sits beside this repository.
BACKBONE = str(pathlib.Path(os.environ.get(
    "JEVRSI_BACKBONE", ROOT.parent / "jev_repro" / "models" / "Qwen3-0.6B")))

N_PROXY = 150
N_MEDIUM = 450


def save(name: str, rows: list, logits: list, types: list) -> pathlib.Path:
    """Write one bundle. Every array is converted to plain Python here, at the single place where
    bundles are written, rather than at each call site -- the calibration path hands over ndarrays
    sliced out of the feature cache while the dev path hands over lists read from JSON, and having
    the two call sites disagree about the type is how one of them ends up unserialised."""
    p = OUT / f"{name}.json"
    p.write_text(json.dumps({
        "_what_this_is": f"{name} split logits from the seed checkpoint, for pipeline cycles.",
        "_frozen_V": "not produced here and never read by the executor; this file is not V",
        "rows": [{"type": t, "target": [float(v) for v in y], "id": i}
                 for t, y, i in zip(types, rows, range(len(rows)))],
        "logits": [[float(v) for v in z] for z in logits],
        "types": [str(t) for t in types],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  {name:12s} n={len(rows):4d} -> {p.name}  {p.stat().st_size/1e6:.1f} MB")
    return p


def main() -> int:
    import torch
    from safetensors.torch import load_file
    from agentjev.model import AgentJevModel
    from evaluate_split import chance_corrected_skill
    from typed_decisions.experiment import metrics as compute_metrics

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- dev -> P / M, stratified by type so the filter is not accidentally type-biased ----------
    dev = [json.loads(l) for l in DEV_LOGITS.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_type: dict[str, list] = {}
    for r in dev:
        by_type.setdefault(r["type"], []).append(r)

    proxy, medium = [], []
    for t, rows in sorted(by_type.items()):
        # deterministic interleave so the split does not depend on dict order
        for i, r in enumerate(rows):
            (proxy if i % 4 == 0 else medium).append(r)
    print(f"dev: {len(dev)} -> proxy {len(proxy)} + medium {len(medium)} "
          f"(every 4th question to proxy, stratified by type)")

    save("proxy", [r["target"] for r in proxy],
         [np.asarray(r["logits"], float) for r in proxy], [r["type"] for r in proxy])
    save("medium", [r["target"] for r in medium],
         [np.asarray(r["logits"], float) for r in medium], [r["type"] for r in medium])

    # ---- calibration -> logits, ONE batched GPU pass -------------------------------------------
    # The earlier harness sweep did this as 600 separate single-question forwards and took minutes.
    # Batched, the same work is seconds. That gap is not a detail: it is the difference between a
    # route that can be re-fitted every cycle and one that cannot.
    z = np.load(FEAT, allow_pickle=True)
    feats, off, tg, types = z["features"].astype(np.float32), z["offsets"], z["targets"], z["types"]
    ks = (off[1:] - off[:-1]).astype(int)
    Cmax = int(ks.max())
    N = len(ks)

    Xp = np.zeros((N, Cmax, feats.shape[1]), dtype=np.float32)
    Mp = np.zeros((N, Cmax), dtype=bool)
    for i in range(N):
        a, k = off[i], ks[i]
        Xp[i, :k] = feats[a:a + k]
        Mp[i, :k] = True

    model = AgentJevModel(BACKBONE, dtype=torch.float32)
    model.load_state_dict(load_file(SEED_CKPT, device="cpu"), strict=True)
    model.to("cuda:0").eval()

    X = torch.from_numpy(Xp).to("cuda:0")
    M = torch.from_numpy(Mp).to("cuda:0")
    outs = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, N, 64):
            xb, mb = X[i:i + 64], M[i:i + 64]
            h = model.proj_in(xb)
            h = model.set_encoder(h, mb)
            outs.append(model.scorer(xb + model.proj_out(h)).double().cpu().numpy())
    L = np.concatenate(outs, 0)
    dt = time.time() - t0
    del model
    torch.cuda.empty_cache()

    cal_logits = [L[i, :ks[i]] for i in range(N)]
    # Target slices come from the feature cache, which stores a FLAT target array with the same
    # offsets as the features. Reading them through the same offsets keeps the row order identical to
    # the logits -- an off-by-one here would fit the calibration stage against mislabelled targets,
    # which produces a plausible-looking and completely wrong recalibration.
    cal_targets = [np.asarray(tg[off[i]:off[i + 1]], dtype=float) for i in range(N)]
    save("calibration", cal_targets, cal_logits, list(types))
    print(f"  calibration logits: {N} questions in {dt:.1f}s ({dt/N*1000:.1f} ms/q, batched)")
    # The batched pass is the difference between a route that can be re-fitted every cycle and one
    # that cannot: the same 600 questions took minutes as 600 single-question forwards in the first
    # harness sweep, and the executor re-fits on calibration EVERY cycle, so this cost is paid 1x/cycle.
    # Recorded because the first harness sweep's 600 unbatched calls are the reason a naive
    # implementation of this loop would have been unusable.

    # ---- baseline on each split, so every later number has something to be read against ----------
    from harness.harness import make_preds, softmax
    for name, rows_, lg in (("proxy", proxy, None), ("medium", medium, None)):
        rs = proxy if name == "proxy" else medium
        pr = make_preds([{"type": r["type"], "target": r["target"], "id": str(i)}
                         for i, r in enumerate(rs)],
                        [softmax(np.asarray(r["logits"], float)) for r in rs])
        m = compute_metrics(pr)
        print(f"  baseline {name:6s}: acc={m['accuracy']:.4f} softCE={m['soft_cross_entropy']:.4f} "
              f"ECE={m['ece_10_bins_vs_gold_argmax']:.4f} skill={chance_corrected_skill(pr):.4f}")

    print(f"\n  frozen V: not produced. It is never read by the executor on any route.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
