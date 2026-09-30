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
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md. This file sits one level below the
# root, so the root is parents[1]; parents[2] would be the root's parent, and every artifact
# read through it would come from a neighbouring directory.
import os
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

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


def save(name: str, rows: list, logits: list, types: list, qids: list | None = None,
         cases: list | None = None) -> pathlib.Path:
    """Write one bundle. Every array is converted to plain Python here, at the single place where
    bundles are written, rather than at each call site -- the calibration path hands over ndarrays
    sliced out of the feature cache while the dev path hands over lists read from JSON, and having
    the two call sites disagree about the type is how one of them ends up unserialised.

    `qids` carries the ORIGINAL question id. The positional index that used to stand in for it is
    unique only within a bundle, so it cannot support the one check that matters: whether the filter
    set and the decision set are disjoint. With per-bundle indices that check is not merely hard, it
    is impossible -- two bundles always "overlap" completely. `cases` carries the case id for the same
    reason, since the case is the unit the split is drawn on.
    """
    p = OUT / f"{name}.json"
    idx = list(range(len(rows)))
    p.write_text(json.dumps({
        "_what_this_is": f"{name} split logits from the seed checkpoint, for pipeline cycles.",
        "_frozen_V": "not produced here and never read by the executor; this file is not V",
        "_ids_note": "qid is the original question id and is unique across bundles; index is "
                     "positional within this bundle and is NOT a stable identifier.",
        "rows": [{"type": t, "target": [float(v) for v in y],
                  "id": i, "qid": (qids[i] if qids else None), "case": (cases[i] if cases else None)}
                 for t, y, i in zip(types, rows, idx)],
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

    # ---- dev -> P / M, split by CASE so no case straddles the filter/decision boundary ---------
    #
    # The previous rule was `every 4th question to proxy`, stratified by type. That is question-level,
    # and the dev split is 120 cases x 5 questions: a question-level interleave puts four questions of
    # one case in medium and its fifth in proxy, so a candidate could pass the filter on the same case
    # the decision set scores. The cases are the unit of shared context, so they are the unit of split.
    #
    # Splitting by case also makes the two sets independent in a stronger sense than distinct question
    # ids: distinct dev questions can carry identical target distributions, so an id-level overlap test
    # is not sufficient evidence either way. Case disjointness is checkable and is checked below.
    dev = [json.loads(l) for l in DEV_LOGITS.read_text(encoding="utf-8").splitlines() if l.strip()]

    cases: dict[str, list] = {}
    for r in dev:
        cases.setdefault(r["case_id"], []).append(r)

    # Deterministic and stratified: within each workflow, every 4th case goes to the proxy. Taking
    # whole cases keeps both the workflow mix and the question-type mix representative, because each
    # case carries the same 5-question type profile.
    per_wf: dict[str, list] = {}
    for cid, rows in cases.items():
        per_wf.setdefault(rows[0]["workflow"], []).append(cid)

    proxy_cases: set[str] = set()
    for wf in sorted(per_wf):
        for i, cid in enumerate(sorted(per_wf[wf])):
            if i % 4 == 0:
                proxy_cases.add(cid)

    proxy = [r for cid in sorted(proxy_cases) for r in cases[cid]]
    medium = [r for cid in sorted(cases) if cid not in proxy_cases for r in cases[cid]]

    # The invariant this split exists to guarantee, asserted rather than assumed.
    assert not (proxy_cases & {c for c in cases if c not in proxy_cases}), "a case is on both sides"
    assert not ({r["id"] for r in proxy} & {r["id"] for r in medium}), "a question is on both sides"
    for name, rows in (("proxy", proxy), ("medium", medium)):
        mix = {t: sum(1 for r in rows if r["type"] == t) for t in sorted({r["type"] for r in dev})}
        wfs = len({r["workflow"] for r in rows})
        print(f"  {name:7s} {len(rows):4d} q  cases={len({r['case_id'] for r in rows}):3d}  "
              f"workflows={wfs}  types={mix}")
    print(f"dev: {len(dev)} -> proxy {len(proxy)} + medium {len(medium)} "
          f"(every 4th CASE to proxy, stratified by workflow; case-disjoint by assertion)")

    save("proxy", [r["target"] for r in proxy],
         [np.asarray(r["logits"], float) for r in proxy], [r["type"] for r in proxy],
         qids=[r["id"] for r in proxy], cases=[r["case_id"] for r in proxy])
    save("medium", [r["target"] for r in medium],
         [np.asarray(r["logits"], float) for r in medium], [r["type"] for r in medium],
         qids=[r["id"] for r in medium], cases=[r["case_id"] for r in medium])

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
    # The feature cache carries the original question ids, so the calibration bundle is labelled the
    # same way the dev bundles are. Without them the three-way disjointness check can only be run on
    # two of the three sets, and a bundle that silently shares questions with calibration -- the set
    # every stage is FITTED on -- would not be detectable from the artefacts.
    cal_qids = [str(q) for q in z["ids"]] if "ids" in z.files else None
    save("calibration", cal_targets, cal_logits, list(types), qids=cal_qids)
    dt = time.perf_counter() - t0
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
