"""
head_route.py -- the one harness route that changes WEIGHTS rather than post-processing logits.

WHY THIS IS NOT A Stage
    Every other outer-level route is a function of logits the model has already produced: take z,
    return p. That is why they cost milliseconds and need no GPU. head.probe_refit does not fit that
    shape. It refits 2.37M head parameters on cached frozen-backbone features, so it changes the map
    from features to logits -- the logits themselves are different numbers afterwards. Wrapping it in
    the Stage abstraction would let it be scored by the same `stage.apply(probs, types)` call as a
    temperature, which is precisely the confusion the proposal's harness-vs-parameter attribution
    exists to prevent. A reader of the log would see a `harness` route that actually trained
    parameters, and the attribution table would be wrong in a way that flatters the cheap route.

    So this module returns LOGITS, and the executor scores those with the same metric code it uses
    for every other route. The distinction is in what the function consumes and returns, not in how
    the number is measured.

THE COST THIS BUYS, MEASURED ON THIS HOST
    backbone 596.0M, one 600-step LoRA replicate ....... 7596 s  (2.11 h)
    head      2.37M, refit on cached features ..........   16.5 s
    ratio .............................................   461.8x

    The feature cache is the reason: cand_vecs [C, 1024] is computed once per checkpoint and every
    later head experiment reads it from disk. The cache is AMORTISED -- 5400 questions in 7.4 min,
    once, then seconds per experiment -- so the first cycle of this route is not cheap and every
    subsequent one is. That asymmetry is recorded in the payload so nobody reads a 16 s cycle as
    evidence that parameter work is free.

WHAT IT IS AND IS NOT
    A seed checkpoint. This compares the head already in the checkpoint against a fresh head of the
    same architecture fit on frozen features. It says nothing about what 600 LoRA steps on the
    BACKBONE buy -- that needs the LoRA checkpoint's features, which is `--checkpoint <lora>.safetensors`.
    And one fit is one draw: no seed dispersion is reported, so a gap smaller than the run-to-run
    spread is not evidence of anything.

    Both limits are written into the payload rather than only into this docstring, because the log is
    what gets read months later.
"""
from __future__ import annotations

import pathlib

import numpy as np

FEATDIR_DEFAULT = "measurement/features"


def feature_path(split: str, tag: str, featdir: pathlib.Path | None = None) -> pathlib.Path:
    return (featdir or pathlib.Path(FEATDIR_DEFAULT)) / f"{split}_{tag}.npz"


def load_features(split: str, tag: str, featdir: pathlib.Path | None = None) -> dict:
    """Ragged cached features, exactly as cache_candidate_features.py wrote them."""
    f = feature_path(split, tag, featdir)
    if not f.exists():
        raise FileNotFoundError(
            f"feature cache {f} not found. head.probe_refit reads CACHED features and will not run a "
            f"backbone pass to make them -- that pass is the 596M cost this route exists to avoid. "
            f"Run: python measurement/cache_candidate_features.py --splits {split}")
    z = np.load(f, allow_pickle=True)
    return {"X": z["features"].astype(np.float32), "off": z["offsets"],
            "y": z["targets"].astype(np.float32), "types": z["types"],
            "n_cand": z["n_cand"], "ids": z["ids"]}


def to_padded(d: dict):
    """[sum(k), D] ragged -> [N, Cmax, D] plus a validity mask.

    The set encoder is permutation-equivariant over a padded candidate SET, so padding must be
    masked, not zero-mean: an unmasked slot contributes a fabricated candidate (feature 0) that the
    encoder will happily mix into the set representation.
    """
    X, off, k = d["X"], d["off"], d["n_cand"]
    N, Cmax = len(k), int(k.max())
    D = X.shape[1]
    Xp = np.zeros((N, Cmax, D), dtype=np.float32)
    M = np.zeros((N, Cmax), dtype=bool)
    Y = np.zeros((N, Cmax), dtype=np.float32)
    for i in range(N):
        a, b = off[i], off[i + 1]
        Xp[i, :k[i]] = X[a:b]
        M[i, :k[i]] = True
        Y[i, :k[i]] = d["y"][a:b]
    return Xp, M, Y, d["types"], k


def refit_head_logits(fit_split: str, score_split: str, tag: str,
                      featdir: pathlib.Path | None = None,
                      epochs: int = 30, lr: float = 3e-4, seed: int = 0,
                      device: str | None = None) -> dict:
    """Fit a fresh head on `fit_split` features; return LOGITS for `score_split`.

    Returns a dict with `logits` [N, Cmax], `mask` [N, Cmax], `targets` [N, Cmax], `types`, plus the
    provenance a log line needs to be checkable later: how long it took, how many parameters, and
    the two limits stated above.
    """
    import torch
    from agentjev.model import CandidateSetEncoder, ScalarScorer   # noqa: E402

    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    tr = load_features(fit_split, tag, featdir)
    sc = load_features(score_split, tag, featdir)
    Xtr, Mtr, Ytr, _, _ = to_padded(tr)
    Xsc, Msc, Ysc, Tsc, _ = to_padded(sc)

    # The model's OWN head class, freshly initialised. Reimplementing the architecture here would
    # make this a comparison of head-vs-reimplementation instead of head-vs-head.
    set_dim, set_heads, set_layers = 256, 4, 2
    head = torch.nn.ModuleDict({
        "proj_in": torch.nn.Linear(Xtr.shape[2], set_dim),
        "set_encoder": CandidateSetEncoder(set_dim, set_heads, set_layers),
        "proj_out": torch.nn.Linear(set_dim, Xtr.shape[2]),
        "scorer": ScalarScorer(Xtr.shape[2], set_dim)}).to(dev)

    def forward(x, mask):
        h = head["set_encoder"](head["proj_in"](x), mask)
        return head["scorer"](x + head["proj_out"](h))

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    Xt, Mt, Yt = (torch.from_numpy(a).to(dev) for a in (Xtr, Mtr, Ytr))

    import time
    t0 = time.perf_counter()
    head.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            logits = forward(Xt[idx], Mt[idx])
            m = Mt[idx]
            # Mask BEFORE the reduction, then average over survivors. Reducing over padded slots puts
            # ~1-2 fabricated candidates (logit 0, target 0) into every question's loss, which is a
            # silently different objective rather than a crash.
            lp = torch.log_softmax(logits.float(), -1)
            per_q = -((Yt[idx] * m) * lp).sum(-1)        # [B]; padded slots contribute exactly 0
            loss = per_q.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
    fit_s = time.perf_counter() - t0

    head.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(Xsc), 64):
            xb = torch.from_numpy(Xsc[i:i + 64]).to(dev)
            mb = torch.from_numpy(Msc[i:i + 64]).to(dev)
            out.append(forward(xb, mb).float().cpu().numpy())
    logits = np.concatenate(out, 0)

    return {
        "logits": logits, "mask": Msc, "targets": Ysc, "types": list(Tsc),
        "n_fit_questions": int(Xtr.shape[0]), "n_score_questions": int(Xsc.shape[0]),
        "head_params": int(sum(p.numel() for p in head.parameters())),
        "fit_seconds": round(fit_s, 2), "epochs": epochs, "lr": lr, "seed": seed,
        "fit_split": fit_split, "score_split": score_split, "tag": tag, "device": dev,
        "_limits": [
            "seed checkpoint: says nothing about what 600 backbone LoRA steps buy",
            "one fit is one draw; no seed dispersion, so a gap below the run-to-run spread is not evidence",
            "the feature cache is amortised: the first cycle pays the backbone pass, later ones do not",
        ],
    }
