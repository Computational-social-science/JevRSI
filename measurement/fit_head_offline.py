"""fit_head_offline.py -- is fine-tuning the lever, or is the head the lever?

THE QUESTION THIS ANSWERS, AT 1/17TH THE COST.

A Floor B replicate trains a 596M backbone for 600 steps at 2.11 h. The architecture splits:

    backbone 596.0M  ->  cand_vecs [C, 1024]     expensive, and what LoRA updates
    head      2.37M  ->  logits [C]              trivial, 0.40% of parameters

cache_candidate_features.py paid the expensive half ONCE: 5400 questions in 7.4 min, while sharing
the GPU with a live training run. Every head experiment after that is seconds. So the question "did those
600 steps of backbone updating buy anything, or would freezing the backbone and refitting 2.37M
parameters have done the same job?" stops being a 2.11 h experiment and becomes a seconds experiment.

THE COMPARISON, AND WHY IT IS FAIR.

    shipped head   the head that already exists in the seed checkpoint, scored on calibration
    refit head     a fresh head, same architecture, fit on CACHED FEATURES ONLY

Both are evaluated the same way, on the same split, with the same metric code. The refit head never sees
a gradient from the backbone, because the backbone never runs again. If refit >= shipped, the backbone
update was not what produced the model's behaviour and the training budget was aimed at the wrong part of
the network. If refit < shipped, the backbone genuinely moved and 2.11 h bought something real.

Either answer is worth having, and the expensive one is the answer you get by default if you never ask.

HONEST LIMITS, STATED UP FRONT.

  * The head is fit on train (4800 questions) and scored on calibration (600). Dev is NOT touched, so this
    is a held-out comparison -- but it is a comparison at the SEED, before any LoRA step, so it says
    nothing yet about what 600 LoRA steps do. That comparison needs the LoRA checkpoint's features and is
    the natural next run of this script (--checkpoint <lora>.safetensors).
  * A single fit is one draw. No seed dispersion is reported here, so a small difference between the two
    arms is NOT evidence of a difference. This script reports the gap and says how big the noise is
    likely to be; it does not claim significance.
  * Caching calibration does not make it a fitting set. It is the scoring set here. If a future version
    fits on calibration, that number stops being held out.

Run: python measurement/fit_head_offline.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md.
ROOT = pathlib.Path(__file__).resolve().parents[2]

import os
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
FEATDIR = pathlib.Path("./measurement/features")
OUT = pathlib.Path("./measurement/head_offline.json")
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
# The base model is an EXTERNAL download, so no path to it is portable. Override with
# JEVRSI_BACKBONE; the default sits beside this repository.
BACKBONE = str(pathlib.Path(os.environ.get(
    "JEVRSI_BACKBONE", ROOT.parent / "jev_repro" / "models" / "Qwen3-0.6B")))


def load_split(tag: str, split: str) -> dict:
    f = FEATDIR / f"{split}_{tag}.npz"
    if not f.exists():
        raise SystemExit(f"[FAIL] {f} not found. Run cache_candidate_features.py for split '{split}'.")
    z = np.load(f, allow_pickle=True)
    return {"X": z["features"].astype(np.float32), "off": z["offsets"], "y": z["targets"].astype(np.float32),
            "types": z["types"], "n_cand": z["n_cand"], "ids": z["ids"]}


def to_padded(d: dict):
    """[sum(k), D] ragged -> [N, Cmax, D] plus a validity mask. Needed because the set encoder is
    permutation-equivariant over a padded candidate SET, so the padding has to be masked, not zero-mean."""
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


def scores(logits: np.ndarray, mask: np.ndarray, y: np.ndarray):
    """The evaluator's own definitions, so these numbers are comparable to every other number here.

    accuracy and chance-corrected skill come from argmax; soft CE and Brier come from the full
    distribution. probe_post_hoc_ceiling.py established that the argmax pair is invariant to
    temperature, so only the head itself can move them.
    """
    acc, sk, ce, br = [], [], [], []
    for i in range(len(mask)):
        m = mask[i]
        p, t = logits[i][m], y[i][m]
        k = len(p)
        if k < 2:
            continue
        am = int(p.argmax())
        acc.append(float(am == int(t.argmax())))
        sk.append((acc[-1] - 1.0 / k) / (1.0 - 1.0 / k))
        e = np.exp(p - p.max())
        e /= e.sum()
        ce.append(float(-np.sum(t * np.log(np.clip(e, 1e-12, None)))))
        br.append(float(np.sum((e - t) ** 2)))
    return {"accuracy": float(np.mean(acc)), "chance_corrected": float(np.mean(sk)),
            "soft_cross_entropy": float(np.mean(ce)), "brier_sum": float(np.mean(br)), "n": len(acc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(SEED_CKPT),
                    help="the checkpoint whose features are cached (tag = its filename stem)")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--score-split", default="calibration")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tag = pathlib.Path(args.checkpoint).stem
    sys.path.insert(0, str(SUBJECT))
    from agentjev.model import AgentJevModel                                  # noqa: E402

    print(f"checkpoint tag : {tag}")
    print(f"fit on         : {args.fit_split}   score on: {args.score_split}")
    print(f"head           : the model's OWN head class, freshly initialised "
          f"({args.epochs} epochs, lr={args.lr}, AdamW)")

    tr = load_split(tag, args.fit_split)
    sc = load_split(tag, args.score_split)
    Xtr, Mtr, Ytr, Ttr, ktr = to_padded(tr)
    Xsc, Msc, Ysc, Tsc, ksc = to_padded(sc)
    print(f"  fit   : {Xtr.shape[0]} questions, max k={Xtr.shape[1]}, dim={Xtr.shape[2]}")
    print(f"  score : {Xsc.shape[0]} questions, max k={Xsc.shape[1]}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # Build ONLY the head, on CPU, with no backbone. This is the whole point: the 596M never loads.
    from agentjev.model import ScalarScorer, CandidateSetEncoder                # noqa: E402
    set_dim, set_heads, set_layers = 256, 4, 2
    proj_in = torch.nn.Linear(1024, set_dim)
    set_enc = CandidateSetEncoder(set_dim, set_heads, set_layers)
    proj_out = torch.nn.Linear(set_dim, 1024)
    scorer = ScalarScorer(1024, set_dim)
    head = torch.nn.ModuleDict({"proj_in": proj_in, "set_encoder": set_enc,
                                "proj_out": proj_out, "scorer": scorer}).to(device)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  head params: {n_head/1e6:.2f}M  (the 596M backbone is NOT loaded -- that is the saving)")

    def head_forward(x, mask):
        h = head["proj_in"](x)
        h = head["set_encoder"](h, mask)
        return head["scorer"](x + head["proj_out"](h))

    # ---- shipped head, for the comparison ------------------------------------
    model = AgentJevModel(BACKBONE, dtype=torch.float32)
    from safetensors.torch import load_file                                   # noqa: E402
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        sl = []
        for i in range(0, len(Xsc), 64):
            xb = torch.from_numpy(Xsc[i:i+64]).to(device)
            mb = torch.from_numpy(Msc[i:i+64]).to(device)
            sl.append(head_forward_shipped(model, xb, mb).float().cpu().numpy())
    shipped_logits = np.concatenate(sl, 0)
    del model
    if device == "cuda:0":
        torch.cuda.empty_cache()
    shipped = scores(shipped_logits, Msc, Ysc)
    print(f"\n  shipped head (in the checkpoint) on {args.score_split}: "
          f"acc={shipped['accuracy']:.4f} skill={shipped['chance_corrected']:.4f} "
          f"CE={shipped['soft_cross_entropy']:.4f}")

    # ---- refit head, on cached features only ---------------------------------
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    Xt = torch.from_numpy(Xtr).to(device)
    Mt = torch.from_numpy(Mtr).to(device)
    Yt = torch.from_numpy(Ytr).to(device)
    t0 = time.time()
    head.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt), device=device)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), 64):
            idx = perm[i:i+64]
            logits = head_forward(Xt[idx], Mt[idx])
            m = Mt[idx]
            # Mask BEFORE the reduction, then average over the survivors. Reducing over the padded
            # slots would put ~1-2 fabricated candidates (logit 0, target 0) into every question's
            # loss, which is a silently different objective rather than a crash.
            lp = torch.log_softmax(logits.float(), -1)
            # Zero the padded slots' contribution BEFORE summing over candidates, then average over
            # questions. Two separate mistakes are being avoided here: reducing over padded slots
            # would add ~1-2 fabricated candidates (logit 0, target 0) to every question's loss, and
            # boolean-masking a [B] tensor with a [B, C] mask is a shape error rather than a warning.
            ym = Yt[idx] * m
            per_q = -(ym * lp).sum(-1)                  # [B], padded slots contribute exactly 0
            loss = per_q.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}/{args.epochs}  loss={tot/max(1,nb):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    fit_s = time.time() - t0
    head.eval()
    with torch.inference_mode():
        rl = []
        for i in range(0, len(Xsc), 64):
            xb = torch.from_numpy(Xsc[i:i+64]).to(device)
            mb = torch.from_numpy(Msc[i:i+64]).to(device)
            rl.append(head_forward(xb, mb).float().cpu().numpy())
    refit_logits = np.concatenate(rl, 0)
    refit = scores(refit_logits, Msc, Ysc)
    print(f"\n  REFIT head (frozen backbone, {fit_s:.0f}s of CPU/GPU compute) on {args.score_split}: "
          f"acc={refit['accuracy']:.4f} skill={refit['chance_corrected']:.4f} "
          f"CE={refit['soft_cross_entropy']:.4f}")

    gap = {k: round(refit[k] - shipped[k], 6) for k in ("accuracy", "chance_corrected",
                                                        "soft_cross_entropy", "brier_sum")}
    print(f"\n{'='*74}\nGAP (refit - shipped)\n{'='*74}")
    for k, v in gap.items():
        print(f"  {k:20s} {v:+.4f}")
    print(f"\n  Cost of the refit: {fit_s:.0f}s.  Cost of one LoRA replicate: 7596s "
          f"({7596/max(1,fit_s):.0f}x).")
    print("\n  READ THIS BEFORE CONCLUDING ANYTHING:")
    print("  * This is the SEED checkpoint, before any LoRA step. It does NOT yet say what 600 steps of")
    print("    backbone updating buy. Re-run with --checkpoint <lora-600>.safetensors to find out.")
    print("  * One fit is one draw. No seed dispersion is reported, so a gap smaller than the run-to-run")
    print("    spread is not evidence of anything. Floor B for THIS head-only recipe is unmeasured.")
    print("  * dev was not touched. test was not touched.")

    pathlib.Path(args.out).write_text(json.dumps({
        "_what_this_is": "Head refit on cached frozen-backbone features, vs the head already in the "
                        "checkpoint. Both scored on the same split with the evaluator's own metrics.",
        "_question": "Is the 596M backbone update the lever, or would 2.37M head parameters on frozen "
                     "features have done the same job?",
        "checkpoint": args.checkpoint, "tag": tag,
        "fit_split": args.fit_split, "score_split": args.score_split,
        "head_params_M": round(n_head / 1e6, 3),
        "backbone_params_M": 596.0,
        "refit_seconds": round(fit_s, 1),
        "lora_replicate_seconds": 7596,
        "speedup_vs_one_lora_replicate": round(7596 / max(1, fit_s), 1),
        "shipped": shipped, "refit": refit, "gap_refit_minus_shipped": gap,
        "_limits": ["seed checkpoint only: says nothing yet about what 600 LoRA steps buy",
                    "single fit, no seed dispersion: a small gap is not evidence",
                    "dev and test untouched"],
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


def head_forward_shipped(model, x, mask):
    """The checkpoint's own head, applied to cached features. Uses the model's real modules so the
    comparison is head-vs-head and not head-vs-reimplementation."""
    h = model.proj_in(x)
    h = model.set_encoder(h, mask)
    return model.scorer(x + model.proj_out(h))


if __name__ == "__main__":
    raise SystemExit(main())
