"""cache_candidate_features.py -- pay the backbone once, then stop paying for it.

THE ECONOMIC ARGUMENT THIS SCRIPT EXISTS TO TEST.

A Floor B replicate costs 2.11 h: 600 optimizer steps over a 596M-parameter backbone. But the
architecture splits cleanly in two:

    encode_candidates(batch) -> cand_vecs [Bq, Cmax, 1024]     <- the 596M backbone. Expensive.
    _score(cand_vecs, mask)  -> logits [Bq, Cmax]              <- a 2.4M head. Trivial.

The head is permutation-equivariant by construction (CandidateSetEncoder has no positional embeddings
over the candidate set), so a replacement head that is also permutation-equivariant drops straight in.
And once cand_vecs is on disk the head can be refit in seconds rather than hours, because the expensive
half of the computation is amortised across every head experiment that will ever be run.

That reframes the project's central question. "Is fine-tuning the lever?" stops being a 2.11 h
experiment and becomes a seconds experiment: fit a fresh head on cached features, compare it against
the head 600 steps of LoRA produced. If the fresh head matches it, the backbone update bought nothing
and the training budget was misallocated. If it loses, the backbone update carries real weight. Both
answers are worth having and both cost one forward pass.

SPLIT SIZES, measured from typed_decisions/prepared/. The 840/120/120 figures in the training logs are
STATES, not questions:

    calibration   600 questions      fitting / calibration
    dev           600 questions      held out -- decision signal
    train        4800 questions      fitting
    test         2000 questions      FROZEN. Refused by this script, always.

The 600-question calibration split is 5x larger than the 120 the pre-registration assumed, which
materially strengthens the cheap route: 600 labelled questions is a workable fitting set for a linear
recalibration or a small probe on frozen features, at 1/20th the cost of one LoRA replicate.

HELD-OUT SAFETY. Only train and calibration are read. Caching dev is permitted but makes dev a fitting
set, so every number computed from it afterwards is a training number; the script says so out loud. The
frozen test split is refused outright.

Run from the agent-jev directory (the evaluator's data paths are relative to it):

    cd ../agent-jev && python measurement/cache_candidate_features.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md. This file sits one level below the
# root, so the root is parents[1]; parents[2] would be the root's parent, and every artifact
# read through it would come from a neighbouring directory.
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
# paths.py sits at the project root and this file lives one level below it, so the root is not
# importable until it is put on sys.path. This script is normally launched with the SUBJECT repo as
# the working directory, which is exactly the case where the root is absent from sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The subject repository is an EXTERNAL clone, so no path to it is portable and none can be
# derived from __file__. config/paths.json declares it and paths.py raises rather than falling back
# silently, so a run cannot attribute numbers to a checkpoint it never loaded.
from paths import require, subject                                              # noqa: E402

SUBJECT = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
OUT = pathlib.Path("./measurement/features")
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
# The base model is an EXTERNAL download, so no path to it is portable. Override with
# JEVRSI_BACKBONE; the default sits beside this repository.
BACKBONE = str(pathlib.Path(os.environ.get(
    "JEVRSI_BACKBONE", ROOT.parent / "jev_repro" / "models" / "Qwen3-0.6B")))
# Deliberately small. A 2.11 h training run may be holding the same card, and this script's entire
# justification is that it is cheap -- it must never be the thing that OOMs a replicate.
DEFAULT_BATCH = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(SEED_CKPT))
    ap.add_argument("--splits", nargs="+", default=["calibration", "train"])
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="debug: cap questions per split")
    args = ap.parse_args()

    if "test" in args.splits:
        raise SystemExit("[FAIL] refusing to cache the frozen test split")
    for p in (SUBJECT / "agentjev" / "model.py", SUBJECT / "scripts" / "evaluate_split.py",
              SUBJECT / "typed_decisions" / "prepared"):
        if not p.exists():
            raise SystemExit(f"[FAIL] subject not mounted or incomplete: {p}")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"[FAIL] checkpoint not found: {args.checkpoint}")

    print(f"splits       : {args.splits}")
    print(f"checkpoint   : {os.path.basename(args.checkpoint)}")
    print(f"backbone     : {BACKBONE}")
    print(f"batch size   : {args.batch_size}")
    print(f"destination  : {args.out}")
    if "dev" in args.splits:
        print("\n  [NOTE] dev is being cached. From this moment dev is a FITTING set. Any score")
        print("         computed from these features is a training number, not a held-out one.")

    # Reuse the frozen evaluator's own loader, tokenizer and collate. Reimplementing any of them
    # would create a second definition of what is in a split, which is exactly the kind of drift that
    # makes a later comparison meaningless.
    sys.path.insert(0, str(SUBJECT))
    sys.path.insert(0, str(SUBJECT / "scripts"))
    from transformers import AutoTokenizer                                     # noqa: E402
    from safetensors.torch import load_file                                    # noqa: E402
    from agentjev.model import AgentJevModel                                    # noqa: E402
    import evaluate_split as ev                                                 # noqa: E402

    tok = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AgentJevModel(BACKBONE, dtype=torch.bfloat16)
    model.load_state_dict(load_file(args.checkpoint, device="cpu"), strict=True)
    model.to("cuda:0").eval()
    # Both accessors yield (name, parameter) pairs -- backbone_named_parameters() delegates to
    # nn.Module.named_parameters(), so unpacking only the parameters is an AttributeError.
    n_head = sum(p.numel() for _, p in model.head_named_parameters())
    n_bb = sum(p.numel() for _, p in model.backbone_named_parameters())
    print(f"loaded strict=True | backbone {n_bb/1e6:.1f}M | head {n_head/1e6:.2f}M "
          f"({n_head/(n_bb+n_head)*100:.2f}% of params)")

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = os.path.basename(args.checkpoint).replace(".safetensors", "")
    manifest = {}

    for split in args.splits:
        rows = ev.load_split(split)
        if args.limit:
            rows = rows[:args.limit]
        rows = ev.tokenize_rows(rows, tok)
        print(f"\n{split}: {len(rows)} questions tokenized", flush=True)

        feats, tgts, types, ids, ks = [], [], [], [], []
        t0 = time.time()
        verified = False
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start + args.batch_size]
                b = ev.batch_rows(chunk, pad_id)     # already on cuda:0 -- do not move it again
                cv = model.encode_candidates(b)      # [Bq, Cmax, 1024]

                # FIDELITY CHECK, on the first batch of the first split. The whole cheap route rests
                # on the claim that a head fitted on these cached features reproduces the model. If the
                # cache is not faithful, every head experiment is measuring the cache instead of the
                # model -- and it would fail silently, because a head can always fit a slightly wrong
                # target. So compare the two paths explicitly, head-on, once, here.
                if not verified:
                    with torch.inference_mode():
                        direct = model._score(cv, b["cand_mask"]).float()
                        again = model(b)["logits"].float()
                    m = b["cand_mask"]
                    d = (direct[m] - again[m]).abs().max().item()
                    rel = d / max(1e-9, again[m].abs().max().item())
                    print(f"  [verify] _score(cand_vecs) vs model(batch)['logits']: "
                          f"max|diff|={d:.3e} rel={rel:.3e} over {int(m.sum())} candidates")
                    if rel > 0.02:
                        raise SystemExit(
                            f"[FAIL] the cached-feature path does not reproduce the model's own "
                            f"logits (rel={rel:.3e} > 0.02). Everything downstream would be measuring "
                            f"the cache rather than the model, so this must stop here rather than "
                            f"produce a confident wrong answer.")
                    verified = True

                for j, r in enumerate(chunk):
                    k = len(r["target"])
                    feats.append(cv[j, :k].to(torch.float16).cpu().numpy())
                    tgts.append(np.asarray(r["target"], dtype=np.float32))
                    types.append(r["question"]["type"])
                    ids.append(str(r.get("id", f"{split}-{start+j}")))
                    ks.append(k)
                if start % (args.batch_size * 50) == 0:
                    el = time.time() - t0
                    n = min(start + args.batch_size, len(rows))
                    print(f"  {n}/{len(rows)}  {el:.0f}s  ({el/max(1,n):.3f} s/q)", flush=True)
        dt = time.time() - t0

        f = outdir / f"{split}_{tag}.npz"
        np.savez_compressed(
            f,
            features=np.concatenate(feats, 0),        # [sum(k), 1024] float16
            offsets=np.cumsum([0] + ks),              # row i -> features[offsets[i]:offsets[i+1]]
            targets=np.concatenate(tgts, 0),
            types=np.array(types), ids=np.array(ids), n_cand=np.array(ks),
        )
        manifest[split] = {"file": f.name, "n_questions": len(rows), "n_candidates": int(sum(ks)),
                           "dim": int(feats[0].shape[1]), "seconds": round(dt, 1),
                           "s_per_question": round(dt / max(1, len(rows)), 4),
                           "MB": round(f.stat().st_size / 1e6, 1)}
        print(f"  -> {f.name}  {manifest[split]['MB']} MB  in {dt:.1f}s")
        del feats
        torch.cuda.empty_cache()

    (outdir / f"manifest_{tag}.json").write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "backbone": BACKBONE,
        "_what_this_is": "Frozen-backbone candidate features, for refitting the 2.4M head offline.",
        "_why": "The backbone is 596M and costs 2.11 h per 600-step replicate; the head is 2.4M and "
                "costs seconds once these features exist. Caching turns 'is fine-tuning the lever?' "
                "from a 2.11 h experiment into a seconds experiment.",
        "_held_out": "test was never read. dev was not cached. If you add dev, it stops being held "
                     "out and every number computed from it is a training number.",
        "_api_note": "Loaded via evaluate_split.load_split/tokenize_rows/batch_rows so the split "
                     "definition and the collation stay single-sourced.",
        "splits": manifest,
    }, indent=2), encoding="utf-8")
    print(f"\nmanifest: {outdir}/manifest_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
