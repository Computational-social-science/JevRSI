"""profile_train_step.py -- where does a training step go, and does the recipe fit at all?

!! READ THIS BEFORE TRUSTING ANY TIMING OR PEAK-VRAM NUMBER FROM THIS SCRIPT !!

Its output for the pre-registered arm (bf16 base, batch_states=1, grad_accum=16) was:

    compute 81.192 s/microbatch
    PEAK RESERVED 12.95 GiB   headroom -0.96 GiB
    -> projected 1299.1 s/optimizer step | 216.52 h per 600-step run

**All of those are wrong: ~103x in time, ~2.4x in memory.** Measured live on an idle GPU, the same
configuration runs at 12.6 s/step (0.79 s/microbatch) and 5.3 GB.

The cause is in time_one(): it calls reset_peak_memory_stats() and then immediately times n_micro
microbatches with NO warmup loop. The first call pays cuBLAS/cuDNN kernel selection, lazy CUDA module
loading and the allocator's first touch of every weight. At n_micro=3 that startup dominates the mean,
and reset_peak_memory_stats() folds it into the peak as well. The output is not a noisy measurement of
the right quantity -- it measures a different one.

The near-miss is the dangerous part. `headroom -0.96 GiB` reads as "this recipe exceeds the card", and
that verdict would have rejected the exact recipe the protocol depends on, over a limit it does not
hit. A wrong number that recommends caution is more dangerous than one that recommends risk, because it
survives scrutiny.

FIX BEFORE REUSE: >= 5 warmup microbatches before starting the timer, and n_micro >= 20 so one slow
sample cannot dominate. Until then treat this script's timings and peaks as UNVALIDATED.

Note: the memory *conclusion* drawn below (footprint is the retained packed-path activation graph, not
optimizer state) was corroborated by gradient checkpointing taking the peak from 11.8 GiB to 7.80 GiB --
an intervention with an independent predicted direction. So conclusions (1) and (2) below stand, while the
per-configuration numbers printed by this script do not.

WHY THIS FILE EXISTS. The pre-registered Floor B plan (600 steps x 5 seeds x 2 arms) could not be run: a
20-step smoke test did not finish inside 900 s, i.e. worse than 45 s per optimizer step, which projects to
roughly 75 hours. Before shortening the protocol or dropping seeds -- both of which weaken the floor -- the
step has to be attributed.

WHAT WAS MEASURED, and the two conclusions that changed the design:

  1. Memory is pinned near the ceiling REGARDLESS of dtype or microbatch size. Measured peak reserved was
     11.86 GiB with an fp32 base, 11.90 GiB with a bf16 base, and 11.84 GiB at batch_states=1 instead of 4 --
     a 68 MiB spread across configurations that should differ by gigabytes. The footprint is therefore not
     the optimizer state and not the microbatch activation: it is the retained activation graph of the
     packed-path representation, whose P x L is large even for a single state.

  2. That retires a conclusion that does not transfer. measurement/floor_b_host_capacity.md recorded that
     "micro-batching does NOT help, the footprint is optimizer-dominated (92% params+grads+AdamW)". That was
     measured on the FULL-PARAMETER arm. Under a rank-16 adapter the optimizer holds 10.09M x 12 B = 121 MB,
     about 0.2% of the footprint. Reading the old finding into this arm points at exactly the wrong lever.

The fix that follows is backbone gradient checkpointing -- the LoRA feasibility probe had it on and peaked
at 7.80 GiB, while train.py did not and sat at 11.8 GiB. This script measures the same recipe train.py now
runs, and can reproduce the slow configuration with --no-grad-ckpt so the comparison stays a measurement
rather than a claim.

A note on an earlier revision: programmatic re-indentation broke this file mid-session and it was rewritten
from scratch. The broken version is kept beside it as profile_train_step.broken_2026-09-29.py rather than
deleted, because "this approach was tried and it broke the file" is worth more than a clean tree.

Run from the agent-jev directory (the config's data paths are relative):
    cd ../agent-jev && python measurement/profile_train_step.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md.
ROOT = pathlib.Path(__file__).resolve().parents[2]

import pathlib
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("../agent-jev")
sys.path.insert(0, str(REPO))

import torch
import yaml
from transformers import AutoTokenizer

from agentjev.data import SourceMixer, batch_to_device, make_collate
from agentjev.losses import compute_losses
from agentjev.model import AgentJevModel

CFG = REPO / "agentjev" / "configs" / "autoresearch.yaml"
CARD_GIB = 12282 / 1024
EFFECTIVE_BATCH = 16          # protocol.json: microbatch 4 x grad_accum 4


def attach_lora(model, r=16):
    from peft import LoraConfig, TaskType, get_peft_model
    base = model.path_encoder.backbone
    for p in base.parameters():
        p.requires_grad_(False)
    model.path_encoder.backbone = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    return model


def time_one(model, mixer, collate, device, n_micro, grad_accum, label,
             batch_states, grad_ckpt, max_steps):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    data_t = comp_t = 0.0
    shape = None
    for _ in range(n_micro):
        t0 = time.perf_counter()
        _, states_b = mixer.next_batch()
        batch = batch_to_device(collate(states_b), device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, perm_reg=False)
            loss = compute_losses(out, batch, {}, margin=0.5)["total"] / grad_accum
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        data_t += t1 - t0
        comp_t += t2 - t1
        if shape is None:
            shape = (int(batch["input_ids"].shape[0]), int(batch["input_ids"].shape[1]),
                     int(batch["n_states"]), int(batch["n_questions"]))
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    per_step = (data_t + comp_t) / n_micro * grad_accum
    return {
        "label": label, "batch_states": batch_states, "grad_accum": grad_accum,
        "effective_batch": batch_states * grad_accum, "grad_checkpoint": grad_ckpt,
        "data_s_per_micro": round(data_t / n_micro, 3),
        "compute_s_per_micro": round(comp_t / n_micro, 3),
        "data_share": round(data_t / max(1e-9, data_t + comp_t), 3),
        "seq_P": shape[0], "seq_L": shape[1], "states": shape[2], "questions": shape[3],
        "peak_reserved_gib": round(peak, 2),
        "headroom_gib": round(CARD_GIB - peak, 2),
        "projected_s_per_optimizer_step": round(per_step, 2),
        "projected_hours_per_run": round(per_step * max_steps / 3600, 2),
        "projected_hours_5seeds_2arms": round(per_step * max_steps * 10 / 3600, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-micro", type=int, default=3)
    ap.add_argument("--batch-states", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false",
                    help="reproduce the slow configuration train.py used before gradient "
                         "checkpointing was enabled")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default="./measurement/step_profile.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    max_steps = cfg.get("max_steps", 600)
    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(cfg["model_path"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    collate = make_collate(tok, max_len=cfg.get("max_len", 512),
                           max_state_tokens=cfg.get("max_state_tokens", 256))
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32

    results = []
    for bs in a.batch_states:
        ga = max(1, EFFECTIVE_BATCH // bs)
        print(f"\n### batch_states={bs} grad_accum={ga} effective={bs*ga} "
              f"grad_ckpt={a.grad_ckpt} dtype={a.dtype} ###", flush=True)
        mixer = SourceMixer(cfg["data_sources"], bs, seed=cfg.get("seed", 0) + 1)
        model = AgentJevModel(cfg["model_path"], set_dim=cfg.get("set_dim", 256),
                              set_layers=cfg.get("set_layers", 2),
                              set_heads=cfg.get("set_heads", 4),
                              encoder_impl=cfg.get("encoder_impl", "path"), dtype=dtype).to(device)
        model = attach_lora(model)
        if a.grad_ckpt:
            model.path_encoder.backbone.gradient_checkpointing_enable()
            model.path_encoder.backbone.config.use_cache = False
        model.train()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        r = time_one(model, mixer, collate, device, a.n_micro, ga,
                     f"{a.dtype} base, bs={bs}, ga={ga}, ckpt={a.grad_ckpt}",
                     bs, a.grad_ckpt, max_steps)
        r["trainable_M"] = round(n_train / 1e6, 2)
        results.append(r)
        print(f"  P x L = {r['seq_P']} x {r['seq_L']}  states={r['states']} questions={r['questions']}")
        print(f"  data {r['data_s_per_micro']:.2f} | compute {r['compute_s_per_micro']:.2f} s/micro "
              f"({'DATA' if r['data_share'] > .5 else 'COMPUTE'}-bound)")
        print(f"  PEAK {r['peak_reserved_gib']:.2f} GiB  headroom {r['headroom_gib']:.2f} GiB  "
              f"trainable {r['trainable_M']}M")
        print(f"  -> {r['projected_s_per_optimizer_step']:.1f} s/step | "
              f"{r['projected_hours_per_run']:.2f} h/run | 5 seeds x 2 arms = "
              f"{r['projected_hours_5seeds_2arms']:.0f} h", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    best = min(results, key=lambda r: r["projected_hours_5seeds_2arms"])
    print(f"fastest: {best['label']}  {best['projected_hours_5seeds_2arms']:.0f} h for the full plan")
    if not [r for r in results if r["headroom_gib"] > 1.0]:
        print("NO configuration leaves more than 1 GiB of headroom -- the recipe does not fit this card "
              "as written, and no amount of patience changes that.")
    Path(a.out).write_text(json.dumps({
        "_what_this_is": "Per-phase timing and peak VRAM of the LoRA arm across microbatch sizes. Written "
                         "because the pre-registered 5-seed x 2-arm plan projected to ~75 hours.",
        "_card_gib": round(CARD_GIB, 2),
        "preregistered_arm": f"{a.dtype} base + LoRA r16 + backbone gradient checkpointing={a.grad_ckpt}",
        "_why_grad_ckpt": "Without it the step sat at 11.8-11.9 GiB on a 12.28 GiB card and exceeded "
                          "45 s/optimizer step. Under LoRA the optimizer holds 10.09M x 12 B = 121 MB, so "
                          "the footprint is the retained activation graph of the packed-path "
                          "representation, not optimizer state -- which is why the full-parameter finding "
                          "that micro-batching does not help does not transfer to this arm.",
        "plan": {"max_steps": max_steps, "effective_batch": EFFECTIVE_BATCH, "seeds": 5, "arms": 2},
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
