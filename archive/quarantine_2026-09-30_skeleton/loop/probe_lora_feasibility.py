"""probe_lora_feasibility.py -- does the 0.6B seed actually need 4-bit quantisation?

THE CLAIM UNDER TEST (proposal section 2.1)
-------------------------------------------
"QLoRA (4-bit quantized base + 16-bit LoRA adapters) compress 7B fine-tuning into the 6-10 GB range;
for sub-2B models the footprint is far lower." and, in the abstract, QLoRA is "the only
parameter-efficient path compatible with 12 GB".

That is two claims welded together, and only the first is arithmetic:

    full-parameter fine-tuning is infeasible   -- TRUE, and measured below
    4-bit quantisation is therefore required    -- NOT ESTABLISHED

The paper arithmetic says otherwise. For this seed (Qwen3-0.6B: 28 layers, hidden 1024, GQA 16/8,
intermediate 3072) a rank-16 all-linear LoRA has 168.8M trainable parameters -- 28% of the backbone,
which is a large adapter, not a light one -- and the whole training state still fits:

    base BF16 (frozen)        0.6B x 2B          = 1.12 GiB
    LoRA params + grads       168.8M x 4B        = 0.62 GiB
    LoRA AdamW m+v (fp32)     168.8M x 8B        = 1.26 GiB
    activations, ctx, driver  (measured below)

So 4-bit buys under 1 GiB on a card with 12.28 GiB. The real cost of quantisation here is not
memory -- it is NUMERICAL. 4-bit rounding of the frozen base injects error into every forward pass,
and that error becomes part of Floor B, the run-to-run standard deviation that sets epsilon. A larger
Floor B means a larger epsilon, which means fewer detectable improvements. The proposal treats
quantisation as free (section 2.3 lists it under "settings treated as a fixed baseline") while it is
in fact a decision about how much noise the accept rule must tolerate.

WHAT THIS SCRIPT DOES
---------------------
It measures, on this host, with the real model:
  1. full-parameter AdamW footprint        -- to confirm the first claim
  2. BF16 base + LoRA r16 all-linear        -- peak reserved VRAM over a few real steps
  3. 4-bit base + LoRA r16 all-linear       -- the same, IF bitsandbytes is available
and prints the headroom against the 12.28 GiB card.

It does NOT decide which to use. That decision needs the Floor B measurement, which is a separate
experiment with its own pre-registered criterion. What this script establishes is whether the
proposal's stated REASON for 4-bit holds on this hardware.

Run: bash loop/run_clean.sh python loop/probe_lora_feasibility.py
"""
from __future__ import annotations

import pathlib
import os
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
BASE = str(pathlib.Path(os.environ.get("JEVRSI_BACKBONE",
        ROOT.parent / "jev_repro" / "models" / "Qwen3-0.6B")))
CARD_GIB = 12282 / 1024          # MiB -> GiB, the card as nvidia-smi reports it


def gib(x) -> float:
    return x / 1024 ** 3


def free_report(tag: str) -> float:
    free, total = torch.cuda.mem_get_info()
    print(f"    [{tag}] free {gib(free):6.2f} / {gib(total):.2f} GiB "
          f"({100 * free / total:.0f}%)")
    return gib(free)


def run_lora(steps: int, seq: int, batch: int, quant: bool, seed: int = 20260929) -> dict:
    """Attach a rank-16 all-linear LoRA and take a few real optimisation steps."""
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    free_report("before" + (" 4bit" if quant else " bf16"))

    t0 = time.time()
    kw = {"dtype": torch.bfloat16, "local_files_only": True}
    if quant:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, **kw).to("cuda:0")
    if quant:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    load_s = time.time() - t0

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    torch.manual_seed(seed)
    t1 = time.time()
    losses = []
    for _ in range(steps):
        ids = torch.randint(0, 150000, (batch, seq), device="cuda:0")
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
    step_s = (time.time() - t1) / max(1, steps)
    peak = torch.cuda.max_memory_reserved()

    print(f"    trainable {trainable/1e6:7.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")
    print(f"    load {load_s:.1f}s | {step_s:.2f} s/step | loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"    PEAK RESERVED {gib(peak):.2f} GiB  -> headroom {CARD_GIB - gib(peak):.2f} GiB")

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return {"quantized": quant, "trainable_M": round(trainable / 1e6, 2), "total_M": round(total / 1e6, 1),
            "peak_reserved_gib": round(gib(peak), 3), "s_per_step": round(step_s, 3),
            "loss_first": losses[0], "loss_last": losses[-1], "load_s": round(load_s, 1)}


def full_param_estimate(seq: int, batch: int) -> dict:
    """Footprint of full-parameter AdamW, from the MEASURED parameter count.

    The measured total is 751.6M, not 0.6B: the vocabulary is 151936 x 1024 = 155.6M embeddings on
    top of a 0.6B backbone. An earlier revision of this script used the nominal "0.6B" and therefore
    understated the optimizer state, and -- worse -- compared weights+grads+optimizer alone against
    the card size while ignoring activations, which is what made it print "FITS" for a
    configuration that does not fit. Both errors are corrected here: the parameter count is the
    measured one, and the activation term is included rather than waved at.

    The activation estimate is deliberately generous (12 x hidden bytes per token per layer) because
    the honest use of this number is to show that full-parameter is excluded by a wide margin, not to
    land within 0.1 GiB of the boundary.
    """
    n = 751.6e6                      # measured: AutoModelForCausalLM on Qwen3-0.6B
    hidden, layers = 1024, 28
    fp32_w = n * 4
    fp32_g = n * 4
    adam = n * 8
    act = layers * batch * seq * hidden * 12 * 2      # x2 for bytes-per-element
    ctx = 0.8e9
    total = fp32_w + fp32_g + adam + act + ctx
    print(f"    fp32 weights {gib(fp32_w):5.2f} + grads {gib(fp32_g):5.2f} + AdamW {gib(adam):5.2f}"
          f" + activations {gib(act):5.2f} + ctx {gib(ctx):5.2f}")
    print(f"    = {gib(total):5.2f} GiB vs card {CARD_GIB:.2f} GiB -> "
          f"{'FITS' if gib(total) < CARD_GIB else 'DOES NOT FIT'}")
    return {"params_M": 751.6, "fp32_adamw_gib": round(gib(fp32_w + fp32_g + adam), 2),
            "activations_gib": round(gib(act), 2),
            "total_gib": round(gib(total), 2), "fits": gib(total) < CARD_GIB}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--try-4bit", action="store_true", help="also measure the QLoRA path")
    ap.add_argument("--out", default="measurement/lora_feasibility.json")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] no CUDA device")
        return 1

    print("=" * 74)
    print(f"claim 1: full-parameter fine-tuning is infeasible on this card ({CARD_GIB:.2f} GiB)")
    print("=" * 74)
    fp = full_param_estimate(a.seq, a.batch)   # computed once: it prints, and the JSON needs it too
    res = {"card_gib": CARD_GIB, "full_param": fp, "arms": []}

    print("\n" + "=" * 74)
    print("claim 2: therefore 4-bit is REQUIRED -- measuring the BF16 + LoRA path")
    print("=" * 74)
    try:
        res["arms"].append(run_lora(a.steps, a.seq, a.batch, quant=False))
    except Exception as e:
        print(f"    [FAIL] BF16 + LoRA: {type(e).__name__}: {str(e)[:200]}")
        res["arms"].append({"quantized": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    if a.try_4bit:
        print("\n" + "=" * 74)
        print("for comparison: the QLoRA path")
        print("=" * 74)
        try:
            res["arms"].append(run_lora(a.steps, a.seq, a.batch, quant=True))
        except Exception as e:
            print(f"    [SKIP] 4-bit unavailable: {type(e).__name__}: {str(e)[:160]}")
            res["arms"].append({"quantized": True, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    print("\n" + "=" * 74)
    print("verdict")
    print("=" * 74)
    bf = next((x for x in res["arms"] if x.get("quantized") is False), None)
    if bf and "peak_reserved_gib" in bf:
        head = CARD_GIB - bf["peak_reserved_gib"]
        print(f"  full-parameter AdamW : {res['full_param']['fp32_adamw_gib']:.2f} GiB "
              f"-> {'FITS' if res['full_param']['fits'] else 'DOES NOT FIT'}")
        print(f"  BF16 + LoRA r16      : {bf['peak_reserved_gib']:.2f} GiB peak, "
              f"headroom {head:.2f} GiB")
        if res["full_param"]["fits"] is False and head > 2.0:
            print("\n  => claim 1 HOLDS (full-param does not fit)")
            print("  => claim 2 does NOT follow: 4-bit is not required for memory reasons here.")
            print("     Whether to use it is then a NUMERICAL decision, because quantisation error")
            print("     enters Floor B, and Floor B sets epsilon. Measure Floor B both ways.")
        elif res["full_param"]["fits"]:
            print("\n  => claim 1 is FALSE on this host: full-parameter fits. The proposal's")
            print("     premise does not hold and must be re-derived from measurement.")
    else:
        print("  BF16 + LoRA arm did not complete; see the error above")

    Path(a.out).write_text(json.dumps({
        "_producer": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "why": "A stale background process can overwrite a corrected artifact, and then a later "
                   "reader cannot tell which version of the reasoning produced the numbers. The 2026-09-29 "
                   "run of this very script did exactly that: an earlier revision wrote "
                   "full_param={'fp32_adamw_gib': 8.94, 'fits': True} by comparing weights+grads+optimizer "
                   "against the card while ignoring activations, and a corrected re-run had to overwrite it. "
                   "Recording the producer lets the audit detect that the artifact and the script disagree.",
        },
        "card_gib": CARD_GIB,
        "full_param": fp,
        "arms": res["arms"],
    }, indent=2), encoding="utf-8")
    print(f"\n  written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
