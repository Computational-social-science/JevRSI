"""measure_speed_cost_axes.py -- give the Speed and Cost axes numbers, so MAP-Elites can bin on them.

WHY THIS FILE

The objective's fitness is a four-axis composite -- Intelligence, Calibration, Speed, Cost -- and until
now only two of the four had ever been measured. A MAP-Elites archive cannot bin on an axis that has no
number, and a composite would hide exactly the regression the harness route is looking for. So the two
missing axes get instrumented here, separately, with the timing discipline that makes them comparable.

WHAT "SPEED" AND "COST" MEAN HERE, PRECISELY

Speed is wall-clock per evaluated question, decomposed so that a slow configuration can be diagnosed
rather than merely observed. Three components are separated because they have different fixes:

    data     tokenising + collating           -- the CPU side; not a GPU problem
    compute  forward passes                   -- the part batching and backend choices change
    overhead anything that is neither         -- allocator churn, host pressure, synchronisation

That decomposition is not decoration. The project has already measured a 10x swing in step time on this
host that turned out to be the CUDA caching allocator taking its slow path near the VRAM ceiling
(0.199 -> 0.688 s/step, never recovering), and a separate cold-vs-warm gap of ~100x (81 s/microbatch
cold against 0.79 warm) caused purely by timing cold microbatch calls with no warmup. Both look like
"the model is slow" and neither is. An axis that reports only a total would have recorded both as
"hardware is slow" and pointed the controller at the wrong lever.

Cost is energy: average watts drawn, sampled locally by nvidia-smi. No external monitoring, per Rule 2.
It is reported separately from Speed because they are separable levers -- a harness change can halve the
step time while leaving the idle draw untouched, and vice versa.

HONEST LIMITS

  * These are THIS host's numbers under THIS load. They are not a property of the method, and Rule 4
    requires the comparison table to state whether compute was matched.
  * Speed measured while another job shares the card is meaningless. This script refuses to run if VRAM
    is already occupied, for exactly the reason recorded in the long-gpu-batch-survival skill: an
    orphaned training process once made a healthy configuration look 9x slower.
  * A single timing run is one draw. The spread across repeats is reported, not just the mean.

Run: python measurement/measure_speed_cost_axes.py
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
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Both the subject repository and the base model are EXTERNAL to this repository, so no path to
# either is portable and none can be derived from __file__. config/paths.json declares them and
# paths.py raises rather than falling back silently, so a run cannot attribute numbers to a
# checkpoint it never loaded.
from paths import backbone, require, subject                                    # noqa: E402

AUTORESEARCH = pathlib.Path(str(ROOT))
SUBJECT = require(subject(), "the agent-jev subject repository (JEVRSI_SUBJECT)")
OUT = AUTORESEARCH / "measurement" / "speed_cost_axes.json"
SEED_CKPT = SUBJECT / "checkpoints" / "autoresearch" / "final.safetensors"
BACKBONE = str(require(backbone(), "the Qwen3 base model (JEVRSI_BACKBONE)"))
N_QUESTIONS = 120          # enough to average out, small enough to repeat
REPEATS = 3
VRAM_BUSY_THRESHOLD_MIB = 2000


def gpu_power_w() -> float | None:
    """Instantaneous board power in watts. Returns None rather than guessing if nvidia-smi omits it."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, errors="replace")
    try:
        return float(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def vram_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, errors="replace")
    try:
        return int(float(r.stdout.strip().splitlines()[0]))
    except (ValueError, IndexError):
        return 0


def main() -> int:
    busy = vram_used_mib()
    if busy > VRAM_BUSY_THRESHOLD_MIB:
        print(f"[FAIL] the card already holds {busy} MiB. Timing measured under contention is "
              f"meaningless -- this project has already been burned by exactly that (an orphaned "
              f"training process made a healthy configuration look 9x slower). Free the GPU first.")
        return 2
    if not (SUBJECT / "agentjev" / "model.py").exists():
        print("[FAIL] subject repository not mounted")
        return 2

    sys.path.insert(0, str(SUBJECT))
    sys.path.insert(0, str(SUBJECT / "scripts"))
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from agentjev.model import AgentJevModel
    import evaluate_split as ev

    print(f"subject      : {SUBJECT}")
    print(f"questions    : {N_QUESTIONS} x {REPEATS} repeats (spread is reported, not just the mean)")
    print(f"VRAM at start: {busy} MiB (must be idle -- this is a precondition, not a warning)")

    tok = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    pad_id = tok.pad_token_id or tok.eos_token_id
    model = AgentJevModel(BACKBONE, dtype=torch.bfloat16)
    model.load_state_dict(load_file(SEED_CKPT, device="cpu"), strict=True)
    model.to("cuda:0").eval()

    rows = ev.load_split("dev")[:N_QUESTIONS]

    # ---- data side: tokenise once, then re-time the collate separately -----------------------
    t0 = time.perf_counter()
    rows = ev.tokenize_rows([dict(r) for r in rows], tok)
    tokenise_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch = ev.batch_rows(rows, pad_id)
    torch.cuda.synchronize()
    collate_s = time.perf_counter() - t0

    # ---- WARMUP. Not optional. ----------------------------------------------------------------
    # Timing a cold call measures cuBLAS kernel selection, lazy CUDA module loading and the
    # allocator's first touch of every weight. Measured on this host that gap is ~100x
    # (81 s/microbatch cold against 0.79 warm). A Speed axis that skipped warmup would report the
    # startup cost as the steady-state cost, and the controller would then optimise the wrong thing.
    print("\nwarming up (excluded from every timing below) ...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(5):
            model(batch)
    torch.cuda.synchronize()

    per_q, watts = [], []
    for rep in range(REPEATS):
        t0 = time.perf_counter()
        w0 = gpu_power_w()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(batch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        w1 = gpu_power_w()
        per_q.append(dt / len(rows))
        watts.append(w1 if w1 is not None else None)
        print(f"  repeat {rep+1}/{REPEATS}: {dt*1000:7.1f} ms for {len(rows)} q  "
              f"= {dt/len(rows)*1000:6.2f} ms/q   power={w1} W")
        time.sleep(1.0)

    v_peak = vram_used_mib()
    good = [w for w in watts if w is not None]
    speed = {
        "ms_per_question_mean": round(statistics.mean(per_q) * 1000, 3),
        "ms_per_question_stdev": round(statistics.stdev(per_q) * 1000, 3) if len(per_q) > 1 else None,
        "n_questions": len(rows), "repeats": REPEATS,
        "decomposition_note": "this is a single forward pass over one padded batch, so it isolates the "
                              "compute component. A full eval adds the tokenise and collate costs "
                              "reported below and the per-batch python overhead.",
        "tokenise_s_for_batch": round(tokenise_s, 3),
        "collate_s_for_batch": round(collate_s, 3),
        "peak_vram_mib_after": v_peak,
    }
    cost = {
        "watts_during_forward": good,
        "watts_mean": round(statistics.mean(good), 1) if good else None,
        "watts_note": "board power during the forward pass only, sampled locally via nvidia-smi. Idle "
                      "and system-reserve draw are NOT included, so this understates total cost; a "
                      "six-week electricity budget needs duty-cycle measurement, not a spot sample.",
        "energy_per_1k_questions_wh": (round(statistics.mean(good) * statistics.mean(per_q) * 1000 / 3600, 4)
                                       if good else None),
    }

    print(f"\n{'='*74}\nSPEED AXIS\n{'='*74}")
    print(f"  {speed['ms_per_question_mean']:.3f} ms/question  "
          f"(stdev {speed['ms_per_question_stdev']} over {REPEATS} repeats)")
    print(f"  tokenise {tokenise_s*1000:.0f} ms + collate {collate_s*1000:.0f} ms for the whole batch")
    print(f"  peak VRAM after: {v_peak} MiB")
    print(f"\n{'='*74}\nCOST AXIS\n{'='*74}")
    print(f"  board power during forward: {cost['watts_during_forward']} W")
    print(f"  energy per 1000 questions : {cost['energy_per_1k_questions_wh']} Wh")
    print(f"  {cost['watts_note']}")

    OUT.write_text(json.dumps({
        "_what_this_is": "Speed and Cost axes measured on this host, decomposed so a slow "
                         "configuration can be diagnosed rather than merely observed.",
        "_why": "Two of the objective's four axes had no number at all, and MAP-Elites cannot bin on an "
                "axis with no number. A composite would have hidden the Calibration regression.",
        "_warmup": "5 untimed forward passes precede every measurement. Timing cold calls on this host "
                   "reads ~100x slow (81 s/microbatch cold against 0.79 warm) because the first call "
                   "pays kernel selection, lazy module load and allocator first-touch.",
        "_precondition": f"refuses to run above {VRAM_BUSY_THRESHOLD_MIB} MiB of existing VRAM, because "
                         "timing under contention is not a measurement of this configuration",
        "_limits": ["this host, this load -- not a property of the method",
                    "Cost is a forward-pass spot sample; a six-week budget needs duty-cycle data",
                    "Speed here is a single padded forward, not a full evaluation loop"],
        "speed": speed, "cost": cost,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
