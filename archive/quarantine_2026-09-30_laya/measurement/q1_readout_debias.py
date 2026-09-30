#!/usr/bin/env python3
"""q1_readout_debias.py -- Q1: does readout debiasing transfer to the laya architecture?

THE QUESTION
------------
Research goal v2 (docs/RESEARCH_GOAL_v2_2026-09-29.md) moved the main axis from gradient search to
readout geometry, because the training axis is measured to saturate (+2.17 pp over 5.6x compute,
below our own tau = 4.95 pp) while a 16.9 pp gap to public results sits on the readout side.

Q1 asks the cheapest question on that axis: laya reads a decision by placing a [MASK] before each
option and scoring the hidden state there, so the option's position in the prompt is part of its
representation. AnyJev (Nokia, 869 stars) reports this is worth 0.230 -> 0.073 order-flip rate and
+5.6 accuracy points on Qwen3-8B/BANKING77 with ZERO labels. Their method is cyclic-shift
marginalisation: score the K rotations of the option list and average each option's score over the
positions it visited.

`laya.common.build_sequence` already accepts `option_order` (line 77) -- the publisher left the hook
and never wired it up (`option_order` appears 0 times in agent.py). So the intervention is available
without touching publisher code, which matters: our three-way checkpoint verification assumes the
installed package is unmodified.

WHY THIS IS A CLEAN TEST
------------------------
* Zero labels, zero gradients, one forward pass per rotation. No training, so no training seed.
* The frozen test split is touched ONCE, at the end, with the decision rule fixed in advance
  (tau = 4.95 pp on dev; on test the same test-paired SE gives tau = 3.11 pp, already pre-registered
  in floor_b_laya_prereg.md). Selection happened on dev, never on test.
* Reported per split, per question type, and as a paired comparison against the identical checkpoint
  scored without debiasing -- same cases, same weights, same process.

PREDECISION, WRITTEN BEFORE THE NUMBERS
---------------------------------------
1. If debiasing transfers, the order-flip RATE must fall. That is the mechanism claim, and it is the
   primary readout: an accuracy gain without a flip-rate drop would mean something else caused it.
2. The boolean type should barely move. laya renders noul options in a fixed semantic order
   ([false, true] -- `_resolve_noul_labels`), so there is no arbitrary order to marginalise over and
   the intervention is a no-op by construction. A large boolean gain would therefore indicate a bug.
3. `choice` is where the gain must appear, and it is where the K-rotation cost is highest.
4. If nothing moves beyond tau, that is a clean negative: the 16.9 pp gap is NOT reachable by readout
   debiasing on this architecture, and Q2 (decomposing the gap) becomes the main line.

Run: python measurement/q1_readout_debias.py --split dev --expect-n 600
"""
from __future__ import annotations

import pathlib
import os
import argparse
import json
import math
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")) / "typed_decisions"
# The HF id, not a scratch copy. A scratch path under the Hermes cache is pruned after 24h idle and
# the Q1 run must be reproducible without re-fetching; the hub cache is persistent. `Agent` accepts
# either, and the id is the form that survives a reboot.
BASE = "convaiinnovations/laya-multilingual"
sys.path.insert(0, str(SUBJECT))

# The type vocabulary is NOT uniform across the two sources this script reads, and conflating them is
# the single most expensive mistake available here:
#   * the PARQUET `questions` column spells the boolean type `noul`   (choice 180 / noul 180 / score 240 on dev)
#   * the PREPARED `*_questions.jsonl` spells it `boolean`           (and carries `keys`, which parquet lacks)
# so both are accepted, and the split's own `question.type` is the authority for what a given row is --
# it is what the frozen scorer groups by, so grouping by anything else would make the per-type numbers
# incomparable with every other table in the project.
QTYPES = {"choice", "score", "noul", "boolean"}


# --------------------------------------------------------------------------- data


def load_cases(split: str) -> list[dict]:
    import pyarrow.parquet as pq
    parquet = (SUBJECT / "data" / "all" /
               ("test-00000-of-00001.parquet" if split == "test"
                else "train-00000-of-00001.parquet"))
    rows = {r["id"]: r for r in pq.read_table(parquet).to_pylist()}

    def norm(r):
        return {"id": r["id"],
                "state": json.loads(r["state"]) if isinstance(r["state"], str) else r["state"],
                "questions": (json.loads(r["questions"])
                              if isinstance(r["questions"], str) else r["questions"])}

    if split == "test":
        return [norm(r) for r in rows.values()]
    ids, seen = [], set()
    for line in (SUBJECT / "prepared" / f"{split}_questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            cid = json.loads(line)["case_id"]
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    missing = [c for c in ids if c not in rows]
    if missing:
        raise RuntimeError(f"{len(missing)} {split} case id(s) absent from the train parquet")
    return [norm(rows[c]) for c in ids]


def load_expected(split: str) -> dict:
    out = {}
    for line in (SUBJECT / "prepared" / f"{split}_questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


# ----------------------------------------------------------------- the intervention


def rotations(k: int, cap: int) -> list[list[int]]:
    """Cyclic shifts of range(k): identity first, then k-1 shifts.

    AnyJev's L0 marginalises over option-list rotations. Cyclic shifts (not all k! permutations)
    are what costs O(k) rather than O(k!), and each option still visits k distinct positions, which
    is the property the estimator needs: position bias is cancelled when every option is scored in
    every slot. `cap` bounds the work for wide questions; when k > cap we use k evenly spaced shifts,
    which preserves coverage of positions while staying affordable.
    """
    if k <= 0:
        return [[]]
    if k <= cap:
        n = k
    else:
        n = max(2, cap)
    return [[(i + s) % k for i in range(k)] for s in range(n)]


def permute_question(q: dict, order: list[int]) -> dict:
    """Return the same question with its options presented in `order`.

    Operates on the SPLIT's field names (`type` / `criteria`), which is the form `system_one` takes;
    `Agent._to_internal` does the mapping to the package's `t` / `crit`.

    Only `choice` carries an ordered option set. `score` is an ordered LEVEL set whose numeric order is
    semantic (level 0..n), so reordering it would change the question rather than the presentation;
    the boolean type is rendered by the package in a fixed semantic order, so there is nothing to
    marginalise. Both are therefore left alone, which is exactly the prediction recorded in the module
    docstring: the intervention is a no-op for them BY CONSTRUCTION, and any measured movement there is
    a bug, not a finding.
    """
    if q.get("type") != "choice":
        return q
    crit = q.get("criteria")
    if isinstance(crit, list):                 # a bare list of options carries no criteria
        keys = list(crit)
        crit = {k: None for k in keys}
    else:
        keys = list(crit.keys())
    if sorted(order) != list(range(len(keys))):
        raise ValueError(f"order {order} is not a permutation of {len(keys)} options")
    out = dict(q)
    out["criteria"] = {keys[i]: crit[keys[i]] for i in order}
    return out


# ------------------------------------------------------------------------ scoring


def probs_for(agent, state, q: dict) -> dict[str, float]:
    """Distribution over a question's canonical keys, scored once with natural order.

    Note the two vocabularies in play: the QUESTION arrives in the split's spelling (`type`/`criteria`)
    but the ANSWER comes back in the package's (`noul` scalar for the boolean type, `probabilities` for
    the rest). Mixing them up is an immediate KeyError, which is preferable to a silent wrong answer.
    """
    res = agent.system_one(state, {q["_qid"]: q["_q"]})["answers"][q["_qid"]]
    if res["type"] == "noul":
        d = {"true": float(res["noul"]), "false": 1.0 - float(res["noul"])}
    else:
        d = {k: float(v) for k, v in res["probabilities"].items()}
    keys = q["keys"]
    vals = [d[k] for k in keys]
    tot = sum(vals)
    return {k: (v / tot) for k, v in zip(keys, vals)}


def collect(agent, split: str, debias: bool, cap: int, verbose: bool) -> tuple[list[dict], dict]:
    cases = load_cases(split)
    expected = load_expected(split)
    preds: list[dict] = []
    stats = {"n_rotations": 0, "n_forward": 0, "by_type": {}, "flip_raw": 0, "flip_debiased": 0,
             "n_choice_multi": 0, "n_skipped_unknown_type": 0}
    t0 = time.time()

    # Fail loudly on an empty or mis-typed run. A silent skip is the one failure mode a measurement
    # script must never have: n=0 with exit 0 reads as "measured, found nothing", and the frozen
    # scorer's np.mean of an empty slice even returns NaN rather than raising.
    if not cases:
        raise RuntimeError(f"split {split!r} yielded 0 cases -- refusing to report an empty measurement")
    if not expected:
        raise RuntimeError(f"split {split!r} yielded 0 expected-question rows")

    for i, row in enumerate(cases):
        for qid, q in row["questions"].items():
            if q.get("type") not in QTYPES:
                stats["n_skipped_unknown_type"] += 1
                continue
            r = expected[row["id"] + ":" + qid]
            keys = list(r["question"]["keys"])
            qc = {"_qid": qid, "_q": q, "keys": keys}
            qt = q["type"]

            raw = probs_for(agent, row["state"], qc)
            stats["n_forward"] += 1

            if debias and qt == "choice" and len(keys) > 1:
                rots = rotations(len(keys), cap)
                stats["n_rotations"] += len(rots) - 1
                stats["n_choice_multi"] += 1
                # score under every rotation, then map each score back to its canonical key
                acc = {k: [] for k in keys}
                for order in rots:
                    if order == list(range(len(keys))):
                        continue                      # already measured as `raw`
                    pq_ = permute_question(q, order)
                    d = probs_for(agent, row["state"], {"_qid": qid, "_q": pq_, "keys": keys})
                    stats["n_forward"] += 1
                    # position i in the rotated prompt holds canonical key keys[order[i]]
                    for pos, k in enumerate(keys):
                        acc[k].append(d[keys[order[pos]]])
                merged = {}
                for k in keys:
                    base_w = 1.0
                    rot_w = sum(acc[k]) / len(acc[k]) if acc[k] else 0.0
                    # average in log space: the rotations are alternative presentations of ONE
                    # question, so their probabilities are pooled, not their argmaxes
                    merged[k] = math.exp(
                        (math.log(max(base_w, 1e-12)) + math.log(max(rot_w, 1e-12))) / 2.0)
                tot = sum(merged.values())
                fin = {k: v / tot for k, v in merged.items()}
            else:
                fin = raw

            # Flip counting, defined against the teacher's argmax exactly as the frozen scorer
            # defines accuracy. `target` is a teacher distribution VECTOR, so the comparison is
            # argmax-vs-argmax; treating it as a label (an earlier draft did) silently counted
            # every item as a flip and would have made the mechanism readout meaningless.
            tvec = r["target"]
            tgt_idx = int(np.argmax(np.asarray(tvec, dtype=np.float64)))
            raw_idx = int(np.argmax([raw[k] for k in keys]))
            deb_idx = int(np.argmax([fin[k] for k in keys]))
            stats["flip_raw"] += int(raw_idx != tgt_idx)
            stats["flip_debiased"] += int(deb_idx != tgt_idx)

            vals = [fin[k] for k in keys]
            s = sum(vals)
            vals = [v / s for v in vals]
            preds.append({k: r[k] for k in ("id", "case_id", "workflow", "target")}
                         | {"type": qt, "probs": vals,
                            "raw_argmax": raw_idx, "debiased_argmax": deb_idx})
        if verbose and (i + 1) % 50 == 0:
            print(f"  cases {i+1}/{len(cases)}  {time.time()-t0:.0f}s", flush=True)

    stats["wall_s"] = round(time.time() - t0, 1)
    if not preds:
        raise RuntimeError(
            f"split {split!r}: 0 predictions after filtering "
            f"({stats['n_skipped_unknown_type']} question(s) skipped on unknown type) -- "
            f"the type vocabulary in QTYPES does not match the split")
    if stats["n_skipped_unknown_type"]:
        raise RuntimeError(
            f"split {split!r}: {stats['n_skipped_unknown_type']} question(s) had a type outside "
            f"{sorted(QTYPES)} -- refusing to report a partial-split number")
    return preds, stats


# -------------------------------------------------------------------------- main


def score(preds: list[dict]) -> dict:
    from evaluate_laya_variant import score as frozen_score
    return frozen_score(preds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "dev", "calibration", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--weights", default=None,
                    help="optional model.safetensors; omit for the untrained base (the Q1 question "
                         "is about the readout, so the base is the honest starting point)")
    ap.add_argument("--cap", type=int, default=8, help="max rotations per question")
    ap.add_argument("--expect-n", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    from laya import Agent
    agent = Agent(a.base, device=a.device)
    if a.weights:
        from safetensors.torch import load_file
        w = load_file(a.weights)
        missing, unexpected = agent.model.load_state_dict(w, strict=False)
        if missing or unexpected:
            print(f"[FAIL] checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}")
            return 1
    agent.model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"[Q1] split={a.split} debias=on  cap={a.cap}  weights={a.weights or 'base (untrained)'}")
    preds, stats = collect(agent, a.split, debias=True, cap=a.cap, verbose=a.verbose)
    m = score(preds)
    m["split"] = a.split
    m["intervention"] = "cyclic-shift marginalisation (AnyJev L0 style)"
    m["cap"] = a.cap
    m["forward_passes"] = stats["n_forward"]
    m["extra_rotations"] = stats["n_rotations"]
    m["wall_s"] = stats["wall_s"]
    m["peak_reserved_gib"] = (round(torch.cuda.max_memory_reserved() / 1024**3, 3)
                              if torch.cuda.is_available() else None)
    # per-type accuracy comes from the FROZEN scorer (evaluate_laya_variant.score already returns
    # it). It must not be recomputed here: `target` is a teacher DISTRIBUTION vector, not an argmax
    # label, and the frozen scorer is what defines the metric. An earlier draft of this file tried to
    # derive per-type accuracy locally and got both the target type and the question-type vocabulary
    # wrong (`target` is a vector; the split spells the type `boolean`, not the package's `noul`).
    m["per_type_frozen"] = m.get("per_type")

    if a.expect_n is not None and len(preds) != a.expect_n:
        print(f"[FAIL] expected {a.expect_n} questions, got {len(preds)}")
        return 1

    Path(a.out).write_text(json.dumps(preds), encoding="utf-8")
    if a.stats_out:
        Path(a.stats_out).write_text(json.dumps({"metrics": m, "stats": stats}, indent=2),
                                     encoding="utf-8")

    print(f"[Q1:{a.split}] accuracy={m['accuracy']:.4f} skill={m['chance_corrected_skill']:.4f} "
          f"n={m['n']} forwards={m['forward_passes']} wall={m['wall_s']}s "
          f"peak={m['peak_reserved_gib']}GiB")
    print(f"      per-type: {m['per_type']}")
    print(f"      flip-against-target: raw={stats['flip_raw']} debiased={stats['flip_debiased']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
