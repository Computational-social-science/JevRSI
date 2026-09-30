"""
harness_routes.py -- the OUTER level's search space, expressed in the ecosystem's own knob format.

THE PROBLEM THIS SOLVES
    agent-jev/scripts/autoresearch_agent.py has a complete, well-tested cycle -- propose, train,
    evaluate, accept or reject, persist -- and a search space of nine knobs that all edit ONE file,
    `agentjev/train.py`, all of them LoRA hyper-parameters:

        lora_rank 32 / 64, lr 2e-4 / 1e-4, weight_decay, grad_accum, warmup_ratio,
        brier weight, margin weight

    Every one of them costs a full training run: 2.11 h measured on this host, against a 40-minute
    timeout in the agent. So the agent as written can only execute the proposal's P1 route, and the
    proposal's PRIMARY route -- "harness evolution ... the vast majority of cycles" -- is not
    reachable through it at all.

    Adding harness knobs to that list is not sufficient, for a structural reason rather than a
    scheduling one: the agent runs TRAIN_CMD unconditionally, so a candidate that changes the harness
    would still pay 2.11 h to produce a checkpoint it does not use. The proposal's Phase 3 is explicit
    -- "Harness-only: skip training, evaluate immediately" -- and a harness candidate that trains
    first is not a harness candidate.

WHAT THIS MODULE IS
    A drop-in for the agent's `knobs` list plus the one hook that lets a route declare it needs no
    training. It does not reimplement the cycle: propose, evaluate, accept, persist, reset, the
    crash circuit breaker and the trajectory log all stay in the agent. This file supplies the
    candidate space and the training decision, which is the only part the proposal requires to be
    different.

    The knob records carry the agent's own four fields -- name, description, pattern, replacement --
    plus `level` and `axis`. `level` is what the new branch reads; the rest are the agent's.

WHY THE PATTERNS TARGET A CONFIG FILE AND NOT CODE
    A harness mutation is a choice of calibration stage, not an edit to a program. Putting the
    mutable surface in a JSON file that the agent rewrites means every candidate is inspectable on
    disk, revertible with `git reset`, and reviewable as a diff -- the same properties that make the
    agent's own train.py approach auditable. Editing code to select a calibration stage would make
    the search space a diff over control flow, which is neither reviewable nor revertible in the same
    way.

THE AXIS, AND WHY IT IS RECORDED BUT NOT GATING
    The agent accepts on ONE declared metric with ONE threshold (its L391-448). `axis` here is
    reporting, not gating: the proposal's multi-objective archive is a reporting structure, and
    knowing which axis a candidate was aimed at is what makes the archive interpretable afterwards.
    A calibration candidate that improves ECE and does not move accuracy is TIED and rejected by the
    agent's own rule -- correctly, because the decision metric did not move. The axis label says why
    the candidate existed, so the negative result is legible instead of looking like a mystery.
"""
from __future__ import annotations

import json
import pathlib

# The file the harness knobs rewrite. Relative to the subject repository root.
HARNESS_CONFIG = "agentjev/configs/harness_search.json"

# Axis vocabulary, from the proposal's four-axis frame (section 5.1) plus the robustness family the
# proposal names in section 6.2. `calibration` candidates that are monotone cannot move accuracy by
# construction, which is a property of the family and is recorded on the knob so a reader does not
# have to rediscover it from a null result.
AXES = ("calibration", "pipeline", "backend", "robustness", "intelligence")


def default_harness_config() -> dict:
    """The starting configuration: the proposal's section 3.2 seed harness.

    L1 per-type temperature scaling is the shipped baseline every candidate is measured against, so
    it is the incumbent and not a candidate. Nothing here is fitted; the temperatures are the ones the
    seed checkpoint ships with, and a candidate that proposes new ones is proposing a measurement.
    """
    return {
        "_what_this_is": "Mutable surface of the OUTER (harness) level. The agent rewrites one field "
                         "per candidate and re-evaluates on cached logits; no training is involved.",
        "_baseline": "L1 per-type temperature scaling, as shipped in the seed checkpoint",
        "stage": "scalar_temperature",
        "params": {
            "temperatures": {
                "boolean": 1.0717734625362931,
                "choice": 1.0352649238413776,
                "score": 1.0717734625362931,
            }
        },
        "history": [],
    }


def load_harness_config(repo_root: pathlib.Path) -> dict:
    p = repo_root / HARNESS_CONFIG
    if not p.exists():
        cfg = default_harness_config()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg
    return json.loads(p.read_text(encoding="utf-8"))


def harness_knobs() -> list[dict]:
    """The outer level's search space, in the agent's knob format.

    Two families, and the split is not cosmetic:

      CALIBRATION (isotonic / Platt) -- post-hoc maps of the probability or the top-1 confidence.
          Isotonic and Platt on top-1 confidence are MONOTONE, so they cannot change an argmax and
          therefore cannot move the agent's decision metric. They are included anyway, and marked,
          because the proposal names them as P0 and because a family that provably cannot help is a
          result worth having in the archive rather than a gap in it. Platt on the LOGITS (a
          per-type affine map in logit space) is NOT monotone and is a genuine candidate.

      ROBUSTNESS (option-order) -- the proposal's section 6.2 order-invariance family. The candidate
          head is permutation-equivariant by construction, so this is expected to be a no-op; it is
          included as a NEGATIVE CONTROL, and if it moves the score the equivariance claim is wrong
          and every other number computed on this model is suspect. That is worth one cycle.

    `pattern` and `replacement` rewrite the config's `stage` field, which is the single switch the
    harness evaluator reads. `params` is carried alongside for the stage that needs it.
    """
    return [
        # ---- calibration: non-monotone, can move the decision metric -------------------------
        {
            "name": "harness_platt_per_type",
            "description": "Per-type Platt scaling in logit space (non-monotone; can move argmax)",
            "level": "harness", "axis": "calibration", "trains": False,
            "pattern": r'("stage"\s*:\s*)"[^"]*"',
            "replacement": r'\1"platt_per_type"',
            "params": {},
        },
        {
            "name": "harness_vector_temperature",
            "description": "Per-dimension temperature (non-monotone; can move argmax)",
            "level": "harness", "axis": "calibration", "trains": False,
            "pattern": r'("stage"\s*:\s*)"[^"]*"',
            "replacement": r'\1"vector_temperature"',
            "params": {"dims": "per_candidate"},
        },
        {
            "name": "harness_non_monotone_window",
            "description": "gamma-parameterised mid-confidence window (non-monotone)",
            "level": "harness", "axis": "calibration", "trains": False,
            "pattern": r'("stage"\s*:\s*)"[^"]*"',
            "replacement": r'\1"non_monotone_window"',
            "params": {"gamma": 0.5},
        },
        # ---- calibration: monotone, CANNOT move the decision metric ----------------------------
        {
            "name": "harness_isotonic_top1",
            "description": "Isotonic map on top-1 confidence. MONOTONE: cannot change argmax, so the "
                           "agent's accuracy rule will report TIED. Included because the proposal "
                           "names it P0 and because a family that provably cannot help belongs in "
                           "the archive as a result, not omitted as a gap. Measured 2026-09-30: "
                           "ECE 0.1240 -> 0.0408, accuracy +0.00 pp.",
            "level": "harness", "axis": "calibration", "trains": False,
            "pattern": r'("stage"\s*:\s*)"[^"]*"',
            "replacement": r'\1"isotonic_top1"',
            "params": {},
        },
        # ---- robustness: expected no-op, run once as a negative control ------------------------
        {
            "name": "harness_option_permutation",
            "description": "Permute the candidate order before scoring. The candidate head is "
                           "permutation-equivariant by construction, so this is a NEGATIVE CONTROL: "
                           "if it moves the score, the equivariance claim is wrong and every other "
                           "number here is suspect.",
            "level": "harness", "axis": "robustness", "trains": False, "negative_control": True,
            "pattern": r'("stage"\s*:\s*)"[^"]*"',
            "replacement": r'\1"option_permutation"',
            "params": {"seed": 0},
        },
    ]


def param_knobs_from_agent(agent_knobs: list[dict]) -> list[dict]:
    """Tag the agent's own knobs as the inner level, without editing the agent.

    The agent's nine knobs are all LoRA hyper-parameters and all require a training run. Tagging
    them here means the branch can tell the two levels apart without the agent knowing the
    difference exists, which is what keeps this an extension rather than a fork.
    """
    out = []
    for k in agent_knobs:
        k2 = dict(k)
        k2.setdefault("level", "param")
        k2.setdefault("axis", "intelligence")
        k2.setdefault("trains", True)
        out.append(k2)
    return out
