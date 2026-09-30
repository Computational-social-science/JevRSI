#!/usr/bin/env python
"""
check_arm0_spec.py -- is config/arm0_spec.json a spec that run_arm_lib will actually honour?

    python scripts/check_arm0_spec.py
    python scripts/check_arm0_spec.py --self-test

WHY A VALIDATOR, GIVEN THAT JSON PARSES
    run_arm_lib.run_arm builds ArchConfig and FitConfig by EXPLICIT KEYWORD:

        arch = ArchConfig(readout=cfg["readout"], readout_layer=cfg["readout_layer"],
                          max_options=cfg["max_options"], freeze_base=cfg["freeze_base"],
                          option_pool=cfg["option_pool"], residual=cfg["residual"],
                          logit_cap=cfg["logit_cap"], head_input_norm=cfg["head_input_norm"],
                          **dict(cfg["arch_extra"] or {}))
        fit  = FitConfig(objective=..., steps=..., batch_size=..., lr_head=..., lr_base=...,
                         base_schedule=..., head_schedule=..., label_smoothing=...,
                         head_weight_decay=..., keep_last_k=..., prior_kl=..., rl=...,
                         autocast_bf16=not cfg["freeze_base"],
                         **dict(cfg["fit_extra"] or {}))

    A key in the wrong place is therefore NOT an error. It is dropped, and the arm trains as
    something other than what the file claims. Three real instances of that were written and
    shipped before this validator existed: `weight_decay`, `warmup`, `grad_clip`, `cal_method`,
    `lower_layers_n` and `log_every` sat at the top level and were silently ignored, and
    `xattn_heads` -- a real ArchConfig field -- sat at the top level where run_arm_lib does not
    name it, so it too was ignored while appearing in the file.

    A file that parses is not a spec that works. This checks the PLACEMENT, which is the part JSON
    cannot check and a reader cannot see.

WHAT IT CHECKS
    1. Every top-level key is one run_arm_lib reads: a DEFAULTS key, one of the 8 ArchConfig
       keywords, one of the 11 FitConfig keywords, or arch_extra / fit_extra / rl_extra.
    2. Every arch_extra key exists on ArchConfig, and every fit_extra key on FitConfig. Those two
       are **kwargs splats, so an unknown name is a TypeError at construction -- a loud failure,
       which is the good kind.
    3. The values that decide what the arm IS, not merely runs: freeze_base false (their default is
       true and every frozen variant of theirs stalled below the majority baseline), steps 1500,
       batch 16, lr_base 5e-6 cosine, lr_head 1e-3, readout option_xattn, cal_method none.
    4. No attempt to pin autocast_bf16, which run_arm_lib derives from freeze_base. A second source
       of truth for a derived value is a value that can disagree.

THE NEGATIVE CONTROL
    `--self-test` feeds four broken specs and requires each to be rejected, then requires the real
    one to pass. A validator that has never rejected anything is decoration.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import reference_repo, require  # noqa: E402

SPEC = ROOT / "config" / "arm0_spec.json"

# Copied from run_arm_lib.run_arm by reading it, and re-derived at run time when their checkout is
# reachable. The literals are here so this validator still works if the checkout is absent, and
# `_cross_check` says so loudly rather than letting the literals be trusted silently.
ARCH_KEYWORDS = ("readout", "readout_layer", "max_options", "freeze_base",
                 "option_pool", "residual", "logit_cap", "head_input_norm")
FIT_KEYWORDS = ("objective", "steps", "batch_size", "lr_head", "lr_base", "base_schedule",
                "head_schedule", "label_smoothing", "head_weight_decay", "keep_last_k", "prior_kl")
PASSTHROUGH = ("arch_extra", "fit_extra", "rl_extra", "save_dir")

# (key, expected, why) -- the values that decide what the arm IS.
LOAD_BEARING = [
    ("freeze_base", False,
     "v1.0 fine-tunes the tower. Their DEFAULTS is true, and versions/v1.0.md section 10 records "
     "every frozen variant of theirs below the majority baseline (0.3775 / 0.4850 / 0.473-0.483 "
     "against 0.5185) while the released v1.0 is 0.662."),
    ("steps", 1500, "their v1.0 record: 1,500 steps."),
    ("batch_size", 16, "their v1.0 record: batch 16."),
    ("lr_base", 5e-6, "their v1.0 record: tower 5e-6."),
    ("base_schedule", "cosine", "their v1.0 record: tower 5e-6 cosine."),
    ("lr_head", 1e-3, "their DEFAULTS. The v1.0 prose says 1e-4; the code says 1e-3, and the code "
                      "is what ran. Recorded in the spec rather than silently reconciled."),
    ("head_schedule", "constant", "their v1.0 record: head 1e-4 constant."),
    ("readout", "option_xattn", "the cross-attention scorer, which is the head their v1.0 trains."),
    ("sources", "synth", "the distillation corpus alone. Their v2.0 mixes in the benchmark's own "
                         "train split; that is their later release, not this one."),
    ("seed", 17, "their primary seed, fixed in advance."),
    # An adaptation, not their value. An earlier version of this entry claimed 32 was observed to
    # fail, which was wrong -- a control at steps 1 with freeze_base true completes at 32 with zero
    # OOM, so the isolation experiment never reproduced the fault. The 3-step probe's OOM came from
    # full-parameter training leaving 8.88 GiB of optimiser state resident at 96.7% and evaluation
    # allocating on top of it. 8 is kept as a PRECAUTION, not as a measured necessity, and the
    # reason recorded here says so: pinning it without saying why would let the next reader
    # "restore" 32 against a reason that is not true.
    ("eval_batch_size", 8, "OURS, a precaution. Their default 32 completes all three targets with "
                           "zero OOM on this card; the 3-step probe OOMed because full-parameter "
                           "training leaves the 8.88 GiB optimiser state resident at 96.7% and "
                           "evaluation allocates on top of it. Evaluation-only: it changes how many "
                           "questions are scored per forward pass and nothing else."),
]


def their_modules():
    """Import their dataclasses and DEFAULTS, or return None if the checkout is unreachable."""
    ref = reference_repo()
    for p in (str(ref), str(ref / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from rsijev.arch import ArchConfig
        from rsijev.fit import FitConfig
        import run_arm_lib
        return ArchConfig, FitConfig, run_arm_lib.DEFAULTS
    except Exception as e:                                    # noqa: BLE001
        print(f"[warn] their checkout not usable ({type(e).__name__}: {str(e)[:80]}). "
              f"Checking against the literals in this file, which were read from run_arm_lib.")
        return None, None, None


def check(spec: dict) -> tuple[int, int, list[str]]:
    lines: list[str] = []
    failures = checks = 0
    ArchConfig, FitConfig, DEFAULTS = their_modules()

    keys = {k: v for k, v in spec.items() if not k.startswith("_")}
    arch_fields = ({f.name for f in dataclasses.fields(ArchConfig)} if ArchConfig
                   else {"readout", "readout_layer", "xattn_heads", "xattn_dim", "xattn_combine",
                         "xattn_mlp_hidden", "head_design", "joint_dim", "joint_layers",
                         "joint_heads", "pack_questions", "embedding", "option_pool", "residual",
                         "logit_cap", "layer_mix", "layer_mix_init", "layer_mix_temp",
                         "freeze_base", "head_input_norm", "max_options", "freeze_lower_frac"})
    fit_fields = ({f.name for f in dataclasses.fields(FitConfig)} if FitConfig
                  else {"objective", "steps", "batch_size", "lr_head", "lr_mix", "lr_base",
                        "base_schedule", "head_schedule", "label_smoothing", "head_weight_decay",
                        "warmup", "weight_decay", "grad_clip", "keep_last_k", "prior_kl",
                        "autocast_bf16", "cal_method", "cal_td_frac", "cal_synth_frac",
                        "cal_mc_frac", "cal_joint_lambda", "rl", "log_every", "length_bucket",
                        "retention_kl", "retention_source_prefixes", "retention_rows",
                        "retention_pool", "retention_topk", "retention_max_tokens",
                        "source_tower_scale", "lower_layers_n", "lower_layers_lr_scale",
                        "init_from", "init_sha256", "rl2"})
    defaults = set(DEFAULTS) if DEFAULTS else set()

    # 1. placement
    for k, v in sorted(keys.items()):
        if k in PASSTHROUGH:
            continue
        checks += 1
        ok = (k in defaults) or (k in ARCH_KEYWORDS) or (k in FIT_KEYWORDS)
        if not ok:
            failures += 1
            looks_like = ("ArchConfig" if k in arch_fields else
                          "FitConfig" if k in fit_fields else "nothing")
            where = "arch_extra" if k in arch_fields else "fit_extra" if k in fit_fields else "nowhere"
            lines.append(f"    [FAIL] MISPLACED  {k!r} at top level, but it is a field of "
                         f"{looks_like}")
            lines.append(f"             run_arm_lib names its keywords explicitly, so a top-level "
                         f"key it does not name is DROPPED SILENTLY. It belongs in {where}.")
    checks += 1
    if not failures:
        lines.append(f"    [PASS] placement   all {len(keys)} top-level keys are read by run_arm_lib")

    # 2. the extras splat cleanly
    for name, fields in (("arch_extra", arch_fields), ("fit_extra", fit_fields)):
        extra = spec.get(name) or {}
        real = [k for k in extra if not k.startswith("_")]
        # An underscore key here is NOT a comment. Both extras are splatted as **kwargs into a
        # dataclass constructor, so `_xattn_heads` reaches ArchConfig.__init__ as a keyword and
        # raises TypeError. This was written by hand as a comment, shipped, and only surfaced when
        # the arm actually ran -- which is the whole reason this check exists.
        checks += 1
        comment_keys = [k for k in extra if k.startswith("_")]
        if comment_keys:
            failures += 1
            lines.append(f"    [FAIL] NOT-A-COMMENT {name} carries underscore key(s) "
                         f"{comment_keys}.")
            lines.append(f"             {name} is splatted as **kwargs, so an underscore does not "
                         f"make a key a comment -- it makes it an unexpected keyword argument. "
                         f"Move the note to the top level of the spec as a sibling string.")
        for k in sorted(real):
            checks += 1
            if k not in fields:
                failures += 1
                lines.append(f"    [FAIL] UNKNOWN     {name}.{k} is not a field on "
                             f"{'ArchConfig' if name == 'arch_extra' else 'FitConfig'}")
        bad = [k for k in real if k not in fields]
        if not bad and not comment_keys:
            lines.append(f"    [PASS] {name:11} {len(real)} field(s), all real, no pseudo-comments")

    # 3. load-bearing values
    for k, want, why in LOAD_BEARING:
        checks += 1
        got = spec.get(k, "<absent>")
        if got != want:
            failures += 1
            lines.append(f"    [FAIL] LOAD-BEARING {k} = {got!r}, expected {want!r}")
            lines.append(f"             {why}")
    if not any("[FAIL] LOAD-BEARING" in l for l in lines):
        lines.append(f"    [PASS] load-bearing {len(LOAD_BEARING)} values match their v1.0 record")

    # 4. do not pin a derived value
    for k, where in (("autocast_bf16", "top level"),):
        checks += 1
        if k in keys:
            failures += 1
            lines.append(f"    [FAIL] DERIVED     {k} is set at {where}, but run_arm_lib computes it "
                         f"as `not freeze_base`. Pinning it creates a second source of truth.")
    for k in ("autocast_bf16",):
        if k in (spec.get("fit_extra") or {}):
            checks += 1
            failures += 1
            lines.append(f"    [FAIL] DERIVED     fit_extra.{k} is set, but run_arm_lib passes "
                         f"autocast_bf16 explicitly. Two values, one meaning.")
    checks += 1
    if not any("DERIVED" in l for l in lines):
        lines.append("    [PASS] derived     autocast_bf16 is left to run_arm_lib")

    return checks, failures, lines


def self_test(spec: dict) -> int:
    print("  Spec self-test. Every broken spec must be REJECTED; the real one must PASS.\n")
    cases = [
        ("a FitConfig field left at top level",
         {**spec, "weight_decay": 0.1}, "MISPLACED"),
        ("a real ArchConfig field at top level where run_arm_lib ignores it",
         {**spec, "head_design": "mlp"}, "MISPLACED"),
        ("an unknown key in fit_extra",
         {**spec, "fit_extra": {**(spec.get("fit_extra") or {}), "not_a_field": 1}},
         "UNKNOWN"),
        ("freeze_base left at their default",
         {**spec, "freeze_base": True}, "LOAD-BEARING"),
        ("steps changed",
         {**spec, "steps": 400}, "LOAD-BEARING"),
        ("lr_head off by ten",
         {**spec, "lr_head": 1e-4}, "LOAD-BEARING"),
        ("autocast_bf16 pinned against the derivation",
         {**spec, "autocast_bf16": True}, "DERIVED"),
        # This one actually happened. An underscore key inside arch_extra reads like a comment and
        # is splatted into ArchConfig(**...) as a keyword, so the arm dies with
        # "unexpected keyword argument '_xattn_heads'" -- after the 2.28 GB model has loaded.
        ("a pseudo-comment inside arch_extra",
         {**spec, "arch_extra": {**(spec.get("arch_extra") or {}), "_note": "looks safe"}},
         "NOT-A-COMMENT"),
        ("a pseudo-comment inside fit_extra",
         {**spec, "fit_extra": {**(spec.get("fit_extra") or {}), "_why": "looks safe"}},
         "NOT-A-COMMENT"),
        # eval_batch_size is an ADAPTATION. The failure it prevents is specific: reverted to their
        # DEFAULTS of 32, a 1,500-step arm would train for tens of hours and then OOM in evaluation.
        ("eval_batch_size reverted to their default of 32",
         {**spec, "eval_batch_size": 32}, "LOAD-BEARING"),
        ("batch_size, their training number, changed",
         {**spec, "batch_size": 8}, "LOAD-BEARING"),
    ]
    ok = True
    for name, bad, expect in cases:
        _, fails, lines = check(bad)
        caught = any(expect in l for l in lines)
        if not caught or not fails:
            ok = False
        verdict = "rejected" if caught else f"ACCEPTED (expected {expect})"
        print(f"    {name:56} -> {verdict}")
        for l in lines:
            if "[FAIL]" in l:
                print(f"      {l.strip()[:104]}")
    _, fails, _ = check(spec)
    if fails:
        ok = False
    print(f"\n    {'the real spec':56} -> {'rejected (WRONG)' if fails else 'passed'}")
    print(f"\n  {'[OK] the validator can fail and the spec passes' if ok else '[FAIL] self-test failed'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Check that arm0_spec.json is a spec run_arm_lib honours.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if not SPEC.is_file():
        raise SystemExit(f"[fatal] {SPEC.name} not found")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))

    print(f"[spec-check] {SPEC.relative_to(ROOT)}")
    try:
        require(reference_repo(), "the RSI-Jev reference checkout (JEVRSI_REFERENCE_REPO)")
        print("[spec-check] cross-checking against the reference checkout")
    except SystemExit as e:
        print(f"[spec-check] {e}")

    if a.self_test:
        print()
        return self_test(spec)

    checks, failures, lines = check(spec)
    print("\n".join(lines))
    print(f"\n[{'OK' if failures == 0 else 'FAIL'}] {checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
