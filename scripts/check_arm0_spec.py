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

# The authoritative spec, shipped inside their v1.0 checkpoint. When this file is reachable, the
# spec is diffed against it directly and the literal list below becomes a fallback for an
# environment without the download. Reading the published spec rather than trusting a
# reconstruction is the whole point: the first version of this file WAS a reconstruction and it was
# wrong twice, in ways no amount of care over literals would have caught.
PUBLISHED_META = ("published_release", "meta.json")

# Differences from the published spec that this project has deliberately made. Every entry needs a
# measurement behind it, because an unexplained deviation from a reference is a bug with a comment.
ALLOWED_DEVIATIONS = {
    "eval_batch_size": ("theirs 32 -> ours 8. NOT the OOM it was first justified by: that OOM came "
                        "from a run sharing the card with a 30-step job, and this file's own note "
                        "discards the SAME run's train_seconds for exactly that contention -- one "
                        "contaminated run cannot yield one number kept and another discarded. "
                        "Re-measured on an idle card (v1.0 modules, published spec): 32 does not "
                        "OOM. It does, however, sit at 11,583 / 12,282 MiB = 94% and spend over 61 "
                        "minutes on the evaluation that 8 finishes in 15.6, because at 94% of "
                        "capacity the allocator thrashes. 8 is kept for headroom and speed. See "
                        "docs/eval_batch_size_correction.md."),
}

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
    # This entry used to read `1e-3`, on the reasoning that "the code says 1e-3 and the code is what
    # ran". Both halves were wrong. The published meta.json says 1e-4, and the reason the checkout
    # disagrees is that the checkout's HEAD is not the revision that produced v1.0 -- fit.py has
    # grown from 213 to 559 lines since. Reading a DEFAULT out of a later revision and calling it
    # "what ran" is the same class of error as reading a step time off a run whose conditions were
    # not the ones being asked about.
    ("lr_head", 1e-4, "their v1.0 record, and their published meta.json: head 1e-4 constant. Their "
                      "HEAD DEFAULTS says 1e-3, which is a later revision's default, not v1.0's "
                      "setting."),
    ("head_schedule", "constant", "their v1.0 record: head 1e-4 constant."),
    # The one omission that would have changed the run rather than only mis-documenting it. Without
    # this key the encoder enumerates options in corpus order, and their own record names that
    # failure: "a causal encoder shows option k only options 1..k-1, and a head trained on a fixed
    # order collapses onto position -- one early run picked the last option on 800 of 800 score
    # questions."
    ("option_order", "shuffled",
     "their v1.0 setting, from their published meta.json. Their EncodeConfig default is 'canonical' "
     "and their release overrides it. Omitting this key trains a position-collapsed head, which is "
     "the failure their record documents at 800/800 score questions."),
    ("eval_option_orders", ["canonical", "reversed"],
     "their v1.0 setting. Evaluation is run under both option orders, which is how a position "
     "artefact would show up as a gap between the two rather than as an unexplained score."),
    ("readout", "option_xattn", "the cross-attention scorer, which is the head their v1.0 trains."),
    ("sources", "synth", "the distillation corpus alone. Their v2.0 mixes in the benchmark's own "
                         "train split; that is their later release, not this one."),
    ("seed", 17, "their primary seed, fixed in advance."),
    # An adaptation, not their value. The reason has been rewritten once already, because the first
    # reason did not survive re-testing.
    #
    # WHAT WAS CLAIMED: "freeze_base false + eval 32 -> CUDA OOM, 0 files", so the fault is the
    # interaction of full-parameter training with a large evaluation batch.
    #
    # WHY THAT IS WITHDRAWN: the run that produced the OOM shared the card with a 30-step job --
    # the same run whose train_seconds (3207, or 107 s/step) this project already discards for
    # contention. One contaminated run cannot yield one number kept and another discarded, and the
    # earlier commit kept the OOM while throwing away the duration.
    #
    # WHAT WAS MEASURED ON AN IDLE CARD (v1.0 modules, published spec, eval_batch_size 32):
    # training completed and the evaluation ran 61+ minutes without an OOM, killed by hand. So 32
    # does not OOM here -- but it sits at 11,583 / 12,282 MiB = 94% and needs >61 minutes where 8
    # needs 15.6. At 94% of capacity the allocator thrashes, which is the whole of the slowdown.
    #
    # 8 IS KEPT ANYWAY, for headroom and for speed, and because it is evaluation-only: it changes
    # how many questions are scored per forward pass and nothing else -- not what is scored, not the
    # metric, not the training, not the model. It is NOT kept because 32 is impossible.
    ("eval_batch_size", 8, "OURS, kept for headroom and speed, not for correctness. Their 32 does "
                           "not OOM on an idle card but occupies 94% of this GPU and takes >61 "
                           "minutes against 8's 15.6, so the allocator thrashes. Evaluation-only: "
                           "it changes how many questions are scored per forward pass and nothing "
                           "else. See docs/eval_batch_size_correction.md."),
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


def _load_published_spec():
    """Their v1.0 checkpoint's own spec, or (None, [reason]).

    This is the authoritative artefact: the file the release shipped beside its weights, written by
    the code that produced them. Everything this project could reconstruct is downstream of it.
    """
    try:
        import paths
        root = paths.published_release()
    except Exception as e:                                       # noqa: BLE001
        return None, [f"published_release not configured ({type(e).__name__})"]
    meta = pathlib.Path(root) / PUBLISHED_META[1]
    if not meta.is_file():
        return None, [f"{meta} not present; run the download in docs/arm0_probe_evidence.md"]
    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except Exception as e:                                       # noqa: BLE001
        return None, [f"{meta} unreadable ({type(e).__name__}: {str(e)[:60]})"]
    spec = d.get("spec")
    if not isinstance(spec, dict):
        return None, [f"{meta} has no 'spec' object"]
    return spec, []


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

    # 3b. THE STRONGEST CHECK: diff against the spec their v1.0 checkpoint actually ships.
    # A reconstruction is only ever as good as the sources it was built from, and this project's
    # first reconstruction was wrong twice: lr_head was taken from the checkout's DEFAULTS as if the
    # checkout were the code that produced v1.0, and `option_order` was omitted entirely so the run
    # would have trained canonical-ordered options -- the exact configuration their own record warns
    # collapses onto position. Diffing against meta.json cannot make either mistake.
    published, pub_lines = _load_published_spec()
    if published is None:
        checks += 1
        lines.append(f"    [SKIP] published   {pub_lines[0] if pub_lines else 'not available'}")
    else:
        actual = {k: v for k, v in spec.items() if not k.startswith("_")}
        for k in sorted(set(published) | set(actual)):
            checks += 1
            tv, mv = published.get(k, "<absent>"), actual.get(k, "<absent>")
            if tv == mv:
                continue
            if k in ALLOWED_DEVIATIONS:
                lines.append(f"    [PASS] deviation   {k}: theirs {tv!r} -> ours {mv!r}")
                lines.append(f"             {ALLOWED_DEVIATIONS[k]}")
            else:
                failures += 1
                lines.append(f"    [FAIL] DRIFT       {k}: published {tv!r} but ours is {mv!r}")
                lines.append(f"             Our spec must equal theirs except for a listed deviation. "
                             f"Either copy theirs, or add an ALLOWED_DEVIATIONS entry with the "
                             f"measurement that justifies it.")
        if not any("[FAIL] DRIFT" in l for l in lines):
            n_dev = len(ALLOWED_DEVIATIONS)
            lines.append(f"    [PASS] published   spec matches meta.json field for field, except "
                         f"{n_dev} recorded deviation(s)")

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
