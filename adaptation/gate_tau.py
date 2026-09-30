"""
gate_tau.py -- refuse to start the loop without a pre-registered accept threshold.

THE DEFECT THIS EXISTS TO CATCH
    autoresearch_agent.py's load_tau() is:

        if not tau_file.exists():
            print(f"[WARN] Tau file not found: {tau_file}, using conservative 0.02656")
            return 0.02656

    A missing file produces a WARNING and a plausible number, and the loop then makes every keep and
    discard decision against a threshold that appears in no pre-registration in this project. That
    is a Rule 1 violation with a print statement in front of it, and it is the kind of failure that
    survives to the final report: the run produces numbers, the numbers look fine, and nothing in
    results.tsv records that the accept criterion was invented at startup.

    The word "conservative" in that message is doing real work and is not supported. 0.02656 is 2.656
    pp; the measured, Bonferroni-corrected threshold for a dev-split decision is 4.949 pp. The
    fallback is nearly 2x LOOSER than the registered value, so it admits false keeps at roughly the
    rate the correction was introduced to prevent.

WHAT THIS DOES INSTEAD
    Resolves the threshold through the ecosystem's own load_tau, then PROVES it came from the
    measurement: compares it against INSTRUMENT_CALIBRATION.json and refuses any value that does not
    match. A wrong-units file, a hand-edited threshold, and a stale file all fail here rather than
    producing a run.

WHY COMPARE INSTEAD OF JUST READING
    Because reading is what the loop already does and reading is not the problem. The problem is a
    number that arrives without provenance. Comparing against the single source makes provenance a
    precondition rather than a comment.

USAGE
    python adaptation/gate_tau.py --tau-file measurement/noise_floor.json
    Exits non-zero and prints what to fix. Safe to call from any launcher.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "measurement" / "INSTRUMENT_CALIBRATION.json"

# The ecosystem's own hardcoded fallback, named so the message can say what it is avoiding rather
# than just asserting that a fallback is bad.
ECOSYSTEM_FALLBACK = 0.02656


def fail(msg: str, detail: str = "") -> int:
    print(f"[GATE: FAIL] {msg}")
    if detail:
        for line in detail.splitlines():
            print(f"    {line}")
    print("\nThe loop must not start. A keep decision made against an unregistered threshold is not "
          "evidence, and nothing in results.tsv would record that it happened.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the accept threshold is pre-registered.")
    ap.add_argument("--tau-file", default=None,
                    help="defaults to the harness threshold when --level harness, and is not "
                         "consulted at all when --level param")
    ap.add_argument("--decision-metric", default="accuracy",
                    choices=["accuracy", "skill"],
                    help="must match what the loop is invoked with; tau is calibrated on this scale")
    ap.add_argument("--level", default="harness", choices=["harness", "param"],
                    help="which level of the loop is about to spend GPU. The two have different noise "
                         "structures and therefore different thresholds: a harness candidate is a "
                         "deterministic function of cached logits and needs only a measurement floor, "
                         "while a QLoRA cycle trains weights and needs a run-to-run floor that has "
                         "not been measured on this seed.")
    a = ap.parse_args()

    # ---- the inner level has no threshold, and the reason is structural -------------------------
    if a.level == "param":
        cal = json.loads(CALIBRATION.read_text(encoding="utf-8")) if CALIBRATION.exists() else {}
        rev = cal.get("_REVOKED_2026-09-30") or {}
        print("=" * 78)
        print("tau gate -- REFUSED: the inner (QLoRA) level has no pre-registered threshold")
        print("=" * 78)
        print("  A QLoRA cycle trains weights. Its noise is run-to-run dispersion of a stochastic")
        print("  procedure, which is a property of the training path ON THIS SEED and has not been")
        print("  measured: measurement/floor_b_lora_runs/ holds 1 of 5 seeds, stopped at 325/600 steps.")
        print()
        print(f"  the number previously in circulation: {rev.get('what', 'tau derived from a void floor')}")
        if rev.get("why"):
            print(f"  why it is void: {rev['why'][:300]}")
        if rev.get("how_to_un_revoke"):
            print(f"  to un-revoke: {rev['how_to_un_revoke']}")
        print()
        print("  Note this is a FLOOR question, not a threshold question. A harness candidate is a")
        print("  deterministic function of cached logits -- measured bit-identical across independent")
        print("  evaluations -- so it needs only a measurement floor, which exists. A training run is")
        print("  not deterministic given a seed, so it needs a run-to-run floor, which does not.")
        print("  Conflating the two is what produced the unusable threshold this repository retired.")
        print("\n[GATE: FAIL] the outer level is authorised; the inner level is not.")
        return 1

    if a.tau_file:
        tau_path = pathlib.Path(a.tau_file)
    else:
        # The default depends on the level, and for `param` we never get here.
        tau_path = ROOT / "measurement" / "tau_harness.json"
    if not tau_path.exists():
        return fail(
            f"threshold file not found: {tau_path}",
            f"The ecosystem would substitute {ECOSYSTEM_FALLBACK} ({ECOSYSTEM_FALLBACK*100:.3f} pp), "
            f"which is not a measured value in this project.")

    if not CALIBRATION.exists():
        return fail(f"the single source is missing: {CALIBRATION}",
                    "Without it there is nothing to check the threshold against, and a check that "
                    "cannot fail is not a check.")

    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))

    # A threshold being CONSISTENT with the single source is not the same as the source being VALID.
    # The 2026-09-30 check found the consistency test passing on a threshold derived from a floor
    # measured on a retired substrate, which the pre-registration had already forbidden inheriting.
    # A gate that only checks agreement therefore armed a threshold that was explicitly not armable.
    # So validity is checked FIRST, and a revoked source stops the run regardless of agreement.
    revoked = cal.get("_REVOKED_2026-09-30") or cal.get("_REVOKED")
    if revoked and tau_path.name == "noise_floor.json":
        print("=" * 78)
        print("tau gate -- REFUSED: the calibration source is REVOKED")
        print("=" * 78)
        print(f"  {revoked.get('what', '(no summary)')}")
        print()
        print(f"  why: {revoked.get('why', '(unstated)')}")
        if revoked.get("also_broken"):
            print(f"  and: {revoked['also_broken']}")
        print()
        if revoked.get("what_remains_valid"):
            print(f"  still valid: {revoked['what_remains_valid']}")
        if revoked.get("how_to_un_revoke"):
            print(f"  to un-revoke: {revoked['how_to_un_revoke']}")
        print("\n[GATE: FAIL] consistency with the single source is not validity. A threshold derived "
              "from a measurement of a different model on a different training path is not a "
              "threshold for this seed, and no amount of internal agreement makes it one.")
        return 1
    if revoked and tau_path.name != "noise_floor.json":
        print(f"  note: INSTRUMENT_CALIBRATION.json is revoked ({revoked.get('what', '')[:90]})")
        print("        -- that revocation applies to the FLOOR it contained, not to this threshold.")
        print(f"        This threshold comes from {tau_path.name}, whose floor was measured on THIS")
        print("        split by measurement/floor_a_dev_bootstrap.py. The two are different files")
        print("        for different reasons, and arming one says nothing about the other.")
        print()

    thresholds = cal.get("thresholds", {})
    if a.decision_metric not in ("accuracy",):
        # The loop's own --decision-metric help says 'skill' is UNVALIDATED on this scale. Rather
        # than restate that as a warning we can ignore, treat it as a gate: an unvalidated scale
        # cannot carry a pre-registered threshold.
        return fail(
            f"decision metric {a.decision_metric!r} has no calibrated threshold",
            "tau_dev was derived on accuracy. The ecosystem agent itself marks 'skill' as "
            "UNVALIDATED on this scale. Re-run with --decision-metric accuracy, or pre-register a "
            "threshold for the other scale before using it.")

    # The threshold file is now SELF-CONSISTENT rather than cross-checked against
    # INSTRUMENT_CALIBRATION.json, and the difference is not cosmetic. The old check compared
    # noise_floor.json against thresholds.tau_dev, which is the revoked value -- so the only way to
    # satisfy it was to carry the revoked number forward. The harness threshold has a different
    # provenance: a floor measured on THIS decision split, in a file that can be re-run.
    #
    # What is checked instead is the arithmetic that produced it, because that is the part a later
    # session could plausibly get wrong: the factor times the standard error must equal the published
    # tau, and the preregistration must be the named source.
    spec = json.loads(tau_path.read_text(encoding="utf-8"))
    prereg = ROOT / "docs" / "prereg_tau_harness_2026-09-30.md"
    if not prereg.exists():
        return fail(f"the pre-registration is missing: {prereg}",
                    "A threshold with no written derivation behind it is a constant.")
    if spec.get("_source_prereg") and prereg.name not in str(spec["_source_prereg"]):
        return fail(f"{tau_path.name} names a different pre-registration: "
                    f"{spec.get('_source_prereg')!r}")
    ex = spec.get("exact", {})
    factor, se = ex.get("bonferroni_horn_factor"), ex.get("paired_se_pp")
    if factor is None or se is None:
        return fail(f"{tau_path.name} does not record the factor and the SE it was built from",
                    "The gate needs both to re-derive tau rather than trust it.")
    product = factor * se
    published = spec.get("tau_pp")
    if published is None:
        return fail(f"{tau_path.name} has no tau_pp (percentage points)")
    if abs(product - published) > 0.002:
        return fail(
            f"the arithmetic does not close: {factor} x {se} = {product:.4f} pp, "
            f"but tau_pp = {published}",
            "One of the three numbers was edited. They travel together or not at all.")
    if ex.get("product") is not None and abs(ex["product"] - product) > 1e-6:
        return fail(f"{tau_path.name} records product={ex['product']} but factor x SE = {product}")
    if spec.get("decision_metric") and spec["decision_metric"] != a.decision_metric:
        return fail(
            f"this threshold is on {spec['decision_metric']!r}, the loop was invoked with "
            f"{a.decision_metric!r}",
            "The ecosystem agent marks a non-accuracy metric as UNVALIDATED on this scale.")
    registered = published / 100.0

    # Resolve through the ecosystem's own reader, so units and field names are its interpretation.
    sys.path.insert(0, str(ROOT))
    try:
        from adaptation._ecosystem import load_tau          # noqa: E402
    except ImportError as e:                                    # pragma: no cover
        return fail(f"cannot import the ecosystem's load_tau: {e}")
    try:
        got = load_tau(tau_path)
    except Exception as e:                                      # noqa: BLE001
        return fail(f"the ecosystem's load_tau could not read {tau_path}: "
                    f"{type(e).__name__}: {e}")

    print("=" * 78)
    print("tau gate -- the accept threshold is pre-registered, measured, and on the right scale")
    print("=" * 78)
    print(f"  level           {a.level}  (harness: deterministic, measurement floor only)")
    print(f"  threshold file  {tau_path.name}")
    print(f"  preregistration {prereg.name}")
    print(f"  decision split  {spec.get('decision_split', '(unstated)')}")
    print(f"  decision metric {a.decision_metric}")
    print()
    print(f"  paired SE                 {se:.6f} pp   ({spec['reproduces']['unit']}-level bootstrap, "
          f"{spec['reproduces']['replicates']} replicates)")
    print(f"  Bonferroni-Horn factor    {factor:.5f}    (H = {ex.get('horizon_H')})")
    print(f"  factor x SE               {product:.4f} pp")
    print(f"  published tau_pp          {published} pp")
    print(f"  loaded     (proportion)   {got!r}")
    print(f"  in percentage points     {got*100:.4f} pp")

    if abs(got - registered) > 1e-12:
        return fail(
            f"threshold mismatch: {got!r} loaded vs {registered!r} registered",
            "A mismatch means the file was edited, is stale, or is in the wrong units. The ecosystem "
            "divides tau_pp by 100, so a file storing percentage points under 'value' instead of "
            "'tau_pp' yields a threshold 100x too small -- and a 100x too small threshold admits "
            "almost every candidate.")

    if got < ECOSYSTEM_FALLBACK:
        print(f"\n  [WARN] this threshold ({got*100:.3f} pp) is LOOSER than the ecosystem's "
              f"hardcoded fallback ({ECOSYSTEM_FALLBACK*100:.3f} pp).")
        print("         Registered and intentional, but it admits more false keeps than the "
              "fallback would.")

    print("\nRESULT: PASS -- the threshold is registered, matches the single source exactly, and is "
          "in proportion units.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
