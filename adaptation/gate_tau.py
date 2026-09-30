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
    ap.add_argument("--tau-file", default=str(ROOT / "measurement" / "noise_floor.json"))
    ap.add_argument("--decision-metric", default="accuracy",
                    choices=["accuracy", "skill"],
                    help="must match what the loop is invoked with; tau is calibrated on this scale")
    a = ap.parse_args()

    tau_path = pathlib.Path(a.tau_file)
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
    if revoked:
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

    key = "tau_dev"
    if key not in thresholds:
        return fail(f"{CALIBRATION.name} has no thresholds.{key}",
                    f"available: {sorted(thresholds)}")
    registered = thresholds[key].get("value")
    if registered is None:
        return fail(f"thresholds.{key} has no 'value' (proportion) field")

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
    print(f"  file            {tau_path}")
    print(f"  single source   {CALIBRATION.name} -> thresholds.{key}")
    print(f"  decision split  {thresholds[key].get('use_for', '(unstated)')}")
    print(f"  decision metric {a.decision_metric}")
    print()
    print(f"  registered (proportion)   {registered!r}")
    print(f"  loaded     (proportion)   {got!r}")
    print(f"  in percentage points     {got*100:.6f} pp")

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
