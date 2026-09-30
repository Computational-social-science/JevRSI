"""test_meaning_boundary.py -- the meaning-boundary TRIGGER must fire where it should.

Why this file exists: the meaning-boundary guardrail is one of JevRSI's claimed contributions
(see docs/JevRSI_target_backward_derivation.md section 4.3). A guardrail that is documented but
untested is a guardrail that may silently not exist. This test pins its behaviour at the exact
published boundaries, and includes a NEGATIVE CONTROL built from this project's own prior-work
number -- if the guardrail cannot flag our own 0.8085 as teacher-quirk fitting, it is decorative.

Run: python measurement/test_meaning_boundary.py
Exit code 0 = all assertions hold.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import the driver module WITHOUT running main(). run_hillclimb.py needs measurement/ on the
# path (it imports `provenance`), which the insert above supplies.
SRC = (HERE / "run_hillclimb.py").read_text(encoding="utf-8")
NS = {"__file__": str(HERE / "run_hillclimb.py"), "__name__": "rh_under_test"}
exec(compile(SRC.split("def main(")[0], "run_hillclimb.py", "exec"), NS)
meaning_zone = NS["meaning_zone"]
REF_TEACHER_SELFAGREE = NS["REF_TEACHER_SELFAGREE"]
REF_SATURATION = NS["REF_SATURATION"]
REF_FACTOR_CEILING = NS["REF_FACTOR_CEILING"]
REF_OBSERVED_SOTA = NS["REF_OBSERVED_SOTA"]

FAILS = []


def check(v, want, label):
    got = meaning_zone(v)
    ok = got == want
    if not ok:
        FAILS.append(f"{label}: meaning_zone({v}) = {got}, want {want}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {v:.4f}  ->  {got:8s}  {label}")


print("== published reference constants (must match their sources verbatim) ==")
for name, val, want in [("REF_PRIOR", NS["REF_PRIOR"], 0.470),
                        ("REF_FACTOR_CEILING", REF_FACTOR_CEILING, 0.704),
                        ("REF_TEACHER_SELFAGREE", REF_TEACHER_SELFAGREE, 0.735),
                        ("REF_SATURATION", REF_SATURATION, 0.750),
                        ("REF_OBSERVED_SOTA", REF_OBSERVED_SOTA, 0.799)]:
    ok = abs(val - want) < 1e-9
    if not ok:
        FAILS.append(f"{name} = {val}, want {want}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {name} = {val}")

print("\n== boundary behaviour: the zone must change ONLY past the boundary ==")
check(0.3500, "below-self-agreement", "our arm, untrained")
check(0.5310, "below-self-agreement", "our arm, trained (old protocol)")
check(0.5567, "below-self-agreement", "our arm, learning-curve plateau")
check(0.6460, "below-self-agreement", "specialist SOTA (ModernBERT-base)")
check(0.7040, "below-self-agreement", "factor ceiling -- AT the boundary")
check(0.7350, "below-self-agreement", "teacher self-agreement -- AT the boundary")
check(0.7351, "above-self-agreement", "just past teacher self-agreement")
check(0.7500, "above-self-agreement", "saturation point -- AT it, not past")
check(0.7501, "above-saturation", "just past saturation")
check(0.7680, "above-saturation", "laya-typed-decisions / meraGPT, measured in this repo")
check(0.7990, "above-saturation", "AnyJev L2 SOTA -- AT the boundary, not past")
check(0.7991, "above-observed-sota", "just past the published SOTA")

print("\n== the reclassification that motivated these constants ==")
print("   0.735 is NOT a ceiling: on this same 400-case split, public measurements reach 0.768 and")
print("   0.799. A system that predicts the scenario is not bound by the teacher's self-agreement, so")
print("   the guardrail must FLAG those scores for a mechanism report, not refuse them.")
for score, why in [(0.7680, "laya-typed-decisions"), (0.7860, "AnyJev L2 Qwen3-4B"),
                   (0.7990, "AnyJev L2 Qwen3-30B-A3B")]:
    z = meaning_zone(score)
    ok = z != "below-self-agreement"
    if not ok:
        FAILS.append(f"{why} at {score} classified {z} -- the guardrail would refuse a published result")
    print(f"  {'ok  ' if ok else 'FAIL'}  {score} -> {z:20s} ({why})")

print("\n== NEGATIVE CONTROL: this project's own prior-work numbers ==")
print("   These MUST be flagged. Reaching them is legitimate; reporting a bare score is not.")
check(0.7680, "above-saturation", "leaderboard no.1 (general, zero-shot)")
check(0.7690, "above-saturation", "Laya typed-decisions, measured in this repo")
check(0.7925, "above-saturation", "AgentJev-0.6B, published figure")
# 0.8085 exceeds the observed SOTA (0.799), so above-observed-sota is the CORRECT zone for it. The
# negative control still fires -- quirk_warning is what the test is really asserting, and the
# end-to-end case below checks that an accepted generation there is still marked.
check(0.8085, "above-observed-sota", "AgentJev-0.6B, re-measured in this repo (beats 0.799 SOTA)")

print("\n== end-to-end: an accepted generation in the quirks zone must be marked ==")
incumbent, acc, tau = 0.7000, 0.8085, 0.0495
delta = acc - incumbent
accepted = delta > tau
zone = meaning_zone(acc)
quirk_warning = bool(accepted and zone != "below-self-agreement")
print(f"  incumbent={incumbent:.4f}  candidate={acc:.4f}  delta={delta*100:+.2f} pp  tau={tau*100:.2f} pp")
print(f"  accepted={accepted}  zone={zone}  quirk_warning={quirk_warning}")
if not (accepted and quirk_warning):
    FAILS.append("end-to-end: an accepted quirks-zone generation was not marked")

print("\n== control: a legitimate accept must NOT be marked ==")
incumbent, acc, tau = 0.5567, 0.6460, 0.0495
delta = acc - incumbent
accepted = delta > tau
zone = meaning_zone(acc)
quirk_warning = bool(accepted and zone != "below-self-agreement")
print(f"  incumbent={incumbent:.4f}  candidate={acc:.4f}  delta={delta*100:+.2f} pp  tau={tau*100:.2f} pp")
print(f"  accepted={accepted}  zone={zone}  quirk_warning={quirk_warning}")
if not (accepted and not quirk_warning):
    FAILS.append("control: a legitimate task-zone accept was wrongly marked")

print()
if FAILS:
    print(f"[FAIL] {len(FAILS)} assertion(s) failed:")
    for f in FAILS:
        print(f"    - {f}")
    sys.exit(1)
print("[OK] meaning-boundary guardrail behaves as specified (13 boundary cases + 2 end-to-end)")
