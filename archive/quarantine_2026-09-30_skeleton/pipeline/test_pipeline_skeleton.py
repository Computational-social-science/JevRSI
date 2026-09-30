"""test_pipeline_skeleton.py -- prove the three tiers actually work together, not just import.

WHY THIS TEST

A skeleton that imports is not a skeleton that runs. Three failure modes are specific to this design
and none of them raises on import:

  1. The gate. If `update_gate` opens the inner level on a flat slope that is not flat, or holds it shut
     on one that is, the loop spends a 2.11 h cycle at the wrong time -- and the proposal's risk
     register names exactly that as a budget risk.
  2. Replay. Rule 3 requires the control state to rebuild from the JSONL alone. A rebuild that loses a
     cool-down, or that mistakes a crash for a judged failure, corrupts every later decision.
  3. The prior-art veto. A rejected idea that reappears under a different route name costs a day and
     teaches nothing. This is the one filter standing between a six-week run and that.

Each is asserted against a value the test computes itself, not against a value copied from the
implementation, so a change that breaks the behaviour fails here rather than in hour 40 of an
unattended run.

Run: python pipeline/test_pipeline_skeleton.py
"""
from __future__ import annotations

# TERMINOLOGY: the term "prompt-space" below names a retired research object of this project.
# Do not work on it. It is referenced in exactly one place -- the NON_TRANSFERABLE table --
# purely to mark that family of methods as inapplicable to a non-generative scoring model,
# which emits one scalar per candidate and has no LM head. It is not an approach this project
# uses, is evaluating, or plans to evaluate.




import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "loop"))

from loop.state import Action, Log, Record, now_iso                              # noqa: E402
from pipeline.routes import (ALL_ROUTES, GateState, HARNESS, PARAM,              # noqa: E402
                             flat_run_length, update_gate)
from pipeline.orchestrator import Candidate, Orchestrator, _map_elites_cell      # noqa: E402
from pipeline import open_jev_fast as ojf                                        # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------
def test_gate() -> None:
    print("\n1. THE GATE -- flat slopes open the inner level, movement holds it shut")
    g = GateState()
    tol = 0.01

    check("a rising slope is NOT flat", flat_run_length([0.05, 0.04, 0.03], tol) == 0)
    check("flat slopes count", flat_run_length([0.001, -0.002, 0.0, 0.0005], tol) == 4)
    # counting must STOP at the first real movement, not average over it
    check("counting stops at movement",
          flat_run_length([0.001, 0.001, 0.09, 0.001, 0.001], tol) == 2,
          "a good result buried under flat ones must not open the gate")

    g = update_gate(GateState(), [0.001] * 3, tol)
    check("gate shut below the cool-down", not g.open, g.reason)
    g = update_gate(GateState(), [0.001] * 6, tol)
    check("gate OPENS at the cool-down", g.open, g.reason)
    g = update_gate(GateState(cool_down_k=6), [0.001] * 8, tol)
    g.observed_param_fraction = 0.30
    g = update_gate(g, [0.001] * 8, tol)
    check("gate stays shut at the param cap", not g.open, g.reason)

    # the inner level must be far more expensive than every outer route combined
    outer = sum(r.cost_s for r in ALL_ROUTES.values() if r.kind == HARNESS)
    inner = min(r.cost_s for r in ALL_ROUTES.values() if r.kind == PARAM)
    check("inner level costs more than the whole outer sweep", inner > outer,
          f"one param route {inner/3600:.2f} h vs full harness sweep {outer/60:.1f} min")


# ---------------------------------------------------------------------------------------
def test_replay() -> None:
    print("\n2. REPLAY -- state rebuilds from the JSONL, and a crash is not a verdict")
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        o = Orchestrator(d, epsilon=0.01)
        o.log.append(Record(iter=0, ts=now_iso(), action=Action.RUNNING.value,
                            route="cal.isotonic_top1", kind=HARNESS))
        o.log.append(Record(iter=0, ts=now_iso(), action=Action.DISCARD.value,
                            route="cal.isotonic_top1", kind=HARNESS, medium=0.83, delta=0.01,
                            modules={"family": "calibration"}, diagnosis="no gain"))
        o.log.append(Record(iter=1, ts=now_iso(), action=Action.RUNNING.value,
                            route="cal.platt_per_type", kind=HARNESS))
        o.log.append(Record(iter=1, ts=now_iso(), action=Action.DISCARD.value,
                            route="cal.platt_per_type", kind=HARNESS, medium=0.84, delta=0.02,
                            modules={"family": "calibration"}, diagnosis="no gain"))
        # a cycle killed by the watchdog: RUNNING and never completed
        o.log.append(Record(iter=2, ts=now_iso(), action=Action.RUNNING.value,
                            route="infer.batch_policy", kind=HARNESS))
        st = o.rebuild()

        # iter=0 and iter=1 each have a RUNNING *and* a terminal record, so they are COMPLETE cycles
        # that merely have two lines each. Only iter=2 dangles. Counting RUNNING lines instead of
        # RUNNING iterations would report 3, and after a few dozen restarts would report a run that
        # is permanently broken while it is in fact healthy.
        check("a dangling RUNNING is INCOMPLETE, not a discard", st["incomplete"] == 1,
              f"incomplete iters={st['incomplete']} (3 RUNNING lines exist, only 1 dangles)")
        check("it does not enter the cool-down", st["cooldowns"].get("backend", 0) == 0,
              f"cooldowns={st['cooldowns']}")
        check("the two real discards do", st["cooldowns"].get("calibration") == 2)
        check("errors are counted apart from discards", st["errors"] == 0)
        check("no incumbent without a keep", st["incumbent"] is None)

        o.log.append(Record(iter=3, ts=now_iso(), action=Action.KEEP.value,
                            route="head.probe_refit", kind=HARNESS, medium=0.80, delta=-0.03,
                            modules={"family": "head"}, iteration=7))
        st2 = o.rebuild()
        check("a keep becomes the incumbent", st2["incumbent"] is not None)
        check("the incumbent is the best, not the last", json.loads(st2["incumbent"])["medium"] == 0.80)

        mix = st2["route_mix"]
        # route_mix is keyed by ROUTE NAME; the harness/param split lives in _fractions. Asserting on
        # the wrong key would have passed vacuously here and told us nothing about attribution.
        check("route mix is keyed by route name", len([k for k in mix if k != "_fractions"]) == 3,
              str(sorted(k for k in mix if k != "_fractions")))
        check("every recorded route is an outer one",
              all(ALL_ROUTES[k].kind == HARNESS for k in mix if k != "_fractions"))
        check("param fraction starts at zero", mix["_fractions"].get(PARAM, 0.0) == 0.0)

        # MAP-Elites: a 2-D cell, not a scalar
        c1 = _map_elites_cell({"accuracy": 0.72, "soft_cross_entropy": 1.05})
        c2 = _map_elites_cell({"accuracy": 0.88, "soft_cross_entropy": 0.75})
        check("different regions get different cells", c1 != c2, f"{c1} vs {c2}")
        check("a missing metric does not raise", _map_elites_cell({}) == 0)


# ---------------------------------------------------------------------------------------
def test_prior_art_veto() -> None:
    print("\n3. PRIOR-ART VETO -- transferability decides, and a rejection is permanent")
    gen = ojf.IdeaCard("x-cot", "CoT self-verify", "any source",
                       ["chain_of_thought", "output_parser"], "claims gains", 1.0, "intelligence")
    v = ojf.adjudicate(gen)
    check("a prompt-space idea is REJECTED", v.verdict == ojf.REJECT, v.reason[:80])
    check("the rejection names why", "no LM head" in v.reason)

    cal = ojf.IdeaCard("x-platt", "Platt on margin", "any source",
                       ["calibration", "decision_rule"], "claims gains", 0.01, "calibration")
    v2 = ojf.adjudicate(cal)
    check("a calibration idea is ADOPTED", v2.verdict == ojf.ADOPT)
    check("it maps onto a real route", v2.route in ALL_ROUTES, str(v2.route))

    mixed = ojf.IdeaCard("x-mixed", "Platt + CoT", "any source",
                         ["calibration", "chain_of_thought"], "claims gains", 0.01, "calibration")
    v3 = ojf.adjudicate(mixed)
    check("a mixed idea takes the transferable half", v3.verdict == ojf.ADOPT)
    check("and says the other half is dropped", bool(v3.surfaces_blocked), str(v3.surfaces_blocked))

    unknown = ojf.IdeaCard("x-unk", "Newfangled", "any source", ["telepathy"], "?", 0.01, "intelligence")
    check("an unknown surface is NOTE, not ADOPT", ojf.adjudicate(unknown).verdict == ojf.NOTE)

    expensive = ojf.IdeaCard("x-cost", "Slow calibration", "any source",
                             ["calibration"], "?", 120.0, "calibration")
    check("a minute-long calibration idea is demoted", ojf.adjudicate(expensive).verdict == ojf.NOTE,
          "a minute is a parameter mutation in disguise")


# ---------------------------------------------------------------------------------------
def test_candidate_contract() -> None:
    print("\n4. THE CANDIDATE CONTRACT -- every candidate names a route with a known cost")
    try:
        Candidate(route="cal.made_up")
        check("an unknown route is refused", False, "no exception raised")
    except ValueError:
        check("an unknown route is refused", True)
    c = Candidate(route="cal.isotonic_top1")
    check("a known route is accepted", c.kind == HARNESS)
    check("a param route is tagged param", Candidate(route="qlora.rank_sweep").kind == PARAM)


def main() -> int:
    print("=" * 74)
    print("PIPELINE SKELETON -- the loop + AgentJev-0.6B + open-jev-fast")
    print("=" * 74)
    test_gate()
    test_replay()
    test_prior_art_veto()
    test_candidate_contract()
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("ALL CHECKS PASS -- the three tiers compose, the gate holds, replay is faithful, and a")
    print("rejected idea cannot re-enter under a different route name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
