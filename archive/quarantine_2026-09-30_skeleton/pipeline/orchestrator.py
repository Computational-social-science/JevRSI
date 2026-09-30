"""orchestrator.py -- the cycle driver: Select -> Mutate -> Evaluate -> Archive -> Persist.

WHAT THIS IS

The skeleton of the pipeline. It owns the ORDER of a cycle and nothing else: the fitness contract
belongs to loop/fitness.py, the noise threshold to loop/epsilon.py, the durable log to loop/state.py,
and the mutation taxonomy to pipeline/routes.py. Keeping those separate is the point -- each of those
files encodes a decision that is expensive to get wrong, and none of them should be reachable by
accident from a change made here.

THE THREE TIERS AND WHERE THEY MEET

    open-jev-fast     proposes an idea and a route, or refuses the idea permanently
    outer level       runs harness routes, seconds per cycle, the default
    inner level       runs parameter routes, minutes per cycle, behind the gate

    ("Outer" and "inner" are structural descriptions of THIS schedule. They are not a claim about the
    borrowed autoresearch harness's design, and they do not invoke bilevel optimization -- see
    docs/TERMINOLOGY.md, which rules that compound term out.)

The orchestrator is the only component that sees all three, and the only place where a prior-art
proposal becomes a cycle. That single choke point is deliberate: it is where "did we already try this"
is checked, so a rejected idea cannot re-enter by a different route name.

WHY EVALUATION IS BATCHED, AND WHY THAT IS A CORRECTNESS REQUIREMENT

A parent that holds a GPU-resident model while spawning a child squeezes the child into an allocator
stall -- measured on this host as a child frozen at step 1 for 632 s with the GPU at 100% and a frozen
log mtime. Worse, the contention made a healthy configuration look 9x slower, which would have been
recorded as evidence about the configuration. So: no model is loaded in this process. The child does
the work and exits; the orchestrator only reads numbers. The rule generalises to the rule in
measurement/measure_speed_cost_axes.py, which refuses to time anything while VRAM is busy.

THE ORDER IS NOT ARBITRARY

RUNNING is appended BEFORE the work starts, so a crash leaves a dangling record that replay can mark
incomplete rather than mistake for a judged failure. Conflating "killed" with "rejected" would teach
the loop that a candidate was bad when it was never evaluated -- and, in a six-week run with a
selection operator, that is a bias that compounds.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import dataclass, field, asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "loop"))

from loop.state import Action, Log, Record, now_iso                      # noqa: E402
from loop.fitness import Decision, evaluate as fitness_evaluate          # noqa: E402
from pipeline.routes import (ALL_ROUTES, GateState, HARNESS, PARAM,      # noqa: E402
                             load_gate, save_gate, update_gate)
from pipeline import open_jev_fast as ojf                                 # noqa: E402


@dataclass
class Candidate:
    """A concrete, runnable mutation. `route` must name a declared route -- see routes.py."""

    route: str
    params: dict = field(default_factory=dict)
    parent_commit: str | None = None
    provenance: str = ""              # where the idea came from; empty means the loop invented it

    def __post_init__(self):
        if self.route not in ALL_ROUTES:
            raise ValueError(
                f"unknown route {self.route!r}. A candidate must map onto a declared route so its cost, "
                f"axis and gate status are known; otherwise the cycle cost is unbounded and the "
                f"harness/param attribution the proposal requires cannot be computed. "
                f"Known routes: {sorted(ALL_ROUTES)}")

    @property
    def kind(self) -> str:
        return ALL_ROUTES[self.route].kind


@dataclass
class Cycle:
    iteration: int
    route: str
    action: str
    primary: float | None = None
    reason: str = ""
    diagnosis: str = ""


class Orchestrator:
    """The loop. Deliberately thin: it sequences work and records outcomes, and holds no model."""

    def __init__(self, run_dir: str | pathlib.Path, epsilon: float,
                 gate_path: str | pathlib.Path | None = None):
        self.run_dir = pathlib.Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.run_dir / "autoresearch.jsonl")
        self.gate_path = pathlib.Path(gate_path or (self.run_dir / "gate.json"))
        self.gate = load_gate(self.gate_path)
        self.epsilon = epsilon
        self.rejected = ojf.rejected_ids()

    # ---- replay ---------------------------------------------------------------------------------
    def rebuild(self) -> dict:
        """Derive control state from the log ALONE, per Rule 3 and loop/state.py's contract.

        A record left as RUNNING by a crash is INCOMPLETE, not a failure. It is counted separately
        and is never allowed to satisfy a cool-down, because a cycle that never ran is not evidence
        that an idea failed.
        """
        recs = [Record.from_json(l) for l in self.log.path.read_text(encoding="utf-8").splitlines()
                if l.strip()] if self.log.path.exists() else []
        kept = [r for r in recs
                if r.action == Action.KEEP.value and r.medium is not None]
        errors = [r for r in recs if r.action == Action.ERROR.value]
        discards = [r for r in recs if r.action == Action.DISCARD.value]

        # A cycle writes RUNNING first and its terminal record second, so counting RUNNING *records*
        # counts every cycle ever started. Incompleteness is a property of an ITERATION, not of a
        # line: an iteration is incomplete when it has a RUNNING record and no terminal record for
        # the same iter. Getting this wrong reports 3 unfinished cycles where there is 1, and after a
        # few dozen restarts it reports a permanently broken run that is actually healthy.
        terminal_iters = {r.iter for r in recs
                          if r.action in (Action.KEEP.value, Action.DISCARD.value, Action.ERROR.value)}
        running_iters = [r.iter for r in recs if r.action == Action.RUNNING.value]
        incomplete = [i for i in running_iters if i not in terminal_iters]

        # cool-downs: consecutive-failure run length per module family, in iteration order.
        # Only a DISCARD extends a cool-down. A KEEP resets it, and an ERROR or a dangling RUNNING
        # leaves it untouched -- a cycle that never produced a number is not a failed idea, and
        # letting it count would make a family look exhausted because the machine kept crashing.
        cool: dict[str, int] = {}
        for r in sorted(recs, key=lambda x: x.iter):
            fam = (r.modules or {}).get("family", "?")
            if r.action in (Action.KEEP.value,):
                cool[fam] = 0
            elif r.action == Action.DISCARD.value:
                cool[fam] = cool.get(fam, 0) + 1

        # recent harness slopes on the PRIMARY (lower is better), most recent first
        done = [r for r in recs if r.medium is not None and r.route == HARNESS]
        done.sort(key=lambda x: x.iter, reverse=True)
        slopes = [r.delta for r in done[: self.gate.cool_down_k] if r.delta is not None]

        return {
            "records": len(recs), "kept": len(kept), "discarded": len(discards),
            "incomplete": len(incomplete), "errors": len(errors),
            # float() is not cosmetic: `kept` is filtered on `medium is not None`, but a None in the
            # key would make min() raise on the next incomparable record rather than skip it.
            "incumbent": (min(kept, key=lambda r: float(r.medium)).to_json() if kept else None),
            "cooldowns": cool, "harness_slopes": slopes,
            "route_mix": _route_mix(recs),
        }

    # ---- selection ------------------------------------------------------------------------------
    def select(self, state: dict) -> Candidate | None:
        """Pick the next cycle. Priority order, with the inner level reachable only through the gate.

        1. never re-propose something the prior-art tier rejected, under any route name
        2. an untried OUTER route whose family is not in cool-down
        3. an open-jev-fast proposal that maps onto an untried route
        4. the INNER level, and only if the gate is open and the param cap is not reached
        """
        self.gate.observed_param_fraction = state["route_mix"].get(PARAM, 0.0)
        self.gate = update_gate(self.gate, state["harness_slopes"], tolerance=self.epsilon)
        save_gate(self.gate, self.gate_path)

        # A route counts as TRIED once it has been ATTEMPTED, whether or not it produced a number.
        #
        # The earlier test was `medium is not None`, which excluded every route that failed -- and a
        # route that fails is precisely the one that must not be re-proposed forever. In a 6-cycle run
        # that put cal.platt_per_type into 10 log records and starved head.probe_refit, which had
        # never run at all. An unimplemented route is not a candidate that keeps its place in the
        # queue; it is a gap in the executor, and the honest response to a gap is to record it (which
        # the cycle record already does) and move on to the routes that can actually be measured.
        tried = {r.route for r in self._records()}
        cool = state["cooldowns"]

        # (1) prior-art proposals, filtered against permanent rejections
        for v in reversed(ojf.ledger()):
            if v["verdict"] != ojf.ADOPT or not v.get("route"):
                continue
            if v["idea_id"] in self.rejected:
                continue
            if v["route"] in tried:
                continue
            if cool.get(ALL_ROUTES[v["route"]].family, 0) >= self.gate.cool_down_k:
                continue
            return Candidate(route=v["route"], provenance=f"open-jev-fast:{v['idea_id']}")

        # (2) untried outer routes
        for name, r in ALL_ROUTES.items():
            if r.kind != HARNESS or name in tried:
                continue
            if cool.get(r.family, 0) >= self.gate.cool_down_k:
                continue
            if "REFERENCE" in r.note.upper():
                continue
            return Candidate(route=name)

        # (4) inner level, gated
        if self.gate.open:
            for name, r in ALL_ROUTES.items():
                if r.kind == PARAM and name not in tried:
                    return Candidate(route=name, provenance="gate")
        return None

    def _records(self) -> list[Record]:
        if not self.log.path.exists():
            return []
        return [Record.from_json(l) for l in self.log.path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    # ---- one cycle ------------------------------------------------------------------------------
    def run_cycle(self, cand: Candidate, run_fn, incumbent_metrics: dict | None) -> Cycle:
        """Execute one candidate. `run_fn(cand) -> (metrics, wall_s, peak_vram_gib, diagnosis)`.

        `run_fn` MUST be a subprocess that loads its own model and exits. This process holds no GPU
        state on purpose -- see the module docstring on the 632 s allocator stall.
        """
        it = len(self._records())
        self.log.append(Record(iter=it, ts=now_iso(), action=Action.RUNNING.value,
                               route=cand.route, kind=cand.kind, parent=cand.parent_commit,
                               modules={"route": cand.route,
                                        "family": ALL_ROUTES[cand.route].family,
                                        "kind": cand.kind},
                               mutation=cand.provenance or f"apply {cand.route}",
                               epsilon=self.epsilon))
        t0 = time.perf_counter()
        try:
            metrics, _wall, vram, diagnosis = run_fn(cand)
        except Exception as e:                                     # noqa: BLE001
            # ERROR is not evidence about the idea. A crash, an OOM or a missing file says nothing
            # about whether the mutation was good, and recording it as a discard would poison the
            # family's cool-down with a verdict nobody earned.
            self.log.append(Record(iter=it, ts=now_iso(), action=Action.ERROR.value,
                                   route=cand.route, kind=cand.kind,
                                   wall_s=time.perf_counter() - t0,
                                   diagnosis=f"{type(e).__name__}: {e}"[:400]))
            return Cycle(it, cand.route, Action.ERROR.value, None, f"{type(e).__name__}: {e}"[:200])

        wall = time.perf_counter() - t0
        if incumbent_metrics is None:
            decision = Decision(True, 0.0, self.epsilon, "first candidate; nothing to compare against", [])
        else:
            decision = fitness_evaluate(metrics, incumbent_metrics, self.epsilon)
        cell = _map_elites_cell(metrics)

        self.log.append(Record(
            iter=it, ts=now_iso(),
            action=Action.KEEP.value if decision.accept else Action.DISCARD.value,
            route=cand.route, kind=cand.kind, medium=metrics.get("soft_cross_entropy"),
            delta=(metrics.get("soft_cross_entropy") - incumbent_metrics.get("soft_cross_entropy"))
            if incumbent_metrics else None,
            epsilon=self.epsilon, peak_vram_gib=vram, wall_s=wall,
            iteration=cell, diagnosis=(decision.reason + " | " + diagnosis)[:400]))
        return Cycle(it, cand.route, Action.KEEP.value if decision.accept else Action.DISCARD.value,
                     metrics.get("soft_cross_entropy"), decision.reason, diagnosis)


def _map_elites_cell(metrics: dict, bins: int = 5) -> int:
    """Discrete archive cell over (Intelligence, Calibration), per the proposal's MAP-Elites grid.

    A composite fitness would put every candidate in one line and destroy the diversity the archive
    exists to preserve, so the cell is a 2-D discretisation instead. Missing metrics fall into bin 0
    rather than raising: an unmeasurable candidate is still a candidate, and dropping it would make
    the archive quietly incomplete.
    """
    def q(v, lo, hi):
        if v is None:
            return 0
        return max(0, min(bins - 1, int((float(v) - lo) / (hi - lo) * bins)))
    acc = metrics.get("accuracy")
    ce = metrics.get("soft_cross_entropy")
    return q(acc, 0.70, 0.90) * bins + q(ce, 0.70, 1.10)


def _route_mix(recs: list[Record]) -> dict:
    mix: dict[str, int] = {}
    for r in recs:
        if r.action in (Action.KEEP.value, Action.DISCARD.value) and r.route:
            mix[r.route] = mix.get(r.route, 0) + 1
    total = sum(mix.values()) or 1
    return {**mix, "_fractions": {k: v / total for k, v in mix.items()}}


if __name__ == "__main__":
    print(__doc__)
    print("This module is the skeleton. Wire `run_fn` to a subprocess that loads the model, then:")
    print("  python pipeline/orchestrator.py --plan     # show what the selector would pick first")
