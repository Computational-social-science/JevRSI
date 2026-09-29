"""routes.py -- the two levels, and the gate between them.

WHAT "BILEVEL" MEANS HERE

The objective's technical posture is harness-first: a harness mutation costs seconds, a parameter
mutation costs minutes to hours, and on a single 12 GB card that ratio decides how many candidates a
six-week run can evaluate. The controller is therefore organised as two levels rather than one flat
search, and the inner level is GATED rather than merely deprioritised:

    OUTER  harness      calibration, scoring pipeline, inference backend     seconds   the default
    INNER  parameter    QLoRA adapters, rank / target / LR sweeps             minutes   only past the gate

The gate is not a schedule and not a preference. It is a measured condition on the outer level's own
recent slope, and it is deliberately hard to open, because the failure mode is asymmetric: a premature
inner cycle burns the budget on the expensive route while the cheap one still had headroom, and that
cannot be undone by reordering later.

WHY ROUTES ARE DECLARED, NOT DISCOVERED

A flat search over "things we could change" will, over 672 hours, spend its cycles wherever the
proposal-writer happened to write more detail. Declaring routes as data -- with their measured cost,
their axis, and their gate -- makes the controller's schedule auditable: every cycle's log line names
the route it came from, so the proposal's requirement that gains be "disaggregated into harness-
attributed vs parameter-attributed" is satisfied by construction rather than by later analysis of a
log that never recorded it.

THE COSTS ARE MEASURED, NOT ESTIMATED

Route costs here come from this host, and each is a measurement with a date:

    harness post-hoc stage      ~10 ms per 600 questions    (measurement/harness_sweep_01.json)
    harness feature-space refit    16 s for 2.37M params    (measurement/head_offline.json)
    harness feature cache          7.4 min, amortised       (measurement/features/manifest_final.json)
    one 600-step LoRA replicate    2.11 h                    (measured, 12.6 s/step clean)

The proposal's own estimate for a QLoRA pass is "typically <= 5 minutes for 1,000 samples". On this
host that is off by more than an order of magnitude, which is why the gate below is expressed in terms
of measured cycle counts rather than the proposal's nominal hours: a controller that trusted the
nominal figure would open the inner level believing it was cheap.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field, asdict

HARNESS = "harness"
PARAM = "param"


@dataclass
class Route:
    """One mutable family, with the axis it acts on and what it costs to try."""

    name: str
    kind: str                                   # harness | param
    axis: str                                   # intelligence | calibration | speed | cost
    cost_s: float                               # measured seconds per cycle
    family: str = ""                            # module family, for cool-downs and FAILURES.md
    gated: bool = False                         # inner level: needs the gate open
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------------------
# The outer level. These are the default. Several are already measured on this host.
# ---------------------------------------------------------------------------------------
HARNESS_ROUTES: list[Route] = [
    Route("cal.scalar_temperature", HARNESS, "calibration", 0.01, "calibration",
          note="REFERENCE ARM ONLY. Monotone, so argmax is invariant and it cannot move the "
               "Intelligence axis. Retained because it is the baseline every other calibration "
               "stage is read against, not because it is a candidate."),
    Route("cal.isotonic_top1", HARNESS, "calibration", 0.01, "calibration",
          note="monotone map on top-1 confidence, fitted on calibration. Cannot move accuracy. "
               "Must be judged on the PRIMARY (soft CE), never on ECE: ECE is a binned statistic "
               "and is gameable -- see loop/fitness.py."),
    Route("cal.platt_per_type", HARNESS, "calibration", 0.01, "calibration"),
    Route("cal.vector_temperature", HARNESS, "intelligence", 0.02, "calibration",
          note="per-dimension temperature. NOT a scalar monotone map, so it is allowed to move "
               "argmax -- the only cheap family that can reach the 87.4% of errors sitting within "
               "margin 1.0 of the boundary."),
    Route("cal.non_monotone_window", HARNESS, "intelligence", 0.01, "calibration",
          note="gamma-parameterised mid-confidence window. Also non-monotone, so also able to move "
               "argmax. Included to MEASURE reachability, not to assume it."),
    Route("head.probe_refit", HARNESS, "intelligence", 16.0, "head",
          note="refit the 2.37M head on cached frozen-backbone features. Measured 16 s and 1.50 pp "
               "from the shipped head (measurement/head_offline.json). The feature cache is "
               "amortised: 7.4 min once, then seconds per experiment."),
    Route("pipe.option_permutation", HARNESS, "intelligence", 0.05, "pipeline",
          note="the head is permutation-equivariant by construction, so this must be a no-op. It is "
               "worth one cycle as a NEGATIVE CONTROL: if it moves the score, the equivariance claim "
               "is wrong and every other number computed here is suspect."),
    Route("pipe.numerical_stability", HARNESS, "intelligence", 0.05, "pipeline"),
    Route("infer.batch_policy", HARNESS, "speed", 0.05, "backend",
          note="this host has already measured a 10x step-time swing from allocator behaviour alone "
               "(0.199 -> 0.688 s/step, never recovering), so batch policy is a first-class Speed "
               "lever and not a micro-optimisation."),
    Route("infer.cuda_graphs", HARNESS, "speed", 300.0, "backend"),
    Route("infer.precision_policy", HARNESS, "speed", 0.05, "backend",
          note="a bf16<->fp32 change moves logits by ~0.0075 mean abs on a fixed machine "
               "(~0.15 pp Top-1), so this is a Speed lever with a known small Intelligence cost."),
]

# ---------------------------------------------------------------------------------------
# The inner level. Gated. Expensive. The proposal makes this secondary for a reason.
# ---------------------------------------------------------------------------------------
PARAM_ROUTES: list[Route] = [
    Route("qlora.rank_sweep", PARAM, "intelligence", 7596.0, "qlora", gated=True,
          note="one 600-step replicate measured at 2.11 h on this host, not the proposal's nominal "
               "5 minutes. A rank sweep of 5 is therefore ~10.5 h."),
    Route("qlora.target_modules", PARAM, "intelligence", 7596.0, "qlora", gated=True),
    Route("qlora.lr_sweep", PARAM, "intelligence", 7596.0, "qlora", gated=True),
    Route("qlora.task_family", PARAM, "intelligence", 7596.0, "qlora", gated=True),
]

ALL_ROUTES: dict[str, Route] = {r.name: r for r in HARNESS_ROUTES + PARAM_ROUTES}


# ---------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------
@dataclass
class GateState:
    """Why the inner level is shut, or open. Persisted, because the reason must survive a restart.

    A gate that is recomputed from memory on restart will quietly re-open or re-shut after a reboot,
    and the log will then contain inner cycles with no gate event explaining them. Rule 3 requires the
    control state to rebuild from the JSONL alone, so the gate's inputs are derived from the log.
    """

    open: bool = False
    reason: str = "inner level shut by default"
    harness_saturated_on: str = ""        # which axis looked flat
    consecutive_flat_cycles: int = 0
    cool_down_k: int = 6                 # flat cycles on the bottleneck axis before the gate opens
    max_param_fraction: float = 0.25     # hard cap, per the proposal's risk register
    observed_param_fraction: float = 0.0
    _notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def flat_run_length(slopes: list[float], tolerance: float) -> int:
    """How many of the most recent harness cycles show no material movement on the bottleneck axis.

    `slopes` is most-recent-first, in units of the PRIMARY metric (lower is better, so a positive
    slope means it got WORSE). A cycle counts as flat when |slope| <= tolerance.

    Counting from the most recent backwards, and stopping at the first non-flat cycle, is deliberate:
    a single good result buried under twelve flat ones is not a reason to spend an hour of GPU, and a
    controller that averaged would open the gate on exactly that case.
    """
    n = 0
    for s in slopes:
        if abs(s) <= tolerance:
            n += 1
        else:
            break
    return n


def update_gate(gate: GateState, slopes: list[float], tolerance: float) -> GateState:
    """Recompute the gate from the recent harness slopes. Pure function of its inputs, so a restart
    that replays the log reaches the same state."""
    run = flat_run_length(slopes, tolerance)
    gate.consecutive_flat_cycles = run
    if run >= gate.cool_down_k and gate.observed_param_fraction < gate.max_param_fraction:
        gate.open = True
        gate.reason = (f"harness flat on the bottleneck axis for {run} consecutive cycles "
                       f"(tolerance {tolerance}); parameter cap {gate.max_param_fraction} not reached")
    else:
        gate.open = False
        if run < gate.cool_down_k:
            gate.reason = f"harness flat for {run}/{gate.cool_down_k} cycles"
        else:
            gate.reason = (f"parameter cycles already at {gate.observed_param_fraction:.2f} of the "
                           f"cap {gate.max_param_fraction}")
    return gate


def load_gate(path: str | pathlib.Path) -> GateState:
    p = pathlib.Path(path)
    if not p.exists():
        return GateState()
    return GateState(**json.loads(p.read_text(encoding="utf-8")))


def save_gate(gate: GateState, path: str | pathlib.Path) -> None:
    pathlib.Path(path).write_text(gate.to_json(), encoding="utf-8")


if __name__ == "__main__":
    print(f"outer level: {len(HARNESS_ROUTES)} harness routes")
    for r in HARNESS_ROUTES:
        print(f"  {r.name:28s} {r.axis:12s} {r.cost_s:8.2f}s")
    print(f"\ninner level: {len(PARAM_ROUTES)} param routes (gated)")
    for r in PARAM_ROUTES:
        print(f"  {r.name:28s} {r.axis:12s} {r.cost_s:8.2f}s")
    tot_h = sum(r.cost_s for r in HARNESS_ROUTES)
    tot_p = sum(r.cost_s for r in PARAM_ROUTES)
    print(f"\ncost asymmetry: full harness sweep {tot_h/60:.1f} min vs one param route "
          f"{tot_p/len(PARAM_ROUTES)/3600:.2f} h")
    print("This ratio is the whole argument for the gate.")
