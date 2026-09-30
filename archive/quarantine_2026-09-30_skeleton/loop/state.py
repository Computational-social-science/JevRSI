"""loop/state.py -- the JSONL state machine: append-only log, full rebuild on restart.

WHAT THE PROPOSAL CLAIMS
------------------------
Section 3.4: "All experimental state is persisted to autoresearch.jsonl, which survives process kills
and host reboots. Each record stores commit hash, metrics, peak VRAM, run status and a short
diagnostic. Control state (eligible candidates, cool-down timers, current best) is recomputed from
the JSONL file on every restart; the cycle resumes from the last interrupted point with no reliance
on in-memory context."

That is a strong claim and it is the kind that fails silently: a state file that is written but never
READ back looks identical to one that works, right up until the machine reboots mid-run. So this
module is built to be tested rather than trusted, and `test_state.py` kills a process mid-write and
rebuilds.

WHY A LOG AND NOT A STATE FILE
------------------------------
A checkpoint file (`state.json`, rewritten each cycle) has one failure mode that a log does not: a
crash during the rewrite leaves a truncated or half-updated file, and there is no earlier version to
fall back on. An append-only log cannot be corrupted by a crash that happens DURING the write, only
by one that happens before the bytes reach the disk -- and a partial final line is detectable and
discardable. For a six-week unattended run, that difference is the whole design.

WHAT "REBUILD" MEANS HERE
-------------------------
Not "restore the last snapshot". The control state is DERIVED by replaying the log:

    incumbent      the best fitness among records whose action was "keep", per the accept rule
    archive        the MAP-Elites cell contents, replayed in order
    cool-downs     per module family, the length of the current consecutive-failure run
    failures       every discarded record, grouped by family (this is what FAILURES.md is built from)
    attempts       the monotonic counter, which must survive a crash mid-cycle without reusing a number

A record that was appended with action "running" and never completed -- a cycle killed by the
watchdog or a reboot -- is treated as INCOMPLETE, not as a failure. Conflating the two would silently
teach the loop that a candidate was bad when in fact it was never judged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class Action(str, Enum):
    RUNNING = "running"   # appended before the cycle starts; may be left dangling by a crash
    KEEP = "keep"
    DISCARD = "discard"
    ERROR = "error"       # the cycle could not be run at all (OOM, crash); not evidence about the idea


@dataclass
class Record:
    """One line of autoresearch.jsonl. Every field the controller needs to rebuild itself."""

    iter: int
    ts: str
    action: str
    parent: str | None = None          # commit hash of the parent configuration
    commit: str | None = None          # commit hash of this configuration
    # Which level produced this record. The proposal's logging schema requires it, and it cannot be
    # reconstructed after the fact: a record that does not say "harness" or "param" is a record from
    # which the harness-vs-parameter attribution the proposal mandates cannot be computed at all.
    # Added after the first records already existed, so it carries a default and old lines still load.
    route: str | None = None           # e.g. "cal.isotonic_top1", "qlora.rank_sweep"
    kind: str | None = None            # "harness" | "param" -- denormalised so a scan needs no join
    modules: dict[str, str] = field(default_factory=dict)   # {"cal": "L1", "lora": "r16"}
    mutation: str = ""                 # the concrete change, human-readable
    proxy: float | None = None         # proxy-set fitness
    medium: float | None = None        # medium-set fitness -- the decision signal
    delta: float | None = None         # medium - parent medium
    epsilon: float | None = None       # the threshold this decision was made against
    peak_vram_gib: float | None = None
    wall_s: float | None = None
    iteration: int | None = None       # archive cell, for MAP-Elites
    diagnosis: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "Record":
        return Record(**json.loads(line))


class Log:
    """Append-only JSONL with full rebuild. One instance per run directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writing ------------------------------------------------------------------------------
    def append(self, rec: Record) -> None:
        """Append one record and fsync.

        The fsync is the point of the whole design: without it a power loss can lose records that
        the OS had already accepted, and the rebuilt state would be silently short. A six-week run
        on a consumer machine will see at least one unclean shutdown.
        """
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(rec.to_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def next_iter(self) -> int:
        """The next attempt number. Derived from the log, never from a counter in memory.

        Reading it back from the log is what makes the number monotonic across a crash: an
        in-memory counter restarts at zero and two cycles can then share a number, which breaks the
        one-record-per-attempt invariant that the trajectory archive depends on.
        """
        return self.rebuild()["next_iter"]

    # ---- reading ------------------------------------------------------------------------------
    def read(self) -> Iterator[Record]:
        """Yield every intact record, in order. A truncated final line is skipped, not fatal.

        Skipping rather than raising is deliberate: a crash during append can leave a partial line,
        and the correct response to "the last cycle was cut off" is "that cycle did not finish",
        which is exactly what RUNNING-then-truncated already means. Raising would turn a recoverable
        state into a dead run.
        """
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Record.from_json(line)
                except (json.JSONDecodeError, TypeError):
                    continue

    def rebuild(self) -> dict[str, Any]:
        """Replay the whole log into control state. This is the function §3.4 depends on."""
        attempts = 0
        incomplete: list[dict] = []
        kept: list[Record] = []
        discarded: list[Record] = []
        errors: list[Record] = []
        # MAP-Elites cell -> the record that currently occupies it
        archive: dict[tuple, dict] = {}
        # module family -> consecutive failure run length
        cool: dict[str, int] = {}
        failures: dict[str, list[dict]] = {}
        best: dict | None = None
        vram: list[float] = []

        for rec in self.read():
            if rec.action == Action.RUNNING.value:
                # Left dangling: the cycle started and never reached a verdict. NOT a failure.
                # It DOES consume an attempt number, though -- the number was spent when the record
                # was appended, and reusing it after a restart would give two cycles one number and
                # break the one-record-per-attempt invariant the trajectory archive depends on.
                incomplete.append(asdict(rec))
                attempts += 1
                continue
            attempts += 1
            fam = rec.modules.get("family") or rec.mutation.split(":")[0] or "unlabelled"
            if rec.action == Action.KEEP.value:
                kept.append(rec)
                cool[fam] = 0
                if rec.iteration is not None:
                    cell = (int(rec.iteration),)
                    prev = archive.get(cell)
                    if prev is None or (rec.medium or -1) > (prev.get("medium") or -1):
                        archive[cell] = asdict(rec)
                if rec.medium is not None and (best is None or rec.medium > best["medium"]):
                    best = asdict(rec)
            elif rec.action == Action.DISCARD.value:
                discarded.append(rec)
                cool[fam] = cool.get(fam, 0) + 1
                failures.setdefault(fam, []).append(
                    {"iter": rec.iter, "mutation": rec.mutation, "delta": rec.delta,
                     "epsilon": rec.epsilon, "diagnosis": rec.diagnosis})
            elif rec.action == Action.ERROR.value:
                errors.append(rec)
                cool[fam] = cool.get(fam, 0) + 1
            if rec.peak_vram_gib is not None:
                vram.append(rec.peak_vram_gib)

        return {
            # Attempt numbers are 1-based, so an empty log yields the FIRST attempt, not zero. A
            # zero-based counter would make the first cycle `iter=0`, which then reads as "no cycle
            # has run" in every downstream summary and in the trajectory archive's directory names.
            "next_iter": attempts + 1,
            "n_records": attempts,
            "n_kept": len(kept),
            "n_discarded": len(discarded),
            "n_errors": len(errors),
            "n_incomplete": len(incomplete),
            "incomplete": incomplete,
            "best": best,
            "archive": {str(k): v for k, v in archive.items()},
            "n_archive_cells": len(archive),
            "cooldowns": cool,
            "families_exhausted": [f for f, n in cool.items() if n >= 3],
            "failures_by_family": failures,
            "peak_vram_gib": max(vram) if vram else None,
            "mean_cycle_wall_s": (sum(r.wall_s for r in kept + discarded if r.wall_s)
                                  / max(1, len([r for r in kept + discarded if r.wall_s]))),
        }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    import sys
    log = Log(sys.argv[1] if len(sys.argv) > 1 else "autoresearch.jsonl")
    print(json.dumps(log.rebuild(), indent=2, ensure_ascii=False)[:2000])
