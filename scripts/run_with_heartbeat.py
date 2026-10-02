#!/usr/bin/env python
"""run_with_heartbeat.py -- give a training run a progress unit it does not otherwise have.

WHAT IT DOES
    Runs the harness's training entry point unchanged, and appends one JSON row per N optimiser
    steps to <arm>/progress.jsonl: step index, wall-clock timestamp, elapsed seconds, and the
    step-to-step interval. Nothing in the reference code is edited; the counter is installed from
    outside by wrapping the optimiser.

WHEN TO REACH FOR IT
    For every arm, in place of calling the harness's entry point directly. The harness prints nothing
    during `fit()`, writes its checkpoint only after training, and has no resume, so a run without
    this wrapper has no progress signal at all -- only completion. The neighbouring tool is
    `health.py status`, which reads the state this wrapper produces plus the GPU; without a heartbeat
    there is nothing for it to read but activity, and activity is what a stuck job emits too.
    Not for evaluation time: the heartbeat stops when training does.

WHAT ITS OUTPUT MEANS
    `progress.jsonl` rows are append-only and timestamped at the moment of the write, which makes them
    the job's own time-stamped log -- the only legitimate basis for projecting its remaining time.
    A rate is fitted from the steady-state window, discarding the first rows as startup transient.
    **Failure looks like an absent or stalled file**: if the file stops growing while the process
    lives, the run is stuck, and that is distinguishable from a slow run for the first time.
    **A severalfold drop in the interval voids any projection made from it** -- re-measure, do not
    keep quoting the old ETA.

WHAT TEST ITS ANSWER MUST PASS
    Run it on a short job (a few steps) whose step count is known from the harness's own post-run
    report, and confirm the last heartbeat row's step index agrees. A counter that drifts from the
    harness's own count is worse than none.

USAGE
    python scripts/run_with_heartbeat.py --arm E:/2026-AI4S/arms/arm1 [--every 25] -- <release_train args>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import runpy
import sys
import time

_count = 0
_last = 0.0
_t0 = 0.0
_hb: pathlib.Path | None = None
_every = 25


def _bump() -> None:
    global _count, _last
    _count += 1
    if _hb is None or _count % _every:
        return
    now = time.time()
    row = {"step": _count, "at": datetime.datetime.fromtimestamp(now).isoformat(timespec="seconds"),
           "elapsed_s": round(now - _t0, 2), "since_last_s": round(now - _last, 2) if _last else None}
    _last = now
    try:
        with _hb.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())     # a heartbeat that sits in a buffer is not a heartbeat
    except OSError:
        pass                          # never let monitoring break the run it monitors


def install(arm_dir: pathlib.Path, every: int, env_dir: pathlib.Path) -> None:
    """Wrap every optimizer instance's step(), which no subclass override can escape."""
    global _hb, _every, _t0
    import torch

    _hb = arm_dir / "progress.jsonl"
    _hb.parent.mkdir(parents=True, exist_ok=True)
    _every = max(1, every)
    _t0 = time.time()
    with _hb.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": 0, "at": datetime.datetime.now().isoformat(timespec="seconds"),
                             "elapsed_s": 0.0, "since_last_s": None,
                             "note": "heartbeat start", "env": str(env_dir)}) + "\n")

    # Wrapping the INSTANCE method rather than torch.optim.Optimizer.step, because subclasses such
    # as AdamW define their own step() and would shadow a base-class patch. init runs before the
    # training loop builds anything, so every optimizer is covered.
    orig_init = torch.optim.Optimizer.__init__

    def wrapped_init(self, *a, **k):
        orig_init(self, *a, **k)
        inner = self.step

        def step_and_count(*aa, **kk):
            r = inner(*aa, **kk)
            _bump()
            return r

        self.step = step_and_count

    torch.optim.Optimizer.__init__ = wrapped_init


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    ap.add_argument("--arm", required=True, help="arm directory; progress.jsonl lands here")
    ap.add_argument("--every", type=int, default=25, help="heartbeat every N optimiser steps")
    ap.add_argument("--env", default=None, help="the code environment to run from")
    ap.add_argument("script", help="the harness entry point, e.g. scripts/release_train.py")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed through verbatim")
    a = ap.parse_args()

    arm = pathlib.Path(a.arm)
    env_dir = pathlib.Path(a.env) if a.env else arm
    script = pathlib.Path(a.script)
    if not script.is_file():
        print(f"[FAIL] {script} not found")
        return 2

    install(arm, a.every, env_dir)
    print(f"[heartbeat] every {a.every} steps -> {arm / 'progress.jsonl'}", flush=True)

    sys.argv = [str(script)] + list(a.args)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
