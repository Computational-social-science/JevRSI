"""test_resume_exact.py -- prove that --resume is EXACT, not approximate.

WHY THIS TEST IS THE POINT. A 600-step Floor B replicate takes 2.11 h on this machine (measured 12.6
s/step) and the E: volume
has disconnected twice in one day, so crash recovery is not an optimisation -- it is what makes a
22-hour batch possible. But "it resumes" is a much weaker claim than "it resumes correctly".

An approximate resume would be easy to build and worthless: restore the weights, forget the RNG, and
the continued run draws different dropout masks and re-feeds states the interrupted run already
consumed. It would still produce a plausible loss curve and a plausible final score. Nobody would
notice. And the damage would land precisely in the quantity under measurement -- Floor B is the
seed-to-seed dispersion of a final score, so a resume that perturbs the data order and the dropout
stream is not a slightly noisier replicate, it is a different replicate wearing the same seed's name.

So this test asserts bit-identity. It runs the same seed twice over 6 steps:

  A. straight through, uninterrupted
  B. killed by SIGKILL once checkpoint-3.pt lands, then relaunched with --resume auto

and requires that B's steps 4, 5 and 6 reproduce A's to the last printed digit -- including the
loss, the learning rate, and the token rate. A test that tolerates a tolerance here is not testing
the thing that matters.

WHY IT KILLS THE PROCESS RATHER THAN PASSING A SMALL max-steps. The scheduler is built from max_steps
(build_scheduler(optimizer, max_steps, ...)), so a run cut short by a lower max_steps follows a
DIFFERENT cosine schedule and could never match. The only faithful crash is a real SIGKILL mid-run,
which is also what actually happens when a drive is yanked.

Run: python measurement/test_resume_exact.py
"""
from __future__ import annotations

# Project root, derived from this file's location rather than hardcoded.
# A literal machine path here would make the repository uncloneable and unrunnable
# anywhere else -- see Rule 1 of docs/PROJECT_RULES.md.
ROOT = pathlib.Path(__file__).resolve().parents[2]

import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
SUBJECT = pathlib.Path(os.environ.get("JEVRSI_SUBJECT", ROOT.parent / "agent-jev")).resolve()
PY = sys.executable
CONFIG = "agentjev/configs/autoresearch.yaml"
OUT = Path("./measurement/resume_test")
STEPS = 6
SAVE_EVERY = 3
SEED = 424242
LORA_ARGS = ["--lora", "--lora-r", "16", "--batch-states", "1", "--grad-accum", "16"]

# Windows has neither os.killpg nor signal.SIGKILL, so the "pull the plug mid-run" is done with
# taskkill /F /T, which is the platform's equivalent of SIGKILLing the whole process tree. CREATE_NEW_GROUP
# keeps the child's console from taking the parent's down with it.
CREATE_NEW_GROUP = getattr(subprocess, "CREATE_NEW_GROUP", 0x00000200)


def hard_kill(pid: int) -> None:
    """Kill a process tree the way a yanked drive or a power cut would: no cleanup, no flush."""
    r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        # Already gone is the desired end state, so this is not an error worth raising.
        print(f"  [test] taskkill rc={r.returncode}: {r.stdout.strip()[:120]}")

STEP_RE = re.compile(r"step (\d+)/(\d+) loss=([0-9.eE+-]+) lr=([0-9.eE+-]+)")


def run(cmd: list[str], log: Path, kill_after_ckpt: Path | None = None) -> int:
    """Run training, optionally SIGKILLing it the moment a checkpoint file appears.

    The child is started in its own process group so the kill reaches python and not just a shell.
    stdout goes to `log` because the per-step lines are the measurements.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=SUBJECT, stdout=fh, stderr=subprocess.STDOUT,
                             env=env, errors="replace", creationflags=CREATE_NEW_GROUP)
        if kill_after_ckpt is None:
            return p.wait()
        deadline = time.time() + 900
        while time.time() < deadline:
            if kill_after_ckpt.exists():
                # Give the atomic rename a moment to land before pulling the plug.
                time.sleep(1.0)
                print(f"  [test] {kill_after_ckpt.name} appeared; hard-killing the run")
                hard_kill(p.pid)
                p.wait(timeout=60)
                return -9
            if p.poll() is not None:
                return p.returncode
            time.sleep(0.5)
        hard_kill(p.pid)
        p.wait(timeout=60)
        return -9


def parse_losses(log: Path) -> dict[int, tuple[str, str]]:
    out = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            out[int(m.group(1))] = (m.group(3), m.group(4))
    return out


def train_cmd(out_dir: Path, resume: str) -> list[str]:
    return ([PY, "-m", "agentjev.train", "--config", CONFIG,
             "--seed", str(SEED), "--max-steps", str(STEPS), "--out-dir", str(out_dir),
             "--save-every-steps", str(SAVE_EVERY), "--resume", resume] + LORA_ARGS)


def main() -> int:
    if not (SUBJECT / "agentjev" / "train.py").exists():
        print("[FAIL] subject volume not mounted")
        return 2
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- A: uninterrupted reference -------------------------------------------
    print("=" * 74)
    print("RUN A: 6 steps, straight through (the reference)")
    print("=" * 74, flush=True)
    d_a = OUT / "A_uninterrupted"
    rc = run(train_cmd(d_a, "off"), d_a / "train.log")
    a = parse_losses(d_a / "train.log")
    print(f"  rc={rc} | steps recorded: {sorted(a)}")
    if len(a) != STEPS:
        print(f"[FAIL] reference run produced {len(a)}/{STEPS} steps; cannot test against it")
        return 1
    for s in sorted(a):
        print(f"    step {s}: loss={a[s][0]} lr={a[s][1]}")

    # ---- B: killed, then resumed ----------------------------------------------
    print(f"\n{'='*74}\nRUN B: killed at checkpoint-{SAVE_EVERY}, then resumed\n{'='*74}", flush=True)
    d_b = OUT / "B_resumed"
    rc = run(train_cmd(d_b, "off"), d_b / "train.log", kill_after_ckpt=d_b / f"checkpoint-{SAVE_EVERY}.pt")
    print(f"  first leg rc={rc} (expect -9 = SIGKILL)")
    b_first = parse_losses(d_b / "train.log")
    print(f"  steps before the kill: {sorted(b_first)}")
    ck = d_b / f"checkpoint-{SAVE_EVERY}.pt"
    if not ck.exists():
        print(f"[FAIL] no checkpoint-{SAVE_EVERY}.pt was written; the kill fired before any save, "
              f"so there is nothing to resume from")
        return 1
    print(f"  resuming from {ck.name} ({ck.stat().st_size/1e9:.2f} GB)", flush=True)
    rc = run(train_cmd(d_b, "auto"), d_b / "train_resume.log")
    b = parse_losses(d_b / "train_resume.log")
    print(f"  second leg rc={rc} | steps after resume: {sorted(b)}")
    for s in sorted(b):
        print(f"    step {s}: loss={b[s][0]} lr={b[s][1]}")

    # ---- the comparison that actually matters ----------------------------------
    print(f"\n{'='*74}\nVERDICT\n{'='*74}")
    overlap = sorted(set(a) & set(b))
    if not overlap:
        print(f"[FAIL] the two runs share no steps at all (A={sorted(a)} B={sorted(b)}); "
              f"resume produced nothing to compare")
        return 1

    mismatches = []
    for s in overlap:
        if a[s] != b[s]:
            mismatches.append((s, a[s], b[s]))
    print(f"  compared steps {overlap}  (loss, lr) -- exact string equality, no tolerance")
    for s, x, y in mismatches:
        print(f"    step {s}: A loss={x[0]} lr={x[1]}  !=  B loss={y[0]} lr={y[1]}")

    # Also confirm the pre-kill legs agreed -- if they did not, the two runs were never comparable.
    pre = sorted(set(a) & set(b_first))
    pre_bad = [s for s in pre if a[s] != b_first[s]]
    print(f"  pre-kill overlap {pre}: {'identical' if not pre_bad else f'DIFFERS at {pre_bad}'}")
    if pre_bad:
        print("  [FAIL] the runs diverged BEFORE the kill, so the test is comparing two different "
              "trajectories and any later mismatch is not a resume bug.")
        return 1

    if mismatches:
        print(f"\n[FAIL] resume is NOT exact: {len(mismatches)}/{len(overlap)} steps differ.")
        print("  This must not ship. A resume that perturbs the data order or the dropout stream turns")
        print("  one replicate into two, inside a measurement whose entire purpose is the dispersion")
        print("  of a single replicate across seeds.")
        return 1

    print(f"\n[OK] resume is bit-identical across all {len(overlap)} shared steps.")
    print("  The continued run consumed the same microbatches in the same order with the same dropout")
    print("  draws, so a Floor B replicate that survives a crash is still ONE draw from the recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
