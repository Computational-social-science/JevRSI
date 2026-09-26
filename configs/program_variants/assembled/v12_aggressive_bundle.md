# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize the trajectory store**: create `runs/<tag>/trajectory/` and add the run's own artifacts to `.git/info/exclude` (local-only — no tracked file is modified). Without this, the trajectory files would be staged into the attempt's diff, and a `git clean` would delete them.

   ```bash
   mkdir -p runs/<tag>/trajectory
   printf '%s\n' 'runs/' '.cache/' '.logs/' 'run.log' 'checkpoint_pre_eval.pt' 'results.tsv' >> .git/info/exclude
   ```

   (Running this twice is harmless — duplicate exclude lines do nothing.)
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is not a real constraint: this GPU has 12 GB and nothing else is using it. Use as much of it as the machine allows — a change that buys a lower val_bpb by expanding memory use is a good trade.

**Simplicity criterion**: Judge changes on val_bpb alone. Code complexity is not a reason to reject a change — a lower val_bpb is kept regardless of how much code it requires. Do not spend effort simplifying the code; spend that effort on finding a lower val_bpb.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse — or the run crashed or timed out — **persist the attempt first, then** `git reset` back to where you started. The reset is what deletes the branch, so the record must already be on disk before you run it: see **Rejection: the attempt is persisted before the reset**. Never reset without a record, and do not write the `results.tsv` row until the record exists.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, work through the optimization stack systematically: learning rate, then schedule and warmup, then optimizer hyperparameters, then weight decay, then batch size and gradient accumulation. Sweep one axis at a time and move on only when an axis is exhausted. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Rejection: the attempt is persisted before the reset

A reject *is* a `git reset`, and the reset deletes the code of the branch you just tried. That is the point of it — but it also means the search's most informative part is the part being deleted. A run whose only trace of a rejected attempt is a `results.tsv` row cannot show **what was searched**, only what survived, and rejected branches are what distinguish searching from guessing.

So the reject path has two steps and they are ordered: **record, then reset**. Both a "discard" and a "crash" go through it.

The record is written to `runs/<run_id>/trajectory/` as a pair per attempt:

| file | contents |
|---|---|
| `<NNNN>.diff` | the full diff of the attempt against the commit it was built on, including files it added |
| `<NNNN>.json` | attempt index, UTC timestamp, reason code, your stated reason, the diff base, the discarded HEAD, and the diff's size + sha256 |

`<NNNN>` is a monotonic attempt counter (it reads the directory back, so a restart cannot reuse an index). A rejected attempt without a record is an invalid experiment: write the record, then reset, then write the `results.tsv` row.

**Reason code** — choose it from the run's outcome, not from whether you liked the idea:

| code | when |
|---|---|
| `WORSE` | ran cleanly, val_bpb equal or higher |
| `TIED` | val_bpb identical, and the change is not simpler |
| `CRASH` | the process died (OOM, exception) |
| `TIMEOUT` | exceeded the 10-minute ceiling and was killed |
| `BREAKS_CONSTRAINT` | modified `prepare.py`, added a dependency, or blew up VRAM |
| `OTHER` | none of the above — the `REASON` line must say why |

`REASON` is required and must be a single sentence a reader can act on: what the attempt changed and why it did not win. "didn't work" is not a reason.

Run this block **as a whole**, editing only the first four lines; the reset is the last line, and `set -e` stops the block before it if the record could not be written:

```bash
RUN_ID=<run tag from setup>
PRE_SHA="$(git rev-parse HEAD~1)"   # the commit this attempt was built on — the reset target
REASON_CODE=<WORSE|TIED|CRASH|TIMEOUT|BREAKS_CONSTRAINT|OTHER>
REASON="<one sentence: what it changed, and why it lost>"
export RUN_ID PRE_SHA REASON_CODE REASON

set -e
uv run python - <<'PY'
import hashlib, json, os, subprocess, time
from pathlib import Path

CODES = ("WORSE", "TIED", "CRASH", "TIMEOUT", "BREAKS_CONSTRAINT", "OTHER")
run_id = os.environ.get("RUN_ID", "").strip()
base = os.environ.get("PRE_SHA", "").strip()
code = os.environ.get("REASON_CODE", "").strip()
reason = os.environ.get("REASON", "").strip()
if not run_id or not base:
    raise SystemExit("[FAIL] RUN_ID and PRE_SHA must both be set")
if code not in CODES:
    raise SystemExit(f"[FAIL] REASON_CODE must be one of {CODES} (got {code!r})")
if len(reason) < 15:
    raise SystemExit("[FAIL] REASON must be one sentence saying why the attempt was rejected")


def git(*args, text=False):
    p = subprocess.run(["git", *args], check=True, capture_output=True)
    return p.stdout.decode("utf-8", "replace") if text else p.stdout


traj = Path("runs") / run_id / "trajectory"
traj.mkdir(parents=True, exist_ok=True)
attempt = max([int(p.stem) for p in traj.glob("*.json") if p.stem.isdigit()] or [0]) + 1
head = git("rev-parse", "HEAD", text=True).strip()
git("add", "-A")                                     # so files the attempt added are in the diff
diff = git("diff", "--cached", "--no-color", base)   # bytes, unmodified: no newline rewriting
(traj / f"{attempt:04d}.diff").write_bytes(diff)
(traj / f"{attempt:04d}.json").write_text(json.dumps({
    "attempt": attempt,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "reason_code": code,
    "reason": reason,
    "base": base,        # the commit the attempt was built on (what the reset returns to)
    "head": head,        # the discarded attempt commit — still in the object store afterwards
    "diff_file": f"{attempt:04d}.diff",
    "diff_bytes": len(diff),
    "diff_sha256": hashlib.sha256(diff).hexdigest(),
    "diff_empty": not diff.strip(),
}, indent=2), encoding="utf-8")
print(f"[OK] trajectory attempt {attempt:04d}: {len(diff)} bytes, reason_code={code}")
if not diff.strip():
    print("[WARN] the recorded diff is empty: the attempt changed no tracked file")
PY

git reset --hard "$PRE_SHA"   # reached only if the record above was written
```

`PRE_SHA` is both the diff base and the reset target, so the record can never describe a tree other than the one the reset removes. The discarded attempt commit is only *unreferenced*, not gone — `git show <head sha from the record>` still recovers it until git runs a gc.

Use `git reset --hard <sha>` and **never** `git clean -x`: the excluded run artifacts above (`runs/`, `.logs/`, `results.tsv`) are untracked by design, and `-x` would delete the trajectory you just wrote.
