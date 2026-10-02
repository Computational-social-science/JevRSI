# TOOLS — the index

Every tool this project owns, in the BootLoops convention: one row in the table, one four-question
page below it. **A tool missing any of the four questions is not finished, however well it runs.**

The four questions, for a reader who has never seen the tool:

1. **What it does** — one plain paragraph.
2. **When to reach for it** — the symptom or task that should route here, *and the neighbouring tool
   that covers the adjacent case*, so the boundary is visible.
3. **What its output means** — units, conventions, and **what failure looks like as distinct from
   success**. A tool whose failure output resembles its success output has that stated in bold.
4. **What test its answer must pass** before anyone believes it.

## Verification classes

Adopted from BootLoops verbatim, because the distinction is the useful part:

- **selftest** — the battery runs green from a clean clone, as shipped.
- **partial** — the legs that can run, run; legs needing absent data or an engine skip or fail closed
  with a named error.
- **smoke** — no battery of record; worked examples and refusal paths verified.
- **data-gated** — operates on data you supply; without it, **refuses loudly by design**.

## Instruments (`measurement/`) and infrastructure (`scripts/`)

Kept apart the way BootLoops keeps `tools/` and `ops/` apart. **Nothing in `measurement/` imports
from `scripts/`; nothing in `scripts/` computes a scientific result.** Infrastructure that produced a
number would be a second production route, which is what the independence rules exist to prevent.

## Index

| tool | what it is | class |
|---|---|---|
| `scripts/run_with_heartbeat.py` | gives a training run the progress unit it otherwise lacks, without editing the reference | selftest |
| `scripts/health.py` | the diagnostic chain — preflight / launch / status / validate / ledger / dashboard; the only supported way to launch a run | selftest |
| `scripts/export_health_state.py` | publishes `viz/state.js` for the live dashboard | selftest |
| `scripts/build_v1_env.py` | builds the v1.0 code environment and verifies it before declaring success | selftest |
| `scripts/check_arm0_spec.py` | spec validator; diffs against the release's published `meta.json` | selftest |
| `scripts/check_edit_guard.py` | EDITABLE / PROTECTED boundary for the subject repository | selftest |
| `scripts/check_synth_decontamination.py` | proves no split shares a state or a 12-word phrase with the distillation corpus | selftest |
| `measurement/report_from_items.py` | recomputes `pooled_top1` — the metric the release is verified in — from per-question rows | selftest |
| `measurement/build_synth_split.py` | converts the teacher corpus into the harness's question schema | partial |
| `paths.py` + `config/paths.json` | the single place a machine-specific location is declared | selftest |
| `viz/health_dashboard.html` | the live dashboard | smoke |
| `evidence/` | banked records — the value of record for each probe | data-gated |

---

## `scripts/run_with_heartbeat.py` — the progress unit

**What it does.** Runs the harness's training entry point unchanged and appends one JSON row per N
optimiser steps to `<arm>/progress.jsonl`: step index, timestamp, elapsed seconds, interval. Nothing
in the reference is edited — the counter is installed from outside by wrapping each optimizer
instance's `step`, which no subclass override can escape.

**When to reach for it.** For every arm, in place of calling the harness directly. A run without it
has **no progress signal at all** — only completion. **Neighbouring tool:** `health.py status`, which
reads what this produces; without a heartbeat there is nothing for it but activity, and activity is
what a stuck job emits too.

**Output.** The job's own time-stamped log — the only legitimate basis for projecting its remaining
time. A rate is fitted from the steady-state window, discarding the first rows as startup transient.
**An absent or stalled file while the process lives means the run is stuck**, distinguishable from a
slow run for the first time. **A severalfold drop in the interval voids any projection made from it**
— re-measure, do not keep quoting the old ETA.

**Believe it when:** a short known-length job's last heartbeat row agrees with the harness's own step
count. The mechanism is verified (12 steps at `every=3` yields `[0,3,6,9,12]`, AdamW's own `step`
covered); **the end-to-end check against the harness's count is still outstanding and must wait for
an idle GPU**, because running it now would contend with arm 0 — the exact error this project has
made most often.

## `scripts/health.py` — the diagnostic chain

**What it does.** Six subcommands over one pipeline. `preflight` checks every precondition for a run
and exits non-zero if any fails. `launch` runs `preflight` itself and refuses to start, then captures
the run's output to `<arm>/run.log`. `status` reports the live state of a running arm, separating
PREPARING from TRAINING from NOT RUNNING. `validate` checks a finished arm's completeness and
plausibility against the published reference numbers. `ledger` lists every arm with its environment
fingerprint. `dashboard` publishes the state the live page renders.

**When to reach for it.** Before spending any GPU time (`preflight`), to start a run (`launch`), to
check on one (`status`), to accept or reject its output (`validate`), to compare arms (`ledger`).
**Neighbouring tool:** `jevrsi_arm_watch.py` (outside this repository) is the cron-facing wrapper
that runs `status` and refreshes the dashboard on a schedule. If you want a one-shot look, run
`status`; if you want it every two hours, that is the watcher's job.

**Output.** Each check prints `[OK] / [WARN] / [FAIL] / [SKIP]` plus the failure it prevents.
**`WARN` never blocks a launch and `FAIL` always does** — deliberately, because a gate that blocks on
a warning gets disabled, and a disabled gate is worse than none. `preflight` exits 1 on any `FAIL`.
Two verdicts are reported separately and must stay separate: **run health** (`status` + `validate`)
answers "is the running arm sound", **launch readiness** (`preflight`) answers "could another run
start". Merging them once made a healthy running arm display FAIL merely because the card was full.

**Believe it when:** it has been shown to fail. Every check that guards a real failure has a
corresponding negative control, and the two that were built this way — the spec diff and the GPU
capacity check — have both been watched firing.

## `scripts/export_health_state.py` — dashboard state

**What it does.** Writes `viz/state.js` (a JS global) and `viz/state.json` holding everything the
dashboard renders: verdict, arm phase, targets versus anchors, all three check batteries, the arm
ledger, and source mtimes with ages.

**When to reach for it.** Never directly — it is called by `health.py dashboard`, which the watcher
runs. **Neighbouring tool:** the dashboard HTML itself, which reads the state; if the page is blank
the fault is here, not there.

**Output.** `window.APP_STATE` as valid JavaScript. **Failure looks like an absent file, and the
page then says so and shows nothing else** — there is deliberately no fallback data, because a
dashboard that invents plausible content is worse than a blank one and this project's costs are
denominated in believing wrong numbers.

**Believe it when:** the page reloads it. The poll is verified by poisoning `window.APP_STATE = null`
and confirming the value returns from disk within two intervals.

## `scripts/build_v1_env.py` — the v1.0 environment

**What it does.** Rebuilds the environment a release's code runs in: the published `code/<pkg>/`
modules plus the checkout's driver, then proves the borrowed driver's imports actually resolve against
the v1.0 modules before declaring success. Also reports which files are byte-identical, newline-only
different, or genuinely different between the two revisions.

**When to reach for it.** Before any run in the v1.0 environment; after any change to the reference
checkout. **Neighbouring tool:** `check_edit_guard.py` guards the *subject* repository's boundary;
this one guards the *reference* revision being used.

**Output.** A built directory plus per-file verdicts. **A `[FAIL]` line means the borrowed driver
would read a different module than the one that trained the release** — the run would be against the
checkout, not the release, and would not be a reproduction.

**Believe it when:** it has rebuilt an existing environment without changing a single source hash.
Verified, and it is also how the "byte-identical" error was caught: raw bytes differ, newlines
normalised they do not.

## `scripts/check_arm0_spec.py` — the spec validator

**What it does.** Checks ARM 0's spec against the spec the release publishes inside its own
checkpoint: field by field, with only explicitly whitelisted deviations allowed, each carrying the
measurement that justifies it.

**When to reach for it.** Before any run, and after any edit to the spec. **Neighbouring tool:**
`health.py preflight`, which runs this as one of its checks; call this one directly when diagnosing a
spec question rather than a launch.

**Output.** `[OK] n/n checks passed`, or `[FAIL]` with the drift named. **Failure pins the exact key
and both values** — `lr_head: published 0.0001 but ours is 0.001` — because a vague spec complaint is
what let two consecutive commits contradict each other.

**Believe it when:** the negative controls fire. Two are recorded: restoring `lr_head: 1e-3` and
restoring `eval_batch_size: 32` each produce the expected `[FAIL]`.

## `measurement/report_from_items.py` — the published metric

**What it does.** Recomputes `pooled_top1` — the metric the release states its own verification in —
from a run's per-question rows, and compares it against the zero-shot anchor and the published
reference.

**When to reach for it.** After any evaluation that produced an `items.jsonl`, to obtain the one
number both sides can be compared on. **Neighbouring tools:** `health.py validate` calls it, and its
`pooled_aurc` / `min_decision_score` come from the harness directly — this is for the metric the
harness does *not* record.

**Output.** Per-target and per-role `pooled_top1` under both option orders, the canonical-versus-
reversed gap, and the headroom to the reference. **A gap where the reference shows none is
evidence about what the model learned, not an error** — it means the readout is tracking presentation
position as well as option content; the file says so rather than flagging it.

**Believe it when:** it reproduces a number from the harness itself. The zero-shot control's
`typed_decisions` AURC is 0.4919 through the harness and the corresponding pooled metric is stable
across two independent runs to full float precision.

---

## Known gaps, recorded rather than covered

- **No concurrent-run safety.** `preflight` checks capacity but does not lease it; two launches could
  both pass and then collide. BootLoops' `ops/turnstile` solves this with a priority token and RAM /
  CPU-width ledgers on a shared machine. Copy that design if concurrency is ever needed.
- **No progress unit for an arm.** The reference harness writes nothing during training, so a running
  arm has no progress signal — only completion. A wrapper that emits a heartbeat row per N steps is
  the fix for arms we control; it does not exist yet. Recorded in `docs/BOOTLOOPS.md`.
- **`build_synth_split.py` is partial.** Its choice-question path was recovered from a silent
  dropping bug once; the battery does not yet cover a third schema variant.
