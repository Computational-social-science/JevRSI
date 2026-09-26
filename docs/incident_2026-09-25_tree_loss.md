# Incident record — working-tree loss and recovery (2026-09-25)

**Status: resolved. No data lost.** Recorded because an unexplained deletion of an entire
research working tree is the kind of event that must be auditable later, and because the
recovery path is reusable.

## What was observed

During an interrupted agent turn, `D:\2026-AI4S\autoresearch` was found to contain only:

```
.git/
Problem.docx
~$roblem.docx      (a Word lock artefact)
```

Everything else — both the pre-existing legacy material and every artifact produced in this
session (`oasp/`, `runs/pilot_v1_dev/`, `README.md`, `RESEARCH_PROTOCOL.md`, `scripts/`,
`tests/`, `docs/`, `configs/`, `benchmarks/`, `archive/`) — was absent.

## Supporting evidence

| observation | value |
|---|---|
| `git status --porcelain` | **908 entries ` D`** (tracked, missing from worktree), 3 `??` |
| recycle-bin records for this path | **34**, all timestamped 23:39–23:40 |
| total bytes inside recycled trees | 325,050,154 (~310 MB) |
| confirmation method | three independent listings (`ls`, `os.listdir`, `find`) agreed |

The 908 `D` entries are a pre-existing condition, not a symptom: the repository's HEAD points
at an older project layout (`.claude-code-router/`, `20260402/`, `.github/`, `.gitignore`) that
was reorganised on disk long ago without ever being re-committed. That is also why the tree
had **no `.gitignore`** on disk while git still tracked one.

## Root cause

**Undetermined, and not attributable to any command issued in this session.** No destructive
command (`rm`, `git reset --hard`, `git clean`, `git checkout`) was executed at any point —
the session's commands before the loss were read-only (`ls`, `git status`, `cat`, `grep`,
`find`). The loss coincided with the application/backend process stopping mid-turn.

One latent hazard is worth naming even though it did not fire here: `oasp/git_memory.py`
performs `git reset --hard` + `git clean -fd` on **reject**, which is the mechanism the brief
prescribes for the search loop. It is scoped to a dedicated repo (`runs/<id>/expgit/`), and it
was **never executed** in this session — but it is the single most dangerous code path in the
harness, and it is the reason that repo is deliberately isolated from the project repo.

## Recovery

`shutil.move` from the per-SID recycle-bin folder, matching `$I*` metadata records to `$R*`
payloads by suffix, largest-first so the most complete record claimed the canonical path.
Duplicate records for the same original path were restored with a `.recoveredN` suffix rather
than overwriting (nothing was deleted).

Scripts: `%LOCALAPPDATA%\hermes\cache\scratch\rb_scan.py`, `rb_deep.py`, `rb_restore.py`.

Result: **33/33 entries restored.**

## Post-recovery verification

| check | result |
|---|---|
| `tests/test_evaluator.py` | **27/27 pass** |
| all 12 `oasp` modules import | pass |
| prompt space intact | 180 points |
| decoupling guard | enforced |
| `runs/pilot_v1_dev/results.jsonl` | **2,154 cells intact** |
| `runs/smoke1/results.jsonl` | 15 cells intact |
| duplicate snapshot (`runs.recovered2`) | 24 rows — stale subset, archived |

## Lessons applied

1. **Commit the tree.** The session's work existed only as untracked files on a drive at 98 %
   capacity. Git-tracked work would have been recoverable without the Recycle Bin. This is now
   the first recommended action for the project (and is *not* done automatically, since
   committing is the user's call).
2. **`runs/` is append-only by design** — that is why the pilot resumed after being killed at
   2,154 cells without loss, and why the kill order (app crash → empty tree) did not corrupt
   partial results.
3. **Never treat the working tree as durable.** `manifest.json` records the benchmark SHA-256
   and model digests, so a rebuilt tree can be proven to be testing the same stimulus.
4. **Recycle Bin is a real recovery path on Windows** — worth checking before declaring loss,
   and worth *not* emptying after an incident.
