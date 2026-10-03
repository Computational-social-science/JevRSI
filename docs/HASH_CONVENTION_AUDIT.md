# Hash convention audit — 2026-10-03

Every check in this repository that compares bytes, what it decided, and whether its verdict
depended on the machine rather than on the artefact.

## What prompted it

Adopting the BootLoops toolkit surfaced a failure mode worth generalising:

```
VendorTamperError: BALLER vendor integrity FAILED (tool refuses to compute):
  geo/mc.py: 732444e7cd290b73 != pin 20508a7a3c058d39
```

The pin is correct. Git for Windows converted LF to CRLF on checkout, and BootLoops ships no
`.gitattributes`, so every sha256 pin over a text file failed on content-identical files. Proven by
hashing both ways: `raw 732444e7...` vs `newline-normalised 20508a7a3c058d39`, and the pin is
`20508a7a3c058d39`.

The question this raised about our own guards is not "do we have the same bug" but "does any
verdict here depend on a fact about the machine instead of a fact about the artefact".

## The four sites, audited by reading rather than by searching

Classifying these by grepping for `hashlib` puts two of them in the wrong column. Each was read.

| Site | What it decides | Was | Now |
|---|---|---|---|
| `scripts/check_edit_guard.py` `sha256()` | whether a PROTECTED file changed | **raw bytes** | **normalised** |
| `scripts/build_v1_env.py` `sha()` | whether two revisions contain the same code | already normalised (default `normalize_newlines=True`) | unchanged |
| `scripts/health.py` `sha_file()` | arm and environment fingerprints | already normalised | unchanged |
| `scripts/health.py` `spec_repo_match` | whether the arm's spec equals the repo's | **inert** — see below | **answers** |

### The one real defect: `check_edit_guard.py`

Measured on the three PROTECTED files:

```
scripts/evaluate_split.py          CRLF 0    raw ec547eb1...  normalised ec547eb1...   same
scripts/autoresearch_agent.py      CRLF 0    raw 2ba2836b...  normalised 2ba2836b...   same
typed_decisions/experiment.py      CRLF 156  raw ca54b8e1...  normalised 27a6f29c...   DIFFERENT
```

The declaration is made from raw bytes, so `experiment.py`'s entry was a property of the checkout
that produced it. Had that file crossed a platform, the guard would have printed:

```
[FAIL] MODIFIED  typed_decisions/experiment.py
```

on a file nobody had touched — and the natural repair would have been to re-declare, which is
precisely how a guard gets quietly disarmed.

The declaration's own `_measured` note says the boundary is *"pinned by content rather than by
import"*. Raw bytes are not content, so the fix **restores** the stated contract rather than
relaxing it. The two files without CRLF were byte-identical under both conventions and their
declarations did not change — the fix is backward-compatible with everything that was already
unambiguous.

### The inert check: `spec_repo_match`

```python
"spec_repo_match": (spec.read_bytes() == (ROOT / "config" / spec.name).read_bytes())
                    if (ROOT / "config" / spec.name).is_file() else None,
```

`spec.name` is `spec.json`, so this looked for `config/spec.json`, which has never existed — the
file is `config/arm0_spec.json`. The `is_file()` guard is good defensive practice, so the field
became `None` on every launch instead of raising, and nothing consumes it, so it was inert rather
than wrong. A check that always answers "no claim" is indistinguishable from a check that has no
opinion, and neither is evidence.

Now it maps `arm0` → `config/arm0_spec.json` and answers:

```
arm0           matches=True  against=config/arm0_spec.json
arm0_probe_v1  matches=None  against=None
```

`None` still means "this repository declares nothing for that arm", which is a different statement
from "the arm's spec differs".

### The label inconsistency: `spec_sha256`

`health.py` writes this field with `sha_file()` (CRLF-normalised). `evidence/arm0_LAUNCH.json` holds
`8712e49a138db22b...`, which is the **raw-byte** hash of the same file — verified: the raw hash
matches and the normalised hash of the same file does not.

That record is history and is left as written. What changes is that new records state the
convention, so the two are no longer two readings of one name:

```json
"spec_sha256": "...",
"spec_sha256_convention": "sha256 over bytes with CRLF normalised to LF"
```

The field is display-only — nothing compares it — so this was a naming defect, not a live false
alarm.

## Proof that the fix is correct

Fewer failures is what a broken guard also produces, so the fix carries its own controls. The
self-test now includes:

```
LF and CRLF forms of one file hash alike     -> accepted (presentation-blind)
a one-character edit hashes differently      -> rejected (content-sensitive)
```

The four original negative controls still reject, and the clean state still passes:

```
[OK] the guard can fail and does not fail spuriously
[OK] 14/14 checks passed
```

## The convention, stated once

**Content hashes normalise CRLF to LF before hashing.** A verdict must be a fact about the
artefact, not about the machine that looked at it.

`.gitattributes` (added at the repository root) decides the checkout policy so the cause stops
appearing; the checkers normalise so the verdict stops depending on it. Both, not either: a
checkout policy only governs checkouts, and a file can still arrive by copy.

Verified: adding `.gitattributes` left all ten CRLF-tracked files clean in `git status`, so it
changes future checkouts without rewriting present content.

## What was not changed

* `evidence/arm0_LAUNCH.json` keeps its raw-byte hash. It is a launch record; rewriting it to match
  a later convention would make the record agree with us at the cost of being true.
* No commit. The three modified files and the new `.gitattributes` are staged for review, not
  committed, because the line-ending policy is a decision about the repository's history.
