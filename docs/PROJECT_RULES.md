# PROJECT RULES — binding on every artefact in this repository

**Recorded 2026-09-29.** These are project-level rules, not preferences. They are enforced by
mechanical checks, because a rule that is only written down is a rule that is eventually broken by a
session that did not read it.

---

## Rule 1 — Every document follows Nature-level scientific standards

This applies to manuscripts, decision records, pre-registrations, reports, READMEs, code comments,
docstrings, and the prose fields of any JSON artifact that a human reads.

Concretely, that means:

| requirement | what it forbids in practice |
|---|---|
| **Every number is traceable to an artifact** | a figure in a document that no script in this repository produced |
| **Provenance is explicit** | "the result is ~0.4 pp" with no split, no seed count, no df, no date |
| **The substrate is named** | a number from one model quoted as if it belonged to another |
| **A claimed absence is stated as an absence, not as a null** | "no effect" where the honest statement is "not measured, and here is why" |
| **Method and data are separated** | a conclusion stated in the same register as a measurement |
| **Negative results are first-class** | a failed experiment omitted from a summary |
| **Uncertainty is always attached** | a point estimate with no CI, no SD, no df |
| **Re-running must reproduce** | a stated procedure that depends on conversational context |

These are not aspirations. They are the properties `scripts/check_object_purity.py`,
`scripts/check_terminology.py` and `scripts/check_language.py` exist to protect, together with
`measurement/audit_pipeline.py` and the pre-flight gate in `measurement/run_floor_b_lora.py`.

## Rule 2 — English is the language of every result

**All documents, comments, docstrings, commit messages, and artifact prose are in English.** This
includes but is not limited to manuscripts, inline code comments, and the descriptive fields of JSON
outputs.

Why this is a rule and not a preference: the artefacts of this project are meant to be read by people
who did not attend the sessions that produced them, and to be checked by reviewers who work in
English. A mixed-language record is also a record whose meaning depends on who wrote it, which is
exactly the property that makes a scientific claim hard to audit.

**Chinese and other non-English material is permitted in exactly two places**, both of which are
citations rather than results:

1. **Quoted foreign-language terms that are functionally load-bearing.** The clearest case is a
   detection pattern: `scripts/check_object_purity.py` matches `落地形态` and `控制变量` because a
   source document used those words, and deleting the Chinese form would silently blind the guard to
   it. Each such exemption is listed in `scripts/check_language.py` and printed by that check, so the
   exemption list cannot grow unnoticed.
2. **Verbatim quotation of a source document**, marked as a quotation.

Everything else — including any sentence an agent is merely more fluent in — is English.

```bash
python scripts/check_language.py
```

## Rule 3 — Vocabulary is checked, not assumed

`docs/TERMINOLOGY.md` fixes the meaning of "autoresearch" (it means karpathy/autoresearch and nothing
else) and bans the compound "bilevel autoresearch". `scripts/check_terminology.py` enforces it.

## Rule 4 — Destructive operations are not delegated

Files are moved, never deleted, when reclassifying work. A subagent that is asked to archive must be
given a move target and a verification step; earlier, three agents asked to quarantine retired work
deleted roughly 92 files instead, and because none of them had ever been committed, nothing was
recoverable. The lesson is recorded here rather than only in a session log, because the failure is
invisible in the result: a clean tree looks exactly like a correct one.

## Rule 5 — A measurement is not a claim until its noise is measured

A borrowed Floor B, a nominal runtime, or a spot-checked timing may be used to PROVISIONALLY plan, and
must be labelled provisional at the point of use. It may not appear in a success criterion, an accept
threshold, or a comparison table without being re-measured on the substrate it is applied to.
`measurement/INSTRUMENT_CALIBRATION.json` records which numbers are currently provisional.

## Rule 6 — A run that stops cleanly is a success

Stopping a batch, abandoning a branch, or reporting "no gain" counts as a completed result and is
written up like one. A six-week run that plateaus is still required to ship its full failure archive
(this is also the objective's own Rule 5).
