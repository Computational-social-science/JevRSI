#!/usr/bin/env python3
"""check_terminology.py -- fail the build on terminology that misattributes work or conflates concepts.

WHY THIS FILE EXISTS

On 2026-09-29 an agent introduced the compound "bilevel autoresearch" while naming the pipeline
skeleton. It reads naturally, nothing errors, and the drift is invisible until a later session inherits
it as vocabulary and reasons from it. That is the same class of failure this repository's
`check_object_purity.py` was written for -- a concept that should be absent surviving into the live
tree -- except that here the concept is not retired, it is *foreign*, and it is welded to a real,
borrowed project.

Two things were wrong with the phrase, and both are checked:

  1. It implies a lineage from karpathy/autoresearch that does not exist. This project BORROWS that
     harness; it does not fork it again or extend it, and calling our own scheduler a kind of
     "autoresearch" implies a derivation the history does not support.
  2. "Bilevel optimization" is a standard term from the optimization literature and this project uses
     no such optimizer. Pairing it with a named external repository makes an unrelated term look like a
     design property of that repository.

The failure mode is silent, so the check has to be mechanical. A human reading a docstring will not
notice, and will not notice again next week.

WHAT IT DOES AND DOES NOT FORBID

It forbids the COMPOUND and a few specific misattributions. It does not forbid the words "outer" and
"inner", does not forbid using "autoresearch harness" (that is the correct usage), and does not touch
`archive/`, which is history and is meant to read exactly as it was written.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (pattern, what it is, why it is wrong). Patterns are matched case-insensitively against the
# normalised text of every scanned file.
FORBIDDEN: list[tuple[str, str, str]] = [
    (r"bi-?\s?level\s+autoresearch",
     "the compound 'bilevel autoresearch'",
     "conflates karpathy/autoresearch (a real borrowed repository) with bilevel optimization (an "
     "unrelated optimisation technique this project does not use). See docs/TERMINOLOGY.md"),
    (r"autoresearch[- ]win-rtx\s+(is|as)\s+(ours|our\s+own)",
     "claiming the Windows fork as our own code",
     "jsegov/autoresearch-win-rtx is jsegov's fork of karpathy/autoresearch. We borrow its ideas; "
     "claiming it is ours misattributes another party's work, which Rule 4's comparison table depends "
     "on being accurate"),
    (r"(we|our)\s+(fork|forked)\s+of\s+(karpathy/)?autoresearch",
     "claiming to have forked autoresearch",
     "this project did not fork it; it consumes a fork that already exists"),
    (r"(goal|objective|aim|target)\s+(is\s+)?(to\s+)?re-?train(ing)?\s+nanochat",
     "retraining nanochat as a stated project goal",
     "nanochat is the harness's own substrate, borrowed as an instrument. The objective's substrate is "
     "typed-decisions on a non-generative scoring model. Naming nanochat while RUNNING the harness is "
     "correct; making it the object of study is not"),
]

# The correct usage, checked positively so the guard also catches an over-correction in which someone
# bans the word "autoresearch" outright and loses the reference to the harness we actually depend on.
REQUIRED_CONTEXT = re.compile(r"karpathy/autoresearch|autoresearch harness|autoresearch-win-rtx",
                              re.IGNORECASE)

SCAN_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".txt"}

# ---------------------------------------------------------------------------------------------
# Ruling documents are exempt, and the exemption is narrow on purpose.
#
# A prohibition that cannot name what it prohibits is not a prohibition. These three files ARE the
# instruments that define the ban -- the SSOT a session loads first, the rules document, and the
# terminology ruling -- so each has to be able to quote the compound in order to forbid it. This is the
# same principle as the language guard's load-bearing-citation exemption: a legitimate non-English or
# forbidden string is a CITATION here, not a use.
#
# The exemption is deliberately not a directory pattern. Adding a fourth file to this list is a
# deliberate act with a written justification, and every entry in force is printed on every run so the
# list cannot grow unnoticed.
# ---------------------------------------------------------------------------------------------
RULING_DOCS = {
    "docs/TERMINOLOGY.md": "the ruling itself: it must quote the compound in order to ban it",
    "docs/PROJECT_RULES.md": "Rule 3 states that the compound is banned, which requires naming it",
    "CURRENT_OBJECT.md": "the SSOT carries the vocabulary ruling so a fresh session cannot miss it",
    "scripts/check_terminology.py": "the guard must be able to name what it forbids",
}


def scan() -> list[tuple[str, str, str, int]]:
    hits = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        rel = p.relative_to(ROOT)
        s = str(rel).replace("\\", "/")
        # archive/ is history, kept deliberately unmodified; .git and build noise are not content.
        if s.startswith(("archive/", ".git/", "runs/")) or "__pycache__" in s:
            continue
        if s in RULING_DOCS:
            continue        # the instruments that define the ban must be able to name it
        # Two files are EXEMPT, and the exemption is narrow and deliberate.
        #   scripts/check_terminology.py  -- the guard must be able to name what it forbids
        #   docs/TERMINOLOGY.md           -- the ruling has to quote the phrase it prohibits
        # Without this the guard fails its own source, which is the fastest way to get a check
        # disabled. Everything else in the live tree is subject to it.
        if s in ("scripts/check_terminology.py", "docs/TERMINOLOGY.md"):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Collapse newlines so a phrase split across two lines is still caught.
        flat = re.sub(r"\s+", " ", text)
        for pat, what, why in FORBIDDEN:
            for m in re.finditer(pat, flat, re.IGNORECASE):
                line = flat[:m.start()].count("\n") + 1
                hits.append((s, what, why, line))
    return hits


def main() -> int:
    print("=" * 74)
    print("terminology check -- autoresearch means karpathy/autoresearch, and nothing else")
    print("=" * 74)
    print(f"root: {ROOT}")
    print(f"\nRuling documents exempt ({len(RULING_DOCS)}) -- each must be able to name what it forbids:")
    for f, why in RULING_DOCS.items():
        print(f"  [exempt] {f:34s} {why}")

    hits = scan()
    if hits:
        print(f"\nVIOLATIONS ({len(hits)}):\n")
        for f, what, why, line in hits:
            print(f"  {f}  (near line {line})")
            print(f"      {what}")
            print(f"      {why}\n")
        print("Fix the wording. Do not suppress the pattern.")
        return 1

    # Positive check: the correct reference must still exist somewhere, or the guard has been
    # over-applied into an outright ban on the word and the link to the harness has been lost.
    readme = ROOT / "docs" / "TERMINOLOGY.md"
    ok_context = readme.exists() and REQUIRED_CONTEXT.search(readme.read_text(encoding="utf-8"))
    print(f"\n  no forbidden compound found in the live tree")
    print(f"  [{'OK  ' if ok_context else 'WARN'}] the correct attribution is documented "
          f"(karpathy/autoresearch named in docs/TERMINOLOGY.md)")
    print("\nRESULT: PASS -- vocabulary matches the project's actual lineage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
