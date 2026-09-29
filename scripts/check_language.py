#!/usr/bin/env python3
"""check_language.py -- enforce Rule 2 of docs/PROJECT_RULES.md: English is the language of results.

WHY A MECHANICAL CHECK

A language rule that lives only in a document is a rule that breaks the first time a session is more
fluent in another language, which on this project is frequently. The check is here because the failure
is otherwise invisible: a half-Chinese decision record still parses, still has every number, and still
looks like a finished document to anyone skimming it. What it loses is not correctness but
reviewability, which is the property the rule exists to protect.

SCOPE, AND THE TWO THINGS THIS DOES NOT FLAG

Flagged: CJK ideographs and CJK punctuation in any human-readable artefact -- markdown, source
comments, docstrings, and the prose fields of JSON.

Not flagged, by design:

  * `archive/` -- history, deliberately kept as written. Rewriting history to satisfy a present rule
    would destroy the record of what was actually decided.
  * binary sources (`.docx`, `.pdf`, `.svg`) -- the original brief and the cited papers. They are
    sources, not results.
  * the two documented exemptions below, which are citations rather than results.

THE EXEMPTION MECHANISM IS DELIBERATELY AWKWARD

An exemption is a literal string plus the reason it exists, and every exemption in force is PRINTED on
every run. That is the point: an exemption list that can be edited silently is indistinguishable from
no check at all, so this one makes itself visible every time it runs. Adding to it is a deliberate act
with a written justification attached, not a quiet edit.

The live case is `scripts/check_object_purity.py`, which matches two Chinese terms because a source
document used them. Deleting the Chinese form would not tidy anything -- it would blind the purity
guard to the exact vocabulary it was written to catch, and the blindness would be invisible until the
retired concept reappeared.

Usage:
    python scripts/check_language.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Han ideographs, plus CJK punctuation that only ever appears in CJK prose.
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

# ---------------------------------------------------------------------------------------------
# Exemptions. Each is (relative path, exact literal, reason). The literal is removed before the
# file is scanned, so the exemption covers precisely that string and nothing adjacent to it.
# ---------------------------------------------------------------------------------------------
EXEMPTIONS: list[tuple[str, str, str]] = [
    ("scripts/check_object_purity.py", "落地形态",
     "detection pattern for a retired concept that a source document expressed in Chinese; "
     "removing the Chinese form would stop the purity guard matching the vocabulary it exists to catch"),
    ("scripts/check_object_purity.py", "控制变量",
     "detection pattern for a retired concept that a source document expressed in Chinese; "
     "same reason -- the pattern is load-bearing, not prose"),
    ("scripts/check_language.py", "落地形态",
     "this file quotes the exemption above so the reason is readable at the point of enforcement"),
    ("scripts/check_language.py", "控制变量",
     "this file quotes the exemption above so the reason is readable at the point of enforcement"),
    ("docs/PROJECT_RULES.md", "落地形态",
     "Rule 2 cites the exemption to show what a legitimate non-English citation looks like"),
    ("docs/PROJECT_RULES.md", "控制变量",
     "Rule 2 cites the exemption to show what a legitimate non-English citation looks like"),
]

SKIP_DIRS = ("archive/", ".git/", "__pycache__", ".cache", ".pytest_cache", "node_modules")
SCAN_SUFFIXES = {".py", ".md", ".json", ".sh", ".yaml", ".yml", ".txt", ".toml", ".cfg"}

# Prose fields of a JSON artifact that are allowed to hold either language, because they hold quoted
# source material rather than results. Everything else in a JSON file is scanned.
JSON_QUOTED_FIELDS = {"title_zh", "quote", "source_text", "original"}


def strip_exemptions(rel: str, text: str) -> tuple[str, list[str]]:
    """Remove exempt literals. Returns the scannable text and the exemptions that applied."""
    applied = []
    for path, literal, _ in EXEMPTIONS:
        if path == rel and literal in text:
            text = text.replace(literal, "")
            applied.append(f"{literal}  ({_why(path, literal)})")
    return text, applied


def _why(path: str, literal: str) -> str:
    for p, lit, reason in EXEMPTIONS:
        if p == path and lit == literal:
            return reason
    return ""


def scan_file(path: pathlib.Path, rel: str) -> tuple[list[tuple[int, str]], list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:                                          # pragma: no cover
        return [(0, f"unreadable: {exc}")], []

    if path.suffix.lower() == ".json":
        import json

        def strip_fields(o):
            """Blank out fields that legitimately hold quoted source material. Defined before use so
            the name is always bound -- a conditionally-defined nested function is a latent
            UnboundLocalError the day the branch order changes."""
            if isinstance(o, dict):
                return {k: ("" if k in JSON_QUOTED_FIELDS else strip_fields(v))
                        for k, v in o.items()}
            if isinstance(o, list):
                return [strip_fields(v) for v in o]
            return o

        try:
            text = json.dumps(strip_fields(json.loads(text)), ensure_ascii=False)
        except (ValueError, TypeError):
            pass                                                    # fall through to raw scan

    text, applied = strip_exemptions(rel, text)
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        found = CJK.findall(line)
        if found:
            hits.append((i, f"{''.join(sorted(set(found)))}  |  {line.strip()[:90]}"))
    return hits, applied


def main() -> int:
    print("=" * 76)
    print("language check -- Rule 2: English is the language of every result")
    print("=" * 76)
    print(f"root: {ROOT}")
    print(f"scanning: {', '.join(sorted(SCAN_SUFFIXES))}  (binary sources and archive/ excluded)")

    violations: list[tuple[str, int, str]] = []
    exemptions_in_force: list[str] = []
    files = 0
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIXES:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_DIRS):
            continue
        files += 1
        hits, applied = scan_file(p, rel)
        for a in applied:
            exemptions_in_force.append(f"{rel}: {a}")
        for line, detail in hits:
            violations.append((rel, line, detail))

    print(f"files scanned: {files}")
    print(f"\nExemptions in force ({len(exemptions_in_force)}) -- every one is a citation, not a result:")
    for e in exemptions_in_force:
        print(f"  [exempt] {e}")
    if not exemptions_in_force:
        print("  none")

    if violations:
        by_file: dict[str, list[tuple[int, str]]] = {}
        for rel, line, detail in violations:
            by_file.setdefault(rel, []).append((line, detail))
        print(f"\nVIOLATIONS ({len(violations)} CJK runs across {len(by_file)} files):\n")
        for rel, rows in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
            print(f"  {rel}   ({len(rows)} lines)")
            for line, detail in rows[:3]:
                print(f"      L{line}: {detail}")
            if len(rows) > 3:
                print(f"      ... and {len(rows)-3} more")
        print("\nRule 2 of docs/PROJECT_RULES.md requires these in English.")
        print("If a CJK string here is a genuine citation, add it to EXEMPTIONS in this file with a")
        print("written reason -- do not suppress the pattern and do not widen the skip list.")
        return 1

    print("\nRESULT: PASS -- every document, comment and docstring in the live tree is English.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
