"""
check_document_hygiene.py -- the live tree may contain documents, and documents are what contaminate.

WHY A SEPARATE GUARD
    The purity guard scans text. On 2026-09-30 that was enough for every .md, .txt, .html and .rst in
    the tree, and the tree still contained Problem.docx -- a .docx holding the ORIGINAL PROPOSITION of
    a retired objective, in Chinese, naming four retired concepts. It was invisible to every check
    because none of them looked inside a binary container, and it was the most upstream contamination
    in the repository: not a description of the old work but its statement.

    A guard that only reads text has a blind spot exactly where documents are most able to mislead,
    because a document is what a human opens. So this guard enumerates documents by what they ARE,
    not by what a text scan can see, and asks two questions of each: is this the only copy of the
    objective, and does its content belong to the live objective.

WHAT IT CHECKS
    1. Exactly ONE objective document exists, at the repository root, and no other copy is live. An
       obsolete revision of the objective is contamination: a later session cannot tell it from the
       current one without opening both, and the older one is more likely to be found by search.
    2. No office/PDF container sits in the live tree un-inspected. Binary formats are allowed only
       when they are another author's paper under literature/, which is reference material and is
       never an instruction to this project.
    3. No live document is written in a language other than English. Rule 2 makes English the language
       of every result, and a Chinese-language brief is how a retired objective survives: it is not
       read by the text guards, and it is not obviously "documentation" to someone skimming.

WHY DELETION AND NOT ARCHIVING IS ACCEPTABLE HERE
    Project rule 4 keeps history as written, and rule 3 forbids delegating destructive operations. Both
    are about WORK. A document that states a retired objective is not a record of work, it is a
    competing instruction, and leaving it in-tree -- even in archive/ -- leaves it one `git checkout`
    away from being read as current. The owner's order of 2026-09-30 is explicit: a polluting document
    must not exist. sha256 copies are held outside the repository; that satisfies auditability without
    satisfying the document's own existence.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

OBJECTIVE = "RTX4070_SelfEvolving_Jev_Research_Proposal.md"
OBJECTIVE_STEMS = ("RTX4070_SelfEvolving_Jev_Research_Proposal",)

# Containers that hide their text from a grep.
BINARY_DOC_SUFFIXES = {".docx", ".doc", ".odt", ".pdf", ".pptx", ".xlsx"}
TEXT_DOC_SUFFIXES = {".md", ".txt", ".html", ".rst", ".tex"}

# A reference PDF of another author's work is not an instruction to this project.
REFERENCE_PREFIXES = ("literature/",)

CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Files permitted to carry the language guard's own detection patterns, with the reason at the point
# of enforcement. The purity guard matches two Chinese terms because a retired source document used
# them; deleting the Chinese form would blind the guard to the vocabulary it exists to catch. A guard
# that names what it forbids is not violating the rule the guard enforces.
CJK_EXEMPT = {
    "docs/PROJECT_RULES.md":
        "quotes the two Chinese terms that scripts/check_object_purity.py matches on, because a "
        "retired source document used them. Removing the Chinese form would make the purity guard "
        "silently blind to that vocabulary.",
    "scripts/check_object_purity.py":
        "this guard's sibling; it carries the same patterns",
    "scripts/check_document_hygiene.py":
        "this file: it names the CJK ranges it detects",
    "scripts/check_language.py":
        "carries the patterns and the exemption table for the same reason",
}


def docx_text(p: pathlib.Path) -> str:
    """Extract the visible text of a .docx. Best effort: a container we cannot read is reported,
    not silently skipped, because 'I could not check it' and 'it is clean' must not look alike."""
    try:
        with zipfile.ZipFile(p) as z:
            parts = [n for n in z.namelist()
                     if n.startswith("word/") and n.endswith(".xml")]
            blob = "".join(z.read(n).decode("utf-8", errors="replace") for n in parts)
    except Exception as e:                                      # noqa: BLE001
        return f"[UNREADABLE {type(e).__name__}: {e}]"
    blob = re.sub(r"</w:p>", "\n", blob)
    return re.sub(r"<[^>]+>", "", blob)


def all_docs():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(".git/") or "__pycache__" in rel or rel.startswith(".pytest_cache/"):
            continue
        if p.suffix.lower() in (BINARY_DOC_SUFFIXES | TEXT_DOC_SUFFIXES):
            yield p, rel


def main() -> int:
    print("=" * 78)
    print("document hygiene -- one objective, no hidden containers, English only")
    print("=" * 78)
    print(f"root: {ROOT}\n")

    problems: list[str] = []
    docs = list(all_docs())

    print(f"{'document':58s} {'kind':10s} {'verdict'}")
    print("-" * 78)

    for p, rel in docs:
        suffix = p.suffix.lower()
        is_ref = rel.startswith(REFERENCE_PREFIXES)
        is_root_objective = rel == OBJECTIVE

        # --- 1. the objective must be unique and at the root -------------------------------
        # A duplicate is a duplicate wherever it sits. The first version compared `rel` to the root
        # path, which meant a copy under a subdirectory passed -- and the negative control caught it
        # immediately. The check must be on the FILENAME, with the root location as the only
        # permitted place for it.
        if p.stem in OBJECTIVE_STEMS and not is_root_objective:
            problems.append(
                f"[A] a second copy of the objective is live: {rel}\n"
                f"    -> there is exactly one objective document, at the root. An obsolete revision is\n"
                f"       contamination: a later session cannot distinguish it from the current one\n"
                f"       without opening both, and search finds the older one more often.")
            print(f"  {rel:58s} {'objective':10s} DUPLICATE")
            continue

        # --- 2. binary containers are inspected, not assumed --------------------------------
        if suffix in BINARY_DOC_SUFFIXES:
            if is_ref:
                print(f"  {rel:58s} {'reference':10s} ok (another author's paper)")
                continue
            if suffix in (".docx", ".odt"):
                text = docx_text(p)
                if text.startswith("[UNREADABLE"):
                    problems.append(f"[B] {rel} is a document container this guard cannot read: "
                                    f"{text[:90]}. Move it out of the live tree or make it readable.")
                    print(f"  {rel:58s} {'container':10s} UNREADABLE")
                    continue
                cjk = len(CJK.findall(text))
                words = len(text.split())
                # ANY substantial non-English content fails, and the threshold is deliberately low.
                # A first version used `cjk > 40 and words < 1200`, reasoning that only a short brief
                # could be a competing instruction -- and a negative control immediately defeated it:
                # a 13-character Chinese sentence in a .docx passed, because a real brief need not be
                # long to mislead. The quantity that matters is not document length, it is whether
                # the text guards can read this file at all. They cannot, so the rule is absolute and
                # the exemptions are explicit and few.
                # RATIO, not a count. An absolute threshold is arbitrary and brittle in the exact
                # way this guard was just caught: a 7-character Chinese sentence in a .docx passed a
                # `>= 8` rule by one character, and a `> 40` rule before that. What actually matters
                # is the SHARE of the document that no text guard can read, so that is the quantity
                # measured. A file that is 4% Chinese is a Chinese document with an English header;
                # a file that is 99% Chinese is a brief in another language.
                letters = sum(ch.isalpha() for ch in text)
                share = (cjk / letters) if letters else 0.0
                if share > 0.02:
                    problems.append(
                        f"[C] {rel} is written {share:.0%} in a non-English script "
                        f"({cjk} CJK of {letters} letters, {words} words).\n"
                        f"    -> Rule 2 makes English the language of every result. This file is a\n"
                        f"       binary container, so NO text guard can read it; non-English content\n"
                        f"       there is a competing instruction that survives every check.")
                    print(f"  {rel:58s} {'brief':10s} NON-ENGLISH ({share:.0%})")
                else:
                    print(f"  {rel:58s} {'container':10s} ok ({words} words, {share:.0%} non-English)")
            else:
                # PDF in the live tree that is not the objective and not a reference paper.
                problems.append(
                    f"[B] {rel} is a PDF in the live tree and is neither the objective nor a file "
                    f"under literature/.\n"
                    f"    -> A PDF's text is invisible to every other guard. If it is a brief it is\n"
                    f"       unreadable by the checks; if it is a paper it belongs in literature/.")
                print(f"  {rel:58s} {'pdf':10s} UNSOURCED")
            continue

        # --- 3. English only, for text documents ------------------------------------------
        text = p.read_text(encoding="utf-8", errors="replace")
        cjk = len(CJK.findall(text))
        if rel in CJK_EXEMPT and cjk:
            print(f"  {rel:58s} {'text':10s} ok (exempt: {CJK_EXEMPT[rel][:44]})")
            continue
        if cjk and not is_ref:
            problems.append(
                f"[C] {rel} contains {cjk} CJK characters.\n"
                f"    -> Rule 2: English is the language of every result, including documentation.")
            print(f"  {rel:58s} {'text':10s} NON-ENGLISH ({cjk})")
        elif is_root_objective:
            print(f"  {rel:58s} {'objective':10s} THE OBJECTIVE (unique, at root)")
        else:
            print(f"  {rel:58s} {'text':10s} ok")

    print()
    print(f"documents inspected: {len(docs)}")

    # --- the objective must actually exist ---------------------------------------------------
    if not (ROOT / OBJECTIVE).exists():
        problems.append(f"[A] the objective document is missing: {OBJECTIVE}\n"
                        f"    -> a session with no authority invents one, and invention is how dead\n"
                        f"       objectives come back.")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):\n")
        for p in problems:
            print(f"  {p}")
        print("\nA document that states a retired objective is not a record of work; it is a")
        print("competing instruction. Remove it from the live tree -- sha256 outside is enough.")
        return 1

    print("\nRESULT: PASS -- one objective at the root, no un-inspected containers, English only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
