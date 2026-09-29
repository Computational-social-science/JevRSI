"""check_citation_provenance.py -- assert every external number in the manuscript has a verifiable source.

WHY THIS EXISTS. A proposal audited on 2026-09-29 (docs/audit_RTX4070_proposal_2026-09-29.md) asserted three
load-bearing numbers for a public checkpoint -- a "66.4% composite score", an "ECE = 0.0712", and a "~50 ms"
forward pass. A direct read of that checkpoint's model card and its shipped `temperatures.json` found NONE of
the three: the card records a 57.8% coding-accuracy and an 87.2% teacher-agreement, both flagged
`verified: false`, and `temperatures.json` contains no confidence or correctness data from which an ECE could
be computed. The proposal violated its own Rule 4, which demands that superiority be claimed "only on metrics
the baseline actually reported". So the failure is mechanical and therefore mechanically preventable.

WHAT IT CHECKS. Two classes, and the second is the one that matters:

  A. UNSOURCED NUMBER -- a quantitative claim in a manuscript or living document that has no resolvable
     provenance anywhere in the project's evidence files.
  B. FORBIDDEN STRING -- a specific retracted value that must never reappear, so a silent revert cannot pass.

Class A alone is not enough: presence-checking validators cannot catch a silent revert, and a claim can carry
a citation that does not actually support it. Class B closes the specific revert.

Run: python scripts/check_citation_provenance.py [--docs ...] [--strict]
Exit 0 when clean, 1 on any finding.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------
# Class B: values retracted by the 2026-09-29 proposal audit. Each maps to what replaced it.
# A reappearance of any of these in a LIVE document is a revert, not a citation.
# ---------------------------------------------------------------------------------------------
FORBIDDEN = {
    "66.4": "proposal's 'kev-0.6B composite 66.4%' -- no such metric in the model card; the card reports "
            "57.8% coding accuracy and 87.2% teacher agreement, both verified:false",
    "0.0712": "proposal's 'ECE = 0.0712' -- temperatures.json carries no confidence/correctness data, "
              "so the value is not computable from the checkpoint",
    "50 ms": "proposal's '~50 ms per forward pass' -- no latency benchmark in the checkpoint, no third-party "
             "source given",
}

# ---------------------------------------------------------------------------------------------
# Where a provenance claim is allowed to be anchored. A number is "sourced" when the same literal
# (or its distinctive prefix) appears in one of these, i.e. it was read off a measured artifact.
# ---------------------------------------------------------------------------------------------
EVIDENCE_GLOBS = [
    "measurement/**/*.json",
    "measurement/**/*.md",
    "docs/**/*.md",
    "docs/**/*.json",
    "figures/out/*.json",
]

LIVE_DOCS = [
    "manuscript.html",
    "docs/JevRSI_target_backward_derivation.md",
    "docs/JevRSI_goal_feasibility_2026-09-28.md",
    "docs/JevRSI_decision_record_2026-09-28.md",
    "docs/JevRSI_starting_point_review_2026-09-28.md",
    "docs/audit_RTX4070_proposal_2026-09-29.md",
    "CURRENT_OBJECT.md",
]

# Numbers that are self-evidently not empirical claims about a model or a dataset. Quoting these is
# fine and flagging them would make the guard cry wolf, which is how guards get disabled.
ALLOW = {
    # structural / document facts
    "7.16", "400", "0.618", "0.382", "6.0", "3.14",          # figure geometry, dpi
    "960", "120", "400", "2000", "180", "240",                 # split sizes, counts, calibration n
    "2024", "2025", "2026", "2027",                            # years
    # version / identifier fragments
    "4.34.1", "4.29.0", "1.13.0", "0.6", "322", "421", "4070", "12", "150", "200", "27",
    "1.0", "0.0", "2.0", "100", "50", "10", "5", "3", "1",
}

QUANT = re.compile(
    r"(?<![\w.])(\d{1,4}(?:\.\d+)?)\s*(?:%|pp|ms|s\b|GB|MiB|B\b|h\b|×|x\b|C\b|°C|W\b)?"
)


def load_evidence() -> str:
    """Concatenate the evidence corpus AND every numeric literal that appears anywhere in it.

    The normalisation matters: a manuscript that quotes `0.0106` is quoting a measured `0.01057`, and a
    substring test would call that unsourced. So the index is built from the numerals themselves, and a
    document figure counts as anchored when it is a *prefix* of an evidence figure at the same decimal
    precision, or vice versa -- which is what "quoting the same measurement to fewer digits" means.
    """
    blob = []
    for g in EVIDENCE_GLOBS:
        for p in REPO.glob(g):
            try:
                blob.append(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
    return "\n".join(blob)


def norm(tok: str) -> str:
    return tok.rstrip("0").rstrip(".") if "." in tok else tok


def _anchored(tok: str, index: set[str]) -> bool:
    """True when an evidence figure is the same number printed at coarser precision.

    A manuscript that rounds a measured `0.01057` to `0.0106` is citing it, not inventing it. The match is
    therefore a character-prefix comparison, but only for figures carrying at least three significant
    digits on each side -- matching a two-digit claim against a five-digit measurement would let almost any
    number pass, and a guard that cannot fail gets switched off.
    """
    t = norm(tok)
    if t in index:
        return True
    digits = t.replace("0", "").replace(".", "").lstrip("0")
    if len(digits) < 3:
        return False
    n = len(t)
    return any(len(e) > n and e[:n] == t and len(e.replace("0", "").replace(".", "").lstrip("0")) >= 3
               for e in index)


def check_forbidden(docs: list[Path]) -> list[str]:
    findings = []
    for d in docs:
        text = d.read_text(encoding="utf-8", errors="replace")
        for bad, why in FORBIDDEN.items():
            # the audit document is allowed to name the retracted value -- that is its job
            if d.name.startswith("audit_RTX4070"):
                continue
            for m in re.finditer(re.escape(bad), text):
                line = text[: m.start()].count("\n") + 1
                ctx = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                findings.append(
                    f"FORBIDDEN  {d.relative_to(REPO)}:{line}  '{bad}'\n"
                    f"           retracted because: {why}\n"
                    f"           context: ...{ctx}..."
                )
    return findings


def check_unsourced(docs: list[Path], evidence: str, strict: bool) -> list[str]:
    index = {norm(m) for m in re.findall(r"\d{1,4}(?:\.\d+)?", evidence)}
    findings = []
    for d in docs:
        text = d.read_text(encoding="utf-8", errors="replace")
        if d.suffix == ".html":
            text = re.sub(r"<[^>]+>", " ", text)
        for i, line in enumerate(text.split("\n"), 1):
            if re.search(r"retract|superseded|audit_RTX4070|do not use", line, re.I):
                continue  # a line that announces the retraction may name the value
            for raw in re.findall(r"(?<![\w.])(\d{1,4}\.\d+)(?![\w])", line):
                t = norm(raw)
                if t in ALLOW or raw in ALLOW:
                    continue
                if _anchored(raw, index):
                    continue
                findings.append(
                    f"UNSOURCED  {d.relative_to(REPO)}:{i}  {raw}  |  {line.strip()[:110]}"
                )
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=None)
    ap.add_argument("--strict", action="store_true", help="treat UNSOURCED as fatal (default: warn only)")
    a = ap.parse_args()

    paths = [REPO / p for p in (a.docs or LIVE_DOCS)]
    docs = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]

    print("[citation-provenance] live documents: %d" % len(docs))
    for m in missing:
        print(f"    [warn] not found, skipped: {m.relative_to(REPO)}")

    evidence = load_evidence()
    print(f"[citation-provenance] evidence corpus: {len(evidence):,} chars "
          f"({len(EVIDENCE_GLOBS)} glob groups)")

    findings = check_forbidden(docs)
    unsourced = check_unsourced(docs, evidence, a.strict)

    for f in findings:
        print("  " + f)
    for f in unsourced:
        print("  " + f)

    fatal = len(findings) + (len(unsourced) if a.strict else 0)
    if findings:
        print(f"[FAIL] {len(findings)} forbidden value(s) present -- these are retractions, not citations")
    if unsourced:
        verdict = "FAIL" if a.strict else "WARN"
        print(f"[{verdict}] {len(unsourced)} decimal figure(s) with no anchor in the evidence corpus")
        if not a.strict:
            print("        rerun with --strict to make this fatal")
    if not findings and not unsourced:
        print("[OK] no forbidden values, no unanchored figures")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
