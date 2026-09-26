"""Mechanical-scorer regression tests.

Run directly: ``python tests/test_evaluator.py`` (no pytest required).

These exist because a scoring bug is invisible in aggregate results: a false
extraction inflates one variant's accuracy and shows up as a *finding* about
prompt sensitivity rather than as a bug. Both directions are pinned — values that
MUST parse and prose that MUST NOT.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasp.benchmark import Item
from oasp.evaluator import chance_level, extract_choice, score

ABC = Item("t1", "q", ("A", "B", "C", "D"), ("w", "x", "y", "z"), "C")
NUMERIC = Item("t2", "q", ("1", "2", "3", "4"), ("w", "x", "y", "z"), "3")

MUST_PARSE = [
    ("Answer: C", "C"),
    ("Answer: (C)", "C"),
    ("answer is c", "C"),
    ("Answer = C", "C"),
    ("C", "C"),
    ("C.", "C"),
    ("C)", "C"),
    ("(C)", "C"),
    ("(C) \n\nHere's why:\n\n*   **Robots", "C"),
    ("C\n", "C"),
    ("  c  ", "C"),
    ("The correct option is C.", "C"),
    ("Answer: C\n", "C"),
]

MUST_NOT_PARSE = [
    "",
    "   ",
    "I cannot answer this question.",
    "The answer is difficult to determine.",
    "I don't know.",
    "None of the above apply here.",
    # The English article must never be read as option A.
    "This is a good answer because it fits the data.",
    "Sunlight is a source of energy for most ecosystems.",
]

# Reasoning-model shapes: the *final* answer must win over earlier deliberation.
MUST_PARSE_LAST = [
    ("Let me weigh these. Option A is tempting. But the answer is C.", "C"),
    ("A seems plausible. B also. The correct option is C.", "C"),
    ("I considered A and B, then concluded D. So D.", "D"),
]

# Values that must resolve to a *specific* option rather than to None.
MUST_EXTRACT_VALUE = [
    ("Answer: 3", NUMERIC, "3"),
    ("3", NUMERIC, "3"),
]


def main() -> int:
    failures: list[str] = []

    for text, expected in MUST_PARSE:
        got, _ = extract_choice(text, ABC.labels)
        if got != expected:
            failures.append(f"expected {expected!r} from {text!r}, got {got!r}")

    for text in MUST_NOT_PARSE:
        got, _ = extract_choice(text, ABC.labels)
        if got is not None:
            failures.append(f"expected None from {text!r}, got {got!r} (false extraction)")

    for text, expected in MUST_PARSE_LAST:
        got, _ = extract_choice(text, ABC.labels)
        if got != expected:
            failures.append(f"last-match: expected {expected!r} from {text!r}, got {got!r}")

    for text, item, expected in MUST_EXTRACT_VALUE:
        got, _ = extract_choice(text, item.labels)
        if got != expected:
            failures.append(f"expected {expected!r} from {text!r}, got {got!r}")

    # "difficult" must not be read as option D — the guard that motivated this file.
    got, _ = extract_choice("I think the answer is difficult", ABC.labels)
    if got is not None:
        failures.append(f"'difficult' false-extracted as {got!r}")

    # Scoring direction.
    if not score("Answer: C", ABC).correct:
        failures.append("gold match failed for 'Answer: C' vs gold C")
    if score("Answer: A", ABC).correct:
        failures.append("wrong answer scored correct")
    if score("garbage", ABC).parse_ok:
        failures.append("unparseable text marked parse_ok")

    # Chance level sanity.
    cl = chance_level([ABC, ABC, NUMERIC])
    if abs(cl - (0.25 + 0.25 + 0.25) / 3) > 1e-9:
        failures.append(f"chance_level wrong: {cl}")

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    n = len(MUST_PARSE) + len(MUST_NOT_PARSE) + len(MUST_EXTRACT_VALUE) + 4
    print(f"[OK] evaluator: {n} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
