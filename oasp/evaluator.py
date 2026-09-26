"""Mechanical scoring. No LLM judge anywhere in this path.

The brief forbids subjective scoring (机械化的数值评测, 不允许主观评分). This module
is therefore the *only* scorer: a deterministic regex extractor on a constrained
answer token, exact-matched against the frozen gold label.

Consequence, stated up front: a prompt variant that fails to elicit a parseable
answer scores 0. That is intentional — format-following is part of the measured
effect for a "landing form" study, not a hidden covariate. The parse rate is
reported alongside accuracy so a reviewer can see how much of a score drop is
extraction failure rather than wrong answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .benchmark import Item

# Priority-ordered extractors. Each returns the first single-character answer.
#
# Every pattern ends with a `(?![A-Za-z0-9])` guard. Without it, prose such as
# "answer is difficult" false-extracts `D` — which would silently *inflate* a
# variant's accuracy and corrupt M1/M2. The guard makes extraction require a
# genuinely delimited answer token.
_ANSWER_PREFIX = re.compile(
    r"answer\s*(?:is|:|=|would be)?\s*[\(\[]?\s*([A-Za-z0-9])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_BARE_ONLY = re.compile(r"^\s*[\(\[]?\s*([A-Za-z0-9])\s*[\)\]]?\s*[.):,]?\s*$")
_BRACKETED_LEAD = re.compile(r"^\s*[\(\[]\s*([A-Za-z0-9])\s*[\)\]]")
_LEADING = re.compile(r"^\s*([A-Za-z0-9])(?![A-Za-z0-9])\s*[.):,]")
_STANDALONE = re.compile(r"(?:^|[\s\(\[])([A-Za-z])(?=[\)\].,;:]|\s*$)")


@dataclass(frozen=True)
class Score:
    correct: bool
    extracted: str | None
    parse_ok: bool
    matched_choice_text: bool = False


def _norm(s: str) -> str:
    return s.strip()


def extract_choice(text: str, valid_labels: tuple[str, ...]) -> tuple[str | None, bool]:
    """Extract the chosen option label. Returns ``(label_or_None, matched_text)``."""
    if not text:
        return None, False

    valid_upper = {v.strip().upper() for v in valid_labels}
    valid_lower = {v.strip().lower() for v in valid_labels}

    def _accept(ch: str) -> str | None:
        if ch.upper() in valid_upper:
            return ch.upper()
        return None

    t = _norm(text)

    # 1. explicit "Answer: X" / "answer is X" / "Answer = X".
    #    Take the LAST match: a reasoning model that deliberates ("...option A is
    #    tempting, but the answer is C") puts its real answer at the end, and the
    #    first match would score the discarded hypothesis.
    m = None
    for m in _ANSWER_PREFIX.finditer(t):
        pass
    if m:
        got = _accept(m.group(1))
        if got:
            return got, False

    # 2. the whole response is just a letter
    m = _BARE_ONLY.match(t)
    if m:
        got = _accept(m.group(1))
        if got:
            return got, False

    # 3. response opens with a bracketed letter, e.g. "(C)" then prose
    m = _BRACKETED_LEAD.match(t)
    if m:
        got = _accept(m.group(1))
        if got:
            return got, False

    # 4. response opens with the letter, then punctuation
    m = _LEADING.match(t)
    if m:
        got = _accept(m.group(1))
        if got:
            return got, False

    # 5. last standalone letter token — the tail of a thinking trace is where
    #    models state their final choice.
    last = None
    for m in _STANDALONE.finditer(t):
        got = _accept(m.group(1))
        if got:
            last = got
    if last is not None:
        return last, False

    return None, False


def extract_by_choice_text(text: str, item: Item) -> str | None:
    """Last resort: the model echoed a choice's text instead of its label.

    Only fires on an exact, case-insensitive substring match of a full option
    text, so it cannot be triggered by a stray keyword. Returns the option label.
    """
    if not text:
        return None
    low = text.lower()
    hits = [
        (len(txt), lbl)
        for lbl, txt in zip(item.labels, item.texts)
        if len(txt.strip()) >= 8 and txt.strip().lower() in low
    ]
    if not hits:
        return None
    hits.sort(reverse=True)  # longest match wins — avoids nested-option ambiguity
    return str(hits[0][1]).strip().upper()


def score(text: str, item: Item) -> Score:
    """Mechanically score one raw completion against one frozen item."""
    extracted, matched = extract_choice(text, item.labels)
    if extracted is None:
        by_text = extract_by_choice_text(text, item)
        if by_text is not None:
            extracted, matched = by_text, True
    if extracted is None:
        return Score(correct=False, extracted=None, parse_ok=False)

    gold = item.answer_key.strip().upper()
    # ARC ships a small number of items with labels like "1".."4" as well as "A".."D";
    # compare on the label alphabet the item actually uses.
    return Score(correct=extracted == gold, extracted=extracted, parse_ok=True,
                 matched_choice_text=matched)


# ---------------------------------------------------------------------------
# Chance level
# ---------------------------------------------------------------------------


def chance_level(items: list[Item]) -> float:
    """Expected accuracy of uniform random guessing given each item's option count."""
    if not items:
        return 0.0
    return sum(1.0 / max(1, len(it.labels)) for it in items) / len(items)


def summarize_parse_rates(rows) -> dict:
    """Parse-rate summary over a result frame (pandas DataFrame)."""
    if rows is None or len(rows) == 0:
        return {}
    out: dict[str, dict] = {}
    if "model" in rows.columns:
        for model, grp in rows.groupby("model", observed=True):
            out[str(model)] = {
                "n": int(len(grp)),
                "parse_rate": round(float(grp["parse_ok"].mean()), 4),
                "accuracy": round(float(grp["correct"].mean()), 4),
                "accuracy_on_parsed": round(float(grp.loc[grp["parse_ok"], "correct"].mean()), 4)
                if grp["parse_ok"].any()
                else None,
            }
    return out
