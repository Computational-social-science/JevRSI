#!/usr/bin/env python
"""
build_synth_split.py -- convert the RSI-Jev distillation corpus into this project's contract.

    python build_synth_split.py --out <dest.jsonl>
    python build_synth_split.py --self-test

WHY THIS EXISTS
    RSI-Jev's v1.0 is a distillation: every question carries a teacher's FULL probability
    distribution rather than a label, and fitting the distribution rather than the label is the
    substance of that release. The corpus is `n4ze3m/typed-decisions-synth` and it is the asset the
    reproduction needs that nothing else here provides.

THE TWO SCHEMAS
    theirs  questions: {qkey: {type, instructions, criteria}}    teacher: {qkey: {probabilities}}
    ours    questions: [{family, qtype, text, candidates, gold: {distribution}, supervision, ...}]

    Their `noul` is our `boolean`. A distribution is POSITIONAL, so candidate order is part of the
    contract and is fixed here rather than inherited.

FOUR SHAPES IN THE REAL CORPUS, each of which cost a bug to find
    1. `noul` questions give a SINGLE SCALAR for the true branch -- {"noul": 0.9333} -- with no false
       entry. The false branch is 1 - p. Not an interpretation imposed here; it is the only reading
       consistent with the key the teacher itself uses, and --self-test asserts every scalar is in
       [0, 1] rather than assuming it.
    2. `choice` questions have a MAP for criteria and probabilities keyed by OPTION NAME. Handling
       only positional indices drops all 9,171 of them -- 65% of the corpus -- while exiting 0 and
       printing a plausible type count. That is why the self-test pins this case by name.
    3. `score` questions have a LIST for criteria and probabilities keyed by INDEX.
    4. 1,278 of the first 3,002 states are PROSE, not JSON ("Transcript: Ivan: ..."). The corpus
       flags this with `state_is_json`. Parsing states unconditionally raises on the first real
       document, so states are passed through as text and never parsed.

    Ordering: the two boolean literals are fixed at [true, false] because they have an intrinsic
    order. Any other map is sorted -- deterministic, because an unstable order makes a positional
    distribution irreproducible.

    Rounding: teacher values are 4 decimals, so a sum can land at 0.9999. A deficit within 0.02 is
    renormalised; a larger one is refused rather than absorbed. The converted vector is NOT rounded
    again, because a target that does not sum to one puts a constant offset into every gradient.

WHAT IS NOT DONE HERE
    No decontamination. That is scripts/check_synth_decontamination.py, and doing it in two places
    is how a leak ends up counted as checked in one and not the other.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import require, synth_corpus  # noqa: E402

TYPE_MAP = {"noul": "boolean", "choice": "choice", "score": "score"}
NUL_CANDIDATES = ["true", "false"]
# Beyond this a deficit is a format change, not rounding. 0.02 is far above the 1e-4 a 4-decimal
# rounding can produce and far below any real corruption.
SUM_TOLERANCE = 0.02


def maybe_json(x):
    """For fields that really are JSON documents: questions, gold, teacher. NOT for state."""
    return json.loads(x) if isinstance(x, str) else x


def state_to_text(state) -> str:
    """The state is text either way. It is not parsed -- see the module docstring, point 4."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def convert_question(qkey: str, spec: dict, teacher_entry: dict, state_id: str):
    """-> (question, None) or (None, reason). Refuses rather than guesses."""
    qtype = TYPE_MAP.get(spec.get("type"))
    if qtype is None:
        return None, f"unknown type {spec.get('type')!r}"

    crit = spec.get("criteria")
    if isinstance(crit, dict):
        if len(crit) == 2 and all(k in crit for k in ("true", "false")):
            candidates = list(NUL_CANDIDATES)
        else:
            candidates = sorted(crit)
    elif isinstance(crit, list):
        candidates = [str(c) for c in crit]
    else:
        return None, "criteria is neither a list nor a map"

    if qtype == "boolean":
        raw = teacher_entry.get("noul")
        if raw is None:
            return None, "boolean question has no teacher noul scalar"
        p = float(raw)
        dist = [p, 1.0 - p]
        candidates = list(NUL_CANDIDATES)
    else:
        probs = teacher_entry.get("probabilities")
        if not isinstance(probs, dict):
            return None, "non-boolean question has no probabilities map"
        if all(str(c) in probs for c in candidates):
            dist = [float(probs[str(c)]) for c in candidates]
        else:
            try:
                dist = [float(probs[str(i)]) for i in range(len(candidates))]
            except (KeyError, TypeError, ValueError) as e:
                return None, (f"probabilities match neither the option names {candidates} "
                              f"nor the indices 0..{len(candidates) - 1}: {e}")

    total = sum(dist)
    if total <= 0:
        return None, "distribution sums to zero"
    if abs(total - 1.0) > 1e-9:
        if abs(total - 1.0) > SUM_TOLERANCE:
            return None, f"distribution sums to {total:.4f}, too far from 1 to renormalise"
        dist = [d / total for d in dist]

    return {
        "family": spec.get("family", qkey),
        "qtype": qtype,
        "text": spec.get("instructions", ""),
        "candidates": candidates,
        "gold": {"distribution": dist},
        "supervision": "known_distribution",
        "weight": 1.0,
        "ordinal": qtype == "score",
        # Composite, because qkey alone is not unique: the corpus reuses "q1"/"tone"/"next_action"
        # across thousands of states, and an audit keyed on qkey collides on 2,294 entries.
        "_source": f"{state_id}::{qkey}",
    }, None


def convert_file(src: pathlib.Path, out: pathlib.Path | None = None):
    stats: Counter = Counter()
    reasons: Counter = Counter()
    types: Counter = Counter()
    nouls: list[float] = []
    rows = []
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        qs = maybe_json(d["questions"])
        gold = maybe_json(d.get("gold", {})) or {}
        teacher = maybe_json(d.get("teacher", {})) or {}
        stats["questions_str_encoded" if isinstance(d.get("questions"), str) else "questions_decoded"] += 1
        stats["state_prose" if not d.get("state_is_json", True) else "state_json"] += 1

        questions = []
        for qkey, spec in qs.items():
            if not isinstance(spec, dict):
                reasons["spec is not an object"] += 1
                continue
            conv, why = convert_question(qkey, spec, teacher.get(qkey) or {}, d["state_id"])
            if conv is None:
                reasons[why] += 1
                continue
            questions.append(conv)
            types[conv["qtype"]] += 1
            if conv["qtype"] == "boolean":
                nouls.append(conv["gold"]["distribution"][0])
        if not questions:
            stats["documents_dropped"] += 1
            continue
        rows.append({
            "id": f"synth_{d['state_id']}",
            "env": d.get("domain", "synth"),
            "state": state_to_text(d["state"]),
            "questions": questions,
            "_provenance": {"corpus": "n4ze3m/typed-decisions-synth",
                            "state_id": d["state_id"],
                            "teacher_model": d.get("teacher_model"),
                            "generator": d.get("generator")},
        })
        stats["documents_kept"] += 1
        stats["questions_kept"] += len(questions)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stats, reasons, types, nouls, rows


def self_test() -> int:
    print("  Conversion self-test. A converter that has never refused anything is not a converter.\n")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"    {name:52} -> {'ok' if cond else 'FAIL'}")
        if not cond:
            ok = False
            if detail:
                print(f"      {detail}")

    # 1. noul: single scalar -> two branches
    q, why = convert_question("kept", {"type": "noul", "instructions": "x",
                                       "criteria": {"true": "a", "false": "b"}},
                              {"noul": 0.9333}, "s1")
    check("noul yields a two-branch distribution", q is not None, why or "")
    if q:
        d = q["gold"]["distribution"]
        check("noul keeps the scalar on the true branch and complements the other",
              abs(d[0] - 0.9333) < 1e-12 and abs(d[1] - (1 - 0.9333)) < 1e-12, str(d))
        check("noul order is [true, false]", q["candidates"] == ["true", "false"], str(q["candidates"]))

    # 2. score: list criteria, index-keyed probabilities
    q, _ = convert_question("spec", {"type": "score", "instructions": "y",
                                     "criteria": ["c0", "c1", "c2", "c3"]},
                            {"probabilities": {"0": 0.0, "1": 0.0333, "2": 0.25, "3": 0.7167}}, "s1")
    check("score keeps the teacher's positional order",
          q is not None and [round(x, 4) for x in q["gold"]["distribution"]]
          == [0.0, 0.0333, 0.25, 0.7167])
    check("score is flagged ordinal", q is not None and q["ordinal"] is True)

    # 3. REGRESSION: choice is a MAP with NAME-keyed probabilities. Handling only indices dropped
    #    all 9,171 choice questions while exiting 0.
    spec = {"type": "choice", "instructions": "tone",
            "criteria": {"dry_sarcastic": "i", "earnest_positive": "s",
                         "hostile_rant": "a", "neutral_report": "f"}}
    entry = {"probabilities": {"earnest_positive": 0.0033, "dry_sarcastic": 0.97,
                               "hostile_rant": 0.02, "neutral_report": 0.0067}}
    q, why = convert_question("tone", spec, entry, "s1")
    check("a keyed choice question converts", q is not None, why or "")
    if q:
        check("choice candidates are sorted deterministically",
              q["candidates"] == sorted(spec["criteria"]), str(q["candidates"]))
        d = [round(x, 4) for x in q["gold"]["distribution"]]
        # sorted() puts dry_sarcastic first and it carries 0.97, so the distribution must follow the
        # CANDIDATE order. Reading the teacher's dict in its own order gives the same multiset
        # attached to the wrong options.
        check("the distribution is realigned to the candidate order", d == [0.97, 0.0033, 0.02, 0.0067],
              str(d))
        check("choice is not flagged ordinal", q["ordinal"] is False)
        check("a choice keyed by the boolean literals still gets [true, false]",
              convert_question("b", {"type": "choice", "criteria": {"false": "", "true": ""}},
                               {"probabilities": {"false": 0.4, "true": 0.6}}, "s1")[0]
              ["candidates"] == ["true", "false"])
        # REGRESSION, found by diffing this rebuild against the output of the previous converter
        # version: a `choice` whose criteria happen to CONTAIN "true" and "false" alongside other
        # options is not a boolean. The earlier test was "are true and false among the keys", which
        # is true for a 4-way choice like {"true","false","misleading","unverifiable"}; that
        # question was then given 2 candidates and REFUSED for a distribution that covered neither
        # names nor indices, so it vanished from the corpus without a word. Two such questions exist
        # in train.jsonl and both were silently lost. The test is now "exactly two options, and they
        # are the boolean literals" -- the size is what makes it a boolean.
        four = convert_question("cv", {"type": "choice", "criteria": {
            "true": "a", "false": "b", "misleading": "c", "unverifiable": "d"}},
            {"probabilities": {"true": 0.01, "false": 0.9333,
                               "misleading": 0.0467, "unverifiable": 0.01}}, "s1")
        check("a 4-way choice containing true/false stays a choice", four[0] is not None, four[1] or "")
        if four[0]:
            check("its four options are all kept",
                  len(four[0]["candidates"]) == 4, str(four[0]["candidates"]))
            check("its qtype is choice, not boolean", four[0]["qtype"] == "choice")

    # 4. refusals
    check("refuses a boolean with no teacher scalar",
          convert_question("x", {"type": "noul", "criteria": {"true": "", "false": ""}}, {}, "s")[0] is None)
    check("refuses probabilities that cover neither names nor indices",
          convert_question("x", {"type": "score", "criteria": ["a", "b"]},
                           {"probabilities": {"0": 1.0}}, "s")[0] is None)
    check("refuses an unknown type",
          convert_question("x", {"type": "mystery", "criteria": ["a"]},
                           {"probabilities": {"0": 1.0}}, "s")[0] is None)
    # A distribution that is merely non-uniform is still a distribution: {"a":0.5,"b":0.5} sums to 1
    # and must convert. What must be refused is a sum far from 1, which is a format change rather
    # than a rounding artefact. An earlier version of this test used the 0.5/0.5 pair and wrongly
    # asserted a refusal -- the test was wrong, not the converter.
    q, _ = convert_question("x", {"type": "choice", "criteria": ["a", "b"]},
                           {"probabilities": {"a": 0.5, "b": 0.5}}, "s")
    check("a uniform distribution is valid, not a defect", q is not None)
    check("refuses a distribution far from 1",
          convert_question("x", {"type": "choice", "criteria": ["a", "b", "c"]},
                           {"probabilities": {"a": 0.5, "b": 0.5, "c": 0.5}}, "s")[0] is None)

    # 5. rounding: a 4-decimal deficit is renormalised, a real one is refused
    q, _ = convert_question("x", {"type": "choice", "criteria": ["a", "b", "c"]},
                           {"probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333}}, "s")
    check("renormalises a 4-decimal rounding deficit",
          q is not None and abs(sum(q["gold"]["distribution"]) - 1.0) < 1e-12)
    check("refuses a deficit beyond rounding",
          convert_question("x", {"type": "choice", "criteria": ["a", "b"]},
                           {"probabilities": {"a": 0.3333, "b": 0.3333}}, "s")[0] is None)

    # 6. states: prose must survive
    prose = "Transcript: Ivan: Let's finalize the design. Judy: I prefer version A."
    check("a prose state passes through unchanged", state_to_text(prose) == prose)
    check("a dict state is serialised", json.loads(state_to_text({"a": 1})) == {"a": 1})
    check("a JSON-string state is not re-serialised", state_to_text('{"id": "R1"}') == '{"id": "R1"}')

    # 7. determinism and the composite source key
    a = convert_question("q1", {"type": "noul", "criteria": {"true": "", "false": ""}},
                         {"noul": 0.5}, "same")[0]
    b = convert_question("q1", {"type": "noul", "criteria": {"true": "", "false": ""}},
                         {"noul": 0.5}, "same")[0]
    check("conversion is deterministic", a == b)
    check("the source key is composite", a["_source"] == "same::q1", a["_source"])

    print(f"\n  {'[OK] the converter is faithful' if ok else '[FAIL] self-test failed'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert the synth corpus to this project's contract.")
    ap.add_argument("--src", default="", help="defaults to <synth_corpus>/data/train.jsonl")
    ap.add_argument("--out", required=False, help="destination .jsonl")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.out:
        raise SystemExit("[fatal] --out is required outside --self-test")

    root = require(synth_corpus(), "the synth corpus (JEVRSI_SYNTH_CORPUS)")
    src = pathlib.Path(a.src) if a.src else root / "data" / "train.jsonl"
    if not src.is_file():
        raise SystemExit(f"[fatal] {src} not found.\n  Download with:\n"
                         f"  HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub "
                         f"import hf_hub_download; hf_hub_download('n4ze3m/typed-decisions-synth', "
                         f"'data/{src.name}', repo_type='dataset', local_dir='<dest>')\"")

    stats, reasons, types, nouls, rows = convert_file(src, pathlib.Path(a.out))
    print(f"[convert] {src.name} -> {a.out}")
    print(f"  documents kept / dropped : {stats['documents_kept']} / {stats['documents_dropped']}")
    print(f"  questions kept           : {stats['questions_kept']}")
    print(f"  by type                  : {dict(sorted(types.items()))}")
    print(f"  state form               : {{'state_json': {stats['state_json']}, "
          f"'state_prose': {stats['state_prose']}}}")
    if nouls:
        inside = all(-1e-9 <= p <= 1 + 1e-9 for p in nouls)
        print(f"  noul scalars in [0,1]    : {inside} (min {min(nouls):.4f}, max {max(nouls):.4f}, "
              f"n={len(nouls)})")
        if not inside:
            raise SystemExit("[fatal] a noul teacher scalar is outside [0, 1]")
    if reasons:
        print(f"  refused                  : {dict(reasons)}")
    print(f"\n[OK] wrote {len(rows)} documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
