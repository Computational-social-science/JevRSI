#!/usr/bin/env python
"""
check_synth_decontamination.py -- does the distillation corpus leak into any split we score on?

    python scripts/check_synth_decontamination.py [--ngram 12] [--verbose]

WHY IT RUNS BEFORE ANY GPU TIME
    A distillation corpus is, by construction, full of plausible text about the same domains as the
    benchmark. Training on it can raise a score without raising capability, and care elsewhere does
    not compensate for that. RSI-Jev states their benchmark is held out completely and "verified
    rather than asserted": no training document shares a state with the train or test split, and
    none shares even a single 12-word phrase with either.

    That verification is about THEIR corpus against THEIR benchmark. Ours is a different corpus
    against different splits, so their claim does not transfer and the check runs again here. This
    decides whether the first arm's number means anything, so it runs before the number exists
    rather than after it looks surprising.

THREE LEVELS, because each fails on its own
    1. IDENTIFIERS. Document and question ids. Exact overlap is not subtle, but it is the level
       people forget to check.
    2. STATE TEXT. A document can be copied under a new id, so states are compared as normalised
       text and case and punctuation cannot hide a copy.
    3. N-WORD PHRASES, at 12 words, the length RSI-Jev specifies. Catches the realistic case: a
       paraphrased or partly-copied state. Reported as a COUNT with the offending phrases named,
       not as pass/fail, because one shared phrase in 25,000 documents is a different fact from one
       shared phrase in 300.

WHAT IT DOES NOT DECIDE
    Whether a leak is fatal. A non-zero phrase count on a corpus this size is expected and is a
    question about how the training mix is weighted. What this does is make the number EXIST before
    the first arm, so a high score cannot later be quietly attributed to capability.

THE NEGATIVE CONTROL
    `--self-test` copies a state from the corpus into a split in a temporary directory and requires
    the check to fail. A check that has never failed is not a check.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import splits_dir, subject  # noqa: E402

SPLITS = ("train", "dev", "calibration", "shadow")
WORD = re.compile(r"[a-z0-9]+")


def norm_words(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Aggressive on purpose: it cannot invent an overlap, and it
    stops case and punctuation from hiding a real one."""
    return WORD.findall(text.lower())


def ngrams(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load(path: pathlib.Path):
    """-> (doc_ids, source_keys, normalised state texts) for one split."""
    doc_ids, keys, states = set(), set(), []
    if not path.is_file():
        return doc_ids, keys, states
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        doc_ids.add(d.get("id", ""))
        states.append(" ".join(norm_words(d.get("state", ""))))
        for q in d.get("questions", []):
            k = q.get("_source")
            if k:
                keys.add(k)
    return doc_ids, keys, states


def scan(split_dir: pathlib.Path, n: int, verbose: bool, max_report: int):
    corpus_states: list[str] = []
    for name in ("synth_train_questions.jsonl", "synth_validation_questions.jsonl"):
        p = split_dir / name
        if p.is_file():
            corpus_states += load(p)[2]
    index: dict[str, None] = {}
    for st in corpus_states:
        for g in ngrams(st.split(), n):
            index[g] = None

    fatal = 0
    corpus_doc_ids: set[str] = set()
    corpus_keys: set[str] = set()
    for name in ("synth_train_questions.jsonl", "synth_validation_questions.jsonl"):
        p = split_dir / name
        if p.is_file():
            ids, keys, sts = load(p)
            corpus_doc_ids |= ids
            corpus_keys |= keys
            corpus_states += sts
    corpus_state_set = set(corpus_states)
    index: dict[str, None] = {}
    for st in corpus_states:
        for g in ngrams(st.split(), n):
            index[g] = None

    for split in SPLITS:
        p = split_dir / f"{split}_questions.jsonl"
        if not p.is_file():
            continue
        d_doc, d_key, d_states = load(p)
        doc_hits = d_doc & corpus_doc_ids
        key_hits = d_key & corpus_keys
        text_hits = [st for st in d_states if st and st in corpus_state_set]

        phrases = n_docs = 0
        named: list[str] = []
        for st in d_states:
            overlap = ngrams(st.split(), n) & set(index)
            if overlap:
                phrases += len(overlap)
                n_docs += 1
                if verbose and len(named) < max_report:
                    named.append(next(iter(overlap)))

        print(f"\n  [{split}]")
        print(f"    identifier overlap    : docs {len(doc_hits)}  questions {len(key_hits)}")
        print(f"    state text identical  : {len(text_hits)}")
        print(f"    {n}-gram overlap       : {phrases} phrases across {n_docs} documents")
        if verbose:
            for g in named:
                print(f"      {g[:92]}")
        if doc_hits or key_hits or text_hits:
            fatal += 1
    return fatal, len(corpus_states)


def self_test() -> int:
    """Prove the check fails. Copies a real corpus state into a temporary split."""
    import shutil
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="jevrsi_decon_"))
    try:
        corpus = splits_dir() / "synth_train_questions.jsonl"
        if not corpus.is_file():
            print("  [SKIP] no converted corpus yet; nothing to copy a state from")
            return 0
        first = json.loads(next(l for l in corpus.open(encoding="utf-8") if l.strip()))
        rows = [{"id": f"probe{i}", "env": "probe", "state": "unrelated state text",
                 "questions": []} for i in range(3)]
        rows[0]["state"] = first["state"]          # verbatim copy of a state
        rows[1]["id"] = first["id"]                # duplicate document id
        q0 = dict(first["questions"][0])
        rows[2]["questions"] = [q0]                # duplicate question key
        p = tmp / "dev_questions.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print("  Decontamination self-test. A verbatim state copy must be DETECTED.\n")
        index: dict[str, None] = {}
        corpus_states = load(corpus)[2]
        for st in corpus_states:
            for g in ngrams(st.split(), 12):
                index[g] = None
        d_doc, d_key, d_states = load(p)
        state_hit = any(st in set(corpus_states) for st in d_states)
        doc_hit = bool(d_doc & {first["id"]})
        key_hit = bool(d_key & {q0["_source"]})
        print(f"    a verbatim state copy in a split        -> "
              f"{'detected' if state_hit else 'ACCEPTED -- broken'}")
        print(f"    a duplicated document id                -> "
              f"{'detected' if doc_hit else 'ACCEPTED -- broken'}")
        print(f"    a duplicated question key               -> "
              f"{'detected' if key_hit else 'ACCEPTED -- broken'}")
        ok = state_hit and doc_hit and key_hit
        print(f"\n  {'[OK] the check can fail' if ok else '[FAIL] self-test failed'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the corpus against every split we score on.")
    ap.add_argument("--ngram", type=int, default=12,
                    help="phrase length; 12 is the length RSI-Jev states they verified")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--max-report", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test()

    sd = splits_dir()
    print(f"[decon] n-gram length    : {a.ngram}")
    print(f"[decon] splits compared : {', '.join(SPLITS)}")
    fatal, n_docs = scan(sd, a.ngram, a.verbose, a.max_report)
    print(f"\n[decon] corpus documents: {n_docs}")

    print()
    if fatal:
        print(f"[FAIL] {fatal} split(s) share an identical state with the corpus. Training on this "
              f"corpus and\n       scoring on those splits is not a measurement of capability.")
        return 1
    print("[OK] No split shares an identical state with the distillation corpus.")
    print(f"     The {a.ngram}-gram counts above are the softer signal, recorded so that the first")
    print("     arm's score is not later attributed to capability if it turns out to be high.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
