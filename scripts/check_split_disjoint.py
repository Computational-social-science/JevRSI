"""
check_split_disjoint.py -- the proposal's partition invariant, as an executable check.

WHAT THIS ENFORCS
    The proposal (section 4.2) requires four DIFFERENT things to exist: the set every stage is FITTED
    on, the set used to FILTER a candidate, the set the KEEP/REVERT decision is made on, and the
    frozen set the claim rests on. "Different" is the whole point, and it is not self-evident from
    the bundles -- it has to be checked against the question ids, on every run, by something that
    cannot be edited to agree with the data.

WHY A CHECK AND NOT A ONCE-OFF NOTE
    Two distinct failures motivated this, and they are different in kind.

    1. The dev split was cut by QUESTION index (`every 4th question to proxy`). The dev split is
       120 cases x 5 questions, so that rule puts four questions of a case in medium and its fifth in
       proxy. A candidate could therefore pass the filter on the very case the decision set scores.

    2. The bundles' row `id` was a positional index, unique only WITHIN a bundle. Every overlap test
       run against it reported total overlap -- 150/150, 450/450 -- which reads as a catastrophic leak
       and is actually an artefact of the identifier. An earlier conclusion of mine ("medium is a
       subset of calibration") came from that test and was wrong. A check whose identifier cannot
       distinguish the two cases it must judge cannot settle either one.

    So the bundles now carry the ORIGINAL question id and the case id, and this reads those. The
    positional `id` is kept for the evaluator's convenience and is explicitly not an identifier.

WHAT COUNTS AS A VIOLATION
    * any two of proxy / medium / calibration sharing a question id
    * proxy and medium sharing a CASE -- the case is the unit of shared context, so a case that
      straddles the filter and the decision boundary is the leak that matters, even when every
      individual question id is distinct
    * a missing qid on any row, which would silently disable the check rather than fail it

FROZEN V
    V is deliberately absent. This check does not look for it and does not require it. A check that
    fails until someone produces a held-out set would pressure the run toward touching V early, which
    is the one thing Rule 1 forbids.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "measurement" / "logit_bundles"
SETS = ("proxy", "medium", "calibration")


def load(name: str) -> dict:
    p = BUNDLES / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"[fatal] {p} not found. Run measurement/build_logit_bundles.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 78)
    print("split check -- the fit set, the filter set and the decision set are three different things")
    print("=" * 78)
    print(f"bundles: {BUNDLES}")

    data = {n: load(n) for n in SETS}
    qids: dict[str, set] = {}
    cases: dict[str, set] = {}
    problems: list[str] = []

    print(f"\n{'set':13s} {'rows':>6s} {'qids':>6s} {'cases':>6s}  types")
    for n, d in data.items():
        rows = d["rows"]
        q = {r.get("qid") for r in rows}
        c = {r.get("case") for r in rows if r.get("case")}
        missing = sum(1 for r in rows if not r.get("qid"))
        qids[n] = {x for x in q if x}
        cases[n] = c
        mix = {}
        for r in rows:
            mix[r["type"]] = mix.get(r["type"], 0) + 1
        print(f"  {n:11s} {len(rows):6d} {len(qids[n]):6d} {len(c):6d}  {mix}")
        if missing:
            problems.append(f"{n}: {missing} row(s) carry no qid, so disjointness is unchecked there")
        if not qids[n]:
            problems.append(f"{n}: no usable qid on any row")

    print("\npairwise question-id overlap (must be 0):")
    for i, a in enumerate(SETS):
        for b in SETS[i + 1:]:
            ov = qids[a] & qids[b]
            ok = not ov
            print(f"  {a:12s} n {b:12s} = {len(ov):5d}   {'[ok]' if ok else '[VIOLATION]'}")
            if not ok:
                problems.append(f"questions shared between {a} and {b}: {len(ov)}")

    # The case-level check applies to the two dev-derived sets. Calibration is a different official
    # split and its rows carry no case id, so asserting a case overlap with it would be meaningless
    # rather than strict.
    if cases["proxy"] and cases["medium"]:
        ov = cases["proxy"] & cases["medium"]
        ok = not ov
        print(f"\ncase overlap, filter vs decision (must be 0):")
        print(f"  proxy cases={len(cases['proxy'])}  medium cases={len(cases['medium'])}  "
              f"union={len(cases['proxy'] | cases['medium'])}")
        print(f"  shared cases = {len(ov):5d}   {'[ok]' if ok else '[VIOLATION]'}")
        if not ok:
            problems.append(f"cases shared between proxy and medium: {len(ov)}")

    print("\nheld-out V:")
    print("  not present, and not required by this check. Requiring it would create pressure to")
    print("  produce and read V during setup, which is exactly what Rule 1 forbids.")

    if problems:
        print(f"\nVIOLATIONS ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        print("\nA keep/revert decision made on a set that overlaps the fit set or the filter set is")
        print("not evidence. Re-cut the split rather than re-running on this one.")
        return 1

    print("\nRESULT: PASS -- fit, filter and decision sets are disjoint at both question and case level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
