#!/usr/bin/env python
"""Recompute the release-comparable metrics from a run's `items.jsonl`.

WHY THIS EXISTS
    Their release publishes its verification in `verify.json` as **pooled top-1**:

        "typed_decisions/canonical": {"pooled_top1": 0.6525, "n": 2000, "agreement_with_record": 1.0}
        "typed_decisions/reversed":  {"pooled_top1": 0.6525, "n": 2000, "agreement_with_record": 1.0}
        "mmlu_pro_1k/canonical":     {"pooled_top1": 0.355,  "n": 1000, "agreement_with_record": 1.0}

    The harness's own records carry `pooled_aurc`, `min_decision_score` and per-type accuracy --
    not `pooled_top1`. So the one number their release states for verification was not being
    computed on this side at all, and the two sides had no metric in common.

    The harness is PROTECTED. Its `evaluate.py` defines what is scored and on which split, and it is
    not this project's to edit. But the per-question rows it writes contain `pred` and `gold`, so the
    published metric is recoverable without touching it -- which is the right place to recover it.

WHAT IT COMPUTES
    `pooled_top1` exactly as they define it: over every question in a (target, role, order) cell,
    the fraction where the arg-max option equals the gold option. Pooled across the cell's types,
    which is why the denominator for `typed_decisions` is 2000 and not 600.

    It also reports, per option order, the gap between canonical and reversed. Their v1.0 shows
    **no gap** (0.6525 on both), and that is a real property worth watching: a model that has learned
    to pick the option *whose content is right* scores the same under either presentation, while one
    that has learned to pick a *position* does not. A gap here is evidence about what the model
    learned, not about the harness.

USAGE
    python measurement/report_from_items.py <items.jsonl> [<items.jsonl> ...]
    python measurement/report_from_items.py --compare <ours.jsonl> --reference 0.6525
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

TARGETS = ("in_distribution", "mmlu_pro_1k", "typed_decisions")
ROLES = ("control", "candidate")
ORDERS = ("canonical", "reversed")

# From their v1.0 verify.json, for a printed comparison. Their release is a 2B backbone and ours is
# 0.6B, so these are a reference point for scale, not a target to be matched.
PUBLISHED_V1 = {
    "typed_decisions": 0.6525,
    "mmlu_pro_1k": 0.355,
}


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cell(rows: list[dict], target: str, role: str, order: str) -> list[dict]:
    return [r for r in rows
            if r.get("target") == target and r.get("role") == role and r.get("option_order") == order]


def pooled_top1(rs: list[dict]) -> tuple[float | None, int]:
    """Fraction of questions whose arg-max option is the gold one, pooled over the cell's types."""
    if not rs:
        return None, 0
    ok = sum(1 for r in rs if r.get("pred") == r.get("gold"))
    return ok / len(rs), len(rs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("items", nargs="*", help="items.jsonl files")
    ap.add_argument("--reference", type=float, default=None,
                    help="the published pooled_top1 to compare against (they publish 0.6525 for "
                         "typed_decisions on a 2B backbone)")
    a = ap.parse_args()
    if not a.items:
        print("no items.jsonl given; the harness writes <name>.items.jsonl beside <name>.jsonl")
        return 2

    for path_str in a.items:
        path = pathlib.Path(path_str)
        if not path.is_file():
            print(f"[skip] {path} not found")
            continue
        rows = load(path)
        if not rows:
            print(f"[skip] {path} is empty")
            continue
        print(f"\n=== {path}  ({len(rows)} rows) ===")
        print(f"{'target':18} {'role':10} {'canonical':>10} {'reversed':>10} {'gap(pp)':>9} {'n':>6}")
        for t in TARGETS:
            for role in ROLES:
                c, nc = pooled_top1(cell(rows, t, role, "canonical"))
                r, nr = pooled_top1(cell(rows, t, role, "reversed"))
                if c is None and r is None:
                    continue
                gap = f"{(c - r) * 100:+9.1f}" if (c is not None and r is not None) else "        —"
                cs = f"{c:10.4f}" if c is not None else "         —"
                rs_ = f"{r:10.4f}" if r is not None else "         —"
                print(f"  {t:16} {role:10} {cs} {rs_} {gap} {nc or nr:6}")

        print(f"\n  reference: their v1.0 publishes pooled_top1 "
              f"{PUBLISHED_V1.get('typed_decisions')} on typed_decisions and "
              f"{PUBLISHED_V1.get('mmlu_pro_1k')} on mmlu_pro_1k (2B backbone)")
        for t, ref in PUBLISHED_V1.items():
            c, n = pooled_top1(cell(rows, t, "control", "canonical"))
            if c is not None:
                print(f"    {t:18} zero-shot control {c:.4f}   their v1.0 {ref:.4f}   "
                      f"headroom {ref - c:+.4f}")
            c, n = pooled_top1(cell(rows, t, "candidate", "canonical"))
            if c is not None:
                print(f"    {t:18} candidate         {c:.4f}   their v1.0 {ref:.4f}   "
                      f"delta    {c - ref:+.4f}")

        # Their v1.0 scores identically under both orders. A gap here is a statement about what the
        # model learned, so it is surfaced rather than left in the table.
        print()
        for t in TARGETS:
            c, _ = pooled_top1(cell(rows, t, "candidate", "canonical"))
            r, _ = pooled_top1(cell(rows, t, "candidate", "reversed"))
            if c is not None and r is not None and abs(c - r) > 1e-9:
                print(f"  [note] {t}: candidate scores differently under the two option orders "
                      f"({c:.4f} vs {r:.4f}, {(c - r) * 100:+.1f} pp). Their v1.0 shows no gap, so a "
                      f"gap here indicates the readout is tracking presentation position as well as "
                      f"option content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
