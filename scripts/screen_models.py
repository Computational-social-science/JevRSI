"""Capability / parse / speed screen.

Purpose: choose the **target set and generation budget from data**, not from
assumption. Two failure modes must be excluded before a prompt-sensitivity matrix
means anything:

* **Capability floor** — a model at chance cannot show a prompt effect, because
  there is no signal for the wording to move. Reported as Wilson lower bound vs
  chance (RESEARCH_PROTOCOL.md §3).
* **Format non-adherence** — a model that never emits a parseable option letter
  scores 0 for every variant. That is a floor in the *scorer*, not in the prompt,
  and it would masquerade as perfect prompt-invariance (a zero-variance cell).

Run: ``python scripts/screen_models.py --items 10 --budgets 64 192``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasp import benchmark, config, evaluator, llm, prompt_space


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="arc_easy")
    ap.add_argument("--items", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[64, 192])
    ap.add_argument("--models", nargs="+", default=sorted(config.TARGET_MODELS))
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args(argv)

    items = benchmark.load_items(args.benchmark)
    split = benchmark.stratified_split(items, n_dev=args.items, n_test=args.items, seed=config.DESIGN["seed"])
    probe = split["dev"]
    chance = evaluator.chance_level(probe)

    # One terse variant (canonical) and one that invites prose, so the screen sees
    # the worst case for parseability rather than only the easy case.
    vs = prompt_space.sample_variants(n=32, seed=config.DESIGN["seed"])
    canonical = vs[0]
    verbose = next(
        v for v in vs if v.features["verb"] == "Answer" and v.features["no_explanation"] is False
    )

    print(f"benchmark={args.benchmark} items={len(probe)} chance={chance:.3f} "
          f"variants={canonical.label},{verbose.label}")
    print()
    hdr = f"{'model':<16}{'budget':>7}{'parse':>8}{'acc':>8}{'acc|parsed':>12}{'mean_ms':>10}{'tok/s':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for mk in args.models:
        spec = config.TARGET_MODELS[mk]
        for budget in args.budgets:
            n = ok = corr = corr_parsed = parsed = 0
            lat = []
            toks = []
            for v in (canonical, verbose):
                for it in probe:
                    comp = llm.chat(
                        spec,
                        prompt_space.render(v.features, it),
                        temperature=config.DESIGN["temperature"],
                        seed=config.DESIGN["seed"],
                        num_predict=budget,
                        timeout=args.timeout,
                    )
                    n += 1
                    lat.append(comp.latency_ms)
                    if comp.completion_tokens:
                        toks.append(comp.completion_tokens)
                    if not comp.ok:
                        continue
                    ok += 1
                    sc = evaluator.score(comp.text, it)
                    corr += int(sc.correct)
                    if sc.parse_ok:
                        parsed += 1
                        corr_parsed += int(sc.correct)
            mean_ms = sum(lat) / len(lat) if lat else 0
            tps = (sum(toks) / (sum(lat) / 1000.0)) if toks and sum(lat) > 0 else 0
            print(
                f"{mk:<16}{budget:>7}{parsed/n:>8.3f}{corr/n:>8.3f}"
                f"{(corr_parsed/parsed if parsed else 0):>12.3f}{mean_ms:>10.0f}{tps:>8.1f}"
            )
            rows.append((mk, budget, parsed / n, corr / n, mean_ms, tps))

    print()
    print("verdict (parse>=0.90 and acc>chance):")
    for mk, budget, pr, acc, mean_ms, tps in rows:
        good = pr >= 0.90 and acc > chance
        print(f"  {mk:<16} budget={budget:<5} {'USABLE' if good else 'EXCLUDE'}"
              f"  (parse={pr:.2f}, acc={acc:.2f} vs chance {chance:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
