"""CLI for the prompt-space search (H2 / H3).

    python scripts/run_landscape.py --run-id search_v1 --generations 40 --items 30

Then analyse with:
    python -m oasp.analyze --run-id search_v1 --landscape
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasp import config, landscape


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark", default="arc_easy", choices=sorted(config.BENCHMARKS))
    ap.add_argument("--models", nargs="+", default=["gemma3_4b"],
                    choices=sorted(config.TARGET_MODELS))
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--items", type=int, default=30, help="items per evaluation (dev split)")
    ap.add_argument("--population", type=int, default=5)
    ap.add_argument("--cold-start-every", type=int, default=10)
    ap.add_argument("--plateau-window", type=int, default=6)
    ap.add_argument("--num-predict", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    result = landscape.run_search(
        run_id=args.run_id,
        benchmark_key=args.benchmark,
        model_keys=args.models,
        generations=args.generations,
        items=args.items,
        population_size=args.population,
        cold_start_every=args.cold_start_every,
        plateau_window=args.plateau_window,
        num_predict=args.num_predict,
        timeout=args.timeout,
        seed=args.seed,
    )
    print()
    print(json.dumps(
        {k: result[k] for k in (
            "run_id", "seed_score", "best_score", "gain_over_seed",
            "accepted_total", "evaluated_total", "fallback_total",
            "mechanism_counts", "reason_code_counts", "stalled", "wall_seconds",
        )},
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
