"""M6 — the stochastic noise floor.

Re-runs ONE fixed prompt (the canonical ``p0``) at temperature > 0 over several
seeds on a fixed item subset. The resulting SD of accuracy is the yardstick the
H1 decision rule compares the prompt-induced spread against
(RESEARCH_PROTOCOL.md §4, M6).

Why a separate run: the main matrix is decoded greedily so that a score
difference between two variants is a *causal* effect of wording. This run
deliberately reintroduces sampling noise to answer a different question — how
large would a spread be if the prompt made no difference at all but decoding were
stochastic? Without it, "range = 12 points" has no scale.

Run: ``python scripts/run_noise_floor.py --run-id noise_v1 --models gemma3_4b ...``
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oasp import benchmark, config, evaluator, llm, prompt_space
from oasp.runner import _append_rows, _git_commit, _sha256_file, load_done_cells


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="noise_v1")
    ap.add_argument("--benchmark", default="arc_easy")
    ap.add_argument("--models", nargs="+", default=["gemma3_4b", "mistral_7b"])
    ap.add_argument("--replicates", type=int, default=config.DESIGN["noise_floor_seeds"])
    ap.add_argument("--items", type=int, default=config.DESIGN["noise_floor_items"])
    ap.add_argument("--temperature", type=float, default=config.DESIGN["noise_floor_temp"])
    ap.add_argument("--num-predict", type=int, default=192)
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args(argv)

    config.assert_decoupled(args.models)

    items_all = benchmark.load_items(args.benchmark)
    split = benchmark.stratified_split(
        items_all, n_dev=args.items, n_test=args.items, seed=config.DESIGN["seed"]
    )
    probe = split["dev"]

    variant = prompt_space.sample_variants(n=1, seed=config.DESIGN["seed"])[0]

    run_dir = config.RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    manifest = {
        "run_id": args.run_id,
        "kind": "noise_floor",
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "exploratory",
        "benchmark": {
            "key": args.benchmark,
            "provenance": benchmark.provenance(args.benchmark),
            "chance_level": round(evaluator.chance_level(probe), 4),
        },
        "protocol": {
            "fixed_variant": variant.label,
            "fixed_variant_features": variant.features,
            "temperature": args.temperature,
            "replicates": args.replicates,
            "items": len(probe),
            "num_predict": args.num_predict,
        },
        "models": [config.TARGET_MODELS[m].model_id for m in args.models],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(config.PROJECT_ROOT),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    done = load_done_cells(results_path)
    print(f"noise floor: {args.replicates} replicates x {len(probe)} items x {len(args.models)} models"
          f" at T={args.temperature}")

    t0 = time.time()
    for mk in args.models:
        spec = config.TARGET_MODELS[mk]
        for rep in range(args.replicates):
            for it in probe:
                cell = f"noise|{mk}|{variant.vid}|{it.item_id}|r{rep}"
                if cell in done:
                    continue
                comp = llm.chat(
                    spec,
                    prompt_space.render(variant.features, it),
                    temperature=args.temperature,
                    seed=config.DESIGN["seed"] + rep,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                sc = (
                    evaluator.score(comp.text, it)
                    if comp.ok
                    else evaluator.Score(False, None, False)
                )
                _append_rows(
                    results_path,
                    [
                        {
                            "cell_id": cell,
                            "split": "dev",
                            "model": mk,
                            "model_id": spec.model_id,
                            "variant": variant.label,
                            "vid": variant.vid,
                            "features": variant.features,
                            "item_id": it.item_id,
                            "replicate": rep,
                            "raw": comp.text,
                            "extracted": sc.extracted,
                            "correct": bool(sc.correct),
                            "parse_ok": bool(sc.parse_ok),
                            "ok": bool(comp.ok),
                            "error": comp.error,
                            "latency_ms": comp.latency_ms,
                            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                    ],
                )
        print(f"  {mk} done ({time.time() - t0:.0f}s elapsed)", flush=True)

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    manifest["results_sha256_after"] = _sha256_file(results_path)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"noise floor complete in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
