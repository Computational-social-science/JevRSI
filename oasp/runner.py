"""Factorial execution with resume.

Every ``(split, model, variant, item)`` cell is one row in an append-only JSONL
keyed by a stable ``cell_id``. Re-running the same ``--run-id`` skips completed
cells instead of overwriting them, so a long matrix can be interrupted and
resumed — and a partial run is still analysable
(RESEARCH_PROTOCOL.md §9).

Concurrency is **per model**: Ollama serialises work for a single model on the
GPU, so parallelising inside a model buys nothing and thrashes VRAM. One worker
per model in flight is the right shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import benchmark, config, evaluator, llm, prompt_space


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_id(split: str, model_key: str, vid: str, item_id: str) -> str:
    return f"{split}|{model_key}|{vid}|{item_id}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return ""


def load_done_cells(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["cell_id"])
            except Exception:  # noqa: BLE001 - tolerate a torn final line
                continue
    return done


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def build_cells(
    *,
    splits: dict[str, list],
    variants: list[prompt_space.Variant],
    model_keys: list[str],
) -> list[dict]:
    cells = []
    for split, items in splits.items():
        for mk in model_keys:
            for v in variants:
                for it in items:
                    cells.append(
                        {
                            "cell_id": _cell_id(split, mk, v.vid, it.item_id),
                            "split": split,
                            "model_key": mk,
                            "variant_label": v.label,
                            "vid": v.vid,
                            "features": v.features,
                            "item_id": it.item_id,
                            "prompt": prompt_space.render(v.features, it),
                        }
                    )
    return cells


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_one_model(
    model_key: str,
    cells: list[dict],
    *,
    results_path: Path,
    temperature: float,
    seed: int,
    num_predict: int,
    timeout: int,
    progress: bool = True,
) -> dict:
    spec = config.TARGET_MODELS[model_key]
    done = load_done_cells(results_path)
    todo = [c for c in cells if c["cell_id"] not in done]

    stats = {"model": model_key, "planned": len(cells), "skipped": len(cells) - len(todo),
             "ran": 0, "failed": 0}

    t_start = time.time()
    for i, c in enumerate(todo, 1):
        comp = llm.chat(
            spec,
            c["prompt"],
            temperature=temperature,
            seed=seed,
            num_predict=num_predict,
            timeout=timeout,
        )
        if comp.ok:
            sc = evaluator.score(comp.text, _item_index_cache[c["item_id"]])
        else:
            sc = evaluator.Score(False, None, False)
            stats["failed"] += 1

        _append_rows(
            results_path,
            [
                {
                    "cell_id": c["cell_id"],
                    "split": c["split"],
                    "model": model_key,
                    "model_id": spec.model_id,
                    "variant": c["variant_label"],
                    "vid": c["vid"],
                    "features": c["features"],
                    "item_id": c["item_id"],
                    "raw": comp.text,
                    "extracted": sc.extracted,
                    "correct": bool(sc.correct),
                    "parse_ok": bool(sc.parse_ok),
                    "matched_choice_text": bool(sc.matched_choice_text),
                    "ok": bool(comp.ok),
                    "error": comp.error,
                    "latency_ms": comp.latency_ms,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            ],
        )
        stats["ran"] += 1
        if progress and (i % 25 == 0 or i == len(todo)):
            el = time.time() - t_start
            rate = i / el if el > 0 else 0.0
            print(
                f"    [{model_key}] {i}/{len(todo)}  "
                f"({rate:.1f}/s, eta {(len(todo)-i)/rate/60:.1f} min)"
                if rate > 0
                else f"    [{model_key}] {i}/{len(todo)}",
                flush=True,
            )
    stats["seconds"] = round(time.time() - t_start, 1)
    return stats


_item_index_cache: dict[str, benchmark.Item] = {}


def run_matrix(
    *,
    run_id: str,
    benchmark_key: str,
    model_keys: list[str],
    n_variants: int,
    n_items_dev: int,
    n_items_test: int,
    splits_to_run: list[str],
    temperature: float | None = None,
    seed: int | None = None,
    num_predict: int | None = None,
    timeout: int = 120,
    concurrency: int | None = None,
) -> dict:
    """Execute the crossed design and return the run manifest."""
    d = config.DESIGN
    temperature = d["temperature"] if temperature is None else temperature
    seed = d["seed"] if seed is None else seed
    num_predict = d["num_predict"] if num_predict is None else num_predict
    concurrency = concurrency or len(model_keys)

    config.assert_decoupled(model_keys)

    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"

    # --- stimulus ---
    items = benchmark.load_items(benchmark_key)
    _item_index_cache.clear()
    _item_index_cache.update({it.item_id: it for it in items})

    split_map = benchmark.stratified_split(
        items, n_dev=n_items_dev, n_test=n_items_test, seed=seed
    )
    splits = {s: split_map[s] for s in splits_to_run}

    # --- prompt space ---
    variants = prompt_space.sample_variants(n=n_variants, seed=seed)
    (run_dir / "variants.json").write_text(
        json.dumps([v.as_row() for v in variants], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cells = build_cells(splits=splits, variants=variants, model_keys=model_keys)

    manifest = {
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "exploratory" if "pilot" in run_id else "confirmatory",
        "benchmark": {
            "key": benchmark_key,
            "provenance": benchmark.provenance(benchmark_key),
            "n_items_total": len(items),
            "n_dev": len(split_map["dev"]),
            "n_test": len(split_map["test"]),
            "splits_run": splits_to_run,
            "chance_level": round(evaluator.chance_level(items), 4),
        },
        "design": {
            **{k: d[k] for k in (
                "temperature", "seed", "num_predict", "n_variants",
                "noise_floor_seeds", "noise_floor_temp", "noise_floor_items",
                "bootstrap_B", "permutation_B",
            )},
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "n_variants": n_variants,
            "n_items_dev": n_items_dev,
            "n_items_test": n_items_test,
            "decision_rules": {k: v for k, v in d.items() if k.startswith(("h1_", "h3_", "bh_", "alpha"))},
        },
        "models": [
            {
                "key": mk,
                "model_id": config.TARGET_MODELS[mk].model_id,
                "backend": config.TARGET_MODELS[mk].backend,
                "family": config.TARGET_MODELS[mk].family,
                "digest": llm.model_digest(config.TARGET_MODELS[mk].model_id),
            }
            for mk in model_keys
        ],
        "decoupling": {
            "targets": [config.TARGET_MODELS[mk].model_id for mk in model_keys],
            "optimizers": [m.model_id for m in config.OPTIMIZER_MODELS.values()],
            "asserted_disjoint": True,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": _git_commit(config.PROJECT_ROOT),
        },
        "n_cells_planned": len(cells),
        "results_sha256_before": _sha256_file(results_path) if results_path.exists() else "",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  run_id={run_id}  cells={len(cells)}  models={model_keys}  concurrency={concurrency}")
    print(f"  benchmark={benchmark_key}  items: dev={len(split_map['dev'])} test={len(split_map['test'])}"
          f"  variants={len(variants)}")

    if not cells:
        print("  nothing to do")
        return manifest

    by_model: dict[str, list[dict]] = {mk: [] for mk in model_keys}
    for c in cells:
        by_model[c["model_key"]].append(c)

    t0 = time.time()
    all_stats = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(
                _run_one_model,
                mk,
                by_model[mk],
                results_path=results_path,
                temperature=temperature,
                seed=seed,
                num_predict=num_predict,
                timeout=timeout,
            ): mk
            for mk in model_keys
            if by_model[mk]
        }
        for fut in as_completed(futs):
            st = fut.result()
            all_stats.append(st)
            print(f"  done {st['model']}: ran={st['ran']} failed={st['failed']} "
                  f"skipped={st['skipped']} in {st['seconds']}s", flush=True)

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    manifest["per_model_stats"] = sorted(all_stats, key=lambda s: s["model"])
    manifest["results_sha256_after"] = _sha256_file(results_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oasp.runner", description="Run the OASP prompt-sensitivity matrix")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--benchmark", default="arc_easy", choices=sorted(config.BENCHMARKS))
    ap.add_argument("--models", nargs="+", default=["gemma3_4b", "qwen3_4b"],
                    choices=sorted(config.TARGET_MODELS))
    ap.add_argument("--variants", type=int, default=config.DESIGN["n_variants"])
    ap.add_argument("--items-dev", type=int, default=config.DESIGN["n_items_dev"])
    ap.add_argument("--items-test", type=int, default=config.DESIGN["n_items_test"])
    ap.add_argument("--splits", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-predict", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args(argv)

    m = run_matrix(
        run_id=args.run_id,
        benchmark_key=args.benchmark,
        model_keys=args.models,
        n_variants=args.variants,
        n_items_dev=args.items_dev,
        n_items_test=args.items_test,
        splits_to_run=args.splits,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
        timeout=args.timeout,
        concurrency=args.concurrency,
    )
    print(json.dumps({k: m[k] for k in ("run_id", "n_cells_planned", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
