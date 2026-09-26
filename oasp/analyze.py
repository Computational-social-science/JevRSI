"""Analysis CLI — turn a run directory into a report and figures.

Usage
-----
    python -m oasp.analyze --run-id pilot_v1_dev --split dev
    python -m oasp.analyze --run-id pilot_v1_dev --split dev --noise-run-id noise_v1
    python -m oasp.analyze --run-id search_v1 --landscape

Writes ``analysis/<run_id>/report.md``, ``report.json`` and figures. Never
overwrites an existing figure (``figures.py::_new_path``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, figures, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oasp.analyze")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--noise-run-id", default=None)
    ap.add_argument("--out", default=None, help="output dir (default analysis/<run_id>)")
    ap.add_argument("--landscape", action="store_true", help="also render landscape figures")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args(argv)

    run_dir = config.RUNS_DIR / args.run_id
    if not run_dir.exists():
        print(f"[FAIL] no such run: {run_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else (config.ANALYSIS_DIR / args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise_dir = (config.RUNS_DIR / args.noise_run_id) if args.noise_run_id else None
    rep = stats.compute_all(run_dir, noise_dir=noise_dir, split=args.split, B=args.bootstrap)

    md = stats.to_markdown(rep)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    chance = (rep["run_manifest"].get("benchmark") or {}).get("chance_level", 0.0)
    noise_sd = (rep.get("m6_noise_floor") or {}).get("max_sd", 0.0)
    figs: list[str] = []
    if rep["m1_spread"]:
        figs.append(str(figures.fig_prompt_sample_distribution(
            rep["m1_spread"], out_dir=out_dir, chance=chance, noise_sd=noise_sd)))
    if rep["m2_ips"]:
        figs.append(str(figures.fig_ips(rep["m2_ips"], out_dir=out_dir)))

    if args.landscape:
        hill_path = run_dir / "hillclimb.json"
        if hill_path.exists():
            hill = json.loads(hill_path.read_text(encoding="utf-8"))
            rows = figures.load_landscape(run_dir)
            if rows:
                figs.append(str(figures.fig_landscape(hill, rows, out_dir=out_dir)))
                figs.append(str(figures.fig_reason_codes(hill, rows, out_dir=out_dir)))

    print(md)
    print("\n--- artifacts ---")
    print(f"report: {out_dir / 'report.md'}")
    print(f"json:   {out_dir / 'report.json'}")
    for f in figs:
        print(f"figure: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
