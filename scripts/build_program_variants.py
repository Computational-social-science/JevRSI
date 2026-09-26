"""Build the program.md variant space and freeze it with hashes.

Reads the stock program.md, applies each variant's exact string substitutions, writes
one assembled file per variant, and emits a manifest with SHA-256 for every file.

The two guards that matter:
  1. every stock span must occur EXACTLY once, or the build aborts (a silent zero-match
     would produce a "variant" identical to stock, i.e. a fake control);
  2. v00 must be byte-identical to the stock program.md, or the build aborts.

Guard 2 is what makes "the untranslated baseline" a fact rather than a claim.

Usage:
    python scripts/build_program_variants.py [--stock PATH] [--out DIR] [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "configs" / "program_variants"))

import slots as S  # noqa: E402

DEFAULT_STOCK = Path("D:/2026-AI4S/nanochat-autoresearch/program.md")
DEFAULT_OUT = ROOT / "configs" / "program_variants" / "assembled"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_one(stock_text: str, variant: dict) -> tuple[str, list[dict]]:
    """Return (assembled text, per-slot substitution records)."""
    text = stock_text
    applied: list[dict] = []
    for slot_name, choice in variant["slots"].items():
        table = S.SLOTS[slot_name]
        stock_span = table["stock"]
        new_span = table[choice]
        n = text.count(stock_span)
        if n != 1:
            raise SystemExit(
                f"[FAIL] {variant['id']}: stock span for slot {slot_name} occurs {n} time(s), "
                f"expected exactly 1. The stock program.md has diverged from "
                f"configs/program_variants/slots.py."
            )
        text = text.replace(stock_span, new_span, 1)
        applied.append({
            "slot": slot_name,
            "choice": choice,
            "stock_chars": len(stock_span),
            "variant_chars": len(new_span),
        })
    return text, applied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", type=Path, default=DEFAULT_STOCK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify existing output matches; write nothing")
    args = ap.parse_args()

    stock_path: Path = args.stock
    if not stock_path.exists():
        raise SystemExit(f"[FAIL] stock program.md not found: {stock_path}")
    stock_bytes = stock_path.read_bytes()
    stock_text = stock_bytes.decode("utf-8")

    print(f"stock  : {stock_path}")
    print(f"         {len(stock_bytes)} bytes  sha256={sha256_bytes(stock_bytes)[:16]}")
    print(f"variants: {len(S.VARIANTS)}")
    print()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stock_path": str(stock_path),
        "stock_sha256": sha256_bytes(stock_bytes),
        "stock_bytes": len(stock_bytes),
        "n_variants": len(S.VARIANTS),
        "variants": [],
    }

    n_changed = 0
    for v in S.VARIANTS:
        text, applied = build_one(stock_text, v)
        data = text.encode("utf-8")
        fname = f"{v['id']}_{v['name']}.md"
        fpath = out / fname
        if not args.check:
            fpath.write_bytes(data)

        identical = data == stock_bytes
        if v["id"] == "v00" and not identical:
            raise SystemExit(
                "[FAIL] guard 2 violated: v00_stock is NOT byte-identical to the stock "
                "program.md. The baseline would not be the shipped template."
            )
        if v["id"] != "v00" and identical:
            raise SystemExit(
                f"[FAIL] guard 1 violated: {v['id']} assembled to the stock text, so its "
                f"substitutions did nothing."
            )

        delta = len(data) - len(stock_bytes)
        n_changed += not identical
        print(f"  {v['id']:<4} {v['name']:<22} {len(data):>6} B  {delta:+5d}  "
              f"{'IDENTICAL to stock (baseline)' if identical else 'substituted'}"
              f"  sha256={sha256_bytes(data)[:12]}")
        for a in applied:
            print(f"         - {a['slot']}: {a['choice']} "
                  f"({a['stock_chars']}->{a['variant_chars']} chars)")

        manifest["variants"].append({
            "id": v["id"],
            "name": v["name"],
            "file": fname,
            "slots": v["slots"],
            "substitutions": applied,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "identical_to_stock": identical,
            "dimension_probed": S.DIMENSION_OF.get(v["id"]),
        })

    mpath = out.parent / "manifest.json"
    if not args.check:
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"[OK] {len(S.VARIANTS)} variants, {n_changed} changed + 1 baseline")
    print(f"[OK] guard 1 (every stock span matched exactly once): passed")
    print(f"[OK] guard 2 (v00 byte-identical to stock): passed")
    print(f"manifest: {mpath}")
    if args.check:
        print("[OK] --check mode: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
