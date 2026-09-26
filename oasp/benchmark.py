"""Frozen benchmark loading.

The stimulus set is downloaded **once** from the HF mirror and frozen to JSONL
with the source file's SHA-256 recorded. Later upstream revisions therefore
cannot silently change the items under an existing run id
(RESEARCH_PROTOCOL.md §9).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from . import config


@dataclass(frozen=True)
class Item:
    item_id: str
    question: str
    labels: tuple[str, ...]
    texts: tuple[str, ...]
    answer_key: str

    def as_row(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "labels": list(self.labels),
            "texts": list(self.texts),
            "answer_key": self.answer_key,
        }

    @staticmethod
    def from_row(row: dict) -> "Item":
        return Item(
            item_id=str(row["item_id"]),
            question=str(row["question"]),
            labels=tuple(row["labels"]),
            texts=tuple(row["texts"]),
            answer_key=str(row["answer_key"]),
        )


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _rows_from_frame(df: pd.DataFrame, fmt: str, limit: int | None) -> list[dict]:
    """Convert an upstream frame into frozen ``Item`` rows.

    Each benchmark ships a different schema, so the parser is selected by the
    registry's ``fmt`` field rather than guessed from column names.
    """
    rows: list[dict] = []
    if fmt == "arc":
        for _, r in df.iterrows():
            ch = r["choices"]
            rows.append(
                Item(
                    item_id=str(r["id"]),
                    question=str(r["question"]).strip(),
                    labels=tuple(str(x) for x in ch["label"]),
                    texts=tuple(str(x) for x in ch["text"]),
                    answer_key=str(r["answerKey"]).strip(),
                ).as_row()
            )
            if limit is not None and len(rows) >= limit:
                break
        return rows

    if fmt == "mmlu_pro":
        # MMLU-Pro has no label column: options are a plain list of strings and the
        # key is a letter with an accompanying 0-based index. Option arrays are
        # VARIABLE length (4-10), so labels must be generated per row and the chance
        # level differs per item.
        # to_dict("records") rather than iterrows(): iterrows yields Series whose
        # scalar indexing defeats static typing, and dicts are what we actually want.
        letters = "ABCDEFGHIJ"
        for r in df.to_dict("records"):
            opts = list(r["options"])
            ans_idx = int(r["answer_index"])
            if not (0 <= ans_idx < len(opts)):
                continue  # malformed upstream row; skip rather than mis-key it
            rows.append(
                Item(
                    item_id=str(r["question_id"]),
                    question=str(r["question"]).strip(),
                    labels=tuple(letters[: len(opts)]),
                    texts=tuple(str(o) for o in opts),
                    answer_key=letters[ans_idx],
                ).as_row()
            )
            if limit is not None and len(rows) >= limit:
                break
        return rows

    if fmt == "supergpqa":
        # SuperGPQA ships as JSONL with NO generated labels: `options` is a plain list
        # and while `answer` holds the full answer TEXT, the key letter is in
        # `answer_letter`. Keying off `answer` would silently mark every item wrong —
        # it is not a letter at all.
        letters = "ABCDEFGHIJ"
        for r in df.to_dict("records"):
            opts = list(r["options"])
            ans = str(r["answer_letter"]).strip().upper()
            if ans not in letters[: len(opts)]:
                continue  # key outside its own option range; skip as malformed
            rows.append(
                Item(
                    item_id=str(r["uuid"]),
                    question=str(r["question"]).strip(),
                    labels=tuple(letters[: len(opts)]),
                    texts=tuple(str(o) for o in opts),
                    answer_key=ans,
                ).as_row()
            )
            if limit is not None and len(rows) >= limit:
                break
        return rows

    raise ValueError(f"unknown benchmark fmt {fmt!r}")


def _load_frame(payload: bytes, fmt: str) -> pd.DataFrame:
    if fmt == "supergpqa":
        return pd.read_json(io.BytesIO(payload), lines=True)
    return pd.read_parquet(io.BytesIO(payload))


def freeze_benchmark(key: str, *, split: str = "test", limit: int | None = None,
                     force: bool = False) -> Path:
    """Download ``key`` from the HF mirror and write ``benchmarks/<key>/items.jsonl``.

    Returns the path to the frozen JSONL. Idempotent unless ``force``.
    """
    spec = config.BENCHMARKS[key]
    out_dir = config.BENCH_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    prov_path = out_dir / "provenance.json"

    if items_path.exists() and prov_path.exists() and not force:
        return items_path

    url = spec.url()
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    payload = resp.content
    src_sha = _sha256_bytes(payload)

    df = _load_frame(payload, spec.fmt)
    rows = _rows_from_frame(df, spec.fmt, limit)

    with items_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Per-item chance: the exact value is 1/len(options) and it varies whenever the
    # benchmark uses variable-length option arrays.
    chances = [1.0 / len(r["texts"]) for r in rows if r["texts"]]
    prov = {
        "key": key,
        "url": url,
        "hf_repo": spec.hf_repo,
        "hf_path": spec.hf_path,
        "fmt": spec.fmt,
        "release": spec.release,
        "source_sha256": src_sha,
        "n_items": len(rows),
        "n_source_rows": int(len(df)),
        "chance_nominal": spec.chance,
        "chance_mean": round(sum(chances) / len(chances), 6) if chances else None,
        "option_count_counts": {str(n): chances.count(1.0 / n) for n in sorted({int(round(1 / c)) for c in chances})}
        if chances
        else {},
        "notes": spec.notes,
        "frozen": True,
    }
    prov_path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
    return items_path


def load_items(key: str) -> list[Item]:
    path = config.BENCH_DIR / key / "items.jsonl"
    if not path.exists():
        path = freeze_benchmark(key)
    out: list[Item] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Item.from_row(json.loads(line)))
    return out


def stratified_split(items: list[Item], *, n_dev: int, n_test: int, seed: int) -> dict[str, list[Item]]:
    """Deterministic split over items.

    Items are sorted by id before permutation so the split does not depend on
    the order the upstream parquet happened to use.
    """
    import random

    ordered = sorted(items, key=lambda it: it.item_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    need = n_dev + n_test
    if need > len(ordered):
        raise ValueError(f"asked for {need} items but benchmark has {len(ordered)}")
    dev = ordered[:n_dev]
    test = ordered[n_dev : n_dev + n_test]
    return {"dev": dev, "test": test}


def render_question_block(item: Item) -> str:
    """The invariant part of the stimulus: question + enumerated options."""
    lines = [f"Question: {item.question}"]
    for lbl, txt in zip(item.labels, item.texts):
        lines.append(f"{lbl}. {txt}")
    return "\n".join(lines)


def provenance(key: str) -> dict:
    p = config.BENCH_DIR / key / "provenance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
