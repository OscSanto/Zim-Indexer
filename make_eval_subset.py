#!/usr/bin/env python3
"""
Create a fixed evaluation subset as normalized JSONL.

The output format is intentionally simple and stable across all supported
source datasets:
  {"item_index": 175, "question": "...", "options": {"A": "...", ...}, "answer_idx": "B"}

That lets evaluate.py and infer.py consume one subset format for MedQA,
MedMCQA, MMLU-Pro, PubMedQA, or GUI CSV sources.

Examples:
  python make_eval_subset.py --dataset data/medqa_test.jsonl --n 100 --shuffle --seed 42 \
    --out data/medqa_fixed100.jsonl

  python make_eval_subset.py --dataset data/medqa_fixed100.jsonl --n 20 \
    --out data/medqa_quick20.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluate import load_dataset


def normalise_rows(dataset_path: Path, n: int | None = None) -> list[dict]:
    limit = n if n is not None else 999_999
    rows = load_dataset(dataset_path, limit)
    out = []
    for row in rows:
        opts = {k: v for k, v in row.get("options", {}).items() if v}
        out.append({
            "item_index": int(row.get("item_index", len(out) + 1)),
            "question": row["question"],
            "options": opts,
            "answer_idx": row["correct_key"],
            "source_dataset": dataset_path.name,
        })
    return out


def _parse_index_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def save_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_subset(
    dataset_path: Path,
    out_path: Path,
    n: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    indices: list[int] | None = None,
) -> list[dict]:
    rows = normalise_rows(dataset_path)
    if indices:
        wanted = set(indices)
        selected = [row for row in rows if int(row["item_index"]) in wanted]
    else:
        selected = list(rows)
        if shuffle:
            random.Random(seed).shuffle(selected)
        if n is not None:
            selected = selected[:n]
    save_jsonl(selected, out_path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed evaluation subset as normalized JSONL")
    parser.add_argument("--dataset", required=True, help="Source dataset path (JSONL/JSON/CSV)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--n", type=int, default=None, help="Number of rows to keep")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before slicing")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42)")
    parser.add_argument("--indices", default=None,
                        help="Comma-separated item_index values to keep (overrides --n/--shuffle)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    selected = build_subset(
        dataset_path,
        out_path,
        n=args.n,
        shuffle=args.shuffle,
        seed=args.seed,
        indices=_parse_index_list(args.indices) if args.indices else None,
    )

    print(f"Source : {dataset_path}")
    print(f"Output : {out_path}")
    print(f"Rows   : {len(selected)}")
    if selected:
        item_indexes = [str(r['item_index']) for r in selected]
        print(f"Items  : {', '.join(item_indexes)}")


if __name__ == "__main__":
    main()
