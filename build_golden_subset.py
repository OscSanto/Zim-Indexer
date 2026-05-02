#!/usr/bin/env python3
"""
Create a gold retrieval subset from a manually reviewed audit CSV.

Rows are kept when:
  - manual_include_in_gold is truthy
  - manual_answer_article_exists is truthy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


_OPTION_KEYS = [chr(code) for code in range(ord("A"), ord("J") + 1)]
_TRUE = {"1", "y", "yes", "true", "t"}


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in _TRUE


def build_golden_subset(review_path: Path, out_path: Path) -> list[dict]:
    rows_out: list[dict] = []
    with open(review_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not (_truthy(row.get("manual_include_in_gold", "")) and
                    _truthy(row.get("manual_answer_article_exists", ""))):
                continue

            opts = {
                key: row.get(f"option_{key}", "")
                for key in _OPTION_KEYS
                if row.get(f"option_{key}", "").strip()
            }
            rows_out.append({
                "item_index": int(row.get("item_index") or row.get("num") or len(rows_out) + 1),
                "question": row["question"],
                "options": opts,
                "answer_idx": row["correct_key"],
                "correct_title": row.get("manual_canonical_article_title", "").strip(),
                "source_review_csv": review_path.name,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a gold subset from a reviewed audit CSV")
    parser.add_argument("--review", required=True, help="Reviewed audit CSV")
    parser.add_argument("--out", required=True, help="Output normalized JSONL path")
    args = parser.parse_args()

    review_path = Path(args.review)
    out_path = Path(args.out)
    rows_out = build_golden_subset(review_path, out_path)

    print(f"Review : {review_path}")
    print(f"Output : {out_path}")
    print(f"Rows   : {len(rows_out)}")


if __name__ == "__main__":
    main()
