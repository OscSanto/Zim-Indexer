#!/usr/bin/env python3
"""
Qualitative comparison: structured vs flat retrieval.

Reads a per-question CSV produced by evaluate.py and prints the cases where
one system found the answer and the other did not. Useful for error analysis
and for the paper's qualitative section.

Usage:
  python compare.py results/medqa_results.csv \\
    --sys-a "Hybrid Struct + Lead" \\
    --sys-b "Hybrid Flat" \\
    --show 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def _rank_col(label: str, rows: list[dict]) -> str | None:
    """Find the column name for the given system label."""
    for col in rows[0].keys() if rows else []:
        if col.endswith("_rank") and label in col:
            return col
    return None


def _hit10_col(label: str, rows: list[dict]) -> str | None:
    for col in rows[0].keys() if rows else []:
        if col.endswith("_hit@10") and label in col:
            return col
    return None


def compare(
    rows: list[dict],
    sys_a: str,
    sys_b: str,
    show: int,
) -> None:
    col_a = _hit10_col(sys_a, rows)
    col_b = _hit10_col(sys_b, rows)

    if not col_a:
        print(f"[error] No hit@10 column found for system: {sys_a!r}")
        print(f"  Available columns: {list(rows[0].keys()) if rows else []}")
        sys.exit(1)
    if not col_b:
        print(f"[error] No hit@10 column found for system: {sys_b!r}")
        sys.exit(1)

    # Cases where A hit and B missed (A advantage)
    a_wins = [r for r in rows if r[col_a] == "1" and r[col_b] == "0"]
    # Cases where B hit and A missed (B advantage)
    b_wins = [r for r in rows if r[col_b] == "1" and r[col_a] == "0"]
    # Both hit
    both   = [r for r in rows if r[col_a] == "1" and r[col_b] == "1"]
    # Both missed
    neither = [r for r in rows if r[col_a] == "0" and r[col_b] == "0"]

    n = len(rows)
    print(f"\n{'─'*70}")
    print(f"  Comparison: {sys_a!r}  vs  {sys_b!r}")
    print(f"  Total questions: {n}")
    print(f"  Both hit    : {len(both):4d}  ({len(both)/n:.1%})")
    print(f"  A only      : {len(a_wins):4d}  ({len(a_wins)/n:.1%})  ← {sys_a} advantage")
    print(f"  B only      : {len(b_wins):4d}  ({len(b_wins)/n:.1%})  ← {sys_b} advantage")
    print(f"  Neither hit : {len(neither):4d}  ({len(neither)/n:.1%})")
    print(f"{'─'*70}")

    def _show_cases(label: str, cases: list[dict], limit: int) -> None:
        if not cases:
            return
        print(f"\n── {label} (showing {min(limit, len(cases))} of {len(cases)}) ──")
        for i, r in enumerate(cases[:limit]):
            rk_a = r.get(f"{sys_a}_rank", "?")
            rk_b = r.get(f"{sys_b}_rank", "?")
            print(f"\n  [{int(r['num']):>3}] {r['question'][:100]}")
            print(f"       Answer: {r['correct_key']} — {r.get('correct_text','')[:60]}")
            print(f"       {sys_a} rank: {rk_a or 'miss'}   {sys_b} rank: {rk_b or 'miss'}")

    _show_cases(f"{sys_a} HIT / {sys_b} MISS  (structure helped)", a_wins, show)
    _show_cases(f"{sys_b} HIT / {sys_a} MISS  (flat beat structure)", b_wins, show)


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative comparison between two retrieval systems"
    )
    parser.add_argument("csv",   help="Per-question CSV from evaluate.py")
    parser.add_argument("--sys-a", default="Hybrid Struct + Lead",
                        help="Label of system A (default: 'Hybrid Struct + Lead')")
    parser.add_argument("--sys-b", default="Hybrid Flat",
                        help="Label of system B (default: 'Hybrid Flat')")
    parser.add_argument("--show", type=int, default=10,
                        help="Max examples to print per category (default 10)")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[error] File not found: {path}")
        sys.exit(1)

    _, rows = _load_csv(path)
    if not rows:
        print("[error] CSV is empty")
        sys.exit(1)

    compare(rows, args.sys_a, args.sys_b, args.show)


if __name__ == "__main__":
    main()
