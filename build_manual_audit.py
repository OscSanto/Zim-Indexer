#!/usr/bin/env python3
"""
Build a manual retrieval-audit CSV for a fixed subset.

The audit CSV is designed for human review. You mark whether the correct answer
has a real article in the ZIM, record the canonical article title, and then
mark whether each retrieval condition surfaced that title.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluate import load_dataset
from indexer.db import open_db, init_fts
from indexer.query import search


_OPTION_KEYS = [chr(code) for code in range(ord("A"), ord("J") + 1)]


def _top_titles(hits: list[dict], k: int = 5) -> list[str]:
    return [(h.get("title") or "").strip() for h in hits[:k]]


def _titles_pipe(hits: list[dict], k: int = 5) -> str:
    return " | ".join(title for title in _top_titles(hits, k) if title)


def _exact_title_matches(con, answer_text: str) -> list[str]:
    if not answer_text.strip():
        return []
    rows = con.execute(
        "SELECT title FROM articles WHERE lower(title) = lower(?) ORDER BY title LIMIT 5",
        (answer_text.strip(),),
    ).fetchall()
    return [r[0] for r in rows]


def _title_search_matches(con, answer_text: str) -> list[str]:
    if not answer_text.strip():
        return []
    try:
        rows = con.execute(
            "SELECT title FROM articles_fts WHERE articles_fts MATCH ? ORDER BY rank LIMIT 5",
            (answer_text.strip(),),
        ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _write_instructions(path: Path) -> None:
    path.write_text(
        "\n".join([
            "Manual Retrieval Audit Instructions",
            "",
            "Goal:",
            "1. Decide whether the correct answer corresponds to a real article in the ZIM.",
            "2. If yes, record the canonical article title.",
            "3. Mark whether each retrieval condition surfaced that article.",
            "",
            "Columns to fill in:",
            "- manual_answer_article_exists: yes / no",
            "- manual_canonical_article_title: exact article title in the ZIM",
            "- manual_include_in_gold: yes / no",
            "- manual_bm25_hit: yes / no",
            "- manual_flat_hit: yes / no",
            "- manual_struct_hit: yes / no",
            "- manual_struct_lead_hit: yes / no",
            "- manual_struct_section_hit: yes / no",
            "- manual_struct_both_hit: yes / no",
            "- manual_notes: optional notes about ambiguity, redirects, or mismatches",
            "",
            "Recommended rule:",
            "- Include a row in the gold subset only if the correct answer clearly maps to a real ZIM article.",
            "- Mark a retrieval hit only if the retrieved title is that article or an unambiguous equivalent.",
        ]),
        encoding="utf-8",
    )


def build_manual_audit(
    dataset_path: Path,
    struct_dir: Path,
    flat_dir: Path | None,
    out_path: Path,
    top_k: int = 10,
) -> tuple[Path, Path, int]:
    questions = load_dataset(dataset_path, 999_999)

    base_cfg = {
        "use_faiss": True,
        "use_title_bm25": True,
        "use_para_bm25": True,
        "use_diversity_cap": False,
        "use_mention_penalty": False,
        "use_nav_boost": False,
        "eval_rrf_k": 60,
        "eval_diversity_max": 6,
        "embed_model": "BAAI/bge-small-en-v1.5",
        "faiss_mmap": True,
    }
    systems: list[tuple[str, Path | None, dict]] = [
            ("bm25", struct_dir, {**base_cfg, "use_faiss": False, "use_lead_augment": False, "use_section_augment": False}),
            ("flat", flat_dir,   {**base_cfg, "use_lead_augment": False, "use_section_augment": False}),
            ("struct", struct_dir, {**base_cfg, "use_lead_augment": False, "use_section_augment": False}),
            ("struct_lead", struct_dir, {**base_cfg, "use_lead_augment": True, "use_section_augment": False}),
            ("struct_section", struct_dir, {**base_cfg, "use_lead_augment": False, "use_section_augment": True}),
        ("struct_both", struct_dir, {**base_cfg, "use_lead_augment": True, "use_section_augment": True}),
    ]

    title_con = open_db(struct_dir / "data.db")
    init_fts(title_con)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "num", "item_index", "question", "correct_key", "correct_text",
        "answer_title_exact_matches", "answer_title_search_matches",
        "manual_answer_article_exists", "manual_canonical_article_title",
        "manual_include_in_gold", "manual_bm25_hit", "manual_flat_hit",
        "manual_struct_hit", "manual_struct_lead_hit", "manual_struct_section_hit",
        "manual_struct_both_hit", "manual_notes",
    ] + [f"option_{k}" for k in _OPTION_KEYS]

    for label, _, _ in systems:
        fields.extend([
            f"{label}_top1_title",
            f"{label}_top5_titles",
        ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, q in enumerate(questions, start=1):
            row = {
                "num": i,
                "item_index": q.get("item_index", i),
                "question": q["question"],
                "correct_key": q["correct_key"],
                "correct_text": q.get("correct_text", ""),
                "answer_title_exact_matches": " | ".join(_exact_title_matches(title_con, q.get("correct_text", ""))),
                "answer_title_search_matches": " | ".join(_title_search_matches(title_con, q.get("correct_text", ""))),
                "manual_answer_article_exists": "",
                "manual_canonical_article_title": "",
                "manual_include_in_gold": "",
                "manual_bm25_hit": "",
                "manual_flat_hit": "",
                "manual_struct_hit": "",
                "manual_struct_lead_hit": "",
                "manual_struct_section_hit": "",
                "manual_struct_both_hit": "",
                "manual_notes": "",
            }

            for opt_key in _OPTION_KEYS:
                row[f"option_{opt_key}"] = q.get("options", {}).get(opt_key, "")

            for label, index_dir, cfg in systems:
                if index_dir is None or not index_dir.exists():
                    hits = []
                else:
                    hits = search(index_dir, q["question"], top_k=top_k, cfg=cfg)
                titles = _top_titles(hits, 5)
                row[f"{label}_top1_title"] = titles[0] if titles else ""
                row[f"{label}_top5_titles"] = _titles_pipe(hits, 5)

            writer.writerow(row)

    title_con.close()
    instructions_path = out_path.with_suffix(".instructions.txt")
    _write_instructions(instructions_path)

    return out_path, instructions_path, len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual retrieval-audit CSV")
    parser.add_argument("--dataset", required=True, help="Fixed subset dataset (recommended: normalized JSONL)")
    parser.add_argument("--structured", required=True, help="Structured index directory")
    parser.add_argument("--flat", default=None, help="Flat index directory (optional)")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-K (default 10)")
    parser.add_argument("--out", required=True, help="Output audit CSV path")
    args = parser.parse_args()

    out_path, instructions_path, n_questions = build_manual_audit(
        Path(args.dataset),
        Path(args.structured),
        Path(args.flat) if args.flat else None,
        Path(args.out),
        top_k=args.top_k,
    )

    print(f"Audit CSV     : {out_path}")
    print(f"Instructions  : {instructions_path}")
    print(f"Questions     : {n_questions}")


if __name__ == "__main__":
    main()
