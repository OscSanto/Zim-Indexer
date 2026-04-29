#!/usr/bin/env python3
"""
Download evaluation datasets from HuggingFace and save as JSONL files
compatible with evaluate.py and infer.py.

Datasets:
  medqa       — MedQA USMLE 4-option (GBaker/MedQA-USMLE-4-options)
  medmcqa     — MedMCQA validation split (medmcqa/medmcqa)
  mmlu_pro    — MMLU-Pro health + medicine categories (TIGER-Lab/MMLU-Pro)
  pubmedqa    — PubMedQA labeled (qiaojin/PubMedQA, pqa_labeled)

Output files (in ./data/):
  data/medqa_test.jsonl
  data/medmcqa_val.jsonl
  data/mmlu_pro_med.jsonl
  data/pubmedqa.jsonl

Requires:
  pip install datasets huggingface_hub

Usage:
  python download_data.py
  python download_data.py --out data/ --datasets medqa mmlu_pro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


# ── MedQA ─────────────────────────────────────────────────────────────────────

def download_medqa(out_dir: Path) -> None:
    """
    GBaker/MedQA-USMLE-4-options
    Schema: {question, options: {A,B,C,D}, answer}
    answer field is the letter key ("A"/"B"/"C"/"D").
    """
    from datasets import load_dataset  # type: ignore
    print("\nDownloading MedQA USMLE...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    records = []
    for row in ds:
        opts = row.get("options", {})
        if not isinstance(opts, dict):
            continue
        answer = str(row.get("answer", "")).strip().upper()
        records.append({
            "question":   row["question"],
            "options":    opts,
            "answer_idx": answer,
        })
    _save_jsonl(records, out_dir / "medqa_test.jsonl")


# ── MedMCQA ───────────────────────────────────────────────────────────────────

def download_medmcqa(out_dir: Path) -> None:
    """
    medmcqa/medmcqa
    Schema: {question, opa, opb, opc, opd, cop: int 0-3}
    Test split has no ground truth; use validation split instead.
    """
    from datasets import load_dataset  # type: ignore
    print("\nDownloading MedMCQA...")
    ds = load_dataset("medmcqa", split="validation")
    records = []
    for row in ds:
        records.append({
            "question": row["question"],
            "opa":      row.get("opa", ""),
            "opb":      row.get("opb", ""),
            "opc":      row.get("opc", ""),
            "opd":      row.get("opd", ""),
            "cop":      int(row.get("cop", 0)),
        })
    _save_jsonl(records, out_dir / "medmcqa_val.jsonl")


# ── MMLU-Pro ──────────────────────────────────────────────────────────────────

_MMLU_MED_CATEGORIES = {
    "health",
    "medicine",
    "biology",
    "chemistry",
    "psychology",
}


def download_mmlu_pro(out_dir: Path) -> None:
    """
    TIGER-Lab/MMLU-Pro
    Schema: {question, options: list[str], answer: letter, category: str}
    Filter to medical/health-adjacent categories.
    """
    from datasets import load_dataset  # type: ignore
    print("\nDownloading MMLU-Pro (medical categories)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    records = []
    for row in ds:
        cat = (row.get("category") or "").lower()
        if cat not in _MMLU_MED_CATEGORIES:
            continue
        opts_list = row.get("options", [])
        records.append({
            "question": row["question"],
            "options":  opts_list,          # list — our loader maps A-J
            "answer":   str(row.get("answer", "A")).strip().upper(),
            "category": cat,
        })
    _save_jsonl(records, out_dir / "mmlu_pro_med.jsonl")


# ── PubMedQA ──────────────────────────────────────────────────────────────────

def download_pubmedqa(out_dir: Path) -> None:
    """
    qiaojin/PubMedQA, config pqa_labeled
    Schema: {pubid, question, context, long_answer, final_decision: yes/no/maybe}
    """
    from datasets import load_dataset  # type: ignore
    print("\nDownloading PubMedQA...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records = []
    for row in ds:
        records.append({
            "question":       row["question"],
            "final_decision": str(row.get("final_decision", "yes")).lower(),
            "pubid":          row.get("pubid", ""),
        })
    _save_jsonl(records, out_dir / "pubmedqa.jsonl")


# ── Main ──────────────────────────────────────────────────────────────────────

_ALL = ["medqa", "medmcqa", "mmlu_pro", "pubmedqa"]

_DOWNLOADERS = {
    "medqa":    download_medqa,
    "medmcqa":  download_medmcqa,
    "mmlu_pro": download_mmlu_pro,
    "pubmedqa": download_pubmedqa,
}


def main():
    parser = argparse.ArgumentParser(description="Download evaluation datasets from HuggingFace")
    parser.add_argument("--out",      default="data",    help="Output directory (default: data/)")
    parser.add_argument("--datasets", nargs="+",
                        choices=_ALL, default=_ALL,
                        help=f"Which datasets to download (default: all). Choices: {_ALL}")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print(f"Output dir: {out_dir.resolve()}")

    for name in args.datasets:
        try:
            _DOWNLOADERS[name](out_dir)
        except Exception as e:
            print(f"\n  [error] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone. Files in data/:")
    for p in sorted(out_dir.glob("*.jsonl")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<30} {size_mb:.1f} MB")

    print("\nUsage examples:")
    print("  python evaluate.py --dataset data/medqa_test.jsonl --structured /path/to/index --n 200")
    print("  python evaluate.py --dataset data/mmlu_pro_med.jsonl --structured /path/to/index --n 200")
    print("  python infer.py --dataset data/medqa_test.jsonl --index /path/to/index --host 192.168.1.x --models mistral:7b biomistral")


if __name__ == "__main__":
    main()
