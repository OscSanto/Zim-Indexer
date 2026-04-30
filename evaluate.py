#!/usr/bin/env python3
"""
Retrieval evaluation for the ZIM indexer paper.

Compares three systems against medical QA datasets:
  1. BM25 Only        — sparse keyword retrieval, no FAISS
  2. Hybrid Flat      — BM25 + FAISS, flat index (no Article/Section metadata)
  3. Hybrid Structured— BM25 + FAISS, structured index + lead augmentation

Metrics: Hit@1, Hit@3, Hit@5, Hit@10, MRR@10

Dataset formats supported:
  - MedQA  JSONL : {"question":..., "options":{"A":...,"B":...,...}, "answer_idx":"A"}
  - MedMCQA JSONL: {"question":..., "opa":..., "opb":..., "opc":..., "opd":..., "cop":0}
  - GUI CSV      : question, option_a, option_b, option_c, option_d, correct

Usage:
  python evaluate.py \\
    --dataset data/medqa_test.jsonl \\
    --structured /path/to/structured_index \\
    --flat       /path/to/flat_index \\
    --n 200 --top-k 10 \\
    --out results/medqa_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _load_medqa_jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            opts = obj.get("options", {})
            key = (obj.get("answer_idx") or obj.get("answer", "")).strip().upper()
            # Some MedQA variants store full answer text in answer_idx, not the letter
            if len(key) > 1:
                for k, v in opts.items():
                    if v.strip().upper() == key:
                        key = k.strip().upper()
                        break
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  key,
                "correct_text": opts.get(key, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_medmcqa_jsonl(path: Path, n: int) -> list[dict]:
    key_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    opt_map  = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj  = json.loads(line)
            opts = {k: obj.get(v, "") for k, v in opt_map.items()}
            cop  = key_map.get(int(obj.get("cop", 0)), "A")
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  cop,
                "correct_text": opts.get(cop, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_gui_csv(path: Path, n: int) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {k.lower(): v for k, v in row.items()}
            letter = r.get("correct", "A").strip().upper()
            opts   = {"A": r.get("option_a",""), "B": r.get("option_b",""),
                      "C": r.get("option_c",""), "D": r.get("option_d","")}
            rows.append({
                "question":     r.get("question", ""),
                "options":      opts,
                "correct_key":  letter,
                "correct_text": opts.get(letter, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_mmlu_pro_jsonl(path: Path, n: int) -> list[dict]:
    """MMLU-Pro: options is a list of strings, answer is a letter A-J."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            opts_list = obj.get("options", [])
            opts = {chr(65 + i): v for i, v in enumerate(opts_list)}
            key = obj.get("answer", "A").strip().upper()
            rows.append({
                "question":     obj["question"],
                "options":      opts,
                "correct_key":  key,
                "correct_text": opts.get(key, ""),
            })
            if len(rows) >= n:
                break
    return rows


def _load_pubmedqa_jsonl(path: Path, n: int) -> list[dict]:
    """PubMedQA: final_decision is yes/no/maybe → mapped to A/B/C."""
    _decision_map = {"yes": "A", "no": "B", "maybe": "C"}
    _opts = {"A": "yes", "B": "no", "C": "maybe"}
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            decision = obj.get("final_decision", "").lower().strip()
            key = _decision_map.get(decision, "A")
            rows.append({
                "question":     obj["question"],
                "options":      _opts,
                "correct_key":  key,
                "correct_text": _opts[key],
            })
            if len(rows) >= n:
                break
    return rows


def load_dataset(path: Path, n: int) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".json"):
        # Peek first line to detect format
        with open(path, encoding="utf-8") as f:
            first = json.loads(f.readline())
        if "opa" in first:
            print(f"  Detected MedMCQA format")
            return _load_medmcqa_jsonl(path, n)
        if "final_decision" in first:
            print(f"  Detected PubMedQA format")
            return _load_pubmedqa_jsonl(path, n)
        if isinstance(first.get("options"), list):
            print(f"  Detected MMLU-Pro format")
            return _load_mmlu_pro_jsonl(path, n)
        print(f"  Detected MedQA format")
        return _load_medqa_jsonl(path, n)
    print(f"  Detected GUI CSV format")
    return _load_gui_csv(path, n)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _hit_in_chunks(correct_text: str, hits: list[dict]) -> int | None:
    """
    Return 1-based rank of first hit whose article title shares a content word
    with the correct answer text.

    MedQA answer options are named entities (drugs, diseases, mechanisms) that
    appear as Wikipedia article titles — title-word overlap is a reliable signal.
    Verbatim body-text search returns 0 because clinical answer phrases rarely
    appear word-for-word in encyclopedic prose.
    """
    import re as _re
    ans_words = set(_re.findall(r'\b\w{4,}\b', correct_text.lower()))
    if not ans_words:
        return None
    for rank, hit in enumerate(hits, 1):
        title_words = set(_re.findall(r'\b\w{4,}\b', (hit.get("title") or "").lower()))
        if ans_words & title_words:
            return rank
    return None


def evaluate_system(
    index_dir: Path,
    questions: list[dict],
    top_k: int,
    cfg: dict,
    label: str,
) -> list[dict]:
    from indexer.query import search
    from indexer.embed import encode
    from indexer.db import open_db, init_fts

    n = len(questions)

    # Batch embed all questions upfront — only when FAISS is actually used
    model_name = cfg.get("embed_model", "BAAI/bge-small-en-v1.5")
    cache_dir  = cfg.get("fastembed_cache_path", None)
    if cfg.get("use_faiss", True):
        print(f"  [{label}]  embedding {n} queries…", end="", flush=True)
        query_vecs = encode([q["question"] for q in questions], model_name, cache_dir)
        print(f" done", flush=True)
    else:
        print(f"  [{label}]  BM25-only — skipping embedding")
        query_vecs = [None] * n

    # Reuse one SQLite connection across all queries
    db_path = index_dir / "data.db"
    con = open_db(db_path)
    init_fts(con)

    results = []
    t0 = time.time()

    for i, q in enumerate(questions):
        sys.stdout.write(f"\r  [{label}]  {i+1}/{n}  ({(i+1)/max(time.time()-t0,0.1):.1f} q/s)   ")
        sys.stdout.flush()

        try:
            hits = search(index_dir, q["question"], top_k=top_k, cfg=cfg,
                          query_vec=query_vecs[i], con=con)
        except Exception as e:
            hits = []
            print(f"\n  [warn] q{i+1} error: {e}")

        rank = _hit_in_chunks(q["correct_text"], hits)
        results.append({
            "idx":          i,
            "question":     q["question"],
            "correct_key":  q["correct_key"],
            "correct_text": q["correct_text"],
            "rank":         rank,
            "hits":         hits,
        })

    con.close()
    elapsed = time.time() - t0
    q_per_s = n / elapsed if elapsed > 0 else 0.0
    print(f"\r  [{label}]  {n}/{n} done in {elapsed:.1f}s ({q_per_s:.2f} q/s)      ")
    for r in results:
        r["elapsed_s"] = round(elapsed, 2)
        r["q_per_s"]   = round(q_per_s, 3)
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], top_k: int = 10) -> dict:
    n     = len(results)
    ranks = [r["rank"] for r in results]

    def hit_at(k):
        c = sum(1 for rk in ranks if rk is not None and rk <= k)
        return c / n if n else 0.0

    mrr = sum(1.0 / rk for rk in ranks if rk is not None and rk <= 10) / n if n else 0.0

    return {
        "n":      n,
        "hit@1":  hit_at(1),
        "hit@3":  hit_at(3),
        "hit@5":  hit_at(5),
        "hit@10": hit_at(10),
        "mrr@10": mrr,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_table(systems: list[tuple[str, dict]]) -> None:
    header = f"{'System':<28} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR@10':>7}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for label, m in systems:
        print(
            f"{label:<28} "
            f"{m['hit@1']:>7.3f} "
            f"{m['hit@3']:>7.3f} "
            f"{m['hit@5']:>7.3f} "
            f"{m['hit@10']:>7.3f} "
            f"{m['mrr@10']:>7.4f}"
        )
    print("─" * len(header))


def export_csv(out_path: Path, systems: list[tuple[str, dict]],
               all_results: list[tuple[str, list[dict]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics sheet
    agg_path = out_path.with_stem(out_path.stem + "_metrics")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "elapsed_s", "q_per_s"])
        for label, m in systems:
            w.writerow([label, m["n"],
                        f"{m['hit@1']:.4f}", f"{m['hit@3']:.4f}",
                        f"{m['hit@5']:.4f}", f"{m['hit@10']:.4f}",
                        f"{m['mrr@10']:.4f}",
                        f"{m.get('elapsed_s', 0):.2f}",
                        f"{m.get('q_per_s', 0):.3f}"])
    print(f"  Metrics → {agg_path}")

    # Per-question sheet
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["num", "question", "correct_key", "correct_text"] + \
                     [f"{label}_rank" for label, _ in all_results] + \
                     [f"{label}_hit@10" for label, _ in all_results]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n = len(all_results[0][1])
        for i in range(n):
            row: dict = {
                "num":          i + 1,
                "question":     all_results[0][1][i]["question"][:120],
                "correct_key":  all_results[0][1][i]["correct_key"],
                "correct_text": all_results[0][1][i]["correct_text"],
            }
            for label, results in all_results:
                rk = results[i]["rank"]
                row[f"{label}_rank"]    = rk if rk is not None else ""
                row[f"{label}_hit@10"]  = 1 if rk is not None and rk <= 10 else 0
            w.writerow(row)
    print(f"  Per-question → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ZIM retrieval evaluation — structured vs flat ablation"
    )
    parser.add_argument("--dataset",    required=True,  help="Path to dataset (JSONL or CSV)")
    parser.add_argument("--structured", required=True,  help="Structured index directory")
    parser.add_argument("--flat",       default=None,   help="Flat index directory (optional)")
    parser.add_argument("--n",          type=int, default=200,  help="Max questions (default 200)")
    parser.add_argument("--top-k",      type=int, default=10,   help="Retrieval top-K (default 10)")
    parser.add_argument("--out",        default="results/eval.csv", help="Output CSV path")
    parser.add_argument("--no-faiss",   action="store_true", help="Disable FAISS (BM25 only mode)")
    parser.add_argument("--lead-augment", action="store_true", help="Enable lead augmentation")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    struct_dir   = Path(args.structured)
    flat_dir     = Path(args.flat) if args.flat else None
    out_path     = Path(args.out)
    top_k        = args.top_k

    print(f"\nDataset : {dataset_path}")
    print(f"N       : {args.n}")
    print(f"Top-K   : {top_k}")

    print(f"\nLoading dataset...")
    questions = load_dataset(dataset_path, args.n)
    print(f"  {len(questions)} questions loaded")

    systems_to_run: list[tuple[str, Path, dict]] = []

    base_cfg: dict = {
        "use_faiss":        True,
        "use_title_bm25":   True,
        "use_para_bm25":    True,
        "use_diversity_cap":   False,  # hand-tuned, disabled for reproducibility
        "use_mention_penalty": False,  # hand-tuned, disabled for reproducibility
        "use_nav_boost":       False,  # hand-tuned, disabled for reproducibility
        "use_lead_augment": False,
        "eval_rrf_k":       60,
        "eval_diversity_max": 6,
        "embed_model":      "BAAI/bge-small-en-v1.5",
        "faiss_mmap":       True,
    }

    # System 1: BM25 only on structured index
    bm25_cfg = {**base_cfg, "use_faiss": False, "use_lead_augment": False}
    systems_to_run.append(("BM25 Only", struct_dir, bm25_cfg))

    # System 2: Hybrid on flat index (if provided)
    if flat_dir:
        flat_cfg = {**base_cfg, "use_lead_augment": False}
        systems_to_run.append(("Hybrid Flat", flat_dir, flat_cfg))

    # System 3: Hybrid structured (no lead augment)
    struct_cfg = {**base_cfg, "use_lead_augment": False}
    systems_to_run.append(("Hybrid Structured", struct_dir, struct_cfg))

    # System 4: Hybrid structured + lead augment
    lead_cfg = {**base_cfg, "use_lead_augment": True}
    systems_to_run.append(("Hybrid Struct + Lead", struct_dir, lead_cfg))

    all_results: list[tuple[str, list[dict]]] = []
    agg_metrics: list[tuple[str, dict]] = []

    ckpt_path = out_path.with_stem(out_path.stem + "_checkpoint")

    # Load completed systems from a previous crashed run
    completed: dict[str, tuple[list[dict], dict]] = {}
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                completed[entry["label"]] = (entry["results"], entry["metrics"])
        print(f"  Resuming — {len(completed)} system(s) already done: {list(completed)}")

    for label, index_dir, cfg in systems_to_run:
        if label in completed:
            print(f"\n  [resume] {label} — loaded from checkpoint")
            results, metrics = completed[label]
            all_results.append((label, results))
            agg_metrics.append((label, metrics))
            continue
        if not index_dir.exists():
            print(f"\n  [skip] {label} — index not found: {index_dir}")
            continue
        print(f"\nRunning: {label}")
        results  = evaluate_system(index_dir, questions, top_k, cfg, label)
        metrics  = compute_metrics(results, top_k)
        metrics["elapsed_s"] = results[0]["elapsed_s"] if results else 0.0
        metrics["q_per_s"]   = results[0]["q_per_s"]   if results else 0.0
        all_results.append((label, results))
        agg_metrics.append((label, metrics))
        # Save checkpoint after each system completes
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"label": label, "results": results, "metrics": metrics},
                               default=str) + "\n")

    print_table(agg_metrics)
    export_csv(out_path, agg_metrics, all_results)
    ckpt_path.unlink(missing_ok=True)  # clean up checkpoint on successful completion


if __name__ == "__main__":
    main()
