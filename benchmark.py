#!/usr/bin/env python3
"""
Latency and memory benchmarking for the paper's hardware section.

Measures, for each query in a sample:
  - Retrieval wall time (ms)
  - Peak RSS memory delta during retrieval (MB)

Reports mean, median, p95, p99 latency and mean RAM overhead.

Usage:
  python benchmark.py \\
    --index   /path/to/structured_index \\
    --dataset data/medqa_test.jsonl \\
    --n       100 \\
    --top-k   5 \\
    --out     results/benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── Sample loader (just questions, no label needed) ───────────────────────────

def _load_questions(path: Path, n: int) -> list[str]:
    suffix = path.suffix.lower()
    questions: list[str] = []

    if suffix in (".jsonl", ".json"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("question", "")
                if q:
                    questions.append(q)
                if len(questions) >= n:
                    break
    else:
        import csv as _csv
        with open(path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                r = {k.lower(): v for k, v in row.items()}
                q = r.get("question", "")
                if q:
                    questions.append(q)
                if len(questions) >= n:
                    break

    return questions


# ── Benchmark loop ────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    import os
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def run_benchmark(
    index_dir: Path,
    questions: list[str],
    top_k: int,
    cfg: dict,
    label: str,
) -> list[dict]:
    from indexer.query import search

    results = []
    n = len(questions)

    for i, q in enumerate(questions):
        gc.collect()
        rss_before = _rss_mb()
        t0 = time.perf_counter()
        try:
            hits = search(index_dir, q, top_k=top_k, cfg=cfg)
        except Exception as e:
            print(f"\n  [warn] q{i+1}: {e}")
            hits = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rss_after  = _rss_mb()

        results.append({
            "idx":        i,
            "question":   q[:80],
            "n_hits":     len(hits),
            "latency_ms": round(elapsed_ms, 2),
            "ram_delta_mb": round(max(0.0, rss_after - rss_before), 2),
        })
        print(f"\r  [{label}] {i+1}/{n}  {elapsed_ms:.0f}ms   ", end="", flush=True)

    print()
    return results


# ── Stats ─────────────────────────────────────────────────────────────────────

def summarise(results: list[dict], label: str) -> dict:
    lats = [r["latency_ms"] for r in results]
    rams = [r["ram_delta_mb"] for r in results]
    lats_s = sorted(lats)
    n = len(lats_s)

    def pct(p):
        idx = min(int(n * p / 100), n - 1)
        return lats_s[idx]

    return {
        "label":       label,
        "n":           n,
        "mean_ms":     round(statistics.mean(lats), 1),
        "median_ms":   round(statistics.median(lats), 1),
        "p95_ms":      round(pct(95), 1),
        "p99_ms":      round(pct(99), 1),
        "min_ms":      round(min(lats), 1),
        "max_ms":      round(max(lats), 1),
        "mean_ram_mb": round(statistics.mean(rams), 2) if _HAS_PSUTIL else "n/a",
    }


def print_table(summaries: list[dict]) -> None:
    hdr = (
        f"{'System':<30} {'Mean':>8} {'Median':>8} "
        f"{'P95':>8} {'P99':>8} {'Max':>8} {'RAM MB':>8}"
    )
    print("\n" + "─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))
    for s in summaries:
        print(
            f"{s['label']:<30} "
            f"{s['mean_ms']:>7.1f}ms "
            f"{s['median_ms']:>7.1f}ms "
            f"{s['p95_ms']:>7.1f}ms "
            f"{s['p99_ms']:>7.1f}ms "
            f"{s['max_ms']:>7.1f}ms "
            f"{str(s['mean_ram_mb']):>7}MB"
        )
    print("─" * len(hdr))


def export_csv(out_path: Path, summaries: list[dict],
               all_results: list[tuple[str, list[dict]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Summary
    agg = out_path.with_stem(out_path.stem + "_summary")
    with open(agg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"  Summary → {agg}")

    # Per-query
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fields = ["idx", "question"] + \
                 [f"{lbl}_ms"  for lbl, _ in all_results] + \
                 [f"{lbl}_ram" for lbl, _ in all_results]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        nq = len(all_results[0][1])
        for i in range(nq):
            row: dict = {
                "idx":      i,
                "question": all_results[0][1][i]["question"],
            }
            for lbl, res in all_results:
                row[f"{lbl}_ms"]  = res[i]["latency_ms"]
                row[f"{lbl}_ram"] = res[i]["ram_delta_mb"]
            w.writerow(row)
    print(f"  Per-query → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrieval latency + RAM benchmark")
    parser.add_argument("--index",   required=True, help="Structured index directory")
    parser.add_argument("--flat",    default=None,  help="Flat index directory (optional)")
    parser.add_argument("--dataset", required=True, help="Dataset path for sample questions")
    parser.add_argument("--n",       type=int, default=100, help="Number of queries (default 100)")
    parser.add_argument("--top-k",   type=int, default=5,   help="Retrieval top-K (default 5)")
    parser.add_argument("--out",     default="results/benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    if not _HAS_PSUTIL:
        print("  [warn] psutil not installed — RAM delta will be 0. pip install psutil")

    print(f"\nDataset : {args.dataset}")
    print(f"N       : {args.n}   Top-K: {args.top_k}")

    questions = _load_questions(Path(args.dataset), args.n)
    print(f"  {len(questions)} questions loaded")

    base_cfg: dict = {
        "use_faiss":           True,
        "use_title_bm25":      True,
        "use_para_bm25":       True,
        "use_diversity_cap":   True,
        "use_mention_penalty": True,
        "use_nav_boost":       True,
        "use_lead_augment":    False,
        "eval_rrf_k":          60,
        "eval_diversity_max":  6,
    }

    all_results:  list[tuple[str, list[dict]]] = []
    all_summaries: list[dict] = []

    index_dir = Path(args.index)
    flat_dir  = Path(args.flat) if args.flat else None

    systems = [("Hybrid Structured", index_dir, base_cfg)]
    if flat_dir and flat_dir.exists():
        flat_cfg = {**base_cfg}
        systems.append(("Hybrid Flat", flat_dir, flat_cfg))

    for label, idir, cfg in systems:
        if not idir.exists():
            print(f"\n  [skip] {label} — index not found: {idir}")
            continue
        print(f"\nRunning: {label}")
        res = run_benchmark(idir, questions, args.top_k, cfg, label)
        all_results.append((label, res))
        all_summaries.append(summarise(res, label))

    if all_summaries:
        print_table(all_summaries)
        export_csv(Path(args.out), all_summaries, all_results)


if __name__ == "__main__":
    main()
