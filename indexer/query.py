"""
Hybrid retrieval with individually toggleable signals for ablation studies.

Signals:
  use_faiss           — dense FAISS cosine similarity
  use_title_bm25      — BM25 over article titles
  use_para_bm25       — BM25 over paragraph text
  use_diversity_cap   — per-article chunk cap before final ranking
  use_mention_penalty — penalise chunks where query terms appear incidentally
  use_nav_boost       — boost articles whose title closely matches the query
  use_lead_augment    — prepend article lead to non-lead hits (query-time)
  use_section_augment — prepend section's first paragraph to non-first hits (query-time)
"""
from __future__ import annotations

import logging
import re
import argparse
from pathlib import Path

import faiss as _faiss
from indexer import index as faiss_index
from indexer.db import (
    open_db, stats, init_fts, title_search,
    get_chunks_for_article, get_chunk_by_id, mark_embedded,
    chunk_text_search, get_article_lead, get_section_first_para,
)

logger = logging.getLogger(__name__)

_TOK_RE = re.compile(r"\b\w+\b")


def _g(cfg, key, default):
    return (cfg or {}).get(key, default)


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _mention_strength(query: str, chunk_text: str) -> float:
    """
    Ported from AIIAB zim_retrieval.py.
    Returns a multiplier in [0.6, 1.0] that penalises chunks where query
    terms appear only once or only in the second half of the text.
    Terms absent from the chunk are not penalised (pure FAISS hit).
    """
    terms = set(_TOK_RE.findall(query.lower()))
    if not terms:
        return 1.0
    text_lower   = chunk_text.lower()
    early_cutoff = max(1, len(text_lower) // 2)
    multipliers: list[float] = []
    for term in terms:
        count = text_lower.count(term)
        if count == 0:
            continue
        m = 1.0
        if count == 1:
            m *= 0.82
        if term not in text_lower[:early_cutoff]:
            m *= 0.88
        multipliers.append(m)
    return sum(multipliers) / len(multipliers) if multipliers else 1.0


def _diversity_cap(ranked: list[tuple[int, float]],
                   chunk_article: dict[int, int],
                   max_per: int) -> list[tuple[int, float]]:
    """
    Ported from AIIAB zim_retrieval.py.
    Allow at most max_per chunks per article in the primary list;
    overflow chunks are appended after so nothing is lost.
    """
    seen: dict[int, int] = {}
    primary, overflow = [], []
    for cid, score in ranked:
        aid = chunk_article.get(cid, -1)
        if seen.get(aid, 0) < max_per:
            seen[aid] = seen.get(aid, 0) + 1
            primary.append((cid, score))
        else:
            overflow.append((cid, score))
    return primary + overflow


def _nav_boost(hits: list[dict], query: str,
               boost: float = 0.50, lead_mult: float = 1.15) -> list[dict]:
    """
    Simplified navigational boost (ported from AIIAB zim_retrieval.py).
    Boosts chunks from articles whose title has ≥60% word overlap with the query.
    Lead sections receive an additional multiplicative boost.
    """
    q_words = set(_TOK_RE.findall(query.lower()))
    if not q_words:
        return hits
    for c in hits:
        t_words = set(_TOK_RE.findall((c.get("title") or "").lower()))
        if not t_words:
            continue
        overlap = len(q_words & t_words) / len(t_words)
        if overlap >= 0.6:
            c["rrf_score"] = min(1.0, c["rrf_score"] + boost * overlap)
            if (c.get("section_title") or "").lower() == "lead":
                c["rrf_score"] = min(1.0, c["rrf_score"] * lead_mult)
    return hits


# ── Main search function ──────────────────────────────────────────────────────

def search(out_dir: Path, query: str, top_k: int = 10,
           threshold: float = 0.0, cfg: dict | None = None) -> list[dict]:
    """
    Hybrid retrieval with toggleable signals.

    cfg keys that control retrieval behaviour:
      use_faiss          (bool, default True)
      use_title_bm25     (bool, default True)
      use_para_bm25      (bool, default True)
      use_diversity_cap  (bool, default True)
      use_mention_penalty(bool, default True)
      use_nav_boost      (bool, default True)
      eval_rrf_k         (int,  default 60)
      eval_diversity_max (int,  default 3)
    """
    from indexer.embed import encode

    model_name = _g(cfg, "embed_model",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    cache_dir  = _g(cfg, "fastembed_cache_path", None)
    dim        = int(_g(cfg, "embed_dim", 384))
    nprobe     = int(_g(cfg, "faiss_nprobe", 64))

    use_faiss       = _g(cfg, "use_faiss",            True)
    use_title       = _g(cfg, "use_title_bm25",       True)
    use_para        = _g(cfg, "use_para_bm25",         True)
    use_div         = _g(cfg, "use_diversity_cap",     True)
    use_mention     = _g(cfg, "use_mention_penalty",   True)
    use_nav         = _g(cfg, "use_nav_boost",         True)
    use_lead_aug    = _g(cfg, "use_lead_augment",      False)
    use_section_aug = _g(cfg, "use_section_augment",   False)
    rrf_k           = int(_g(cfg, "eval_rrf_k",        60))
    div_max         = int(_g(cfg, "eval_diversity_max", 6))

    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"
    if not db_path.exists():
        raise FileNotFoundError(f"No index found at {out_dir}")

    con = open_db(db_path)
    init_fts(con)

    fetch_n = top_k * 6

    faiss_rank_map:  dict[int, int]   = {}
    faiss_score_map: dict[int, float] = {}
    title_rank_map:  dict[int, int]   = {}
    para_rank_map:   dict[int, int]   = {}

    use_mmap = _g(cfg, "faiss_mmap", False)
    if use_faiss and idx_path.exists():
        idx = faiss_index.load_or_create(idx_path, dim=dim, nprobe=nprobe, mmap=use_mmap)
        if idx.ntotal > 0:
            qvec    = encode([query], model_name, cache_dir)[0]
            results = faiss_index.search(idx, qvec, top_k=fetch_n)
            faiss_rank_map  = {cid: rank for rank, (cid, _) in enumerate(results)}
            faiss_score_map = {cid: sc   for cid, sc in results}

    if use_title:
        for article_id, article_rank in title_search(con, query, limit=top_k * 3):
            for chunk_id, _ in get_chunks_for_article(con, article_id):
                if chunk_id not in title_rank_map:
                    title_rank_map[chunk_id] = article_rank

    if use_para:
        for rank, (cid, _) in enumerate(chunk_text_search(con, query, limit=fetch_n)):
            para_rank_map[cid] = rank

    all_ids = set(faiss_rank_map) | set(title_rank_map) | set(para_rank_map)
    if not all_ids:
        con.close()
        return []

    rrf_scores = {
        cid: (
            (1.0 / (rrf_k + faiss_rank_map[cid]) if cid in faiss_rank_map else 0.0) +
            (1.0 / (rrf_k + title_rank_map[cid]) if cid in title_rank_map else 0.0) +
            (1.0 / (rrf_k + para_rank_map[cid])  if cid in para_rank_map  else 0.0)
        )
        for cid in all_ids
    }

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if use_div and div_max > 0:
        cid_list = [cid for cid, _ in ranked]
        ph   = ",".join("?" * len(cid_list))
        rows = con.execute(
            f"SELECT id, article_id FROM chunks WHERE id IN ({ph})", cid_list
        ).fetchall()
        chunk_article = {r[0]: r[1] for r in rows}
        ranked = _diversity_cap(ranked, chunk_article, div_max)

    hits:  list[dict]       = []
    seen:  set[int]         = set()

    for cid, rrf in ranked:
        if len(hits) >= top_k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        chunk = get_chunk_by_id(con, cid)
        if not chunk:
            continue
        fscore = faiss_score_map.get(cid, 0.0)
        if use_faiss and fscore < threshold:
            continue

        adj = rrf * _mention_strength(query, chunk["text"]) if use_mention else rrf
        chunk.update({
            "rrf_score":     round(adj, 6),
            "faiss_score":   round(fscore, 4),
            "in_faiss":      cid in faiss_rank_map,
            "in_title_bm25": cid in title_rank_map,
            "in_para_bm25":  cid in para_rank_map,
        })
        hits.append(chunk)

    if use_nav and hits:
        hits = _nav_boost(hits, query)
        hits.sort(key=lambda c: c["rrf_score"], reverse=True)

    if use_lead_aug:
        for hit in hits:
            if hit.get("section_title") != "Lead":
                lead = get_article_lead(con, hit["article_id"])
                if lead:
                    hit["lead_context"] = lead

    if use_section_aug:
        for hit in hits:
            sec = hit.get("section_title", "")
            # Only augment non-lead, non-infobox sections
            if sec in ("Lead", "") or sec.startswith("Infobox"):
                continue
            first = get_section_first_para(con, hit["article_id"], sec)
            # Don't duplicate: skip if this hit IS the first paragraph
            if first and first != hit.get("text"):
                hit["section_context"] = first

    con.close()

    if logger.isEnabledFor(logging.DEBUG):
        for i, hit in enumerate(hits[:top_k], 1):
            parts = [f"[{i}] rrf={hit['rrf_score']:.5f}  {hit['title']!r}  §{hit['section_title']!r}"]
            if hit.get("lead_context"):
                parts.append(f"  lead_context: {hit['lead_context'][:120]!r}…")
            if hit.get("section_context"):
                parts.append(f"  section_context: {hit['section_context'][:120]!r}…")
            parts.append(f"  text: {hit['text'][:120]!r}…")
            logger.debug("\n".join(parts))

    return hits[:top_k]


# ── Index status ──────────────────────────────────────────────────────────────

def status(out_dir: Path) -> dict:
    db_path  = out_dir / "data.db"
    idx_path = out_dir / "faiss.index"
    if not db_path.exists():
        return {}

    db_size  = db_path.stat().st_size  if db_path.exists()  else 0
    idx_size = idx_path.stat().st_size if idx_path.exists() else 0

    con = open_db(db_path)
    s   = stats(con)
    con.close()

    result = {
        "db_size":  db_size,
        "idx_size": idx_size,
        "articles": s.get("articles", 0),
        "chunks":   s.get("chunks",   0),
        "embedded": s.get("embedded", 0),
    }
    if idx_path.exists() and idx_size > 100:
        try:
            idx   = faiss_index.load_or_create(idx_path)
            inner = _faiss.downcast_index(idx.index)
            result["faiss_vectors"] = idx.ntotal
            result["faiss_type"]    = type(inner).__name__
            result["faiss_dim"]     = inner.d
        except Exception:
            pass
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Query a ZIM FAISS index")
    parser.add_argument("out_dir")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top",       type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--status",    action="store_true")
    args    = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.status:
        for k, v in status(out_dir).items():
            print(f"  {k}: {v}")
    elif args.query:
        hits = search(out_dir, args.query, top_k=args.top, threshold=args.threshold)
        for i, h in enumerate(hits, 1):
            print(f"\n[{i}] rrf={h['rrf_score']:.5f}  {h['title']!r}  §{h['section_title']!r}")
            if args.verbose:
                print(f"    {h['text'][:300]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
