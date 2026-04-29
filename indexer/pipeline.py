"""
ZIM indexing pipeline with progress callbacks and stop-event support.

Two-step pipeline (keeps peak RAM low):

  # Step 1 — extract articles → SQLite (no embedding model loaded)
  run_extract(zim_path, cfg=cfg)

  # Step 2 — embed pending chunks → FAISS (no ZIM in RAM)
  run_embed(zim_path, cfg=cfg)

  # Both steps sequentially
  run_both(zim_path, cfg=cfg)

Progress/log callbacks:
  log(msg: str)                     — one log line
  progress(phase, done, total)      — progress update
    phase: 'extract' | 'embed' | 'train'

Stop support:
  stop: threading.Event — set() to request early exit
"""

from __future__ import annotations

import gc
import re
import time
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from libzim.reader import Archive

from indexer import db, extract
from indexer import index as faiss_index

# ── Defaults (all overridable via cfg dict) ───────────────────────────────────

_DEFAULTS = {
    "embed_model":         "BAAI/bge-small-en-v1.5",
    "fastembed_cache_path": str(Path.home() / ".cache" / "fastembed"),
    "embed_batch_size":    16,
    "faiss_ivf_threshold": 500_000,
    "faiss_nprobe":        64,
    "faiss_save_every":    128,
    "priority_lead":       3,
    "priority_infobox":    10,
    "priority_prose":      12,
    "skip_namespaces":     "Category,Template,Portal,File,Help,Special,Talk,Wikipedia,User,MediaWiki,Module",
    "skip_regex":          r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:)|\(disambiguation\)",
    "min_prose_chars":     200,
    "min_infobox_rows":    3,
    "embed_dim":           384,
    "flat_chunks":         False,
}

_NO_STOP = threading.Event()  # sentinel — never set


def _cfg(d: dict | None, key: str):
    return (d or {}).get(key, _DEFAULTS[key])


def _output_dir(zim_path: Path, cfg: dict | None = None) -> Path:
    mode = (cfg or {}).get("output_dir_mode", "auto")
    flat = bool((cfg or {}).get("flat_chunks", False))
    if mode == "custom":
        custom = (cfg or {}).get("output_dir_custom", "").strip()
        if custom:
            d = Path(custom)
            d.mkdir(parents=True, exist_ok=True)
            return d
    stem = zim_path.stem + ("_flat" if flat else "")
    d = zim_path.parent / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_chunk_text(title: str, section_title: str, text: str) -> str:
    return f"Article: {title}\nSection: {section_title}\nText: {text}"


def _build_infobox_text(title: str, header: str, label: str, value: str) -> str:
    return f"Article: {title}\nInfobox: {header}\nFact: {label} = {value}"


def _build_chunk_text_flat(_title: str, _section: str, text: str) -> str:
    return text


def _build_infobox_text_flat(_title: str, _header: str, label: str, value: str) -> str:
    return f"{label} = {value}"


def _is_redirect(entry) -> bool:
    try:
        return entry.is_redirect
    except Exception:
        return False


# ── Step 1: Extract ───────────────────────────────────────────────────────────

def run_extract(
    zim_path: Path,
    cfg:      Optional[dict]          = None,
    log:      Callable                = print,
    progress: Optional[Callable]      = None,
    stop:     threading.Event         = _NO_STOP,
) -> None:
    """
    Iterate ZIM, extract article text, store in SQLite.
    Does NOT load the embedding model — keeps RAM low during extraction.
    """
    zim_path = zim_path.resolve()
    out_dir  = _output_dir(zim_path, cfg)
    db_path  = out_dir / "data.db"

    skip_ns   = set(_cfg(cfg, "skip_namespaces").split(","))
    skip_re   = re.compile(_cfg(cfg, "skip_regex"), re.I)
    min_prose = int(_cfg(cfg, "min_prose_chars"))
    min_ib    = int(_cfg(cfg, "min_infobox_rows"))
    p_lead    = int(_cfg(cfg, "priority_lead"))
    p_ib      = int(_cfg(cfg, "priority_infobox"))
    p_prose   = int(_cfg(cfg, "priority_prose"))
    flat      = bool(_cfg(cfg, "flat_chunks"))

    _prose_text   = _build_chunk_text_flat   if flat else _build_chunk_text
    _infobox_text = _build_infobox_text_flat if flat else _build_infobox_text

    log(f"\n{'='*60}")
    log(f"[extract] ZIM:    {zim_path}")
    log(f"[extract] Output: {db_path}")
    log(f"[extract] Mode:   {'flat (no metadata)' if flat else 'structured (Article/Section/Text)'}")
    log(f"{'='*60}\n")

    con     = db.open_db(db_path)
    archive = Archive(str(zim_path))
    total   = archive.entry_count
    log(f"ZIM entries: {total:,}")

    t0           = time.time()
    seen         = 0
    skipped      = 0
    already_done = 0
    new_articles = 0
    total_chunks = 0

    for i in range(total):
        if stop.is_set():
            log("[extract] Stopped by user.")
            break

        seen += 1

        if progress and seen % 5_000 == 0:
            progress("extract", seen, total)

        if seen % 50_000 == 0:
            elapsed = time.time() - t0
            log(f"  [scan] {seen:,}/{total:,} entries scanned | "
                f"new={new_articles:,} skip={skipped:,} | {seen/elapsed:.0f} entries/s")

        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            skipped += 1
            continue

        title = entry.title.strip()
        path  = str(entry.path)

        if not title or _is_redirect(entry):
            skipped += 1
            continue
        if skip_re.search(title):
            skipped += 1
            continue
        ns = path.split("/")[0] if "/" in path else ""
        if ns in skip_ns:
            skipped += 1
            continue

        if db.article_exists(con, title):
            already_done += 1
            continue

        try:
            item   = entry.get_item()
            html   = bytes(item.content).decode("utf-8", errors="replace")
            parsed = extract.extract(html)
        except Exception:
            skipped += 1
            continue

        if not parsed or not parsed["lead"]:
            skipped += 1
            continue

        prose_chars  = len(parsed["lead"]) + sum(
            len(p) for sec in parsed["sections"] for p in sec["paragraphs"]
        )
        infobox_rows = len(parsed["infobox"]["rows"]) if parsed.get("infobox") else 0
        if prose_chars < min_prose and infobox_rows < min_ib:
            skipped += 1
            continue

        url = path if path.startswith("A/") else f"A/{path}"

        lead_units = [
            {"section_title": "Lead",
             "text": _prose_text(title, "Lead", para)}
            for para in (parsed.get("lead_paragraphs") or [parsed["lead"]])
            if para and para.strip()
        ]

        infobox_chunks = []
        if parsed.get("infobox"):
            ib     = parsed["infobox"]
            header = ib["header"]
            for row in ib["rows"]:
                infobox_chunks.append({
                    "section_title": f"Infobox: {header}",
                    "text": _infobox_text(title, header, row["label"], row["value"]),
                })

        _sec_lists = [
            {"title": sec["title"], "paras": sec["paragraphs"]}
            for sec in parsed["sections"] if sec["paragraphs"]
        ]
        prose_chunks = []
        max_depth = max((len(s["paras"]) for s in _sec_lists), default=0)
        for _round in range(max_depth):
            for _sec in _sec_lists:
                if _round < len(_sec["paras"]):
                    prose_chunks.append({
                        "section_title": _sec["title"],
                        "text": _prose_text(title, _sec["title"],
                                            _sec["paras"][_round]),
                    })

        ordered = (
            lead_units[:p_lead]
            + infobox_chunks[:p_ib]
            + prose_chunks[:p_prose]
            + lead_units[p_lead:]
            + prose_chunks[p_prose:]
            + infobox_chunks[p_ib:]
        )

        all_chunks = [{**c, "chunk_index": i} for i, c in enumerate(ordered)]

        article_id = db.insert_article(con, title, url, str(zim_path))
        db.insert_chunks(con, article_id, all_chunks)
        new_articles += 1
        total_chunks += len(all_chunks)

        del html, parsed, all_chunks

        if new_articles % 500 == 0:
            con.commit()

        if new_articles % 1000 == 0:
            gc.collect()
            elapsed = time.time() - t0
            log(f"  [{seen:>8,}/{total:,}] new={new_articles:,} skip={skipped:,} "
                f"resume={already_done:,} | chunks={total_chunks:,} "
                f"| {new_articles/elapsed:.1f} art/s")

    con.commit()
    con.close()

    if progress:
        progress("extract", total, total)

    elapsed = time.time() - t0
    log(f"\n[extract] Done in {elapsed/60:.1f} min — "
        f"{new_articles:,} articles, {total_chunks:,} chunks")


# ── Step 2: Embed ─────────────────────────────────────────────────────────────

def run_embed(
    zim_path: Path,
    cfg:      Optional[dict]          = None,
    log:      Callable                = print,
    progress: Optional[Callable]      = None,
    stop:     threading.Event         = _NO_STOP,
) -> None:
    """
    Embed every pending chunk into FAISS.
    Runs as a separate step — starts with a clean memory slate after extraction.
    """
    from indexer.embed import encode

    zim_path  = zim_path.resolve()
    out_dir   = _output_dir(zim_path, cfg)
    db_path   = out_dir / "data.db"
    idx_path  = out_dir / "faiss.index"

    model_name = _cfg(cfg, "embed_model")
    cache_dir  = _cfg(cfg, "fastembed_cache_path")
    batch_size = int(_cfg(cfg, "embed_batch_size"))
    ivf_thresh = int(_cfg(cfg, "faiss_ivf_threshold"))
    nprobe     = int(_cfg(cfg, "faiss_nprobe"))
    save_every = int(_cfg(cfg, "faiss_save_every"))
    dim        = int(_cfg(cfg, "embed_dim"))

    from indexer.embed import gpu_available, gpu_info
    _gpu = gpu_info() if gpu_available() else None

    log(f"\n{'='*60}")
    log(f"[embed] DB:    {db_path}")
    log(f"[embed] FAISS: {idx_path}")
    log(f"[embed] Model: {model_name}  batch={batch_size}  dim={dim}")
    if _gpu:
        log(f"[embed] GPU:   {_gpu} detected — embedding will use GPU")
    else:
        log(f"[embed] GPU:   none detected — embedding will use CPU")
    log(f"{'='*60}\n")

    con = db.open_db(db_path)
    idx = faiss_index.load_or_create(idx_path, dim=dim, nprobe=nprobe)

    total_pending = con.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded = 0"
    ).fetchone()[0]

    if not total_pending:
        log("  No pending chunks — already fully embedded.")
        con.close()
        if progress:
            progress("embed", 0, 0)
        return

    log(f"  {total_pending:,} pending chunks to embed (batch={batch_size})...")
    t0   = time.time()
    done = 0

    # ── IVFFlat training pre-pass ─────────────────────────────────────────────
    if idx.ntotal == 0 and total_pending >= ivf_thresh:
        import math
        nlist    = max(4, min(int(math.sqrt(total_pending)), 4096))
        train_n  = min(max(nlist * 39, 5_000), total_pending)
        log(f"  Training IVFFlat — fetching {train_n:,} vectors (nlist={nlist})...")

        if progress:
            progress("train", 0, train_n)

        train_ids       = []
        train_vecs_list = []
        cur = con.execute(
            "SELECT id, text FROM chunks WHERE embedded = 0 ORDER BY id LIMIT ?",
            (train_n,),
        )
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            if stop.is_set():
                log("[embed] Stopped during training.")
                con.close()
                return
            ids   = [r[0] for r in rows]
            texts = [r[1] for r in rows]
            train_ids.extend(ids)
            train_vecs_list.append(encode(texts, model_name, cache_dir))
            if progress:
                progress("train", len(train_ids), train_n)
            del ids, texts, rows

        train_vecs = np.vstack(train_vecs_list)
        del train_vecs_list
        gc.collect()

        idx = faiss_index.make_ivf(total_pending, train_vecs, nprobe=nprobe)
        faiss_index.add_vectors(idx, train_ids, train_vecs)
        faiss_index.save(idx, idx_path)       # FAISS first
        db.mark_embedded(con, train_ids)      # then DB
        done += len(train_ids)

        del train_vecs, train_ids
        gc.collect()
        log(f"  Training done. Continuing with remaining chunks...")

    # ── Prefetch queue for overlapping DB reads with GPU encoding ────────────
    import queue
    from indexer.embed import encode_parallel, NUM_STREAMS

    prefetch_q: queue.Queue = queue.Queue(maxsize=NUM_STREAMS * 2)
    last_fetched_id = 0  # track last chunk ID put in queue

    def prefetch_worker():
        """Fetch batches from DB in background, feed to queue."""
        nonlocal last_fetched_id
        # Open separate read-only connection for thread safety
        read_con = db.open_db(db_path)
        try:
            while not stop.is_set():
                rows = read_con.execute(
                    "SELECT id, text FROM chunks WHERE embedded = 0 AND id > ? "
                    "ORDER BY id LIMIT ?",
                    (last_fetched_id, batch_size),
                ).fetchall()
                if not rows:
                    break
                prefetch_q.put(rows)
                last_fetched_id = rows[-1][0]
        finally:
            read_con.close()
            prefetch_q.put(None)  # sentinel to signal done

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    while True:
        if stop.is_set():
            log("[embed] Stopped by user.")
            break

        # Collect multiple batches for parallel GPU processing
        batches_data = []
        for _ in range(NUM_STREAMS):
            try:
                batch = prefetch_q.get(timeout=0.5)
            except queue.Empty:
                if stop.is_set():
                    break
                continue
            if batch is None:  # sentinel from prefetch thread
                break
            batches_data.append(batch)

        if not batches_data:
            break

        # Prepare data for parallel encoding
        all_chunk_ids = []
        all_texts = []
        batch_texts = []
        for batch in batches_data:
            chunk_ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]
            all_chunk_ids.extend(chunk_ids)
            batch_texts.append(texts)

        # Parallel GPU encoding across multiple CUDA streams
        vecs_list = encode_parallel(batch_texts, model_name, cache_dir)
        all_vecs = np.vstack(vecs_list)

        faiss_index.add_vectors(idx, all_chunk_ids, all_vecs)

        del batch_texts, vecs_list, all_vecs
        done += len(all_chunk_ids)

        if done % save_every == 0:
            faiss_index.save(idx, idx_path)  # FAISS to disk first
            db.mark_embedded(con, all_chunk_ids)  # then DB — never ahead of FAISS
            gc.collect()
        else:
            db.mark_embedded(con, all_chunk_ids)

        if progress:
            progress("embed", done, total_pending)

        if done % (batch_size * NUM_STREAMS * 5) == 0 or done >= total_pending:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 0
            log(f"  Embedded {done:,}/{total_pending:,} ({rate:.0f} chunks/s)")

    prefetch_thread.join(timeout=1.0)  # clean up

    faiss_index.save(idx, idx_path)

    log("  Building FTS5 title index...")
    from indexer.db import init_fts
    init_fts(con)
    log("  FTS5 ready.")

    s = db.stats(con)
    con.close()
    elapsed = time.time() - t0
    log(f"\n[embed] Done in {elapsed/60:.1f} min — "
        f"{s['embedded']:,}/{s['chunks']:,} chunks embedded | "
        f"{idx.ntotal:,} FAISS vectors")


# ── Combined ──────────────────────────────────────────────────────────────────

def run_both(
    zim_path: Path,
    cfg:      Optional[dict]          = None,
    log:      Callable                = print,
    progress: Optional[Callable]      = None,
    stop:     threading.Event         = _NO_STOP,
) -> None:
    run_extract(zim_path, cfg=cfg, log=log, progress=progress, stop=stop)
    if not stop.is_set():
        gc.collect()
        run_embed(zim_path, cfg=cfg, log=log, progress=progress, stop=stop)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _make_terminal_progress() -> Callable:
    """Returns a progress callback that renders a live progress bar to stdout."""
    import sys
    _state: dict = {"phase": None, "t0": time.time(), "last_pct": -1}
    BAR = 38

    def _fmt(seconds: float) -> str:
        s = int(seconds)
        if s < 60:   return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:   return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"

    def progress(phase: str, done: int, total: int) -> None:
        if phase != _state["phase"]:
            if _state["phase"] is not None:
                sys.stdout.write("\n")
            _state["phase"] = phase
            _state["t0"]    = time.time()
            _state["last_pct"] = -1

        pct     = int(done / total * 100) if total > 0 else 100
        elapsed = time.time() - _state["t0"]

        if pct == _state["last_pct"] and done < total:
            return
        _state["last_pct"] = pct

        filled  = int(BAR * pct / 100)
        bar     = "█" * filled + "░" * (BAR - filled)

        if done > 0 and total > done and elapsed > 0:
            eta  = elapsed / done * (total - done)
            rate = done / elapsed
            suffix = f"  ETA {_fmt(eta)}  {rate:.0f}/s  elapsed {_fmt(elapsed)}"
        else:
            suffix = f"  elapsed {_fmt(elapsed)}"

        label = f"[{phase:8}]"
        line  = f"\r{label} [{bar}] {pct:3d}%  {done:,}/{total:,}{suffix}   "
        sys.stdout.write(line)
        sys.stdout.flush()

        if done >= total:
            sys.stdout.write("\n")

    return progress


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZIM indexer")
    parser.add_argument("zim", help="Path to .zim file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract", action="store_true")
    mode.add_argument("--embed",   action="store_true")
    args = parser.parse_args()
    zim_path  = Path(args.zim)
    progress  = _make_terminal_progress()
    if args.extract:
        run_extract(zim_path, progress=progress)
    elif args.embed:
        run_embed(zim_path, progress=progress)
    else:
        run_both(zim_path, progress=progress)


if __name__ == "__main__":
    main()
