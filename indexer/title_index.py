"""
Title index builder — embeds one vector per article title into SQLite.
Used for fast semantic title search independent of the chunk FAISS index.
"""
from __future__ import annotations

import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

_SKIP_RE = re.compile(
    r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:|"
    r"Wikipedia:|User:|MediaWiki:|Module:)",
    re.I,
)
_BATCH = 256


def _title_db_path(zim_path: Path) -> Path:
    return zim_path.parent / (zim_path.stem + "_titles.db")


def open_title_db(zim_path: Path) -> sqlite3.Connection:
    db_path = _title_db_path(zim_path)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS titles (
            id    INTEGER PRIMARY KEY,
            path  TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            vec   BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_titles_path ON titles(path);
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
    """)
    con.commit()
    return con


def title_db_count(zim_path: Path) -> int:
    db = _title_db_path(zim_path)
    if not db.exists():
        return 0
    try:
        con   = sqlite3.connect(str(db))
        count = con.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        con.close()
        return count
    except Exception:
        return 0


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def load_all_vecs(con: sqlite3.Connection) -> tuple[np.ndarray, list[str], list[str]]:
    rows = con.execute("SELECT path, title, vec FROM titles").fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32), [], []
    paths  = [r[0] for r in rows]
    titles = [r[1] for r in rows]
    vecs   = np.stack([_blob_to_vec(r[2]) for r in rows])
    return vecs, paths, titles


def build(zim_path: Path, cfg: dict | None = None,
          progress_cb=None, log=print) -> int:
    """
    Build (or resume) the title index.
    progress_cb(done, total) is called periodically if provided.
    Returns number of titles newly embedded.
    """
    from libzim.reader import Archive
    from indexer.embed import encode

    model_name = (cfg or {}).get(
        "embed_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    cache_dir = (cfg or {}).get("fastembed_cache_path", None)

    archive  = Archive(str(zim_path))
    con      = open_title_db(zim_path)
    existing = set(row[0] for row in con.execute("SELECT path FROM titles").fetchall())

    todo: list[tuple[str, str]] = []
    for i in range(archive.article_count):
        try:
            entry = archive._get_entry_by_id(i)
            if entry.is_redirect:
                continue
            path  = entry.path
            title = entry.title or path
            if _SKIP_RE.match(path) or _SKIP_RE.match(title):
                continue
            if path not in existing:
                todo.append((path, title))
        except Exception:
            continue

    total   = len(todo)
    done    = 0
    written = 0

    for batch_start in range(0, total, _BATCH):
        batch  = todo[batch_start: batch_start + _BATCH]
        texts  = [t for _, t in batch]
        try:
            vecs = encode(texts, model_name, cache_dir)
        except Exception as e:
            log(f"[title_index] embed batch failed: {e}")
            done += len(batch)
            continue

        rows = [
            (path, title, _vec_to_blob(vecs[j]))
            for j, (path, title) in enumerate(batch)
        ]
        con.executemany(
            "INSERT OR IGNORE INTO titles (path, title, vec) VALUES (?, ?, ?)", rows
        )
        con.commit()
        done    += len(batch)
        written += len(batch)
        if progress_cb:
            progress_cb(done, total)

    con.close()
    return written
