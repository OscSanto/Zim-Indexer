"""
SQLite helpers for the ZIM indexer.
"""
import sqlite3
import time
from pathlib import Path


def open_db(db_path: Path) -> sqlite3.Connection:
    schema = Path(__file__).parent / "schema.sql"
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.executescript(schema.read_text())
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.commit()
    return con


def article_exists(con: sqlite3.Connection, title: str) -> bool:
    return con.execute(
        "SELECT 1 FROM articles WHERE title = ?", (title,)
    ).fetchone() is not None


def insert_article(con: sqlite3.Connection, title: str, url: str, zim_path: str) -> int:
    cur = con.execute(
        "INSERT INTO articles (title, url, zim_path, indexed_at) VALUES (?, ?, ?, ?)",
        (title, url, zim_path, time.time()),
    )
    return cur.lastrowid


def insert_chunks(con: sqlite3.Connection, article_id: int, chunks: list[dict]) -> None:
    con.executemany(
        "INSERT INTO chunks (article_id, section_title, chunk_index, text, embedded) "
        "VALUES (?, ?, ?, ?, 0)",
        [(article_id, c["section_title"], c["chunk_index"], c["text"]) for c in chunks],
    )


def mark_embedded(con: sqlite3.Connection, chunk_ids: list[int]):
    con.executemany(
        "UPDATE chunks SET embedded = 1 WHERE id = ?",
        [(cid,) for cid in chunk_ids],
    )
    con.commit()


def get_chunk_by_id(con: sqlite3.Connection, chunk_id: int) -> dict | None:
    row = con.execute(
        """SELECT c.id, c.article_id, c.section_title, c.chunk_index, c.text,
                  a.title, a.url
           FROM chunks c JOIN articles a ON c.article_id = a.id
           WHERE c.id = ?""",
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "chunk_id":      row[0],
        "article_id":    row[1],
        "section_title": row[2],
        "chunk_index":   row[3],
        "text":          row[4],
        "title":         row[5],
        "url":           row[6],
    }


def get_chunks_by_ids(con: sqlite3.Connection, chunk_ids: list[int]) -> list[dict]:
    """Batch fetch chunk metadata for a list of chunk IDs."""
    if not chunk_ids:
        return []
    ph   = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"SELECT c.id, c.article_id, c.section_title, c.chunk_index, c.text, "
        f"       a.title, a.url "
        f"FROM chunks c JOIN articles a ON c.article_id = a.id "
        f"WHERE c.id IN ({ph})",
        chunk_ids,
    ).fetchall()
    return [
        {
            "chunk_id":      r[0],
            "article_id":    r[1],
            "section_title": r[2],
            "chunk_index":   r[3],
            "text":          r[4],
            "title":         r[5],
            "url":           r[6],
        }
        for r in rows
    ]


def get_chunks_for_article(con: sqlite3.Connection,
                           article_id: int,
                           limit: int | None = None) -> list[tuple[int, str]]:
    """Return all chunk IDs for one article in chunk order."""
    sql = (
        "SELECT id, section_title FROM chunks "
        "WHERE article_id = ? "
        "ORDER BY chunk_index"
    )
    params: list[int] = [article_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [(int(r[0]), r[1]) for r in con.execute(sql, params).fetchall()]


def get_article_lead(con: sqlite3.Connection, article_id: int) -> str | None:
    """Return the first Lead chunk text for an article, or None."""
    row = con.execute(
        "SELECT text FROM chunks WHERE article_id = ? AND section_title = 'Lead' "
        "ORDER BY chunk_index LIMIT 1",
        (article_id,),
    ).fetchone()
    return row[0] if row else None


def get_section_first_para(con: sqlite3.Connection,
                           article_id: int,
                           section_title: str) -> str | None:
    """Return the first chunk text for a given section, or None."""
    row = con.execute(
        "SELECT text FROM chunks WHERE article_id = ? AND section_title = ? "
        "ORDER BY chunk_index LIMIT 1",
        (article_id, section_title),
    ).fetchone()
    return row[0] if row else None


# ── FTS helpers ───────────────────────────────────────────────────────────────

def _para_text(text: str) -> str:
    """Strip the 'Article/Section/Text:' prefix — return only the paragraph content.
    Handles both prose chunks ('\\nText: …') and infobox chunks ('\\nFact: …').
    """
    for prefix in ("\nText: ", "\nFact: "):
        idx = text.find(prefix)
        if idx != -1:
            return text[idx + len(prefix):]
    return text


def _fts_words(query: str) -> list[str]:
    """Tokenise query into quoted FTS5 terms, skipping very short words."""
    return [f'"{w}"' for w in query.replace("_", " ").split() if len(w) >= 2]


def init_fts(con: sqlite3.Connection) -> None:
    """
    Build / incrementally update both FTS5 indexes.
      articles_fts — BM25 over article titles
      chunks_fts   — BM25 over clean paragraph text

    Safe to call multiple times; only indexes rows not yet present.
    """
    # ── articles_fts ─────────────────────────────────────────────────────────
    try:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, tokenize='porter unicode61'
            )
        """)
    except Exception:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, tokenize='unicode61'
            )
        """)
    con.execute("""
        INSERT INTO articles_fts(rowid, title)
        SELECT id, title FROM articles
        WHERE id NOT IN (SELECT rowid FROM articles_fts)
    """)

    # ── chunks_fts ────────────────────────────────────────────────────────────
    # Contentless FTS5 table — we populate it ourselves from chunks.text.
    # fts_meta tracks the last chunk id that was indexed so we can do
    # incremental updates without scanning the whole table each startup.
    try:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                para_text, tokenize='porter unicode61', content=''
            )
        """)
    except Exception:
        con.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                para_text, tokenize='unicode61', content=''
            )
        """)

    row = con.execute(
        "SELECT value FROM fts_meta WHERE key = 'chunks_fts_last_id'"
    ).fetchone()
    last_id = int(row[0]) if row else 0

    new_rows = con.execute(
        "SELECT id, text FROM chunks WHERE id > ? ORDER BY id",
        (last_id,),
    ).fetchall()

    if new_rows:
        con.executemany(
            "INSERT INTO chunks_fts(rowid, para_text) VALUES (?, ?)",
            [(r[0], _para_text(r[1])) for r in new_rows],
        )
        new_last = new_rows[-1][0]
        con.execute(
            "INSERT OR REPLACE INTO fts_meta(key, value) VALUES ('chunks_fts_last_id', ?)",
            (str(new_last),),
        )

    con.commit()


def title_search_scored(con: sqlite3.Connection,
                         query: str,
                         limit: int = 30) -> list[tuple[int, float]]:
    """
    FTS5 BM25 search on article titles.
    Returns [(article_id, bm25_score), ...] — higher score is better.
    """
    words = _fts_words(query)
    if not words:
        return []
    try:
        rows = con.execute(
            "SELECT rowid, -rank FROM articles_fts "
            "WHERE articles_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(words), limit),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
    except Exception:
        return []


def title_search(con: sqlite3.Connection,
                 query: str,
                 limit: int = 30) -> list[tuple[int, int]]:
    """
    Compatibility wrapper for the CLI retriever.
    Returns [(article_id, one_based_rank), ...].
    """
    return [(article_id, rank) for rank, (article_id, _score)
            in enumerate(title_search_scored(con, query, limit), start=1)]


def chunk_text_search(con: sqlite3.Connection,
                       query: str,
                       limit: int = 40) -> list[tuple[int, float]]:
    """
    FTS5 BM25 search on clean paragraph text.
    Returns [(chunk_id, bm25_score), ...] — higher score is better.
    """
    words = _fts_words(query)
    if not words:
        return []
    try:
        rows = con.execute(
            "SELECT rowid, -rank FROM chunks_fts "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(words), limit),
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]
    except Exception:
        return []


def stats(con: sqlite3.Connection) -> dict:
    articles = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    chunks   = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = con.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 1").fetchone()[0]
    return {"articles": articles, "chunks": chunks, "embedded": embedded}
