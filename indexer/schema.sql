-- Phase 1 ZIM embedding index schema
-- FAISS vector ID == chunks.id (via IndexIDMap)

CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    url         TEXT    NOT NULL,
    zim_path    TEXT    NOT NULL,
    indexed_at  REAL    NOT NULL   -- unix timestamp
);

CREATE INDEX IF NOT EXISTS idx_articles_title ON articles(title);

-- All extracted sections stored here — even ones not yet embedded.
-- Phase 1 embeds chunk_index 0,1,2 (lead + 2 sections).
-- Phase 2 can embed the rest without re-parsing the ZIM.
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id    INTEGER NOT NULL REFERENCES articles(id),
    section_title TEXT    NOT NULL,
    chunk_index   INTEGER NOT NULL,   -- 0=lead, 1=section1, 2=section2, ...
    text          TEXT    NOT NULL,
    embedded      INTEGER NOT NULL DEFAULT 0  -- 0=pending, 1=done
);

CREATE INDEX IF NOT EXISTS idx_chunks_article   ON chunks(article_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded  ON chunks(embedded);

-- Tracks FTS5 chunk index state so init_fts() can do incremental updates.
CREATE TABLE IF NOT EXISTS fts_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
