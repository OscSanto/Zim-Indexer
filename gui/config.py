"""
Persistent JSON configuration for the ZIM Indexer GUI.

Stored at: ~/.config/zim-indexer/config.json
"""
import json
from pathlib import Path

CONFIG_DIR  = Path.home() / ".config" / "zim-indexer"
CONFIG_FILE = CONFIG_DIR / "config.json"

# ── Embed model registry ──────────────────────────────────────────────────────
# (id, dim, size_mb, languages, short description)
EMBED_MODELS: list[tuple] = [
    (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        384, 470, "50+ languages",
        "AIIAB default. Multilingual, balanced quality/speed. ~470 MB download.",
    ),
    (
        "Snowflake/snowflake-arctic-embed-xs",
        384, 23, "English",
        "Smallest model. Only ~23 MB. Best for very low storage/RAM. English only.",
    ),
    (
        "BAAI/bge-small-en-v1.5",
        384, 130, "English",
        "Small English model. Good quality/size ratio. ~130 MB.",
    ),
    (
        "BAAI/bge-base-en-v1.5",
        768, 430, "English",
        "Higher quality, 768-dim output. Needs AIIAB config update + full re-index. ~430 MB.",
    ),
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        384, 90, "English",
        "Very small and fast. English only. Lower semantic quality. ~90 MB.",
    ),
]

MODEL_NAMES: list[str] = [m[0] for m in EMBED_MODELS]
MODEL_DIM:   dict[str, int] = {m[0]: m[1] for m in EMBED_MODELS}
MODEL_INFO:  dict[str, tuple] = {m[0]: m for m in EMBED_MODELS}

DEFAULTS: dict = {
    # Embedding
    "embed_model":          "BAAI/bge-small-en-v1.5",
    "fastembed_cache_path": str(Path.home() / ".cache" / "fastembed"),
    "embed_batch_size":     128,
    # FAISS
    "faiss_ivf_threshold":  500_000,
    "faiss_nprobe":         64,
    "faiss_save_every":     1024,
    "faiss_mmap":           False,
    # Pipeline priorities
    "priority_lead":        3,
    "priority_infobox":     10,
    "priority_prose":       12,
    # Content filtering
    "skip_namespaces":
        "Category,Template,Portal,File,Help,Special,Talk,Wikipedia,User,MediaWiki,Module",
    "skip_regex":
        r"^(Category:|Template:|Portal:|File:|Help:|Special:|Talk:)|\(disambiguation\)",
    "min_prose_chars":      200,
    "min_infobox_rows":     3,
    # Output
    "output_dir_mode":      "auto",
    "output_dir_custom":    "",
    "flat_chunks":          False,
    # Retrieval signal toggles (used by Search + Evaluate tabs)
    "use_faiss":            True,
    "use_title_bm25":       True,
    "use_para_bm25":        True,
    # Disabled by default — hand-tuned thresholds with no principled basis.
    # Turn ON only to compare against the heuristic-augmented baseline.
    "use_diversity_cap":    False,
    "use_mention_penalty":  False,
    "use_nav_boost":        False,
    "use_lead_augment":     False,
    "use_section_augment":  False,
    "eval_rrf_k":           60,
    "eval_diversity_max":   6,
    "eval_top_k":           10,
    # Kiwix
    "kiwix_base_url":       "http://127.0.0.1/kiwix/viewer",
    "kiwix_book_name":      "",
    # Last-used paths (not shown in Settings)
    "last_zim_dir":         str(Path.home()),
    "last_index_dir":       str(Path.home()),
    "last_browse_zim":      "",
}


class Config:
    def __init__(self) -> None:
        self._d: dict = dict(DEFAULTS)
        self.load()

    # ── dict-like access ──────────────────────────────────────────────────────

    def __getitem__(self, key: str):
        return self._d[key]

    def __setitem__(self, key: str, value) -> None:
        self._d[key] = value

    def get(self, key: str, default=None):
        return self._d.get(key, default)

    def as_dict(self) -> dict:
        return dict(self._d)

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        if CONFIG_FILE.exists():
            try:
                saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for k, v in saved.items():
                    if k in DEFAULTS:
                        self._d[k] = v
            except Exception:
                pass

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(self._d, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self._d = dict(DEFAULTS)
        self.save()

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def embed_dim(self) -> int:
        return MODEL_DIM.get(self._d["embed_model"], 384)

    def as_engine_cfg(self) -> dict:
        """Return a cfg dict suitable for passing to pipeline/query functions."""
        d = dict(self._d)
        d["embed_dim"] = self.embed_dim
        return d


# Module-level singleton shared across all GUI components
cfg = Config()
