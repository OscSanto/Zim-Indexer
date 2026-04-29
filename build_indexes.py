#!/usr/bin/env python3
"""
Build structured and flat indexes sequentially overnight.

Usage:
    python build_indexes.py /path/to/wikipedia.zim

Order:
    1. Extract structured  (wikipedia/)
    2. Embed   structured
    3. Extract flat        (wikipedia_flat/)
    4. Embed   flat

Resumable — safe to Ctrl-C and re-run, picks up where it left off.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path


def _step(label: str, fn, *args, **kwargs) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\n[{ts}] ── {label} ──", flush=True)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
    except Exception as e:
        print(f"\nERROR in {label}: {e}")
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {label} done in {elapsed/60:.1f} min", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python build_indexes.py /path/to/file.zim")
        sys.exit(1)

    zim_path = Path(sys.argv[1]).resolve()
    if not zim_path.exists():
        print(f"ZIM file not found: {zim_path}")
        sys.exit(1)

    BASE_CFG = {
        "embed_model":      "BAAI/bge-small-en-v1.5",
        "embed_batch_size": 128,
        "faiss_nprobe":     64,
        "faiss_save_every": 1024,
        "output_dir_mode":  "auto",
        "min_prose_chars":  200,
        "min_infobox_rows": 3,
        "skip_namespaces":
            "Category,Template,Portal,File,Help,Special,"
            "Talk,Wikipedia,User,MediaWiki,Module",
    }

    struct_cfg = {**BASE_CFG, "flat_chunks": False}
    flat_cfg   = {**BASE_CFG, "flat_chunks": True}

    struct_dir = zim_path.parent / zim_path.stem
    flat_dir   = zim_path.parent / (zim_path.stem + "_flat")

    print(f"\nZIM        : {zim_path}")
    print(f"Structured : {struct_dir}")
    print(f"Flat       : {flat_dir}")
    print(f"Model      : {BASE_CFG['embed_model']}")

    from indexer.pipeline import run_extract, run_embed

    _step("Extract structured", run_extract, zim_path, struct_cfg)
    _step("Embed   structured", run_embed,   zim_path, struct_cfg)
    _step("Extract flat",       run_extract, zim_path, flat_cfg)
    _step("Embed   flat",       run_embed,   zim_path, flat_cfg)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"  Structured : {struct_dir}")
    print(f"  Flat       : {flat_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
