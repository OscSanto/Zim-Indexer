"""
FAISS index management — configurable dimension and nprobe.

Index type chosen automatically by corpus size:
  n < ivf_threshold  → IndexFlatIP   (exact, no training needed)
  n >= ivf_threshold → IndexIVFFlat  (approximate, faster at scale)

Both are wrapped in IndexIDMap so FAISS vector ID == SQLite chunks.id.
"""
import numpy as np
import faiss
from pathlib import Path

_NPROBE_DEFAULT    = 64
_IVF_MIN_N_DEFAULT = 500_000

_INDEX_CACHE: dict[tuple, "faiss.IndexIDMap"] = {}


def _make_flat(dim: int) -> faiss.IndexIDMap:
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def make_ivf(n_total: int, train_vecs: np.ndarray,
             nprobe: int = _NPROBE_DEFAULT) -> faiss.IndexIDMap:
    """Create and train an IVFFlat index. Must be called before adding any vectors."""
    dim       = train_vecs.shape[1]
    nlist     = max(4, min(int(np.sqrt(n_total)), 4096))
    quantizer = faiss.IndexFlatIP(dim)
    inner     = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    inner.train(train_vecs)
    inner.nprobe = min(nlist, nprobe)
    idx = faiss.IndexIDMap(inner)
    print(f"  IVFFlat trained: nlist={nlist} nprobe={inner.nprobe} "
          f"train_n={len(train_vecs):,}", flush=True)
    return idx


def load_or_create(index_path: Path, dim: int = 384,
                   nprobe: int = _NPROBE_DEFAULT,
                   mmap: bool = False) -> faiss.IndexIDMap:
    key = (str(index_path), nprobe, mmap)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    if index_path.exists() and index_path.stat().st_size > 100:
        io_flags = faiss.IO_FLAG_MMAP if mmap else 0
        idx   = faiss.read_index(str(index_path), io_flags)
        inner = faiss.downcast_index(idx.index)
        if hasattr(inner, "nprobe"):
            inner.nprobe = min(nprobe, getattr(inner, "nlist", nprobe))
        if hasattr(inner, "make_direct_map"):
            inner.make_direct_map()
        print(f"  Loaded FAISS index: {idx.ntotal:,} vectors from {index_path}",
              flush=True)
        _INDEX_CACHE[key] = idx
        return idx

    print(f"  Creating new FAISS index (dim={dim}, FlatIP)", flush=True)
    idx = _make_flat(dim)
    _INDEX_CACHE[key] = idx
    return idx


def save(index: faiss.IndexIDMap, index_path: Path) -> None:
    tmp = index_path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    if index_path.exists():
        index_path.unlink()
    tmp.rename(index_path)


def add_vectors(index: faiss.IndexIDMap, chunk_ids: list[int],
                vectors: np.ndarray) -> None:
    """vectors: (N, dim) float32 L2-normalised. chunk_ids: SQLite chunk IDs."""
    index.add_with_ids(vectors, np.array(chunk_ids, dtype=np.int64))


def search(index: faiss.IndexIDMap, query_vec: np.ndarray,
           top_k: int = 10) -> list[tuple[int, float]]:
    """query_vec: (dim,) float32 L2-normalised. Returns [(chunk_id, score), ...]."""
    q = query_vec.reshape(1, -1).astype(np.float32)
    scores, ids = index.search(q, top_k)
    return [
        (int(cid), float(score))
        for cid, score in zip(ids[0], scores[0])
        if cid != -1
    ]
