"""
Embedding via fastembed — configurable model name.

Supports parallel multi-stream GPU inference for higher utilization.
Multiple model instances run inference simultaneously on different CUDA streams.
"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

_lock = threading.Lock()
_model_pools: dict[str, list[TextEmbedding]] = {}
_pool_locks: dict[str, threading.Semaphore] = {}

# Number of parallel model instances (CUDA streams).
# Increase for higher GPU utilization if you have spare VRAM.
# Safe default is 2; 12 works on an RTX 2070 Ti (8GB).
NUM_STREAMS = 8


def unload_models() -> None:
    """Clear the model cache, forcing fresh GPU session on next use."""
    with _lock:
        _model_pools.clear()
        _pool_locks.clear()


def _detect_providers() -> list[str]:
    """Return ONNX execution providers: CUDA > ROCm > DirectML > CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "ROCMExecutionProvider" in available:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def gpu_available() -> bool:
    """True if any GPU provider (CUDA, ROCm, or DirectML) is available."""
    providers = _detect_providers()
    return providers[0] != "CPUExecutionProvider"


def gpu_info() -> str:
    """Human-readable GPU provider name, or 'CPU'."""
    providers = _detect_providers()
    return {
        "CUDAExecutionProvider": "NVIDIA CUDA",
        "ROCMExecutionProvider": "AMD ROCm",
        "DmlExecutionProvider":  "AMD/Intel DirectML",
    }.get(providers[0], "CPU")


def _init_model_pool(model_name: str, cache_dir: str | None = None) -> None:
    """Initialize pool of model instances for parallel inference."""
    if model_name in _model_pools:
        return

    cache = cache_dir or str(Path.home() / ".cache" / "fastembed")
    providers = _detect_providers()
    device = gpu_info()
    os.environ.setdefault("FASTEMBED_CACHE_PATH", cache)

    n_streams = NUM_STREAMS if gpu_available() else 1
    print(f"  Loading {n_streams}x embed model {model_name!r} on {device} ...", flush=True)

    try:
        pool = []
        for i in range(n_streams):
            m = TextEmbedding(model_name, cache_dir=cache, providers=providers)
            if i == 0:
                list(m.embed(["warmup"]))  # warmup first instance only; catches OOM early
            pool.append(m)
    except Exception as e:
        if providers[0] != "CPUExecutionProvider":
            print(f"  [warn] GPU init failed ({e}); falling back to CPU.", flush=True)
            providers = ["CPUExecutionProvider"]
            n_streams = 1
            pool = [TextEmbedding(model_name, cache_dir=cache, providers=providers)]
            list(pool[0].embed(["warmup"]))
            device = "CPU"
        else:
            raise

    _model_pools[model_name] = pool
    _pool_locks[model_name] = threading.Semaphore(n_streams)
    print(f"  Embed model ready ({device}, {n_streams} streams).", flush=True)


def _get_model_from_pool(model_name: str) -> tuple[TextEmbedding, int]:
    """Get an available model instance from the pool."""
    pool = _model_pools[model_name]
    # Find first available model
    for i, model in enumerate(pool):
        return model, i
    return pool[0], 0


def encode(texts: list[str], model_name: str,
           cache_dir: str | None = None) -> np.ndarray:
    """
    Encode texts → L2-normalised (N, dim) float32 array.
    Thread-safe, uses single model instance.
    """
    with _lock:
        _init_model_pool(model_name, cache_dir)
        model = _model_pools[model_name][0]
        vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def encode_parallel(batches: list[list[str]], model_name: str,
                    cache_dir: str | None = None) -> list[np.ndarray]:
    """
    Encode multiple batches in parallel using multiple CUDA streams.
    Returns list of L2-normalised (N, dim) float32 arrays, one per batch.
    """
    with _lock:
        _init_model_pool(model_name, cache_dir)

    pool = _model_pools[model_name]
    n_streams = len(pool)
    results = [None] * len(batches)

    # Per-model locks to prevent concurrent use of same instance
    model_locks = [threading.Lock() for _ in range(n_streams)]

    def process_batch(batch_idx: int, texts: list[str]) -> tuple[int, np.ndarray]:
        # Round-robin assignment to model instances
        model_idx = batch_idx % n_streams
        with model_locks[model_idx]:
            model = pool[model_idx]
            vecs = np.array(list(model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return batch_idx, vecs / norms

    # Process all batches in parallel
    with ThreadPoolExecutor(max_workers=n_streams) as executor:
        futures = [
            executor.submit(process_batch, i, batch)
            for i, batch in enumerate(batches)
        ]
        for future in as_completed(futures):
            idx, vecs = future.result()
            results[idx] = vecs

    return results
