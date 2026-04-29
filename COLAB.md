# Running ZIM Indexer on Google Colab

No GUI — CLI only. Indexes are built on Colab's local disk and copied to Drive when done.

**Recommended runtime:** A100 GPU (Colab Pro) or T4 (free tier, slower)
**Estimated time:** ~1 hour on A100, ~4–6 hours on T4

---

## Cell 1 — Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## Cell 2 — Set paths and verify ZIM

```python
import os

ZIM_DRIVE = "/content/drive/MyDrive/rag_pipeline/wikipedia_en_medicine_maxi_2026-04.zim"

print("Exists:", os.path.exists(ZIM_DRIVE))
print("Size MB:", os.path.getsize(ZIM_DRIVE) // 1_000_000)
```

> Change `ZIM_DRIVE` to match your actual Drive path.

---

## Cell 3 — Clone repo

```python
%cd /content
!rm -rf /content/zim-indexer
!git clone https://github.com/OscSanto/zim-indexer.git /content/zim-indexer
%cd /content/zim-indexer
```

---

## Cell 4 — Install dependencies (GPU version)

```python
!pip uninstall -y onnxruntime
!pip install -q fastembed-gpu faiss-cpu libzim beautifulsoup4 numpy requests psutil
```

> `fastembed-gpu` bundles the correct `onnxruntime-gpu` version. Do NOT use
> `fastembed` (CPU-only) or manually install `onnxruntime-gpu` — version
> conflicts will cause `AttributeError: module 'onnxruntime' has no attribute 'SessionOptions'`.

---

## Cell 5 — Copy ZIM to local disk

```python
!rsync -ah --progress "$ZIM_DRIVE" /content/wikipedia.zim
```

```python
import os
print("Local exists:", os.path.exists("/content/wikipedia.zim"))
print("Local size MB:", os.path.getsize("/content/wikipedia.zim") // 1_000_000)
```

> Always index to local disk (`/content/`), not Drive. Writing millions of rows
> directly to Drive over FUSE is 10–20× slower.

---

## Cell 6 — Verify repo and GPU

```python
%cd /content/zim-indexer
!ls
!nvidia-smi
```

> If `nvidia-smi` errors, go to **Runtime → Change runtime type → GPU (T4 or A100)**
> and reconnect before continuing.

---

## Cell 7 — Build all indexes (live output)

```python
%cd /content/zim-indexer

import subprocess, sys

proc = subprocess.Popen(
    [sys.executable, "build_indexes.py", "/content/wikipedia.zim"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for line in proc.stdout:
    print(line, end="", flush=True)

proc.wait()
print("Exit code:", proc.returncode)
```

This runs all 4 steps sequentially:
1. Extract structured → `/content/wikipedia/`
2. Embed structured (GPU)
3. Extract flat → `/content/wikipedia_flat/`
4. Embed flat (GPU)

Progress lines show rate, ETA, GPU %, VRAM, and disk free:
```
Embedded 20,480/1,475,646 (805 chunks/s)  ETA 0h30m  gpu=84% mem=28637/97887MB  disk=196.2GB free
```

---

## Cell 8 — Copy finished indexes to Drive

```python
!rsync -ah --progress /content/wikipedia      "/content/drive/MyDrive/rag_pipeline/wikipedia"
!rsync -ah --progress /content/wikipedia_flat "/content/drive/MyDrive/rag_pipeline/wikipedia_flat"
```

> Each index is ~2–3 GB. Do this as soon as the build finishes — Colab
> local disk is wiped when the session ends.

---

## Cell 9 — Run evaluation (structured only)

```python
!mkdir -p /content/drive/MyDrive/rag_pipeline/results

%cd /content/zim-indexer

!python evaluate.py \
  --dataset data/medqa_test.jsonl \
  --structured /content/wikipedia \
  --n 500 \
  --top-k 10 \
  --out /content/drive/MyDrive/rag_pipeline/results/medqa_results.csv
```

---

## Cell 10 — Run evaluation (structured + flat comparison)

```python
!python evaluate.py \
  --dataset data/medqa_test.jsonl \
  --structured /content/wikipedia \
  --flat       /content/wikipedia_flat \
  --n 500 \
  --top-k 10 \
  --out /content/drive/MyDrive/rag_pipeline/results/medqa_results_with_flat.csv
```

> Use Cell 10 only after both indexes are fully built. Omitting `--flat`
> runs 3 systems (BM25 Only, Hybrid Structured, Hybrid Struct + Lead).
> Adding `--flat` adds Hybrid Flat for the full 4-system comparison.

---

## Resuming after a disconnection

Colab local disk survives a **runtime restart** but is wiped on full
**session disconnect**. If the session is still alive, just re-run from Cell 7 —
the pipeline resumes automatically:

- Extract: scans fast (`new=0`), skips already-extracted articles
- Embed: loads existing `faiss.index` checkpoint, skips `embedded=1` chunks

After a restart, re-run **Cell 1 → Cell 4 → Cell 7** in order (skip 3 and 5
if the disk files are still there).

Check what survived:
```python
import os
print(os.path.exists("/content/wikipedia.zim"))
print(os.path.exists("/content/wikipedia/faiss.index"))
```

If the ZIM is gone, re-run Cell 5. If the index is gone, copy it back from Drive:
```python
!rsync -ah --progress "/content/drive/MyDrive/rag_pipeline/wikipedia" /content/wikipedia
```

---

## Notes and considerations

**GPU not detected (`gpu: none detected`)**
Run Cell 4 again — `fastembed-gpu` must be installed after a runtime restart.
Also check `!nvidia-smi` to confirm a GPU is allocated.

**`AttributeError: module 'onnxruntime' has no attribute 'SessionOptions'`**
Version conflict between `onnxruntime` and `onnxruntime-gpu`. Fix:
```python
!pip uninstall -y onnxruntime onnxruntime-gpu fastembed
!pip install fastembed-gpu
```

**`faiss.tmp` rename error**
Stale file from a previous crashed run. Fix:
```python
!rm -f /content/wikipedia/faiss.tmp
!rm -f /content/wikipedia_flat/faiss.tmp
```
Then re-run Cell 7.

**Slow embed speed (~200 chunks/s instead of 800+)**
SQLite prefetch bottleneck. Make sure you pulled the latest code (`git pull`)
which fetches all 8 batches in one SQL query instead of 8 separate queries.

**ZIM file name**
The local ZIM path (`/content/wikipedia.zim`) determines the index directory name
(`/content/wikipedia/`). Keep the local name consistent across sessions or the
pipeline will create a new index instead of resuming the existing one.

**Drive storage needed**
| File | Size |
|---|---|
| ZIM file (stays on Drive, not copied) | ~2.2 GB |
| `wikipedia/` structured index | ~2–3 GB |
| `wikipedia_flat/` flat index | ~2–3 GB |
| Results CSVs | < 10 MB |
