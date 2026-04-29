# Running ZIM Indexer on Google Colab

The GUI does not work on Colab. This guide uses the CLI scripts only.

The ZIM file stays on Google Drive throughout. The index is built on Colab's
local disk (fast SSD, ~100 GB free) and then copied to Drive when done.

---

## Step 1 — Open a new notebook and set the runtime

Runtime → Change runtime type → **T4 GPU**

---

## Step 2 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Verify your ZIM file is visible:

```python
import os
ZIM = "/content/drive/MyDrive/wikipedia.zim"   # ← change this path
print(os.path.exists(ZIM), os.path.getsize(ZIM) // 1_000_000, "MB")
```

---

## Step 3 — Clone the repo and install dependencies

```python
!git clone https://github.com/YOUR_USERNAME/zim-indexer.git /content/zim-indexer
%cd /content/zim-indexer
```

```python
!pip install -q fastembed faiss-cpu libzim beautifulsoup4 numpy requests psutil
```

---

## Step 4 — Build the indexes

Indexes are written to Colab local disk (next to the ZIM path).
Because the ZIM is on Drive, "auto" mode would write to Drive too — which is
slow for millions of write operations. Instead, copy the ZIM locally first:

```python
# Copy ZIM to local disk (fast writes during indexing)
!cp "{ZIM}" /content/wikipedia.zim
```

Then build both structured and flat indexes:

```python
!python build_indexes.py /content/wikipedia.zim
```

This runs four steps in order:
1. Extract structured → `/content/wikipedia/`
2. Embed structured (uses T4 GPU automatically)
3. Extract flat → `/content/wikipedia_flat/`
4. Embed flat

Embedding 2.5 million chunks takes roughly **2–4 hours** on a T4.

> **Tip:** To avoid losing progress if the session disconnects, stream output
> to a log file and watch it in a second cell:
>
> ```python
> !python build_indexes.py /content/wikipedia.zim > /content/build.log 2>&1 &
> !tail -f /content/build.log
> ```

The pipeline is **resumable** — if the session drops, re-run the same command
and it picks up from where it left off (already-extracted articles and
already-embedded chunks are skipped).

---

## Step 5 — Copy the finished indexes to Drive

```python
!cp -r /content/wikipedia      "/content/drive/MyDrive/wikipedia"
!cp -r /content/wikipedia_flat "/content/drive/MyDrive/wikipedia_flat"
```

Storage needed on Drive:

| Directory | Approximate size |
|---|---|
| `wikipedia/` | ~2–3 GB |
| `wikipedia_flat/` | ~2–3 GB |

You do **not** need to copy the ZIM back — it was never moved.

---

## Step 6 — Run evaluation (optional)

```python
!python evaluate.py \
  --dataset data/medqa_test.jsonl \
  --structured /content/wikipedia \
  --flat       /content/wikipedia_flat \
  --n 500 --top-k 10 \
  --out /content/drive/MyDrive/results/medqa_results.csv
```

Results are written directly to Drive.

---

## Resuming a later session

If you come back after the session expired, the local disk is gone but the
Drive indexes are intact. Copy them back to local disk before querying:

```python
!cp -r "/content/drive/MyDrive/wikipedia"      /content/wikipedia
!cp -r "/content/drive/MyDrive/wikipedia_flat" /content/wikipedia_flat
```

Then run evaluate or infer directly against `/content/wikipedia`.

---

## Changing the embedding model

The default is `BAAI/bge-small-en-v1.5`. To use a different model, pass it
via a config override before running `build_indexes.py`:

```python
import sys
sys.argv = ["build_indexes.py", "/content/wikipedia.zim"]

# Patch the config before the script reads it
from indexer import pipeline
# Edit BASE_CFG in build_indexes.py directly, or override here:
```

Or edit `build_indexes.py` line 49 directly:

```python
"embed_model": "BAAI/bge-small-en-v1.5",   # already the default
```

---

## Troubleshooting

**Out of memory during embed**
Reduce `embed_batch_size` in `build_indexes.py` `BASE_CFG`:
```python
"embed_batch_size": 32,   # default 128; lower if T4 OOM
```

**Session disconnects mid-embed**
Re-run the same `build_indexes.py` command. It skips already-embedded chunks
(`embedded=1` flag in SQLite) and continues from the last checkpoint.

**Drive writes are slow**
Always index to local disk (`/content/`) and copy to Drive only at the end.
Writing millions of rows directly to Drive over FUSE is 10–20× slower.

**`libzim` not found**
```python
!pip install libzim --pre   # use pre-release if stable build fails on Colab
```
