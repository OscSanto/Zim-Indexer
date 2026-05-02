# ZIM Indexer

Structure-aware hybrid retrieval over Wikipedia ZIM files for offline medical question answering.

Extracts, embeds, and queries Wikipedia content from `.zim` files using a combination of BM25 keyword search and FAISS dense vector search — with no internet connection required at query time.

---

## Features

- **Hybrid retrieval** — Reciprocal Rank Fusion (RRF) over BM25 (title + paragraph) and FAISS dense vectors
- **Structured chunking** — preserves `Article / Section / Text` hierarchy for context-aware retrieval
- **Flat chunking** — strips metadata for ablation comparison
- **Offline** — all inference runs locally via [Ollama](https://ollama.com); embeddings via [fastembed](https://github.com/qdrant/fastembed)
- **Resumable** — safe to stop and restart; picks up from where it left off
- **GUI** — Tkinter interface for indexing, searching, browsing, and evaluation
- **CLI tools** — headless pipeline for overnight builds and batch evaluation

---

## Requirements

- Python 3.10+
- A `.zim` file (e.g. Wikipedia for Medicine from [Kiwix](https://www.kiwix.org/en/content/))
- Ollama running locally (for LLM inference in the Evaluate tab)

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional — rich HTML rendering in the Browse tab:

```bash
pip install tkinterweb
```

---

## Quick Start

### 1. Launch the GUI

```bash
python main.py
```

### 2. Index a ZIM file

1. Open the **Index** tab
2. Browse to your `.zim` file
3. Click **Build All Indexes** to run Extract + Embed for both Structured and Flat indexes sequentially
   - Or use **Both Steps** / **Extract Only** / **Embed Only** for the currently selected index type

Progress, ETA, and logs are shown in real time. The process is resumable — if stopped, re-clicking the same button continues from where it left off.

### 3. Search

Open the **Search** tab, select your index directory, and type a query. Results show article title, section, and matched text.

### 4. Browse

The **Browse** tab renders ZIM articles as HTML (or plain text if `tkinterweb` is not installed).

---

## Overnight Build (CLI)

To build both structured and flat indexes in one unattended run:

```bash
python build_indexes.py /path/to/wikipedia.zim
```

With logging to file:

```bash
nohup python build_indexes.py /path/to/wikipedia.zim > build.log 2>&1 &
tail -f build.log
```

Steps run in order:
1. Extract structured → `wikipedia/`
2. Embed structured
3. Extract flat → `wikipedia_flat/`
4. Embed flat

---

## Evaluation

### Download datasets

```bash
python download_data.py
```

Downloads MedQA, MedMCQA, MMLU-Pro (medicine), and PubMedQA into `data/`.

### Run retrieval evaluation

```bash
python evaluate.py \
  --dataset data/medqa_test.jsonl \
  --structured /path/to/wikipedia \
  --flat       /path/to/wikipedia_flat \
  --n 200 --top-k 10 \
  --out results/medqa_results.csv
```

Compares: BM25 Only, Hybrid Flat, Hybrid Structured, Hybrid Structured + Lead Augmentation.

Metrics: Hit@1, Hit@3, Hit@5, Hit@10, MRR@10.

Per-question output now includes `item_index` so fixed subsets stay traceable.

### Run LLM inference evaluation (GUI)

Open the **Evaluate** tab:

1. Select a dataset file (JSONL or CSV)
2. Select an index directory
3. Pick an Ollama model from the list
4. Choose retrieval conditions (No RAG, Structured RAG, Flat RAG)
5. Set context budgets (e.g. `128,512,0` for 128 tok / 512 tok / unlimited)
6. Click **Run Evaluation**

Results are saved to CSV. Use **Compare** to load two result files side by side.

The GUI **Evaluate** tab also includes a **Subset And Review Workflow** section:

- `Create Fixed 100` — one-time frozen subset for paper runs
- `Create Quick 20` — short smoke-test subset
- `Build Audit CSV` — manual review sheet for checking whether the correct answer maps to a real ZIM article title
- `Build Gold Subset` — reviewed retrieval-only subset derived from the audit CSV
- `View CSV` / `Open Any CSV…` — lightweight in-app CSV viewer for audit and results files

### Fixed subsets and manual review

For paper runs, freeze a fixed subset once and reuse it for every condition:

```bash
python make_eval_subset.py \
  --dataset data/medqa_test.jsonl \
  --n 100 --shuffle --seed 42 \
  --out data/medqa_fixed100.jsonl
```

For fast smoke tests, create a smaller fixed subset:

```bash
python make_eval_subset.py \
  --dataset data/medqa_fixed100.jsonl \
  --n 20 \
  --out data/medqa_quick20.jsonl
```

To build a manual retrieval-audit CSV for human review:

```bash
python build_manual_audit.py \
  --dataset data/medqa_fixed100.jsonl \
  --structured /path/to/wikipedia \
  --flat /path/to/wikipedia_flat \
  --top-k 10 \
  --out results/medqa_fixed100_audit.csv
```

This writes:

- `results/medqa_fixed100_audit.csv` — one row per QA item with answer text, options, candidate title matches, and retrieved top titles for each condition
- `results/medqa_fixed100_audit.instructions.txt` — columns to mark manually

After manual review, build the gold retrieval subset:

```bash
python build_golden_subset.py \
  --review results/medqa_fixed100_audit.csv \
  --out data/medqa_fixed100_gold.jsonl
```

Recommended paper workflow:

1. Run end-to-end QA accuracy on the full fixed 100-question subset.
2. Run retrieval evaluation on the manually reviewed gold subset only.
3. Keep `No Retrieval` as the main baseline for LLM accuracy comparisons.

---

## Index Structure

Each index lives in a directory next to the ZIM file:

```
wikipedia/          ← structured index
  data.db           ← SQLite: articles, chunks, BM25 FTS
  faiss.index       ← FAISS IVF flat index
  faiss_ids.npy     ← chunk ID mapping

wikipedia_flat/     ← flat index (no Article/Section prefixes)
  data.db
  faiss.index
  faiss_ids.npy
```

---

## Embedding Models

Default: `BAAI/bge-small-en-v1.5` (384-dim, ~130 MB, English)

| Model | Dim | Size | Languages |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~130 MB | English |
| `BAAI/bge-base-en-v1.5` | 768 | ~430 MB | English |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~470 MB | 50+ languages |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90 MB | English |
| `Snowflake/snowflake-arctic-embed-xs` | 384 | ~23 MB | English |

Change the model in the **Settings** tab. After changing models, click **Reset Embedded Flags** in the Index tab, delete the old `faiss.index`, and re-run Embed.

---

## Configuration

Settings are stored at `~/.config/zim-indexer/config.json` and editable via the **Settings** tab in the GUI.

---

## Project Layout

```
main.py                 ← GUI entry point
build_indexes.py        ← CLI overnight build script
evaluate.py             ← CLI retrieval evaluation
infer.py                ← CLI LLM inference evaluation
download_data.py        ← download benchmark datasets

gui/
  app.py                ← Tkinter app shell
  index_tab.py          ← indexing UI
  search_tab.py         ← search UI
  browse_tab.py         ← article browser
  evaluate_tab.py       ← LLM evaluation UI
  settings_tab.py       ← settings UI
  config.py             ← persistent config

indexer/
  pipeline.py           ← extract + embed orchestration
  extract.py            ← ZIM parsing and chunking
  embed.py              ← fastembed wrapper (GPU/CPU)
  index.py              ← FAISS index build/load
  query.py              ← hybrid retrieval (BM25 + FAISS + RRF)
  db.py                 ← SQLite helpers
  title_index.py        ← BM25 title index
  schema.sql            ← database schema

data/                   ← evaluation datasets
results/                ← evaluation output CSVs
```
