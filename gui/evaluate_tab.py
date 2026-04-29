"""Evaluate tab — retrieval evaluation and LLM inference against QA datasets."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path

from gui.config import cfg


class EvaluateTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self._q: queue.Queue = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

        self._build_ui()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    _DEFAULT_PROMPT = (
        "You are a medical expert answering multiple-choice questions. "
        "Reply with only the answer letter."
    )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        # ── Paths ─────────────────────────────────────────────────────────────
        path_frame = ttk.LabelFrame(self, text="Paths", padding=8)
        path_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        path_frame.columnconfigure(1, weight=1)

        def _row(frame, label, var, browse_cmd, r):
            ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(frame, textvariable=var)
            e.grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            ttk.Button(frame, text="Browse…", command=browse_cmd, width=9).grid(
                row=r, column=2, sticky="e", pady=2
            )
            return e

        self._dataset_var   = tk.StringVar()
        self._struct_var    = tk.StringVar(value=cfg.get("last_index_dir", ""))
        self._flat_var      = tk.StringVar()
        self._out_var       = tk.StringVar(value="results")
        self._dataset_var.trace_add("write", lambda *_: self._on_dataset_change())

        _row(path_frame, "Dataset:",          self._dataset_var, self._browse_dataset, 0)
        _row(path_frame, "Structured index:", self._struct_var,  self._browse_struct,  1)
        _row(path_frame, "Flat index:",       self._flat_var,    self._browse_flat,    2)
        _row(path_frame, "Output folder:",    self._out_var,     self._browse_out,     3)

        ttk.Label(path_frame,
                  text="Flat index is optional — leave blank to skip the flat baseline.",
                  foreground="gray", font=("TkDefaultFont", 9),
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ── Parameters ────────────────────────────────────────────────────────
        param_frame = ttk.LabelFrame(self, text="Parameters", padding=8)
        param_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        # Row 0: N slider + count label + Top-K
        ttk.Label(param_frame, text="Questions:").grid(row=0, column=0, sticky="w", padx=4)
        self._n_var = tk.IntVar(value=200)
        self._n_scale = tk.Scale(
            param_frame, variable=self._n_var,
            from_=10, to=10000, resolution=10,
            orient="horizontal", length=200, showvalue=False,
        )
        self._n_scale.grid(row=0, column=1, sticky="ew", padx=4)
        param_frame.columnconfigure(1, weight=1)
        self._n_count_lbl = ttk.Label(param_frame, text="200 of ? available",
                                       foreground="gray", width=22)
        self._n_count_lbl.grid(row=0, column=2, sticky="w", padx=4)
        self._n_var.trace_add("write", lambda *_: self._update_n_label())

        ttk.Label(param_frame, text="Top-K:").grid(row=0, column=3, sticky="w", padx=(16, 4))
        self._topk_var = tk.StringVar(value=str(cfg.get("eval_top_k", 10)))
        ttk.Spinbox(param_frame, textvariable=self._topk_var, from_=1, to=50,
                    increment=1, width=5).grid(row=0, column=4, sticky="w", padx=4)

        # Row 1: Shuffle option
        self._shuffle_var = tk.BooleanVar(value=False)
        self._seed_var    = tk.StringVar(value="42")
        shuffle_frame = ttk.Frame(param_frame)
        shuffle_frame.grid(row=1, column=0, columnspan=5, sticky="w", pady=(2, 0))
        ttk.Checkbutton(shuffle_frame, text="Shuffle questions",
                        variable=self._shuffle_var).pack(side="left", padx=4)
        ttk.Label(shuffle_frame, text="Seed:").pack(side="left", padx=(8, 2))
        ttk.Entry(shuffle_frame, textvariable=self._seed_var, width=6).pack(side="left")
        ttk.Label(shuffle_frame,
                  text="  (unchecked = first N questions in file order)",
                  foreground="gray", font=("TkDefaultFont", 9)).pack(side="left")

        # Row 2: Context budget (LLM mode)
        budget_frame = ttk.Frame(param_frame)
        budget_frame.grid(row=2, column=0, columnspan=5, sticky="w", pady=(2, 0))
        ttk.Label(budget_frame, text="Ctx budgets (tokens):").pack(side="left", padx=4)
        self._budgets_var = tk.StringVar(value="128 512 0")
        ttk.Entry(budget_frame, textvariable=self._budgets_var, width=18).pack(side="left")
        ttk.Label(budget_frame,
                  text="  space-separated; 0 = unlimited  (LLM mode only)",
                  foreground="gray", font=("TkDefaultFont", 9)).pack(side="left")

        # Row 3: Mode selector
        ttk.Label(param_frame, text="Mode:").grid(row=3, column=0, sticky="w", padx=4, pady=(6, 0))
        self._mode_var = tk.StringVar(value="retrieval")
        mode_frame = ttk.Frame(param_frame)
        mode_frame.grid(row=3, column=1, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Radiobutton(mode_frame, text="Retrieval eval  (Hit@K, MRR)",
                        variable=self._mode_var, value="retrieval",
                        command=self._on_mode_change).pack(side="left", padx=4)
        ttk.Radiobutton(mode_frame, text="LLM inference  (accuracy via Ollama)",
                        variable=self._mode_var, value="llm",
                        command=self._on_mode_change).pack(side="left", padx=16)

        # Row 4: Ollama settings (shown only in LLM mode)
        self._ollama_frame = ttk.Frame(param_frame)
        self._ollama_frame.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(4, 0))

        # Row 0: host + port + refresh button
        ttk.Label(self._ollama_frame, text="Ollama host:").grid(row=0, column=0, sticky="w", padx=4)
        self._host_var = tk.StringVar(value="localhost")
        host_entry = ttk.Entry(self._ollama_frame, textvariable=self._host_var, width=18)
        host_entry.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(self._ollama_frame, text="Port:").grid(row=0, column=2, sticky="w", padx=(12, 4))
        self._port_var = tk.StringVar(value="11434")
        ttk.Entry(self._ollama_frame, textvariable=self._port_var, width=7).grid(
            row=0, column=3, sticky="w", padx=4)

        self._fetch_btn = ttk.Button(self._ollama_frame, text="↻ Fetch models",
                                     command=self._fetch_models, width=14)
        self._fetch_btn.grid(row=0, column=4, sticky="w", padx=(12, 4))

        self._model_status = ttk.Label(self._ollama_frame, text="", foreground="gray",
                                       font=("TkDefaultFont", 9))
        self._model_status.grid(row=0, column=5, sticky="w", padx=4)
        self._ollama_frame.columnconfigure(5, weight=1)

        # Row 1: model listbox (multi-select)
        ttk.Label(self._ollama_frame, text="Models:").grid(
            row=1, column=0, sticky="nw", padx=4, pady=(6, 0))
        lb_frame = ttk.Frame(self._ollama_frame)
        lb_frame.grid(row=1, column=1, columnspan=5, sticky="ew", pady=(6, 0))
        lb_frame.columnconfigure(0, weight=1)

        self._model_lb = tk.Listbox(lb_frame, selectmode="multiple", height=4,
                                    exportselection=False, font=("TkFixedFont", 9))
        self._model_lb.grid(row=0, column=0, sticky="ew")
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self._model_lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._model_lb.configure(yscrollcommand=sb.set)

        ttk.Label(lb_frame,
                  text="Ctrl-click to select multiple. Click ↻ to load from Ollama.",
                  foreground="gray", font=("TkDefaultFont", 9),
                  ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # ── System prompt (LLM mode only) ─────────────────────────────────────
        self._prompt_frame = ttk.LabelFrame(self, text="System Prompt  (LLM mode)", padding=6)
        self._prompt_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 4))
        self._prompt_frame.columnconfigure(0, weight=1)

        prompt_bar = ttk.Frame(self._prompt_frame)
        prompt_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(prompt_bar,
                  text="Leave blank to send no system prompt  —  empty = LLM uses its own defaults",
                  foreground="gray", font=("TkDefaultFont", 9)).pack(side="left")
        ttk.Button(prompt_bar, text="Reset to default",
                   command=self._reset_prompt, width=16).pack(side="right")

        self._prompt_text = scrolledtext.ScrolledText(
            self._prompt_frame, height=3, wrap="word", font=("TkDefaultFont", 9))
        self._prompt_text.insert("1.0", self._DEFAULT_PROMPT)
        self._prompt_text.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(8, 4))
        btn_frame.grid(row=3, column=0, sticky="ew")

        self._btn_run = ttk.Button(btn_frame, text="Run Evaluation",
                                   command=self._start, width=18)
        self._btn_run.pack(side="left", padx=4)

        self._btn_compare = ttk.Button(btn_frame, text="Compare (diff)",
                                       command=self._run_compare, width=16)
        self._btn_compare.pack(side="left", padx=4)

        self._btn_stop = ttk.Button(btn_frame, text="Stop",
                                    command=self._request_stop,
                                    state="disabled", width=8)
        self._btn_stop.pack(side="left", padx=4)

        self._status_lbl = ttk.Label(btn_frame, text="", foreground="gray")
        self._status_lbl.pack(side="right", padx=8)

        # ── Results + Log ─────────────────────────────────────────────────────
        results_frame = ttk.LabelFrame(self, text="Results", padding=8)
        results_frame.grid(row=4, column=0, sticky="nsew", padx=8, pady=4)
        results_frame.columnconfigure(0, weight=1)

        # Treeview for metric table
        cols = ("system", "n", "hit1", "hit3", "hit5", "hit10", "mrr10")
        self._tree = ttk.Treeview(results_frame, columns=cols,
                                  show="headings", height=5)
        headers = {
            "system": ("System", 240),
            "n":      ("N",       50),
            "hit1":   ("Hit@1",   70),
            "hit3":   ("Hit@3",   70),
            "hit5":   ("Hit@5",   70),
            "hit10":  ("Hit@10",  70),
            "mrr10":  ("MRR@10",  70),
        }
        for cid, (head, width) in headers.items():
            self._tree.heading(cid, text=head)
            self._tree.column(cid, width=width,
                              anchor="center" if cid != "system" else "w",
                              stretch=(cid == "system"))
        self._tree.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        # Progress bar under the table
        self._pb = ttk.Progressbar(results_frame, mode="determinate")
        self._pb.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        # Log
        ttk.Label(results_frame, text="Log:").grid(row=2, column=0, sticky="w")
        self._log = scrolledtext.ScrolledText(
            results_frame, height=10, state="disabled",
            wrap="word", font=("TkFixedFont", 9),
        )
        self._log.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        results_frame.rowconfigure(3, weight=1)

        self._on_mode_change()   # set initial visibility after widgets exist

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        show = self._mode_var.get() == "llm"
        state = "normal" if show else "disabled"
        for child in self._ollama_frame.winfo_children():
            if isinstance(child, (ttk.Label, ttk.Frame, tk.Frame)):
                continue
            child.configure(state=state)
        self._model_lb.configure(state=state)
        # Show/hide the system prompt frame
        if show:
            self._prompt_frame.grid()
        else:
            self._prompt_frame.grid_remove()
        if show:
            for cid, head in [("hit1", "Accuracy"), ("hit3", "Ctx Tok"),
                               ("hit5", "Helped%"),  ("hit10", "Hurt%"), ("mrr10", "")]:
                self._tree.heading(cid, text=head)
        else:
            for cid, head in [("hit1", "Hit@1"), ("hit3", "Hit@3"),
                               ("hit5", "Hit@5"), ("hit10", "Hit@10"), ("mrr10", "MRR@10")]:
                self._tree.heading(cid, text=head)

    def _update_n_label(self) -> None:
        try:
            n = self._n_var.get()
        except Exception:
            return
        total = getattr(self, "_dataset_total", None)
        total_str = f"of {total:,}" if total else "of ?"
        self._n_count_lbl.configure(text=f"{n:,} {total_str} available")

    def _on_dataset_change(self) -> None:
        path_str = self._dataset_var.get().strip()
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            return

        def _count():
            try:
                if path.suffix.lower() in (".jsonl", ".json"):
                    count = sum(1 for ln in open(path, encoding="utf-8") if ln.strip())
                else:
                    import csv as _csv
                    with open(path, newline="", encoding="utf-8") as f:
                        count = sum(1 for _ in _csv.DictReader(f))
            except Exception:
                count = None
            self.after(0, lambda: self._apply_dataset_count(count))

        threading.Thread(target=_count, daemon=True).start()

    def _apply_dataset_count(self, count: int | None) -> None:
        self._dataset_total = count
        if count:
            self._n_scale.configure(to=count)
            if self._n_var.get() > count:
                self._n_var.set(count)
        self._update_n_label()

    def _reset_prompt(self) -> None:
        self._prompt_text.delete("1.0", "end")
        self._prompt_text.insert("1.0", self._DEFAULT_PROMPT)

    # ── Model fetcher ─────────────────────────────────────────────────────────

    _EMBED_PATTERNS = ("embed", "minilm", "e5-", "bge-", "gte-", "nomic")

    def _fetch_models(self) -> None:
        host = self._host_var.get().strip() or "localhost"
        try:
            port = int(self._port_var.get().strip() or "11434")
        except ValueError:
            self._model_status.configure(text="Bad port", foreground="red")
            return

        self._fetch_btn.configure(state="disabled")
        self._model_status.configure(text="Fetching…", foreground="gray")

        def _do_fetch():
            try:
                import requests as _rq
                resp = _rq.get(f"http://{host}:{port}/api/tags", timeout=5)
                resp.raise_for_status()
                models_raw = resp.json().get("models", [])
                names = []
                for m in models_raw:
                    name = m.get("name", "")
                    low  = name.lower()
                    if any(p in low for p in self._EMBED_PATTERNS):
                        continue
                    size_gb = m.get("size", 0) / 1e9
                    label = f"{name}  ({size_gb:.1f} GB)" if size_gb else name
                    names.append((name, label))
                self.after(0, lambda: self._populate_models(names, None))
            except Exception as e:
                self.after(0, lambda: self._populate_models(None, str(e)))

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _populate_models(self, names: list | None, error: str | None) -> None:
        self._fetch_btn.configure(state="normal")
        if error:
            self._model_status.configure(text=f"Error: {error}", foreground="red")
            return
        self._model_lb.delete(0, "end")
        for real_name, _label in (names or []):
            self._model_lb.insert("end", real_name)
        count = len(names or [])
        self._model_status.configure(
            text=f"{count} model{'s' if count != 1 else ''} found",
            foreground="green" if count else "orange",
        )

    # ── File pickers ──────────────────────────────────────────────────────────

    def _browse_dataset(self) -> None:
        p = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[("JSONL / JSON", "*.jsonl *.json"),
                       ("CSV", "*.csv"), ("All files", "*")],
        )
        if p:
            self._dataset_var.set(p)

    def _browse_struct(self) -> None:
        p = filedialog.askdirectory(title="Select structured index directory")
        if p:
            self._struct_var.set(p)
            cfg["last_index_dir"] = p
            cfg.save()

    def _browse_flat(self) -> None:
        p = filedialog.askdirectory(title="Select flat index directory")
        if p:
            self._flat_var.set(p)

    def _browse_out(self) -> None:
        p = filedialog.askdirectory(title="Select output folder for CSVs")
        if p:
            self._out_var.set(p)

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate(self) -> bool:
        dataset = self._dataset_var.get().strip()
        struct  = self._struct_var.get().strip()
        if not dataset:
            self._log_append("[error] No dataset selected.\n")
            return False
        if not Path(dataset).exists():
            self._log_append(f"[error] Dataset not found: {dataset}\n")
            return False
        if not struct:
            self._log_append("[error] No structured index directory selected.\n")
            return False
        if not Path(struct).exists():
            self._log_append(f"[error] Structured index not found: {struct}\n")
            return False
        try:
            int(self._topk_var.get())
        except ValueError:
            self._log_append("[error] Top-K must be an integer.\n")
            return False
        if self._mode_var.get() == "llm":
            if not self._model_lb.curselection():
                self._log_append("[error] No models selected. Click ↻ Fetch models, then select at least one.\n")
                return False
            try:
                int(self._port_var.get().strip())
            except ValueError:
                self._log_append("[error] Port must be an integer.\n")
                return False
        return True

    # ── Run evaluation ────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._log_clear()
        if not self._validate():
            return

        dataset   = Path(self._dataset_var.get().strip())
        struct    = Path(self._struct_var.get().strip())
        flat_str  = self._flat_var.get().strip()
        flat      = Path(flat_str) if flat_str else None
        out_dir   = Path(self._out_var.get().strip() or "results")
        n         = self._n_var.get()
        top_k     = int(self._topk_var.get())
        mode      = self._mode_var.get()
        shuffle   = self._shuffle_var.get()
        try:
            seed = int(self._seed_var.get())
        except ValueError:
            seed = 42

        # Clear results table
        for row in self._tree.get_children():
            self._tree.delete(row)

        self._pb["value"] = 0
        self._set_running(True)
        stop_event = threading.Event()
        self._stop_event = stop_event
        q = self._q

        if mode == "retrieval":
            self._run_retrieval(dataset, struct, flat, out_dir, n, top_k,
                                shuffle, seed, stop_event, q)
        else:
            host          = self._host_var.get().strip() or "localhost"
            port          = int(self._port_var.get().strip() or "11434")
            models        = [self._model_lb.get(i) for i in self._model_lb.curselection()]
            system_prompt = self._prompt_text.get("1.0", "end").strip()
            raw_budgets   = self._budgets_var.get().split()
            try:
                ctx_budgets = [int(b) for b in raw_budgets]
            except ValueError:
                ctx_budgets = [128, 512, 0]
            self._run_llm(dataset, struct, flat, out_dir, n, top_k,
                          host, port, models, system_prompt, ctx_budgets,
                          shuffle, seed, stop_event, q)

    def _run_retrieval(self, dataset, struct, flat, out_dir,
                       n, top_k, shuffle, seed, stop_event, q) -> None:
        def _worker():
            try:
                import sys, random, re as _re
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from evaluate import load_dataset, compute_metrics, export_csv
                from indexer.query import search

                q.put(("log", f"Loading dataset: {dataset.name}\n"))
                questions = load_dataset(dataset, 999_999)
                if shuffle:
                    random.Random(seed).shuffle(questions)
                    q.put(("log", f"  Shuffled with seed={seed}\n"))
                questions = questions[:n]
                q.put(("log", f"  {len(questions)} questions loaded\n"))

                base_cfg: dict = {
                    "use_faiss":           True,
                    "use_title_bm25":      True,
                    "use_para_bm25":       True,
                    "use_diversity_cap":   False,  # hand-tuned, disabled for reproducibility
                    "use_mention_penalty": False,  # hand-tuned, disabled for reproducibility
                    "use_nav_boost":       False,  # hand-tuned, disabled for reproducibility
                    "use_lead_augment":    False,
                    "eval_rrf_k":          60,
                    "eval_diversity_max":  6,
                    **cfg.as_engine_cfg(),
                }

                systems_to_run = []
                bm25_cfg = {**base_cfg, "use_faiss": False,
                            "use_lead_augment": False, "use_section_augment": False}
                systems_to_run.append(("BM25 Only", struct, bm25_cfg))
                if flat and flat.exists():
                    systems_to_run.append(("Hybrid Flat", flat,
                        {**base_cfg, "use_lead_augment": False, "use_section_augment": False}))
                systems_to_run.append(("Hybrid Structured", struct,
                    {**base_cfg, "use_lead_augment": False, "use_section_augment": False}))
                systems_to_run.append(("Struct+Lead", struct,
                    {**base_cfg, "use_lead_augment": True,  "use_section_augment": False}))
                systems_to_run.append(("Struct+Section", struct,
                    {**base_cfg, "use_lead_augment": False, "use_section_augment": True}))
                systems_to_run.append(("Struct+Both", struct,
                    {**base_cfg, "use_lead_augment": True,  "use_section_augment": True}))

                all_results = []
                agg_metrics = []
                total_systems = len(systems_to_run)

                for si, (label, index_dir, sys_cfg) in enumerate(systems_to_run):
                    if stop_event.is_set():
                        break
                    q.put(("log", f"\nRunning: {label}\n"))

                    nq = len(questions)
                    results = []
                    t0 = time.time()

                    for i, question in enumerate(questions):
                        if stop_event.is_set():
                            break
                        try:
                            hits = search(index_dir, question["question"],
                                          top_k=top_k, cfg=sys_cfg)
                        except Exception as e:
                            hits = []
                            q.put(("log", f"  [warn] q{i+1}: {e}\n"))

                        ans_words = set(_re.findall(
                            r'\b\w{4,}\b',
                            (question.get("correct_text") or "").lower()))
                        rank = None
                        for rk, hit in enumerate(hits, 1):
                            title_words = set(_re.findall(
                                r'\b\w{4,}\b',
                                (hit.get("title") or "").lower()))
                            if ans_words & title_words:
                                rank = rk
                                break

                        results.append({
                            "idx":          i,
                            "question":     question["question"],
                            "correct_key":  question["correct_key"],
                            "correct_text": question.get("correct_text", ""),
                            "rank":         rank,
                            "hits":         hits,
                        })

                        pct = int((si * nq + i + 1) / (total_systems * nq) * 100)
                        q.put(("progress", pct))
                        if (i + 1) % 20 == 0:
                            elapsed = time.time() - t0
                            rate = (i + 1) / max(elapsed, 0.1)
                            eta = (nq - i - 1) / rate
                            q.put(("log",
                                   f"  {i+1}/{nq}  {rate:.1f} q/s  "
                                   f"ETA {int(eta//60)}m{int(eta%60):02d}s\n"))

                    metrics = compute_metrics(results, top_k)
                    all_results.append((label, results))
                    agg_metrics.append((label, metrics))
                    q.put(("result", label, metrics))
                    q.put(("log",
                           f"  {label}: "
                           f"Hit@1={metrics['hit@1']:.3f}  "
                           f"Hit@3={metrics['hit@3']:.3f}  "
                           f"Hit@5={metrics['hit@5']:.3f}  "
                           f"Hit@10={metrics['hit@10']:.3f}  "
                           f"MRR@10={metrics['mrr@10']:.4f}\n"))

                if not stop_event.is_set() and all_results:
                    stem = dataset.stem + "_results"
                    out_path = out_dir / f"{stem}.csv"
                    export_csv(out_path, agg_metrics, all_results)
                    q.put(("log", f"\nSaved → {out_path}\n"))

                q.put(("done",) if not stop_event.is_set() else ("stopped",))

            except Exception:
                q.put(("error", traceback.format_exc()))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def _run_llm(self, dataset, struct, flat, out_dir, n, top_k,
                 host, port, models, system_prompt, ctx_budgets,
                 shuffle, seed, stop_event, q) -> None:
        def _worker():
            try:
                import sys, random
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from infer import (load_dataset, accuracy, export_csv,
                                   _build_prompt, ollama_generate, _parse_letter)
                from indexer.query import search as _search

                q.put(("log", f"Loading dataset: {dataset.name}\n"))
                questions = load_dataset(dataset, 999_999)
                if shuffle:
                    random.Random(seed).shuffle(questions)
                    q.put(("log", f"  Shuffled with seed={seed}\n"))
                questions = questions[:n]
                nq = len(questions)
                q.put(("log", f"  {nq} questions loaded\n"))

                _base_rag = {
                    "use_faiss": True, "use_title_bm25": True, "use_para_bm25": True,
                    "use_diversity_cap": False,   # hand-tuned, disabled for reproducibility
                    "use_mention_penalty": False,  # hand-tuned, disabled for reproducibility
                    "use_nav_boost": False,        # hand-tuned, disabled for reproducibility
                    "eval_rrf_k": 60, "eval_diversity_max": 6,
                    **cfg.as_engine_cfg(),
                }
                all_conditions = [
                    ("No Retrieval",   None,   {}),
                    ("Flat RAG",       flat,   {**_base_rag, "use_lead_augment": False, "use_section_augment": False}),
                    ("Struct RAG",     struct, {**_base_rag, "use_lead_augment": False, "use_section_augment": False}),
                    ("Struct+Lead",    struct, {**_base_rag, "use_lead_augment": True,  "use_section_augment": False}),
                    ("Struct+Section", struct, {**_base_rag, "use_lead_augment": False, "use_section_augment": True}),
                    ("Struct+Both",    struct, {**_base_rag, "use_lead_augment": True,  "use_section_augment": True}),
                ]
                conditions = [(l, idx, c) for l, idx, c in all_conditions
                              if idx is None or (idx is not None and idx.exists())]

                # Resume: detect already-completed labels in existing CSV
                import csv as _csv
                out_dir.mkdir(parents=True, exist_ok=True)
                stem          = dataset.stem + "_infer"
                out_path      = out_dir / f"{stem}.csv"
                done_labels: set[str] = set()
                if out_path.exists():
                    try:
                        with open(out_path, newline="", encoding="utf-8") as _f:
                            _cols = next(_csv.reader(_f), [])
                        done_labels = {c[:-3] for c in _cols if c.endswith("_ok")}
                        if done_labels:
                            q.put(("log", f"  Resuming — already done: {sorted(done_labels)}\n"))
                            print(f"  Resuming — already done: {sorted(done_labels)}", flush=True)
                    except Exception:
                        pass

                budgets     = sorted(set(ctx_budgets))
                budget_strs = [f"{b}tok" if b > 0 else "unlimited" for b in budgets]
                q.put(("log", f"  Conditions to run: {[l for l,_,_ in conditions]}\n"))
                q.put(("log", f"  Context budgets: {budget_strs}\n"))
                q.put(("log", f"  System prompt: {repr(system_prompt[:80]) if system_prompt else '(none)'}\n"))

                total_steps = len(models) * len(conditions) * len(budgets) * nq
                all_results: list = []
                agg_systems: list = []
                step = 0

                for model in models:
                    if stop_event.is_set():
                        break
                    short = model.replace(":", "-")

                    # Warm-up: load model into RAM before measurement begins
                    q.put(("log", f"\nWarming up {model}…\n"))
                    ollama_generate(
                        "Answer with a single letter A, B, C or D.\nAnswer:",
                        model, host, port, timeout=180,
                    )
                    q.put(("log", "  Model ready.\n"))

                    for budget in budgets:
                        if stop_event.is_set():
                            break
                        max_ctx = budget if budget > 0 else None
                        budget_tag = f"{budget}tok" if budget > 0 else "unlimited"
                        no_rag_correct: list[bool] | None = None

                        for cond_name, idx_dir, cond_cfg in conditions:
                                label = f"{short} / {cond_name} [{budget_tag}]"
                                if label in done_labels:
                                    q.put(("log", f"  [skip] {label} — already in CSV\n"))
                                    print(f"  [skip] {label}", flush=True)
                                    step += nq
                                    q.put(("progress", int(step / max(total_steps, 1) * 100)))
                                    continue
                                if stop_event.is_set():
                                    break
                                q.put(("log", f"\nRunning: {label}\n"))

                                # Print retrieval config to terminal only
                                _sep = "─" * 60
                                _flag_keys = [
                                    "use_faiss", "use_title_bm25", "use_para_bm25",
                                    "use_diversity_cap", "use_mention_penalty", "use_nav_boost",
                                    "use_lead_augment", "use_section_augment",
                                    "eval_rrf_k", "eval_diversity_max",
                                ]
                                _flags = "  ".join(
                                    f"{k.replace('use_','').replace('eval_','')}="
                                    f"{'ON' if v is True else ('OFF' if v is False else v)}"
                                    for k, v in cond_cfg.items() if k in _flag_keys
                                )
                                print(f"\n{_sep}", flush=True)
                                print(f"CONDITION: {label}", flush=True)
                                print(f"CTX BUDGET:{budget_tag}", flush=True)
                                print(f"INDEX:     {idx_dir or 'none'}", flush=True)
                                print(f"FLAGS:     {_flags}", flush=True)
                                print(_sep, flush=True)

                                results = []
                                total_ctx_chars = 0

                                for i, question in enumerate(questions):
                                    if stop_event.is_set():
                                        break
                                    hits = []
                                    if idx_dir is not None:
                                        try:
                                            hits = _search(idx_dir, question["question"],
                                                           top_k=top_k, cfg=cond_cfg)
                                        except Exception as e:
                                            q.put(("log", f"  [warn] q{i+1}: {e}\n"))

                                    prompt, ctx_chars = _build_prompt(
                                        question, hits or None,
                                        max_ctx_tokens=max_ctx,
                                        system_prompt=system_prompt or None,
                                    )
                                    total_ctx_chars += ctx_chars

                                    # Print question + context to terminal
                                    print(f"\n  Q{i+1}: {question['question'][:80]}", flush=True)
                                    if ctx_chars > 0:
                                        ctx_start = prompt.find("Use the following context")
                                        ctx_end   = prompt.find("\nQuestion:")
                                        if ctx_start != -1 and ctx_end != -1:
                                            ctx_block = prompt[ctx_start:ctx_end].strip()
                                            # Count how many chunks made it in
                                            import re as _re2
                                            n_used = len(_re2.findall(r'^\[\d+\]', ctx_block, _re2.MULTILINE))
                                            budget_label = f"{max_ctx}tok" if max_ctx else "unlimited"
                                            print(f"  CONTEXT  budget={budget_label}"
                                                  f"  used={n_used}/{len(hits)} chunks"
                                                  f"  {ctx_chars} chars (~{ctx_chars//4} tok)", flush=True)
                                            for ri, h in enumerate(hits[:n_used], 1):
                                                snippet = (h.get("text") or "").replace("\n", " ")[:80]
                                                print(f"    [{ri}] rrf={h.get('rrf_score',0):.4f}"
                                                      f"  {h.get('title','?')!r}"
                                                      f"  §{h.get('section_title','?')!r}"
                                                      f"  \"{snippet}\"", flush=True)
                                            print(ctx_block, flush=True)
                                    else:
                                        print(f"  CONTEXT  none (no retrieval)", flush=True)
                                    gen       = ollama_generate(prompt, model, host, port)
                                    predicted = _parse_letter(gen["response"] or "") if gen["response"] else None
                                    correct   = predicted == question["correct_key"] if predicted else False
                                    mark = "✓" if correct else "✗"
                                    results.append({
                                        "idx": i, "question": question["question"],
                                        "correct_key": question["correct_key"],
                                        "correct_text": question.get("correct_text", ""),
                                        "predicted": predicted or "", "correct": correct,
                                        "raw_response": (gen["response"] or "").strip()[:80],
                                        "total_s":   round(gen["total_s"],   3),
                                        "prefill_s": round(gen["prefill_s"], 3),
                                        "gen_s":     round(gen["gen_s"],     3),
                                    })
                                    raw_snippet = (gen["response"] or "").strip().replace("\n", " ")[:60]
                                    top_titles = " | ".join(
                                        h.get("title", "?")[:30] for h in (hits or [])[:3]
                                    ) or "—"
                                    q_line = (f"  q{i+1:>3}/{nq}  {mark}"
                                              f"  pred={predicted or '?'}"
                                              f"  ans={question['correct_key']}"
                                              f"  gen={gen['gen_s']:.1f}s"
                                              f"  tok={gen['gen_tokens']}"
                                              f"  ctx=[{top_titles}]"
                                              f'  raw="{raw_snippet}"\n')
                                    q.put(("log", q_line))
                                    print(q_line, end="", flush=True)
                                    step += 1
                                    q.put(("progress", int(step / max(total_steps, 1) * 100)))
                                    if (i + 1) % 10 == 0:
                                        acc_so_far = sum(r["correct"] for r in results) / len(results)
                                        summary = f"  ── {i+1}/{nq}  running acc={acc_so_far:.3f}\n"
                                        q.put(("log", summary))
                                        print(summary, end="", flush=True)

                                acc     = accuracy(results)
                                avg_tok = int(total_ctx_chars / max(nq, 1) / 4)

                                # Latency stats from collected timings
                                import statistics as _st
                                prefill_times = [r["prefill_s"] for r in results]
                                gen_times     = [r["gen_s"]     for r in results]

                                def _pct(lst, p):
                                    if not lst:
                                        return 0.0
                                    s = sorted(lst)
                                    idx = (len(s) - 1) * p / 100
                                    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
                                    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

                                mean_pre = _st.mean(prefill_times) if prefill_times else 0.0
                                mean_gen = _st.mean(gen_times)     if gen_times     else 0.0
                                std_pre  = _st.stdev(prefill_times) if len(prefill_times) > 1 else 0.0
                                std_gen  = _st.stdev(gen_times)     if len(gen_times)     > 1 else 0.0
                                p50_gen  = _pct(gen_times, 50)
                                p95_gen  = _pct(gen_times, 95)

                                # Context helped/hurt vs no-retrieval baseline
                                helped_pct = hurt_pct = None
                                if no_rag_correct is not None and len(results) == len(no_rag_correct):
                                    helped = sum(1 for j, r in enumerate(results)
                                                 if r["correct"] and not no_rag_correct[j])
                                    hurt   = sum(1 for j, r in enumerate(results)
                                                 if not r["correct"] and no_rag_correct[j])
                                    helped_pct = helped / nq * 100
                                    hurt_pct   = hurt   / nq * 100

                                if cond_name == "No Retrieval":
                                    no_rag_correct = [r["correct"] for r in results]

                                for r in results:
                                    r["avg_ctx_tokens"] = avg_tok
                                all_results.append((label, results))
                                agg_systems.append({
                                    "label":          label,
                                    "acc":            acc,
                                    "n_correct":      sum(r["correct"] for r in results),
                                    "n_total":        nq,
                                    "avg_ctx_tokens": avg_tok,
                                })
                                q.put(("infer_result", label, acc, nq, avg_tok, helped_pct, hurt_pct))
                                _hlp = f"+{helped_pct:.1f}%" if helped_pct is not None else "+?%"
                                _hrt = f"-{hurt_pct:.1f}%"  if hurt_pct  is not None else "-?%"
                                summary_lines = (
                                    f"\n── {label} ──\n"
                                    f"  acc={acc:.3f}  ctx≈{avg_tok}tok  {_hlp}  {_hrt}\n"
                                    f"  prefill {mean_pre:.2f}±{std_pre:.2f}s"
                                    f"  gen {mean_gen:.2f}±{std_gen:.2f}s"
                                    f"  gen p50={p50_gen:.2f}s p95={p95_gen:.2f}s\n"
                                )
                                q.put(("log", summary_lines))
                                print(summary_lines, end="", flush=True)

                if all_results:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem     = dataset.stem + "_infer"
                    out_path = out_dir / f"{stem}.csv"
                    export_csv(out_path, agg_systems, all_results)
                    if stop_event.is_set():
                        q.put(("log", f"\nPartial results saved ({len(all_results)} condition(s)) → {out_path}\n"))
                    else:
                        q.put(("log", f"\nSaved → {out_path}\n"))

                q.put(("done",) if not stop_event.is_set() else ("stopped",))

            except Exception:
                q.put(("error", traceback.format_exc()))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    # ── Compare ───────────────────────────────────────────────────────────────

    def _run_compare(self) -> None:
        out_dir = Path(self._out_var.get().strip() or "results")
        dataset = Path(self._dataset_var.get().strip())
        if dataset.name:
            stem = dataset.stem + "_results.csv"
            candidate = out_dir / stem
        else:
            candidate = None

        csv_file = filedialog.askopenfilename(
            title="Select per-question results CSV",
            initialdir=str(out_dir),
            initialfile=str(candidate.name) if candidate else "",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not csv_file:
            return

        self._log_clear()
        self._log_append(f"Comparing: {Path(csv_file).name}\n\n")

        try:
            import csv as _csv
            with open(csv_file, newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            if not rows:
                self._log_append("[error] CSV is empty.\n")
                return

            # Find binary outcome columns — retrieval uses _hit@10, infer uses _ok
            hit_cols = [c for c in rows[0].keys() if c.endswith("_hit@10")]
            ok_cols  = [c for c in rows[0].keys() if c.endswith("_ok")]
            suffix   = "_hit@10" if len(hit_cols) >= 2 else ("_ok" if len(ok_cols) >= 2 else None)
            if suffix is None:
                self._log_append(
                    "[error] Need at least 2 systems with _hit@10 or _ok columns.\n"
                    f"  Found hit@10: {hit_cols}  ok: {ok_cols}\n"
                )
                return
            hit_cols = hit_cols if suffix == "_hit@10" else ok_cols

            # Default: last two systems (Struct+Lead vs Flat or BM25)
            col_a = hit_cols[-1]
            col_b = hit_cols[0] if hit_cols[0] != col_a else hit_cols[1]
            sys_a = col_a.replace(suffix, "")
            sys_b = col_b.replace(suffix, "")

            n     = len(rows)
            a_wins  = [r for r in rows if r[col_a] == "1" and r[col_b] == "0"]
            b_wins  = [r for r in rows if r[col_b] == "1" and r[col_a] == "0"]
            both    = [r for r in rows if r[col_a] == "1" and r[col_b] == "1"]
            neither = [r for r in rows if r[col_a] == "0" and r[col_b] == "0"]

            self._log_append(
                f"System A: {sys_a}\n"
                f"System B: {sys_b}\n"
                f"{'─'*60}\n"
                f"Both hit    : {len(both):4d}  ({len(both)/n:.1%})\n"
                f"A only      : {len(a_wins):4d}  ({len(a_wins)/n:.1%})  ← {sys_a} advantage\n"
                f"B only      : {len(b_wins):4d}  ({len(b_wins)/n:.1%})  ← {sys_b} advantage\n"
                f"Neither hit : {len(neither):4d}  ({len(neither)/n:.1%})\n"
                f"{'─'*60}\n"
            )

            def _show(label, cases, limit=8):
                if not cases:
                    return
                self._log_append(f"\n── {label} ({len(cases)} cases, showing {min(limit, len(cases))}) ──\n")
                for r in cases[:limit]:
                    # retrieval CSVs have _rank; infer CSVs have _pred
                    extra_a = r.get(f"{sys_a}_rank") or r.get(f"{sys_a}_pred") or "?"
                    extra_b = r.get(f"{sys_b}_rank") or r.get(f"{sys_b}_pred") or "?"
                    self._log_append(
                        f"\n  [{r['num']:>3}] {r['question'][:90]}\n"
                        f"       Answer: {r['correct_key']} — {r.get('correct_text','')[:55]}\n"
                        f"       {sys_a}: {extra_a}   {sys_b}: {extra_b}\n"
                    )

            _show(f"{sys_a} HIT / {sys_b} MISS  (structure helped)", a_wins)
            _show(f"{sys_b} HIT / {sys_a} MISS  (flat beat structure)", b_wins)

        except Exception:
            self._log_append(f"\n[error]\n{traceback.format_exc()}\n")

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log_append(msg[1])
                elif kind == "progress":
                    self._pb["value"] = msg[1]
                elif kind == "result":
                    # retrieval mode result row
                    _, label, m = msg
                    self._tree.insert("", "end", values=(
                        label, m["n"],
                        f"{m['hit@1']:.3f}", f"{m['hit@3']:.3f}",
                        f"{m['hit@5']:.3f}", f"{m['hit@10']:.3f}",
                        f"{m['mrr@10']:.4f}",
                    ))
                elif kind == "infer_result":
                    _, label, acc, n, avg_tok, helped_pct, hurt_pct = msg
                    tok_str  = f"~{avg_tok}" if avg_tok else "—"
                    hlp_str  = f"{helped_pct:.1f}%" if helped_pct is not None else "—"
                    hrt_str  = f"{hurt_pct:.1f}%"  if hurt_pct  is not None else "—"
                    self._tree.insert("", "end", values=(
                        label, n, f"{acc:.3f}", tok_str, hlp_str, hrt_str, ""
                    ))
                elif kind == "done":
                    self._pb["value"] = 100
                    self._log_append("\n✓ Evaluation complete.\n")
                    self._set_running(False)
                    self._status_lbl.configure(text="Done", foreground="green")
                elif kind == "stopped":
                    self._log_append("\n— Stopped by user.\n")
                    self._set_running(False)
                    self._status_lbl.configure(text="Stopped", foreground="gray")
                elif kind == "error":
                    self._log_append(f"\n[ERROR]\n{msg[1]}\n")
                    self._set_running(False)
                    self._status_lbl.configure(text="Error", foreground="red")
        except queue.Empty:
            pass
        self.after(150, self._poll)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _request_stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self._btn_stop.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state_run  = "disabled" if running else "normal"
        state_stop = "normal"   if running else "disabled"
        self._btn_run.configure(state=state_run)
        self._btn_compare.configure(state=state_run)
        self._btn_stop.configure(state=state_stop)
        if running:
            self._status_lbl.configure(text="Running…", foreground="blue")

    def _log_append(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_clear(self) -> None:
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._status_lbl.configure(text="")
        for row in self._tree.get_children():
            self._tree.delete(row)
        self._pb["value"] = 0
