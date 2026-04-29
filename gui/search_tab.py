"""Search tab — single query + golden-dataset evaluation with retrieval ablation toggles."""
from __future__ import annotations

import csv
import queue
import traceback
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import ttk, filedialog, scrolledtext

from gui.config import cfg

# ── Toggle metadata ───────────────────────────────────────────────────────────

_TOGGLES = [
    ("use_faiss",          "FAISS (Dense)",
     "Semantic similarity via FAISS cosine search. Finds relevant chunks even "
     "when the query uses different wording than the source text. Core signal."),
    ("use_title_bm25",     "Title BM25",
     "Keyword search over article titles using BM25 ranking. Strong when the "
     "query names a specific condition, drug, or procedure directly."),
    ("use_para_bm25",      "Paragraph BM25",
     "Keyword search over paragraph text (not just titles). Complements title "
     "BM25 by matching content inside articles, not only their headings."),
    ("use_diversity_cap",  "Diversity Cap",
     "Limits how many chunks from the same article enter the candidate pool "
     "(default: 3 per article). Prevents one dominant article from crowding "
     "out others before scoring."),
    ("use_nav_boost",      "Navigational Boost",
     "Boosts articles whose title has ≥60% word overlap with the query. Helps "
     "when the query names a specific entity (e.g. 'Metformin mechanism'). "
     "Lead sections receive an extra ×1.15 boost."),
    ("use_mention_penalty","Mention Penalty",
     "Penalises chunks where query terms appear only once (×0.82) or only in "
     "the second half of the text (×0.88). Favours chunks where terms are "
     "prominent and repeated. Ported from AIIAB retrieval pipeline."),
    ("use_lead_augment",    "Lead Augment",
     "Appends the article's lead section to each non-lead retrieved chunk. "
     "Provides article-level context to the LLM alongside the specific chunk. "
     "Adds a 'lead_context' field — does not affect Hit@K retrieval metrics."),
    ("use_section_augment", "Section Augment",
     "Appends the section's opening paragraph to non-first-paragraph chunks. "
     "Wikipedia style mandates the first paragraph summarises the section, so "
     "this injects a section-level 'abstract' around deeper hits. "
     "Adds a 'section_context' field — does not affect Hit@K retrieval metrics."),
]


class SearchTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self._q:        queue.Queue      = queue.Queue()
        self._stop_evt: threading.Event  = threading.Event()
        self._hits:     list[dict]       = []
        self._eval_rows: list[dict]      = []   # per-question results
        self._golden:   list[dict]       = []   # loaded CSV rows
        self._toggle_vars: dict[str, tk.BooleanVar] = {}
        self._build_ui()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Shared top: index dir + Kiwix URL ────────────────────────────────
        top = ttk.LabelFrame(self, text="Index", padding=8)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Index dir:").grid(row=0, column=0, sticky="w")
        self._dir_var = tk.StringVar(value=cfg.get("last_index_dir", ""))
        ttk.Entry(top, textvariable=self._dir_var).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse…", command=self._browse_dir).grid(
            row=0, column=2, sticky="e")
        self._dir_var.trace_add("write", lambda *_: self._on_dir_change())

        ttk.Label(top, text="Kiwix viewer:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._kiwix_var = tk.StringVar(value=cfg.get("kiwix_base_url", "http://127.0.0.1/kiwix/viewer"))
        ttk.Entry(top, textvariable=self._kiwix_var).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        self._kiwix_var.trace_add("write", lambda *_: self._save_kiwix())

        ttk.Label(top, text="Book name:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._book_var = tk.StringVar(value=cfg.get("kiwix_book_name", ""))
        self._book_entry = ttk.Entry(top, textvariable=self._book_var)
        self._book_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Label(top, text="(auto from dir name if blank)",
                  foreground="gray", font=("TkDefaultFont", 8)).grid(
            row=2, column=2, sticky="w", padx=4, pady=(4, 0))
        self._book_var.trace_add("write", lambda *_: self._save_kiwix())

        # ── Retrieval config ──────────────────────────────────────────────────
        rcfg = ttk.LabelFrame(self, text="Retrieval Configuration", padding=8)
        rcfg.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        rcfg.columnconfigure(0, weight=1)

        toggle_frame = ttk.Frame(rcfg)
        toggle_frame.grid(row=0, column=0, sticky="ew")

        for col, (key, label, _desc) in enumerate(_TOGGLES):
            var = tk.BooleanVar(value=cfg.get(key, True))
            self._toggle_vars[key] = var
            cb = ttk.Checkbutton(toggle_frame, text=label, variable=var,
                                  command=self._save_toggles)
            cb.grid(row=0, column=col, sticky="w", padx=6)
            cb.bind("<Enter>", lambda e, k=key: self._show_desc(k))

        # Numeric params
        param_frame = ttk.Frame(rcfg)
        param_frame.grid(row=1, column=0, sticky="w", pady=(6, 0))

        ttk.Label(param_frame, text="RRF K:").pack(side="left")
        self._rrf_k_var = tk.IntVar(value=cfg.get("eval_rrf_k", 60))
        ttk.Spinbox(param_frame, textvariable=self._rrf_k_var,
                    from_=1, to=500, increment=10, width=6,
                    command=self._save_toggles).pack(side="left", padx=(2, 14))

        ttk.Label(param_frame, text="Max chunks/article:").pack(side="left")
        self._div_max_var = tk.IntVar(value=cfg.get("eval_diversity_max", 6))
        ttk.Spinbox(param_frame, textvariable=self._div_max_var,
                    from_=1, to=20, increment=1, width=4,
                    command=self._save_toggles).pack(side="left", padx=(2, 14))

        ttk.Label(param_frame, text="Top-K:").pack(side="left")
        self._topk_var = tk.IntVar(value=cfg.get("eval_top_k", 10))
        ttk.Spinbox(param_frame, textvariable=self._topk_var,
                    from_=1, to=50, increment=1, width=4,
                    command=self._save_toggles).pack(side="left", padx=(2, 0))

        # Description label
        self._desc_lbl = ttk.Label(rcfg, text="Hover a toggle to see what it does.",
                                   foreground="#555", wraplength=860,
                                   font=("TkDefaultFont", 9), justify="left")
        self._desc_lbl.grid(row=2, column=0, sticky="w", pady=(4, 0))

        # ── Sub-notebook: Query | Evaluate ────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

        query_tab = ttk.Frame(nb)
        eval_tab  = ttk.Frame(nb)
        nb.add(query_tab, text="  Query  ")
        nb.add(eval_tab,  text="  Evaluate  ")

        self._build_query_tab(query_tab)
        self._build_eval_tab(eval_tab)

    # ── Query sub-tab ─────────────────────────────────────────────────────────

    def _build_query_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        bar = ttk.Frame(parent, padding=(4, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(1, weight=1)

        ttk.Label(bar, text="Query:").grid(row=0, column=0, sticky="w")
        self._query_var = tk.StringVar()
        qe = ttk.Entry(bar, textvariable=self._query_var)
        qe.grid(row=0, column=1, sticky="ew", padx=4)
        qe.bind("<Return>", lambda _: self._search())

        ttk.Label(bar, text="Threshold:").grid(row=0, column=2, sticky="w", padx=(8, 2))
        self._thresh_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(bar, textvariable=self._thresh_var, from_=0.0, to=1.0,
                    increment=0.05, format="%.2f", width=6).grid(
            row=0, column=3, sticky="w", padx=(0, 8))

        self._btn_search = ttk.Button(bar, text="Search", command=self._search, width=10)
        self._btn_search.grid(row=0, column=4, sticky="e")

        self._q_status = ttk.Label(bar, text="", foreground="gray")
        self._q_status.grid(row=1, column=0, columnspan=5, sticky="w", pady=(2, 0))

        # Results pane
        pane = ttk.PanedWindow(parent, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew")

        # Left: treeview
        lf = ttk.Frame(pane)
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        pane.add(lf, weight=3)

        cols = ("rank", "title", "section", "rrf", "faiss", "src")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", selectmode="browse")
        for col, hdr, w in [("rank","#",40),("title","Title",220),
                              ("section","Section",180),("rrf","RRF",65),
                              ("faiss","Cosine",65),("src","Signals",60)]:
            self._tree.heading(col, text=hdr)
            self._tree.column(col, width=w, minwidth=30,
                               stretch=(col in ("title","section")))
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_q_select)

        # Right: preview
        rf = ttk.LabelFrame(pane, text="Preview", padding=4)
        rf.rowconfigure(1, weight=1)
        rf.columnconfigure(0, weight=1)
        pane.add(rf, weight=2)

        self._btn_kiwix = ttk.Button(rf, text="Open in Kiwix ↗",
                                      command=self._open_kiwix, state="disabled")
        self._btn_kiwix.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._preview = scrolledtext.ScrolledText(
            rf, wrap="word", state="disabled", font=("TkDefaultFont", 9))
        self._preview.grid(row=1, column=0, sticky="nsew")

    # ── Evaluate sub-tab ──────────────────────────────────────────────────────

    def _build_eval_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # Control bar
        bar = ttk.Frame(parent, padding=(4, 4))
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(2, weight=1)

        ttk.Button(bar, text="Load CSV…", command=self._load_golden).grid(
            row=0, column=0, sticky="w")
        self._csv_lbl = ttk.Label(bar, text="No dataset loaded.", foreground="gray")
        self._csv_lbl.grid(row=0, column=1, sticky="w", padx=8)

        self._btn_run = ttk.Button(bar, text="Run Evaluation",
                                    command=self._run_eval, state="disabled")
        self._btn_run.grid(row=0, column=3, sticky="e", padx=(0, 4))
        self._btn_stop_eval = ttk.Button(bar, text="Stop",
                                          command=self._stop_eval, state="disabled", width=6)
        self._btn_stop_eval.grid(row=0, column=4, sticky="e")

        self._btn_export = ttk.Button(bar, text="Export CSV…",
                                       command=self._export_csv, state="disabled")
        self._btn_export.grid(row=0, column=5, sticky="e", padx=(8, 0))

        # Progress
        prog_frame = ttk.Frame(parent, padding=(4, 2))
        prog_frame.grid(row=1, column=0, sticky="ew")
        prog_frame.columnconfigure(1, weight=1)

        self._eval_pb = ttk.Progressbar(prog_frame, mode="determinate", length=300)
        self._eval_pb.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._eval_status = ttk.Label(prog_frame, text="", foreground="gray")
        self._eval_status.grid(row=0, column=1, sticky="w")

        # Main pane
        pane = ttk.PanedWindow(parent, orient="horizontal")
        pane.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

        # Left: per-question results
        lf = ttk.Frame(pane)
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        pane.add(lf, weight=3)

        ecols = ("num", "question", "correct", "hit", "rank", "found_in")
        self._etree = ttk.Treeview(lf, columns=ecols, show="headings", selectmode="browse")
        for col, hdr, w in [
            ("num",      "#",        35),
            ("question", "Question", 280),
            ("correct",  "Correct",  120),
            ("hit",      "Hit@K",    55),
            ("rank",     "Rank",     45),
            ("found_in", "Found In", 160),
        ]:
            self._etree.heading(col, text=hdr)
            self._etree.column(col, width=w, minwidth=30,
                                stretch=(col in ("question", "found_in")))
        evsb = ttk.Scrollbar(lf, orient="vertical", command=self._etree.yview)
        self._etree.configure(yscrollcommand=evsb.set)
        self._etree.grid(row=0, column=0, sticky="nsew")
        evsb.grid(row=0, column=1, sticky="ns")
        self._etree.bind("<<TreeviewSelect>>", self._on_eval_select)
        self._etree.tag_configure("hit",  foreground="green")
        self._etree.tag_configure("miss", foreground="red")

        # Right: metrics + detail
        rf = ttk.Frame(pane)
        rf.rowconfigure(1, weight=1)
        rf.columnconfigure(0, weight=1)
        pane.add(rf, weight=2)

        # Metrics panel
        mf = ttk.LabelFrame(rf, text="Aggregate Metrics", padding=8)
        mf.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        mf.columnconfigure(1, weight=1)

        self._metric_vars: dict[str, tk.StringVar] = {}
        for i, name in enumerate(["Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR@10"]):
            ttk.Label(mf, text=f"{name}:", font=("TkDefaultFont", 9, "bold")).grid(
                row=i, column=0, sticky="w", pady=1)
            var = tk.StringVar(value="—")
            self._metric_vars[name] = var
            ttk.Label(mf, textvariable=var, font=("TkDefaultFont", 9)).grid(
                row=i, column=1, sticky="w", padx=8)

        # Detail view
        detail_lf = ttk.LabelFrame(rf, text="Question Detail", padding=4)
        detail_lf.grid(row=1, column=0, sticky="nsew")
        detail_lf.rowconfigure(0, weight=1)
        detail_lf.columnconfigure(0, weight=1)

        self._detail = scrolledtext.ScrolledText(
            detail_lf, wrap="word", state="disabled", font=("TkDefaultFont", 9))
        self._detail.grid(row=0, column=0, sticky="nsew")

    # ── Query actions ─────────────────────────────────────────────────────────

    def _browse_dir(self) -> None:
        init = self._dir_var.get().strip() or cfg.get("last_index_dir", str(Path.home()))
        path = filedialog.askdirectory(initialdir=init, title="Select index directory")
        if path:
            self._dir_var.set(path)
            cfg["last_index_dir"] = path
            cfg.save()

    def _on_dir_change(self) -> None:
        cfg["last_index_dir"] = self._dir_var.get().strip()
        cfg.save()

    def _save_kiwix(self) -> None:
        cfg["kiwix_base_url"]   = self._kiwix_var.get().strip()
        cfg["kiwix_book_name"]  = self._book_var.get().strip()
        cfg.save()

    def _kiwix_book(self) -> str:
        """Book name: explicit override or derived from index dir name."""
        override = self._book_var.get().strip()
        if override:
            return override
        d = self._dir_var.get().strip()
        return Path(d).name if d else ""

    def _save_toggles(self) -> None:
        for key, var in self._toggle_vars.items():
            cfg[key] = var.get()
        cfg["eval_rrf_k"]         = self._rrf_k_var.get()
        cfg["eval_diversity_max"] = self._div_max_var.get()
        cfg["eval_top_k"]         = self._topk_var.get()
        cfg.save()

    def _show_desc(self, key: str) -> None:
        for k, _label, desc in _TOGGLES:
            if k == key:
                self._desc_lbl.configure(text=desc)
                return

    def _build_run_cfg(self) -> dict:
        c = cfg.as_engine_cfg()
        for key, var in self._toggle_vars.items():
            c[key] = var.get()
        c["eval_rrf_k"]         = self._rrf_k_var.get()
        c["eval_diversity_max"] = self._div_max_var.get()
        return c

    def _search(self) -> None:
        out_dir = self._dir_var.get().strip()
        query   = self._query_var.get().strip()
        if not out_dir or not query:
            return
        top_k   = self._topk_var.get()
        thresh  = self._thresh_var.get()
        run_cfg = self._build_run_cfg()
        q       = self._q

        self._btn_search.configure(state="disabled")
        self._q_status.configure(text="Searching…")
        self._tree.delete(*self._tree.get_children())
        self._set_preview("")

        def _run():
            try:
                from indexer.query import search
                hits = search(Path(out_dir), query, top_k=top_k,
                              threshold=thresh, cfg=run_cfg)
                q.put(("q_results", hits))
            except FileNotFoundError as e:
                q.put(("q_error", f"Index not found — {e}"))
            except Exception:
                q.put(("q_error", traceback.format_exc()))

        threading.Thread(target=_run, daemon=True).start()

    def _open_kiwix(self) -> None:
        sel = self._tree.selection()
        if not sel or not self._hits:
            return
        idx     = int(sel[0])
        hit     = self._hits[idx]
        base    = self._kiwix_var.get().strip().rstrip("/")
        book    = self._kiwix_book()
        article = hit.get("url", "").lstrip("/")
        if article.startswith("A/"):
            article = article[2:]
        webbrowser.open(f"{base}#{book}/{article}")

    # ── Evaluate actions ──────────────────────────────────────────────────────

    def _load_golden(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
            title="Load golden dataset",
        )
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            required = {"question", "option_a", "option_b", "option_c", "option_d", "correct"}
            if not required.issubset({k.lower() for k in (rows[0].keys() if rows else [])}):
                self._csv_lbl.configure(
                    text="CSV missing required columns: question, option_a-d, correct",
                    foreground="red")
                return
            # Normalise keys to lowercase
            self._golden = [{k.lower(): v for k, v in r.items()} for r in rows]
            self._csv_lbl.configure(
                text=f"{len(self._golden)} questions loaded  —  {Path(path).name}",
                foreground="gray")
            self._btn_run.configure(state="normal")
            self._eval_rows = []
            self._etree.delete(*self._etree.get_children())
            self._reset_metrics()
        except Exception as e:
            self._csv_lbl.configure(text=f"Error: {e}", foreground="red")

    def _run_eval(self) -> None:
        if not self._golden:
            return
        out_dir = self._dir_var.get().strip()
        if not out_dir:
            self._eval_status.configure(text="Select an index directory first.")
            return

        self._stop_evt.clear()
        self._eval_rows = []
        self._etree.delete(*self._etree.get_children())
        self._reset_metrics()
        self._btn_run.configure(state="disabled")
        self._btn_stop_eval.configure(state="normal")
        self._btn_export.configure(state="disabled")
        self._eval_pb["maximum"] = len(self._golden)
        self._eval_pb["value"]   = 0

        run_cfg = self._build_run_cfg()
        top_k   = self._topk_var.get()
        golden  = list(self._golden)
        q       = self._q
        stop    = self._stop_evt

        def _run():
            from indexer.query import search
            for i, row in enumerate(golden):
                if stop.is_set():
                    q.put(("eval_stopped",))
                    return
                question      = row["question"]
                letter        = row["correct"].strip().upper()
                opt_map       = {"A": row["option_a"], "B": row["option_b"],
                                 "C": row["option_c"], "D": row["option_d"]}
                correct_text  = opt_map.get(letter, "")
                try:
                    hits = search(Path(out_dir), question, top_k=top_k, cfg=run_cfg)
                except Exception as e:
                    q.put(("eval_result", {
                        "idx": i, "question": question,
                        "correct_letter": letter, "correct_text": correct_text,
                        "rank": None, "found_in": f"ERROR: {e}", "hits": [],
                    }))
                    continue

                rank      = None
                found_in  = ""
                needle    = correct_text.lower()
                for pos, hit in enumerate(hits, 1):
                    if needle and needle in hit.get("text", "").lower():
                        rank     = pos
                        found_in = hit.get("title", "")
                        break

                q.put(("eval_result", {
                    "idx":            i,
                    "question":       question,
                    "correct_letter": letter,
                    "correct_text":   correct_text,
                    "rank":           rank,
                    "found_in":       found_in,
                    "hits":           hits,
                }))
            q.put(("eval_done",))

        threading.Thread(target=_run, daemon=True).start()

    def _stop_eval(self) -> None:
        self._stop_evt.set()
        self._btn_stop_eval.configure(state="disabled")

    def _export_csv(self) -> None:
        if not self._eval_rows:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Export evaluation results",
        )
        if not path:
            return
        fieldnames = ["num", "question", "correct_letter", "correct_text",
                      "rank", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "found_in"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._eval_rows:
                rank = r["rank"]
                writer.writerow({
                    "num":           r["idx"] + 1,
                    "question":      r["question"],
                    "correct_letter":r["correct_letter"],
                    "correct_text":  r["correct_text"],
                    "rank":          rank if rank is not None else "",
                    "hit_at_1":      1 if rank and rank <= 1  else 0,
                    "hit_at_3":      1 if rank and rank <= 3  else 0,
                    "hit_at_5":      1 if rank and rank <= 5  else 0,
                    "hit_at_10":     1 if rank and rank <= 10 else 0,
                    "found_in":      r["found_in"],
                })
        # Append aggregate row
        with open(path, "a", newline="", encoding="utf-8") as f:
            for name, var in self._metric_vars.items():
                f.write(f"\n{name},{var.get()}")

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]

                if kind == "q_results":
                    self._show_q_results(msg[1])
                elif kind == "q_error":
                    self._q_status.configure(text=f"Error: {msg[1]}", foreground="red")
                    self._btn_search.configure(state="normal")

                elif kind == "eval_result":
                    self._add_eval_row(msg[1])
                elif kind == "eval_done":
                    self._eval_status.configure(text="Evaluation complete.")
                    self._btn_run.configure(state="normal")
                    self._btn_stop_eval.configure(state="disabled")
                    self._btn_export.configure(state="normal")
                elif kind == "eval_stopped":
                    self._eval_status.configure(text="Stopped.")
                    self._btn_run.configure(state="normal")
                    self._btn_stop_eval.configure(state="disabled")
                    if self._eval_rows:
                        self._btn_export.configure(state="normal")

        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ── Query results display ─────────────────────────────────────────────────

    def _show_q_results(self, hits: list[dict]) -> None:
        self._hits = hits
        self._tree.delete(*self._tree.get_children())
        self._set_preview("")
        for i, h in enumerate(hits):
            sources = ("F" if h.get("in_faiss")      else "") + \
                      ("T" if h.get("in_title_bm25") else "") + \
                      ("P" if h.get("in_para_bm25")  else "")
            self._tree.insert("", "end", iid=str(i), values=(
                i + 1,
                h.get("title", ""),
                h.get("section_title", ""),
                f"{h.get('rrf_score', 0):.5f}",
                f"{h.get('faiss_score', 0):.3f}",
                sources,
            ))
        n = len(hits)
        if n == 0:
            self._q_status.configure(
                text="No results. Is embedding complete? Check the index dir is correct.",
                foreground="orange")
        else:
            self._q_status.configure(
                text=f"{n} result{'s' if n != 1 else ''}.  "
                     "Signals: F=FAISS  T=TitleBM25  P=ParaBM25",
                foreground="gray")
        self._btn_search.configure(state="normal")
        if hits:
            self._tree.selection_set("0")
            self._btn_kiwix.configure(state="normal")
        else:
            self._btn_kiwix.configure(state="disabled")

    def _on_q_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel or not self._hits:
            self._btn_kiwix.configure(state="disabled")
            return
        self._btn_kiwix.configure(state="normal")
        h = self._hits[int(sel[0])]
        lines = [
            f"Title:    {h.get('title','')}",
            f"Section:  {h.get('section_title','')}",
            f"RRF:      {h.get('rrf_score',0):.6f}   Cosine: {h.get('faiss_score',0):.4f}",
            f"Signals:  {'FAISS ' if h.get('in_faiss') else ''}"
            f"{'TitleBM25 ' if h.get('in_title_bm25') else ''}"
            f"{'ParaBM25' if h.get('in_para_bm25') else ''}",
            "",
            h.get("text", ""),
        ]
        self._set_preview("\n".join(lines))

    # ── Eval results display ──────────────────────────────────────────────────

    def _add_eval_row(self, r: dict) -> None:
        self._eval_rows.append(r)
        self._eval_pb["value"] = len(self._eval_rows)
        n    = len(self._eval_rows)
        done = len(self._golden)
        self._eval_status.configure(text=f"{n} / {done} questions evaluated")

        rank = r["rank"]
        top_k = self._topk_var.get()
        hit   = rank is not None and rank <= top_k
        tag   = "hit" if hit else "miss"
        self._etree.insert("", "end", iid=str(r["idx"]), tags=(tag,), values=(
            r["idx"] + 1,
            r["question"][:60] + ("…" if len(r["question"]) > 60 else ""),
            f"{r['correct_letter']}: {r['correct_text'][:25]}",
            "✓" if hit else "✗",
            str(rank) if rank is not None else "—",
            r["found_in"][:30] if r["found_in"] else "—",
        ))
        self._etree.see(str(r["idx"]))
        self._update_metrics()

    def _on_eval_select(self, _event=None) -> None:
        sel = self._etree.selection()
        if not sel:
            return
        idx = int(sel[0])
        row = next((r for r in self._eval_rows if r["idx"] == idx), None)
        if not row:
            return

        opt_map = {}
        if idx < len(self._golden):
            g = self._golden[idx]
            opt_map = {"A": g.get("option_a",""), "B": g.get("option_b",""),
                       "C": g.get("option_c",""), "D": g.get("option_d","")}

        lines = [
            f"Q{idx+1}: {row['question']}",
            "",
            f"  A: {opt_map.get('A','')}",
            f"  B: {opt_map.get('B','')}",
            f"  C: {opt_map.get('C','')}",
            f"  D: {opt_map.get('D','')}",
            f"\n  Correct: {row['correct_letter']}: {row['correct_text']}",
            f"  Rank found: {row['rank'] if row['rank'] else 'Not found in top-K'}",
            f"  Found in article: {row['found_in'] or '—'}",
            "\n── Top retrieved chunks ─────────────────",
        ]
        for i, h in enumerate(row.get("hits", []), 1):
            lines.append(f"\n[{i}] {h.get('title','')} § {h.get('section_title','')}")
            lines.append(f"    RRF={h.get('rrf_score',0):.5f}")
            lines.append(f"    {h.get('text','')[:200]}…")

        self._set_detail("\n".join(lines))

    def _update_metrics(self) -> None:
        rows  = self._eval_rows
        n     = len(rows)
        if n == 0:
            return
        ranks = [r["rank"] for r in rows]

        def hit_at(k):
            c = sum(1 for rk in ranks if rk is not None and rk <= k)
            return f"{c} / {n}  ({100*c/n:.1f}%)"

        mrr = sum(1/rk for rk in ranks if rk is not None and rk <= 10) / n

        self._metric_vars["Hit@1"].set(hit_at(1))
        self._metric_vars["Hit@3"].set(hit_at(3))
        self._metric_vars["Hit@5"].set(hit_at(5))
        self._metric_vars["Hit@10"].set(hit_at(10))
        self._metric_vars["MRR@10"].set(f"{mrr:.4f}")

    def _reset_metrics(self) -> None:
        for var in self._metric_vars.values():
            var.set("—")

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _set_preview(self, text: str) -> None:
        self._preview.configure(state="normal")
        self._preview.delete("1.0", "end")
        if text:
            self._preview.insert("end", text)
        self._preview.configure(state="disabled")

    def _set_detail(self, text: str) -> None:
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("end", text)
        self._detail.configure(state="disabled")
