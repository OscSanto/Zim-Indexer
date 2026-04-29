"""Settings tab — all configurable pipeline parameters with descriptions."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

from gui.config import cfg, EMBED_MODELS, MODEL_NAMES


_HELP: dict[str, str] = {
    "embed_model": (
        "Embedding model used to turn text into vectors. Must match whatever "
        "AIIAB is configured to use (same dim). Changing after indexing requires "
        "a full re-index. Multilingual models work on any ZIM; English models are "
        "faster and smaller but only for English content."
    ),
    "fastembed_cache_path": (
        "Directory where fastembed caches downloaded model weights. Survives "
        "reboots. Shared with other fastembed users on the same machine."
    ),
    "embed_batch_size": (
        "Number of text chunks embedded per GPU/CPU call. Larger = faster on "
        "strong hardware. Reduce to 4–8 on low-RAM machines or if you get OOM."
    ),
    "faiss_ivf_threshold": (
        "Chunk count above which an IVFFlat approximate index is built instead "
        "of a flat exact index. IVFFlat is much faster to search but requires a "
        "training pass. Leave at 500 000 unless you know what you're doing."
    ),
    "faiss_nprobe": (
        "Number of IVF cells probed per query. Higher = more accurate but slower "
        "search. 64 is a good default. Only affects IVFFlat indexes."
    ),
    "faiss_save_every": (
        "Save the FAISS index to disk every N chunks embedded. Lower values "
        "reduce data loss on crash but add I/O overhead."
    ),
    "priority_lead": (
        "Max lead (intro) paragraphs kept per article before switching to other "
        "content types. Lead text usually has the best summary."
    ),
    "priority_infobox": (
        "Max infobox fact-rows kept per article. Infoboxes are great for "
        "factual queries (dates, measurements, names)."
    ),
    "priority_prose": (
        "Max body prose chunks kept per article. Increase for deep articles; "
        "decrease to save storage and embedding time."
    ),
    "skip_namespaces": (
        "Comma-separated ZIM path prefixes to ignore. Wikipedia-style ZIMs "
        "have Category, Template, File, etc. pages that aren't article content."
    ),
    "skip_regex": (
        "Python regex applied to article titles; matching titles are skipped. "
        "The default filters out disambiguation and meta pages."
    ),
    "min_prose_chars": (
        "Articles with fewer total prose characters than this are skipped. "
        "Filters out stub and redirect-like pages. 200 is a reasonable minimum."
    ),
    "min_infobox_rows": (
        "Articles whose only content is a tiny infobox (fewer rows than this) "
        "are skipped. Prevents embedding near-empty tables."
    ),
}


class SettingsTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self._vars: dict[str, tk.Variable] = {}
        self._build_ui()
        self._load_from_cfg()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Outer canvas + scrollbar for the whole tab
        canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas, padding=8)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(event):
            canvas.itemconfig(win_id, width=event.width)

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_resize)

        # Mouse-wheel scrolling
        def _on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)

        inner.columnconfigure(0, weight=1)

        row = 0

        # ── Embedding ─────────────────────────────────────────────────────────
        row = self._group(inner, row, "Embedding")

        row = self._combo(inner, row,
            key="embed_model",
            label="Embedding model",
            values=MODEL_NAMES,
            on_change=self._on_model_change,
        )

        self._model_info_lbl = ttk.Label(
            inner, text="", foreground="#555", wraplength=700,
            font=("TkDefaultFont", 9), justify="left",
        )
        self._model_info_lbl.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 4))
        row += 1

        model_btn_row = ttk.Frame(inner)
        model_btn_row.grid(row=row, column=0, sticky="w", padx=20, pady=(0, 8))
        row += 1
        ttk.Button(
            model_btn_row, text="Save Embed Model",
            command=self._save_embed_model,
        ).pack(side="left", padx=(0, 8))
        self._model_saved_lbl = ttk.Label(model_btn_row, text="", foreground="green")
        self._model_saved_lbl.pack(side="left")

        row = self._entry(inner, row,
            key="fastembed_cache_path",
            label="fastembed cache path",
            browse="dir",
        )
        row = self._spinbox(inner, row,
            key="embed_batch_size",
            label="Embed batch size",
            from_=1, to=512, step=4,
        )

        # ── FAISS ─────────────────────────────────────────────────────────────
        row = self._group(inner, row, "FAISS Index")

        row = self._spinbox(inner, row,
            key="faiss_ivf_threshold",
            label="IVF threshold (chunks)",
            from_=10_000, to=10_000_000, step=50_000,
        )
        row = self._spinbox(inner, row,
            key="faiss_nprobe",
            label="nprobe",
            from_=1, to=4096, step=8,
        )
        row = self._spinbox(inner, row,
            key="faiss_save_every",
            label="Save every N chunks",
            from_=16, to=10_000, step=16,
        )

        # ── Pipeline priorities ───────────────────────────────────────────────
        row = self._group(inner, row, "Pipeline Priorities")

        row = self._spinbox(inner, row,
            key="priority_lead",
            label="Max lead chunks",
            from_=0, to=20, step=1,
        )
        row = self._spinbox(inner, row,
            key="priority_infobox",
            label="Max infobox rows",
            from_=0, to=50, step=1,
        )
        row = self._spinbox(inner, row,
            key="priority_prose",
            label="Max prose chunks",
            from_=0, to=100, step=1,
        )

        # ── Content Filtering ─────────────────────────────────────────────────
        row = self._group(inner, row, "Content Filtering")

        row = self._entry(inner, row, key="skip_namespaces", label="Skip namespaces")
        row = self._entry(inner, row, key="skip_regex",      label="Skip regex")
        row = self._spinbox(inner, row,
            key="min_prose_chars",
            label="Min prose chars",
            from_=0, to=10_000, step=50,
        )
        row = self._spinbox(inner, row,
            key="min_infobox_rows",
            label="Min infobox rows",
            from_=0, to=20, step=1,
        )

        # ── Output ────────────────────────────────────────────────────────────
        row = self._group(inner, row, "Output Directory")

        out_frame = ttk.Frame(inner)
        out_frame.grid(row=row, column=0, sticky="ew", padx=12, pady=2)
        out_frame.columnconfigure(1, weight=1)
        row += 1

        self._out_mode_var = tk.StringVar()
        ttk.Radiobutton(
            out_frame, text="Auto (sibling of ZIM file)",
            variable=self._out_mode_var, value="auto",
            command=self._on_out_mode,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Radiobutton(
            out_frame, text="Custom:",
            variable=self._out_mode_var, value="custom",
            command=self._on_out_mode,
        ).grid(row=1, column=0, sticky="w")

        self._out_custom_var = tk.StringVar()
        self._out_entry = ttk.Entry(out_frame, textvariable=self._out_custom_var)
        self._out_entry.grid(row=1, column=1, sticky="ew", padx=4)
        self._out_btn = ttk.Button(
            out_frame, text="Browse…", command=self._browse_out
        )
        self._out_btn.grid(row=1, column=2, sticky="e")

        # ── Save / Reset ──────────────────────────────────────────────────────
        row = self._group(inner, row, "")  # spacer

        btn_row = ttk.Frame(inner)
        btn_row.grid(row=row, column=0, sticky="ew", padx=12, pady=8)
        row += 1

        ttk.Button(btn_row, text="Save Settings", command=self._save).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Reset to Defaults", command=self._reset).pack(side="left", padx=4)
        self._saved_lbl = ttk.Label(btn_row, text="", foreground="green")
        self._saved_lbl.pack(side="left", padx=12)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _group(self, parent, row: int, title: str) -> int:
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=row, column=0, sticky="ew", padx=4, pady=(8, 2))
        row += 1
        if title:
            ttk.Label(parent, text=title,
                      font=("TkDefaultFont", 10, "bold")).grid(
                row=row, column=0, sticky="w", padx=8, pady=(0, 4))
            row += 1
        return row

    def _combo(self, parent, row: int, *, key: str, label: str,
               values: list[str], on_change=None) -> int:
        var = tk.StringVar()
        self._vars[key] = var

        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="ew", padx=12, pady=2)
        f.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(f, text=label + ":", width=24, anchor="w").grid(
            row=0, column=0, sticky="w")
        cb = ttk.Combobox(f, textvariable=var, values=values, state="readonly")
        cb.grid(row=0, column=1, sticky="ew")
        if on_change:
            var.trace_add("write", lambda *_: on_change())

        if key in _HELP:
            ttk.Label(
                parent, text=_HELP[key], foreground="#555",
                wraplength=700, font=("TkDefaultFont", 9), justify="left",
            ).grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 4))
            row += 1
        return row

    def _entry(self, parent, row: int, *, key: str, label: str,
               browse: str | None = None) -> int:
        var = tk.StringVar()
        self._vars[key] = var

        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="ew", padx=12, pady=2)
        f.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(f, text=label + ":", width=24, anchor="w").grid(
            row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=var).grid(row=0, column=1, sticky="ew", padx=4)
        if browse == "dir":
            ttk.Button(f, text="Browse…",
                       command=lambda k=key, v=var: self._browse_dir(v)).grid(
                row=0, column=2, sticky="e")

        if key in _HELP:
            ttk.Label(
                parent, text=_HELP[key], foreground="#555",
                wraplength=700, font=("TkDefaultFont", 9), justify="left",
            ).grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 4))
            row += 1
        return row

    def _spinbox(self, parent, row: int, *, key: str, label: str,
                 from_: int, to: int, step: int = 1) -> int:
        var = tk.IntVar()
        self._vars[key] = var

        f = ttk.Frame(parent)
        f.grid(row=row, column=0, sticky="ew", padx=12, pady=2)
        row += 1

        ttk.Label(f, text=label + ":", width=24, anchor="w").grid(
            row=0, column=0, sticky="w")
        ttk.Spinbox(f, textvariable=var, from_=from_, to=to,
                    increment=step, width=12).grid(row=0, column=1, sticky="w")

        if key in _HELP:
            ttk.Label(
                parent, text=_HELP[key], foreground="#555",
                wraplength=700, font=("TkDefaultFont", 9), justify="left",
            ).grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 4))
            row += 1
        return row

    # ── Load / Save / Reset ───────────────────────────────────────────────────

    def _load_from_cfg(self) -> None:
        for key, var in self._vars.items():
            val = cfg.get(key)
            if val is not None:
                try:
                    var.set(val)
                except Exception:
                    pass
        self._out_mode_var.set(cfg["output_dir_mode"])
        self._out_custom_var.set(cfg["output_dir_custom"])
        self._on_out_mode()
        self._on_model_change()

    def _save(self) -> None:
        for key, var in self._vars.items():
            try:
                cfg[key] = var.get()
            except Exception:
                pass
        cfg["output_dir_mode"]   = self._out_mode_var.get()
        cfg["output_dir_custom"] = self._out_custom_var.get()
        cfg.save()
        self._saved_lbl.configure(text="Saved.")
        self.after(2000, lambda: self._saved_lbl.configure(text=""))

    def _reset(self) -> None:
        cfg.reset()
        self._load_from_cfg()
        self._saved_lbl.configure(text="Reset to defaults.")
        self.after(2000, lambda: self._saved_lbl.configure(text=""))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_embed_model(self) -> None:
        model = self._vars.get("embed_model")
        if not model:
            return
        cfg["embed_model"] = model.get()
        cfg.save()
        self._model_saved_lbl.configure(text=f"Saved — {model.get()}")
        self.after(3000, lambda: self._model_saved_lbl.configure(text=""))

    def _on_model_change(self) -> None:
        model = self._vars.get("embed_model")
        if not model:
            return
        name = model.get()
        from gui.config import MODEL_INFO
        info = MODEL_INFO.get(name)
        if info:
            _, dim, size_mb, langs, desc = info
            self._model_info_lbl.configure(
                text=f"dim={dim}  size≈{size_mb} MB  languages={langs}  —  {desc}"
            )
        else:
            self._model_info_lbl.configure(text="")

    def _on_out_mode(self) -> None:
        mode = self._out_mode_var.get()
        state = "normal" if mode == "custom" else "disabled"
        self._out_entry.configure(state=state)
        self._out_btn.configure(state=state)

    def _browse_dir(self, var: tk.StringVar) -> None:
        current = var.get().strip() or str(Path.home())
        path = filedialog.askdirectory(initialdir=current)
        if path:
            var.set(path)

    def _browse_out(self) -> None:
        self._browse_dir(self._out_custom_var)
