"""Index tab — ZIM file picker, pipeline controls, progress bars, log stream."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path

from gui.config import cfg


class IndexTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self._q: queue.Queue = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._phase_totals: dict[str, int] = {}

        self._build_ui()
        self._refresh_output_display()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        # ── ZIM file ──────────────────────────────────────────────────────────
        zim_frame = ttk.LabelFrame(self, text="ZIM File", padding=8)
        zim_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        zim_frame.columnconfigure(1, weight=1)

        ttk.Label(zim_frame, text="Path:").grid(row=0, column=0, sticky="w")
        self._zim_var = tk.StringVar()
        self._zim_entry = ttk.Entry(zim_frame, textvariable=self._zim_var)
        self._zim_entry.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(zim_frame, text="Browse…", command=self._browse_zim).grid(
            row=0, column=2, sticky="e"
        )

        # ── Output directory ──────────────────────────────────────────────────
        out_frame = ttk.LabelFrame(self, text="Output Directory", padding=8)
        out_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        out_frame.columnconfigure(1, weight=1)

        self._out_mode = tk.StringVar(value=cfg["output_dir_mode"])
        ttk.Radiobutton(
            out_frame, text="Auto (next to ZIM file)",
            variable=self._out_mode, value="auto",
            command=self._on_out_mode_change,
        ).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Radiobutton(
            out_frame, text="Custom:",
            variable=self._out_mode, value="custom",
            command=self._on_out_mode_change,
        ).grid(row=1, column=0, sticky="w")

        self._out_var = tk.StringVar(value=cfg["output_dir_custom"])
        self._out_entry = ttk.Entry(out_frame, textvariable=self._out_var)
        self._out_entry.grid(row=1, column=1, sticky="ew", padx=4)
        self._out_browse_btn = ttk.Button(
            out_frame, text="Browse…", command=self._browse_out
        )
        self._out_browse_btn.grid(row=1, column=2, sticky="e")

        self._out_info = ttk.Label(out_frame, text="", foreground="gray",
                                   font=("TkDefaultFont", 9))
        self._out_info.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self._refresh_output_display()

        # ── GPU status ────────────────────────────────────────────────────────
        self._gpu_lbl = ttk.Label(self, text="Checking GPU…", foreground="gray",
                                  font=("TkDefaultFont", 9))
        self._gpu_lbl.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 2))
        self.after(100, self._check_gpu)

        # ── Chunk mode ────────────────────────────────────────────────────────
        mode_frame = ttk.Frame(self, padding=(8, 2))
        mode_frame.grid(row=3, column=0, sticky="ew", padx=0)

        self._flat_var = tk.BooleanVar(value=cfg.get("flat_chunks", False))
        ttk.Checkbutton(
            mode_frame,
            text="Flat chunks — strip Article/Section metadata (baseline index for ablation)",
            variable=self._flat_var,
            command=self._on_flat_change,
        ).pack(side="left", padx=8)
        self._flat_info = ttk.Label(
            mode_frame, text="", foreground="gray", font=("TkDefaultFont", 9)
        )
        self._flat_info.pack(side="left")
        self._refresh_output_display()

        # ── Action buttons ────────────────────────────────────────────────────
        btn_frame = ttk.Frame(self, padding=(8, 4))
        btn_frame.grid(row=4, column=0, sticky="ew", padx=0)

        self._btn_build_all = ttk.Button(
            btn_frame, text="Build All Indexes",
            command=self._start_all, width=18,
        )
        self._btn_build_all.pack(side="left", padx=4)
        ttk.Label(btn_frame,
                  text="← Extract+Embed for Structured AND Flat",
                  foreground="gray", font=("TkDefaultFont", 9),
                  ).pack(side="left", padx=(0, 4))

        ttk.Separator(btn_frame, orient="vertical").pack(side="left", fill="y", padx=8)

        self._btn_both = ttk.Button(
            btn_frame, text="Both Steps", command=lambda: self._start("both"), width=14
        )
        self._btn_both.pack(side="left", padx=4)

        self._btn_extract = ttk.Button(
            btn_frame, text="Extract Only", command=lambda: self._start("extract"), width=14
        )
        self._btn_extract.pack(side="left", padx=4)

        self._btn_embed = ttk.Button(
            btn_frame, text="Embed Only", command=lambda: self._start("embed"), width=14
        )
        self._btn_embed.pack(side="left", padx=4)

        self._btn_stop = ttk.Button(
            btn_frame, text="Stop", command=self._request_stop,
            state="disabled", width=8,
        )
        self._btn_stop.pack(side="left", padx=4)

        ttk.Separator(btn_frame, orient="vertical").pack(side="left", fill="y", padx=8)

        self._btn_reset_embedded = ttk.Button(
            btn_frame,
            text="Reset Embedded Flags",
            command=self._reset_embedded,
            width=22,
        )
        self._btn_reset_embedded.pack(side="left", padx=4)
        ttk.Label(btn_frame,
                  text="← clears embedded=0 on all chunks so Embed re-runs with a new model",
                  foreground="gray", font=("TkDefaultFont", 9),
                  ).pack(side="left", padx=2)

        self._status_lbl = ttk.Label(btn_frame, text="", foreground="gray")
        self._status_lbl.pack(side="right", padx=8)

        # ── Progress bars ─────────────────────────────────────────────────────
        prog_frame = ttk.LabelFrame(self, text="Progress", padding=8)
        prog_frame.grid(row=5, column=0, sticky="nsew", padx=8, pady=4)
        prog_frame.columnconfigure(1, weight=1)
        prog_frame.rowconfigure(4, weight=1)

        ttk.Label(prog_frame, text="Extract:").grid(row=0, column=0, sticky="w")
        self._pb_extract = ttk.Progressbar(prog_frame, length=400, mode="determinate")
        self._pb_extract.grid(row=0, column=1, sticky="ew", padx=4)
        self._lbl_extract = ttk.Label(prog_frame, text="", width=20)
        self._lbl_extract.grid(row=0, column=2, sticky="w")

        ttk.Label(prog_frame, text="Embed:").grid(row=1, column=0, sticky="w")
        self._pb_embed = ttk.Progressbar(prog_frame, length=400, mode="determinate")
        self._pb_embed.grid(row=1, column=1, sticky="ew", padx=4)
        self._lbl_embed = ttk.Label(prog_frame, text="", width=20)
        self._lbl_embed.grid(row=1, column=2, sticky="w")

        ttk.Label(prog_frame, text="Train:").grid(row=2, column=0, sticky="w")
        self._pb_train = ttk.Progressbar(prog_frame, length=400, mode="determinate")
        self._pb_train.grid(row=2, column=1, sticky="ew", padx=4)
        self._lbl_train = ttk.Label(prog_frame, text="", width=20)
        self._lbl_train.grid(row=2, column=2, sticky="w")

        self._eta_lbl = ttk.Label(prog_frame, text="", foreground="gray",
                                  font=("TkDefaultFont", 9))
        self._eta_lbl.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 2))

        ttk.Label(prog_frame, text="Log:").grid(row=4, column=0, sticky="nw", pady=(4, 0))
        self._log = scrolledtext.ScrolledText(
            prog_frame, height=14, state="disabled",
            wrap="word", font=("TkFixedFont", 9),
        )
        self._log.grid(row=4, column=1, columnspan=2, sticky="nsew", padx=4, pady=(4, 0))

    # ── GPU check ─────────────────────────────────────────────────────────────

    def _check_gpu(self) -> None:
        try:
            from indexer.embed import gpu_available, gpu_info
            if gpu_available():
                self._gpu_lbl.configure(
                    text=f"GPU: {gpu_info()} available — embedding will use GPU",
                    foreground="green",
                )
            else:
                self._gpu_lbl.configure(
                    text="GPU: not available — embedding will use CPU",
                    foreground="gray",
                )
        except Exception:
            self._gpu_lbl.configure(text="GPU: unknown", foreground="gray")

    # ── File pickers ──────────────────────────────────────────────────────────

    def _browse_zim(self) -> None:
        init_dir = cfg.get("last_zim_dir", str(Path.home()))
        path = filedialog.askopenfilename(
            initialdir=init_dir,
            filetypes=[("ZIM files", "*.zim"), ("All files", "*")],
            title="Select ZIM file",
        )
        if path:
            self._zim_var.set(path)
            cfg["last_zim_dir"] = str(Path(path).parent)
            cfg.save()
            self._refresh_output_display()

    def _browse_out(self) -> None:
        init_dir = cfg.get("last_index_dir", str(Path.home()))
        path = filedialog.askdirectory(initialdir=init_dir, title="Select output directory")
        if path:
            self._out_var.set(path)
            cfg["last_index_dir"] = path
            cfg.save()
            self._refresh_output_display()

    # ── Output mode helpers ───────────────────────────────────────────────────

    def _on_flat_change(self) -> None:
        cfg["flat_chunks"] = self._flat_var.get()
        cfg.save()
        self._refresh_output_display()

    def _on_out_mode_change(self) -> None:
        mode = self._out_mode.get()
        state = "normal" if mode == "custom" else "disabled"
        self._out_entry.configure(state=state)
        self._out_browse_btn.configure(state=state)
        cfg["output_dir_mode"] = mode
        cfg.save()
        self._refresh_output_display()

    def _refresh_output_display(self) -> None:
        mode = self._out_mode.get()
        flat = getattr(self, "_flat_var", None) and self._flat_var.get()
        suffix = "_flat" if flat else ""
        if mode == "auto":
            self._out_entry.configure(state="disabled")
            self._out_browse_btn.configure(state="disabled")
            zim = self._zim_var.get().strip()
            if zim:
                derived = str(Path(zim).parent / (Path(zim).stem + suffix))
                self._out_info.configure(text=f"→ {derived}")
            else:
                self._out_info.configure(text="→ (select a ZIM file first)")
        else:
            self._out_entry.configure(state="normal")
            self._out_browse_btn.configure(state="normal")
            self._out_info.configure(text="")
        if hasattr(self, "_flat_info"):
            self._flat_info.configure(
                text="Output → …_flat/" if flat else ""
            )

    # ── Pipeline start/stop ───────────────────────────────────────────────────

    def _start_all(self) -> None:
        """Run Extract+Embed for structured index, then Extract+Embed for flat index."""
        zim = self._zim_var.get().strip()
        if not zim:
            self._log_append("[error] No ZIM file selected.\n")
            return
        if not Path(zim).exists():
            self._log_append(f"[error] File not found: {zim}\n")
            return

        cfg["output_dir_mode"]   = self._out_mode.get()
        cfg["output_dir_custom"] = self._out_var.get().strip()
        cfg.save()

        self._reset_ui()
        self._set_running(True)
        self._t0 = time.time()
        self._phase_totals = {}

        stop_event = threading.Event()
        self._stop_event = stop_event
        q = self._q

        base = cfg.as_engine_cfg()
        struct_cfg = {**base, "flat_chunks": False}
        flat_cfg   = {**base, "flat_chunks": True}

        def _run():
            try:
                from indexer import pipeline
                zim_path = Path(zim)

                def log_cb(msg: str) -> None:
                    q.put(("log", msg + "\n"))

                def progress_cb(phase: str, done: int, total: int) -> None:
                    q.put(("progress", phase, done, total))

                steps = [
                    ("Extract structured", pipeline.run_extract, struct_cfg),
                    ("Embed   structured", pipeline.run_embed,   struct_cfg),
                    ("Extract flat",       pipeline.run_extract, flat_cfg),
                    ("Embed   flat",       pipeline.run_embed,   flat_cfg),
                ]

                for step_label, fn, step_cfg in steps:
                    if stop_event.is_set():
                        break
                    ts = time.strftime("%H:%M:%S")
                    q.put(("log", f"\n[{ts}] ── {step_label} ──\n"))
                    fn(zim_path, cfg=step_cfg,
                       log=log_cb, progress=progress_cb, stop=stop_event)

                if stop_event.is_set():
                    q.put(("stopped",))
                else:
                    q.put(("done", time.time() - self._t0))
            except Exception:
                q.put(("error", traceback.format_exc()))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _start(self, mode: str) -> None:
        zim = self._zim_var.get().strip()
        if not zim:
            self._log_append("[error] No ZIM file selected.\n")
            return
        if not Path(zim).exists():
            self._log_append(f"[error] File not found: {zim}\n")
            return

        # Persist output dir setting
        cfg["output_dir_mode"]   = self._out_mode.get()
        cfg["output_dir_custom"] = self._out_var.get().strip()
        cfg.save()

        self._reset_ui()
        self._set_running(True)
        self._t0 = time.time()
        self._phase_totals = {}

        stop_event = threading.Event()
        self._stop_event = stop_event
        q = self._q

        run_cfg = cfg.as_engine_cfg()
        run_cfg["flat_chunks"] = self._flat_var.get()

        def _run():
            try:
                from indexer import pipeline
                zim_path = Path(zim)

                def log_cb(msg: str) -> None:
                    q.put(("log", msg + "\n"))

                def progress_cb(phase: str, done: int, total: int) -> None:
                    q.put(("progress", phase, done, total))

                if mode == "extract":
                    pipeline.run_extract(zim_path, cfg=run_cfg,
                                         log=log_cb, progress=progress_cb,
                                         stop=stop_event)
                elif mode == "embed":
                    pipeline.run_embed(zim_path, cfg=run_cfg,
                                       log=log_cb, progress=progress_cb,
                                       stop=stop_event)
                else:
                    pipeline.run_both(zim_path, cfg=run_cfg,
                                      log=log_cb, progress=progress_cb,
                                      stop=stop_event)

                if stop_event.is_set():
                    q.put(("stopped",))
                else:
                    q.put(("done", time.time() - self._t0))
            except Exception:
                q.put(("error", traceback.format_exc()))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def _request_stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        self._btn_stop.configure(state="disabled")

    def request_stop(self) -> None:
        """Called by app.py on window close."""
        self._request_stop()

    # ── Queue polling (main thread) ───────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log_append(msg[1])
                elif kind == "progress":
                    _, phase, done, total = msg
                    self._update_progress(phase, done, total)
                elif kind == "done":
                    elapsed = msg[1]
                    self._log_append(f"\n✓ Finished in {_fmt_time(elapsed)}.\n")
                    self._set_running(False)
                    self._eta_lbl.configure(
                        text=f"Completed in {_fmt_time(elapsed)}"
                    )
                elif kind == "stopped":
                    self._log_append("\n— Stopped by user.\n")
                    self._set_running(False)
                    self._eta_lbl.configure(text="Stopped.")
                    # Clear cached embedding model to ensure fresh GPU session on restart
                    try:
                        from indexer.embed import unload_models
                        unload_models()
                    except Exception:
                        pass
                elif kind == "error":
                    self._log_append(f"\n[ERROR]\n{msg[1]}\n")
                    self._set_running(False)
                    self._eta_lbl.configure(text="Error — see log.")
                    # Clear cached embedding model to ensure fresh GPU session on retry
                    try:
                        from indexer.embed import unload_models
                        unload_models()
                    except Exception:
                        pass
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ── Progress update ───────────────────────────────────────────────────────

    def _update_progress(self, phase: str, done: int, total: int) -> None:
        if total <= 0:
            pct = 100
        else:
            pct = int(done / total * 100)

        self._phase_totals[phase] = total

        elapsed = time.time() - self._t0
        if done > 0 and total > done:
            eta = elapsed / done * (total - done)
            rate = done / elapsed if elapsed > 0 else 0
            eta_str = _fmt_time(eta)
            self._eta_lbl.configure(
                text=f"[{phase}]  {done:,} / {total:,}  —  "
                     f"ETA {eta_str}  ({rate:.0f}/s)  "
                     f"Elapsed {_fmt_time(elapsed)}"
            )
        else:
            self._eta_lbl.configure(
                text=f"[{phase}]  {done:,} / {total:,}  —  "
                     f"Elapsed {_fmt_time(elapsed)}"
            )

        label_text = f"{pct}%  ({done:,}/{total:,})" if total else f"{done:,}"

        if phase == "extract":
            self._pb_extract["value"] = pct
            self._lbl_extract.configure(text=label_text)
        elif phase == "embed":
            self._pb_embed["value"] = pct
            self._lbl_embed.configure(text=label_text)
        elif phase == "train":
            self._pb_train["value"] = pct
            self._lbl_train.configure(text=label_text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_append(self, text: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", text)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state_run  = "disabled" if running else "normal"
        state_stop = "normal"   if running else "disabled"
        self._btn_build_all.configure(state=state_run)
        self._btn_both.configure(state=state_run)
        self._btn_extract.configure(state=state_run)
        self._btn_embed.configure(state=state_run)
        self._btn_stop.configure(state=state_stop)
        self._btn_reset_embedded.configure(state=state_run)

    def _reset_embedded(self) -> None:
        mode = self._out_mode.get()
        flat = self._flat_var.get()
        suffix = "_flat" if flat else ""
        if mode == "auto":
            zim = self._zim_var.get().strip()
            if not zim:
                self._status_lbl.configure(text="Select a ZIM file first", foreground="red")
                return
            out_dir = Path(zim).parent / (Path(zim).stem + suffix)
        else:
            custom = self._out_var.get().strip()
            if not custom:
                self._status_lbl.configure(text="No output dir set", foreground="red")
                return
            out_dir = Path(custom)
        db_path = out_dir / "data.db"
        if not db_path.exists():
            self._status_lbl.configure(text=f"No data.db in {out_dir}", foreground="red")
            return
        import sqlite3, tkinter.messagebox as _mb
        if not _mb.askyesno(
            "Reset embedded flags",
            f"Set embedded=0 for all chunks in:\n{db_path}\n\n"
            "Do this before re-embedding with a new model.\n"
            "The FAISS index should also be deleted manually.",
        ):
            return
        try:
            con = sqlite3.connect(db_path)
            cur = con.execute("UPDATE chunks SET embedded=0")
            con.commit()
            n = cur.rowcount
            con.close()
            self._status_lbl.configure(
                text=f"Reset {n:,} chunks → embedded=0", foreground="green")
            self._log_append(f"Reset {n:,} chunks to embedded=0. "
                             f"Delete faiss.index then click Embed Only.\n")
        except Exception as e:
            self._status_lbl.configure(text=f"Error: {e}", foreground="red")

    def _reset_ui(self) -> None:
        self._pb_extract["value"] = 0
        self._pb_embed["value"]   = 0
        self._pb_train["value"]   = 0
        self._lbl_extract.configure(text="")
        self._lbl_embed.configure(text="")
        self._lbl_train.configure(text="")
        self._eta_lbl.configure(text="")
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"
