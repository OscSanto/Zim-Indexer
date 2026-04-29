"""Browse tab — open a ZIM file and read articles like Kiwix."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from pathlib import Path

from gui.config import cfg

# tkinterweb is optional — fall back to plain text if not installed
try:
    from tkinterweb import HtmlFrame
    _HAVE_HTMLFRAME = True
except ImportError:
    _HAVE_HTMLFRAME = False


class BrowseTab(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent)
        self._archive = None
        self._entries: list[tuple[str, str]] = []   # (path, title)
        self._filtered: list[int] = []               # indices into _entries
        self._q: queue.Queue = queue.Queue()
        self._build_ui()
        last = cfg.get("last_browse_zim", "")
        if last and Path(last).exists():
            self._zim_var.set(last)
            self._open_zim(last)
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Top bar
        top = ttk.Frame(self, padding=(8, 6))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="ZIM file:").grid(row=0, column=0, sticky="w")
        self._zim_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._zim_var).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="Browse…", command=self._browse_zim).grid(
            row=0, column=2, sticky="e")
        ttk.Button(top, text="Open", command=lambda: self._open_zim(self._zim_var.get())).grid(
            row=0, column=3, sticky="e", padx=(4, 0))

        self._info_lbl = ttk.Label(top, text="", foreground="gray",
                                   font=("TkDefaultFont", 9))
        self._info_lbl.grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # Main pane: article list (left) + viewer (right)
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # ── Left panel ────────────────────────────────────────────────────────
        left = ttk.Frame(pane)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        pane.add(left, weight=1)

        self._filter_var = tk.StringVar()
        self._filter_var.trace_add("write", lambda *_: self._apply_filter())
        filter_entry = ttk.Entry(left, textvariable=self._filter_var,
                                  font=("TkDefaultFont", 9))
        filter_entry.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        ttk.Label(left, text="filter", foreground="gray",
                  font=("TkDefaultFont", 8)).place(in_=filter_entry,
                  relx=1.0, rely=0.5, anchor="e", x=-4)

        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self._listbox = tk.Listbox(list_frame, selectmode="browse",
                                    font=("TkDefaultFont", 9),
                                    activestyle="none")
        vsb = ttk.Scrollbar(list_frame, orient="vertical",
                             command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=vsb.set)
        self._listbox.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._listbox.bind("<<ListboxSelect>>", self._on_article_select)

        self._count_lbl = ttk.Label(left, text="", foreground="gray",
                                    font=("TkDefaultFont", 8))
        self._count_lbl.grid(row=2, column=0, sticky="w")

        # ── Right panel ───────────────────────────────────────────────────────
        right = ttk.Frame(pane)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        pane.add(right, weight=3)

        nav_bar = ttk.Frame(right)
        nav_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        nav_bar.columnconfigure(1, weight=1)

        self._title_lbl = ttk.Label(nav_bar, text="", font=("TkDefaultFont", 10, "bold"))
        self._title_lbl.grid(row=0, column=1, sticky="w", padx=4)

        if _HAVE_HTMLFRAME:
            self._html_view = HtmlFrame(right, horizontal_scrollbar=False)
            self._html_view.grid(row=1, column=0, sticky="nsew")
            self._plain_view = None
        else:
            self._html_view = None
            self._plain_view = scrolledtext.ScrolledText(
                right, wrap="word", state="disabled",
                font=("TkDefaultFont", 9),
            )
            self._plain_view.grid(row=1, column=0, sticky="nsew")
            ttk.Label(
                right,
                text="Install tkinterweb for HTML rendering  (pip install tkinterweb)",
                foreground="gray", font=("TkDefaultFont", 8),
            ).grid(row=2, column=0, sticky="w")

    # ── ZIM open / browse ─────────────────────────────────────────────────────

    def _browse_zim(self) -> None:
        init = cfg.get("last_zim_dir", str(Path.home()))
        path = filedialog.askopenfilename(
            initialdir=init,
            filetypes=[("ZIM files", "*.zim"), ("All files", "*")],
            title="Select ZIM file",
        )
        if path:
            self._zim_var.set(path)
            cfg["last_zim_dir"] = str(Path(path).parent)
            cfg.save()
            self._open_zim(path)

    def _open_zim(self, path: str) -> None:
        path = path.strip()
        if not path or not Path(path).exists():
            return
        self._info_lbl.configure(text="Opening…")
        self._listbox.delete(0, "end")
        self._entries = []
        self._filtered = []

        q = self._q

        def _load():
            try:
                from libzim.reader import Archive
                archive = Archive(path)
                entries: list[tuple[str, str]] = []
                for i in range(archive.article_count):
                    try:
                        entry = archive._get_entry_by_id(i)
                        if entry.is_redirect:
                            continue
                        entries.append((entry.path, entry.title or entry.path))
                    except Exception:
                        continue
                q.put(("opened", archive, entries, path))
            except Exception as e:
                q.put(("open_error", str(e)))

        threading.Thread(target=_load, daemon=True).start()

    # ── Article display ───────────────────────────────────────────────────────

    def _show_article(self, entry_path: str) -> None:
        if not self._archive:
            return
        try:
            entry = self._archive.get_entry_by_path(entry_path)
            if entry.is_redirect:
                entry = entry.get_redirect_entry()
            item  = entry.get_item()
            title = entry.title or entry.path
            self._title_lbl.configure(text=title)

            mime = str(item.mimetype)
            if "html" in mime:
                html = bytes(item.content).decode("utf-8", errors="replace")
                self._render_html(html)
            else:
                raw = bytes(item.content).decode("utf-8", errors="replace")
                self._render_plain(raw)
        except Exception as e:
            self._render_plain(f"[Error reading article]\n{e}")

    def _render_html(self, html: str) -> None:
        if _HAVE_HTMLFRAME and self._html_view:
            self._html_view.load_html(html)
        else:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n")
            except Exception:
                text = html
            self._render_plain(text)

    def _render_plain(self, text: str) -> None:
        if self._plain_view:
            self._plain_view.configure(state="normal")
            self._plain_view.delete("1.0", "end")
            self._plain_view.insert("end", text)
            self._plain_view.configure(state="disabled")
        elif _HAVE_HTMLFRAME and self._html_view:
            safe = text.replace("&", "&amp;").replace("<", "&lt;")
            self._html_view.load_html(f"<pre>{safe}</pre>")

    # ── Filter ────────────────────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        needle = self._filter_var.get().strip().lower()
        self._listbox.delete(0, "end")
        if needle:
            self._filtered = [
                i for i, (_, t) in enumerate(self._entries)
                if needle in t.lower()
            ]
        else:
            self._filtered = list(range(len(self._entries)))

        for idx in self._filtered:
            self._listbox.insert("end", self._entries[idx][1])

        n = len(self._filtered)
        total = len(self._entries)
        self._count_lbl.configure(
            text=f"{n:,} / {total:,} articles" if needle else f"{total:,} articles"
        )

    def _on_article_select(self, _event=None) -> None:
        sel = self._listbox.curselection()
        if not sel:
            return
        list_idx = sel[0]
        if list_idx >= len(self._filtered):
            return
        entry_path = self._entries[self._filtered[list_idx]][0]
        threading.Thread(
            target=lambda: self._q.put(("show", entry_path)),
            daemon=True,
        ).start()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "opened":
                    _, archive, entries, path = msg
                    self._archive = archive
                    self._entries = entries
                    cfg["last_browse_zim"] = path
                    cfg.save()
                    self._apply_filter()
                    self._info_lbl.configure(
                        text=f"{len(entries):,} articles  —  {Path(path).name}"
                    )
                elif msg[0] == "open_error":
                    self._info_lbl.configure(
                        text=f"Error: {msg[1]}", )
                elif msg[0] == "show":
                    self._show_article(msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)
