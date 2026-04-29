"""Main Tk window — creates the notebook and mounts the four tabs."""
import tkinter as tk
from tkinter import ttk


def launch() -> None:
    root = tk.Tk()
    root.title("ZIM Indexer")
    root.geometry("980x740")
    root.minsize(820, 600)

    # Use a clean theme across platforms
    style = ttk.Style(root)
    for t in ("clam", "alt", "default"):
        if t in style.theme_names():
            style.theme_use(t)
            break

    # Slightly larger default font
    import tkinter.font as tkfont
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(size=10)
    root.option_add("*Font", default_font)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=6, pady=6)

    # Import tabs here to avoid circular deps
    from gui.index_tab    import IndexTab
    from gui.settings_tab import SettingsTab
    from gui.search_tab   import SearchTab
    from gui.browse_tab   import BrowseTab
    from gui.evaluate_tab import EvaluateTab

    index_tab    = IndexTab(notebook)
    settings_tab = SettingsTab(notebook)
    search_tab   = SearchTab(notebook)
    browse_tab   = BrowseTab(notebook)
    evaluate_tab = EvaluateTab(notebook)

    notebook.add(index_tab,    text="  Index  ")
    notebook.add(settings_tab, text="  Settings  ")
    notebook.add(search_tab,   text="  Search  ")
    notebook.add(browse_tab,   text="  Browse  ")
    notebook.add(evaluate_tab, text="  Evaluate  ")

    root.protocol("WM_DELETE_WINDOW", lambda: _on_close(root, index_tab))
    root.mainloop()


def _on_close(root: tk.Tk, index_tab) -> None:
    # Signal any running indexing job to stop before exit
    try:
        index_tab.request_stop()
    except Exception:
        pass
    root.destroy()
