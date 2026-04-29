"""
HTML extraction from ZIM article content.

Wikipedia ZIM HTML structure:
  <h1> = article title (always present, not a section boundary)
  <div class="mw-parser-output"> = all article content
  <h2>/<h3> = actual section headings

Returns:
  {
    "lead":     str,                                       # joined lead units
    "lead_paragraphs": [str],                              # semantic lead units
    "infobox":  {"header": str, "rows": [{"label": str, "value": str}]} | None,
    "sections": [{"title": str, "paragraphs": [str]}, ...],  # section-bounded semantic units
  }
"""
import re
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_JUNK_SECTIONS = {
    "references", "external links", "see also", "notes", "further reading",
    "bibliography", "footnotes", "citations", "sources",
}

_CITE_RE = re.compile(r"\[\w{0,8}\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def _clean_text(text: str) -> str:
    text = _CITE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_RE.split(text) if p.strip()]
    return parts or [text]


def _semantic_units(texts: list[str], soft_chars: int = 420, min_chars: int = 50) -> list[str]:
    """
    Produce section-bounded text units:
      - keep existing paragraph/list boundaries
      - split very long items on sentence boundaries
      - merge only short adjacent units within the same section
    """
    units: list[str] = []
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        if len(text) <= soft_chars:
            units.append(text)
            continue

        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            units.append(text)
            continue

        buf: list[str] = []
        buf_len = 0
        for sent in sentences:
            sent_len = len(sent) + (1 if buf else 0)
            if buf and buf_len + sent_len > soft_chars:
                units.append(" ".join(buf).strip())
                buf = [sent]
                buf_len = len(sent)
            else:
                buf.append(sent)
                buf_len += sent_len
        if buf:
            units.append(" ".join(buf).strip())

    merged: list[str] = []
    i = 0
    while i < len(units):
        current = units[i]
        if len(current) < min_chars and i + 1 < len(units):
            current = (current + " " + units[i + 1]).strip()
            i += 1
        merged.append(current)
        i += 1
    return merged


def _extract_infobox(soup) -> dict | None:
    """
    Extract the first infobox table as a header + list of label/value rows.
    Called BEFORE tables are removed from the DOM.
    """
    infobox = None
    for table in soup.find_all("table"):
        classes = " ".join(table.get("class") or [])
        if "infobox" in classes:
            infobox = table
            break

    if not infobox:
        return None

    header = ""
    current_group = ""
    rows   = []

    for tr in infobox.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")

        # Header/group row — Wikipedia infoboxes use th-only rows both for
        # the main title and for grouped subheaders such as "Area".
        if th and not td:
            text = _clean_text(th.get_text(" "))
            if text and len(text) < 100:
                classes = set(th.get("class") or [])
                if not header or classes & {"infobox-above", "infobox-title"}:
                    header = text
                else:
                    current_group = text
            continue

        # Data row — th label + td value
        if th and td:
            label = _clean_text(th.get_text(" "))
            value = _clean_text(td.get_text(" "))
            if label.startswith("•") and current_group:
                label = f"{current_group} {label}"
            # Skip empty, overly long values (images, nested tables), or pure numbers with no label
            if label and value and len(label) < 80 and 2 < len(value) < 250:
                rows.append({"label": label, "value": value})

    if not rows:
        return None

    return {"header": header or "Facts", "rows": rows}


def extract(html: str) -> dict | None:
    """
    Parse article HTML. Returns None if the article has no usable content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise that doesn't affect infobox or main content
    for tag in soup(["script", "style", "sup"]):
        tag.decompose()

    # ── Extract infobox BEFORE removing tables ────────────────────────────────
    infobox_data = _extract_infobox(soup)

    # Now remove tables and junk classes
    for tag in soup(["table"]):
        tag.decompose()

    _JUNK_CLASSES = {"navbox", "navbox-styles", "infobox", "reflist",
                     "mw-editsection", "toc", "sistersitebox",
                     "vertical-navbox", "hatnote", "authority-control"}
    for el in soup.find_all(True):
        classes = set(el.attrs.get("class") or []) if el.attrs else set()
        if classes & _JUNK_CLASSES:
            el.decompose()

    # Scope to mw-parser-output — all article content lives here.
    content = soup.find(class_="mw-parser-output") or soup.find("body") or soup

    # ── Lead: all <p> before any h2/h3 ───────────────────────────────────────
    lead_parts: list[str] = []
    for el in content.children:
        if getattr(el, "name", None) in ("h2", "h3"):
            break
        if getattr(el, "name", None) == "p":
            text = _clean_text(el.get_text(" "))
            if len(text) >= 50 and not text.endswith(":"):
                lead_parts.append(text)
    lead_units = _semantic_units(lead_parts)
    lead = " ".join(lead_units)

    if not lead:
        return None

    # ── Sections: each <p>/<ul>/<ol> is its own paragraph entry ─────────────
    # h2 → h3 nesting produces titles like "Treatment > Chemotherapy",
    # giving embeddings a richer, more discriminative section identity.
    sections:      list[dict] = []
    current_h2:    str | None = None
    current_h3:    str | None = None
    current_paras: list[str]  = []

    def _section_title() -> str | None:
        if current_h2 is None:
            return None
        return f"{current_h2} > {current_h3}" if current_h3 else current_h2

    def _flush():
        title = _section_title()
        if title is None or not current_paras:
            return
        if (current_h2 or "").lower() in _JUNK_SECTIONS:
            return
        if (current_h3 or "").lower() in _JUNK_SECTIONS:
            return
        sections.append({"title": title, "paragraphs": _semantic_units(current_paras)})

    def _list_text(el) -> str:
        """Join top-level <li> items into a readable sentence-like string."""
        items = [_clean_text(li.get_text(" ")) for li in el.find_all("li", recursive=False)]
        items = [i for i in items if i]
        return "; ".join(items)

    for el in content.find_all(["h2", "h3", "p", "ul", "ol"], recursive=True):
        if el.name == "h2":
            _flush()
            current_h2 = _clean_text(el.get_text(" "))
            current_h3 = None
            current_paras = []
        elif el.name == "h3":
            _flush()
            current_h3 = _clean_text(el.get_text(" "))
            current_paras = []
        elif el.name == "p":
            text = _clean_text(el.get_text(" "))
            # Skip orphaned list-header paragraphs (end with ":" — their list
            # content follows in a <ul>/<ol> and is captured separately below)
            if len(text) >= 60 and not text.endswith(":"):
                current_paras.append(text)
        elif el.name in ("ul", "ol") and current_h2 is not None:
            text = _list_text(el)
            if len(text) >= 60:
                current_paras.append(text)

    _flush()

    return {"lead": lead, "lead_paragraphs": lead_units, "infobox": infobox_data, "sections": sections}
