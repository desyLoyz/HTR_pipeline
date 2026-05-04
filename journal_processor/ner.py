"""
NER stage orchestration
=======================
Reads a PAGE-XML file produced by the layout + transcription stages, runs
Gemini-based NER on each TextRegion's transcription, and writes a new
PAGE-XML file in which:

(1) Each detected entity is encoded **inline** on its TextRegion using the
    Transkribus ``custom``-attribute convention (offset / length syntax),
    e.g.::

        <TextRegion id="r02"
            custom="type:ParagraphRegion
                    namedentity {offset:14; length:7; type:Location;}
                    namedentity {offset:79; length:23; type:Animal;}">

    This is the de-facto standard used by Transkribus, eScriptorium, OCR-D
    and the page2tei converter for inline tagging — ``offset`` is the
    character index into the region's <Unicode> string, ``length`` is the
    number of characters covered by the entity.  Multiple entities can be
    attached to one region, and a single entity that occurs more than once
    in the same region produces one custom-attribute entry per occurrence.

(2) A denormalised ``<NamedEntities>`` block is also written under the
    ``<Page>`` element, with one ``<NamedEntity>`` per occurrence carrying
    ``regionRef``, ``offset``, ``length``, ``type``, ``text`` and an
    optional ``context`` snippet.  This index is regenerated from the
    inline tags on every run so the two representations cannot drift
    apart.  The GUI consumes this index for fast lookup.  Tools that need
    fine-grained token offsets (training-data extraction, evaluation,
    Aletheia / Transkribus integrations) should read the inline ``custom``
    attributes directly via ``parse_inline_custom_entities``.

Original PAGE-XML files in ``output/pagexml`` are NOT modified — annotated
copies are written to ``output/pagexml_ner``.  The Create_GUIs.py viewer
prefers ``pagexml_ner`` if it exists and falls back to ``pagexml`` otherwise.
"""

from __future__ import annotations

import logging
import re
import xml.dom.minidom as minidom
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .ner import Entity, perform_ner

log = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Markup we strip before sending text to the NER prompt — these tags are
# stored *literally* (escaped) inside <Unicode>, so leaving them in place
# would only confuse the model.  Offsets are computed against the RAW
# <Unicode> string so they round-trip cleanly.
_MARKUP_TAG_RE = re.compile(
    r"</?(?:u|sup|sub|s|del|ins|mark|b|i|em|strong|small)\b[^>]*>",
    flags=re.IGNORECASE,
)

# Tag name used in the inline ``custom`` attribute.  Lowercase keeps it in
# line with Transkribus's convention (``person``, ``place``, etc.) while a
# single generic tag avoids forcing downstream tools to know our 11-class
# scheme — the actual class lives in the ``type`` body field.
INLINE_TAG_NAME = "namedentity"

# Region types whose transcribed text is uninteresting for NER.
_NER_SKIP_TYPES = frozenset({"PageNumberRegion", "ImageRegion", "ObjectRegion"})


# ---------------------------------------------------------------------------
# Markup stripping with offset map
# ---------------------------------------------------------------------------

def _strip_with_map(raw: str) -> Tuple[str, List[int]]:
    """Strip <u>/<sup>/… and join soft hyphens; return (clean, raw_idx_of_clean).

    ``raw_idx_of_clean[i]`` gives the offset in ``raw`` that produced
    ``clean[i]``.  The map lets callers project a (start, end) span found
    in the cleaned text back onto raw-text offsets, so the canonical raw
    <Unicode> string remains the single source of truth for the inline
    ``custom`` tags.
    """
    clean_chars: List[str] = []
    raw_idx: List[int] = []

    i = 0
    n = len(raw)
    while i < n:
        # 1. Skip a markup tag like "<u>" or "</sup>"
        m = _MARKUP_TAG_RE.match(raw, i)
        if m:
            i = m.end()
            continue
        # 2. Soft hyphen at line break: "-\n" disappears
        if raw[i] == "-" and i + 1 < n and raw[i + 1] == "\n":
            i += 2
            continue
        # 3. Regular character — keep it
        clean_chars.append(raw[i])
        raw_idx.append(i)
        i += 1

    return "".join(clean_chars), raw_idx


def _project_to_raw(span_in_clean: Tuple[int, int],
                    raw_idx_of_clean: List[int],
                    raw_len: int) -> Tuple[int, int]:
    """Project a half-open ``[start, end)`` interval over the cleaned text
    back to a half-open interval over the raw text that *covers* the same
    span (extending across any intervening markup tags).
    """
    s_clean, e_clean = span_in_clean
    n_clean = len(raw_idx_of_clean)
    if s_clean >= e_clean or n_clean == 0:
        return (raw_len, raw_len)

    raw_start = raw_idx_of_clean[s_clean]
    raw_last = raw_idx_of_clean[min(e_clean - 1, n_clean - 1)]
    return (raw_start, raw_last + 1)


# ---------------------------------------------------------------------------
# Custom-attribute syntax (Transkribus convention)
# ---------------------------------------------------------------------------
#
#   tagName {key1:val1; key2:val2;}
#
# multiple tags space-separated on one attribute:
#
#   readingOrder {index:0;} namedentity {offset:14; length:7; type:Location;}

_TAG_RE = re.compile(
    r"(?P<name>[A-Za-z_][\w\-]*)\s*\{(?P<body>[^}]*)\}",
    flags=re.UNICODE,
)


def _parse_custom(attr: str) -> List[Tuple[str, Dict[str, str]]]:
    """Parse a Transkribus-style custom attribute into ``[(tag, {k:v}), …]``.

    Bareword tokens without a ``{...}`` block (such as the legacy
    ``type:ParagraphRegion`` form already present in this codebase) are
    preserved as a single ``("type:ParagraphRegion", {})`` entry so they
    round-trip through serialisation untouched.
    """
    if not attr:
        return []
    out: List[Tuple[str, Dict[str, str]]] = []
    pos = 0
    for m in _TAG_RE.finditer(attr):
        gap = attr[pos:m.start()].strip()
        if gap:
            for token in gap.split():
                out.append((token, {}))
        body: Dict[str, str] = {}
        for piece in m.group("body").split(";"):
            piece = piece.strip()
            if not piece:
                continue
            if ":" in piece:
                k, _, v = piece.partition(":")
                body[k.strip()] = v.strip()
            else:
                body[piece] = ""
        out.append((m.group("name"), body))
        pos = m.end()
    tail = attr[pos:].strip()
    if tail:
        for token in tail.split():
            out.append((token, {}))
    return out


def _format_custom(parts: List[Tuple[str, Dict[str, str]]]) -> str:
    """Serialise ``[(tag, {k:v}), …]`` back into a custom-attribute string."""
    pieces: List[str] = []
    for name, body in parts:
        if not body:
            pieces.append(name)
            continue
        kv = " ".join(f"{k}:{v};" for k, v in body.items())
        pieces.append(f"{name} {{{kv}}}")
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# PAGE-XML I/O
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    """Strip the XML namespace from an ElementTree tag."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _read_pagexml(
    xml_path: Path,
) -> Tuple[ET.ElementTree, ET.Element, List[Dict[str, Any]]]:
    """Parse a PAGE XML file and return ``(tree, page_element, regions)``.

    ``regions`` is a list of ``{id, type, text, element}`` for every
    TextRegion that has transcribed Unicode content.  ``element`` is the
    live ET element so callers can mutate the ``custom`` attribute in
    place.
    """
    ET.register_namespace("", PAGE_NS)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = None
    for child in root.iter():
        if _local(child.tag) == "Page":
            page = child
            break
    if page is None:
        raise ValueError(f"No <Page> element found in {xml_path}")

    regions: List[Dict[str, Any]] = []
    for child in page:
        if _local(child.tag) != "TextRegion":
            continue
        rid = child.get("id", "")
        custom = child.get("custom", "")
        rtype = ""
        m = re.search(r"type:([A-Za-z]+)", custom)
        if m:
            rtype = m.group(1)

        text = ""
        for sub in child.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                text = sub.text
                break

        if text:
            regions.append({
                "id": rid,
                "type": rtype,
                "text": text,
                "element": child,
            })

    return tree, page, regions


def _write_pretty(tree: ET.ElementTree, out_path: Path) -> None:
    """Pretty-print and write the XML, matching the original generator style.

    Important: we only drop the leading ``<?xml ?>`` declaration that
    minidom adds — we do NOT filter blank lines, because blank lines
    inside the ``<Unicode>`` text content (paragraph breaks in the
    transcription) are semantically significant and shifting them by even
    one character invalidates every offset stored in the inline
    ``custom`` attributes.
    """
    xml_str = ET.tostring(tree.getroot(), encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)
    pretty = "\n".join(pretty.splitlines()[1:])
    out_path.write_text(pretty, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entity → offset bookkeeping
# ---------------------------------------------------------------------------

# Word-character set (Unicode-aware) used to detect word boundaries.  We
# treat anything that is NOT a unicode word character as a boundary so
# that "Mai" inside "Maine" doesn't get matched as a separate entity.
_WORD_RE = re.compile(r"\w", flags=re.UNICODE)


def _find_all_offsets(haystack: str, needle: str,
                      prefer_whole_word: bool = True) -> List[int]:
    """Return every char-index where ``needle`` occurs inside ``haystack``.

    If at least one whole-word match exists and ``prefer_whole_word`` is
    set, only whole-word matches are returned.  Otherwise all substring
    positions are returned (entity strings often contain spaces / dashes
    so naive whole-word filtering would discard real hits).
    """
    if not needle:
        return []
    positions: List[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not prefer_whole_word or not positions:
        return positions

    n = len(needle)
    h_len = len(haystack)
    whole: List[int] = []
    for p in positions:
        left_ok  = p == 0          or not _WORD_RE.match(haystack[p - 1])
        right_ok = p + n == h_len  or not _WORD_RE.match(haystack[p + n])
        if left_ok and right_ok:
            whole.append(p)
    return whole or positions


def _occurrences_for_region(
    raw_text: str,
    clean_text: str,
    raw_idx_of_clean: List[int],
    entities: List[Entity],
) -> List[Dict[str, Any]]:
    """Locate every occurrence of every entity inside one region's raw text.

    The model receives the cleaned text (no markup, hyphens joined) so we
    search the cleaned form first and project the hit back to the raw
    string used for inline offsets.  We then also try a literal raw match
    to catch entities short enough to live entirely outside any markup
    span — the union covers both.

    Returns a list of dicts ``{offset, length, type, text, context}``
    sorted by offset for deterministic output.
    """
    out: List[Dict[str, Any]] = []
    raw_len = len(raw_text)

    for ent in entities:
        seen_raw_starts: set = set()

        # Primary: search in cleaned text, project back to raw.
        for off_clean in _find_all_offsets(clean_text, ent.text):
            raw_start, raw_end = _project_to_raw(
                (off_clean, off_clean + len(ent.text)),
                raw_idx_of_clean, raw_len,
            )
            if raw_start in seen_raw_starts:
                continue
            seen_raw_starts.add(raw_start)
            out.append({
                "offset":  raw_start,
                "length":  raw_end - raw_start,
                "type":    ent.entity_type,
                "text":    raw_text[raw_start:raw_end],
                "context": ent.context,
            })

        # Fallback: literal raw match (covers entities matching raw verbatim).
        for off_raw in _find_all_offsets(raw_text, ent.text):
            if off_raw in seen_raw_starts:
                continue
            seen_raw_starts.add(off_raw)
            out.append({
                "offset":  off_raw,
                "length":  len(ent.text),
                "type":    ent.entity_type,
                "text":    ent.text,
                "context": ent.context,
            })

    out.sort(key=lambda r: (r["offset"], r["length"]))
    return out


def _attach_inline_tags(
    region_el: ET.Element,
    occurrences: List[Dict[str, Any]],
) -> None:
    """Replace existing ``namedentity`` tags on ``region_el`` with fresh ones."""
    parts = _parse_custom(region_el.get("custom", ""))
    parts = [(n, b) for n, b in parts if n != INLINE_TAG_NAME]

    for occ in occurrences:
        body = {
            "offset": str(occ["offset"]),
            "length": str(occ["length"]),
            "type":   occ["type"],
        }
        parts.append((INLINE_TAG_NAME, body))

    serialised = _format_custom(parts)
    if serialised:
        region_el.set("custom", serialised)
    elif "custom" in region_el.attrib:
        del region_el.attrib["custom"]


def _add_named_entities_index(
    page: ET.Element,
    flat: List[Dict[str, Any]],
) -> ET.Element:
    """(Re)build the ``<NamedEntities>`` index block under ``<Page>``.
    """
    for existing in list(page):
        if _local(existing.tag) == "NamedEntities":
            page.remove(existing)

    ns = f"{{{PAGE_NS}}}"
    block = ET.SubElement(page, f"{ns}NamedEntities")
    for rec in flat:
        attrs = {
            "regionRef": rec["regionRef"],
            "offset":    str(rec["offset"]),
            "length":    str(rec["length"]),
            "type":      rec["type"],
        }
        if rec.get("context"):
            attrs["context"] = rec["context"]
        ET.SubElement(block, f"{ns}NamedEntity", attrs)
    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_pagexml(
    client: Any,
    xml_path: Path,
    out_path: Path,
    entity_types: Dict[str, str],
    model_id: str,
    thinking_level: str = "low",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Run NER on one PAGE-XML file and write an annotated copy.

    Returns a small summary dict with keys ``page``, ``status``,
    ``n_entities``.  ``n_entities`` counts *occurrences* (the size of the
    index block), not distinct (text, type) pairs.
    """
    page_id = xml_path.stem

    if skip_existing and out_path.exists():
        return {"page": page_id, "status": "skipped", "n_entities": -1}

    try:
        tree, page_el, regions = _read_pagexml(xml_path)
    except Exception as exc:
        log.error("Failed to parse %s: %s", xml_path, exc)
        return {"page": page_id, "status": "parse_error", "n_entities": 0}

    region_views: List[Dict[str, Any]] = []
    prompt_pieces: List[str] = []
    for r in regions:
        if r["type"] in _NER_SKIP_TYPES:
            continue
        clean, idx_map = _strip_with_map(r["text"])
        if not clean.strip():
            continue
        prompt_pieces.append(clean)
        region_views.append({**r, "clean": clean, "idx_map": idx_map})

    combined = "\n\n".join(prompt_pieces)

    if not combined.strip():
        _add_named_entities_index(page_el, [])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pretty(tree, out_path)
        return {"page": page_id, "status": "empty", "n_entities": 0}

    entities = perform_ner(
        client=client,
        text=combined,
        entity_types=entity_types,
        model_id=model_id,
        thinking_level=thinking_level,
        page_id=page_id,
    )

    flat_index: List[Dict[str, Any]] = []
    for view in region_views:
        occs = _occurrences_for_region(
            view["text"], view["clean"], view["idx_map"], entities,
        )
        _attach_inline_tags(view["element"], occs)
        for occ in occs:
            flat_index.append({
                "regionRef": view["id"],
                "offset":    occ["offset"],
                "length":    occ["length"],
                "type":      occ["type"],
                "text":      occ["text"],
                "context":   occ["context"],
            })

    # Wipe stale namedentity tags from regions excluded from NER
    skipped_ids = {r["id"] for r in regions if r["type"] in _NER_SKIP_TYPES}
    for r in regions:
        if r["id"] in skipped_ids:
            _attach_inline_tags(r["element"], [])

    _add_named_entities_index(page_el, flat_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pretty(tree, out_path)

    return {"page": page_id, "status": "ok", "n_entities": len(flat_index)}


def parse_named_entities(xml_path: Path) -> List[Dict[str, Any]]:
    """Read NamedEntity records from the ``<NamedEntities>`` index block
    Returns ``[{text, entity_type, region_ref, offset, length, context}, …]``.
    """
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Build a region_id → Unicode-text map first
    region_text: Dict[str, str] = {}
    for tr in root.iter():
        if _local(tr.tag) != "TextRegion":
            continue
        rid = tr.get("id", "")
        for sub in tr.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                region_text[rid] = sub.text
                break

    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _local(el.tag) != "NamedEntity":
            continue

        def _to_int(s: Optional[str]) -> Optional[int]:
            try:
                return int(s) if s is not None and s != "" else None
            except (TypeError, ValueError):
                return None

        rid    = el.get("regionRef") or ""
        offset = _to_int(el.get("offset"))
        length = _to_int(el.get("length"))

        if rid and offset is not None and length is not None:
            src = region_text.get(rid, "")
            text = src[offset:offset + length] if 0 <= offset < len(src) else ""
        else:
            # Legacy files might still carry text="…" — fall back to it
            text = el.get("text", "")

        out.append({
            "text":        text,
            "entity_type": el.get("type", ""),
            "region_ref":  rid or None,
            "offset":      offset,
            "length":      length,
            "context":     el.get("context") or None,
        })
    return out


def parse_inline_custom_entities(xml_path: Path) -> List[Dict[str, Any]]:
    """Read entities from the inline TextRegion ``custom`` attributes.

    This is the canonical store; ``parse_named_entities`` reads the
    denormalised index block.
    """
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _local(el.tag) != "TextRegion":
            continue
        rid = el.get("id", "")
        custom = el.get("custom", "")
        if not custom or INLINE_TAG_NAME not in custom:
            continue
        text = ""
        for sub in el.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                text = sub.text
                break
        for name, body in _parse_custom(custom):
            if name != INLINE_TAG_NAME:
                continue
            try:
                offset = int(body.get("offset", "-1"))
                length = int(body.get("length", "-1"))
            except ValueError:
                continue
            etype = body.get("type", "")
            etext = text[offset:offset + length] if 0 <= offset < len(text) else ""
            out.append({
                "region_ref":  rid,
                "offset":      offset,
                "length":      length,
                "entity_type": etype,
                "text":        etext,
            })
    return out
