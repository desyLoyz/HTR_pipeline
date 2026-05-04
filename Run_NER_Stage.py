"""
============================================================
HistOrniGraph — NER Stage Runner (Step 3, Colab)
============================================================
Runs Named Entity Recognition on every PAGE-XML page produced by the
layout + transcription stages and writes annotated copies under
``pagexml_ner/`` next to the original ``pagexml/`` folder.

Usage in Colab
--------------
1. Mount Google Drive
2. Set GEMINI_API_KEY (e.g. via google.colab.userdata.get('GEMINI_API_KEY')
   or os.environ — the google-genai client picks it up automatically).
3. Set BOOKS_ROOT_DIR (or BOOK_ROOT_DIR for a single book) below.
4. Run all cells.

Output structure
----------------
For every book folder ``<root>/<book>/pagexml/*.xml`` the script writes
``<root>/<book>/pagexml_ner/*.xml``.  The annotated XML files are
identical to the originals plus a ``<NamedEntities>`` block under
``<Page>`` listing every detected entity with its region reference.

The original ``pagexml/`` folder is never modified — re-runs are safe.
By default already-annotated pages are skipped (set ``SKIP_EXISTING=False``
to force re-processing).
============================================================
"""

import os
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# CONFIGURATION — change these for your run
# ═══════════════════════════════════════════════════════════

# Process EITHER one book (BOOK_ROOT_DIR) OR all books under BOOKS_ROOT_DIR.
# Allow notebook overrides via globals().
BOOK_ROOT_DIR  = globals().get("BOOK_ROOT_DIR_OVERRIDE",
    "")  # e.g. "/content/drive/MyDrive/HistOrniGraph_output/Laubmann_01_gemini"

BOOKS_ROOT_DIR = globals().get("BOOKS_ROOT_DIR_OVERRIDE",
    "/content/drive/MyDrive/HistOrniGraph_output")

# Gemini model and thinking level
NER_MODEL_ID        = globals().get("NER_MODEL_ID_OVERRIDE",        "gemini-3-flash-preview")
NER_THINKING_LEVEL  = globals().get("NER_THINKING_LEVEL_OVERRIDE",  "low")

# Skip pages whose annotated XML already exists in pagexml_ner/
SKIP_EXISTING = globals().get("SKIP_EXISTING_OVERRIDE", True)

# Subdirectory names — change only if your output layout differs
PAGEXML_SUBDIR     = "pagexml"
PAGEXML_NER_SUBDIR = "pagexml_ner"


# ═══════════════════════════════════════════════════════════
# CLIENT INITIALISATION
# ═══════════════════════════════════════════════════════════

def _init_client():
    """Create an authenticated google-genai client.

    Picks up the API key from (in order):
      1. ``google.colab.userdata.get('GEMINI_API_KEY')``
      2. ``os.environ['GEMINI_API_KEY']``
      3. ``os.environ['GOOGLE_API_KEY']``
    """
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "google-genai is not installed. Run:\n"
            "    !pip install -q google-genai"
        ) from exc

    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get("GEMINI_API_KEY")
            if key:
                os.environ["GEMINI_API_KEY"] = key
        except Exception:
            pass

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit(
            "No API key found. Set GEMINI_API_KEY in Colab "
            "(Tools → User secrets) or os.environ before running."
        )

    return genai.Client(http_options={"api_version": "v1alpha"})


# ═══════════════════════════════════════════════════════════
# BOOK / PAGE DISCOVERY
# ═══════════════════════════════════════════════════════════

def _list_books(books_root: Path) -> list:
    """Return every immediate sub-folder of ``books_root`` that has a pagexml/ dir."""
    if not books_root.is_dir():
        return []
    out = []
    for sub in sorted(books_root.iterdir()):
        if sub.is_dir() and (sub / PAGEXML_SUBDIR).is_dir():
            out.append(sub)
    return out


def _list_pages(book_dir: Path) -> list:
    """Return every page XML file under ``book_dir/pagexml``, naturally sorted."""
    src = book_dir / PAGEXML_SUBDIR
    if not src.is_dir():
        return []
    files = [p for p in src.iterdir() if p.suffix.lower() == ".xml"]

    def _key(p: Path):
        import re
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r"(\d+)", p.stem)]

    return sorted(files, key=_key)


# ═══════════════════════════════════════════════════════════
# CORE LOOP
# ═══════════════════════════════════════════════════════════

def process_book(client, book_dir: Path) -> dict:
    """Run NER on every page of one book."""
    # Ensure the journal_processor package is importable when this file is
    # placed next to the ``journal_processor/`` folder (typical Colab layout).
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from journal_processor.config import ENTITY_TYPES
    from journal_processor.ner_stage import annotate_pagexml

    src_dir = book_dir / PAGEXML_SUBDIR
    out_dir = book_dir / PAGEXML_NER_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = _list_pages(book_dir)
    print(f"\n📚 {book_dir.name}: {len(pages)} page(s)")
    if not pages:
        return {"book": book_dir.name, "pages": 0, "ok": 0, "skipped": 0,
                "errors": 0, "total_entities": 0}

    ok = skipped = errors = total_ents = 0
    t0 = time.time()

    for idx, page_xml in enumerate(pages, 1):
        out_xml = out_dir / page_xml.name
        try:
            res = annotate_pagexml(
                client=client,
                xml_path=page_xml,
                out_path=out_xml,
                entity_types=ENTITY_TYPES,
                model_id=NER_MODEL_ID,
                thinking_level=NER_THINKING_LEVEL,
                skip_existing=SKIP_EXISTING,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"   [{idx}/{len(pages)}] ✗ {page_xml.stem}: {exc}")
            continue

        if res["status"] == "skipped":
            skipped += 1
        elif res["status"] in ("ok", "empty"):
            ok += 1
            total_ents += max(0, res["n_entities"])
            print(f"   [{idx}/{len(pages)}] ✓ {res['page']}: "
                  f"{res['n_entities']} entit{'y' if res['n_entities'] == 1 else 'ies'}")
        else:
            errors += 1
            print(f"   [{idx}/{len(pages)}] ✗ {res['page']}: {res['status']}")

    elapsed = time.time() - t0
    print(f"   → done in {elapsed:.1f}s   "
          f"ok={ok}  skipped={skipped}  errors={errors}  entities={total_ents}")
    return {"book": book_dir.name, "pages": len(pages),
            "ok": ok, "skipped": skipped, "errors": errors,
            "total_entities": total_ents, "elapsed_s": round(elapsed, 1)}


def main() -> None:
    print("🔬 HistOrniGraph — NER Stage Runner")
    print(f"   Model:    {NER_MODEL_ID}")
    print(f"   Thinking: {NER_THINKING_LEVEL}")
    print(f"   Skip existing: {SKIP_EXISTING}")
    print()

    client = _init_client()

    # Resolve target book(s)
    if BOOK_ROOT_DIR:
        single = Path(BOOK_ROOT_DIR.rstrip("/"))
        if not single.is_dir():
            raise SystemExit(f"❌ BOOK_ROOT_DIR not found: {single}")
        books = [single]
        print(f"   Target: single book → {single}")
    else:
        root = Path(BOOKS_ROOT_DIR.rstrip("/"))
        if not root.is_dir():
            raise SystemExit(f"❌ BOOKS_ROOT_DIR not found: {root}")
        books = _list_books(root)
        print(f"   Target: {len(books)} book(s) under {root}")
        if not books:
            print("   ⚠ No book folders with a pagexml/ subdirectory found.")
            return

    summary = []
    for b in books:
        summary.append(process_book(client, b))

    # Final summary
    print("\n📊 Summary")
    print(f"   {'Book':<32}  {'Pages':>6}  {'OK':>4}  {'Skip':>4}  "
          f"{'Err':>4}  {'Entities':>8}")
    for s in summary:
        print(f"   {s['book']:<32}  {s['pages']:>6}  {s['ok']:>4}  "
              f"{s['skipped']:>4}  {s['errors']:>4}  {s['total_entities']:>8}")

    grand_total = sum(s["total_entities"] for s in summary)
    print(f"\n   ✅ Total entities: {grand_total}")
    print(f"   Output: <book>/{PAGEXML_NER_SUBDIR}/*.xml")
    print(f"\n   Next: re-run Create_GUIs.py for each book to view entities in the GUI.")


if __name__ == "__main__":
    main()
