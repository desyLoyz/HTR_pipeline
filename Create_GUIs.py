"""
============================================================
HistOrniGraph — Transcription Validation GUI Generator
============================================================
Run this script in Google Colab to generate a standalone HTML
validation interface for each book in your corpus.

Images are referenced via Google Drive URLs so the HTML file
works standalone — no need to keep folder structure intact.

Usage in Colab:
    1. Mount Google Drive
    2. Set BOOK_ROOT_DIR below
    3. Run all cells
    4. HTML file is saved into the book root folder
    5. Download the HTML and open in any browser

IMPORTANT: The 'pages' folder must be shared as
"Anyone with the link can view" for images to load.
The script will offer to do this automatically.

SESSION PERSISTENCE:
    - Auto-saves progress to browser localStorage after every edit
    - On re-opening the same HTML file in the same browser, a restore
      banner appears offering to pick up where you left off
    - Export edits as JSON (includes last-viewed page)
    - Import a previously exported JSON to restore edits + position
      (works across browsers or machines)

VALIDATION:
    - Mark pages as "Done" after correcting transcription
    - Mark pages as "Redo" to flag them for reprocessing
    - Automatic CER (Character Error Rate) and WER (Word Error Rate)
      calculation between original and corrected text
    - Metrics are displayed inline and saved to the exported JSON

IMAGE ROTATION:
    - Rotate the page image 90° CW/CCW or 180° using the controls
      in the image panel header
    - PageXML region overlays are re-projected to match the rotated view
    - Rotation is saved per-page in the exported JSON
============================================================
"""

import os
import re
import json
import html as html_module
from pathlib import Path
from xml.etree import ElementTree as ET

# ═══════════════════════════════════════════════════════════
# CONFIGURATION — Change these for each book
# ═══════════════════════════════════════════════════════════
BOOK_ROOT_DIR = globals().get('BOOK_ROOT_DIR_OVERRIDE', 
    "/content/drive/MyDrive/HistOrniGraph_output/Laubmann_01_gemini")
    
# Subdirectory names (change only if your structure differs)
MD_SUBDIR = "md"
PAGES_SUBDIR = "pages"
PAGEXML_SUBDIR = "pagexml"
PAGEXML_NER_SUBDIR = "pagexml_ner"  # preferred over PAGEXML_SUBDIR if it exists

# Output filename
OUTPUT_FILENAME = "Laubmann_01_gemini_validation_gui.html"

# Image extensions to look for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Set to True to automatically share the pages folder (anyone with link)
AUTO_SHARE_PAGES_FOLDER = True

# ── Named-Entity tag set (mirror of journal_processor/config.py) ──────────
# Kept here verbatim so this script stays standalone (Colab cell paste-friendly).
ENTITY_COLORS = {
    "Animal":               "#c62828",
    "Artefact":             "#e65100",
    "Environment":          "#2e7d32",
    "Environmental Impact": "#bf360c",
    "Person":               "#6a1b9a",
    "Location":             "#1565c0",
    "Organisation":         "#37474f",
    "Natural Object":       "#5d4037",
    "Plant":                "#558b2f",
    "Resource":             "#f9a825",
    "Climate":              "#546e7a",
}
ENTITY_LABELS = {
    "Animal":               "Tiere",
    "Artefact":             "Artefakte",
    "Environment":          "Umgebung",
    "Environmental Impact": "Umwelteinflüsse",
    "Person":               "Personen",
    "Location":             "Orte",
    "Organisation":         "Organisationen",
    "Natural Object":       "Naturobjekte",
    "Plant":                "Pflanzen",
    "Resource":             "Ressourcen",
    "Climate":              "Klima",
}


# ═══════════════════════════════════════════════════════════
# GOOGLE DRIVE HELPERS
# ═══════════════════════════════════════════════════════════

_drive_service = None

def get_drive_service():
    """Authenticate and return a Google Drive API service."""
    global _drive_service
    if _drive_service is not None:
        return _drive_service

    try:
        from google.colab import auth
        from googleapiclient.discovery import build
        auth.authenticate_user()
        _drive_service = build('drive', 'v3')
        print("   ✓ Google Drive API authenticated")
        return _drive_service
    except ImportError:
        print("   ⚠ Not running in Colab — Drive API unavailable.")
        print("     Images will use relative paths as fallback.")
        return None


def find_drive_folder_id(drive_path_from_mydrive):
    service = get_drive_service()
    if not service:
        return None

    parts = [p for p in drive_path_from_mydrive.split("/") if p]
    parent_id = "root"

    for part in parts:
        escaped = part.replace("'", "\\'")
        results = service.files().list(
            q=f"name='{escaped}' and '{parent_id}' in parents "
              f"and mimeType='application/vnd.google-apps.folder' "
              f"and trashed=false",
            fields="files(id, name)",
            pageSize=10
        ).execute()
        files = results.get("files", [])
        if not files:
            print(f"   ⚠ Folder not found in Drive: '{part}' under parent {parent_id}")
            return None
        parent_id = files[0]["id"]

    return parent_id


def list_files_in_folder(folder_id):
    service = get_drive_service()
    if not service:
        return {}

    file_map = {}
    page_token = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name)",
            pageSize=500,
            pageToken=page_token
        ).execute()

        for f in results.get("files", []):
            file_map[f["name"]] = f["id"]

        page_token = results.get("nextPageToken")
        if not page_token:
            break

    return file_map


def share_folder_anyone_with_link(folder_id):
    service = get_drive_service()
    if not service:
        return False

    try:
        service.permissions().create(
            fileId=folder_id,
            body={"type": "anyone", "role": "reader"},
            fields="id"
        ).execute()
        return True
    except Exception as e:
        print(f"   ⚠ Could not share folder: {e}")
        return False


def get_drive_relative_path(local_path):
    markers = ["/content/drive/MyDrive/", "/content/drive/My Drive/"]
    for marker in markers:
        if local_path.startswith(marker):
            return local_path[len(marker):]
    return None


def build_image_url(file_id):
    return f"https://drive.google.com/thumbnail?id={file_id}&sz=w2000"


def resolve_image_urls(root_dir):
    pages_local = os.path.join(root_dir, PAGES_SUBDIR)
    drive_rel = get_drive_relative_path(pages_local)

    if drive_rel is None:
        print("   ⚠ Could not determine Drive path. Using relative paths.")
        return {}

    print(f"   Resolving Drive folder: My Drive/{drive_rel}")
    folder_id = find_drive_folder_id(drive_rel)
    if not folder_id:
        return {}

    print(f"   ✓ Found pages folder ID: {folder_id}")

    if AUTO_SHARE_PAGES_FOLDER:
        print("   Sharing pages folder (anyone with link → viewer)...")
        if share_folder_anyone_with_link(folder_id):
            print("   ✓ Folder shared successfully")

    print("   Listing image files...")
    file_map = list_files_in_folder(folder_id)
    print(f"   ✓ Found {len(file_map)} files in pages folder")

    url_map = {}
    for filename, file_id in file_map.items():
        url_map[filename] = build_image_url(file_id)

    return url_map


# ═══════════════════════════════════════════════════════════
# FILE DISCOVERY & PARSING
# ═══════════════════════════════════════════════════════════

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]


def discover_files(root_dir):
    md_dir = os.path.join(root_dir, MD_SUBDIR)
    pages_dir = os.path.join(root_dir, PAGES_SUBDIR)

    # Prefer NER-annotated pagexml when present, otherwise fall back to the
    # original pagexml from the layout/HTR stages.
    pagexml_ner_dir = os.path.join(root_dir, PAGEXML_NER_SUBDIR)
    pagexml_plain_dir = os.path.join(root_dir, PAGEXML_SUBDIR)
    if os.path.isdir(pagexml_ner_dir) and any(
        f.endswith(".xml") for f in os.listdir(pagexml_ner_dir)
    ):
        pagexml_dir = pagexml_ner_dir
        ner_active = True
    else:
        pagexml_dir = pagexml_plain_dir
        ner_active = False

    md_files = {}
    if os.path.isdir(md_dir):
        for f in os.listdir(md_dir):
            if f.endswith(".md"):
                md_files[Path(f).stem] = os.path.join(md_dir, f)

    page_files = {}
    if os.path.isdir(pages_dir):
        for f in os.listdir(pages_dir):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                page_files[Path(f).stem] = f

    pagexml_files = {}
    if os.path.isdir(pagexml_dir):
        for f in os.listdir(pagexml_dir):
            if f.endswith(".xml"):
                pagexml_files[Path(f).stem] = os.path.join(pagexml_dir, f)

    all_stems = sorted(
        set(list(md_files.keys()) + list(page_files.keys()) + list(pagexml_files.keys())),
        key=natural_sort_key
    )

    pages = []
    for stem in all_stems:
        pages.append({
            "stem": stem,
            "image_filename": page_files.get(stem, None),
            "md_path": md_files.get(stem, None),
            "pagexml_path": pagexml_files.get(stem, None),
        })
    return pages, ner_active


def parse_pagexml(xml_path):
    if xml_path is None or not os.path.exists(xml_path):
        return {"width": 0, "height": 0, "regions": [], "entities": []}

    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {}
    m = re.match(r'\{(.*?)\}', root.tag)
    if m:
        ns['p'] = m.group(1)

    def find_first(element, tag):
        if ns:
            return element.find(f'p:{tag}', ns)
        return element.find(tag)

    page_el = find_first(root, 'Page')
    if page_el is None:
        page_el = root.find('.//Page')
    if page_el is None:
        for child in root:
            if 'Page' in child.tag:
                page_el = child
                break

    page_width = int(page_el.get('imageWidth', 0)) if page_el is not None else 0
    page_height = int(page_el.get('imageHeight', 0)) if page_el is not None else 0

    regions = []
    raw_entity_elements = []  # collected first, resolved against region texts below
    region_text_by_id = {}    # region id → <Unicode> text content

    if page_el is not None:
        for region in page_el:
            tag_name = region.tag.split('}')[-1] if '}' in region.tag else region.tag

            # ── NamedEntities block (added by ner_stage.py) ─────────────
            if tag_name == 'NamedEntities':
                for ent in region:
                    ent_tag = ent.tag.split('}')[-1] if '}' in ent.tag else ent.tag
                    if ent_tag != 'NamedEntity':
                        continue
                    raw_entity_elements.append(ent)
                continue

            region_id = region.get('id', '')
            region_type = region.get('type', tag_name)
            custom = region.get('custom', '')

            # Capture the region's transcription so we can slice entities
            # from it later (the canonical text source for offset/length).
            for child in region:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag == 'TextEquiv':
                    for sub in child:
                        sub_tag = sub.tag.split('}')[-1] if '}' in sub.tag else sub.tag
                        if sub_tag == 'Unicode' and sub.text:
                            region_text_by_id[region_id] = sub.text
                            break
                    break

            coords_el = None
            for child in region:
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag == 'Coords':
                    coords_el = child
                    break

            if coords_el is not None:
                points_str = coords_el.get('points', '')
                if points_str:
                    points = []
                    for pt in points_str.strip().split():
                        parts = pt.split(',')
                        if len(parts) == 2:
                            points.append([int(parts[0]), int(parts[1])])
                    if points:
                        regions.append({
                            "id": region_id,
                            "type": region_type,
                            "custom": custom,
                            "points": points,
                        })

    # ── Resolve entity text from region <Unicode> using offset+length ────
    #
    # Storing offset+length on <NamedEntity> and slicing at read time is
    # the format used by ner_stage.py (Transkribus-style inline tags).
    # Older files that still carry text="…" are handled by falling back
    # to that attribute when offset/length are absent or invalid.
    def _to_int(s):
        try:
            return int(s) if s not in (None, "") else None
        except (TypeError, ValueError):
            return None

    entities = []
    for ent in raw_entity_elements:
        rid    = ent.get("regionRef", "")
        offset = _to_int(ent.get("offset"))
        length = _to_int(ent.get("length"))
        text   = ""
        if rid and offset is not None and length is not None:
            src = region_text_by_id.get(rid, "")
            if 0 <= offset < len(src) and length > 0:
                text = src[offset:offset + length]
        if not text:
            text = ent.get("text", "")  # legacy fallback
        entities.append({
            "text":      text,
            "type":      ent.get("type", ""),
            "regionRef": rid,
            "offset":    offset if offset is not None else -1,
            "length":    length if length is not None else -1,
            "context":   ent.get("context", ""),
        })

    return {
        "width": page_width,
        "height": page_height,
        "regions": regions,
        "entities": entities,
    }


def read_markdown(md_path):
    if md_path is None or not os.path.exists(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


# ═══════════════════════════════════════════════════════════
# BUILD DATA
# ═══════════════════════════════════════════════════════════

def build_data(root_dir, image_urls):
    pages_info, ner_active = discover_files(root_dir)
    book_name = os.path.basename(root_dir.rstrip("/"))

    data = {
        "bookName": book_name,
        "nerActive": ner_active,
        "entityColors": ENTITY_COLORS,
        "entityLabels": ENTITY_LABELS,
        "pages": [],
    }

    for p in pages_info:
        md_content = read_markdown(p["md_path"])
        pagexml_data = parse_pagexml(p["pagexml_path"])

        img_url = None
        if p["image_filename"]:
            img_url = image_urls.get(
                p["image_filename"],
                f'{PAGES_SUBDIR}/{p["image_filename"]}'
            )

        data["pages"].append({
            "stem": p["stem"],
            "imageFilename": p["image_filename"],
            "imagePath": img_url,
            "markdown": md_content,
            "pagexml": pagexml_data,
        })

    return data


# ═══════════════════════════════════════════════════════════
# GENERATE HTML
# ═══════════════════════════════════════════════════════════

def generate_html(data):
    data_json = json.dumps(data, ensure_ascii=False)

    html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>""" + html_module.escape(data["bookName"]) + r""" — Transcription Validator</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Literata:ital,opsz,wght@0,7..72,300;0,7..72,400;0,7..72,600;1,7..72,400&display=swap');

  :root {
    --bg-primary: #faf8f5;
    --bg-secondary: #f0ece6;
    --bg-tertiary: #e8e3db;
    --bg-panel: #ffffff;
    --border: #d4cfc7;
    --border-active: #8b6914;
    --text-primary: #2c2418;
    --text-secondary: #6b6050;
    --text-muted: #968c7a;
    --accent: #8b6914;
    --accent-subtle: rgba(139, 105, 20, 0.08);
    --accent-hover: #a37d1a;
    --success: #3a7d44;
    --success-subtle: rgba(58, 125, 68, 0.08);
    --warning: #c07d16;
    --warning-subtle: rgba(192, 125, 22, 0.10);
    --danger: #c04030;
    --region-text: rgba(30, 100, 200, 0.5);
    --region-image: rgba(40, 160, 60, 0.5);
    --region-separator: rgba(200, 140, 20, 0.5);
    --region-other: rgba(200, 60, 50, 0.5);
    --region-table: rgba(140, 80, 220, 0.5);
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
    --shadow-md: 0 2px 8px rgba(0,0,0,0.08);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Literata', Georgia, serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* ── TOP BAR ── */
  .topbar {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    padding: 0 20px;
    height: 52px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
    z-index: 100;
    box-shadow: var(--shadow-sm);
  }

  .topbar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.3px;
    white-space: nowrap;
  }

  .topbar-sep {
    width: 1px;
    height: 24px;
    background: var(--border);
  }

  .page-nav {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .page-nav button {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    color: var(--text-primary);
    width: 32px;
    height: 32px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    font-size: 16px;
  }

  .page-nav button:hover { border-color: var(--accent); background: var(--accent-subtle); }
  .page-nav button:disabled { opacity: 0.3; cursor: default; border-color: var(--border); background: var(--bg-panel); }

  .page-indicator {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-secondary);
    min-width: 80px;
    text-align: center;
  }

  .page-indicator strong { color: var(--text-primary); }

  .page-select {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    max-width: 200px;
  }

  /* ── PROGRESS COUNTER ── */
  .progress-counter {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }

  .progress-counter .progress-bar-bg {
    width: 60px;
    height: 5px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-counter .progress-bar-fill {
    height: 100%;
    background: var(--success);
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .topbar-controls {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .toggle-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .toggle-btn:hover { border-color: var(--accent); color: var(--text-primary); }
  .toggle-btn.active { background: var(--accent-subtle); border-color: var(--accent); color: var(--accent); font-weight: 500; }

  .save-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--success);
    background: var(--success-subtle);
    color: var(--success);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
    font-weight: 500;
  }

  .save-btn:hover { background: rgba(58, 125, 68, 0.15); }
  .save-btn.has-changes { animation: pulse-green 2s infinite; }

  /* ── IMPORT BUTTON ── */
  .import-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .import-btn:hover { border-color: var(--accent); color: var(--text-primary); background: var(--accent-subtle); }

  @keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(58, 125, 68, 0.25); }
    50% { box-shadow: 0 0 0 6px rgba(58, 125, 68, 0); }
  }

  /* ── RESTORE BANNER ── */
  .restore-banner {
    background: var(--warning-subtle);
    border-bottom: 1px solid rgba(192, 125, 22, 0.25);
    padding: 0 20px;
    height: 40px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-secondary);
    animation: slideDown 0.2s ease;
  }

  @keyframes slideDown {
    from { height: 0; opacity: 0; padding-top: 0; padding-bottom: 0; }
    to   { height: 40px; opacity: 1; }
  }

  .restore-banner .banner-icon { font-size: 14px; }
  .restore-banner .banner-text { flex: 1; }
  .restore-banner .banner-text strong { color: var(--text-primary); }

  .restore-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 5px;
    border: 1px solid var(--warning);
    background: rgba(192, 125, 22, 0.15);
    color: var(--warning);
    cursor: pointer;
    font-weight: 500;
    transition: all 0.15s;
  }

  .restore-btn:hover { background: rgba(192, 125, 22, 0.25); }

  .dismiss-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 10px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .dismiss-btn:hover { color: var(--text-secondary); border-color: var(--text-muted); }

  /* ── TOAST NOTIFICATION ── */
  .toast {
    position: fixed;
    bottom: 40px;
    left: 50%;
    transform: translateX(-50%) translateY(0);
    background: var(--text-primary);
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    padding: 9px 18px;
    border-radius: 7px;
    z-index: 9999;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s, transform 0.2s;
    white-space: nowrap;
    box-shadow: var(--shadow-md);
  }

  .toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(-4px);
  }

  /* ── MAIN SPLIT ── */
  .main-container {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-width: 0;
  }

  .panel-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    flex-wrap: wrap;
  }

  .panel-header-right {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .divider {
    width: 4px;
    background: var(--border);
    cursor: col-resize;
    flex-shrink: 0;
    transition: background 0.15s;
  }

  .divider:hover, .divider.dragging { background: var(--accent); }

  /* ── ROTATION CONTROLS ── */
  .rotation-controls {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .rot-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    width: 26px;
    height: 26px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    padding: 0;
    line-height: 1;
  }

  .rot-btn:hover { border-color: var(--accent); color: var(--text-primary); background: var(--accent-subtle); }
  .rot-btn.active { background: var(--accent-subtle); border-color: var(--accent); color: var(--accent); }

  .rot-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    min-width: 28px;
    text-align: center;
  }

  /* ── IMAGE PANEL ── */
  .image-viewport {
    flex: 1;
    overflow: hidden;
    position: relative;
    background: var(--bg-tertiary);
    cursor: grab;
  }

  .image-viewport:active { cursor: grabbing; }

  .image-container {
    position: absolute;
    transform-origin: 0 0;
    will-change: transform;
  }

  .image-container img {
    display: block;
    max-width: none;
    image-rendering: auto;
  }

  .image-container img.loading { opacity: 0.3; }

  .region-overlay {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  }

  .region-polygon {
    fill-opacity: 0.12;
    stroke-width: 2.5;
    stroke-opacity: 0.85;
  }

  .region-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    fill: #fff;
    paint-order: stroke;
    stroke: rgba(0,0,0,0.7);
    stroke-width: 3px;
    pointer-events: none;
  }

  .zoom-controls {
    position: absolute;
    bottom: 12px;
    right: 12px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    z-index: 10;
  }

  .zoom-controls button {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-primary);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s;
    box-shadow: var(--shadow-sm);
  }

  .zoom-controls button:hover { border-color: var(--accent); background: var(--accent-subtle); }

  .zoom-level {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    text-align: center;
    padding: 2px;
    background: var(--bg-panel);
    border-radius: 4px;
    box-shadow: var(--shadow-sm);
  }

  /* ── EDITOR PANEL ── */
  .editor-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .editor-tabs {
    display: flex;
    gap: 0;
    padding: 0 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }

  .editor-tab {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 8px 14px;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
  }

  .editor-tab:hover { color: var(--text-secondary); }
  .editor-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .editor-content {
    flex: 1;
    overflow: hidden;
    position: relative;
  }

  .editor-textarea {
    width: 100%;
    height: 100%;
    background: var(--bg-panel);
    color: var(--text-primary);
    border: none;
    padding: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.75;
    resize: none;
    outline: none;
    tab-size: 4;
  }

  .editor-textarea::selection { background: rgba(139, 105, 20, 0.15); }
  .editor-textarea::placeholder { color: var(--text-muted); }

  /* ── MARKDOWN PREVIEW ── */
  .markdown-preview {
    width: 100%;
    height: 100%;
    overflow-y: auto;
    padding: 32px 36px;
    background: var(--bg-panel);
    display: none;
    line-height: 1.8;
  }

  .markdown-preview h1 { font-size: 1.65em; margin: 0.8em 0 0.4em; font-weight: 600; color: var(--text-primary); border-bottom: 2px solid var(--bg-tertiary); padding-bottom: 0.25em; }
  .markdown-preview h2 { font-size: 1.35em; margin: 0.7em 0 0.35em; font-weight: 600; color: var(--text-primary); border-bottom: 1px solid var(--bg-tertiary); padding-bottom: 0.2em; }
  .markdown-preview h3 { font-size: 1.12em; margin: 0.6em 0 0.25em; font-weight: 600; color: var(--text-secondary); }
  .markdown-preview p { margin: 0.55em 0; line-height: 1.85; color: var(--text-primary); text-align: justify; hyphens: auto; }
  .markdown-preview ul, .markdown-preview ol { margin: 0.5em 0 0.5em 1.6em; }
  .markdown-preview li { margin: 0.3em 0; line-height: 1.7; }
  .markdown-preview em { font-style: italic; }
  .markdown-preview strong { font-weight: 600; color: var(--text-primary); }
  .markdown-preview code { font-family: 'IBM Plex Mono', monospace; font-size: 0.85em; background: var(--bg-secondary); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border); color: var(--accent); }
  .markdown-preview blockquote { border-left: 3px solid var(--accent); padding: 8px 16px; color: var(--text-secondary); margin: 0.6em 0; font-style: italic; background: var(--accent-subtle); border-radius: 0 6px 6px 0; }
  .markdown-preview table { border-collapse: collapse; margin: 0.6em 0; width: 100%; font-size: 0.92em; }
  .markdown-preview th, .markdown-preview td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
  .markdown-preview th { background: var(--bg-secondary); font-weight: 600; color: var(--text-primary); }
  .markdown-preview td { background: var(--bg-panel); }
  .markdown-preview tr:hover td { background: var(--accent-subtle); }
  .markdown-preview hr { border: none; height: 1px; background: var(--border); margin: 1.2em 0; }
  .markdown-preview a { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }

  /* ── DONE BUTTON & METRICS ── */
  .done-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .done-btn:hover { border-color: var(--success); color: var(--success); background: var(--success-subtle); }
  .done-btn.is-done { border-color: var(--success); background: var(--success); color: #fff; font-weight: 500; }
  .done-btn.is-done:hover { background: #2e6636; border-color: #2e6636; }

  /* ── REDO BUTTON ── */
  .redo-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .redo-btn:hover { border-color: var(--warning); color: var(--warning); background: var(--warning-subtle); }
  .redo-btn.is-redo { border-color: var(--warning); background: var(--warning); color: #fff; font-weight: 500; }
  .redo-btn.is-redo:hover { background: #a36812; border-color: #a36812; }

  .metrics-bar {
    flex-shrink: 0;
    background: var(--success-subtle);
    border-bottom: 1px solid rgba(58, 125, 68, 0.18);
    padding: 0 16px;
    height: 0;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-secondary);
    transition: height 0.25s ease;
  }

  .metrics-bar.visible { height: 38px; }

  .metric-item { display: flex; align-items: center; gap: 6px; }
  .metric-label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
  .metric-value { font-weight: 600; font-size: 12px; }
  .metric-value.excellent { color: var(--success); }
  .metric-value.good { color: #5a9e3a; }
  .metric-value.moderate { color: var(--warning); }
  .metric-value.poor { color: var(--danger); }
  .metric-sep { width: 1px; height: 18px; background: rgba(58, 125, 68, 0.2); }
  .metrics-bar .done-timestamp { margin-left: auto; font-size: 10px; color: var(--text-muted); }

  /* ── PAGE SELECT done/redo indicators ── */
  .page-select option.opt-done { color: var(--success); }
  .page-select option.opt-redo { color: var(--warning); }

  /* ── STATUS BAR ── */
  .statusbar {
    height: 28px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--text-muted);
    gap: 16px;
    flex-shrink: 0;
  }

  .status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 4px; }
  .status-dot.clean { background: var(--success); }
  .status-dot.dirty { background: var(--warning); }

  .legend { display: flex; gap: 12px; align-items: center; }
  .legend-item { display: flex; align-items: center; gap: 4px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: var(--text-muted); }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; border: 1px solid rgba(0,0,0,0.15); }

  .no-image { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-family: 'IBM Plex Mono', monospace; font-size: 13px; flex-direction: column; gap: 8px; }
  .no-image .hint { font-size: 11px; max-width: 380px; text-align: center; line-height: 1.5; }

  kbd { display: inline-block; padding: 1px 5px; font-family: 'IBM Plex Mono', monospace; font-size: 10px; background: var(--bg-panel); border: 1px solid var(--border); border-radius: 3px; color: var(--text-muted); box-shadow: 0 1px 0 var(--border); }

  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

  .image-loading-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: none; flex-direction: column; align-items: center; gap: 10px; z-index: 5; }
  .image-loading-overlay.active { display: flex; }
  .spinner { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner-text { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--text-muted); }

  /* ── Named-Entity highlighting (preview tab) ── */
  .entity {
    padding: 1px 3px;
    border-radius: 3px;
    border-bottom: 2px solid currentColor;
    background-color: color-mix(in srgb, currentColor 14%, transparent);
    color: inherit;
  }
  .entity-tag {
    display: inline-block;
    margin-left: 4px;
    padding: 0 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 600;
    line-height: 14px;
    color: #fff;
    border-radius: 2px;
    vertical-align: 2px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
  }
  .entity-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 10px;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    align-items: center;
  }
  .entity-legend.hidden { display: none; }
  .entity-legend-title {
    font-weight: 600;
    color: var(--text-secondary);
    margin-right: 4px;
  }
  .entity-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    cursor: default;
    user-select: none;
    color: var(--text-secondary);
  }
  .entity-legend-item.empty { opacity: 0.35; }
  .entity-legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  .entity-legend-count {
    font-weight: 600;
    color: var(--text-primary);
  }

  @media (max-width: 800px) {
    .main-container { flex-direction: column; }
    .divider { width: auto; height: 4px; cursor: row-resize; }
  }
</style>
</head>
<body>

<!-- Top Bar -->
<div class="topbar">
  <div class="topbar-title" id="bookTitle"></div>
  <div class="topbar-sep"></div>
  <div class="page-nav">
    <button id="btnPrev" title="Previous (&#8592;)">&#8249;</button>
    <span class="page-indicator"><strong id="pageNum">1</strong> / <span id="pageTotal">0</span></span>
    <button id="btnNext" title="Next (&#8594;)">&#8250;</button>
    <select class="page-select" id="pageSelect"></select>
  </div>
  <div class="topbar-sep"></div>
  <div class="progress-counter" id="progressCounter">
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressBarFill" style="width:0%"></div></div>
    <span id="progressText">0 / 0 done</span>
  </div>
  <div class="topbar-controls">
    <button class="toggle-btn" id="btnRegions" title="Toggle layout regions (R)">&#9638; Regions</button>
    <button class="toggle-btn active" id="btnEntities" title="Toggle named-entity highlights (E)">&#9873; Entities</button>
    <button class="toggle-btn" id="btnFitWidth" title="Fit to width (F)">&#10530; Fit</button>
    <button class="import-btn" id="btnImport" title="Import a previously exported session JSON">&#8593; Import session</button>
    <input type="file" id="sessionFileInput" accept=".json" style="display:none">
    <button class="save-btn" id="btnExport" title="Export all edits as JSON">&#8595; Export edits</button>
  </div>
</div>

<!-- Restore Banner -->
<div class="restore-banner" id="restoreBanner" style="display:none">
  <span class="banner-icon">&#128190;</span>
  <span class="banner-text" id="bannerText"></span>
  <button class="restore-btn" id="btnRestore">Restore session</button>
  <button class="dismiss-btn" id="btnDismiss">Dismiss</button>
</div>

<!-- Toast notification -->
<div class="toast" id="toast"></div>

<!-- Main -->
<div class="main-container">
  <div class="panel" id="imagePanel">
    <div class="panel-header">
      <span>Original Page</span>
      <!-- Rotation controls -->
      <div class="rotation-controls" title="Rotate image">
        <button class="rot-btn" id="btnRotCCW" title="Rotate 90° counter-clockwise (&#91;)">&#8634;</button>
        <span class="rot-label" id="rotLabel">0°</span>
        <button class="rot-btn" id="btnRotCW" title="Rotate 90° clockwise (&#93;)">&#8635;</button>
        <button class="rot-btn" id="btnRotReset" title="Reset rotation (Backslash)" style="font-size:10px;width:32px;">0°&#8617;</button>
      </div>
      <div class="legend" id="regionLegend" style="display:none">
        <div class="legend-item"><div class="legend-swatch" style="background:var(--region-text)"></div>Text</div>
        <div class="legend-item"><div class="legend-swatch" style="background:var(--region-image)"></div>Image</div>
        <div class="legend-item"><div class="legend-swatch" style="background:var(--region-table)"></div>Table</div>
        <div class="legend-item"><div class="legend-swatch" style="background:var(--region-separator)"></div>Sep</div>
        <div class="legend-item"><div class="legend-swatch" style="background:var(--region-other)"></div>Other</div>
      </div>
    </div>
    <div class="image-viewport" id="imageViewport">
      <div class="image-container" id="imageContainer">
        <img id="pageImage" alt="Page image" referrerpolicy="no-referrer" />
        <svg class="region-overlay" id="regionOverlay" style="display:none"></svg>
      </div>
      <div class="image-loading-overlay" id="imageLoading">
        <div class="spinner"></div>
        <span class="spinner-text">Loading from Drive...</span>
      </div>
      <div class="no-image" id="noImage" style="display:none">
        <span>No image available</span>
        <span class="hint"></span>
      </div>
      <div class="zoom-controls">
        <button id="btnZoomIn" title="Zoom in (+)">+</button>
        <div class="zoom-level" id="zoomLevel">100%</div>
        <button id="btnZoomOut" title="Zoom out (-)">-</button>
        <button id="btnZoomReset" title="Reset (0)">&#8857;</button>
      </div>
    </div>
  </div>

  <div class="divider" id="divider"></div>

  <div class="panel" id="editorPanel">
    <div class="panel-header">
      <span>Transcription</span>
      <div class="panel-header-right">
        <span id="charCount" style="font-size:10px"></span>
        <button class="redo-btn" id="btnRedo" title="Flag this page for reprocessing (Shift+D)">
          <span id="redoIcon">&#9744;</span>
          <span id="redoLabel">Redo</span>
        </button>
        <button class="done-btn" id="btnDone" title="Mark this page as validated (D)">
          <span id="doneIcon">&#9744;</span>
          <span id="doneLabel">Mark done</span>
        </button>
      </div>
    </div>
    <!-- Metrics bar — shown when page is marked done -->
    <div class="metrics-bar" id="metricsBar">
      <div class="metric-item">
        <span class="metric-label">CER</span>
        <span class="metric-value" id="metricCER">—</span>
      </div>
      <div class="metric-sep"></div>
      <div class="metric-item">
        <span class="metric-label">WER</span>
        <span class="metric-value" id="metricWER">—</span>
      </div>
      <div class="metric-sep"></div>
      <div class="metric-item">
        <span class="metric-label">Edits</span>
        <span class="metric-value" id="metricEdits" style="color:var(--text-primary)">—</span>
      </div>
      <div class="metric-sep"></div>
      <div class="metric-item">
        <span class="metric-label">Rotation</span>
        <span class="metric-value" id="metricRotation" style="color:var(--text-primary)">—</span>
      </div>
      <span class="done-timestamp" id="doneTimestamp"></span>
    </div>
    <div class="editor-wrapper">
      <div class="entity-legend hidden" id="entityLegend"></div>
      <div class="editor-tabs">
        <div class="editor-tab active" data-tab="edit">Edit</div>
        <div class="editor-tab" data-tab="preview">Preview</div>
      </div>
      <div class="editor-content">
        <textarea class="editor-textarea" id="editorTextarea" spellcheck="false" placeholder="No transcription for this page..."></textarea>
        <div class="markdown-preview" id="markdownPreview"></div>
      </div>
    </div>
  </div>
</div>

<!-- Status Bar -->
<div class="statusbar">
  <span><span class="status-dot clean" id="statusDot"></span> <span id="statusText">Clean</span></span>
  <span id="pageStem"></span>
  <span id="autoSaveStatus" style="color:var(--text-muted)"></span>
  <span style="margin-left:auto">
    <kbd>&#8592;</kbd><kbd>&#8594;</kbd> nav &nbsp;
    <kbd>D</kbd> done &nbsp;
    <kbd>Shift+D</kbd> redo &nbsp;
    <kbd>R</kbd> regions &nbsp;
    <kbd>E</kbd> entities &nbsp;
    <kbd>F</kbd> fit &nbsp;
    <kbd>[</kbd><kbd>]</kbd> rotate &nbsp;
    <kbd>+</kbd><kbd>-</kbd> zoom
  </span>
</div>

<script>
const DATA = """ + data_json + r""";

const STORAGE_KEY = 'histornigraph_session_' + DATA.bookName;

let currentPageIdx = 0;
let showRegions = false;
let showEntities = true;     // entity highlighting in preview (default ON)
let zoom = 1, panX = 0, panY = 0;
let isPanning = false, panStartX = 0, panStartY = 0;
let edits = {};
let donePages = {};
let redoPages = {};
// Per-page rotation store: { stem: 0|90|180|270 }
let pageRotations = {};
let imageRotation = 0; // current page's rotation in degrees (0,90,180,270)
let activeTab = 'edit';
let pageLoaded = false;
let autoSaveTimer = null;
let toastTimer = null;

const $ = id => document.getElementById(id);
const bookTitle = $('bookTitle');
const pageNum = $('pageNum');
const pageTotal = $('pageTotal');
const pageSelect = $('pageSelect');
const btnPrev = $('btnPrev');
const btnNext = $('btnNext');
const btnRegions = $('btnRegions');
const btnEntities = $('btnEntities');
const btnFitWidth = $('btnFitWidth');
const btnExport = $('btnExport');
const btnImport = $('btnImport');
const btnDone = $('btnDone');
const btnRedo = $('btnRedo');
const btnRotCW = $('btnRotCW');
const btnRotCCW = $('btnRotCCW');
const btnRotReset = $('btnRotReset');
const rotLabel = $('rotLabel');
const sessionFileInput = $('sessionFileInput');
const restoreBanner = $('restoreBanner');
const bannerText = $('bannerText');
const btnRestore = $('btnRestore');
const btnDismiss = $('btnDismiss');
const regionLegend = $('regionLegend');
const imageViewport = $('imageViewport');
const imageContainer = $('imageContainer');
const pageImage = $('pageImage');
const regionOverlay = $('regionOverlay');
const noImage = $('noImage');
const imageLoading = $('imageLoading');
const zoomLevel = $('zoomLevel');
const editorTextarea = $('editorTextarea');
const markdownPreview = $('markdownPreview');
const statusDot = $('statusDot');
const statusText = $('statusText');
const pageStem = $('pageStem');
const charCount = $('charCount');
const autoSaveStatus = $('autoSaveStatus');
const metricsBar = $('metricsBar');
const metricCER = $('metricCER');
const metricWER = $('metricWER');
const metricEdits = $('metricEdits');
const metricRotation = $('metricRotation');
const doneTimestamp = $('doneTimestamp');
const doneIcon = $('doneIcon');
const doneLabel = $('doneLabel');
const redoIcon = $('redoIcon');
const redoLabel = $('redoLabel');
const progressBarFill = $('progressBarFill');
const progressText = $('progressText');

// ══════════════════════════════════════════════════════════
// TOAST
// ══════════════════════════════════════════════════════════

function showToast(msg, duration) {
  duration = duration || 2800;
  var t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(function() { t.classList.remove('show'); }, duration);
}

// ══════════════════════════════════════════════════════════
// IMAGE ROTATION
// ══════════════════════════════════════════════════════════

/**
 * Rotate a single point (px, py) around the center of image
 * (origW × origH) by `degrees` clockwise.
 * Returns the new [x, y] in the rotated coordinate space.
 *
 * For each rotation step the canvas swaps w/h:
 *   0°  → canvas is origW × origH,  point unchanged
 *   90° → canvas is origH × origW,  (x,y) → (origH-y, x)
 *  180° → canvas is origW × origH,  (x,y) → (origW-x, origH-y)
 *  270° → canvas is origH × origW,  (x,y) → (y, origW-x)
 */
function rotatePoint(px, py, origW, origH, degrees) {
  var d = ((degrees % 360) + 360) % 360;
  if (d === 0)   return [px, py];
  if (d === 90)  return [origH - py, px];
  if (d === 180) return [origW - px, origH - py];
  if (d === 270) return [py, origW - px];
  return [px, py];
}

/**
 * Return the canvas dimensions after rotating origW × origH by degrees.
 */
function rotatedDimensions(origW, origH, degrees) {
  var d = ((degrees % 360) + 360) % 360;
  if (d === 90 || d === 270) return [origH, origW];
  return [origW, origH];
}

function setRotation(deg) {
  imageRotation = ((deg % 360) + 360) % 360;
  var page = DATA.pages[currentPageIdx];
  if (imageRotation === 0) {
    delete pageRotations[page.stem];
  } else {
    pageRotations[page.stem] = imageRotation;
  }
  rotLabel.textContent = imageRotation + '°';

  // Highlight active rotation button
  btnRotCW.classList.toggle('active', imageRotation !== 0);
  btnRotCCW.classList.toggle('active', imageRotation !== 0);

  updateTransform();
  drawRegions();
  scheduleAutoSave();
}

function rotateCW()  { setRotation(imageRotation + 90);  fitToWidth(); }
function rotateCCW() { setRotation(imageRotation - 90);  fitToWidth(); }
function rotateReset(){ setRotation(0); fitToWidth(); }

// ══════════════════════════════════════════════════════════
// CER / WER CALCULATION
// ══════════════════════════════════════════════════════════

function levenshteinDistance(a, b) {
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  if (a.length > b.length) { var tmp = a; a = b; b = tmp; }
  var prev = new Array(a.length + 1);
  var curr = new Array(a.length + 1);
  for (var i = 0; i <= a.length; i++) prev[i] = i;
  for (var j = 1; j <= b.length; j++) {
    curr[0] = j;
    for (var i = 1; i <= a.length; i++) {
      if (a[i - 1] === b[j - 1]) { curr[i] = prev[i - 1]; }
      else { curr[i] = 1 + Math.min(prev[i - 1], prev[i], curr[i - 1]); }
    }
    var swap = prev; prev = curr; curr = swap;
  }
  return prev[a.length];
}

function computeCER(reference, hypothesis) {
  var refChars = Array.from(reference);
  var hypChars = Array.from(hypothesis);
  var dist = levenshteinDistance(refChars, hypChars);
  var refLen = Math.max(refChars.length, 1);
  return { cer: dist / refLen, distance: dist, refLen: refChars.length };
}

function computeWER(reference, hypothesis) {
  var refWords = reference.split(/\s+/).filter(function(w) { return w.length > 0; });
  var hypWords = hypothesis.split(/\s+/).filter(function(w) { return w.length > 0; });
  var dist = levenshteinDistance(refWords, hypWords);
  var refLen = Math.max(refWords.length, 1);
  return { wer: dist / refLen, distance: dist, refLen: refWords.length };
}

function rateTier(rate) {
  if (rate <= 0.02) return 'excellent';
  if (rate <= 0.08) return 'good';
  if (rate <= 0.20) return 'moderate';
  return 'poor';
}

function formatRate(rate) { return (rate * 100).toFixed(2) + '%'; }

// ══════════════════════════════════════════════════════════
// MARK AS DONE
// ══════════════════════════════════════════════════════════

function togglePageDone() {
  saveCurrentEdits();
  var page = DATA.pages[currentPageIdx];
  var stem = page.stem;

  if (donePages[stem]) {
    delete donePages[stem];
    showToast('Page ' + (currentPageIdx + 1) + ' unmarked');
  } else {
    var original = page.markdown;
    var corrected = editorTextarea.value;
    var cerResult = computeCER(original, corrected);
    var werResult = computeWER(original, corrected);

    donePages[stem] = {
      cer: cerResult.cer,
      wer: werResult.wer,
      charEdits: cerResult.distance,
      wordEdits: werResult.distance,
      refChars: cerResult.refLen,
      refWords: werResult.refLen,
      imageRotation: imageRotation,  // ← save rotation at time of marking done
      doneAt: new Date().toISOString()
    };

    showToast('Page ' + (currentPageIdx + 1) + ' done \u2014 CER ' + formatRate(cerResult.cer) + ', WER ' + formatRate(werResult.wer));
  }

  updateDoneDisplay();
  updateRedoDisplay();
  updateProgress();
  updatePageSelectLabels();
  scheduleAutoSave();
}

function updateDoneDisplay() {
  var page = DATA.pages[currentPageIdx];
  var info = donePages[page.stem];

  if (info) {
    btnDone.classList.add('is-done');
    doneIcon.innerHTML = '&#9745;';
    doneLabel.textContent = 'Done';

    metricCER.textContent = formatRate(info.cer);
    metricCER.className = 'metric-value ' + rateTier(info.cer);
    metricWER.textContent = formatRate(info.wer);
    metricWER.className = 'metric-value ' + rateTier(info.wer);
    metricEdits.textContent = info.charEdits + ' chars / ' + info.wordEdits + ' words';
    metricRotation.textContent = (info.imageRotation !== undefined ? info.imageRotation : 0) + '°';

    try {
      var d = new Date(info.doneAt);
      doneTimestamp.textContent = 'Validated ' + d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
        ' at ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    } catch(e) { doneTimestamp.textContent = ''; }

    metricsBar.classList.add('visible');
  } else {
    btnDone.classList.remove('is-done');
    doneIcon.innerHTML = '&#9744;';
    doneLabel.textContent = 'Mark done';
    metricCER.textContent = '\u2014'; metricCER.className = 'metric-value';
    metricWER.textContent = '\u2014'; metricWER.className = 'metric-value';
    metricEdits.textContent = '\u2014';
    metricRotation.textContent = '\u2014';
    doneTimestamp.textContent = '';
    metricsBar.classList.remove('visible');
  }
}

// ══════════════════════════════════════════════════════════
// MARK AS REDO
// ══════════════════════════════════════════════════════════

function togglePageRedo() {
  saveCurrentEdits();
  var page = DATA.pages[currentPageIdx];
  var stem = page.stem;
  if (redoPages[stem]) {
    delete redoPages[stem];
    showToast('Page ' + (currentPageIdx + 1) + ' redo flag removed');
  } else {
    redoPages[stem] = { markedAt: new Date().toISOString() };
    showToast('Page ' + (currentPageIdx + 1) + ' flagged for redo');
  }
  updateRedoDisplay();
  updateProgress();
  updatePageSelectLabels();
  scheduleAutoSave();
}

function updateRedoDisplay() {
  var page = DATA.pages[currentPageIdx];
  var isRedo = !!redoPages[page.stem];
  if (isRedo) {
    btnRedo.classList.add('is-redo');
    redoIcon.innerHTML = '&#9745;';
    redoLabel.textContent = 'Redo';
  } else {
    btnRedo.classList.remove('is-redo');
    redoIcon.innerHTML = '&#9744;';
    redoLabel.textContent = 'Redo';
  }
}

function updateProgress() {
  var total = DATA.pages.length;
  var done = Object.keys(donePages).length;
  var redo = Object.keys(redoPages).length;
  var pct = total > 0 ? (done / total * 100) : 0;
  progressBarFill.style.width = pct.toFixed(1) + '%';
  var txt = done + ' / ' + total + ' done';
  if (redo > 0) txt += ', ' + redo + ' redo';
  progressText.textContent = txt;
}

function updatePageSelectLabels() {
  var options = pageSelect.options;
  for (var i = 0; i < options.length; i++) {
    var stem = DATA.pages[i].stem;
    var isDone = !!donePages[stem];
    var isRedo = !!redoPages[stem];
    var prefix = '';
    if (isDone && isRedo) prefix = '\u2713\u21bb ';
    else if (isDone) prefix = '\u2713 ';
    else if (isRedo) prefix = '\u21bb ';
    options[i].textContent = prefix + stem;
    options[i].classList.toggle('opt-done', isDone && !isRedo);
    options[i].classList.toggle('opt-redo', isRedo);
  }
}

// ══════════════════════════════════════════════════════════
// SESSION PERSISTENCE
// ══════════════════════════════════════════════════════════

function buildSessionObject(pageIdx) {
  var session = {
    _session: {
      bookName: DATA.bookName,
      lastPage: pageIdx !== undefined ? pageIdx : currentPageIdx,
      savedAt: new Date().toISOString(),
      editCount: Object.keys(edits).length,
      doneCount: Object.keys(donePages).length,
      redoCount: Object.keys(redoPages).length,
      rotationCount: Object.keys(pageRotations).length
    },
    _donePages: {},
    _redoPages: {},
    _pageRotations: {}
  };
  Object.keys(edits).forEach(function(stem) { session[stem] = edits[stem]; });
  Object.keys(donePages).forEach(function(stem) { session._donePages[stem] = donePages[stem]; });
  Object.keys(redoPages).forEach(function(stem) { session._redoPages[stem] = redoPages[stem]; });
  Object.keys(pageRotations).forEach(function(stem) { session._pageRotations[stem] = pageRotations[stem]; });
  return session;
}

function autoSaveToLocalStorage() {
  try {
    var session = buildSessionObject();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
    var now = new Date();
    var hh = String(now.getHours()).padStart(2, '0');
    var mm = String(now.getMinutes()).padStart(2, '0');
    var ss = String(now.getSeconds()).padStart(2, '0');
    autoSaveStatus.textContent = 'Auto-saved ' + hh + ':' + mm + ':' + ss;
  } catch(e) { autoSaveStatus.textContent = ''; }
}

function scheduleAutoSave() {
  clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(function() {
    saveCurrentEdits();
    autoSaveToLocalStorage();
  }, 1500);
}

function applySession(session, source) {
  source = source || 'session';
  var meta = session._session || {};
  var restoredEdits = 0, restoredDone = 0, restoredRedo = 0, restoredRot = 0;

  Object.keys(session).forEach(function(key) {
    if (key === '_session' || key === '_donePages' || key === '_redoPages' || key === '_pageRotations') return;
    var val = session[key];
    if (typeof val === 'string') {
      edits[key] = val; restoredEdits++;
    } else if (val && typeof val.edited === 'string') {
      if (val.modified) { edits[key] = val.edited; restoredEdits++; }
    }
  });

  if (session._donePages) {
    Object.keys(session._donePages).forEach(function(stem) {
      donePages[stem] = session._donePages[stem]; restoredDone++;
    });
  }
  if (session._redoPages) {
    Object.keys(session._redoPages).forEach(function(stem) {
      redoPages[stem] = session._redoPages[stem]; restoredRedo++;
    });
  }
  if (session._pageRotations) {
    Object.keys(session._pageRotations).forEach(function(stem) {
      pageRotations[stem] = session._pageRotations[stem]; restoredRot++;
    });
  }

  var targetPage = (meta.lastPage !== undefined) ? parseInt(meta.lastPage) : 0;
  targetPage = Math.max(0, Math.min(targetPage, DATA.pages.length - 1));

  pageLoaded = false;
  loadPage(targetPage);
  updateProgress();
  updatePageSelectLabels();

  var msg = '\u2713 Session restored \u2014 ' + restoredEdits + ' edit(s), ' + restoredDone + ' done, ' + restoredRedo + ' redo';
  if (restoredRot > 0) msg += ', ' + restoredRot + ' rotation(s)';
  msg += ', page ' + (targetPage + 1);
  showToast(msg, 3500);
  autoSaveToLocalStorage();
}

function checkLocalStorageSession() {
  try {
    var raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    var session = JSON.parse(raw);
    var meta = session._session;
    if (!meta || meta.bookName !== DATA.bookName) return;
    var hasData = Object.keys(session).filter(function(k) {
      return k !== '_session' && k !== '_donePages' && k !== '_redoPages' && k !== '_pageRotations';
    }).length > 0;
    if (!hasData && meta.editCount === 0 && (!meta.doneCount || meta.doneCount === 0) &&
        (!meta.redoCount || meta.redoCount === 0) && (!meta.rotationCount || meta.rotationCount === 0)) return;

    var savedAt = '';
    try {
      var d = new Date(meta.savedAt);
      savedAt = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
                ' at ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    } catch(e) { savedAt = meta.savedAt || ''; }

    var editCount = meta.editCount || 0;
    var doneCount = meta.doneCount || 0;
    var redoCount = meta.redoCount || 0;
    var rotCount = meta.rotationCount || 0;
    var lastPageNum = (meta.lastPage !== undefined) ? meta.lastPage + 1 : '?';
    var details = editCount + ' edited, ' + doneCount + ' done';
    if (redoCount > 0) details += ', ' + redoCount + ' redo';
    if (rotCount > 0) details += ', ' + rotCount + ' rotated';
    bannerText.innerHTML =
      'Auto-save found from <strong>' + savedAt + '</strong> \u2014 ' +
      details + ', last on page <strong>' + lastPageNum + '</strong>.';

    restoreBanner._sessionData = session;
    restoreBanner.style.display = 'flex';
  } catch(e) {
    try { localStorage.removeItem(STORAGE_KEY); } catch(e2) {}
  }
}

// ══════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════

function init() {
  bookTitle.textContent = DATA.bookName;
  pageTotal.textContent = DATA.pages.length;
  DATA.pages.forEach(function(p, i) {
    var opt = document.createElement('option');
    opt.value = i;
    opt.textContent = p.stem;
    pageSelect.appendChild(opt);
  });
  loadPage(0);
  setupEvents();
  updateProgress();
  checkLocalStorageSession();
}

// ══════════════════════════════════════════════════════════
// PAGE LOADING
// ══════════════════════════════════════════════════════════

function loadPage(idx) {
  if (idx < 0 || idx >= DATA.pages.length) return;
  saveCurrentEdits();
  currentPageIdx = idx;
  var page = DATA.pages[idx];

  pageNum.textContent = idx + 1;
  pageSelect.value = idx;
  pageStem.textContent = page.stem;
  btnPrev.disabled = idx === 0;
  btnNext.disabled = idx === DATA.pages.length - 1;

  // Restore per-page rotation
  imageRotation = pageRotations[page.stem] || 0;
  rotLabel.textContent = imageRotation + '°';
  btnRotCW.classList.toggle('active', imageRotation !== 0);
  btnRotCCW.classList.toggle('active', imageRotation !== 0);

  if (page.imagePath) {
    noImage.style.display = 'none';
    imageContainer.style.display = 'block';
    imageLoading.classList.add('active');
    pageImage.style.display = 'none';

    pageImage.onload = function() {
      imageLoading.classList.remove('active');
      pageImage.style.display = 'block';
      fitToWidth();
      drawRegions();
    };
    pageImage.onerror = function() {
      imageLoading.classList.remove('active');
      pageImage.style.display = 'none';
      imageContainer.style.display = 'none';
      noImage.style.display = 'flex';
      var hint = noImage.querySelector('.hint');
      if (page.imagePath.includes('drive.google.com')) {
        hint.textContent = 'Drive image failed to load. Ensure the pages folder is shared as "Anyone with the link can view".';
      } else {
        hint.textContent = 'Image path: ' + page.imagePath;
      }
    };
    pageImage.src = page.imagePath;
  } else {
    noImage.style.display = 'flex';
    noImage.querySelector('.hint').textContent = '';
    imageContainer.style.display = 'none';
    imageLoading.classList.remove('active');
  }

  var md = edits[page.stem] !== undefined ? edits[page.stem] : page.markdown;
  editorTextarea.value = md;
  updatePreview(md);
  updateCharCount(md);
  updateDirtyState();
  updateDoneDisplay();
  updateRedoDisplay();
  updateEntityLegend();
  pageLoaded = true;

  scheduleAutoSave();
}

function saveCurrentEdits() {
  if (!pageLoaded || DATA.pages.length === 0) return;
  var page = DATA.pages[currentPageIdx];
  var currentText = editorTextarea.value;
  if (currentText !== page.markdown) {
    edits[page.stem] = currentText;
  } else {
    delete edits[page.stem];
  }
}

// ══════════════════════════════════════════════════════════
// REGIONS / SVG
// ══════════════════════════════════════════════════════════

function drawRegions() {
  var page = DATA.pages[currentPageIdx];
  var pxml = page.pagexml;
  if (!pxml || pxml.regions.length === 0) { regionOverlay.innerHTML = ''; return; }

  // Original image dimensions from PageXML (fall back to natural img size)
  var origW = pxml.width || pageImage.naturalWidth;
  var origH = pxml.height || pageImage.naturalHeight;

  // Canvas dimensions after rotation
  var dims = rotatedDimensions(origW, origH, imageRotation);
  var canvasW = dims[0];
  var canvasH = dims[1];

  regionOverlay.setAttribute('width', canvasW);
  regionOverlay.setAttribute('height', canvasH);
  regionOverlay.setAttribute('viewBox', '0 0 ' + canvasW + ' ' + canvasH);
  regionOverlay.style.width = canvasW + 'px';
  regionOverlay.style.height = canvasH + 'px';

  var svg = '';
  pxml.regions.forEach(function(r) {
    var color = getRegionColor(r.type, r.custom);

    // Rotate all polygon points
    var rotatedPts = r.points.map(function(p) {
      return rotatePoint(p[0], p[1], origW, origH, imageRotation);
    });

    var pts = rotatedPts.map(function(p) { return p.join(','); }).join(' ');
    var minX = Math.min.apply(null, rotatedPts.map(function(p) { return p[0]; }));
    var minY = Math.min.apply(null, rotatedPts.map(function(p) { return p[1]; }));
    var label = r.custom ? r.custom.replace(/^type:/, '') : r.type;
    svg += '<polygon class="region-polygon" points="' + pts + '" fill="' + color + '" stroke="' + color + '" />';
    svg += '<text class="region-label" x="' + (minX + 4) + '" y="' + (minY + 14) + '">' + escHtml(label) + (r.id ? ' #' + escHtml(r.id) : '') + '</text>';
  });
  regionOverlay.innerHTML = svg;
}

function getRegionColor(type, custom) {
  var t = ((custom || '') + ' ' + type).toLowerCase();
  if (t.includes('text') || t.includes('paragraph') || t.includes('heading') || t.includes('caption')) return 'var(--region-text)';
  if (t.includes('image') || t.includes('graphic') || t.includes('figure')) return 'var(--region-image)';
  if (t.includes('table')) return 'var(--region-table)';
  if (t.includes('separator') || t.includes('border')) return 'var(--region-separator)';
  return 'var(--region-other)';
}

function escHtml(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ══════════════════════════════════════════════════════════
// ZOOM / PAN  (rotation is a CSS transform on the img+svg wrapper)
// ══════════════════════════════════════════════════════════

/**
 * Canonical page dimensions for the current page.
 *
 * We trust the PageXML's <Page imageWidth/imageHeight> over the image's
 * naturalWidth/Height because Drive thumbnails are capped at sz=w2000:
 * a 3350-px-wide unsplit page is served to the browser as 2000 px, so
 * naturalWidth=2000 while the PageXML region polygons are in 3350-coord
 * space.  Forcing the rendered image and SVG to share PageXML's coord
 * space realigns full pages without affecting split pages (which are
 * under the cap and so happen to match either way).
 *
 * Falls back to naturalWidth/Height if the PageXML is missing.
 */
function pageOrigDims() {
  var page = DATA.pages[currentPageIdx];
  var pxml = page && page.pagexml;
  var w = (pxml && pxml.width)  || pageImage.naturalWidth  || 0;
  var h = (pxml && pxml.height) || pageImage.naturalHeight || 0;
  return [w, h];
}

function updateTransform() {
  // The image itself is rotated via CSS on the img element.
  // The imageContainer is then panned/zoomed as before.
  // We apply rotation to the img and SVG overlay together via a wrapper transform.
  pageImage.style.transform = 'rotate(' + imageRotation + 'deg)';
  pageImage.style.transformOrigin = 'top left';

  // Use PageXML's logical page dims (not naturalWidth/Height) so that the
  // visible image, SVG overlay and polygon coordinates all share one
  // coordinate space — see pageOrigDims() comment.
  var origDims = pageOrigDims();
  var origW = origDims[0];
  var origH = origDims[1];

  // Pin the rendered image to the canonical dims so it scales (up or down)
  // to match the polygon coordinate space.  This is the actual fix for
  // the full-page overlay misalignment.
  if (origW > 0 && origH > 0) {
    pageImage.style.width  = origW + 'px';
    pageImage.style.height = origH + 'px';
  }

  var dims = rotatedDimensions(origW, origH, imageRotation);
  var canvasW = dims[0];
  var canvasH = dims[1];

  // Offset the image so top-left of rotated image aligns with container origin
  var offsetX = 0, offsetY = 0;
  if (imageRotation === 90)  { offsetX = origH; offsetY = 0; }
  if (imageRotation === 180) { offsetX = origW; offsetY = origH; }
  if (imageRotation === 270) { offsetX = 0;     offsetY = origW; }

  pageImage.style.marginLeft = offsetX + 'px';
  pageImage.style.marginTop  = offsetY + 'px';

  // Resize the container to the rotated canvas dimensions
  imageContainer.style.width  = canvasW + 'px';
  imageContainer.style.height = canvasH + 'px';

  // Also position SVG overlay
  regionOverlay.style.left = '0px';
  regionOverlay.style.top  = '0px';

  // Apply pan+zoom to the outer container
  imageContainer.style.transform = 'translate(' + panX + 'px, ' + panY + 'px) scale(' + zoom + ')';
  zoomLevel.textContent = Math.round(zoom * 100) + '%';
}

function zoomTo(newZoom, cx, cy) {
  var old = zoom;
  zoom = Math.max(0.05, Math.min(15, newZoom));
  if (cx !== undefined && cy !== undefined) {
    panX = cx - (cx - panX) * (zoom / old);
    panY = cy - (cy - panY) * (zoom / old);
  }
  updateTransform();
}

function fitToWidth() {
  var vw = imageViewport.clientWidth;
  var origDims = pageOrigDims();
  var origW = origDims[0] || 1;
  var origH = origDims[1] || 1;
  var dims = rotatedDimensions(origW, origH, imageRotation);
  var canvasW = dims[0];
  zoom = (vw - 20) / canvasW;
  panX = 10; panY = 10;
  updateTransform();
}

// ══════════════════════════════════════════════════════════
// MARKDOWN PREVIEW
// ══════════════════════════════════════════════════════════

function renderMarkdown(text) {
  var h = escHtml(text);
  var safeTags = ['u', 'sup', 'sub', 's', 'del', 'ins', 'mark', 'b', 'i', 'em', 'strong', 'br', 'small', 'abbr', 'span'];
  safeTags.forEach(function(tag) {
    h = h.replace(new RegExp('&lt;(' + tag + ')(\\s[^&]*?)?&gt;', 'gi'), '<$1$2>');
    h = h.replace(new RegExp('&lt;/(' + tag + ')&gt;', 'gi'), '</$1>');
  });
  h = h.replace(/&lt;br\s*\/?&gt;/gi, '<br/>');
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  h = h.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  h = h.replace(/^---+$/gm, '<hr>');
  h = h.replace(/\|(.+)\|/g, function(match) {
    if (match.indexOf('---') !== -1) return '';
    var cells = match.split('|').filter(function(c) { return c.trim(); });
    return '<tr>' + cells.map(function(c) { return '<td>' + c.trim() + '</td>'; }).join('') + '</tr>';
  });
  h = h.split('\n').map(function(line) {
    var t = line.trim();
    if (!t) return '';
    if (/^<(h[1-6]|li|blockquote|tr|table|hr)/.test(t)) return t;
    return '<p>' + t + '</p>';
  }).join('\n');
  h = h.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
  h = h.replace(/(<tr>.*<\/tr>\n?)+/g, '<table>$&</table>');
  return h;
}

function updatePreview(text) {
  markdownPreview.innerHTML = renderMarkdown(text);
  applyEntityHighlights();
}
function updateCharCount(text) {
  charCount.textContent = text.split('\n').length + ' lines \u00b7 ' + text.length + ' chars';
}

// ══════════════════════════════════════════════════════════
// NAMED ENTITIES — preview highlighting + legend
// ══════════════════════════════════════════════════════════

const ENTITY_COLORS = DATA.entityColors || {};
const ENTITY_LABELS = DATA.entityLabels || {};

function currentPageEntities() {
  var page = DATA.pages[currentPageIdx];
  var pxml = page && page.pagexml;
  return (pxml && Array.isArray(pxml.entities)) ? pxml.entities : [];
}

function escRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Walk text nodes inside the markdown preview and wrap occurrences of
 * entity texts in <span class="entity"> tags. Longest entities are
 * matched first so "grauen Fliegenschnäpper" beats "Fliegenschnäpper"
 * to a shared span. Skips text already inside .entity / <code> / <pre>.
 */
function applyEntityHighlights() {
  if (!showEntities) return;
  var entities = currentPageEntities();
  if (!entities.length) return;

  // Sort longest-first to handle overlapping/nested entity texts deterministically.
  var sorted = entities.slice().sort(function(a, b) {
    return (b.text || '').length - (a.text || '').length;
  });

  // Combined regex (case-sensitive — the NER prompt requires exact-case matches).
  var pattern = sorted
    .filter(function(e) { return e && e.text; })
    .map(function(e) { return escRegex(e.text); })
    .join('|');
  if (!pattern) return;
  var rx = new RegExp(pattern);

  // Build a quick lookup: text → entity (longest already wins via sort order).
  var byText = {};
  sorted.forEach(function(e) {
    if (e && e.text && byText[e.text] === undefined) byText[e.text] = e;
  });

  function shouldSkip(node) {
    var p = node.parentNode;
    while (p && p !== markdownPreview) {
      if (p.nodeType === 1) {
        var tag = p.tagName;
        if (tag === 'CODE' || tag === 'PRE') return true;
        if (p.classList && p.classList.contains('entity')) return true;
      }
      p = p.parentNode;
    }
    return false;
  }

  // Collect text nodes first — modifying the tree mid-walk is fragile.
  var walker = document.createTreeWalker(markdownPreview, NodeFilter.SHOW_TEXT, null);
  var nodes = [];
  var n;
  while ((n = walker.nextNode())) nodes.push(n);

  nodes.forEach(function(node) {
    if (!node.nodeValue || shouldSkip(node)) return;
    var txt = node.nodeValue;
    if (!rx.test(txt)) return;

    var frag = document.createDocumentFragment();
    var lastIdx = 0;
    var globalRx = new RegExp(pattern, 'g');
    var m;
    while ((m = globalRx.exec(txt)) !== null) {
      var matchText = m[0];
      var ent = byText[matchText];
      if (!ent) { lastIdx = globalRx.lastIndex; continue; }

      // Push leading text
      if (m.index > lastIdx) {
        frag.appendChild(document.createTextNode(txt.slice(lastIdx, m.index)));
      }
      // Build span
      var span = document.createElement('span');
      span.className = 'entity';
      var color = ENTITY_COLORS[ent.entity_type || ent.type] || '#666';
      span.style.color = color;
      span.title = (ENTITY_LABELS[ent.entity_type || ent.type]
                    || ent.entity_type || ent.type || '')
                   + (ent.context ? '  —  ' + ent.context : '');
      span.textContent = matchText;
      frag.appendChild(span);
      lastIdx = globalRx.lastIndex;

      // Guard against zero-width matches
      if (m.index === globalRx.lastIndex) globalRx.lastIndex++;
    }
    // Trailing text
    if (lastIdx < txt.length) {
      frag.appendChild(document.createTextNode(txt.slice(lastIdx)));
    }
    node.parentNode.replaceChild(frag, node);
  });
}

/**
 * Rebuild the entity legend strip above the editor tabs.  One chip per
 * entity type with a colour swatch and the count of occurrences on the
 * current page; types with zero hits are dimmed.  The whole strip is
 * hidden when the page has no entities at all (or NER hasn't been run).
 */
function updateEntityLegend() {
  var legend = $('entityLegend');
  if (!legend) return;
  var entities = currentPageEntities();

  if (!showEntities || !entities.length) {
    legend.classList.add('hidden');
    legend.innerHTML = '';
    return;
  }

  // Count per type (preserve the canonical type order from ENTITY_LABELS)
  var counts = {};
  entities.forEach(function(e) {
    var t = e.entity_type || e.type;
    if (!t) return;
    counts[t] = (counts[t] || 0) + 1;
  });

  var types = Object.keys(ENTITY_LABELS);
  // Append any unknown types we encountered (shouldn't happen, but be safe)
  Object.keys(counts).forEach(function(t) {
    if (types.indexOf(t) === -1) types.push(t);
  });

  var html = '<span class="entity-legend-title">Entities</span>';
  types.forEach(function(t) {
    var c = counts[t] || 0;
    var color = ENTITY_COLORS[t] || '#666';
    var label = ENTITY_LABELS[t] || t;
    html += '<span class="entity-legend-item' + (c === 0 ? ' empty' : '') + '" '
          + 'title="' + escHtml(t) + '">'
          + '<span class="entity-legend-swatch" style="background:' + color + '"></span>'
          + escHtml(label)
          + '<span class="entity-legend-count">' + c + '</span>'
          + '</span>';
  });
  legend.innerHTML = html;
  legend.classList.remove('hidden');
}

function updateDirtyState() {
  var page = DATA.pages[currentPageIdx];
  var isDirty = editorTextarea.value !== page.markdown;
  var hasAny = Object.keys(edits).length > 0 || isDirty;
  statusDot.className = 'status-dot ' + (isDirty ? 'dirty' : 'clean');
  statusText.textContent = isDirty ? 'Modified' : 'Clean';
  btnExport.classList.toggle('has-changes', hasAny);
}

// ══════════════════════════════════════════════════════════
// EXPORT — includes per-page rotation
// ══════════════════════════════════════════════════════════

function exportEdits() {
  saveCurrentEdits();
  var output = {
    _session: {
      bookName: DATA.bookName,
      lastPage: currentPageIdx,
      exportedAt: new Date().toISOString(),
      editCount: Object.keys(edits).length,
      doneCount: Object.keys(donePages).length,
      redoCount: Object.keys(redoPages).length,
      rotationCount: Object.keys(pageRotations).length
    },
    _donePages: {},
    _redoPages: {},
    _pageRotations: {}
  };

  Object.keys(donePages).forEach(function(stem) { output._donePages[stem] = donePages[stem]; });
  Object.keys(redoPages).forEach(function(stem) { output._redoPages[stem] = redoPages[stem]; });
  // Export all per-page rotations (non-zero only, already filtered at set time)
  Object.keys(pageRotations).forEach(function(stem) { output._pageRotations[stem] = pageRotations[stem]; });

  // Aggregate metrics
  var totalCerDist = 0, totalCerRef = 0, totalWerDist = 0, totalWerRef = 0;
  Object.keys(donePages).forEach(function(stem) {
    var m = donePages[stem];
    totalCerDist += m.charEdits; totalCerRef += m.refChars;
    totalWerDist += m.wordEdits; totalWerRef += m.refWords;
  });

  if (Object.keys(donePages).length > 0) {
    output._session.aggregateMetrics = {
      totalPages: DATA.pages.length,
      donePages: Object.keys(donePages).length,
      redoPages: Object.keys(redoPages).length,
      rotatedPages: Object.keys(pageRotations).length,
      microCER: totalCerRef > 0 ? totalCerDist / totalCerRef : 0,
      microWER: totalWerRef > 0 ? totalWerDist / totalWerRef : 0,
      totalCharEdits: totalCerDist,
      totalWordEdits: totalWerDist,
      totalRefChars: totalCerRef,
      totalRefWords: totalWerRef
    };
  }

  DATA.pages.forEach(function(p) {
    var mod = edits[p.stem] !== undefined;
    var rot = pageRotations[p.stem] || 0;
    var hasRedo = !!redoPages[p.stem];

    if (mod || rot || hasRedo) {
      output[p.stem] = {
        original: p.markdown,
        edited: mod ? edits[p.stem] : p.markdown,
        modified: mod,
        imageRotation: rot  // ← always written (0 if untouched)
      };
      if (donePages[p.stem]) { output[p.stem].metrics = donePages[p.stem]; }
      if (hasRedo) { output[p.stem].redo = redoPages[p.stem]; }
    }
  });

  var blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url;
  a.download = DATA.bookName + '_edits.json';
  a.click();
  URL.revokeObjectURL(url);

  var parts = [Object.keys(edits).length + ' modified', Object.keys(donePages).length + ' done'];
  if (Object.keys(redoPages).length > 0) parts.push(Object.keys(redoPages).length + ' redo');
  if (Object.keys(pageRotations).length > 0) parts.push(Object.keys(pageRotations).length + ' rotated');
  showToast('Exported ' + parts.join(', ') + ' page(s)');
}

// ══════════════════════════════════════════════════════════
// IMPORT
// ══════════════════════════════════════════════════════════

function importSessionFromFile(file) {
  var reader = new FileReader();
  reader.onload = function(e) {
    try {
      var session = JSON.parse(e.target.result);
      if (session._session && session._session.bookName &&
          session._session.bookName !== DATA.bookName) {
        showToast('\u26a0 This session is for "' + session._session.bookName +
                  '", not "' + DATA.bookName + '"', 4000);
        return;
      }

      if (!session._donePages) {
        session._donePages = {};
        Object.keys(session).forEach(function(key) {
          if (key === '_session' || key === '_donePages' || key === '_redoPages' || key === '_pageRotations') return;
          var val = session[key];
          if (val && val.metrics) { session._donePages[key] = val.metrics; }
        });
      }
      if (!session._redoPages) {
        session._redoPages = {};
        Object.keys(session).forEach(function(key) {
          if (key === '_session' || key === '_donePages' || key === '_redoPages' || key === '_pageRotations') return;
          var val = session[key];
          if (val && val.redo) { session._redoPages[key] = val.redo; }
        });
      }
      // Extract rotations from page entries if _pageRotations block absent
      if (!session._pageRotations) {
        session._pageRotations = {};
        Object.keys(session).forEach(function(key) {
          if (key === '_session' || key === '_donePages' || key === '_redoPages' || key === '_pageRotations') return;
          var val = session[key];
          if (val && val.imageRotation) { session._pageRotations[key] = val.imageRotation; }
        });
      }

      applySession(session, 'import');
    } catch(err) {
      showToast('\u26a0 Could not parse session file: ' + err.message, 4000);
    }
  };
  reader.readAsText(file);
}

// ══════════════════════════════════════════════════════════
// EVENT WIRING
// ══════════════════════════════════════════════════════════

function setupEvents() {
  btnPrev.addEventListener('click', function() { loadPage(currentPageIdx - 1); });
  btnNext.addEventListener('click', function() { loadPage(currentPageIdx + 1); });
  pageSelect.addEventListener('change', function() { loadPage(parseInt(pageSelect.value)); });

  btnRegions.addEventListener('click', function() {
    showRegions = !showRegions;
    btnRegions.classList.toggle('active', showRegions);
    regionOverlay.style.display = showRegions ? 'block' : 'none';
    regionLegend.style.display = showRegions ? 'flex' : 'none';
  });

  btnEntities.addEventListener('click', function() {
    showEntities = !showEntities;
    btnEntities.classList.toggle('active', showEntities);
    var page = DATA.pages[currentPageIdx];
    if (page) {
      // Re-render preview from the current text so entity spans appear/disappear cleanly
      var md = edits[page.stem] !== undefined ? edits[page.stem] : page.markdown;
      updatePreview(md);
    }
    updateEntityLegend();
  });

  btnFitWidth.addEventListener('click', fitToWidth);
  $('btnZoomIn').addEventListener('click', function() { zoomTo(zoom * 1.3); });
  $('btnZoomOut').addEventListener('click', function() { zoomTo(zoom / 1.3); });
  $('btnZoomReset').addEventListener('click', function() { zoom = 1; panX = 0; panY = 0; updateTransform(); });

  // Rotation buttons
  btnRotCW.addEventListener('click', rotateCW);
  btnRotCCW.addEventListener('click', rotateCCW);
  btnRotReset.addEventListener('click', rotateReset);

  imageViewport.addEventListener('wheel', function(e) {
    e.preventDefault();
    var rect = imageViewport.getBoundingClientRect();
    zoomTo(zoom * (e.deltaY < 0 ? 1.12 : 1/1.12), e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  imageViewport.addEventListener('mousedown', function(e) {
    if (e.button !== 0) return;
    isPanning = true;
    panStartX = e.clientX - panX;
    panStartY = e.clientY - panY;
    e.preventDefault();
  });
  window.addEventListener('mousemove', function(e) {
    if (!isPanning) return;
    panX = e.clientX - panStartX; panY = e.clientY - panStartY;
    updateTransform();
  });
  window.addEventListener('mouseup', function() { isPanning = false; });

  var touchStartX, touchStartY;
  imageViewport.addEventListener('touchstart', function(e) {
    if (e.touches.length === 1) {
      isPanning = true;
      touchStartX = e.touches[0].clientX - panX;
      touchStartY = e.touches[0].clientY - panY;
    }
  });
  imageViewport.addEventListener('touchmove', function(e) {
    if (!isPanning || e.touches.length !== 1) return;
    e.preventDefault();
    panX = e.touches[0].clientX - touchStartX;
    panY = e.touches[0].clientY - touchStartY;
    updateTransform();
  }, { passive: false });
  imageViewport.addEventListener('touchend', function() { isPanning = false; });

  editorTextarea.addEventListener('input', function() {
    var t = editorTextarea.value;
    updatePreview(t); updateCharCount(t); updateDirtyState();
    scheduleAutoSave();
  });

  document.querySelectorAll('.editor-tab').forEach(function(tab) {
    tab.addEventListener('click', function() {
      activeTab = tab.dataset.tab;
      document.querySelectorAll('.editor-tab').forEach(function(t) { t.classList.remove('active'); });
      tab.classList.add('active');
      editorTextarea.style.display = activeTab === 'edit' ? 'block' : 'none';
      markdownPreview.style.display = activeTab === 'preview' ? 'block' : 'none';
    });
  });

  btnExport.addEventListener('click', exportEdits);
  btnDone.addEventListener('click', togglePageDone);
  btnRedo.addEventListener('click', togglePageRedo);
  btnImport.addEventListener('click', function() { sessionFileInput.click(); });
  sessionFileInput.addEventListener('change', function(e) {
    var file = e.target.files && e.target.files[0];
    if (file) importSessionFromFile(file);
    sessionFileInput.value = '';
  });

  btnRestore.addEventListener('click', function() {
    restoreBanner.style.display = 'none';
    if (restoreBanner._sessionData) { applySession(restoreBanner._sessionData, 'autosave'); }
  });
  btnDismiss.addEventListener('click', function() {
    restoreBanner.style.display = 'none';
    try { localStorage.removeItem(STORAGE_KEY); } catch(e) {}
  });

  document.addEventListener('keydown', function(e) {
    if (e.target === editorTextarea) return;
    if (e.key === 'ArrowLeft')  { loadPage(currentPageIdx - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { loadPage(currentPageIdx + 1); e.preventDefault(); }
    if (e.key === 'r' || e.key === 'R') { if (!e.shiftKey) btnRegions.click(); }
    if (e.key === 'e' || e.key === 'E') { if (!e.shiftKey) btnEntities.click(); }
    if (e.key === 'f' || e.key === 'F') fitToWidth();
    if ((e.key === 'd' || e.key === 'D') && e.shiftKey) { togglePageRedo(); e.preventDefault(); }
    else if (e.key === 'd' || e.key === 'D') togglePageDone();
    // [ = rotate CCW,  ] = rotate CW,  \ = reset rotation
    if (e.key === '[') { rotateCCW(); e.preventDefault(); }
    if (e.key === ']') { rotateCW();  e.preventDefault(); }
    if (e.key === '\\') { rotateReset(); e.preventDefault(); }
    if (e.key === '+' || e.key === '=') zoomTo(zoom * 1.3);
    if (e.key === '-') zoomTo(zoom / 1.3);
    if (e.key === '0') { zoom = 1; panX = 0; panY = 0; updateTransform(); }
  });

  window.addEventListener('beforeunload', function() {
    saveCurrentEdits();
    autoSaveToLocalStorage();
  });

  var divider = $('divider');
  var isDragging = false;
  divider.addEventListener('mousedown', function(e) { isDragging = true; divider.classList.add('dragging'); e.preventDefault(); });
  window.addEventListener('mousemove', function(e) {
    if (!isDragging) return;
    var rect = document.querySelector('.main-container').getBoundingClientRect();
    var ratio = Math.max(0.2, Math.min(0.8, (e.clientX - rect.left) / rect.width));
    $('imagePanel').style.flex = '0 0 ' + (ratio * 100) + '%';
    $('editorPanel').style.flex = '0 0 ' + ((1 - ratio) * 100) + '%';
  });
  window.addEventListener('mouseup', function() { isDragging = false; divider.classList.remove('dragging'); });
}

init();
</script>
</body>
</html>"""
    return html_template


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    root = BOOK_ROOT_DIR.rstrip("/")
    book_name = os.path.basename(root)
    print(f"\U0001f4da HistOrniGraph Validation GUI Generator")
    print(f"   Book: {book_name}")
    print(f"   Path: {root}")
    print()

    if not os.path.isdir(root):
        print(f"\u274c Directory not found: {root}")
        return

    print("\U0001f517 Resolving Google Drive image URLs...")
    image_urls = resolve_image_urls(root)

    if not image_urls:
        print("   \u26a0 No Drive URLs resolved. Images will use relative paths.")
        print("     (This is fine if you open the HTML from within the book folder)")
    print()

    print("\U0001f4c4 Building page data...")
    data = build_data(root, image_urls)
    total = len(data['pages'])
    with_img = sum(1 for p in data['pages'] if p['imagePath'])
    with_drive = sum(1 for p in data['pages'] if p['imagePath'] and 'drive.google.com' in str(p['imagePath']))
    with_md = sum(1 for p in data['pages'] if p['markdown'])
    with_xml = sum(1 for p in data['pages'] if p['pagexml']['regions'])
    with_ents = sum(1 for p in data['pages'] if p['pagexml'].get('entities'))
    total_ents = sum(len(p['pagexml'].get('entities', [])) for p in data['pages'])
    print(f"   Pages: {total}")
    print(f"   With images: {with_img} ({with_drive} via Drive URLs)")
    print(f"   With transcriptions: {with_md}")
    print(f"   With layout regions: {with_xml}")
    if data.get('nerActive'):
        print(f"   With named entities: {with_ents} pages, {total_ents} entities total"
              f" (using {PAGEXML_NER_SUBDIR}/)")
    else:
        print(f"   Named entities: not yet run "
              f"(no {PAGEXML_NER_SUBDIR}/ folder — run Run_NER_Stage.py first)")
    print()

    print("\U0001f3d7\ufe0f  Generating HTML...")
    html = generate_html(data)

    output_path = os.path.join(root, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"   \u2705 Saved: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    print()
    print("\U0001f4a1 Usage:")
    print(f"   1. Download {OUTPUT_FILENAME}")
    print(f"   2. Open in any browser")
    if with_drive > 0:
        print(f"   3. Images load from Google Drive (folder is shared)")
    else:
        print(f"   3. Keep the HTML next to the '{PAGES_SUBDIR}/' folder for images")
    print()
    print("\U0001f504 Image rotation:")
    print(f"   \u2022 Use the ↺ ↻ buttons in the image panel header to rotate 90°")
    print(f"   \u2022 Keyboard: [ = CCW, ] = CW, \\ = reset rotation")
    print(f"   \u2022 PageXML region overlays track the rotation automatically")
    print(f"   \u2022 Per-page rotation is saved in localStorage and exported JSON")
    print()
    print("\U0001f4be Session persistence:")
    print(f"   \u2022 Edits auto-save to browser localStorage every 1.5 s")
    print(f"   \u2022 Re-opening the same HTML in the same browser will offer to restore")
    print(f"   \u2022 Use 'Export edits' to save a portable JSON (works across browsers/machines)")
    print(f"   \u2022 Use 'Import session' to load a previously exported JSON")
    print()
    print("\U0001f4ca Validation metrics:")
    print(f"   \u2022 Mark pages as 'Done' after correcting — triggers CER/WER calculation")
    print(f"   \u2022 Mark pages as 'Redo' to flag them for reprocessing")
    print(f"   \u2022 Metrics (incl. rotation) are saved per-page in the exported JSON")
    print(f"   \u2022 Keyboard shortcuts: D to toggle done, Shift+D to toggle redo")
    print()
    print("   To generate for another book, change BOOK_ROOT_DIR and re-run!")


if __name__ == "__main__":
    main()
