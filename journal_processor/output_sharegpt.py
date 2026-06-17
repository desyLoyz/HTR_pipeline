"""Generate ShareGPT-format JSONL training data.

Each line is a conversation turn: system prompt → full page image → assistant
structured extraction for one region record.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import PipelineConfig, RECORD_TYPE

log = logging.getLogger(__name__)


def _save_resized_image(
    img: Image.Image, save_path: Path, max_side: int = 1024
) -> None:
    """Resize (if needed) and save a PIL image to disk."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(save_path, format="PNG")


def _format_assistant_response(rec: Dict[str, Any]) -> str:
    """Build the structured assistant turn from one extracted record."""
    text = rec.get("transcription", {}).get("text", "")
    xml_block = rec.get("transcription", {}).get("xml", "")
    notes = rec.get("transcription", {}).get("uncertainty_notes") or []

    parts = [
        f"## Region Record {rec.get('reading_order', '')}",
        "",
        f"Record number: {rec.get('record_number', '')}",
        f"Date: {rec.get('date', '')}",
        f"Marginal reference: {rec.get('marginal_reference') or ''}",
        f"End line: {rec.get('end_line', '')}",
        "",
        "## Diplomatic transcription",
        text,
    ]
    if xml_block:
        parts.extend(["", "## Categorized structure", xml_block.strip()])
    if notes:
        parts.extend(["", "## Notes on certainty", *[f"- {n}" for n in notes]])
    return "\n".join(parts)


def build_sharegpt_entries(
    page_id: str,
    page_image: Image.Image,
    records: List[Dict[str, Any]],
    cfg: PipelineConfig,
    image_save_dir: Path,
) -> List[Dict[str, Any]]:
    """Return ShareGPT conversation dicts — one per extracted record."""
    entries: List[Dict[str, Any]] = []
    image_save_dir.mkdir(parents=True, exist_ok=True)

    page_image_filename = f"{page_id}_page.png"
    page_image_path = image_save_dir / page_image_filename
    if not page_image_path.exists():
        _save_resized_image(page_image, page_image_path)

    for rec in records:
        if rec.get("type") != RECORD_TYPE:
            continue

        assistant_text = _format_assistant_response(rec)
        if not assistant_text.strip():
            continue

        entry = {
            "id": f"{page_id}_{rec['id']}",
            "messages": [
                {
                    "role": "system",
                    "content": cfg.sharegpt_system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "<image>\n"
                        "Extract the numbered and dated archival register record "
                        f"with record number {rec.get('record_number', '') or '(see image)'}."
                    ),
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ],
            "images": [str(page_image_path)],
            "metadata": {
                "page_id": page_id,
                "record_id": rec["id"],
                "record_type": rec.get("type"),
                "record_number": rec.get("record_number"),
                "date": rec.get("date"),
            },
        }
        entries.append(entry)

    return entries


def append_sharegpt(entries: List[Dict[str, Any]], output_path: Path) -> None:
    """Append entries to a JSONL file (one JSON object per line)."""
    with open(output_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.debug("Appended %d ShareGPT entries to %s", len(entries), output_path.name)
