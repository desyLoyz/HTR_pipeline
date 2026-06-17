"""Generate Markdown reconstruction of each page from extracted records."""

import logging
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def generate_md(
    page_id: str,
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a Markdown file with all extracted region records on the page."""
    lines: List[str] = [f"# {page_id}", ""]

    for rec in sorted(records, key=lambda r: r["reading_order"]):
        text = rec.get("transcription", {}).get("text", "")
        if not text:
            continue

        n = rec.get("reading_order", 0)
        lines.append(f"## Region Record {n}")
        lines.append("")
        lines.append("| Tag | Extracted value |")
        lines.append("|---|---|")
        lines.append(f"| **Record number** | **{rec.get('record_number', '')}** |")
        lines.append(f"| **Date** | **{rec.get('date', '')}** |")
        marginal = rec.get("marginal_reference") or ""
        lines.append(f"| **Marginal note / later reference** | **{marginal}** |")
        lines.append(
            f"| **Record type** | {rec.get('record_type', 'Numbered dated archival regest / region record')} |"
        )
        end_line = rec.get("end_line", "")
        lines.append(f"| **End line / seal formula** | `{end_line}` |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Diplomatic transcription")
        lines.append("")
        lines.append("```text")
        lines.append(text)
        lines.append("```")
        lines.append("")

        xml_block = rec.get("transcription", {}).get("xml", "")
        if xml_block:
            lines.append("---")
            lines.append("")
            lines.append("## Categorized structure")
            lines.append("")
            lines.append("```xml")
            lines.append(xml_block.strip())
            lines.append("```")
            lines.append("")

        notes = rec.get("transcription", {}).get("uncertainty_notes") or []
        if notes:
            lines.append("---")
            lines.append("")
            lines.append("## Notes on certainty")
            lines.append("")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("")

    md_path = output_dir / f"{page_id}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.debug("Wrote %s", md_path.name)
    return md_path
