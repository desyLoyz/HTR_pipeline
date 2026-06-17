"""Generate PAGE XML (PAGE Content Schema) for each page.

Each extracted region record becomes a TextRegion.  Bounding boxes are
approximate vertical bands (single-pass mode does not detect layout boxes).
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom

log = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def _coords_str(bbox: Dict[str, int]) -> str:
    """Convert bbox dict → PAGE XML Points string (polygon)."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    return f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"


def generate_pagexml(
    page_id: str,
    records: List[Dict[str, Any]],
    image_dims: Dict[str, int],
    image_filename: str,
    output_dir: Path,
) -> Path:
    """Write a PAGE XML file for one page."""

    root = Element("PcGts", xmlns=PAGE_NS)
    metadata = SubElement(root, "Metadata")
    SubElement(metadata, "Creator").text = "journal_processor"
    SubElement(metadata, "Created").text = datetime.now(timezone.utc).isoformat()

    page = SubElement(
        root,
        "Page",
        imageFilename=image_filename,
        imageWidth=str(image_dims["width"]),
        imageHeight=str(image_dims["height"]),
    )

    reading_order_el = SubElement(page, "ReadingOrder")
    og = SubElement(reading_order_el, "OrderedGroup", id="reading_order")

    for rec in sorted(records, key=lambda r: r["reading_order"]):
        rid = rec["id"]
        bbox = rec.get("bbox", {"x": 0, "y": 0, "width": image_dims["width"], "height": image_dims["height"]})

        SubElement(og, "RegionRefIndexed", index=str(rec["reading_order"]), regionRef=rid)

        custom_parts = [
            f"type:{rec.get('type', 'RegionRecord')}",
            f"record_number:{rec.get('record_number', '')}",
            f"date:{rec.get('date', '')}",
        ]
        if rec.get("marginal_reference"):
            custom_parts.append(f"marginal_reference:{rec['marginal_reference']}")
        if rec.get("end_line"):
            custom_parts.append(f"end_line:{rec['end_line']}")

        region_el = SubElement(page, "TextRegion", id=rid, custom=" ".join(custom_parts))
        SubElement(region_el, "Coords", points=_coords_str(bbox))

        text = rec.get("transcription", {}).get("text", "")
        if text:
            te = SubElement(region_el, "TextEquiv")
            SubElement(te, "Unicode").text = text

    xml_str = tostring(root, encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)
    pretty = "\n".join(pretty.splitlines()[1:])

    out_path = output_dir / f"{page_id}.xml"
    out_path.write_text(pretty, encoding="utf-8")
    log.debug("Wrote %s", out_path.name)
    return out_path
