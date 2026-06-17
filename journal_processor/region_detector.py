"""Single-pass region detection and structured extraction using Gemini."""

import json
import logging
import re
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .utils import MIME_BY_EXT, clean_llm_json

log = logging.getLogger(__name__)


# ── Prompt ───────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """\
# Task: Region Detection and Structured Extraction from German Archival Register Page

You are analyzing a scanned historical German archival register page written mainly \
in 19th-century German Kurrent handwriting, with some Latin-script marginalia, \
numbers, dates, and file references.

Your task is to detect, segment, and transcribe each distinct numbered and dated record \
on this page.

## 1. Definition of a Region Record

A **region record** is one complete archival entry. It usually consists of:

1. A **record number** on the far left margin or at the beginning of a line.
2. An associated **date**, usually written near the number.
3. Optional **marginal references**, often written left of the main text, e.g. `jetzt: Kurbaiern 20381`.
4. The **main body text**, written in Kurrent.
5. A closing or source line, often beginning and ending with `/: ... :/`, typically mentioning:
   - `Or.`
   - `Perg.`
   - `Siegel`
   - `anh. Siegel`
   - `abgeg. Siegel`
   - or similar archival/seal information.

Each numbered and dated entry should be treated as one distinct record.

## 2. Region Boundaries

Detect the start and end of each record carefully.

### Start of a record

A record usually begins where one or more of the following appear:

- A far-left **record number**, e.g. `7.`, `8.`, `9.`, `10.`
- A date immediately after or near the number, e.g. `1340, Novemb. 26.`
- A red/blue/black marginal mark or underline next to a new entry
- A new block of main text aligned horizontally with a date/number

### End of a record

A record usually ends at the line containing an archival source or seal formula, often shaped like:

- `/: Or. (Perg.) mit anh. Siegel ... :/`
- `/: Or. (Perg.) ... Siegel ... :/`
- `/: ... Siegel ... :/`

If the closing seal/source line is unclear, end the record immediately before the next \
detected numbered/date entry.

## 3. Transcription Rules

Use **diplomatic transcription**.

Preserve:

- original spelling
- capitalization
- abbreviations
- punctuation
- line breaks where possible
- hyphenation at line breaks
- marginalia
- dates and numbers exactly as written
- source/seal formulas exactly as written

Do **not**:

- modernize spelling
- translate the text
- silently expand abbreviations
- merge uncertain words into guessed modern forms
- omit marginal references
- omit the final source/seal line

## 4. Handling Mixed Scripts

The document uses mixed scripts.

Recognize that:

- Main body text is usually German Kurrent.
- Dates, numbers, marginal references, headings, and some proper names may be in Latin \
cursive or print-like script.
- Marginal annotations such as `jetzt: Kurbaiern 20381` belong to the same record if \
horizontally aligned with that record.

## 5. Tagging Rules

For every record, identify and tag in `xml_structure`:

### Required tags

- `<record_number>`
- `<date>`
- `<body>`
- `<end_line>`

### Optional but important tags

- `<marginal_reference>`
- `<datum_line>`
- `<place_names>`
- `<personal_names>`
- `<archive_reference>`
- `<seal_information>`
- `<uncertain_readings>`

## 6. Marginal Notes

Marginal notes should not be treated as separate records unless they have their own \
number and date.

Associate a marginal note with the record whose horizontal text block it aligns with.

## 7. Uncertainty Marking

Use these conventions in `diplomatic_transcription`:

- `[unclear]` = unreadable word or passage
- `word[?]` = tentative reading
- `[... ]` = omitted or illegible portion inside a word/phrase
- Do not invent missing words.
- If the seal formula is partly unreadable, transcribe the visible part and mark the \
rest as `[unclear]`.

## 8. Record Completeness

Detect every **full** numbered and dated record on the page (up to {max_records} records).

Do not start with partial text at the top of the page if that text belongs to a previous \
incomplete entry. Ignore incomplete continuation text above the first clearly numbered entry.

Each record must have:

1. visible record number,
2. visible date,
3. main text,
4. closing source/seal line (or end before the next record if unclear).

## 9. Quality Control Checklist

Before finalizing, verify for each record:

- Correct start of the record
- Far-left number included
- Date included
- Marginal notes associated correctly
- Complete body transcribed
- Closing `/: ... :/` source/seal line included
- No text from previous or next record
- Spelling and abbreviations preserved
- Uncertain readings marked instead of guessed

## 10. Output format (JSON only)

Return **only** a JSON object — no commentary, no markdown fences, no reasoning text.

Schema:

{{"records": [
  {{
    "record_number": "<string>",
    "date": "<string>",
    "marginal_reference": "<string or null>",
    "record_type": "Numbered dated archival regest / region record",
    "end_line": "<string>",
    "diplomatic_transcription": "<complete diplomatic transcription>",
    "xml_structure": "<record>...</record>",
    "uncertainty_notes": ["<note>", "..."]
  }}
]}}

Rules:

- Include every distinct numbered and dated record on the page, in top-to-bottom order.
- Maximum {max_records} records.
- `xml_structure` must be a single well-formed `<record>...</record>` element.
- `uncertainty_notes` may be an empty array.
"""


class RegionDetector:
    """Single-pass detection and structured extraction with Gemini."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    def detect(
        self,
        image_path: Path,
        full_dimensions: Optional[Dict[str, int]] = None,
        gemini_scale: float = 1.0,
        max_regions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract all region records from a page image in one Gemini call."""
        from google.genai import types

        _ = gemini_scale  # retained for pipeline API compatibility
        effective_max = max_regions if max_regions is not None else self.cfg.max_records

        ext = image_path.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "image/png")
        img_bytes = image_path.read_bytes()

        if full_dimensions:
            full_w = int(full_dimensions["width"])
            full_h = int(full_dimensions["height"])
        else:
            from PIL import Image
            with Image.open(image_path) as img:
                full_w, full_h = img.size

        prompt = EXTRACTION_PROMPT.format(max_records=effective_max)

        raw = ""
        data: Dict[str, Any] = {}
        for attempt in range(self.cfg.detection_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model_id,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.cfg.detection_temperature,
                        max_output_tokens=65536,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.cfg.detection_thinking
                        ),
                    ),
                )
                raw = resp.text or ""
                data = self._parse_response(raw)
                if data.get("records") is not None:
                    break
                raise ValueError("Response missing 'records' array")
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt < self.cfg.detection_retries:
                    log.warning(
                        "Parse failed for %s (attempt %d/%d), retrying…",
                        image_path.name, attempt + 1, self.cfg.detection_retries + 1,
                    )
                    continue
                log.error("Parse error for %s: %s\nRaw: %s", image_path.name, exc, raw[:400])
                return self._error(image_path, f"Parse: {exc}", raw)
            except Exception as exc:
                log.error("Extraction failed for %s: %s", image_path.name, exc)
                return self._error(image_path, str(exc), traceback.format_exc())

        records = self._validate_records(
            data.get("records", []), effective_max, full_w, full_h,
        )

        return {
            "status": "success",
            "image_path": str(image_path),
            "image_dimensions": {"width": full_w, "height": full_h},
            "records": records,
            "total_records": len(records),
            "reading_order": [r["id"] for r in records],
            # Legacy key kept for callers that still look for "regions"
            "regions": records,
            "total_regions": len(records),
        }

    # ── parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """Parse JSON from model output, with markdown/XML fallbacks."""
        cleaned = clean_llm_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        return {"records": RegionDetector._parse_records_from_markup(raw)}

    @staticmethod
    def _parse_records_from_markup(raw: str) -> List[Dict[str, Any]]:
        """Best-effort extraction when the model returns markdown/XML instead of JSON."""
        records: List[Dict[str, Any]] = []
        xml_blocks = re.findall(r"(<record>.*?</record>)", raw, re.DOTALL | re.IGNORECASE)
        transcriptions = re.findall(
            r"##\s*Diplomatic transcription\s*```(?:text)?\s*(.*?)```",
            raw, re.DOTALL | re.IGNORECASE,
        )

        for idx, xml_block in enumerate(xml_blocks):
            fields = RegionDetector._parse_record_xml(xml_block)
            if idx < len(transcriptions):
                fields["diplomatic_transcription"] = transcriptions[idx].strip()
            fields.setdefault("record_type", "Numbered dated archival regest / region record")
            fields["xml_structure"] = xml_block
            fields.setdefault("uncertainty_notes", [])
            records.append(fields)

        if records:
            return records

        # Last resort: one record from the full raw text
        return [{
            "record_number": "",
            "date": "",
            "marginal_reference": None,
            "record_type": "Numbered dated archival regest / region record",
            "end_line": "",
            "diplomatic_transcription": raw.strip(),
            "xml_structure": "",
            "uncertainty_notes": [],
        }]

    @staticmethod
    def _parse_record_xml(xml_block: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "record_number": "",
            "date": "",
            "marginal_reference": None,
            "end_line": "",
            "uncertainty_notes": [],
        }
        try:
            root = ET.fromstring(xml_block)
        except ET.ParseError:
            return fields

        def _text(tag: str) -> str:
            el = root.find(tag)
            return (el.text or "").strip() if el is not None else ""

        fields["record_number"] = _text("record_number")
        fields["date"] = _text("date")
        marginal = _text("marginal_reference")
        fields["marginal_reference"] = marginal or None
        fields["end_line"] = _text("end_line")
        body = _text("body")
        if body:
            fields["diplomatic_transcription"] = body
        return fields

    # ── validation ──────────────────────────────────────────────────────

    def _validate_records(
        self,
        raw_records: List[Dict],
        max_records: int,
        img_w: int,
        img_h: int,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        total = min(len(raw_records), max_records)
        band_h = max(img_h // max(total, 1), 1)

        for i, rec in enumerate(raw_records[:max_records]):
            transcription = (
                rec.get("diplomatic_transcription")
                or rec.get("transcription")
                or ""
            ).strip()
            uncertainty = rec.get("uncertainty_notes") or []
            if isinstance(uncertainty, str):
                uncertainty = [uncertainty] if uncertainty else []

            xml_structure = rec.get("xml_structure") or ""
            if not xml_structure and transcription:
                xml_structure = self._synthesise_xml(rec, transcription)

            entry: Dict[str, Any] = {
                "id": "",
                "type": "RegionRecord",
                "reading_order": i + 1,
                "record_number": str(rec.get("record_number") or "").strip(),
                "date": str(rec.get("date") or "").strip(),
                "marginal_reference": rec.get("marginal_reference"),
                "record_type": rec.get(
                    "record_type",
                    "Numbered dated archival regest / region record",
                ),
                "end_line": str(rec.get("end_line") or "").strip(),
                "bbox": self._estimate_bbox(i, total, img_w, img_h, band_h),
                "transcription": {
                    "text": transcription,
                    "xml": xml_structure,
                    "uncertainty_notes": uncertainty,
                },
            }
            out.append(entry)

        for idx, record in enumerate(out):
            record["id"] = f"r{idx + 1:02d}"
            record["reading_order"] = idx + 1

        return out

    @staticmethod
    def _estimate_bbox(
        index: int, total: int, img_w: int, img_h: int, band_h: int,
    ) -> Dict[str, int]:
        """Approximate vertical band for PAGE XML (no bbox detection in single-pass mode)."""
        if total <= 0:
            return {"x": 0, "y": 0, "width": img_w, "height": img_h}
        y = min(index * band_h, max(img_h - band_h, 0))
        height = band_h if index < total - 1 else img_h - y
        return {"x": 0, "y": y, "width": img_w, "height": max(height, 1)}

    @staticmethod
    def _synthesise_xml(rec: Dict[str, Any], transcription: str) -> str:
        marginal = rec.get("marginal_reference") or ""
        marginal_tag = (
            f"  <marginal_reference>{marginal}</marginal_reference>\n"
            if marginal else ""
        )
        return (
            "<record>\n"
            f"  <record_number>{rec.get('record_number', '')}</record_number>\n"
            f"  <date>{rec.get('date', '')}</date>\n"
            f"{marginal_tag}"
            f"  <body>\n{transcription}\n  </body>\n"
            f"  <end_line>{rec.get('end_line', '')}</end_line>\n"
            "</record>"
        )

    @staticmethod
    def _error(path: Path, msg: str, detail: str = "") -> Dict[str, Any]:
        return {
            "status": "error",
            "image_path": str(path),
            "error": msg,
            "detail": detail[:500],
            "records": [],
            "total_records": 0,
            "regions": [],
            "total_regions": 0,
        }
