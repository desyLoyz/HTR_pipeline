"""Per-region transcription using Gemini 3 Flash Preview.

Each region type gets a specialised prompt:
  • Text regions  → exact line-by-line transcription, underline/superscript markup
  • Tables        → Markdown table preserving rows/cols
  • Images/Objects→ short description + any visible text
  • PageNumber    → skipped (already extracted during detection)
"""

import io
import logging
from typing import Any, Dict

from PIL import Image

from .config import PipelineConfig

log = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────────────────

_TEXT_PROMPT = """\
Transcribe the German text in this image exactly as written, \
character by character, line by line. It is part of an archival index for court records.

Region type: {region_type}

Rules:
- Preserve line breaks exactly as they appear.
- Mark underlined words like this: <u>word</u>
- Mark superscript (e.g. reference numbers) like this: <sup>3</sup>
- Use [?] for uncertain characters and [illegible] for unreadable words.
- Do NOT add interpretations, translations, or commentary.

Output ONLY the transcription."""

_TABLE_PROMPT = """\
Transcribe this table from a German ornithologist's journal into Markdown \
table format.

Expected dimensions: ~{rows} rows × {cols} columns.

Rules:
- Use | to separate columns and a header-separator row (---).
- Preserve the original cell content exactly.
- Mark uncertain text with [?], illegible text with [illegible].
- If a cell is empty, leave it blank between pipes.

Output ONLY the Markdown table."""

_IMAGE_PROMPT = """\
Describe this image/drawing region from a German ornithologist's journal.

Provide:
1. A short description (1-3 sentences) of what is depicted.
2. If any text or labels are visible, transcribe them exactly.
3. Metadata: drawing_type (sketch / diagram / map / photograph / other).

Format your answer as:
DESCRIPTION: …
TEXT: … (or "none")
DRAWING_TYPE: …"""

_LIST_PROMPT = """\
Transcribe the German image regions as a List. \
character by character.

Rules:
- Output each list item as a Markdown bullet: start the line with "- ".
- Preserve the original text of each item exactly.
- Mark underlined words like this: <u>word</u>
- Mark superscript (e.g. reference numbers) like this: <sup>3</sup>
- Use [?] for uncertain characters and [illegible] for unreadable words.
- Do NOT add interpretations, translations, or commentary.

Output ONLY the Markdown list."""

_OBJECT_PROMPT = """\
Describe this object/element from a German ornithologist's journal page.

Provide:
1. A short description (1-3 sentences).
2. Transcribe any visible text exactly.
3. Metadata: object_type (stamp / seal / letterhead / printed_form / other).

Format your answer as:
DESCRIPTION: …
TEXT: … (or "none")
OBJECT_TYPE: …"""


class Transcriber:
    """Region-level transcription with Gemini 3 Flash."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    # ── public API ──────────────────────────────────────────────────────

    def transcribe_region(
        self,
        region_image: Image.Image,
        region: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transcribe a single cropped region image.

        *region* is the detection dict (with type, metadata, etc.).
        Returns a dict with ``status``, ``text``, and optional metadata.
        """
        rtype = region["type"]

        # PageNumberRegion – number already extracted during detection
        if rtype == "PageNumberRegion":
            return {
                "status": "success",
                "text": str(region.get("page_number", "")),
                "skipped": True,
                "skip_reason": "page_number_from_detection",
            }

        # Folded insert – content is not visible; skip transcription
        if region.get("insert_state") == "folded":
            return {
                "status": "success",
                "text": "",
                "skipped": True,
                "skip_reason": "folded_insert",
            }

        prompt = self._build_prompt(region)
        return self._call(region_image, prompt, rtype)

    # ── prompt routing ──────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(region: Dict) -> str:
        rtype = region["type"]
        if rtype == "TableRegion":
            return _TABLE_PROMPT.format(
                rows=region.get("rows", "?"),
                cols=region.get("cols", "?"),
            )
        if rtype == "ListRegion":
            return _LIST_PROMPT
        if rtype == "ImageRegion":
            return _IMAGE_PROMPT
        if rtype == "ObjectRegion":
            return _OBJECT_PROMPT
        # ParagraphRegion, FootnoteRegion, MarginaliaRegion
        return _TEXT_PROMPT.format(region_type=rtype)

    # ── Gemini call ─────────────────────────────────────────────────────

    def _call(
        self, image: Image.Image, prompt: str, rtype: str
    ) -> Dict[str, Any]:
        from google.genai import types

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        try:
            resp = self.client.models.generate_content(
                model=self.cfg.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=self.cfg.transcription_temperature,
                    max_output_tokens=8192,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.cfg.transcription_thinking
                    ),
                ),
            )
            text = resp.text.strip()

            result: Dict[str, Any] = {"status": "success", "text": text}

            # Parse structured fields for image / object descriptions
            if rtype == "ImageRegion":
                result.update(self._parse_image_desc(text))
            elif rtype == "ObjectRegion":
                result.update(self._parse_object_desc(text))

            return result

        except Exception as exc:
            log.error("Transcription failed (%s): %s", rtype, exc)
            return {"status": "error", "error": str(exc), "text": ""}

    # ── structured-description parsers ──────────────────────────────────

    @staticmethod
    def _parse_image_desc(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for line in text.splitlines():
            if line.upper().startswith("DESCRIPTION:"):
                out["description"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TEXT:"):
                out["visible_text"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DRAWING_TYPE:"):
                out["drawing_type"] = line.split(":", 1)[1].strip()
        return out

    @staticmethod
    def _parse_object_desc(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for line in text.splitlines():
            if line.upper().startswith("DESCRIPTION:"):
                out["description"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("TEXT:"):
                out["visible_text"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("OBJECT_TYPE:"):
                out["object_type"] = line.split(":", 1)[1].strip()
        return out
