"""
NER stage — Named Entity Recognition on transcribed page text.
==============================================================
Runs after the layout + transcription stages.  Operates on the combined
plain-text content of a page (assembled from PAGE-XML TextRegions) and
returns a list of Entity objects.

Adapted from the original NER project (Reuß ornithologist pipeline) for use
inside the HistOrniGraph code-base.  The renderer matches entities back to
the source text by literal substring lookup — LLMs are unreliable at
exact character offsets, so start_char / end_char are kept at -1.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import clean_llm_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """One Named Entity detected on a page."""
    text: str
    entity_type: str
    start_char: int = -1
    end_char: int = -1
    context: Optional[str] = None
    region_ref: Optional[str] = None  # PAGE XML region id where the entity appears

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
        if self.context:
            d["context"] = self.context
        if self.region_ref:
            d["region_ref"] = self.region_ref
        return d


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

NER_PROMPT_TEMPLATE = """\
Du bist ein Experte für Named Entity Recognition (NER) in historischen \
deutschen Texten.

Analysiere den folgenden Text und identifiziere alle Entitäten der unten \
genannten Kategorien.

ENTITÄTSKATEGORIEN:
{entity_descriptions}

WICHTIGE ANWEISUNGEN – BITTE GENAU BEACHTEN:

1. STRENGE KRITERIEN FÜR "Person":
   - NUR einzelne, namentlich genannte historische Persönlichkeiten annotieren.
   - Korrekte Beispiele: "Kaiser Karl IV.", "Herzog Ernst", "Fürst Reuß", "Martin Luther"
   - NICHT annotieren: Berufsgruppen (Bauern, Bergleute), Bevölkerungsgruppen (Einwohner),
     generische Begriffe (Volk, Menschen), Ethnien (Sorben, Germanen).
   - Im Zweifel: NICHT als Person annotieren!

2. STRENGE KRITERIEN FÜR "Location":
   - NUR konkret identifizierbare geographische Orte mit Eigennamen.
   - Korrekte Beispiele: "Weimar", "Thüringen", "Böhmen", "Sachsen", "Magdeburg"
   - NICHT annotieren: abstrakte Gebietsbezeichnungen (Unterland, Hochfläche),
     generische Landschaftsbegriffe (Thal, Plateau), relative Ortsangaben (im Osten).
   - Im Zweifel: NICHT als Location annotieren!

3. ALLGEMEINE REGELN:
   - Annotiere nur EINDEUTIGE Entitäten.
   - Der "text"-Wert muss EXAKT mit dem Originaltext übereinstimmen \
(Groß-/Kleinschreibung beachten).
   - Bei überlappenden Entitäten: wähle die spezifischere Kategorie.

TEXT ZUR ANALYSE:
```
{text}
```

Antworte NUR mit einem JSON-Array (kein Markdown, kein Kommentar):
[
    {{
        "text": "exakter Text der Entität",
        "entity_type": "Kategorie aus der Liste oben",
        "context": "...kurzer Satz in dem die Entität vorkommt..."
    }}
]

Gib ein leeres Array [] zurück, wenn keine Entitäten gefunden werden.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Robustly extract a JSON *array* from an LLM response.

    Handles:
      • Plain JSON arrays (response_mime_type='application/json' guarantees this)
      • Markdown ```json fences
      • Surrounding commentary / thinking text
    """
    if not text:
        return []
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract the first balanced array
    start = text.find("[")
    if start == -1:
        # Maybe the model returned a {"entities": [...]} object — try clean_llm_json
        cleaned = clean_llm_json(text)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for key in ("entities", "items", "results", "data"):
                    val = obj.get(key)
                    if isinstance(val, list):
                        return val
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        return []

    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        log.warning("NER JSON parse failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Core NER function
# ---------------------------------------------------------------------------

def perform_ner(
    client: Any,                          # google.genai.Client
    text: str,
    entity_types: Dict[str, str],
    model_id: str,
    thinking_level: str = "low",
    max_attempts: int = 2,
    page_id: Optional[str] = None,
) -> List[Entity]:
    """Run NER on plain text and return a list of Entity objects.

    Parameters
    ----------
    client          : authenticated google.genai.Client
    text            : the page text (regions concatenated, markup stripped)
    entity_types    : dict mapping entity-type → German description
    model_id        : Gemini model id (e.g. "gemini-3-flash-preview")
    thinking_level  : "low" / "medium" / "high"
    max_attempts    : retries on JSON / API failure
    page_id         : optional page identifier (used only for log output)
    """
    if not text or not text.strip():
        return []

    # Lazy import so the rest of the package works even when google-genai is
    # not installed (e.g. when only running Create_GUIs.py).
    from google.genai import types

    entity_descriptions = "\n".join(
        f"- **{etype}**: {desc}" for etype, desc in entity_types.items()
    )
    prompt = NER_PROMPT_TEMPLATE.format(
        entity_descriptions=entity_descriptions,
        text=text,
    )

    raw_data: List[Dict[str, Any]] = []
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=thinking_level,
                    ),
                    response_mime_type="application/json",
                ),
            )
            raw_data = _parse_json_array(response.text or "")
            break
        except Exception as exc:  # noqa: BLE001 — network / API / parsing
            last_err = exc
            log.warning(
                "NER attempt %d/%d failed%s: %s",
                attempt, max_attempts,
                f" (page={page_id})" if page_id else "",
                exc,
            )

    if not raw_data and last_err is not None:
        log.error("NER giving up on %s after %d attempts: %s",
                  page_id or "<text>", max_attempts, last_err)
        return []

    valid_types = set(entity_types.keys())
    seen: set = set()
    entities: List[Entity] = []

    for item in raw_data:
        if not isinstance(item, dict):
            continue
        etype = item.get("entity_type", "")
        etext = str(item.get("text", "")).strip()
        if etype not in valid_types or not etext:
            continue
        key = (etext, etype)
        if key in seen:
            continue
        seen.add(key)
        entities.append(Entity(
            text=etext,
            entity_type=etype,
            start_char=-1,
            end_char=-1,
            context=item.get("context"),
        ))

    return entities
