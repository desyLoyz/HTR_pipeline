"""Configuration for the journal processing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Entity types  (NER stage — historical German ornithologist's journals)
# ---------------------------------------------------------------------------
ENTITY_TYPES: Dict[str, str] = {
    "Animal": (
        "Tier, Tiergruppe oder Tierart (z. B. Wolf, Forelle, Rinderherde)"
    ),
    "Artefact": (
        "Menschengemachtes, unbelebtes Artefakt (z. B. Brücke, Mühle, Eisenbahn)"
    ),
    "Environment": (
        "Biotop/Habitat, natürliche Umgebung, kein Eigenname einer Stadt/Ort "
        "(z. B. Wald, Uferzone, Auenlandschaft)"
    ),
    "Environmental Impact": (
        "Umweltauswirkung/Effekt (z. B. Überschwemmung, Erosion, Abholzung)"
    ),
    "Person": (
        "NUR einzelne, namentlich identifizierbare historische Persönlichkeiten "
        "mit Eigennamen (z. B. Kaiser Karl IV., Herzog Ernst, Fürst Reuß). "
        "KEINE Berufsgruppen, Bevölkerungsgruppen, Völker oder generische Bezeichnungen."
    ),
    "Location": (
        "NUR eindeutig identifizierbare, konkrete geographische Orte mit Eigennamen: "
        "Länder, Regionen, Städte, Dörfer (z. B. Weimar, Thüringen, Böhmen, Sachsen). "
        "KEINE abstrakten Gebietsbezeichnungen."
    ),
    "Organisation": (
        "Organisation/Verband/Institution (z. B. Universität Jena, Forstamt Saalfeld, "
        "Kloster Ettal)"
    ),
    "Natural Object": (
        "Natürlich vorkommendes Objekt ohne Veränderung durch menschliches Zutun "
        "(z. B. Donau, Fichtelgebirge, Lech, Brocken)"
    ),
    "Plant": "Pflanze/Pflanzenart (z. B. Eiche, Buche, Weizen)",
    "Resource": (
        "Natürlich vorkommende Ressource (z. B. Holz, Erz, Kohle, Quellwasser)"
    ),
    "Climate": (
        "Klima-/Wetter-/Temperatur-Phänomen (z. B. Frost, Dürre, Schneesturm, Regen)"
    ),
}

# Entity colours and labels for the HTML viewer / GUI
ENTITY_COLORS: Dict[str, str] = {
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
ENTITY_LABELS: Dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Archival register record taxonomy (single-pass extraction)
# ---------------------------------------------------------------------------
RECORD_TYPE = "RegionRecord"

MAX_RECORDS_PER_PAGE = 8

# Default long-edge cap for images sent to Gemini (full-res kept for crops).
GEMINI_MAX_LONG_EDGE = 3072

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable knobs live here.

    Each input scan is treated as a single page.  Full-resolution working
    copies are kept for crops and exports; a downscaled JPEG is sent to Gemini.
    """

    # I/O paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")

    # Models
    model_id: str = MODEL_ID

    # Gemini image sizing (0 = use GEMINI_MAX_LONG_EDGE default)
    gemini_max_long_edge: int = GEMINI_MAX_LONG_EDGE

    # Pre-processing
    deskew: bool = False                # optional deskew step
    enhance_contrast: bool = False      # optional CLAHE contrast boost

    # Single-pass extraction (detection + diplomatic transcription)
    max_records: int = MAX_RECORDS_PER_PAGE
    # Gemini 3 docs: keep temperature at 1.0 (default); values < 1.0 may cause
    # looping or degraded performance on complex tasks.
    detection_temperature: float = 1.0
    detection_thinking: str = "low"
    detection_retries: int = 3          # retry on bad JSON from Gemini

    # Output formats (all enabled by default)
    output_md: bool = True
    output_pagexml: bool = True
    output_sharegpt: bool = True

    # ShareGPT
    sharegpt_system_prompt: str = (
        "Detect and transcribe numbered archival register records from this "
        "German Kurrent manuscript page using diplomatic transcription."
    )

    # NER (Stage 7 – run separately in Colab via Run_NER_Stage.py)
    ner_model_id: str = MODEL_ID
    ner_thinking_level: str = "low"
    ner_retries: int = 2

    # Concurrency
    workers: int = 4                    # parallel page processing threads

    def ensure_dirs(self) -> None:
        """Create output sub-directories."""
        for sub in ("pages", "pages_gemini", "records", "md", "pagexml", "pagexml_ner",
                    "sharegpt", "sharegpt/images"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
