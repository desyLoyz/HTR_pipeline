"""Scan preparation: automatic rotation detection for single-page scans.

Rotation is inferred from aspect ratio — no model call:
  landscape (width >= height) → upright
  portrait  (height > width)  → rotate 90° clockwise before processing
"""

import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image

log = logging.getLogger(__name__)

# If height/width exceeds this, treat the scan as sideways.
_PORTRAIT_THRESHOLD = 1.1


def detect_rotation(image_path: Path, force_rotation: int = 0) -> int:
    """Return clockwise rotation in degrees (0 or 90)."""
    if force_rotation:
        return force_rotation
    with Image.open(image_path) as img:
        if img.height > img.width * _PORTRAIT_THRESHOLD:
            log.info(
                "%s is portrait (%dx%d) — auto rotation 90°",
                image_path.name, img.width, img.height,
            )
            return 90
    return 0


class ScanAnalyzer:
    """Lightweight per-scan checks (rotation only).

    Not used by the main pipeline — kept for optional standalone use.
    """

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg

    def analyse(self, image_path: Path, force_rotation: int = 0) -> Dict[str, Any]:
        rotation = detect_rotation(image_path, force_rotation)
        return {"rotation": rotation}
