"""Downscale page images to a Gemini-friendly resolution.

Gemini vision models accept large inputs but latency and timeouts grow quickly
with multi-megapixel PNGs.  We keep full-resolution working copies for crops
and exports, and send a downscaled JPEG to the API.
"""

import logging
from pathlib import Path
from typing import Tuple

from PIL import Image

from .config import PipelineConfig

log = logging.getLogger(__name__)

# Long-edge cap used when cfg.gemini_max_long_edge is 0 (disabled).
_DEFAULT_MAX_LONG_EDGE = 3072


def gemini_scale_factor(width: int, height: int, max_long_edge: int) -> float:
    """Return uniform scale so max(width, height) <= max_long_edge (or 1.0)."""
    if max_long_edge <= 0:
        return 1.0
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return 1.0
    return max_long_edge / long_edge


def downscale_image(img: Image.Image, max_long_edge: int) -> Tuple[Image.Image, float]:
    """Return *(resized image, scale)* where scale maps full-res → gemini pixels."""
    scale = gemini_scale_factor(img.width, img.height, max_long_edge)
    if scale >= 1.0:
        return img, 1.0
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    return resized, scale


def scale_bbox_to_full(bbox: dict, scale: float) -> dict:
    """Map a pixel bbox from the Gemini image back to full-resolution coords."""
    if scale >= 1.0:
        return dict(bbox)
    inv = 1.0 / scale
    return {
        "x": int(bbox["x"] * inv),
        "y": int(bbox["y"] * inv),
        "width": int(bbox["width"] * inv),
        "height": int(bbox["height"] * inv),
    }


def downscale_page_for_gemini(page_path: Path, cfg: PipelineConfig) -> Tuple[Path, float]:
    """Write a JPEG under ``output_dir/pages_gemini/`` and return *(path, scale)*."""
    max_edge = cfg.gemini_max_long_edge or _DEFAULT_MAX_LONG_EDGE
    gemini_dir = cfg.output_dir / "pages_gemini"
    gemini_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(page_path).convert("RGB")
    gemini_img, scale = downscale_image(img, max_edge)
    gemini_path = gemini_dir / f"{page_path.stem}.jpg"

    if scale < 1.0:
        gemini_img.save(gemini_path, "JPEG", quality=92, optimize=True)
        log.info(
            "Downscaled %s %dx%d → %dx%d (scale=%.3f) for Gemini",
            page_path.name,
            img.width, img.height,
            gemini_img.width, gemini_img.height,
            scale,
        )
    else:
        # Already within limits — still write a compact JPEG for the API payload.
        gemini_img.save(gemini_path, "JPEG", quality=92, optimize=True)
        log.debug("Gemini copy for %s at full resolution (%dx%d)", page_path.name, img.width, img.height)

    return gemini_path, scale
