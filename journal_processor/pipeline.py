"""Main processing pipeline.

Flow per scan:
    [copy] → [preprocess] → [downscale for Gemini] → [extract] → [output]

Each input image is one page.  Full-resolution PNGs live under ``pages/``;
compact JPEGs for the API under ``pages_gemini/``.  Region detection and
diplomatic transcription happen in a single Gemini call per page.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from .config import PipelineConfig
from .image_scaling import downscale_page_for_gemini
from .output_md import generate_md
from .output_pagexml import generate_pagexml
from .output_sharegpt import append_sharegpt, build_sharegpt_entries
from .preprocessor import preprocess_page
from .region_detector import RegionDetector
from .utils import natural_sort_key

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class Pipeline:
    """End-to-end archival register processing pipeline."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self._init_client()

    def _init_client(self) -> None:
        from google import genai

        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
        )
        self.detector = RegionDetector(self.client, self.cfg)

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline.  Returns a summary dict."""
        t0 = time.time()
        summary: Dict[str, Any] = {
            "mode": "single_pass",
            "pages_processed": 0,
            "errors": [],
        }

        scans = self._find_scans()
        if not scans:
            log.error("No images found in %s", self.cfg.input_dir)
            return summary
        log.info("Found %d scan(s) in %s", len(scans), self.cfg.input_dir)

        log.info("=== Stage 2: Prepare page images ===")
        page_tasks = self._prepare_pages(scans, summary)
        if not page_tasks:
            return summary

        log.info("=== Stage 3: Pre-processing %d page image(s) ===", len(page_tasks))
        for page_path in page_tasks:
            preprocess_page(page_path, self.cfg)

        log.info("=== Stage 4: Downscaling for Gemini ===")
        gemini_tasks: List[Tuple[Path, Path, float]] = []
        for page_path in page_tasks:
            try:
                gemini_path, scale = downscale_page_for_gemini(page_path, self.cfg)
                gemini_tasks.append((page_path, gemini_path, scale))
            except Exception as exc:
                log.error("Downscale failed for %s: %s", page_path.name, exc)
                summary["errors"].append({"page": page_path.name, "error": f"downscale: {exc}"})

        if not gemini_tasks:
            return summary

        log.info("=== Stages 5-6: Extract → Output ===")
        sharegpt_path = self.cfg.output_dir / "sharegpt" / "training_data.jsonl"
        if sharegpt_path.exists():
            sharegpt_path.unlink()

        if self.cfg.workers > 1:
            self._run_parallel(gemini_tasks, sharegpt_path, summary)
        else:
            self._run_sequential(gemini_tasks, sharegpt_path, summary)

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["pages_processed"] = len(gemini_tasks) - len(summary["errors"])
        summary["routing"] = {"total_page_images": len(gemini_tasks)}

        summary_path = self.cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(
            "Done. %d page image(s) in %.0fs (%d error(s))",
            summary["pages_processed"], elapsed, len(summary["errors"]),
        )
        return summary

    def _find_scans(self) -> List[Path]:
        return sorted(
            [p for p in self.cfg.input_dir.iterdir()
             if p.suffix.lower() in _IMAGE_EXTS],
            key=natural_sort_key,
        )

    def _prepare_pages(self, scans: List[Path], summary: Dict) -> List[Path]:
        """Write full-res PNGs under ``pages/`` (no rotation)."""
        pages_dir = self.cfg.output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        tasks: List[Path] = []

        for scan in scans:
            dest = pages_dir / (scan.stem + ".png")

            try:
                if not dest.exists() or dest.stat().st_mtime < scan.stat().st_mtime:
                    img = Image.open(scan).convert("RGB")
                    img.save(dest, "PNG")
                tasks.append(dest)
            except Exception as exc:
                log.error("Preparing page failed for %s: %s", scan.name, exc)
                summary["errors"].append({"page": scan.name, "error": f"prepare: {exc}"})

        log.info("Prepared %d page image(s)", len(tasks))
        return tasks

    def _run_sequential(
        self,
        tasks: List[Tuple[Path, Path, float]],
        sharegpt_path: Path,
        summary: Dict,
    ) -> None:
        for idx, (page_path, gemini_path, scale) in enumerate(tasks, 1):
            log.info("[%d/%d] %s", idx, len(tasks), page_path.name)
            try:
                self._process_page(page_path, gemini_path, scale, sharegpt_path)
            except Exception as exc:
                log.error("Failed %s: %s", page_path.name, exc)
                summary["errors"].append({"page": page_path.name, "error": str(exc)})

    def _run_parallel(
        self,
        tasks: List[Tuple[Path, Path, float]],
        sharegpt_path: Path,
        summary: Dict,
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {
                pool.submit(self._process_page, pp, gp, sc, sharegpt_path): pp
                for pp, gp, sc in tasks
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                pp = futures[future]
                try:
                    future.result()
                    log.info("[%d/%d] ✓ %s", done, len(tasks), pp.name)
                except Exception as exc:
                    log.error("[%d/%d] ✗ %s: %s", done, len(tasks), pp.name, exc)
                    summary["errors"].append({"page": pp.name, "error": str(exc)})

    def _process_page(
        self,
        page_path: Path,
        gemini_path: Path,
        gemini_scale: float,
        sharegpt_path: Path,
    ) -> None:
        pid = page_path.stem
        page_img = Image.open(page_path).convert("RGB")

        result = self.detector.detect(
            gemini_path,
            full_dimensions={"width": page_img.width, "height": page_img.height},
            gemini_scale=gemini_scale,
        )

        if result["status"] != "success":
            raise RuntimeError(f"Extraction failed: {result.get('error', 'unknown')}")

        records = result["records"]
        dims = result["image_dimensions"]

        if self.cfg.output_md:
            generate_md(pid, records, self.cfg.output_dir / "md")

        if self.cfg.output_pagexml:
            generate_pagexml(
                pid, records, dims, page_path.name,
                self.cfg.output_dir / "pagexml",
            )

        if self.cfg.output_sharegpt:
            sharegpt_images_dir = self.cfg.output_dir / "sharegpt" / "images"
            entries = build_sharegpt_entries(
                pid, page_img, records, self.cfg, sharegpt_images_dir,
            )
            if entries:
                append_sharegpt(entries, sharegpt_path)

        page_json = self.cfg.output_dir / "records" / f"{pid}.json"
        page_json.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
