#!/usr/bin/env python3
"""Batch pipeline runner for selective reprocessing across Laubmann folders.

Edit the JOBS list below, then run:
    python batch_run.py

Each job entry is a dict with these keys:
─────────────────────────────────────────────────────────────────────────────
  folder    str   Input folder path (absolute, or relative to INPUT_BASE_DIR)
  output    str   Output folder path (absolute, or relative to OUTPUT_BASE_DIR)
            This is the same output dir you normally point --output at.
            When reprocessing specific images the old output files for those
            image numbers are deleted first (pages/, regions/, md/, pagexml/)
            before the pipeline writes fresh results — exactly like the
            PowerShell reprocess scripts did.

  images    list of int  |  "all"
            Which scans to process.  Give the trailing number from the
            filename as an int: for "NL_Laubmann_20_0042.jpg" use 42.
            The script matches the pattern *_NNNN.* (4-digit zero-padded,
            any prefix, any extension).
            Use "all" to process every image in the folder without filtering.

  workers   int   Parallel threads (default 4)
─────────────────────────────────────────────────────────────────────────────

Equivalent PowerShell one-liner for reference:
    python run.py --input %tempDir% --output %outputDir%
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from journal_processor import Pipeline, PipelineConfig

# ─── Base directories ─────────────────────────────────────────────────────────
# Folder/output values in each job are resolved relative to these.
# Use absolute paths in each job to ignore these entirely.

INPUT_BASE_DIR  = Path(r"G:\My Drive\HistOrniGraph")           # ← edit me
OUTPUT_BASE_DIR = Path(r"G:\My Drive\HistOrniGraph_output")    # ← edit me

# ─── Job list ─────────────────────────────────────────────────────────────────

JOBS: List[Dict[str, Any]] = [


	{
        "folder":  r"NL Laubmann_34/NL Laubmann_34",
        "output":  "Laubmann_34_gemini",
        "images":  [6,13,20,22,26,29,31,42,72],
    },
    	{
        "folder":  r"NL Laubmann_34/NL Laubmann_34",
        "output":  "Laubmann_34_gemini",
        "images":  [4,14,15,16,17,18,21,24,28,30,34,35,36,37,38,39,40,41,44,45,46,49,50,51,52,53,54,55,56,57,58,59,60,63,67,68],
    },
    
    
   
    # ── All of folder 27, auto mode ───────────────────────────────────────────
    # {
    #     "folder":  "Gemini Laubmann 27",
    #     "output":  "Laubmann_27_gemini",
    #     "images":  "all",
    #     "mode":    "auto",
    # },

    # ── Specific images from folder 22, force split ───────────────────────────
    # {
    #     "folder":  "Gemini Laubmann 22",
    #     "output":  "Laubmann_22_gemini",
    #     "images":  [16, 17, 24, 26],
    #     "mode":    "split",
    # },

    # ── Sideways scans in folder 21: rotate 90° then split ───────────────────
    # {
    #     "folder":  "Gemini Laubmann 21",
    #     "output":  "Laubmann_21_gemini",
    #     "images":  [17, 18, 19, 20, 21],
    #     "mode":    "split",
    #     "rotate":  90,
    # },

 
]

# ─── Implementation ───────────────────────────────────────────────────────────

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
# Output subdirs that store per-image files and need to be cleaned before rerun
_OUTPUT_SUBDIRS = ("pages", "regions", "md", "pagexml", "sharegpt/images")


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _find_image(folder: Path, number: int) -> Path:
    """Return the image whose filename ends with _{number:04d}.<ext>."""
    suffix = f"_{number:04d}"
    matches = [
        f for f in folder.iterdir()
        if f.suffix.lower() in _IMAGE_EXTS and f.stem.endswith(suffix)
    ]
    if not matches:
        # Fallback: stem is exactly the 4-digit number (e.g. "0042.jpg")
        exact = f"{number:04d}"
        matches = [
            f for f in folder.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and f.stem == exact
        ]
    if not matches:
        sample = [f.name for f in folder.iterdir() if f.suffix.lower() in _IMAGE_EXTS][:8]
        raise FileNotFoundError(
            f"No image for number {number} (pattern *_{number:04d}.*) in {folder}\n"
            f"  Sample files: {sample}"
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous: multiple files match number {number} in {folder}: {[m.name for m in matches]}")
    return matches[0]


def _clear_old_outputs(output_dir: Path, numbers: List[int]) -> None:
    """Delete existing output files/dirs for the given image numbers."""
    deleted = 0
    for num in numbers:
        for sub in _OUTPUT_SUBDIRS:
            sub_dir = output_dir / sub
            if not sub_dir.exists():
                continue
            for entry in sub_dir.rglob(f"*_{num:04d}*"):
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                deleted += 1
    if deleted:
        log.info("  Cleared %d old output item(s) for %d image number(s).", deleted, len(numbers))


def _run_job(job: Dict[str, Any], job_idx: int) -> bool:
    label = f"[job {job_idx + 1}] {job.get('folder', '?')}"

    input_folder  = _resolve(INPUT_BASE_DIR,  job["folder"])
    output_folder = _resolve(OUTPUT_BASE_DIR, job.get("output", Path(job["folder"]).name + "_out"))

    if not input_folder.exists():
        log.error("%s  Input folder not found: %s", label, input_folder)
        return False

    workers = int(job.get("workers", 4))
    images  = job.get("images", "all")

    tmp_dir = None
    try:
        if images == "all":
            effective_input = input_folder
        else:
            # 1. Clear old outputs for these specific image numbers
            _clear_old_outputs(output_folder, images)

            # 2. Copy selected images to a temp dir so the pipeline only sees them
            tmp_dir = Path(tempfile.mkdtemp(prefix="histornigraph_"))
            log.info("%s  Staging %d image(s) in temp dir …", label, len(images))
            for num in images:
                src = _find_image(input_folder, num)
                shutil.copy2(src, tmp_dir / src.name)
                log.info("  + %s", src.name)
            effective_input = tmp_dir

        log.info(
            "%s  images=%s → %s",
            label,
            "all" if images == "all" else len(images),
            output_folder,
        )

        cfg = PipelineConfig(
            input_dir  = effective_input,
            output_dir = output_folder,
            workers    = workers,
        )

        pipeline = Pipeline(cfg)
        summary  = pipeline.run()

        errors = summary.get("errors", [])
        if errors:
            log.warning("%s  Finished with %d error(s).", label, len(errors))
            for e in errors:
                log.warning("  ✗ %s: %s", e.get("page", "?"), e.get("error", "?"))
        else:
            log.info("%s  ✓ Done — %d page image(s) processed.", label, summary.get("pages_processed", 0))
        return len(errors) == 0

    except Exception as exc:
        log.error("%s  Job failed: %s", label, exc, exc_info=True)
        return False
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not JOBS:
        print("No jobs defined in JOBS list — nothing to do.")
        sys.exit(0)

    print(f"\nRunning {len(JOBS)} job(s) …\n{'─' * 60}")

    results = []
    for idx, job in enumerate(JOBS):
        ok = _run_job(job, idx)
        results.append(ok)
        print()

    passed = sum(results)
    failed = len(results) - passed
    print(f"{'─' * 60}")
    print(f"Summary: {passed}/{len(results)} job(s) succeeded" +
          (f", {failed} failed" if failed else "") + ".")
    if failed:
        sys.exit(1)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
