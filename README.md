# HTR Pipeline for archival registers (for court files)

Pipeline for transcription of **handwritten, scanned historical German archival register pages** (court files), from the **13th century onward**.

## Pipeline stages

```
1. Prepare     input scan → normalized full-resolution page PNG
2. Preprocess  optional deskew / contrast enhancement
3. Downscale   optional API-friendly copy for Gemini
4. Extract     region detection + diplomatic transcription (single call per page)
5. Output      records JSON · (optional) Markdown · PAGE XML · ShareGPT JSONL
```

## Output structure

```
output/
├── pages/            # normalized full-resolution page PNGs
├── pages_gemini/     # downscaled JPEGs used for the API (optional)
├── records/          # per-page structured extraction JSON
├── md/               # (optional) Markdown page reconstructions
├── pagexml/          # (optional) PAGE XML with layout + transcription
├── sharegpt/         # (optional) training_data.jsonl (+ images/)
└── summary.json
```

## Record / region types

| Type | Metadata | Transcription |
|------|----------|---------------|
| ParagraphRegion | line_count | exact line-by-line, `<u>` / `<sup>` markup |
| ListRegion | line_count | exact line-by-line |
| TableRegion | rows, cols | Markdown table |
| ObjectRegion | — | description + object_type |
| PageNumberRegion | page_number | skipped (extracted in detection) |
| MarginaliaRegion | line_count | exact transcription |
| FootnoteRegion | line_count | exact transcription |
| ImageRegion | — | description + drawing_type |

## Quick start

```bash
# Install
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-key"

# Run
python run.py -i /path/to/scans -o /path/to/output

# Options
python run.py -i scans/ -o out/ \
    --workers 8 \
    --max-records 8 \
    --deskew \
    --enhance-contrast \
    -v
```

## Notes

- Each input image is treated as **one page**.
- Region detection and diplomatic transcription are performed in a **single Gemini call per page**.
- `PageNumberRegion` extraction happens during detection, saving one extra call/step.
- ShareGPT output only includes `ParagraphRegion`, `ListRegion`, `TableRegion`, and `FootnoteRegion`.
