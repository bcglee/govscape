# AI modified: 2026-06-29T00:00:00Z ae90b40f0be6148f154a65bc4f211dfdfed48490
"""Small runner to extract PDFs from a folder, run OCR single vs multi-threaded,
and write timing results to CSV for analysis.

Usage:
    python tests/test_ocr_benchmark_runner.py

Defaults read PDFs from `tests/test_data/large/PDFs` and writes outputs to
`data/ocr_benchmark_output` and CSV to `data/ocr_benchmark_output/ocr_timings.csv`.
"""

import csv
import sys
from pathlib import Path

from govscape.config import DataModel
from govscape.processing.ocr_processing_stage import OCRProcessingStage
from govscape.processing.pdf_extraction_stage import PDFExtractionStage


def main(
    pdf_dir: str | Path = "tests/test_data/large/PDFs",
    out_data_dir: str | Path = "data/ocr_benchmark_output",
    threads_list=(1, 4),
    batch_size: int = 500,
    cpu_count: int = 2,
):
    pdf_dir = Path(pdf_dir)
    out_data_dir = Path(out_data_dir)
    out_data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_data_dir / "ocr_timings.csv"

    pdf_files = [str(p) for p in pdf_dir.glob("**/*.pdf")]
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    data_model = DataModel(str(out_data_dir))

    # Extract PDFs to images and text using existing stage
    print(
        f"Converting {len(pdf_files)} PDFs to images in {data_model.image_directory}..."
    )
    pdf_stage = PDFExtractionStage(data_model, pdf_files, cpu_count)
    pdf_stage.run()

    # Validate OCR stage
    ocr_stage = OCRProcessingStage(data_model)
    try:
        ocr_stage.validate()
    except Exception as e:
        print(f"OCR validation failed: {e}")
        sys.exit(2)

    rows = []

    # Always run a single-thread baseline first
    print("Running single-threaded OCR baseline...")
    res_single = ocr_stage.run(threads=1, batch_size=batch_size, compare=False)
    rows.append(
        {
            "mode": "single",
            "threads": 1,
            "batch_size": batch_size,
            "single_time": res_single.get("single_time"),
            "multi_time": res_single.get("multi_time"),
            "processed_count": res_single.get("processed_count"),
            "error_count": res_single.get("error_count"),
        }
    )

    # Run multi-threaded tests
    for t in threads_list:
        if t <= 1:
            continue
        print(f"Running multi-threaded OCR with {t} threads...")
        # Create a fresh stage to ensure engines are fresh
        ocr_stage_mt = OCRProcessingStage(data_model)
        res = ocr_stage_mt.run(threads=t, batch_size=batch_size, compare=True)
        rows.append(
            {
                "mode": "multi",
                "threads": t,
                "batch_size": batch_size,
                "single_time": res.get("single_time"),
                "multi_time": res.get("multi_time"),
                "processed_count": res.get("processed_count"),
                "error_count": res.get("error_count"),
            }
        )

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "mode",
            "threads",
            "batch_size",
            "single_time",
            "multi_time",
            "processed_count",
            "error_count",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote timings to {csv_path}")


if __name__ == "__main__":
    main()
