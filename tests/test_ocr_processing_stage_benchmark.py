#!/usr/bin/env python3
"""Benchmark runtime of single-threaded vs multi-threaded OCR processing.

This script converts the PDFs from tests/test_data/large/PDFs into page
images, then benchmarks OCRProcessingStage using the repository's real OCR
implementations.

Supported OCR implementations: easyocr, paddleocr, olmocr, ocrmypdf.

Usage:
    poetry run python tests/test_ocr_processing_stage_benchmark.py
    poetry run python tests/test_ocr_processing_stage_benchmark.py \
        --ocr-types easyocr paddleocr olmocr ocrmypdf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage
from govscape.processing.pdf_extraction_stage import _convert_single_pdf

PDF_SOURCE_ROOT = Path(__file__).resolve().parent / "test_data" / "large" / "PDFs"
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "benchmarking_material"

OCR_ENGINE_KWARGS: dict[str, dict[str, Any]] = {
    "easyocr": {"languages": ["en"], "gpu": False},
    "paddleocr": {"language": "en", "use_gpu": False},
    "olmocr": {"model_name": "default"},
    "ocrmypdf": {"language": "eng", "output_type": "txt"},
}


def convert_sample_pdfs(
    source_pdf_dir: Path,
    data_model: DataModel,
    pdf_count: int | None,
) -> int:
    if not source_pdf_dir.is_dir():
        raise ValueError(f"Source PDF directory not found: {source_pdf_dir}")

    pdf_files = sorted(source_pdf_dir.glob("*.pdf"))
    if pdf_count is not None:
        pdf_files = pdf_files[:pdf_count]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {source_pdf_dir}")

    for path in (
        data_model.image_directory,
        data_model.txt_directory,
        data_model.metadata_directory,
    ):
        if os.path.isdir(path):
            shutil.rmtree(path)

    os.makedirs(data_model.image_directory, exist_ok=True)
    os.makedirs(data_model.txt_directory, exist_ok=True)
    os.makedirs(data_model.metadata_directory, exist_ok=True)

    total_pages = 0
    for pdf_file in pdf_files:
        if _convert_single_pdf(data_model, str(pdf_file)):
            page_images = list(
                Path(data_model.image_directory).joinpath(pdf_file.stem).glob("*.jpeg")
            )
            total_pages += len(page_images)

    return total_pages


def benchmark_ocr_processing_stage(
    data_root: Path,
    pdf_count: int | None,
    ocr_type: str,
    threads: int,
    batch_size: int,
) -> dict[str, Any]:
    if ocr_type not in OCR_ENGINE_KWARGS:
        raise ValueError(
            f"Unsupported OCR type {ocr_type!r}. "
            f"Supported types: {', '.join(sorted(OCR_ENGINE_KWARGS))}"
        )

    data_model = DataModel(str(data_root))
    if os.path.isdir(data_model.txt_directory):
        shutil.rmtree(data_model.txt_directory)
    os.makedirs(data_model.txt_directory, exist_ok=True)

    total_pages = convert_sample_pdfs(
        source_pdf_dir=PDF_SOURCE_ROOT,
        data_model=data_model,
        pdf_count=pdf_count,
    )

    stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type=ocr_type,
        **OCR_ENGINE_KWARGS[ocr_type],
    )
    stage.validate()

    result = stage.run(
        threads=threads,
        batch_size=batch_size,
        compare=True,
    )

    if result is None:
        raise RuntimeError("OCRProcessingStage.run returned no result")

    return {
        "data_root": str(data_root),
        "pdf_count": pdf_count,
        "ocr_type": ocr_type,
        "total_pages": total_pages,
        "threads": threads,
        "batch_size": batch_size,
        "single_time": result["single_time"],
        "multi_time": result["multi_time"],
        "processed_count": result["processed_count"],
        "error_count": result["error_count"],
    }


def format_summary(summary: dict[str, Any]) -> str:
    ratio = (
        summary["multi_time"] / summary["single_time"]
        if summary["single_time"]
        else float("inf")
    )
    lines = [
        "OCR Processing Stage Benchmark",
        "===============================",
        f"data_root: {summary['data_root']}",
        f"ocr_type: {summary['ocr_type']}",
        f"pdf_count: {summary['pdf_count']}",
        f"total_pages: {summary['total_pages']}",
        f"threads: {summary['threads']}",
        f"batch_size: {summary['batch_size']}",
        "",
        f"single_threaded_time: {summary['single_time']:.6f} seconds",
        f"multi_threaded_time: {summary['multi_time']:.6f} seconds",
        f"multi/single ratio: {ratio:.4f}",
        f"processed_count: {summary['processed_count']}",
        f"error_count: {summary['error_count']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCRProcessingStage single-threaded vs multi-threaded benchmark."
        )
    )
    parser.add_argument(
        "--ocr-types",
        nargs="+",
        choices=sorted(OCR_ENGINE_KWARGS),
        default=sorted(OCR_ENGINE_KWARGS),
        help=(
            "OCR engine types to benchmark. "
            "Defaults to all supported engines if omitted."
        ),
    )
    parser.add_argument(
        "--pdf-count",
        type=int,
        default=None,
        help=(
            "Optional number of PDFs to convert from tests/test_data/large/PDFs. "
            "If omitted, all PDFs are used."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of worker threads for multi-threaded benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size used by OCRProcessingStage.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output root for benchmark data and summary.",
    )

    args = parser.parse_args(argv)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for ocr_type in args.ocr_types:
        summary = benchmark_ocr_processing_stage(
            data_root=output_root,
            pdf_count=args.pdf_count,
            ocr_type=ocr_type,
            threads=args.threads,
            batch_size=args.batch_size,
        )
        summaries.append(summary)

        summary_text = format_summary(summary)
        summary_path = (
            output_root / f"ocr_processing_stage_benchmark_summary_{ocr_type}.txt"
        )
        summary_path.write_text(summary_text, encoding="utf-8")
        print(summary_text)
        print(f"\nSummary written to: {summary_path}\n")

    all_json_path = output_root / "ocr_processing_stage_benchmark_results.json"
    all_json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"All JSON results written to: {all_json_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
