"""CodSpeed benchmarks for the PDF processing pipeline.

Run locally with `poetry run pytest tests/benchmarks --codspeed`. Without the
`--codspeed` flag, these run once as ordinary correctness-checking tests.

These stages do real file I/O and (for extraction/image embedding) spawn
worker processes, so CI measures them with CodSpeed's `walltime` instrument
rather than CPU simulation (see the `benchmarks` job in
`.github/workflows/ci.yml`). Output/correctness regressions (as opposed to
performance regressions) are covered separately by
`tests/test_pdf_processing_regressions.py`, using `pytest-regressions`.

Note: `PDFExtractionStage`/`PageImageEmbeddingStage` spawn their worker pool
with `multiprocessing.get_context("spawn")`, so every call re-imports
`govscape.processing` (and, transitively, `torch`/`transformers` via
`visual_embedding_models`) in each worker. That import alone costs several
seconds, dwarfing the actual per-PDF parsing work, so
`test_single_pdf_conversion_benchmark` below measures just the core
conversion logic in-process (no spawn) to isolate that signal; the other
benchmarks intentionally measure the full, spawn-inclusive stage cost since
that is what real pipeline runs pay.

These benchmarks use the real `ST` (sentence-transformers/all-mpnet-base-v2)
text model and `CLIP` (openai/clip-vit-base-patch32) visual model -- the same
defaults `scripts/serving/start_api_server.py` uses in production -- rather
than the cheap `Dummy` models, over the full ~1,660-page sample corpus (see
`conftest.py`). This is deliberately expensive: CLIP alone downloads a 1.2GB
model and takes roughly half a second per page on CPU, so the image embedding
and end-to-end benchmarks can each take on the order of 10+ minutes in CI.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from govscape.config import DataModel
from govscape.pdf_processing_pipeline import PDFProcessingPipeline
from govscape.processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)
from govscape.processing.pdf_extraction_stage import _convert_single_pdf

CPU_COUNT = os.cpu_count() or 1


def _fresh_data_model(base_dir: Path) -> DataModel:
    return DataModel((base_dir / uuid.uuid4().hex).as_posix())


def test_single_pdf_conversion_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark converting one PDF to text/images/metadata, in-process.

    Calls the same per-PDF conversion logic `PDFExtractionStage` dispatches to
    its worker pool, but in-process (no `multiprocessing` spawn), isolating
    the actual parsing cost from process-startup/import overhead.
    """
    pdf_file = sample_pdf_files[0]

    def run():
        data_model = _fresh_data_model(tmp_path)
        ok = _convert_single_pdf(data_model, pdf_file)
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return ok

    assert benchmark(run) is True


def test_pdf_extraction_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark converting the full PDF sample to per-page text, images, and
    metadata."""

    def run():
        data_model = _fresh_data_model(tmp_path)
        stage = PDFExtractionStage(
            data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
        )
        parsed_count = stage.run()
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return parsed_count

    parsed_count = benchmark(run)
    assert parsed_count == len(sample_pdf_files)


def test_end_to_end_pipeline_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark the full PDF -> text/image -> embeddings pipeline (no OCR),
    using the real ST/CLIP models (see module docstring)."""

    def run():
        data_model = _fresh_data_model(tmp_path)
        pipeline = PDFProcessingPipeline(
            data_model.data_dir, text_model_type="ST", visual_model_type="CLIP"
        )
        timings = pipeline.process_pdfs(
            sample_pdf_files,
            do_text_embedding=True,
            do_img_embedding=True,
            do_metadata_collection=True,
            do_ocr=False,
        )
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return timings

    timings = benchmark(run)
    assert len(timings) == 4


def test_text_embedding_benchmark(benchmark, sample_pdf_files, tmp_path_factory):
    """Benchmark embedding the extracted page text with the real ST model.

    The `TextEmbeddingStage` (and its model) is built once, outside the
    benchmarked callable, so repeated rounds measure encoding throughput
    rather than re-paying the one-time model download/load cost.
    """
    data_dir = tmp_path_factory.mktemp("text_embedding_benchmark")
    data_model = DataModel(data_dir.as_posix())
    PDFExtractionStage(
        data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
    ).run()
    stage = TextEmbeddingStage(data_model=data_model, model_type="ST")

    benchmark(stage.run)


def test_image_embedding_benchmark(benchmark, sample_pdf_files, tmp_path_factory):
    """Benchmark embedding the extracted page images with the real CLIP model.

    The `PageImageEmbeddingStage` (and its model) is built once, outside the
    benchmarked callable, so repeated rounds measure encoding throughput
    rather than re-paying the one-time model download/load cost.
    """
    data_dir = tmp_path_factory.mktemp("image_embedding_benchmark")
    data_model = DataModel(data_dir.as_posix())
    PDFExtractionStage(
        data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
    ).run()
    stage = PageImageEmbeddingStage(
        data_model=data_model, model_type="CLIP", cpu_count=CPU_COUNT
    )

    benchmark(stage.run)
