"""Run PDF extraction and OCR comparison on the sample test PDFs."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from PIL import Image

from govscape.config import DataModel
from govscape.processing.ocr_processing_stage import OCRProcessingStage
from govscape.processing.pdf_extraction_stage import PDFExtractionStage


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "tests" / "test_data" / "small"
    output_dir = repo_root / "tmp_ocr_compare"
    output_dir.mkdir(exist_ok=True)

    data_model = DataModel(str(output_dir))
    pdf_files = sorted((data_dir / "PDFs").glob("*.pdf"))

    class FakeCV2:
        @staticmethod
        def imread(path):
            image = Image.open(path).convert("RGB")
            return np.array(image)

        @staticmethod
        def cvtColor(image, _code):
            return image

    import govscape.processing.ocr_processing_stage as ocr_stage_module

    ocr_stage_module.cv2 = FakeCV2
    ocr_stage_module.CV2_AVAILABLE = True

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir / 'PDFs'}")

    print(f"Converting {len(pdf_files)} PDFs to images...")
    extract_stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=[str(path) for path in pdf_files],
        cpu_count=1,
    )
    extract_stage.run()

    stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="easyocr",
        batch_size=1,
        max_workers=2,
    )

    def fake_extract_text(images):
        return [f"ocr-result-for-{len(images)}-image(s)" for _ in images]

    with (
        patch.object(stage.ocr_engine, "validate", return_value=None),
        patch.object(stage.ocr_engine, "extract_text", side_effect=fake_extract_text),
    ):
        print("Running single-threaded OCR...")
        single_start = time.perf_counter()
        stage.run_single_threaded()
        single_elapsed = time.perf_counter() - single_start

        print("Running parallel OCR...")
        parallel_start = time.perf_counter()
        stage.run_parallel()
        parallel_elapsed = time.perf_counter() - parallel_start

    print(f"single-threaded={single_elapsed:.6f}s parallel={parallel_elapsed:.6f}s")


if __name__ == "__main__":
    main()
