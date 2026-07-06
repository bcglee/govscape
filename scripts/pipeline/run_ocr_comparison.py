"""Run PDF extraction and OCR comparison on the sample test PDFs."""

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from PIL import Image

from govscape.config import DataModel


class FakeOCR:
    def validate(self) -> None:
        return None

    def extract_text(self, images):
        return [f"ocr-result-for-{len(images)}-image(s)" for _ in images]


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        pyproj = candidate / "pyproject.toml"
        gov_dir = candidate / "govscape"
        if pyproj.exists() and gov_dir.exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from script path")


def main() -> None:
    repo_root = _find_repo_root(Path(__file__).resolve())
    data_dir = repo_root / "tests" / "test_data" / "small"
    output_dir = repo_root / "tmp_ocr_compare"
    output_dir.mkdir(exist_ok=True)

    data_model = DataModel(str(output_dir))
    pdf_files = sorted((data_dir / "PDFs").glob("*.pdf"))

    class FakeCV2:
        COLOR_BGR2RGB = 4

        @staticmethod
        def imread(path):
            image = Image.open(path).convert("RGB")
            return np.array(image)

        @staticmethod
        def cvtColor(image, _code):
            return image

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir / 'PDFs'}")

    print(f"Using repository root: {repo_root}")
    print(f"Converting {len(pdf_files)} PDFs to images...")

    import govscape.processing.ocr_processing_stage as ocr_stage_module
    from govscape.processing.ocr_processing_stage import OCRProcessingStage
    from govscape.processing.pdf_extraction_stage import PDFExtractionStage

    ocr_stage_module.cv2 = FakeCV2
    ocr_stage_module.CV2_AVAILABLE = True

    extract_stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=[str(path) for path in pdf_files],
        cpu_count=1,
    )
    extract_stage.run()

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    with (
        patch(
            "govscape.processing.ocr_processing_stage._build_ocr_engine",
            return_value=FakeOCR(),
        ),
        patch(
            "govscape.processing.ocr_processing_stage.ProcessPoolExecutor",
            FakeExecutor,
        ),
    ):
        serial_stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
            batch_size=1,
            max_workers=1,
        )
        multiprocessing_stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type="easyocr",
            batch_size=1,
            max_workers=2,
        )

        print("Running serial processing...")
        serial_start = time.perf_counter()
        serial_stage.run_single_threaded()
        serial_elapsed = time.perf_counter() - serial_start

        print("Running multiprocessing OCR...")
        parallel_start = time.perf_counter()
        multiprocessing_stage.run_parallel()
        parallel_elapsed = time.perf_counter() - parallel_start

    print(f"serial={serial_elapsed:.6f}s multiprocessing={parallel_elapsed:.6f}s")


if __name__ == "__main__":
    main()
