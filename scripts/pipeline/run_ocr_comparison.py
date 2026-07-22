"""Run PDF extraction and OCR comparison on the sample test PDFs."""

import argparse
import shutil
import subprocess
import sys
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PDF extraction and OCR comparison on sample test PDFs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests" / "test_data" / "small",
        help="Path to the sample PDF test data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tmp_ocr_compare",
        help="Directory to write OCR outputs and temporary files.",
    )
    parser.add_argument(
        "--ocr-type",
        choices=["easyocr", "paddleocr", "olmocr", "ocrmypdf"],
        default="easyocr",
        help="OCR backend to benchmark.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        choices=range(1, 6),
        metavar="N",
        help="Maximum number of workers for multiprocessing OCR (1-5).",
    )
    parser.add_argument(
        "--fake-ocr",
        action="store_true",
        help="Use fake OCR instead of the real OCR backend.",
    )
    return parser.parse_args()


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        pyproj = candidate / "pyproject.toml"
        gov_dir = candidate / "govscape"
        if pyproj.exists() and gov_dir.exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from script path")


def _is_module_available(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _install_dependencies(packages: list[str]) -> None:
    if not packages:
        return

    print(f"Installing missing OCR dependencies: {packages}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        check=True,
    )


def _ensure_ocr_dependencies(ocr_type: str) -> None:
    if ocr_type == "easyocr":
        module_requirements = ["easyocr", "cv2"]
        package_requirements = ["easyocr", "opencv-python-headless"]
    elif ocr_type == "paddleocr":
        module_requirements = ["paddleocr", "paddle", "cv2"]
        package_requirements = ["paddleocr", "paddlepaddle", "opencv-python-headless"]
    elif ocr_type == "olmocr":
        module_requirements = ["olmocr"]
        package_requirements = ["olmocr"]
    elif ocr_type == "ocrmypdf":
        module_requirements = ["ocrmypdf", "pytesseract", "cv2"]
        package_requirements = ["ocrmypdf", "pytesseract", "opencv-python-headless"]
    else:
        raise ValueError(f"Unsupported OCR type: {ocr_type}")

    missing_packages = []
    for module_name, package_name in zip(
        module_requirements,
        package_requirements,
        strict=True,
    ):
        if not _is_module_available(module_name):
            missing_packages.append(package_name)

    if missing_packages:
        _install_dependencies(missing_packages)

    if ocr_type == "olmocr" and _is_module_available("olmocr"):
        import olmocr

        if not hasattr(olmocr, "OLMOcr"):
            raise RuntimeError(
                "Installed olmocr package does not expose a compatible `olmocr.OLMOcr` "
                "API. Please install a compatible olmocr release or use a different "
                "OCR backend."
            )

    if ocr_type == "ocrmypdf" and shutil.which("tesseract") is None:
        print(
            "WARNING: Tesseract binary not found in PATH. "
            "ocrmypdf requires the system tesseract executable to perform OCR. "
            "Install it separately if you want real OCR results.",
        )


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root(Path(__file__).resolve())
    data_dir = args.data_dir
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True)

    data_model = DataModel(str(output_dir))
    pdf_files = sorted((data_dir / "PDFs").glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir / 'PDFs'}")

    print(f"Using repository root: {repo_root}")
    print(f"Output directory: {output_dir}")
    print(f"Converting {len(pdf_files)} PDFs to images...")
    print(
        "Running with "
        f"{'fake OCR' if args.fake_ocr else 'real OCR backend'} "
        f"({args.ocr_type})"
    )
    print(f"Multiprocessing stage will use up to {args.max_workers} workers")

    print("Loading OCR and PDF extraction modules...")
    if not args.fake_ocr:
        _ensure_ocr_dependencies(args.ocr_type)

    import govscape.processing.ocr_processing_stage as ocr_stage_module
    from govscape.processing.ocr_processing_stage import OCRProcessingStage
    from govscape.processing.pdf_extraction_stage import PDFExtractionStage

    print("Initializing PDF extraction stage...")
    extract_stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=[str(path) for path in pdf_files],
        cpu_count=1,
    )
    print("Running PDF extraction stage...")
    extract_stage.run()
    print("PDF extraction complete.")

    class FakeExecutor:
        def __init__(
            self, max_workers=None, mp_context=None, initializer=None, initargs=()
        ):
            self._initializer = initializer
            self._initargs = initargs

        def __enter__(self):
            if self._initializer is not None:
                self._initializer(*self._initargs)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, items):
            return [fn(item) for item in items]

    class FakeCV2:
        COLOR_BGR2RGB = 4

        @staticmethod
        def imread(path):
            image = Image.open(path).convert("RGB")
            return np.array(image)

        @staticmethod
        def cvtColor(image, _code):
            return image

    if args.fake_ocr:
        print(
            "Running in fake OCR mode; using synthetic OCR engine and serial "
            "execution mock."
        )
        ocr_stage_module.cv2 = FakeCV2
        ocr_stage_module.CV2_AVAILABLE = True

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
                ocr_type=args.ocr_type,
                batch_size=1,
                max_workers=1,
            )
            multiprocessing_stage = OCRProcessingStage(
                data_model=data_model,
                ocr_type=args.ocr_type,
                batch_size=1,
                max_workers=args.max_workers,
            )

            print("Validating fake OCR stage...")
            serial_stage.validate()
            print("Starting serial fake OCR processing...")
            serial_start = time.perf_counter()
            serial_stage.run_single_threaded()
            serial_elapsed = time.perf_counter() - serial_start
            print("Serial fake OCR processing complete.")

            print("Starting multiprocessing fake OCR processing...")
            parallel_start = time.perf_counter()
            multiprocessing_stage.run_parallel()
            parallel_elapsed = time.perf_counter() - parallel_start
            print("Multiprocessing fake OCR processing complete.")
    else:
        print("Running with real OCR backend.")
        serial_stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type=args.ocr_type,
            batch_size=1,
            max_workers=1,
        )
        multiprocessing_stage = OCRProcessingStage(
            data_model=data_model,
            ocr_type=args.ocr_type,
            batch_size=1,
            max_workers=args.max_workers,
        )

        print("Validating real OCR stage...")
        serial_stage.validate()
        print("Starting serial OCR processing...")
        serial_start = time.perf_counter()
        serial_stage.run_single_threaded()
        serial_elapsed = time.perf_counter() - serial_start
        print("Serial OCR processing complete.")

        print("Starting multiprocessing OCR processing...")
        parallel_start = time.perf_counter()
        multiprocessing_stage.run_parallel()
        parallel_elapsed = time.perf_counter() - parallel_start
        print("Multiprocessing OCR processing complete.")

    print(f"serial={serial_elapsed:.6f}s multiprocessing={parallel_elapsed:.6f}s")


if __name__ == "__main__":
    main()
