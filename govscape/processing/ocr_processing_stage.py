"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import contextlib
import logging
import os
import time

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from ..config import DataModel
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage


def _build_ocr_engine(ocr_type: str, **kwargs) -> BaseOCR:
    """Factory function to build OCR engines.

    Args:
        ocr_type: Type of OCR engine ('easyocr', 'paddleocr', 'olmocr', 'ocrmypdf').
        **kwargs: Additional arguments to pass to the OCR engine constructor.

    Returns:
        An initialized OCR engine.

    Raises:
        ValueError: If ocr_type is not supported.
    """
    from .ocr import EasyOCRImpl, OcrMyPDFImpl, OLMOcrImpl, PaddleOCRImpl

    ocr_engines = {
        "easyocr": EasyOCRImpl,
        "paddleocr": PaddleOCRImpl,
        "olmocr": OLMOcrImpl,
        "ocrmypdf": OcrMyPDFImpl,
    }

    if ocr_type not in ocr_engines:
        raise ValueError(
            f"Unsupported OCR type: {ocr_type}. "
            f"Must be one of: {list(ocr_engines.keys())}"
        )

    engine_class = ocr_engines[ocr_type]
    return engine_class(**kwargs)


class OCRProcessingStage(ProcessingStage):
    """Processing stage that performs OCR on PDF page images.

    This stage:
    1. Reads page images from {image_directory}/{digest}/{digest}_{pg_no}.jpeg
    2. Applies OCR using the specified engine
    3. Saves extracted text to {txt_directory}/{digest}/{digest}_{pg_no}.txt

    Following the DATA_MODEL.md protocol for text file organization.
    """

    def __init__(self, data_model: DataModel, ocr_type: str = "easyocr", **ocr_kwargs):
        """Initialize the OCR Processing Stage.

        Args:
            data_model: DataModel instance defining directory structure.
            ocr_type: Type of OCR engine to use (default: 'easyocr').
            **ocr_kwargs: Additional arguments to pass to the OCR engine.
                For EasyOCR: languages=['en', ...], gpu=False
                For PaddleOCR: language='en', use_gpu=False
                For OLMOcr: model_name='default'
                For OcrMyPDF: language='eng', output_type='txt'
        """
        self.data_model = data_model
        self.ocr_type = ocr_type
        self.ocr_kwargs = ocr_kwargs
        # primary engine used for validation and single-threaded runs
        self.ocr_engine = _build_ocr_engine(ocr_type, **ocr_kwargs)
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate that the image directory exists and OCR engine is initialized."""
        if not CV2_AVAILABLE:
            raise ImportError(
                "cv2 (OpenCV) is required for OCR processing. "
                "Install it with: pip install opencv-python"
            )

        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: "
                f"{self.data_model.image_directory}"
            )

        try:
            self.ocr_engine.validate()
        except Exception as e:
            raise ValueError(f"OCR engine validation failed: {e}") from e

    def _prepare_images(self, images: list) -> list:
        """Convert images to the expected color format for the configured OCR engine."""
        prepared: list = []
        for img in images:
            img_use = img
            if "paddle" not in self.ocr_type:
                with contextlib.suppress(Exception):
                    img_use = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prepared.append(img_use)
        return prepared

    def _process_image_batch(
        self, engine, batch_images: list[tuple], batch_paths: list[str]
    ):
        """Run OCR on a batch of images with the provided engine."""
        images = [img for img, _ in batch_images]
        prepared_images = self._prepare_images(images)

        try:
            batch_texts = engine.extract_text_batch(prepared_images)
        except Exception as exc:
            self.logger.error("OCR batch failed: %s", exc)
            batch_texts = [""] * len(batch_images)

        texts: list[str] = []
        errors = 0
        for text, img_path in zip(batch_texts, batch_paths, strict=True):
            if not text:
                errors += 1
                self.logger.error("OCR failed for image %s", img_path)
            texts.append(text)

        return texts, errors

    def _run_batch(self, batch: list):
        """Process a batch of image tuples via the configured OCR engine."""
        if not batch:
            return []

        batch_images = [item for item in batch if isinstance(item, tuple)]
        if not batch_images:
            return []

        batch_paths = [img_path for _, img_path in batch_images]
        texts, _ = self._process_image_batch(self.ocr_engine, batch_images, batch_paths)
        return texts

    def run_single_threaded(
        self, all_images: list, batch_size: int = 1000
    ) -> tuple[list[str], int, float]:
        """Process ``all_images`` on the primary OCR engine in the current thread.

        Returns (texts, errors, elapsed_seconds).
        """
        texts: list[str] = []
        errors = 0
        t0 = time.perf_counter()

        for start in range(0, len(all_images), batch_size):
            batch = all_images[start : start + batch_size]
            batch_texts, batch_errors = self._process_image_batch(
                self.ocr_engine,
                batch,
                [img_path for _, img_path in batch],
            )
            texts.extend(batch_texts)
            errors += batch_errors

        elapsed = time.perf_counter() - t0
        return texts, errors, elapsed

    def run(self, threads: int = 1, batch_size: int = 1000, compare: bool = False):
        """Run OCR on all PDF page images and save extracted text.

        If `threads` > 1, runs OCR in a ThreadPoolExecutor with one engine
        instance per worker to avoid engine thread-safety issues. Returns a
        summary dict with counts and optional timing information.
        """
        os.makedirs(self.data_model.txt_directory, exist_ok=True)

        all_images: list = []
        all_metadata: list[tuple[str, int]] = []
        error_count = 0

        # Collect all images and metadata first
        for digest_dir in os.scandir(self.data_model.image_directory):
            if not digest_dir.is_dir():
                continue

            digest = digest_dir.name
            os.makedirs(self.data_model.txt_pdf_directory(digest), exist_ok=True)

            page_files = sorted(
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")]
            )

            for page_file in page_files:
                image_path = os.path.join(digest_dir.path, page_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.warning("Failed to read image: %s", image_path)
                        error_count += 1
                        continue

                    page_num = int(page_file.split("_")[-1].replace(".jpeg", ""))
                    all_images.append((image, image_path))
                    all_metadata.append((digest, page_num))
                except Exception as e:
                    self.logger.error("Error loading %s: %s", image_path, e)
                    error_count += 1

        if not all_images:
            self.logger.info(
                "OCR processing complete. Processed: 0, Errors: %d", error_count
            )
            return None

        from concurrent.futures import ThreadPoolExecutor

        image_batches = [
            all_images[start : start + batch_size]
            for start in range(0, len(all_images), batch_size)
        ]

        def _handle_batch(idx_and_batch):
            idx, batch = idx_and_batch
            engine = engines[idx % len(engines)]
            batch_texts, _ = self._process_image_batch(
                engine,
                batch,
                [img_path for _, img_path in batch],
            )
            return batch_texts

        # Optionally compare single-threaded vs multi-threaded timings
        single_time = None
        multi_time = None
        texts: list[str] = []

        if compare and threads > 1:
            _single_texts, _single_errors, single_time = self.run_single_threaded(
                all_images, batch_size=batch_size
            )
            self.logger.info("Single-threaded OCR extract time: %.3fs", single_time)

        if threads <= 1:
            texts, errors, multi_time = self.run_single_threaded(
                all_images, batch_size=batch_size
            )
            error_count += errors
        else:
            # Multi-threaded execution: create one engine instance per worker
            engines = [
                _build_ocr_engine(self.ocr_type, **self.ocr_kwargs)
                for _ in range(threads)
            ]

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=threads) as exc:
                # map preserves input order
                all_batch_texts = list(exc.map(_handle_batch, enumerate(image_batches)))
            texts = [text for batch_texts in all_batch_texts for text in batch_texts]
            multi_time = time.perf_counter() - t0
            self.logger.info(
                "Multi-threaded OCR extract time (%d workers): %.3fs",
                threads,
                multi_time,
            )

        processed_count = 0

        def _write_text(digest, page_num, text):
            try:
                txt_output_path = self.data_model.txt_page_path(digest, page_num)
                os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
                with open(txt_output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                return True
            except Exception as e:
                self.logger.error(
                    "Error writing OCR text for %s page %d: %s",
                    digest,
                    page_num,
                    e,
                )
                return False

        for (digest, page_num), text in zip(all_metadata, texts, strict=True):
            if _write_text(digest, page_num, text):
                processed_count += 1
            else:
                error_count += 1

        self.logger.info(
            "OCR processing complete. Processed: %d, Errors: %d",
            processed_count,
            error_count,
        )

        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "single_time": single_time,
            "multi_time": multi_time,
            "threads": threads,
        }
