"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import contextlib
import logging
import os

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

from ..config import DataModel
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage


def _build_ocr_engine(ocr_type: str, **kwargs) -> BaseOCR:
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
            f"Must be one of: {list(ocr_engines.keys())}",
        )

    return ocr_engines[ocr_type](**kwargs)


class OCRProcessingStage(ProcessingStage):
    """Processing stage that performs OCR on PDF page images.

    Reads images from {image_directory}/{digest}/{digest}_{pg_no}.jpeg and
    writes extracted text to {txt_directory}/{digest}/{digest}_{pg_no}.txt.
    """

    def __init__(self, data_model: DataModel, ocr_type: str = "easyocr", **ocr_kwargs):
        self.data_model = data_model
        self.ocr_engine = _build_ocr_engine(ocr_type, **ocr_kwargs)
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        if not CV2_AVAILABLE:
            raise ImportError(
                "cv2 (OpenCV) is required for OCR processing. "
                "Install it with: pip install opencv-python",
            )

        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: "
                f"{self.data_model.image_directory}",
            )

        try:
            self.ocr_engine.validate()
        except Exception as e:
            raise ValueError(f"OCR engine validation failed: {e}") from e

    def run(self):
        os.makedirs(self.data_model.txt_directory, exist_ok=True)

        error_count = 0
        all_images: list = []
        all_metadata: list[tuple[str, int]] = []
        engine_name = self.ocr_engine.__class__.__name__.lower()

        for digest_dir in os.scandir(self.data_model.image_directory):
            if not digest_dir.is_dir():
                continue

            digest = digest_dir.name
            os.makedirs(self.data_model.txt_pdf_directory(digest), exist_ok=True)

            page_files = sorted(
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")],
            )

            for page_file in page_files:
                image_path = os.path.join(digest_dir.path, page_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {image_path}")
                        error_count += 1
                        continue
                    if "paddle" not in engine_name:
                        with contextlib.suppress(Exception):
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    page_num = int(page_file.split("_")[-1].replace(".jpeg", ""))
                    all_images.append(image)
                    all_metadata.append((digest, page_num))
                except Exception as e:
                    self.logger.error(f"Error loading {image_path}: {e}")
                    error_count += 1

        if not all_images:
            self.logger.info(
                "OCR processing complete. Processed: 0, Errors: %d", error_count
            )
            return

        _BATCH_SIZE = 1000
        all_texts: list[str] = []
        for batch_start in range(0, len(all_images), _BATCH_SIZE):
            batch = all_images[batch_start : batch_start + _BATCH_SIZE]
            try:
                all_texts.extend(self.ocr_engine.extract_text(batch))
            except Exception as e:
                self.logger.error(
                    f"OCR failed for batch starting at index {batch_start}: {e}"
                )
                all_texts.extend("" for _ in batch)
                error_count += len(batch)

        processed_count = 0
        for (digest, page_num), text in zip(all_metadata, all_texts, strict=True):
            if self._write_page_text(digest, page_num, text):
                processed_count += 1
            else:
                error_count += 1

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}",
        )

    def _write_page_text(self, digest: str, page_num: int, text: str) -> bool:
        try:
            txt_output_path = self.data_model.txt_page_path(digest, page_num)
            os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(text)
            self.logger.debug(f"Processed: {txt_output_path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Error writing OCR text for {digest} page {page_num}: {e}"
            )
            return False
