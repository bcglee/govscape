"""PaddleOCR implementation."""

import logging
from typing import Any

import numpy as np

from .base_ocr import BaseOCR

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class PaddleOCRImpl(BaseOCR):
    """OCR implementation using PaddleOCR.

    PaddleOCR is a multilingual OCR toolkit with high accuracy and efficiency.
    """

    def __init__(self, language: str = "en", use_gpu: bool = False):
        """Initialize PaddleOCR.

        Args:
            language: Language code (e.g., 'en', 'ch', 'fr'). Defaults to 'en'.
            use_gpu: Whether to use GPU for inference. Defaults to False.
        """
        self.language = language
        self.use_gpu = use_gpu
        self.ocr: Any = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate PaddleOCR installation and initialize the OCR engine."""
        if PaddleOCR is None:
            raise ImportError(
                "paddleocr is not installed. Install it with: "
                "pip install paddleocr paddlepaddle"
            )

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang=self.language, use_gpu=self.use_gpu
            )
            self.logger.info(
                f"PaddleOCR initialized with language: {self.language}, "
                f"GPU: {self.use_gpu}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of images using PaddleOCR.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image.
        """
        if self.ocr is None:
            self.validate()

        return [self._extract_single(image) for image in images]

    def _extract_single(self, image: np.ndarray) -> str:
        try:
            result = self.ocr.ocr(image, cls=True)
            text_lines: list[str] = []
            if result:
                for line in result:
                    text_lines.extend(detection[1][0] for detection in line)
            return "\n".join(text_lines)
        except Exception as e:
            self.logger.error(f"Error during PaddleOCR text extraction: {e}")
            return ""
