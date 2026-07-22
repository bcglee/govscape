"""EasyOCR implementation."""

import logging
from typing import Any

import numpy as np

from .base_ocr import BaseOCR

try:
    import easyocr
except ImportError:
    easyocr = None


class EasyOCRImpl(BaseOCR):
    """OCR implementation using EasyOCR.

    EasyOCR is a Python library for OCR supporting 80+ languages.
    """

    def __init__(
        self,
        languages: list | None = None,
        gpu: bool = False,
        batch_size: int = 256,
    ):
        """Initialize EasyOCR.

        Args:
            languages: List of language codes (e.g., ['en', 'fr']). Defaults to ['en'].
            gpu: Whether to use GPU for inference. Defaults to False.
            batch_size: Number of detected text crops recognized per forward pass.
                Larger values batch the recognition network and are much faster on
                GPU (~1.7x at 256 vs EasyOCR's per-crop default of 1); it is capped
                internally by the number of crops on a page.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.batch_size = batch_size
        self.reader: Any = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate EasyOCR installation and initialize the reader."""
        if easyocr is None:
            raise ImportError(
                "easyocr is not installed. Install it with: pip install easyocr"
            )

        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            self.logger.info(
                f"EasyOCR reader initialized with languages: {self.languages}, "
                f"GPU: {self.gpu}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}") from e

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of images using EasyOCR.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image.
        """
        if self.reader is None:
            self.validate()

        return [self._extract_single(image) for image in images]

    def _extract_single(self, image: np.ndarray) -> str:
        try:
            results = self.reader.readtext(image, batch_size=self.batch_size)
            text_lines = [detection[1] for detection in results]
            return "\n".join(text_lines)
        except Exception as e:
            self.logger.error(f"Error during EasyOCR text extraction: {e}")
            return ""
