"""OLMOcr OCR implementation."""

import logging
from typing import Any

import numpy as np

from .base_ocr import BaseOCR

try:
    import olmocr
except ImportError:
    olmocr = None


def _get_olmocr_model_class() -> type | None:
    if olmocr is None:
        return None

    if hasattr(olmocr, "OLMOcr"):
        return olmocr.OLMOcr

    return None


class OLMOcrImpl(BaseOCR):
    """OCR implementation using OLMOcr.

    OLMOcr is an open-source language model-based OCR system.
    """

    def __init__(self, model_name: str = "default"):
        """Initialize OLMOcr.

        Args:
            model_name: The OLMOcr model to use. Defaults to "default".
        """
        self.model_name = model_name
        self.model: Any = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate OLMOcr installation and initialize the model."""
        if olmocr is None:
            raise ImportError(
                "olmocr is not installed. Install it with: pip install olmocr"
            )

        model_class = _get_olmocr_model_class()
        if model_class is None:
            raise RuntimeError(
                "Installed olmocr package does not provide a compatible `OLMOcr` "
                "API. Please install a compatible olmocr release or use a different "
                "OCR backend."
            )

        try:
            self.model = model_class(model_name=self.model_name)
            self.logger.info(
                f"OLMOcr model '{self.model_name}' initialized successfully"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OLMOcr: {e}") from e

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of images using OLMOcr.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image.
        """
        if self.model is None:
            self.validate()

        return [self._extract_single(image) for image in images]

    def _extract_single(self, image: np.ndarray) -> str:
        try:
            result = self.model.recognize(image)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            self.logger.error(f"Error during OLMOcr text extraction: {e}")
            return ""
