"""Abstract base class for OCR implementations."""

from abc import ABC, abstractmethod

import numpy as np


class BaseOCR(ABC):
    """Abstract base class for OCR engines.

    Subclasses should implement text extraction from PDF pages.
    """

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from a page image.

        Args:
            image: A numpy array representing a page image (from PIL or cv2).

        Returns:
            Extracted text as a string.
        """

    @abstractmethod
    def validate(self) -> None:
        """Validate OCR engine initialization and required dependencies."""

    def extract_text_batch(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of page images."""
        return [self.extract_text(image) for image in images]
