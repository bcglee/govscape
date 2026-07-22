"""OcrMyPDF implementation."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PIL import Image

from .base_ocr import BaseOCR

try:
    import ocrmypdf
    import pytesseract
except ImportError:
    ocrmypdf = None
    pytesseract = None


class OcrMyPDFImpl(BaseOCR):
    """OCR implementation using OcrMyPDF.

    OcrMyPDF adds an OCR text layer to scanned PDFs.
    Note: This implementation extracts text from a page image by converting
    to PDF, running OCR, and extracting text back.
    """

    def __init__(
        self,
        language: str = "eng",
        output_type: str = "txt",
        max_workers: int | None = None,
    ):
        """Initialize OcrMyPDF.

        Args:
            language: Tesseract language code (e.g., 'eng', 'fra'). Defaults to 'eng'.
            output_type: Output type ('txt' or 'searchable_pdf'). Defaults to 'txt'.
            max_workers: Number of pages to OCR concurrently. Each Tesseract call is
                single-threaded, so running a batch of pages in parallel gives a
                near-linear speedup (~6x) on a multi-core host. Defaults to
                ``min(8, os.cpu_count())``; set to 1 to disable concurrency (e.g.
                when the surrounding pipeline already parallelizes across processes).
        """
        self.language = language
        self.output_type = output_type
        self.max_workers = max_workers or min(8, os.cpu_count() or 1)
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate OcrMyPDF installation and dependencies."""
        if ocrmypdf is None:
            raise ImportError(
                "ocrmypdf is not installed. Install it with: pip install ocrmypdf"
            )

        if pytesseract is None:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            )

        self.logger.info(
            f"OcrMyPDF initialized with language: {self.language}, "
            f"output: {self.output_type}"
        )

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of images using OcrMyPDF/Tesseract.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image.
        """
        if pytesseract is None:
            self.validate()

        if self.max_workers <= 1 or len(images) <= 1:
            return [self._extract_single(image) for image in images]

        # Tesseract is single-threaded per call and runs in a subprocess (so the
        # GIL is not a bottleneck); OCR the batch's pages concurrently. map()
        # preserves input order, keeping results aligned with pages.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._extract_single, images))

    def _extract_single(self, image: np.ndarray) -> str:
        try:
            # Convert numpy array to PIL Image if needed
            pil_image = (
                Image.fromarray(image.astype("uint8"))
                if isinstance(image, np.ndarray)
                else image
            )
            # Extract text using pytesseract (which uses Tesseract OCR)
            return pytesseract.image_to_string(pil_image, lang=self.language)
        except Exception as e:
            self.logger.error(f"Error during OcrMyPDF text extraction: {e}")
            return ""
