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
        self.ocr_type = ocr_type
        self.ocr_kwargs = ocr_kwargs
        # primary engine used for validation and single-threaded runs
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

    def run(self, threads: int = 1, batch_size: int = 1000, compare: bool = False):
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
            return None
        # Create batches
        batches: list[list] = [
            all_images[i : i + batch_size]
            for i in range(0, len(all_images), batch_size)
        ]

        def _handle_batch(batch_idx: int, batch: list) -> tuple[list[str], int]:
            try:
                return self.ocr_engine.extract_text(batch), 0
            except Exception as e:
                start_idx = batch_idx * batch_size
                self.logger.error(
                    "OCR failed for batch %d (start %d): %s",
                    batch_idx,
                    start_idx,
                    e,
                )
                return ["" for _ in batch], len(batch)

        def _run_single_threaded(batches_to_run: list[list]) -> tuple[list[str], int]:
            texts: list[str] = []
            errors = 0
            for idx, batch in enumerate(batches_to_run):
                res_texts, res_errors = _handle_batch(idx, batch)
                texts.extend(res_texts)
                errors += res_errors
            return texts, errors

        import time

        # Optionally compare single-threaded vs multi-threaded run times
        single_time = None
        multi_time = None
        all_texts: list[str] = []

        if compare and threads > 1:
            t0 = time.perf_counter()
            _single_texts, single_errors = _run_single_threaded(batches)
            single_time = time.perf_counter() - t0
            self.logger.info(f"Single-threaded OCR extract time: {single_time:.3f}s")

        if threads <= 1:
            t0 = time.perf_counter()
            all_texts, errors = _run_single_threaded(batches)
            multi_time = time.perf_counter() - t0
            error_count += errors
        else:
            # Multi-threaded execution: create one engine instance per worker
            from concurrent.futures import ThreadPoolExecutor

            engines = [
                _build_ocr_engine(self.ocr_type, **self.ocr_kwargs)
                for _ in range(threads)
            ]

            def _worker(batch_index_and_batch):
                idx, batch = batch_index_and_batch
                engine = engines[idx % len(engines)]
                try:
                    return engine.extract_text(batch)
                except Exception as e:
                    start_idx = idx * batch_size
                    self.logger.error(
                        "OCR failed for batch %d (start %d): %s",
                        idx,
                        start_idx,
                        e,
                    )
                    return ["" for _ in batch]

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=threads) as exc:
                # submit batches preserving their index so we can restore order
                futures = [
                    exc.submit(_worker, (i, batch)) for i, batch in enumerate(batches)
                ]

                # collect results in order of submission (which matches batch order)
                def _get_future_result(idx: int, fut):
                    try:
                        return fut.result()
                    except Exception as e:  # pragma: no cover - defensive
                        self.logger.error("Unexpected worker error: %s", e)
                        return ["" for _ in batches[idx]]

                for i, fut in enumerate(futures):
                    res = _get_future_result(i, fut)
                    all_texts.extend(res)
            multi_time = time.perf_counter() - t0
            self.logger.info(
                "Multi-threaded OCR extract time (%d workers): %.3fs",
                threads,
                multi_time,
            )

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

        # Return a summary for external analysis (timings may be None)
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "single_time": single_time,
            "multi_time": multi_time,
            "threads": threads,
            "batch_size": batch_size,
        }

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
