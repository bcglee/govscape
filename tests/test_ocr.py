"""Tests for OCR implementations and OCRProcessingStage."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage
from govscape.processing.ocr import (
    EasyOCRImpl,
    OcrMyPDFImpl,
    OLMOcrImpl,
    PaddleOCRImpl,
)


def _create_test_image(text: str, size=(300, 80)) -> np.ndarray:
    """Create a simple white RGB image with black text.

    Returns a numpy array representing the image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        return np.array(img)
    except Exception:
        # Fall back to a blank numpy image if PIL is not available
        return np.full((size[1], size[0], 3), 255, dtype=np.uint8)


def _is_olmocr_compatible() -> bool:
    # olmOCR runs a ~7B vision-language model (large download + slow, GPU-bound
    # inference), so its real-inference test is opt-in via RUN_OLMOCR_MODEL=1
    # rather than run on every suite invocation.
    if not os.environ.get("RUN_OLMOCR_MODEL"):
        return False
    try:
        import transformers
    except ImportError:
        return False

    return hasattr(transformers, "Qwen2_5_VLForConditionalGeneration")


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing and yield DataModel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_model = DataModel(tmpdir)
        os.makedirs(data_model.image_directory, exist_ok=True)
        yield tmpdir, data_model


class _StaticTextOCR:
    """Fake OCR engine returning a fixed string for every image.

    Lets us exercise the OCRProcessingStage orchestration (batching, worker
    dispatch, metadata mapping, writing) without loading a real OCR model.
    """

    TEXT = "fake-ocr-text"

    def validate(self) -> None:
        return None

    def extract_text(self, images):
        return [self.TEXT for _ in images]


class _InlineProcessPoolExecutor:
    """Drop-in ProcessPoolExecutor substitute that runs everything in-process.

    It honors ``initializer``/``initargs`` exactly like the real executor, so the
    per-worker engine setup in ``run`` is exercised, while avoiding the
    spawning of real processes (and loading of real OCR models) during tests.
    """

    map_calls = 0

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
        type(self).map_calls += 1
        return [fn(item) for item in items]


OCR_IMPLS = [
    (EasyOCRImpl, {"languages": ["en"], "gpu": False}, "easyocr"),
    (PaddleOCRImpl, {"language": "en", "use_gpu": False}, "paddleocr"),
    (OLMOcrImpl, {"model_name": "default"}, "olmocr"),
    (OcrMyPDFImpl, {"language": "eng"}, None),
]


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_implementations_on_sample_images(impl_class, init_args, skip_pkg):
    """Functionality-oriented test: each OCR implementation should extract expected text

    This test will be skipped for implementations whose dependencies are not installed.
    """
    # Skip based on known package name when provided
    if skip_pkg:
        pytest.importorskip(skip_pkg)

    # Skip olmocr if the installed package is not compatible with the expected API.
    if impl_class is OLMOcrImpl and not _is_olmocr_compatible():
        pytest.skip(
            "Installed olmocr package does not expose a compatible `OLMOcr` API."
        )

    # OcrMyPDF requires both ocrmypdf and pytesseract
    if impl_class is OcrMyPDFImpl:
        pytest.importorskip("ocrmypdf")
        pytest.importorskip("pytesseract")

    expected_pairs = [
        ("TEST ONE", _create_test_image("TEST ONE")),
        ("HELLO 123", _create_test_image("HELLO 123")),
    ]

    # Instantiate and validate the OCR engine
    ocr = impl_class(**init_args)
    try:
        ocr.validate()
    except ImportError:
        pytest.skip("Required OCR dependency is not available")

    # Extract the whole batch at once and assert each result contains its substring
    images = [img for _expected, img in expected_pairs]
    extracted = ocr.extract_text(images)
    assert isinstance(extracted, list)
    assert len(extracted) == len(expected_pairs)
    for (expected, _img), text in zip(expected_pairs, extracted, strict=True):
        assert isinstance(text, str)
        assert expected.lower().split()[0] in text.lower()


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_processing_stage_writes_txt(
    impl_class, init_args, skip_pkg, temp_data_dir
):
    """Test OCRProcessingStage pipeline writes text files using different OCR engines.

    The OCR engine's actual `extract_text` is mocked to return deterministic text so the
    stage's file-writing behavior can be asserted for each implementation.
    """
    pytest.importorskip("cv2")

    _, data_model = temp_data_dir

    # Map impl_class to ocr_type string expected by OCRProcessingStage
    impl_to_type = {
        EasyOCRImpl: "easyocr",
        PaddleOCRImpl: "paddleocr",
        OLMOcrImpl: "olmocr",
        OcrMyPDFImpl: "ocrmypdf",
    }

    ocr_type = impl_to_type[impl_class]

    stage = OCRProcessingStage(data_model=data_model, ocr_type=ocr_type, **init_args)

    # Create a sample image on disk that cv2 can read
    import cv2

    digest = "abc123def456abc123def456abc123def45"
    img_dir = os.path.join(data_model.image_directory, digest)
    os.makedirs(img_dir, exist_ok=True)

    img = _create_test_image("PIPELINE TEST")
    img_path = os.path.join(img_dir, f"{digest}_0.jpeg")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mocked_text = "pipeline extracted text"

    # Patch engine methods to avoid external OCR dependencies during pipeline test.
    # extract_text is batched, so it returns a list with one entry per image.
    # run() dispatches batches to worker processes that rebuild the engine via
    # _build_ocr_engine, so patch that (and run the pool inline) to keep the test
    # hermetic and independent of the multiprocessing start method.
    worker_engine = MagicMock()
    worker_engine.extract_text.return_value = [mocked_text]
    with (
        patch.object(stage.ocr_engine, "validate", return_value=None),
        patch.object(stage.ocr_engine, "extract_text", return_value=[mocked_text]),
        patch(
            "govscape.processing.ocr_processing_stage._build_ocr_engine",
            return_value=worker_engine,
        ),
        patch(
            "govscape.processing.ocr_processing_stage.ProcessPoolExecutor",
            _InlineProcessPoolExecutor,
        ),
    ):
        # Validate and run stage
        stage.validate()
        stage.run()

    # Verify txt directory and file created with expected content
    txt_file = data_model.txt_page_path(digest, 0)
    assert os.path.exists(txt_file)
    with open(txt_file, encoding="utf-8") as f:
        content = f.read()
        assert content == mocked_text


def test_run_dispatches_batches_through_process_pool(temp_data_dir):
    """run() should dispatch batches through a ProcessPoolExecutor, building the
    engine via the per-worker initializer."""
    pytest.importorskip("cv2")

    _, data_model = temp_data_dir
    import cv2

    digest = "processpool123456"
    img_dir = os.path.join(data_model.image_directory, digest)
    os.makedirs(img_dir, exist_ok=True)

    img_path = os.path.join(img_dir, f"{digest}_0.jpeg")
    img = _create_test_image("PROCESS POOL")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="easyocr",
        batch_size=1,
        max_workers=2,
    )

    _InlineProcessPoolExecutor.map_calls = 0
    with (
        patch.object(stage.ocr_engine, "validate", return_value=None),
        patch(
            "govscape.processing.ocr_processing_stage._build_ocr_engine",
            return_value=_StaticTextOCR(),
        ),
        patch(
            "govscape.processing.ocr_processing_stage.ProcessPoolExecutor",
            _InlineProcessPoolExecutor,
        ),
    ):
        stage.run()

    assert _InlineProcessPoolExecutor.map_calls == 1
    txt_file = data_model.txt_page_path(digest, 0)
    assert os.path.exists(txt_file)
    with open(txt_file, encoding="utf-8") as f:
        assert f.read() == _StaticTextOCR.TEXT
