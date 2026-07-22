"""olmOCR implementation.

olmOCR is a document-OCR vision-language model (Qwen2.5-VL fine-tune). Upstream
`olmocr` drives it through an async pipeline that POSTs pages to a served vLLM
instance; that is heavy to orchestrate and pins a specific torch build. Instead
we load the published bf16 checkpoint directly with transformers and run it
locally, which fits the ``BaseOCR.extract_text`` contract and needs no server.
We reuse olmocr's own prompt builder and front-matter parser so the input and
output handling match upstream exactly.
"""

import logging
from typing import Any

import numpy as np

from PIL import Image

from .base_ocr import BaseOCR

try:
    import torch
    from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt
    from olmocr.train.front_matter import FrontMatterParser
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    _OLMOCR_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when deps are missing
    torch = None  # type: ignore[assignment]
    _OLMOCR_AVAILABLE = False

# bf16 checkpoint (no vLLM/FP8 required), loadable directly with transformers.
DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025"


class OLMOcrImpl(BaseOCR):
    """OCR implementation using the olmOCR vision-language model via transformers."""

    def __init__(
        self,
        model_name: str = "default",
        gpu: bool = True,
        target_longest_image_dim: int = 1288,
        max_new_tokens: int = 4096,
    ):
        """Initialize olmOCR.

        Args:
            model_name: HuggingFace model id (or local path). "default" resolves to
                the published bf16 checkpoint ``allenai/olmOCR-2-7B-1025``.
            gpu: Use the GPU if available. Defaults to True.
            target_longest_image_dim: Longest-side pixel size each page is resized
                to before OCR (upstream olmOCR default is 1288).
            max_new_tokens: Upper bound on generated tokens per page.
        """
        self.model_name = (
            DEFAULT_MODEL if model_name in (None, "default") else model_name
        )
        self.gpu = gpu
        self.target_longest_image_dim = target_longest_image_dim
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Load the olmOCR model and processor."""
        if not _OLMOCR_AVAILABLE:
            raise ImportError(
                "olmOCR requires `olmocr` and `transformers`. Install with: "
                "poetry install --extras olmocr"
            )

        try:
            self.device = "cuda" if (self.gpu and torch.cuda.is_available()) else "cpu"
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype
            ).to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.logger.info(
                f"olmOCR model '{self.model_name}' loaded on {self.device}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize olmOCR: {e}") from e

    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of page images using olmOCR.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image.
        """
        if self.model is None:
            self.validate()
        if not images:
            return []

        prompt = build_no_anchoring_v4_yaml_prompt()
        return [self._extract_single(image, prompt) for image in images]

    def _extract_single(self, image: np.ndarray, prompt: str) -> str:
        try:
            pil_image = self._prepare_image(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[chat_text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            new_tokens = generated[0][inputs.input_ids.shape[1] :]
            decoded = self.processor.decode(new_tokens, skip_special_tokens=True)
            return self._parse_response(decoded)
        except Exception as e:
            self.logger.error(f"Error during olmOCR text extraction: {e}")
            return ""

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        pil_image = (
            Image.fromarray(image.astype("uint8"))
            if isinstance(image, np.ndarray)
            else image
        )
        pil_image = pil_image.convert("RGB")
        width, height = pil_image.size
        longest = max(width, height)
        if longest > self.target_longest_image_dim:
            scale = self.target_longest_image_dim / longest
            pil_image = pil_image.resize((int(width * scale), int(height * scale)))
        return pil_image

    def _parse_response(self, model_output: str) -> str:
        """Parse olmOCR's YAML-front-matter response into plain page text."""
        try:
            parser = FrontMatterParser(front_matter_class=PageResponse)
            front_matter, text = parser._extract_front_matter_and_text(model_output)
            page_response = parser._parse_front_matter(front_matter, text)
            return page_response.natural_text or ""
        except Exception:
            # If the model didn't emit valid front matter, return the raw text.
            return model_output.strip()
