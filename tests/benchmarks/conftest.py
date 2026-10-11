"""Shared fixtures for the CodSpeed PDF-processing benchmark suite.

These benchmarks use the full real-world PDF sample under `tests/test_data/`
(100 government PDFs, ~1,660 pages, ~96MB), rather than a small synthetic
sample, so measurements reflect realistic documents. That corpus is paired
with CDX crawl metadata (see `CONTRIBUTOR.md` -> "Running Govscape Locally"
for how it was generated): `tests/test_data/cdx/digests_manifest.csv` lists
each PDF's filename, digest, and source URL, and is used here as the
authoritative file list.

Because `pytest-codspeed` repeatedly invokes each benchmarked callable, and
this corpus is large, individual CI runs can take several minutes (see the
docstring in `test_pdf_processing_benchmarks.py` for details per benchmark).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
CDX_DIR = TEST_DATA_DIR / "cdx"
PDF_DIR = TEST_DATA_DIR / "pdfs"


@pytest.fixture(scope="module")
def sample_pdf_files() -> list[str]:
    """The full real-world PDF sample, as listed in the CDX digest manifest."""
    with open(CDX_DIR / "digests_manifest.csv", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    return sorted(str(PDF_DIR / row["file"]) for row in manifest_rows)
