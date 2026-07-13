# AI modified: 2026-07-13 9572ec45
"""Build the date-extraction eval corpus from downloaded sample PDFs.

Runs PDFExtractionStage over the PDFs, OCRs pages whose extracted text is
empty (scanned pages), and writes corpus/manifest.jsonl with one record per
successfully processed document.
"""

import argparse
import glob
import json
import logging
import os

import numpy as np

import duckdb
from PIL import Image

from govscape.config import DataModel
from govscape.processing.ocr_processing_stage import _build_ocr_engine
from govscape.processing.pdf_extraction_stage import PDFExtractionStage

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

MIN_TEXT_CHARS = 20


def _page_numbers(data_model: DataModel, digest: str) -> list[int]:
    img_dir = data_model.img_pdf_directory(digest)
    pages = [
        int(f.rsplit("_", 1)[1].removesuffix(".jpeg"))
        for f in os.listdir(img_dir)
        if f.endswith(".jpeg")
    ]
    return sorted(pages)


def _page_text_len(data_model: DataModel, digest: str, pg_no: int) -> int:
    path = data_model.txt_page_path(digest, pg_no)
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return len(f.read().strip())


def _ocr_empty_pages(data_model: DataModel, digests: list[str], ocr_type: str) -> dict:
    """OCR pages with no embedded text. Returns digest -> list of OCR'd pages."""
    targets: list[tuple[str, int]] = [
        (digest, pg_no)
        for digest in digests
        for pg_no in _page_numbers(data_model, digest)
        if _page_text_len(data_model, digest, pg_no) < MIN_TEXT_CHARS
    ]

    logging.info("OCR needed for %d pages", len(targets))
    if not targets:
        return {}

    engine = _build_ocr_engine(ocr_type)
    engine.validate()
    ocred: dict[str, list[int]] = {}
    for i, (digest, pg_no) in enumerate(targets):
        image = np.asarray(Image.open(data_model.img_page_path(digest, pg_no)))
        text = engine.extract_text([image])[0]
        if len(text.strip()) >= MIN_TEXT_CHARS:
            with open(
                data_model.txt_page_path(digest, pg_no), "w", encoding="utf-8"
            ) as f:
                f.write(text)
            ocred.setdefault(digest, []).append(pg_no)
        if (i + 1) % 100 == 0:
            logging.info("OCR progress: %d/%d pages", i + 1, len(targets))
    return ocred


def _crawl_info(cdx_parquet: str, digests: list[str]) -> dict:
    rows = (
        duckdb.connect()
        .execute(
            "SELECT digest, min(crawl_date), any_value(url) "
            "FROM read_parquet(?) WHERE digest IN "
            f"({','.join('?' for _ in digests)}) GROUP BY digest",
            [cdx_parquet, *digests],
        )
        .fetchall()
    )
    return {digest: {"crawl_date": cd, "url": url} for digest, cd, url in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf_dir", required=True, help="Directory of sample PDFs")
    parser.add_argument("--data_dir", required=True, help="DataModel output directory")
    parser.add_argument("--cdx_parquet", required=True, help="CDX parquet path")
    parser.add_argument("--manifest", required=True, help="Output manifest.jsonl path")
    parser.add_argument("--ocr_type", default="ocrmypdf")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu_count", type=int, default=os.cpu_count())
    args = parser.parse_args()

    pdf_files = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if args.limit:
        pdf_files = pdf_files[: args.limit]
    logging.info("Extracting %d PDFs", len(pdf_files))

    data_model = DataModel(args.data_dir)
    stage = PDFExtractionStage(data_model, pdf_files, args.cpu_count)
    stage.validate()
    ok_count = stage.run()
    logging.info("Extraction succeeded for %d/%d PDFs", ok_count, len(pdf_files))

    digests = [
        os.path.splitext(os.path.basename(f))[0]
        for f in pdf_files
        if os.path.isfile(
            data_model.metadata_file_path(os.path.splitext(os.path.basename(f))[0])
        )
    ]

    ocred = _ocr_empty_pages(data_model, digests, args.ocr_type)
    crawl_info = _crawl_info(args.cdx_parquet, digests)

    with open(args.manifest, "w", encoding="utf-8") as out:
        for digest in digests:
            with open(data_model.metadata_file_path(digest), encoding="utf-8") as f:
                meta = json.load(f)
            pages = _page_numbers(data_model, digest)
            record = {
                "digest": digest,
                "pdf_path": os.path.join(args.pdf_dir, f"{digest}.pdf"),
                "num_pages": meta["num_pages"],
                "pretty_name": meta["pretty_name"],
                "embedded_creation_date": meta["creation_date"],
                "url": crawl_info.get(digest, {}).get("url"),
                "first_crawl_date": crawl_info.get(digest, {}).get("crawl_date"),
                "ocr_pages": ocred.get(digest, []),
                "empty_pages": [
                    p
                    for p in pages
                    if _page_text_len(data_model, digest, p) < MIN_TEXT_CHARS
                ],
            }
            out.write(json.dumps(record) + "\n")
    logging.info("Wrote manifest with %d records to %s", len(digests), args.manifest)


if __name__ == "__main__":
    main()
