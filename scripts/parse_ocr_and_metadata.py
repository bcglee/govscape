from __future__ import annotations

import json
import argparse
import logging
import os
import subprocess
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import random

import fitz
import pyarrow as pa
import pyarrow.parquet as pq

PDF_BUCKET = "s3://eot-pdf-archive/pdfs/"
OCR_BUCKET = "s3://eot-pdf-archive/ai2-olmocr/"
BATCH_SIZE = 100

OUTPUT_DIR = Path("data/parsed")
OCR_DIR = OUTPUT_DIR / "ocr"
METADATA_DIR = OUTPUT_DIR / "metadata"

ERROR_LOG = OUTPUT_DIR / "errors.log"
SAMPLE_SIZE = 1000
SAMPLE_SEED = 42
SAMPLE_IDS_CSV = OUTPUT_DIR / "sampled_pdf_ids.csv"

SCHEMA = pa.schema([
    ("pdf_id", pa.string()),
    ("page_no", pa.int32()),
    ("txt", pa.string()),
])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def s5cmd_ls(prefix: str) -> list[str]:
    result = subprocess.run(
        ["s5cmd", "ls", prefix],
        capture_output=True, text=True, check=True,
    )
    paths = []
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if parts:
            paths.append(parts[-1])
    return paths

def list_pdf_ids() -> set[str]:
    paths = s5cmd_ls(PDF_BUCKET)
    ids = set()
    for p in paths:
        name = p.rsplit("/", 1)[-1]
        if name.endswith(".pdf"):
            ids.add(name[:-4])
    return ids

def sample_pdf_ids(all_ids: set[str], n: int, seed: int) -> list[str]:
    """Uniformly sample n PDF IDs from the full set and freeze them to CSV."""
    rng = random.Random(seed)
    population = sorted(all_ids)  # sort first so the sample is reproducible across runs
    sampled = rng.sample(population, min(n, len(population)))
    with open(SAMPLE_IDS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf_id"])
        for pid in sampled:
            writer.writerow([pid])
    return sampled

def list_jsonl_files() -> set[str]:
    paths = s5cmd_ls(OCR_BUCKET)
    return {p.rsplit("/", 1)[-1] for p in paths if p.endswith(".jsonl")}


def extract_pdf_id(source_file: str) -> str:
    name = source_file.rsplit("/", 1)[-1]
    if name.endswith(".pdf"):
        return name[:-4]
    return name


def extract_metadata_pages(pdf_path: Path) -> list[tuple[int, str | None]]:
    """Extract text from each page. Returns (page_no, txt) pairs.
    txt is None if no text layer exists, "" if text layer is present but empty.
    """
    pages = []
    doc = fitz.open(pdf_path)
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text()
        if text:
            pages.append((page_no, text))
        else:
            blocks = page.get_text("dict")["blocks"]
            has_text_layer = any(b["type"] == 0 for b in blocks)
            pages.append((page_no, "" if has_text_layer else None))
    doc.close()
    return pages


def flush_parquet(rows: list[tuple[str, int, str | None]], out_dir: Path, worker_id: str):
    if not rows:
        return
    table = pa.table({
        "pdf_id": [r[0] for r in rows],
        "page_no": [r[1] for r in rows],
        "txt": [r[2] for r in rows],
    }, schema=SCHEMA)
    shard_name = f"shard_{worker_id}_{uuid.uuid4().hex[:8]}.parquet"
    pq.write_table(table, out_dir / shard_name)


def log_error(pdf_id: str, error: Exception):
    msg = f"DOWNLOAD FAILED: {pdf_id} — {error}"
    print(msg)
    logger.error(msg)
    with open(ERROR_LOG, "a") as f:
        f.write(msg + "\n")


def download_pdf(pdf_id: str, dest: Path) -> bool:
    s3_key = f"{PDF_BUCKET}{pdf_id}.pdf"
    try:
        subprocess.run(
            ["s5cmd", "cp", s3_key, str(dest)],
            capture_output=True, text=True, check=True,
        )
        return True
    except Exception as e:
        log_error(pdf_id, e)
        return False


def process_jsonl_sample_worker(
    jsonl_filenames: list[str], sampled_ids: set[str], worker_id: str
) -> set[str]:
    """Same as process_jsonl_worker, but only processes documents whose PDF
    is in sampled_ids. Returns the subset of sampled IDs that had OCR."""

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    matched_ids: set[str] = set()
    ocr_rows: list[tuple[str, int, str | None]] = []
    metadata_rows: list[tuple[str, int, str | None]] = []
    pdfs_in_batch = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for jsonl_name in jsonl_filenames:
            jsonl_s3 = f"{OCR_BUCKET}{jsonl_name}"
            local_jsonl = Path(tmpdir) / jsonl_name
            subprocess.run(
                ["s5cmd", "cp", jsonl_s3, str(local_jsonl)],
                capture_output=True, text=True, check=True,
            )

            with open(local_jsonl, encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    source_file = doc["metadata"]["Source-File"]
                    pdf_id = extract_pdf_id(source_file)
                    if pdf_id not in sampled_ids:
                        continue  # not in sample, skip entirely

                    matched_ids.add(pdf_id)
                    text = doc["text"]
                    page_numbers = doc["attributes"]["pdf_page_numbers"]
                    for entry in page_numbers:
                        start, end, page_no = entry[0], entry[1], entry[2]
                        ocr_rows.append((pdf_id, page_no, text[start:end]))

                    pdf_path = Path(tmpdir) / f"{pdf_id}.pdf"
                    if download_pdf(pdf_id, pdf_path):
                        for page_no, txt in extract_metadata_pages(pdf_path):
                            metadata_rows.append((pdf_id, page_no, txt))
                        pdf_path.unlink()

                    pdfs_in_batch += 1
                    if pdfs_in_batch >= BATCH_SIZE:
                        flush_parquet(ocr_rows, OCR_DIR, worker_id)
                        flush_parquet(metadata_rows, METADATA_DIR, worker_id)
                        ocr_rows.clear()
                        metadata_rows.clear()
                        pdfs_in_batch = 0

            local_jsonl.unlink(missing_ok=True)

    flush_parquet(ocr_rows, OCR_DIR, worker_id)
    flush_parquet(metadata_rows, METADATA_DIR, worker_id)
    return matched_ids

def process_jsonl_worker(jsonl_filenames: list[str], worker_id: str) -> set[str]:
    """Process all documents in the assigned JSONL files. Returns referenced pdf_ids."""

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    referenced_pdf_ids: set[str] = set()
    ocr_rows: list[tuple[str, int, str | None]] = []
    metadata_rows: list[tuple[str, int, str | None]] = []
    pdfs_in_batch = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for jsonl_name in jsonl_filenames:
            jsonl_s3 = f"{OCR_BUCKET}{jsonl_name}"
            local_jsonl = Path(tmpdir) / jsonl_name
            subprocess.run(
                ["s5cmd", "cp", jsonl_s3, str(local_jsonl)],
                capture_output=True, text=True, check=True,
            )

            with open(local_jsonl, encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc["text"]
                    page_numbers = doc["attributes"]["pdf_page_numbers"]
                    source_file = doc["metadata"]["Source-File"]
                    pdf_id = extract_pdf_id(source_file)

                    referenced_pdf_ids.add(pdf_id)

                    for entry in page_numbers:
                        start, end, page_no = entry[0], entry[1], entry[2]
                        ocr_rows.append((pdf_id, page_no, text[start:end]))

                    pdf_path = Path(tmpdir) / f"{pdf_id}.pdf"
                    if download_pdf(pdf_id, pdf_path):
                        for page_no, txt in extract_metadata_pages(pdf_path):
                            metadata_rows.append((pdf_id, page_no, txt))
                        pdf_path.unlink()

                    pdfs_in_batch += 1
                    if pdfs_in_batch >= BATCH_SIZE:
                        flush_parquet(ocr_rows, OCR_DIR, worker_id)
                        flush_parquet(metadata_rows, METADATA_DIR, worker_id)
                        ocr_rows.clear()
                        metadata_rows.clear()
                        pdfs_in_batch = 0

            local_jsonl.unlink(missing_ok=True)

    flush_parquet(ocr_rows, OCR_DIR, worker_id)
    flush_parquet(metadata_rows, METADATA_DIR, worker_id)
    return referenced_pdf_ids


def process_leftover_worker(pdf_ids: list[str], worker_id: str):
    """Download PDFs with no OCR counterpart and extract metadata triples."""

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_rows: list[tuple[str, int, str | None]] = []
    pdfs_in_batch = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for pdf_id in pdf_ids:
            pdf_path = Path(tmpdir) / f"{pdf_id}.pdf"
            if download_pdf(pdf_id, pdf_path):
                pages = extract_metadata_pages(pdf_path)
                for page_no, txt in pages:
                    metadata_rows.append((pdf_id, page_no, txt))
                pdf_path.unlink()

            pdfs_in_batch += 1
            if pdfs_in_batch >= BATCH_SIZE:
                flush_parquet(metadata_rows, METADATA_DIR, worker_id)
                metadata_rows.clear()
                pdfs_in_batch = 0

    flush_parquet(metadata_rows, METADATA_DIR, worker_id)


def distribute(items: list, n_workers: int) -> list[list]:
    chunks = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    return [c for c in chunks if c]

def run_sample(n: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Listing PDF bucket...")
    all_pdf_ids = list_pdf_ids()
    logger.info(f"Found {len(all_pdf_ids)} PDFs in bucket")

    sampled = sample_pdf_ids(all_pdf_ids, n, seed)
    sampled_set = set(sampled)
    logger.info(f"Sampled {len(sampled)} PDF IDs (seed={seed}) → {SAMPLE_IDS_CSV}")

    logger.info("Listing OCR JSONL files...")
    jsonl_files = sorted(list_jsonl_files())

    n_workers = os.cpu_count() or 4
    chunks = distribute(jsonl_files, n_workers)

    logger.info(f"Scanning {len(jsonl_files)} JSONL files for sampled PDFs...")
    matched_ids: set[str] = set()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_jsonl_sample_worker, chunk, sampled_set, f"s{i}")
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            matched_ids.update(future.result())

    # Sampled PDFs with no OCR: extract metadata so the no-OCR buckets are covered.
    no_ocr_ids = list(sampled_set - matched_ids)
    logger.info(f"{len(matched_ids)} sampled PDFs had OCR; {len(no_ocr_ids)} did not")

    if no_ocr_ids:
        leftover_chunks = distribute(no_ocr_ids, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_leftover_worker, chunk, f"snoocr{i}")
                for i, chunk in enumerate(leftover_chunks)
            ]
            for future in futures:
                future.result()

    logger.info("Sample run done.")

def run_full():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Listing PDF bucket...")
    all_pdf_ids = list_pdf_ids()
    logger.info(f"Found {len(all_pdf_ids)} PDFs in bucket")

    logger.info("Listing OCR JSONL files...")
    jsonl_files = sorted(list_jsonl_files())
    logger.info(f"Found {len(jsonl_files)} JSONL files")

    n_workers = os.cpu_count() or 4
    chunks = distribute(jsonl_files, n_workers)

    logger.info(f"Processing {len(jsonl_files)} JSONL files across {len(chunks)} workers...")
    referenced_pdf_ids: set[str] = set()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(process_jsonl_worker, chunk, f"w{i}")
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            referenced_pdf_ids.update(future.result())

    leftover_ids = list(all_pdf_ids - referenced_pdf_ids)
    logger.info(f"{len(leftover_ids)} PDFs have no OCR counterpart, processing metadata...")

    if leftover_ids:
        leftover_chunks = distribute(leftover_ids, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_leftover_worker, chunk, f"left{i}")
                for i, chunk in enumerate(leftover_chunks)
            ]
            for future in futures:
                future.result()

    logger.info("Done.")

def main():
    parser = argparse.ArgumentParser(description="Parse OCR and metadata triples.")
    parser.add_argument(
        "mode",
        choices=["full", "sample"],
        help="full: process all documents. sample: process a random PDF-ID sample.",
    )
    parser.add_argument(
        "--n", type=int, default=SAMPLE_SIZE,
        help=f"sample size (default {SAMPLE_SIZE}, only used in sample mode)",
    )
    parser.add_argument(
        "--seed", type=int, default=SAMPLE_SEED,
        help=f"sample seed (default {SAMPLE_SEED}, only used in sample mode)",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        run_sample(n=args.n, seed=args.seed)
    else:
        run_full()

if __name__ == "__main__":
    main()