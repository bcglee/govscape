from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from govscape.config import DataModel
from govscape.data_loader import DataLoader, RemoteDirectoryIterator, build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_BUCKET = "eot-pdf-archive"
OCR_PREFIX = "ai2-olmocr/"

LOCAL_DATA_DIR = Path("data/ocr_staging")
MANIFEST_PATH = LOCAL_DATA_DIR / "completed.manifest"
ERROR_LOG = LOCAL_DATA_DIR / "errors.log"

def extract_pdf_id(source_file: str) -> str:
    name = source_file.rsplit("/", 1)[-1]
    if name.endswith(".pdf"):
        return name[:-4]
    return name

def log_error(label: str, error: Exception):
    msg = f"{label} — {error}"
    logging.error(msg)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(f"ERROR: {msg}\n")

def parse_jsonl(jsonl_path: str, ocr_text_dir: str, ocr_metadata_dir: str, worker_id: str) -> int:
    """Split one local OLMOCR JSONL into per-page .txt files under out_dir."""
    pages_staged = 0
    fallback_docs = 0
    fallback_pages = 0
    docs_seen = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except Exception as e:
                log_error(f"JSONL parse in {jsonl_path}", e)
                continue

            docs_seen += 1
            text = doc["text"]
            digest = extract_pdf_id(doc["metadata"]["Source-File"])
            digest_text_dir = Path(ocr_text_dir) / digest
            digest_text_dir.mkdir(parents=True, exist_ok=True)

            digest_metadata_dir = Path(ocr_metadata_dir) / digest
            digest_metadata_dir.mkdir(parents=True, exist_ok=True)

            for start, end, page_no in doc["attributes"]["pdf_page_numbers"]:
                (digest_text_dir / f"{digest}_{page_no}.txt").write_text(
                    text[start:end], encoding="utf-8"
                )
                pages_staged += 1
            
            (digest_metadata_dir / "metadata.json").write_text(json.dumps(doc), encoding="utf-8")

            doc_fallback = doc["metadata"].get("total-fallback-pages", 0)
            if doc_fallback > 0:
                fallback_docs += 1
                fallback_pages += doc_fallback


    logging.info(
        "[%s] %s: %d pages, %d/%d docs needed fallback",
        worker_id, Path(jsonl_path).name, pages_staged, fallback_docs, docs_seen,
    )
    return {
        "pages": pages_staged,
        "docs": docs_seen,
        "fallback_docs": fallback_docs,
        "fallback_pages": fallback_pages,
    }

def run(
    data_loader: DataLoader,
    remote_dm: DataModel,
    batch_size: int,
):
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = str(LOCAL_DATA_DIR / "jsonl_in")
    ocr_text_dir = str(LOCAL_DATA_DIR / "pages_text_out")
    ocr_metadata_dir = str(LOCAL_DATA_DIR / "pages_metadata_out")

    remote_checkpoint = os.path.join(
        remote_dm.checkpoints_directory, "ocr_extraction.checkpoint"
    )

    totals = {"pages": 0, "docs": 0, "fallback_docs": 0, "fallback_pages": 0}

    with RemoteDirectoryIterator(
        data_loader=data_loader,
        prefix=OCR_PREFIX,
        remote_checkpoint_path=remote_checkpoint,
        local_checkpoint_path=str(LOCAL_DATA_DIR / "ocr_extraction.checkpoint"),
        local_dir=in_dir,
        use_multiprocessing=False,
    ) as it, ProcessPoolExecutor() as pool:
        batch_num = 0
        while not it.finished:
            jsonl_paths = it.download_batch(max_keys=batch_size)

            if jsonl_paths:
                futures = {
                    pool.submit(parse_jsonl, p, ocr_text_dir, ocr_metadata_dir, f"w{i}"): p
                    for i, p in enumerate(jsonl_paths)
                }
                for fut in futures:
                    try:
                        counts = fut.result()
                        for k in totals:
                            totals[k] += counts[k]
                    except Exception as e:
                        log_error(f"parse failed {futures[fut]}", e)
                data_loader.upload_directory(local_dir=ocr_text_dir, remote_prefix=remote_dm.ocr_text_directory, compress=True)
                data_loader.upload_directory(local_dir=ocr_metadata_dir, remote_prefix=remote_dm.ocr_metadata_directory, compress=True)

            it.save_checkpoint()
            shutil.rmtree(in_dir, ignore_errors=True)
            shutil.rmtree(ocr_text_dir, ignore_errors=True)
            shutil.rmtree(ocr_metadata_dir, ignore_errors=True)
            batch_num += 1
            logging.info(
                "Batch %d complete (finished=%s) — running totals: "
                "%d docs, %d needed fallback (%d fallback pages)",
                batch_num, it.finished,
                totals["docs"], totals["fallback_docs"], totals["fallback_pages"],
            )
    
    pct = (100 * totals["fallback_docs"] / totals["docs"]) if totals["docs"] else 0.0
    logging.info(
        "Done. %d docs total, %d needed OCR fallback (%.1f%%), %d fallback pages.",
        totals["docs"], totals["fallback_docs"], pct, totals["fallback_pages"],
    )
    logging.info("Done.")

def main():
    parser = base_argument_parser(
        description="Extract per-page OCR text from OLMOCR JSONL files to S3.",
    )
    parser.set_defaults(bucket_name=DEFAULT_BUCKET)
    args = parser.parse_args()

    data_loader = build_data_loader(
        args.backend,
        args.bucket_name
    )
    remote_dm = DataModel(args.remote_data_dir)

    run(data_loader, remote_dm, args.batch_size)

if __name__ == "__main__":
    main()