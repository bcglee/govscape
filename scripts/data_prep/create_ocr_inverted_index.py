# AI modified: 2026-06-23 00:00:00 942a1884
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from govscape.config import DataModel
from govscape.data_loader import DataLoader, build_data_loader
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

_worker_data_loader: DataLoader | None = None
_worker_remote_ocr_dir: str | None = None

def _init_worker(
    backend: str,
    bucket_name: str,
    local_base_dir: str | None,
    remote_ocr_dir: str,
) -> None:
    global _worker_data_loader, _worker_remote_ocr_dir
    _worker_data_loader = build_data_loader(
        backend,
        bucket_name,
        local_base_dir=local_base_dir,
    )
    _worker_remote_ocr_dir = remote_ocr_dir

def list_all_keys(data_loader: DataLoader) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        result = data_loader.list_objects(
            OCR_PREFIX,
            max_keys=100000,
            continuation_token=token,
        )
        keys.extend(result.keys)
        if not result.is_truncated:
            break
        token = result.continuation_token
    return keys

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

def distribute(items: list, n_workers: int) -> list[list]:
    chunks: list[list] = [[] for _ in range(n_workers)]
    for i, item in enumerate(items):
        chunks[i % n_workers].append(item)
    return [c for c in chunks if c]

def load_manifest(
    data_loader: DataLoader,
    remote_manifest_path: str,
) -> set[str]:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if data_loader.exists(remote_manifest_path):
        data_loader.download_file(remote_manifest_path, str(MANIFEST_PATH))
    if MANIFEST_PATH.exists():
        return {line for line in MANIFEST_PATH.read_text().splitlines() if line.strip()}
    return set()

def save_manifest(
    data_loader: DataLoader,
    remote_manifest_path: str,
    completed: set[str],
):
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(sorted(completed)) + "\n")
    data_loader.upload_file(str(MANIFEST_PATH), remote_manifest_path)

def extract_ocr_worker(jsonl_filenames: list[str], worker_id: str):
    # TODO: Add some error handling for the workers / transition out of global structure
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "ocr"

        for jsonl_name in jsonl_filenames:
            local_jsonl = Path(tmpdir) / jsonl_name
            try:
                _worker_data_loader.download_file(
                    f"{OCR_PREFIX}{jsonl_name}",
                    str(local_jsonl),
                )
            except Exception as e:
                log_error(f"JSONL download {jsonl_name}", e)
                continue

            staging_dir.mkdir(exist_ok=True)
            pages_staged = 0

            with open(local_jsonl, encoding="utf-8") as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                    except Exception as e:
                        log_error(f"JSONL parse in {jsonl_name}", e)
                        continue

                    text = doc["text"]
                    digest = extract_pdf_id(doc["metadata"]["Source-File"])
                    digest_dir = staging_dir / digest
                    digest_dir.mkdir(parents=True, exist_ok=True)

                    for start, end, page_no in doc["attributes"]["pdf_page_numbers"]:
                        page_path = digest_dir / f"{digest}_{page_no}.txt"
                        page_path.write_text(text[start:end], encoding="utf-8")
                        pages_staged += 1

            local_jsonl.unlink(missing_ok=True)

            if pages_staged > 0:
                _worker_data_loader.upload_directory(
                    str(staging_dir),
                    _worker_remote_ocr_dir,
                )
                shutil.rmtree(staging_dir)

            logging.info(
                "[%s] %s: uploaded %d pages",
                worker_id,
                jsonl_name,
                pages_staged,
            )

def run(
    data_loader: DataLoader,
    backend: str,
    bucket_name: str,
    local_base_dir: str | None,
    remote_dm: DataModel,
    batch_size: int,
):
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    remote_manifest = os.path.join(
        remote_dm.checkpoints_directory,
        "ocr_extraction.manifest",
    )
    completed = load_manifest(data_loader, remote_manifest)

    all_jsonls = sorted(list_all_keys(data_loader))
    remaining = [j for j in all_jsonls if j not in completed]
    logging.info(
        "%d JSONL files total, %d already done, %d remaining",
        len(all_jsonls),
        len(completed),
        len(remaining),
    )

    if not remaining:
        logging.info("Nothing to do.")
        return

    n_workers = os.cpu_count() or 4

    for chunk_start in range(0, len(remaining), batch_size):
        chunk = remaining[chunk_start : chunk_start + batch_size]
        worker_batches = distribute(chunk, n_workers)

        logging.info(
            "Processing chunk %d-%d (%d JSONL files, %d workers)...",
            chunk_start,
            chunk_start + len(chunk),
            len(chunk),
            len(worker_batches),
        )

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(backend, bucket_name, local_base_dir, remote_dm.ocr_directory),
        ) as executor:
            futures = [
                executor.submit(extract_ocr_worker, batch, f"w{i}")
                for i, batch in enumerate(worker_batches)
            ]
            for fut in futures:
                fut.result()

        completed.update(chunk)
        save_manifest(data_loader, remote_manifest, completed)
        logging.info(
            "Checkpoint saved: %d / %d JSONL files complete",
            len(completed),
            len(all_jsonls),
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
        args.bucket_name,
        local_base_dir=args.local_base_dir,
    )
    remote_dm = DataModel(args.remote_data_dir)

    run(
        data_loader,
        args.backend,
        args.bucket_name,
        args.local_base_dir,
        remote_dm,
        args.batch_size,
    )

if __name__ == "__main__":
    main()