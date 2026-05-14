import logging
import multiprocessing
import os
import tempfile
from collections import defaultdict

import pandas as pd
import pypdfium2
import requests
from govscape.data_loader import DataLoader, build_data_loader
from govscape.utils import base_argument_parser
from warcio.archiveiterator import ArchiveIterator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

_USER_AGENT = "govscape/0.1 (PDF Retrieval Script; kdeeds@cs.washington.edu)"

_worker_data_loader: DataLoader | None = None


def _init_worker(backend: str, bucket_name: str, local_base_dir: str) -> None:
    global _worker_data_loader
    _worker_data_loader = build_data_loader(backend, bucket_name, local_base_dir)


def _is_parseable_pdf(data: bytes) -> bool:
    try:
        pypdfium2.PdfDocument(data)
        return True
    except Exception:
        return False


def _process_one_pdf(args: tuple) -> str:
    digest, filename, offset, length, output_directory = args
    assert _worker_data_loader is not None

    output_path = os.path.join(output_directory, digest + ".pdf")

    try:
        if _worker_data_loader.exists(output_path):
            return "skipped"
    except Exception:
        pass

    s3_url = f"https://eotarchive.s3.amazonaws.com/{filename}"
    byte_range = f"bytes={offset}-{offset + length - 1}"

    try:
        response = requests.get(
            s3_url,
            headers={"user-agent": _USER_AGENT, "Range": byte_range},
            stream=True,
        )
        for record in ArchiveIterator(response.raw):
            if record.rec_type != "response":
                continue
            is_pdf = (
                record.rec_headers.get("Content-Type") == "application/pdf"
            ) or (".pdf" in record.rec_headers.get("WARC-Target-URI", ""))
            if not is_pdf:
                return "invalid_content"
            data = record.content_stream().read()
            if not _is_parseable_pdf(data):
                return "parse_failed"
            _worker_data_loader.upload_bytes(data, output_path)
            return "uploaded"
    except Exception as e:
        logging.warning("Error processing %s: %s", filename, e)
        return "error"

    return "invalid_content"


def main() -> None:
    parser = base_argument_parser(description="Retrieve PDFs from S3 & store them.")
    parser.add_argument("--bucket", required=True, help="Output S3 bucket name")
    parser.add_argument(
        "--cdx_parquet", required=True, help="Remote key of CDX parquet file in the output bucket"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Remote key prefix for output PDFs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count() * 4,
        help="Number of worker processes (default: cpu_count * 4)",
    )
    parser.add_argument(
        "--max_pdfs",
        type=int,
        default=None,
        help="Limit total PDFs processed (useful for testing)",
    )
    args = parser.parse_args()

    data_loader = build_data_loader(args.backend, args.bucket, args.local_base_dir)
    with tempfile.TemporaryDirectory(prefix="retrieve_pdfs_") as tmp_dir:
        local_parquet = os.path.join(tmp_dir, "cdx.parquet")
        logging.info("Downloading %s", args.cdx_parquet)
        data_loader.download_file(args.cdx_parquet, local_parquet)
        df = pd.read_parquet(local_parquet)
    logging.info("Loaded %d CDX entries from %s", len(df), args.cdx_parquet)

    worker_args = [
        (row.digest, row.filename, int(row.offset), int(row.length), args.output_dir)
        for row in df.itertuples(index=False)
    ]
    if args.max_pdfs is not None:
        worker_args = worker_args[: args.max_pdfs]

    num_workers = min(args.num_workers, len(worker_args))
    logging.info("Processing %d PDFs across %d workers", len(worker_args), num_workers)

    counts: dict[str, int] = defaultdict(int)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(args.backend, args.bucket, args.local_base_dir),
    ) as pool:
        for status in pool.imap_unordered(_process_one_pdf, worker_args):
            counts[status] += 1
            total = sum(counts.values())
            if total % 100 == 0:
                logging.info("Progress: %d processed — %s", total, dict(counts))

    logging.info("Done. Final counts: %s", dict(counts))


if __name__ == "__main__":
    main()
