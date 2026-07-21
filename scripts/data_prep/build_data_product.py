"""Build the eota-ocr data product on source.coop"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from ocr_common import (
    ARCHIVE_PREFIX, COOP_BUCKET, DIGEST_RE, NATIVE_BUCKET, NATIVE_PREFIX,
    PDF_KEY_TEMPLATE, PRODUCT_PREFIX,
    CoopWriter, JsonCheckpoint, download_with_retry, is_not_found,
    list_keys, load_ignore_list, make_coop_anon_client, make_error_logger,
    setup_logging, tar_pages,
)

AWS_BUCKET = "eot-pdf-archive"
BCGL_BUCKET = "bcgl-public-bucket"
DEV_SERVING_PREFIX = "dev-serving/txt/"

LOCAL_DIR = Path("data/data_product")
ERROR_LOG = str(LOCAL_DIR / "errors.log")
log_error = make_error_logger(ERROR_LOG)

# bulk copy ocr_text + ocr_metadata
def digest_from_key(key: str) -> str | None:
    """ocr_text/{digest}.tar.gz or ocr_metadata/{digest}/metadata.json -> digest."""
    parts = key.split("/")
    if len(parts) < 2:
        return None
    candidate = parts[1].removesuffix(".tar.gz")
    return candidate if DIGEST_RE.match(candidate) else None

def copy_one_key(writer, key):
    writer.copy_object(AWS_BUCKET, key, f"{PRODUCT_PREFIX}/{key}")
    return key

def phase_copy(writer: CoopWriter, exclude: set[str], threads: int) -> None:
    if not writer.native:
        raise SystemExit("phase copy requires --native (server-side CopyObject); "
                         "the proxy endpoint does not support it")
    aws_client = boto3.client("s3")
    for prefix in ("ocr_text/", "ocr_metadata/"):
        ckpt = JsonCheckpoint(str(LOCAL_DIR / f"copy_{prefix.rstrip('/')}.ckpt"))
        if ckpt.state.get("finished"):
            logging.info("phase copy %s already finished — skipping", prefix)
            continue
        token = ckpt.state.get("token")
        copied = ckpt.state.get("copied", 0)
        skipped = ckpt.state.get("skipped", 0)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            while True:
                keys, token = list_keys(aws_client, AWS_BUCKET, prefix,
                                        continuation=token, max_keys=1000)
                work, batch_skipped = [], 0
                for k in keys:
                    d = digest_from_key(k)
                    if d is None or d in exclude:
                        batch_skipped += 1
                        continue
                    work.append(k)
                futures = [pool.submit(copy_one_key, writer, k) for k in work]
                for fut in futures:
                    fut.result()
                    copied += 1
                skipped += batch_skipped
                ckpt.state = {"token": token, "copied": copied, "skipped": skipped,
                              "finished": token is None}
                ckpt.save()
                logging.info("copy %s: %d copied, %d skipped", prefix, copied, skipped)
                if token is None:
                    break

# filtered CDX (DuckDB semi-join over parquet)
def phase_cdx(writer: CoopWriter, include: set[str]) -> None:
    import duckdb
    cdx_local = str(LOCAL_DIR / "complete_cdx.parquet")
    filtered = str(LOCAL_DIR / "cdx_filtered.parquet")
    include_file = str(LOCAL_DIR / "include_digests.txt")
    Path(include_file).write_text("\n".join(sorted(include)))

    if not os.path.exists(cdx_local):
        logging.info("downloading complete_cdx.parquet")
        download_with_retry(make_coop_anon_client(), COOP_BUCKET,
                            f"{ARCHIVE_PREFIX}/cdx/complete_cdx.parquet", cdx_local)

    con = duckdb.connect()
    con.sql(f"""
        COPY (
            SELECT c.* FROM read_parquet('{cdx_local}') c
            SEMI JOIN read_csv('{include_file}',
                               columns={{'digest': 'VARCHAR'}}, header=false) i
            ON c.digest = i.digest
        ) TO '{filtered}' (FORMAT parquet, COMPRESSION zstd)
    """)
    rows, digests = con.sql(
        f"SELECT COUNT(*), COUNT(DISTINCT digest) FROM read_parquet('{filtered}')"
    ).fetchone()
    logging.info("cdx filtered: %d rows, %d distinct digests (include set: %d)",
                 rows, digests, len(include))
    if digests < len(include):
        logging.warning("%d include digests have NO CDX entry — investigate before publish",
                        len(include) - digests)
    writer.upload_file(filtered, f"{PRODUCT_PREFIX}/cdx/complete_cdx.parquet")


# extracted text + PDFs, per digest

_ANON = None
_WRITER = None
_BCGL = None
pdfium = None
_CREDS_PATH = None

# Use Data Loader and Remote Directory Iterator and if they don't work
# If we do not know why they work, then we can try to scale and adapt them
def _init_worker(creds_path: str, native: bool) -> None:
    global _ANON, _WRITER, _BCGL, pdfium
    from botocore.config import Config
    cfg = Config(max_pool_connections=60, retries={"max_attempts": 3, "mode": "standard"})
    _ANON = make_coop_anon_client()
    _WRITER = CoopWriter(creds_path, native=native)
    _BCGL = boto3.client("s3", config=cfg)
    import pypdfium2 as pdfium

def _extract_pdf_pages(pdf_path: str) -> dict[int, str]:
    """pypdfium2 text per page, returned 1-indexed."""
    pages: dict[int, str] = {}
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            pages[i + 1] = tp.get_text_bounded()
            tp.close(); page.close()
    finally:
        pdf.close()
    return pages

def _fetch_devserving_pages(digest, tmp_dir):
    prefix = f"{DEV_SERVING_PREFIX}{digest}/"
    all_keys = []
    keys, token = list_keys(_BCGL, BCGL_BUCKET, prefix)
    all_keys.extend(keys)
    while token:
        keys, token = list_keys(_BCGL, BCGL_BUCKET, prefix, continuation=token)
        all_keys.extend(keys)
    if not all_keys:
        return None

    def fetch_one(key):
        stem = Path(key).stem
        try:
            pg0 = int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return None
        obj = _BCGL.get_object(Bucket=BCGL_BUCKET, Key=key)
        return pg0 + 1, obj["Body"].read().decode("utf-8", errors="replace")

    pages = {}
    if len(all_keys) == 1:
        r = fetch_one(all_keys[0])
        if r: pages[r[0]] = r[1]
        return pages or None

    n = min(16, len(all_keys))
    with ThreadPoolExecutor(max_workers=n) as ex:
        for r in ex.map(fetch_one, all_keys):
            if r:
                pages[r[0]] = r[1]
    return pages or None

def process_digest(digest: str, work_dir: str,
                   skip_pdf_upload: bool) -> dict:
    result = {"digest": digest, "counts": collections.Counter(),
              "miss": False, "error": None}
    c = result["counts"]
    tmp_dir = os.path.join(work_dir, digest)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        pdf_key = PDF_KEY_TEMPLATE.format(digest=digest)
        pdf_local = os.path.join(tmp_dir, f"{digest}.pdf")

        if not skip_pdf_upload:
            try:
                _WRITER.copy_object(NATIVE_BUCKET, f"{NATIVE_PREFIX}{pdf_key}",
                                    f"{PRODUCT_PREFIX}/pdfs/{digest}.pdf")
                c["pdf_copied"] += 1
            except ClientError as e:
                if is_not_found(e):
                    result["miss"] = True
                    c["miss"] += 1
                    return result
                raise
        else:
            try:
                _WRITER.client.head_object(Bucket=NATIVE_BUCKET,
                                           Key=f"{NATIVE_PREFIX}{pdf_key}")
            except ClientError as e:
                if is_not_found(e):
                    result["miss"] = True
                    c["miss"] += 1
                    return result
                raise

        pages = _fetch_devserving_pages(digest, tmp_dir)
        if pages is not None:
            c["devserving_hit"] += 1
        else:
            if not os.path.exists(pdf_local):
                download_with_retry(_WRITER.client, NATIVE_BUCKET,
                                    f"{NATIVE_PREFIX}{pdf_key}", pdf_local)
            pages = _extract_pdf_pages(pdf_local)
            c["pypdfium_extracted"] += 1
        tar_path = tar_pages(pages, digest, tmp_dir)
        _WRITER.upload_file(
            tar_path, f"{PRODUCT_PREFIX}/extracted_text/{digest}.tar.gz")
        c["text_uploaded"] += 1
    except Exception as e:
        log_error(f"process failed {digest}", e)
        result["error"] = str(e)
        c["errors"] += 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result

def phase_text(creds_path, native, exclude, workers, batch_size, skip_pdf_upload):
    aws_client = boto3.client("s3")
    work_dir = str(LOCAL_DIR / "text_work")
    all_misses_path = LOCAL_DIR / "all_misses.txt"
    ckpt = JsonCheckpoint(str(LOCAL_DIR / "text.ckpt"))
    if ckpt.state.get("finished"):
        logging.info("phase text already finished"); return
    token = ckpt.state.get("token")
    totals = collections.Counter(ckpt.state.get("totals", {}))

    def _make_pool():
        return ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                   initargs=(creds_path, native),
                                   max_tasks_per_child=500)
    pool = _make_pool()
    try:
        while True:
            keys, token = list_keys(aws_client, AWS_BUCKET, "ocr_text/",
                                    continuation=token, max_keys=batch_size)
            digests = [d for d in (digest_from_key(k) for k in keys)
                       if d and d not in exclude]
            try:
                futures = {pool.submit(process_digest, d, work_dir,
                                       skip_pdf_upload): d for d in digests}
                batch_misses = []
                for fut in futures:
                    try:
                        r = fut.result()
                        totals.update(r["counts"])
                        if r["miss"]:
                            batch_misses.append(r["digest"])
                    except Exception as e:
                        log_error(f"worker error {futures[fut]}", e)
                        totals["errors"] += 1
            except BrokenProcessPool:
                logging.error("pool broke — rebuilding, re-running this batch")
                pool.shutdown(wait=False)
                pool = _make_pool()
                continue
            if batch_misses:
                with open(all_misses_path, "a") as f:
                    f.write("\n".join(batch_misses) + "\n")
            ckpt.state = {"token": token, "totals": dict(totals),
                          "finished": token is None}
            ckpt.save()
            logging.info("text phase: %s", dict(totals))
            if token is None:
                break
    finally:
        pool.shutdown(wait=False)
    logging.info("phase text done: %s", dict(totals))

def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["text", "copy", "cdx", "all"], default="all")
    p.add_argument("--native", action="store_true",
                   help="write via native AWS bucket (Option 3 grant) — no creds file")
    p.add_argument("--creds", default="source_coop_creds.json",
                   help="source.coop credentials JSON (refresh by overwriting)")
    p.add_argument("--fallback-ignore", default="fallback_ignore.txt")
    p.add_argument("--miss-list", default="misses.txt",
                   help="pre-known misses (profiler); silences their logging only")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--copy-threads", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--skip-pdf-upload", action="store_true",
                   help="omit pdfs/ (e.g. if source.coop copies them backend-side)")
    args = p.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    exclude = load_ignore_list(args.fallback_ignore)
    writer = CoopWriter(args.creds, native=args.native)
    all_misses_path = str(LOCAL_DIR / "all_misses.txt")

    # Text extraction phase
    if args.phase in ("text", "all"):
        phase_text(args.creds, args.native, exclude,
                   args.workers, args.batch_size, args.skip_pdf_upload)
                   
    all_misses = load_ignore_list(all_misses_path)
    if args.phase in ("copy", "cdx") and not JsonCheckpoint(
            str(LOCAL_DIR / "text.ckpt")).state.get("finished"):
        logging.warning("phase text has NOT finished — all_misses.txt is incomplete; "
                        "copy/cdx run now would violate the consistency invariant")

    if args.phase in ("copy", "all"):
        phase_copy(writer, exclude | all_misses, args.copy_threads)

    if args.phase in ("cdx", "all"):
        logging.info("building include set from ocr_text listing...")
        aws_client = boto3.client("s3")
        include: set[str] = set()
        token = None
        excluded_all = exclude | all_misses
        while True:
            keys, token = list_keys(aws_client, AWS_BUCKET, "ocr_text/",
                                    continuation=token, max_keys=1000)
            include.update(d for d in (digest_from_key(k) for k in keys)
                           if d and d not in excluded_all)
            if token is None:
                break
        logging.info("include set: %d digests", len(include))
        phase_cdx(writer, include)

if __name__ == "__main__":
    main()