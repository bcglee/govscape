"""Shared utilities for the GovScape data-product build and OCR agreement analysis."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tarfile
import time
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# ---- Source.coop constants ----
COOP_ENDPOINT = "https://data.source.coop"
COOP_BUCKET = "govscape"
ARCHIVE_PREFIX = "eota-pdf-archive"
PRODUCT_PREFIX = "eota-ocr"
PDF_KEY_TEMPLATE = f"{ARCHIVE_PREFIX}/pdfs/{{digest}}.pdf"

DIGEST_RE = re.compile(r"^[A-Z2-7]{32}$")    # base32 SHA-1

TRANSIENT_CODES = {
    "500", "502", "503", "SlowDown",
    "520", "521", "522", "523", "524", "525", "526",
}
AUTH_ERROR_CODES = {
    "ExpiredToken", "InvalidToken", "InvalidAccessKeyId",
    "SignatureDoesNotMatch", "AccessDenied", "403",
}
class AuthExhausted(Exception): pass

def make_coop_anon_client():
    """Anonymous read client for public source.coop products."""
    return boto3.client(
        "s3", endpoint_url=COOP_ENDPOINT,
        config=Config(signature_version=UNSIGNED),
    )

def make_coop_authed_client(creds_path: str):
    """Authed client from the source.coop credentials JSON (expiring STS-style)."""
    with open(creds_path) as f:
        c = json.load(f)
    return boto3.client(
        "s3",
        aws_access_key_id=c["aws_access_key_id"],
        aws_secret_access_key=c["aws_secret_access_key"],
        aws_session_token=c["aws_session_token"],
        region_name=c.get("region_name", "us-east-1"),
        endpoint_url=c.get("endpoint_url", COOP_ENDPOINT),
    )

class CoopWriter:
    """Authed writer with two self-healing behaviors"""

    def __init__(self, creds_path: str, attempts: int = 4) -> None:
        self.creds_path = creds_path
        self.attempts = attempts
        self.client = make_coop_authed_client(creds_path)

    def _reload(self, reason: str) -> None:
        logging.warning("auth error (%s) — reloading creds from %s",
                        reason, self.creds_path)
        self.client = make_coop_authed_client(self.creds_path)

    def _with_retry(self, op_desc: str, fn) -> None:
        """Every exit is success-or-raise — no silent fall-through."""
        attempt = 0
        auth_reloads = 0
        while True:
            try:
                fn()
                return
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code", ""))
                if code in AUTH_ERROR_CODES:
                    auth_reloads += 1
                    self._reload(code)
                    if auth_reloads % 5 == 0:      # once per ~5 min of stall
                        logging.warning("PAUSED on stale creds (%s) — %d reloads; "
                                        "overwrite %s to resume", op_desc,
                                        auth_reloads, self.creds_path)
                    time.sleep(min(60, 5 * auth_reloads))
                    continue
                if code in TRANSIENT_CODES and attempt < self.attempts - 1:
                    attempt += 1
                    logging.warning("transient %s on %s, retry %d",
                                    code, op_desc, attempt)
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                raise
            except Exception as e:
                msg = str(e)
                if any(c in msg for c in AUTH_ERROR_CODES):
                    auth_reloads += 1
                    self._reload("wrapped auth error")
                    if auth_reloads % 5 == 0:
                        logging.warning("PAUSED on stale creds (%s) — %d reloads; "
                                        "overwrite %s to resume", op_desc,
                                        auth_reloads, self.creds_path)
                    time.sleep(min(60, 5 * auth_reloads))
                    continue
                if attempt < self.attempts - 1:
                    attempt += 1
                    logging.warning("upload error on %s (%s), retry %d",
                                    op_desc, msg[:120], attempt)
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                raise

    def upload_file(self, local_path: str, key: str) -> None:
        self._with_retry(key, lambda: self.client.upload_file(
            local_path, COOP_BUCKET, key))

    def put_bytes(self, data: bytes, key: str) -> None:
        self._with_retry(key, lambda: self.client.put_object(
            Bucket=COOP_BUCKET, Key=key, Body=data))

def download_with_retry(client, bucket: str, key: str, local_path: str,
                        attempts: int = 4) -> None:
    """Download with exponential backoff + jitter on transient (CDN) errors.
    404 and other non-transient errors raise immediately (fail fast)."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    for attempt in range(attempts):
        try:
            client.download_file(bucket, key, local_path)
            return
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in TRANSIENT_CODES and attempt < attempts - 1:
                logging.warning("transient %s on %s, retry %d", code, key, attempt + 1)
                time.sleep((2 ** attempt) + random.random())
                continue
            raise

def is_not_found(error: ClientError) -> bool:
    return str(error.response.get("Error", {}).get("Code", "")) in ("404", "NoSuchKey")

def list_keys(client, bucket: str, prefix: str,
              continuation: str | None = None, max_keys: int = 1000):
    """One page of keys. Returns (keys, next_continuation_or_None)."""
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys}
    if continuation:
        kwargs["ContinuationToken"] = continuation
    resp = client.list_objects_v2(**kwargs)
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    return keys, resp.get("NextContinuationToken")

def tar_pages(page_texts: dict[int, str], digest: str, out_dir: str) -> str:
    """Write pages (1-indexed page_no -> text) as {digest}.tar.gz containing
    {digest}_{page_no}.txt. Returns the tar path."""
    os.makedirs(out_dir, exist_ok=True)
    staging = Path(out_dir) / f"_{digest}_pages"
    staging.mkdir(exist_ok=True)
    tar_path = os.path.join(out_dir, f"{digest}.tar.gz")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for pg in sorted(page_texts):
                fname = f"{digest}_{pg}.txt"
                fpath = staging / fname
                fpath.write_text(page_texts[pg], encoding="utf-8")
                tar.add(fpath, arcname=fname)
    finally:
        for f in staging.glob("*"):
            f.unlink()
        staging.rmdir()
    return tar_path

def untar_pages(tar_path: str, digest: str) -> dict[int, str]:
    """Read {digest}.tar.gz -> {page_no: text} (page numbers as stored, 1-indexed)."""
    pages: dict[int, str] = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            stem = Path(member.name).stem
            try:
                pg = int(stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            f = tar.extractfile(member)
            if f:
                pages[pg] = f.read().decode("utf-8", errors="replace")
    return pages

def load_ignore_list(path: str) -> set[str]:
    if not os.path.exists(path):
        logging.warning("Ignore list %s not found — proceeding without it.", path)
        return set()
    ignored = {line.strip() for line in open(path) if line.strip()}
    logging.info("Loaded %d digests from %s", len(ignored), path)
    return ignored

class JsonCheckpoint:
    """Tiny local checkpoint: a JSON dict persisted after every batch."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.state: dict = {}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state))

def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

def make_error_logger(error_log_path: str):
    def _log_error(label: str, error) -> None:
        msg = f"{label} — {error}"
        logging.error("!!! %s", msg)
        Path(error_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "a") as f:
            f.write(f"ERROR: {msg}\n")
    return _log_error