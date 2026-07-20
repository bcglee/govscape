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

NATIVE_BUCKET = "us-west-2.opendata.source.coop"
NATIVE_REGION = "us-west-2"
NATIVE_PREFIX = "govscape/"          # product paths sit under this in the native bucket
OWNER_ACL = "bucket-owner-full-control"   # REQUIRED on every native write, per their docs

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
    def __init__(self, creds_path: str | None = None, attempts: int = 4,
                 native: bool = False) -> None:
        self.native = native
        self.attempts = attempts
        self.creds_path = creds_path
        if native:
            self.client = boto3.client("s3", region_name=NATIVE_REGION)  # instance role
            self.bucket, self.prefix = NATIVE_BUCKET, NATIVE_PREFIX
        else:
            self.client = make_coop_authed_client(creds_path)
            self.bucket, self.prefix = COOP_BUCKET, ""

    def _reload(self, reason: str) -> None:
        if self.native:
            raise RuntimeError(f"auth error in native mode ({reason}) — "
                               "this is a real permissions problem, not stale creds")
        logging.warning("auth error (%s) — reloading creds from %s", reason, self.creds_path)
        self.client = make_coop_authed_client(self.creds_path)

    def upload_file(self, local_path: str, key: str) -> None:
        extra = {"ACL": OWNER_ACL} if self.native else None
        self._with_retry(key, lambda: self.client.upload_file(
            local_path, self.bucket, self.prefix + key, ExtraArgs=extra))

    def put_bytes(self, data: bytes, key: str) -> None:
        kwargs = {"Bucket": self.bucket, "Key": self.prefix + key, "Body": data}
        if self.native:
            kwargs["ACL"] = OWNER_ACL
        self._with_retry(key, lambda: self.client.put_object(**kwargs))

    def copy_object(self, src_bucket: str, src_key: str, dest_key: str) -> None:
        """Server-side copy — zero bytes through the instance. Native mode only."""
        self._with_retry(dest_key, lambda: self.client.copy_object(
            Bucket=self.bucket, Key=self.prefix + dest_key,
            CopySource={"Bucket": src_bucket, "Key": src_key}, ACL=OWNER_ACL))
            
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
                    if auth_reloads % 5 == 0:
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
            except Exception as e:  # S3UploadFailedError wraps ClientError;
                # connection resets arrive as generic exceptions
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
                                    op_desc, msg[:300], attempt)
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                raise

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