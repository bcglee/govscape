from __future__ import annotations

import argparse
import io
import logging
import os
import re
import tarfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from botocore.config import Config
from botocore.exceptions import ClientError
from rapidfuzz.distance import LCSseq, Levenshtein

from govscape.data_loader import S3DataLoader
from ocr_common import (
    DIGEST_RE, NATIVE_BUCKET, NATIVE_PREFIX, NATIVE_REGION, PRODUCT_PREFIX,
    JsonCheckpoint, is_not_found, load_ignore_list, make_error_logger,
    setup_logging,
)
import json

# Target
PRODUCT_KEY_PREFIX = f"{NATIVE_PREFIX}{PRODUCT_PREFIX}"   # govscape/eota-ocr
PARQUET_BUCKET = "eot-pdf-archive"
PARQUET_REGION = "us-east-2"
PARQUET_REMOTE_PREFIX = "performance/ocr_metrics"
MAX_POOL_CONNECTIONS = 60
LOCAL_DIR = Path("data/ocr_agreement")
ERROR_LOG = str(LOCAL_DIR / "errors.log")
log_error = make_error_logger(ERROR_LOG)

TOKENIZER_NAME = "o200k_base"

# Normalization
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]") 
_TAG_RE = re.compile(r"<[^>\n]{1,120}>")
_MD_RE = re.compile(r"(^#{1,6}\s+)|(\*{1,3})|(`{1,3})|(^\s*[-*+]\s+)", re.M)

def normalize_l1(text: str) -> str:
    """NFKC + control-char strip + whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def normalize_l2(text: str) -> str:
    """L1 + HTML/markdown markup strip + lowercase."""
    text = _TAG_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    return normalize_l1(text).lower()

# Grabage Ratio of control characters
# NOTE: There is a vectorized way of doing this if throughput is poor
def garbage_ratio(text: str) -> float:
    """Fraction of non-whitespace chars that are undecodable (raw text input)."""
    if not text:
        return 0.0
    bad = denom = 0
    for ch in text:
        if ch == "\ufffd":
            bad += 1; denom += 1
            continue
        if ch in "\n\t\r\u200c\u200d" or ch == " ":
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("C"):
            bad += 1
        elif cat == "Zs" or ch.isspace():
            continue
        elif not (ch.isalnum() or cat.startswith(("P", "S", "M"))):
            bad += 1
        denom += 1
    return bad / denom if denom else 0.0

# Metrics
_ENCODER = None
def _encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)
    return _ENCODER

def _pair_metrics(a: str, b: str) -> dict:
    words_a, words_b = a.split(), b.split()
    la, lb = len(a), len(b)
    out = {
        "len_ratio_char": (la / lb) if lb else float("inf") if la else 1.0,
        "norm_edit_distance_char": Levenshtein.normalized_distance(a, b),
        "norm_edit_distance_word": Levenshtein.normalized_distance(words_a, words_b),
    }
    lcs = LCSseq.similarity(words_a, words_b)
    p = lcs / len(words_a) if words_a else 0.0
    r = lcs / len(words_b) if words_b else 0.0
    out["rouge_l"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    sa, sb = set(words_a), set(words_b)
    inter, union = len(sa & sb), len(sa | sb)
    out["jaccard"] = inter / union if union else 1.0
    out["containment_ocr"] = inter / len(sa) if sa else 0.0
    out["containment_pdf"] = inter / len(sb) if sb else 0.0
    return out

def compute_page_metrics(ocr_text: str, pdf_text: str) -> dict:
    enc = _encoder()
    l1_o, l1_p = normalize_l1(ocr_text), normalize_l1(pdf_text)
    l2_o, l2_p = normalize_l2(ocr_text), normalize_l2(pdf_text)
    # Batch encoding
    t_ocr_raw, t_pdf_raw, t_ocr_l1, t_pdf_l1, t_ocr_l2, t_pdf_l2 = (
        len(e) for e in enc.encode_ordinary_batch(
            [ocr_text, pdf_text, l1_o, l1_p, l2_o, l2_p]))
    row = {
        "tokens_ocr_raw": t_ocr_raw, "tokens_pdf_raw": t_pdf_raw,
        "tokens_ocr_l1": t_ocr_l1, "tokens_pdf_l1": t_pdf_l1,
        "tokens_ocr_l2": t_ocr_l2, "tokens_pdf_l2": t_pdf_l2,
        "garbage_ratio_ocr_raw": round(garbage_ratio(ocr_text), 4),
        "garbage_ratio_pdf_raw": round(garbage_ratio(pdf_text), 4),
    }
    row["len_ratio_token_raw"] = (
        row["tokens_ocr_raw"] / row["tokens_pdf_raw"]
        if row["tokens_pdf_raw"] else float("inf"))
    row["len_ratio_token_l1"] = (
        row["tokens_ocr_l1"] / row["tokens_pdf_l1"]
        if row["tokens_pdf_l1"] else float("inf"))
    for level, (a, b) in (("l1", (l1_o, l1_p)), ("l2", (l2_o, l2_p))):
        for k, v in _pair_metrics(a, b).items():
            row[f"{k}_{level}"] = round(v, 4) if v != float("inf") else -1.0
    for k in ("len_ratio_token_raw", "len_ratio_token_l1"):
        if row[k] == float("inf"):
            row[k] = -1.0
    return row

# Pulling tar bytes into RAM (safe and quicker cause our data, but maybe unscalable?)
def untar_pages_from_bytes(raw: bytes) -> dict[int, str]:
    """Read a .tar.gz's page files straight from memory into the form {page_no: text}"""
    pages: dict[int, str] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
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

# Worker 
_DL = None
def _init_worker() -> None:
    global _DL
    cfg = Config(max_pool_connections=MAX_POOL_CONNECTIONS,
                 retries={"max_attempts": 4, "mode": "standard"},
                 region_name=NATIVE_REGION)
    _DL = S3DataLoader(bucket_name=NATIVE_BUCKET, config=cfg)
    _encoder()

def _get_tar_bytes(key: str) -> bytes:
    return _DL.s3.get_object(Bucket=NATIVE_BUCKET, Key=key)["Body"].read()

def process_digest(digest: str) -> list[dict]:
    """Return per-page metric rows for one digest. Empty list on hard error"""
    try:
        ocr_key = f"{PRODUCT_KEY_PREFIX}/ocr_text/{digest}.tar.gz"
        ext_key = f"{PRODUCT_KEY_PREFIX}/extracted_text/{digest}.tar.gz"
        with ThreadPoolExecutor(max_workers=2) as tp:
            ocr_fut = tp.submit(_get_tar_bytes, ocr_key)
            ext_fut = tp.submit(_get_tar_bytes, ext_key)
            ocr_pages = untar_pages_from_bytes(ocr_fut.result())
            try:
                pdf_pages = untar_pages_from_bytes(ext_fut.result())
                ext_missing = False
            except ClientError as e:
                if is_not_found(e):
                    pdf_pages, ext_missing = {}, True
                else:
                    raise

        enc = _encoder()
        rows: list[dict] = []
        # NOTE: Maybe here too we can batch compute across pages using threads on top of the process pool. Thoughts? Its compute.
        for pg in sorted(set(ocr_pages) | set(pdf_pages)):
            ocr_t = ocr_pages.get(pg, "")
            pdf_t = pdf_pages.get(pg, "")
            has_ocr, has_pdf = bool(ocr_t.strip()), bool(pdf_t.strip())
            if has_ocr and has_pdf:
                m = compute_page_metrics(ocr_t, pdf_t)
                rows.append({"digest": digest, "page": pg, "bucket": "both",
                             "extracted_tar_missing": ext_missing, **m})
            else:
                bucket = ("ocr_only" if has_ocr
                          else "pdf_text_only" if has_pdf else "neither")
                tok = len(enc.encode_ordinary(ocr_t or pdf_t))
                rows.append({"digest": digest, "page": pg, "bucket": bucket,
                             "extracted_tar_missing": ext_missing,
                             "tokens_present": tok})
        return {"digest": digest, "rows": rows, "error": None}
    except Exception as e:
        log_error(f"analysis failed {digest}", e)
        return {"digest": digest, "rows": [], "error": repr(e)}

# Parquet file
_FLOAT, _INT, _STR, _BOOL = pa.float64(), pa.int64(), pa.string(), pa.bool_()
_METRIC_FLOAT_COLS = [
    "garbage_ratio_ocr_raw", "garbage_ratio_pdf_raw",
    "len_ratio_token_raw", "len_ratio_token_l1",
]
for _lvl in ("l1", "l2"):
    _METRIC_FLOAT_COLS += [
        f"len_ratio_char_{_lvl}", f"norm_edit_distance_char_{_lvl}",
        f"norm_edit_distance_word_{_lvl}", f"rouge_l_{_lvl}",
        f"jaccard_{_lvl}", f"containment_ocr_{_lvl}", f"containment_pdf_{_lvl}",
    ]
_METRIC_INT_COLS = [
    "tokens_ocr_raw", "tokens_pdf_raw", "tokens_ocr_l1", "tokens_pdf_l1",
    "tokens_ocr_l2", "tokens_pdf_l2",
]
PARQUET_SCHEMA = pa.schema(
    [("digest", _STR), ("page", _INT), ("bucket", _STR),
     ("extracted_tar_missing", _BOOL), ("tokens_present", _INT)]
    + [(c, _INT) for c in _METRIC_INT_COLS]
    + [(c, _FLOAT) for c in _METRIC_FLOAT_COLS]
)

def write_batch_parquet(rows: list[dict], path: str) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA),
                   path, compression="zstd")

MISS_LEDGER = LOCAL_DIR / "analysis_misses.jsonl"
# Main run functions
def run(args) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = LOCAL_DIR / "metrics_out"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    lister = S3DataLoader(
        bucket_name=NATIVE_BUCKET,
        config=Config(max_pool_connections=MAX_POOL_CONNECTIONS,
                      region_name=NATIVE_REGION))
    parquet_client = boto3.client("s3", region_name=PARQUET_REGION)
    exclude = load_ignore_list(args.ignore_list)
    list_prefix = f"{PRODUCT_KEY_PREFIX}/ocr_text/"

    ckpt = JsonCheckpoint(str(LOCAL_DIR / "analysis.ckpt"))
    token = ckpt.state.get("token")
    batch_num = ckpt.state.get("batch_num", 0)
    digests_done = ckpt.state.get("digests_done", 0)

    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as pool:
        def _list_next(tok):
            res = lister.list_objects(list_prefix, max_keys=args.batch_size,
                                    continuation_token=tok)
            digs = []
            for k in res.keys:
                d = Path(k).name.removesuffix(".tar.gz")
                if DIGEST_RE.match(d) and d not in exclude:
                    digs.append(d)
            return digs, res.continuation_token

        list_pool = ThreadPoolExecutor(max_workers=1)
        digests, token = _list_next(token) 
        while digests:
            next_future = list_pool.submit(_list_next, token)

            logging.info("batch %d: %d digests submitted", batch_num, len(digests))
            batch_rows = []
            future_to_digest = {pool.submit(process_digest, d): d for d in digests}
            for fut in as_completed(future_to_digest):
                d = future_to_digest[fut]
                digests_done += 1
                try:
                    r = fut.result()
                except Exception as e:
                    log_error(f"worker crashed on {d}", e)
                    with open(MISS_LEDGER, "a") as f:
                        f.write(json.dumps({"digest": d, "reason": "worker_crash",
                                            "error": repr(e), "batch": batch_num}) + "\n")
                    continue
                if r["error"] is not None:
                    with open(MISS_LEDGER, "a") as f:
                        f.write(json.dumps({"digest": d, "reason": "process_error",
                                            "error": r["error"], "batch": batch_num}) + "\n")
                elif not r["rows"]:
                    with open(MISS_LEDGER, "a") as f:
                        f.write(json.dumps({"digest": d, "reason": "zero_rows",
                                            "error": None, "batch": batch_num}) + "\n")
                batch_rows.extend(r["rows"])

            if batch_rows:
                fname = f"metrics_batch_{batch_num:06d}.parquet"
                fpath = str(metrics_dir / fname)
                write_batch_parquet(batch_rows, fpath)
                if not args.no_cloud_sync:
                    parquet_client.upload_file(
                        fpath, PARQUET_BUCKET, f"{PARQUET_REMOTE_PREFIX}/{fname}")

            batch_num += 1
            # collect prefetched next batch (listed during compute above)
            digests, token = next_future.result()
            ckpt.state = {"token": token, "batch_num": batch_num,
                        "digests_done": digests_done, "finished": token is None}
            ckpt.save()
            logging.info("batch %d done — %d processed", batch_num, digests_done)

            if args.sample and digests_done >= args.sample:
                break

        list_pool.shutdown()
        logging.info("==== EXTRACTION DONE ==== %d digests, %d batches",
                    digests_done, batch_num)

def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=14,
                   help="CPU-bound (edit distances); leave headroom under vCPU count")
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--ignore-list", default="fallback_ignore.txt")
    p.add_argument("--sample", type=int, default=0,
                   help="stop after N digests (0 = full corpus)")
    p.add_argument("--no-cloud-sync", action="store_true",
                   help="keep parquet local only")
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()