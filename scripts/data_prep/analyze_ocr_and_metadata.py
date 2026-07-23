from __future__ import annotations

import argparse
import io
import json
import logging
import re
import tarfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

# Target
PRODUCT_KEY_PREFIX = f"{NATIVE_PREFIX}{PRODUCT_PREFIX}"   # govscape/eota-ocr
PARQUET_BUCKET = "eot-pdf-archive"
PARQUET_REGION = "us-east-2"
PARQUET_REMOTE_PREFIX = "performance/ocr_metrics"
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
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def normalize_l2(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    return normalize_l1(text).lower()

def garbage_ratio(text: str) -> float:
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

# ---- Metrics ----
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

def untar_pages_from_bytes(raw: bytes) -> dict[int, str]:
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

_S3 = None
_THREADS = 32

def _init_proc(threads: int) -> None:
    global _S3, _THREADS
    _THREADS = threads
    cfg = Config(max_pool_connections=max(threads * 2, 64),
                 retries={"max_attempts": 4, "mode": "standard"},
                 region_name=NATIVE_REGION)
    _S3 = boto3.client("s3", config=cfg)
    _encoder()

def _get_tar_bytes(key: str) -> bytes:
    return _S3.get_object(Bucket=NATIVE_BUCKET, Key=key)["Body"].read()

def _process_one(digest: str) -> dict:
    """One digest, run inside a worker thread. Sequential fetches — concurrency
    comes from the thread pool running many digests at once."""
    try:
        ocr_key = f"{PRODUCT_KEY_PREFIX}/ocr_text/{digest}.tar.gz"
        ext_key = f"{PRODUCT_KEY_PREFIX}/extracted_text/{digest}.tar.gz"
        ocr_pages = untar_pages_from_bytes(_get_tar_bytes(ocr_key))
        try:
            pdf_pages = untar_pages_from_bytes(_get_tar_bytes(ext_key))
            ext_missing = False
        except ClientError as e:
            if is_not_found(e):
                pdf_pages, ext_missing = {}, True
            else:
                raise
        enc = _encoder()
        rows = []
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


def process_chunk(digests: list[str]) -> list[dict]:
    """Runs in a worker process usign threadpools"""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=_THREADS) as tp:
        for r in tp.map(_process_one, digests):
            results.append(r)
    return results


# ---- Parquet ----
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

def _ledger(reason: str, digest: str, error, batch_num: int) -> None:
    with open(MISS_LEDGER, "a") as f:
        f.write(json.dumps({"digest": digest, "reason": reason,
                            "error": error, "batch": batch_num}) + "\n")

# Main
def run(args) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = LOCAL_DIR / "metrics_out"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    lister = S3DataLoader(
        bucket_name=NATIVE_BUCKET,
        config=Config(max_pool_connections=64, region_name=NATIVE_REGION))
    parquet_client = boto3.client("s3", region_name=PARQUET_REGION)
    exclude = load_ignore_list(args.ignore_list)
    list_prefix = f"{PRODUCT_KEY_PREFIX}/ocr_text/"

    ckpt = JsonCheckpoint(str(LOCAL_DIR / "analysis.ckpt"))
    token = ckpt.state.get("token")
    batch_num = ckpt.state.get("batch_num", 0)
    digests_done = ckpt.state.get("digests_done", 0)

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

    with ProcessPoolExecutor(max_workers=args.procs,
                             initializer=_init_proc,
                             initargs=(args.threads,)) as pool:
        while digests:
            next_future = list_pool.submit(_list_next, token)

            # split batch into args.procs chunks, one per worker process
            chunks = [digests[i::args.procs] for i in range(args.procs)]
            chunks = [c for c in chunks if c]
            logging.info("batch %d: %d digests over %d procs x %d threads",
                         batch_num, len(digests), len(chunks), args.threads)

            batch_rows: list[dict] = []
            fut_to_chunk = {pool.submit(process_chunk, c): i
                            for i, c in enumerate(chunks)}
            for fut in as_completed(fut_to_chunk):
                try:
                    chunk_results = fut.result()
                except Exception as e:  # whole chunk process died
                    log_error(f"chunk {fut_to_chunk[fut]} crashed", e)
                    _ledger("chunk_crash", f"CHUNK_{fut_to_chunk[fut]}",
                            repr(e), batch_num)
                    continue
                for r in chunk_results:
                    digests_done += 1
                    if r["error"] is not None:
                        _ledger("process_error", r["digest"], r["error"], batch_num)
                    elif not r["rows"]:
                        _ledger("zero_rows", r["digest"], None, batch_num)
                    batch_rows.extend(r["rows"])

            if batch_rows:
                fname = f"metrics_batch_{batch_num:06d}.parquet"
                fpath = str(metrics_dir / fname)
                write_batch_parquet(batch_rows, fpath)
                if not args.no_cloud_sync:
                    parquet_client.upload_file(
                        fpath, PARQUET_BUCKET, f"{PARQUET_REMOTE_PREFIX}/{fname}")

            batch_num += 1
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
    p.add_argument("--procs", type=int, default=8,
                   help="worker processes (~= cores used for compute/GIL work)")
    p.add_argument("--threads", type=int, default=32,
                   help="fetch threads PER process; total concurrency = procs*threads")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="digests listed+processed per batch (split across procs)")
    p.add_argument("--ignore-list", default="fallback_ignore.txt")
    p.add_argument("--sample", type=int, default=0,
                   help="stop after N digests (0 = full corpus)")
    p.add_argument("--no-cloud-sync", action="store_true",
                   help="keep parquet local only")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()