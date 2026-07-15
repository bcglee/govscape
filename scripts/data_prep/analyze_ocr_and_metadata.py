from __future__ import annotations

import argparse
import collections
import heapq
import json
import logging
import time
import os
import random
import re
import shutil
import tarfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

import boto3
import tiktoken
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import LCSseq, Levenshtein

from govscape.config import DataModel
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_BUCKET = "eot-pdf-archive"
OCR_TEXT_PREFIX = "ocr_text/"
PDF_BUCKET = "govscape"
PDF_KEY_TEMPLATE = "eota-pdf-archive/pdfs/{digest}.pdf"
PDF_ENDPOINT = "https://data.source.coop"
EXTRACTED_TEXT_PREFIX = "pdf_extracted_text"
METRICS_REMOTE_SUBDIR = "ocr_metrics"
TRANSIENT_CODES = {"500", "502", "503", "SlowDown",
                   "520", "521", "522", "523", "524", "525", "526"}

LOCAL_DATA_DIR = Path("data/ocr_agreement")
ERROR_LOG = LOCAL_DATA_DIR / "errors.log"

TOKENIZER_NAME = "o200k_base"  # GPT-4o; single tokenizer per arXiv:2506.08300 discussion
GARBAGE_THRESHOLD = 0.15
LEN_RATIO_LOW, LEN_RATIO_HIGH = 0.2, 5.0
LOW_AGREEMENT_ROUGE, LOW_AGREEMENT_JACCARD = 0.30, 0.30
SCRAMBLE_JACCARD, SCRAMBLE_ROUGE = 0.70, 0.40
CONTAINMENT_HIGH, CONTAINMENT_LOW = 0.90, 0.50
MILD_ROUGE = 0.70

HIGH_FLAGS = {"GARBAGE_OCR", "GARBAGE_PDF", "LOW_AGREEMENT", "EXTREME_LEN_RATIO"}
MEDIUM_FLAGS = {"ORDER_SCRAMBLE", "PDF_MISSING_CONTENT", "OCR_MISSING_CONTENT"}
# olmOCR-paper alignment buckets, for comparability with their reported 0.875 avg.
ALIGN_BUCKETS = [("low", 0.0, 0.70), ("medium", 0.70, 0.95), ("high", 0.95, 1.01)]

# Normalization constants & functions

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>\n]{1,120}>") # HTML/XML tags
_MD_RE = re.compile(r"(^#{1,6}\s+)|(\*{1,3})|(`{1,3})|(^\s*[-*+]\s+)", re.M)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_l1(text: str) -> str:
    """Minimal: NFKC + whitespace collapse."""
    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()

def normalize_l2(text: str) -> str:
    """Aggressive: L1 + strip HTML/markdown + lowercase + strip punctuation."""
    text = _TAG_RE.sub(" ", text)
    text = _MD_RE.sub(" ", text)
    text = normalize_l1(text).lower()
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()

def garbage_ratio(text: str) -> float:
    """Fraction of chars that are U+FFFD, control, or otherwise non-word/space/punct."""
    if not text:
        return 0.0
    bad_chars = 0
    for ch in text:
        if ch == "\ufffd":
            bad_chars += 1
            continue
        cat = unicodedata.category(ch)
        if cat.startswith(("C",)) and ch not in "\n\t\r":
            bad_chars += 1
        elif not (ch.isalnum() or ch.isspace() or cat.startswith(("P", "S", "M"))):
            bad_chars += 1
    return bad_chars / len(text)

# Metrics constants & functions
_ENC = None

# Lazy loader for the tokenizer 
def _encoder():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding(TOKENIZER_NAME)
    return _ENC

def _pair_metrics(text_a: str, text_b: str) -> dict:
    """Symmetric agreement metrics between two normalized texts (a=OCR, b=PDF)"""
    words_a, words_b = text_a.split(), text_b.split()
    la, lb = len(text_a), len(text_b)
    out = {
        "len_ratio_char": (la / lb) if lb else float("inf") if la else 1.0,
        "norm_edit_distance_char": Levenshtein.normalized_distance(text_a, text_b),
        "norm_edit_distance_word": Levenshtein.normalized_distance(words_a, words_b),
    }
    lcs = LCSseq.similarity(words_a, words_b)
    p = lcs / len(words_a) if words_a else 0.0
    r = lcs / len(words_b) if words_b else 0.0
    out["rouge_l"] = (2 * p * r / (p + r)) if (p + r) else 0.0
    set_a, set_b = set(words_a), set(words_b)
    inter, union = len(set_a & set_b), len(set_a | set_b)
    out["jaccard"] = inter / union if union else 1.0
    out["containment_ocr"] = inter / len(set_a) if set_a else 0.0  # OCR vocab found in PDF
    out["containment_pdf"] = inter / len(set_b) if set_b else 0.0  # PDF vocab found in OCR
    return out

def compute_page_metrics(ocr_text: str, pdf_text: str) -> dict:
    enc = _encoder()
    # Normalize the text at both levels
    l1_ocr, l1_pdf = normalize_l1(ocr_text), normalize_l1(pdf_text)
    l2_ocr, l2_pdf = normalize_l2(ocr_text), normalize_l2(pdf_text)

    # Compute the metrics for the raw text
    row = {
        "tokens_ocr_raw": len(enc.encode(ocr_text, disallowed_special=())),
        "tokens_pdf_raw": len(enc.encode(pdf_text, disallowed_special=())),
        "garbage_ratio_ocr_raw": round(garbage_ratio(ocr_text), 4),
        "garbage_ratio_pdf_raw": round(garbage_ratio(pdf_text), 4),
        "tokens_ocr_l1": len(enc.encode(l1_ocr, disallowed_special=())),
        "tokens_pdf_l1": len(enc.encode(l1_pdf, disallowed_special=())),
        "garbage_ratio_ocr_l1": round(garbage_ratio(l1_ocr), 4),
        "garbage_ratio_pdf_l1": round(garbage_ratio(l1_pdf), 4),
    }

    # Compute the length ratio for the raw text and l1 normalized text
    row["len_ratio_token_raw"] = (
        row["tokens_ocr_raw"] / row["tokens_pdf_raw"] if row["tokens_pdf_raw"] else float("inf")
    )
    row["len_ratio_token_l1"] = (
        row["tokens_ocr_l1"] / row["tokens_pdf_l1"] if row["tokens_pdf_l1"] else float("inf")
    )

    for level, (a, b) in (("l1", (l1_ocr, l1_pdf)), ("l2", (l2_ocr, l2_pdf))):
        for k, v in _pair_metrics(a, b).items():
            row[f"{k}_{level}"] = round(v, 4) if v != float("inf") else -1.0
    return row

def flag_page(m: dict) -> tuple[list[str], str]:
    """Derive flags + severity label from L2 metrics. Combinations tell the story."""
    flags = []
    if m["garbage_ratio_ocr_raw"] > GARBAGE_THRESHOLD:
        flags.append("GARBAGE_OCR")
    if m["garbage_ratio_pdf_raw"] > GARBAGE_THRESHOLD:
        flags.append("GARBAGE_PDF")
    ratio = m["len_ratio_char_l2"]
    if ratio == -1.0 or ratio < LEN_RATIO_LOW or ratio > LEN_RATIO_HIGH:
        flags.append("EXTREME_LEN_RATIO")
    rouge, jac = m["rouge_l_l2"], m["jaccard_l2"]
    if rouge < LOW_AGREEMENT_ROUGE and jac < LOW_AGREEMENT_JACCARD:
        flags.append("LOW_AGREEMENT")
    if jac >= SCRAMBLE_JACCARD and rouge < SCRAMBLE_ROUGE:
        flags.append("ORDER_SCRAMBLE")  # same words, jumbled order
    if m["containment_pdf_l2"] >= CONTAINMENT_HIGH and m["containment_ocr_l2"] < CONTAINMENT_LOW:
        flags.append("PDF_MISSING_CONTENT")
    if m["containment_ocr_l2"] >= CONTAINMENT_HIGH and m["containment_pdf_l2"] < CONTAINMENT_LOW:
        flags.append("OCR_MISSING_CONTENT")
    if any(f in HIGH_FLAGS for f in flags):
        sev = "high"
    elif any(f in MEDIUM_FLAGS for f in flags):
        sev = "medium"
    elif flags or rouge < MILD_ROUGE:
        sev = "low"
    else:
        sev = "ok"
    return flags, sev

# Worker constants & functions
_PDF_S3 = None
pdfium = None
 
def _init_worker() -> None:
    """Per-process setup"""
    global _PDF_S3, pdfium
    _PDF_S3 = boto3.client(
        "s3",
        endpoint_url=PDF_ENDPOINT,
        config=Config(signature_version=UNSIGNED),
    )
    import pypdfium2 as pdfium
    _encoder()
 
def _log_error(label: str, error: Exception | str) -> None:
    msg = f"{label} — {error}"
    logging.error("!!! %s", msg)
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(f"ERROR: {msg}\n")
 
def _extract_pdf_pages(pdf_path: str) -> list[str]:
    """Per-page text via pypdfium2"""
    pages: list[str] = []
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_bounded())
            textpage.close()
            page.close()
    finally:
        pdf.close()
    return pages

def process_digest(
    digest: str,
    ocr_page_paths: list[str],
    work_dir: str,
    out_tar_dir: str,
    profile_only: bool,
) -> dict:
    """One digest: download PDF, extract, compare against OCR pages, tar output."""
    result: dict = {
        "digest": digest,
        "rows": [],
        "buckets": collections.Counter(),
        "totals": collections.Counter(),
        "profile": collections.Counter(),
        "error": None,
    }

    ocr_pages: dict[int, str] = {}
    for p in ocr_page_paths:
        try:
            pg = int(Path(p).stem.rsplit("_", 1)[1]) - 1
            ocr_pages[pg] = Path(p).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            _log_error(f"OCR page read {p}", e)

    # Download the source PDF
    pdf_local = os.path.join(work_dir, f"{digest}.pdf")
    key = PDF_KEY_TEMPLATE.format(digest=digest)
    try:
        os.makedirs(work_dir, exist_ok=True)
        for attempt in range(4):
            try:
                _PDF_S3.download_file(PDF_BUCKET, key, pdf_local)
                break
            except ClientError as e:
                code = str(e.response.get("Error", {}).get("Code", ""))
                if code in TRANSIENT_CODES and attempt < 3:
                    logging.warning("transient %s on %s, retry %d", code, key, attempt + 1)
                    time.sleep((2 ** attempt) + random.random())
                    continue
                raise
    except Exception as e:
        _log_error(f"PDF DOWNLOAD FAILED s3://{PDF_BUCKET}/{key}", e)
        result["error"] = f"pdf_download: {e}"
        result["buckets"]["error_digests"] += 1
        return result

    try:
        pdf_pages = _extract_pdf_pages(pdf_local)
    except Exception as e:
        _log_error(f"pypdfium2 extraction failed {digest}", e)
        result["error"] = f"pdf_extract: {e}"
        result["buckets"]["error_digests"] += 1
        os.remove(pdf_local)
        return result
    finally:
        if os.path.exists(pdf_local):
            os.remove(pdf_local)

    if profile_only:
        for text, side in [(t, "ocr") for t in ocr_pages.values()] + [
            (t, "pdf") for t in pdf_pages
        ]:
            for name, count in profile_text(text).items():
                result["profile"][f"{side}:{name}"] += count
        result["buckets"]["profiled_digests"] += 1
        return result

    # Write extracted pages and tar as {digest}.tar.gz with contents {digest}_{pg}.txt
    digest_dir = os.path.join(work_dir, digest)
    os.makedirs(digest_dir, exist_ok=True)
    for pg, text in enumerate(pdf_pages):
        Path(digest_dir, f"{digest}_{pg + 1}.txt").write_text(text, encoding="utf-8")
    os.makedirs(out_tar_dir, exist_ok=True)
    tar_path = os.path.join(out_tar_dir, f"{digest}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for pg in range(len(pdf_pages)):
            fname = f"{digest}_{pg + 1}.txt"
            tar.add(os.path.join(digest_dir, fname), arcname=fname)
    shutil.rmtree(digest_dir, ignore_errors=True)

    # Bucket pages and compute metrics
    b, totals = result["buckets"], result["totals"]
    enc = _encoder()
    n_pages = max(len(pdf_pages), max(ocr_pages, default=-1) + 1)
    
    for pg in range(n_pages):
        ocr_t = ocr_pages.get(pg, "")
        pdf_t = pdf_pages[pg] if pg < len(pdf_pages) else ""
        has_ocr, has_pdf = bool(ocr_t.strip()), bool(pdf_t.strip())
        if has_ocr and has_pdf:
            b["both"] += 1
            m = compute_page_metrics(ocr_t, pdf_t)
            flags, sev = flag_page(m)
            totals["tokens_ocr"] += m["tokens_ocr_l1"]
            totals["tokens_pdf"] += m["tokens_pdf_l1"]
            result["rows"].append(
                {"digest": digest, "page": pg, "bucket": "both",
                 "flags": flags, "severity": sev, **m}
            )
        else:
            bucket = ("ocr_only" if has_ocr
                      else "pdf_text_only" if has_pdf else "neither")
            b[bucket] += 1
            tok = len(enc.encode(normalize_l1(ocr_t or pdf_t), disallowed_special=()))
            if has_ocr:
                totals["tokens_ocr"] += tok
            elif has_pdf:
                totals["tokens_pdf"] += tok
            result["rows"].append(
                {"digest": digest, "page": pg, "bucket": bucket,
                 "tokens": tok, "flags": [], "severity": "ok"}
            )
    b["ok_digests"] += 1
    return result

# Profiler constants & functions

PROFILE_PATTERNS = {
    "html_tags": _TAG_RE,
    "markdown_artifacts": re.compile(r"(^#{1,6}\s)|(\*\*)|(```)", re.M),
    "vlm_refusal": re.compile(
        r"(i'?m (unable|sorry|not able))|(cannot (transcribe|extract|process))|(as an ai)",
        re.I,
    ),
    "replacement_char": re.compile("\ufffd"),
    "control_chars": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),
    "hyphen_linebreak": re.compile(r"\w-\n\w"),
    "ligatures": re.compile(r"[\ufb00-\ufb06]"),
}

def profile_text(text: str) -> dict[str, int]:
    counts = {name: len(rx.findall(text)) for name, rx in PROFILE_PATTERNS.items()}
    counts["pages_seen"] = 1
    counts["pages_with_any_hit"] = int(any(v for k, v in counts.items() if k != "pages_seen"))
    return counts

# Aggregation constants & functions
HIST_KEYS = ("rouge_l_l2", "jaccard_l2", "norm_edit_distance_char_l2")
HIST_LABELS = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]

def _flag_counts(row: dict) -> tuple[int, int, int]:
    n_high = sum(1 for f in row["flags"] if f in HIGH_FLAGS)
    n_med = sum(1 for f in row["flags"] if f in MEDIUM_FLAGS)
    n_low = int(not row["flags"] and row["severity"] == "low")
    return n_high, n_med, n_low

def _hist_bin(value: float) -> str:
    return HIST_LABELS[min(int(value * 10), 9)]

def _align_bucket(rouge: float) -> str:
    return next(name for name, lo, hi in ALIGN_BUCKETS if lo <= rouge < hi)

class TopK:
    """Worst (digest, page) tuples per flag: most severity flags hit,
    then highest char-level disagreement (CER, L2)."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.heaps: dict[str, list] = collections.defaultdict(list)
        self._counter = 0 

    @staticmethod
    def score(row: dict) -> tuple:
        n_high, n_med, n_low = _flag_counts(row)
        cer = row.get("norm_edit_distance_char_l2", 0.0)
        return (n_high, n_med, n_low, cer)

    def add(self, row: dict) -> None:
        s = self.score(row)
        entry = {
            "digest": row["digest"], "page": row["page"],
            "severity": row["severity"], "flags": row["flags"],
            "score": list(s),
            "cer_l2": row.get("norm_edit_distance_char_l2"),
            "rouge_l_l2": row.get("rouge_l_l2"),
            "jaccard_l2": row.get("jaccard_l2"),
        }
        for flag in row["flags"] or (["ANY"] if row["severity"] != "ok" else []):
            heap = self.heaps[flag]
            self._counter += 1
            item = (s, self._counter, entry)
            if len(heap) < self.k:
                heapq.heappush(heap, item)
            elif s > heap[0][0]:
                heapq.heapreplace(heap, item)

    def dump(self) -> dict:
        return {
            flag: [e for _, _, e in sorted(heap, key=lambda t: (-t[0][0], -t[0][1], -t[0][2], -t[0][3]))]
            for flag, heap in self.heaps.items()
        }

class Aggregator:
    def __init__(self) -> None:
        self.buckets = collections.Counter()
        self.totals = collections.Counter()
        self.severity = collections.Counter()
        self.flags = collections.Counter()
        self.hists = {k: collections.Counter() for k in HIST_KEYS}
        self.align = collections.Counter()

    def add_result(self, result: dict) -> None:
        self.buckets.update(result["buckets"])
        self.totals.update(result["totals"])
        for row in result["rows"]:
            self.severity[row["severity"]] += 1
            self.flags.update(row["flags"])
            if row["bucket"] != "both":
                continue
            for k in HIST_KEYS:
                self.hists[k][_hist_bin(row[k])] += 1
            self.align[_align_bucket(row["rouge_l_l2"])] += 1

    def summary(self) -> dict:
        return {
            "page_buckets": dict(self.buckets),
            "token_totals": {
                "tokenizer": TOKENIZER_NAME,
                "tokens_ocr_total": self.totals["tokens_ocr"],
                "tokens_pdf_total": self.totals["tokens_pdf"],
            },
            "severity_counts": dict(self.severity),
            "flag_counts": dict(self.flags),
            "alignment_buckets_olmocr_style": dict(self.align),
            "metric_histograms": {k: dict(v) for k, v in self.hists.items()},
        }

# Main Loop

def group_by_digest(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for p in paths:
        if not p.endswith(".txt"):
            continue
        stem = Path(p).stem
        if "_" in stem:
            groups[stem.rsplit("_", 1)[0]].append(p)
    return groups

def load_ignore_list(path: str) -> set[str]:
    if not os.path.exists(path):
        logging.warning("Ignore list %s not found — proceeding without it.", path)
        return set()
    ignored = {line.strip() for line in open(path) if line.strip()}
    logging.info("Loaded %d ignored digests from %s", len(ignored), path)
    return ignored

def write_batch_parquet(rows: list[dict], path: str) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")

def run(args) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    in_dir = str(LOCAL_DATA_DIR / "ocr_in")
    work_dir = str(LOCAL_DATA_DIR / "work")
    out_tar_dir = str(LOCAL_DATA_DIR / "extracted_tars")
    metrics_dir = LOCAL_DATA_DIR / "metrics_out"

    data_loader = build_data_loader(args.backend, args.bucket_name)
    remote_dm = DataModel(args.remote_data_dir)
    ignored = load_ignore_list(args.ignore_list)

    ckpt_name = "ocr_agreement_profile" if args.profile_only else "ocr_agreement"
    remote_ckpt = os.path.join(remote_dm.checkpoints_directory, f"{ckpt_name}.checkpoint")

    agg = Aggregator()
    topk = TopK(args.top_k)
    profile_totals = collections.Counter()
    digests_done = 0
    rng = random.Random(1337)

    with RemoteDirectoryIterator(
        data_loader=data_loader,
        prefix=OCR_TEXT_PREFIX,
        remote_checkpoint_path=remote_ckpt,
        local_checkpoint_path=str(LOCAL_DATA_DIR / f"{ckpt_name}.checkpoint"),
        local_dir=in_dir,
        use_multiprocessing=False,
    ) as it, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
    ) as pool:
        batch_num = 0
        while not it.finished:
            paths = it.download_batch(max_keys=args.batch_size)
            groups = {d: ps for d, ps in group_by_digest(paths).items() if d not in ignored}
            skipped = len(group_by_digest(paths)) - len(groups)
            if skipped:
                logging.info("Skipped %d ignored digests in this batch.", skipped)

            if args.profile_only and args.sample and groups:
                # Downsample so total profiled digests approaches --sample.
                budget = max(0, args.sample - digests_done)
                keys = list(groups)
                if budget < len(keys):
                    keys = rng.sample(keys, budget)
                groups = {k: groups[k] for k in keys}

            if groups:
                futures = {
                    pool.submit(
                        process_digest,
                        d, ps, work_dir, out_tar_dir, args.profile_only
                        ): d for d, ps in groups.items()
                }
                batch_rows: list[dict] = []
                for fut in futures:
                    try:
                        result = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _log_error(f"worker crashed on {futures[fut]}", e)
                        agg.buckets["error_digests"] += 1
                        continue
                    digests_done += 1
                    if args.profile_only:
                        profile_totals.update(result["profile"])
                        continue
                    agg.add_result(result)
                    for row in result["rows"]:
                        if row["severity"] != "ok":
                            topk.add(row)
                    batch_rows.extend(result["rows"])

                if not args.profile_only:
                    # Upload re-extracted per-digest tars.
                    if os.path.isdir(out_tar_dir) and os.listdir(out_tar_dir):
                        data_loader.upload_directory(
                            local_dir=out_tar_dir,
                            remote_prefix=os.path.join(
                                args.remote_data_dir, EXTRACTED_TEXT_PREFIX
                            ),
                            compress=False,
                        )

                    # Upload batch metrics paraquet
                    if batch_rows:
                        metrics_dir.mkdir(parents=True, exist_ok=True)
                        write_batch_parquet(
                            batch_rows, str(metrics_dir / f"metrics_batch_{batch_num:06d}.parquet")
                        )
            it.save_checkpoint()
            for d in (in_dir, work_dir, out_tar_dir):
                shutil.rmtree(d, ignore_errors=True)
            batch_num += 1
            logging.info("Batch %d done — %d digests processed so far.", batch_num, digests_done)
            if args.profile_only and args.sample and digests_done >= args.sample:
                break

    # Final Reports
    if args.profile_only:
        pages = profile_totals.pop("ocr:pages_seen", 0) + profile_totals.pop("pdf:pages_seen", 0)
        logging.info("Profiler report over %d digests / %d pages:", digests_done, pages)
        for name, count in profile_totals.most_common():
            logging.info("  %-28s %d", name, count)
        report = {"digests": digests_done, "pages": pages, "counts": dict(profile_totals)}
        LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (LOCAL_DATA_DIR / "profile_report.json").write_text(json.dumps(report, indent=2))
        logging.info("Report written to %s", LOCAL_DATA_DIR / "profile_report.json")
        return

    summary = agg.summary()
    flagged = topk.dump()
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOCAL_DATA_DIR / "summary.json"
    flagged_path = LOCAL_DATA_DIR / "flagged_top_k.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    flagged_path.write_text(json.dumps(flagged, indent=2))
    # flagged_top_k.json stays LOCAL only (minimize uploads; ask Kyle if we
    # ever want it in performance/). summary.json is still uploaded.
    data_loader.upload_file(
        str(summary_path),
        os.path.join(remote_dm.performance_directory, METRICS_REMOTE_SUBDIR, "summary.json"),
    )

    logging.info("==== FINAL SUMMARY ====")
    logging.info("Page buckets: %s", summary["page_buckets"])
    logging.info(
        "Token totals (%s): OCR=%d  PDF=%d",
        TOKENIZER_NAME,
        summary["token_totals"]["tokens_ocr_total"],
        summary["token_totals"]["tokens_pdf_total"],
    )
    logging.info("Severity: %s", summary["severity_counts"])
    logging.info("Flags: %s", summary["flag_counts"])
    logging.info("olmOCR-style alignment buckets: %s", summary["alignment_buckets_olmocr_style"])
    logging.info(
        "Flagged tuples per category (top-%d) saved LOCALLY at %s (not uploaded)",
        args.top_k, flagged_path,
    )

def main() -> None:
    parser = base_argument_parser(
        description="Compare OCR text vs pypdfium2-extracted PDF text; re-upload extracted pages.",
    )
    parser.set_defaults(bucket_name=DEFAULT_BUCKET)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ignore-list", default="fallback_ignore.txt",
                        help="Local path to fallback ignore list (digests to skip).")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Worst tuples to keep per flag category.")
    parser.add_argument("--profile-only", action="store_true",
                        help="Stage-1 formatting profiler; no metrics or uploads.")
    parser.add_argument("--sample", type=int, default=3000,
                        help="Digest sample size for --profile-only.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()