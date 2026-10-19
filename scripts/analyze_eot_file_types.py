#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import duckdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PDF_SOURCE_BUCKET = "https://data.source.coop/govscape"
SOURCE_COOP_RAW_CDX_CANDIDATES = [
    "https://data.source.coop/govscape/eot-archive/cdx",
    "https://data.source.coop/govscape/eot-archive/cdx/*.parquet",
    "https://data.source.coop/govscape/eota-pdf-archive/cdx/complete_cdx.parquet",
    "https://data.source.coop/govscape/eota-ocr/",
]

MULTIPART_EXTENSIONS = (
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".csv.gz",
    ".json.gz",
    ".xml.gz",
    ".html.gz",
    ".txt.gz",
)

FILE_CLASS_MAP = {
    "pdf": "document",
    "doc": "document",
    "docx": "document",
    "rtf": "document",
    "txt": "document",
    "csv": "data",
    "tsv": "data",
    "xls": "data",
    "xlsx": "data",
    "json": "data",
    "xml": "data",
    "ppt": "presentation",
    "pptx": "presentation",
    "zip": "archive",
    "tar": "archive",
    "gz": "archive",
    "bz2": "archive",
    "xz": "archive",
    "tgz": "archive",
    "rar": "archive",
    "7z": "archive",
    "jpeg": "image",
    "jpg": "image",
    "png": "image",
    "gif": "image",
    "tif": "image",
    "tiff": "image",
    "svg": "image",
    "webp": "image",
    "bmp": "image",
    "mp3": "audio",
    "wav": "audio",
    "m4a": "audio",
    "ogg": "audio",
    "aac": "audio",
    "flac": "audio",
    "mp4": "video",
    "mov": "video",
    "webm": "video",
    "mpeg": "video",
    "mpg": "video",
    "avi": "video",
    "mkv": "video",
    "html": "web",
    "htm": "web",
    "css": "web",
    "js": "web",
    "woff": "web",
    "woff2": "web",
    "ttf": "web",
    "otf": "web",
}

GENERIC_MIME_TYPES = {
    "application/octet-stream",
    "text/plain",
    "application/unknown",
    "unknown",
}

MIME_EXTENSION_HINTS = {
    "application/pdf": {"pdf"},
    "text/csv": {"csv"},
    "application/csv": {"csv"},
    "application/vnd.ms-excel": {"xls", "csv"},
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {"xlsx"},
    "application/vnd.ms-word": {"doc"},
    "application/msword": {"doc"},
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {"docx"},
    "application/vnd.ms-powerpoint": {"ppt"},
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": {
        "pptx"
    },
    "application/json": {"json"},
    "application/xml": {"xml"},
    "text/xml": {"xml"},
    "application/zip": {"zip"},
    "application/gzip": {"gz", "tgz"},
    "application/x-gzip": {"gz", "tgz"},
    "audio/mpeg": {"mp3"},
    "audio/mp4": {"m4a", "mp4"},
    "audio/ogg": {"ogg"},
    "video/mp4": {"mp4"},
    "video/webm": {"webm"},
    "video/quicktime": {"mov"},
    "image/jpeg": {"jpg", "jpeg"},
    "image/png": {"png"},
    "image/gif": {"gif"},
    "image/svg+xml": {"svg"},
    "image/webp": {"webp"},
    "text/html": {"html", "htm"},
    "text/css": {"css"},
    "application/javascript": {"js"},
    "text/javascript": {"js"},
}


def normalize_mime(raw_mime: str | None) -> str:
    if raw_mime is None:
        return "unknown"
    value = raw_mime.strip()
    if not value:
        return "unknown"
    mime = value.split(";", 1)[0].strip().lower()
    return mime or "unknown"


def extract_url_extension(url: str | None) -> str:
    if not url:
        return "unknown"
    parsed = urlsplit(url)
    path = parsed.path or ""
    if not path:
        return "unknown"
    path = path.rstrip("/")
    if not path:
        return "unknown"
    basename = os.path.basename(path)
    if not basename or basename in {".", ".."}:
        return "unknown"
    lower_name = basename.lower()
    for suffix in sorted(
        (candidate for candidate in MULTIPART_EXTENSIONS), key=len, reverse=True
    ):
        if lower_name.endswith(suffix):
            return suffix.lstrip(".")
    if lower_name.endswith("."):
        return "unknown"
    if "." not in basename:
        return "unknown"
    _, ext = os.path.splitext(basename)
    ext_value = ext.lstrip(".").lower()
    return ext_value if ext_value else "unknown"


def summarize_category(value: str | None) -> str:
    normalized = (value or "unknown").strip().lower()
    if not normalized:
        return "unknown"
    if normalized in FILE_CLASS_MAP:
        return FILE_CLASS_MAP[normalized]
    if normalized.endswith(".gz"):
        normalized = "gz"
    # Generic MIME-type fallbacks.
    if normalized.startswith("application/"):
        if "pdf" in normalized:
            return "document"
        if "json" in normalized:
            return "data"
        if "xml" in normalized:
            return "data"
        if "zip" in normalized or "gzip" in normalized or "tar" in normalized:
            return "archive"
        if "excel" in normalized or "spreadsheet" in normalized:
            return "data"
        if "powerpoint" in normalized or "presentation" in normalized:
            return "presentation"
        if "word" in normalized or "document" in normalized:
            return "document"
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("audio/"):
        return "audio"
    if normalized.startswith("video/"):
        return "video"
    if normalized.startswith("text/"):
        if "html" in normalized:
            return "web"
        if "csv" in normalized or "json" in normalized or "xml" in normalized:
            return "data"
        if "plain" in normalized:
            return "document"
        return "web"
    return "unknown"


def classify_record(
    *,
    url: str | None,
    mime: str | None,
    status: str | None,
    digest: str | None,
    length: str | int | None = None,
    filename: str | None = None,
    offset: str | int | None = None,
    capture_date: str | None = None,
    collection: str | None = None,
) -> dict:
    normalized_mime = normalize_mime(mime)
    extension = extract_url_extension(url)
    category = summarize_category(extension)
    mime_category = (
        normalized_mime.split("/", 1)[0] if normalized_mime != "unknown" else "unknown"
    )
    mime_hints = MIME_EXTENSION_HINTS.get(normalized_mime, set())
    extension_matches_mime = True
    if (
        extension == "unknown"
        or normalized_mime == "unknown"
        or normalized_mime in GENERIC_MIME_TYPES
    ):
        extension_matches_mime = False
    else:
        extension_matches_mime = extension in mime_hints or (
            extension in {"html", "htm"}
            and normalized_mime in {"text/html", "application/xhtml+xml"}
        )
    return {
        "url": url,
        "mime": normalized_mime,
        "mime_category": mime_category,
        "extension": extension,
        "category": category,
        "extension_matches_mime": extension_matches_mime,
        "status": (status or "unknown").strip() or "unknown",
        "digest": (digest or "").strip(),
        "length": int(length)
        if length is not None and str(length).strip() not in {"", "None"}
        else 0,
        "filename": filename,
        "offset": int(offset)
        if offset is not None and str(offset).strip() not in {"", "None"}
        else 0,
        "capture_date": capture_date,
        "collection": collection,
    }


def parse_cdxj_line(raw_line: str | None) -> dict | None:
    if raw_line is None:
        return None
    text = raw_line.strip()
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    payload = text[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    timestamp = data.get("timestamp")
    return {
        "url": data.get("url"),
        "mime": data.get("mime"),
        "status": data.get("status"),
        "digest": data.get("digest"),
        "filename": data.get("filename"),
        "offset": data.get("offset"),
        "length": data.get("length"),
        "capture_date": timestamp,
        "collection": data.get("collection"),
    }


def determine_raw_data_blocker(input_path: str | None = None) -> str | None:
    if input_path is None:
        return None
    normalized = input_path.strip().lower()
    if "complete_cdx.parquet" in normalized or "eota-pdf-archive" in normalized:
        return (
            "The supplied input appears to be the PDF-selected Govscape "
            "dataset, not the raw unfiltered EOT CDX corpus."
        )
    if "eot-index/collections/" in normalized or normalized.startswith(
        "s3://eotarchive/"
    ):
        return None
    return None


def iter_cdxj_records(
    input_spec: str, *, max_rows: int | None = None, offset: int = 0
) -> list[dict]:
    records: list[dict] = []
    seen = 0
    skipped = 0
    with gzip.open(input_spec, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if skipped < offset:
                skipped += 1
                continue
            parsed = parse_cdxj_line(raw_line)
            if parsed is None:
                continue
            records.append(parsed)
            seen += 1
            if max_rows is not None and seen >= max_rows:
                break
    return records


def query_cdx_rows(
    input_spec: str,
    *,
    max_rows: int | None = None,
    collection: str | None = None,
    offset: int = 0,
) -> list[dict]:
    con = duckdb.connect()
    try:
        if input_spec.endswith(".parquet"):
            sql = "SELECT * FROM read_parquet(?)"
            params: list[object] = [input_spec]
            if max_rows is not None:
                sql += " LIMIT ?"
                params.append(max_rows)
            rows = con.execute(sql, params).fetchall()
            column_names = [
                "url",
                "mime",
                "status",
                "digest",
                "filename",
                "offset",
                "length",
                "capture_date",
                "collection",
            ]
            if rows and isinstance(rows[0], tuple):
                return [dict(zip(column_names, row, strict=True)) for row in rows]
            return []
        if input_spec.lower().endswith(".gz") and os.path.exists(input_spec):
            return iter_cdxj_records(input_spec, max_rows=max_rows, offset=offset)
        rows = con.execute(
            """
            SELECT raw_line
            FROM read_csv(
                ?,
                header=false,
                columns={'raw_line': 'VARCHAR'},
                compression='gzip',
                delim='\n',
                quote='',
                escape='',
                all_varchar=true,
                max_line_size=10000000
            )
            WHERE raw_line IS NOT NULL AND trim(raw_line) <> ''
            """,
            [input_spec],
        ).fetchall()
        parsed_rows: list[dict] = []
        for (raw_line,) in rows:
            if offset > 0:
                offset -= 1
                continue
            parsed = parse_cdxj_line(raw_line)
            if parsed is None:
                continue
            parsed_rows.append(parsed)
            if max_rows is not None and len(parsed_rows) >= max_rows:
                break
        return parsed_rows
    finally:
        con.close()


def infer_collection_name(input_spec: str, collection: str | None) -> str:
    if collection:
        return collection
    if "EOT-" in input_spec.upper():
        return input_spec[input_spec.upper().index("EOT-") :]
    if "eot" in input_spec.lower() and "2024" in input_spec:
        return "EOT-2024"
    return "unknown"


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summarise_by_key(rows: list[dict], key_name: str) -> list[dict]:
    buckets: defaultdict[str, dict] = defaultdict(dict)
    for row in rows:
        key = row.get(key_name, "unknown")
        if key is None:
            key = "unknown"
        entry = buckets[key]
        entry["count"] = entry.get("count", 0) + 1
        entry["distinct_urls"] = entry.get("distinct_urls", set())
        entry["distinct_urls"].add(row.get("url") or "")
        entry["distinct_digests"] = entry.get("distinct_digests", set())
        digest = row.get("digest")
        if digest:
            entry["distinct_digests"].add(digest)
        entry["total_length"] = entry.get("total_length", 0) + _safe_int(
            row.get("length")
        )
    result: list[dict] = []
    for key, entry in buckets.items():
        result.append(
            {
                "key": key,
                "capture_count": entry["count"],
                "distinct_urls": len(entry["distinct_urls"] - {""}),
                "distinct_digests": len(entry["distinct_digests"]),
                "total_length": entry["total_length"],
            }
        )
    result.sort(
        key=lambda item: (-item["capture_count"], -(item["total_length"]), item["key"])
    )
    return result


def build_extension_rows(rows: list[dict]) -> list[dict]:
    parts: defaultdict[str, dict] = defaultdict(dict)
    for row in rows:
        rec = classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
            length=row.get("length"),
            filename=row.get("filename"),
            offset=row.get("offset"),
            capture_date=row.get("capture_date"),
            collection=row.get("collection"),
        )
        ext = rec["extension"]
        entry = parts[ext]
        entry["capture_count"] = entry.get("capture_count", 0) + 1
        entry["distinct_urls"] = entry.get("distinct_urls", set())
        entry["distinct_urls"].add(rec["url"] or "")
        entry["distinct_digests"] = entry.get("distinct_digests", set())
        if rec["digest"]:
            entry["distinct_digests"].add(rec["digest"])
        entry["total_length"] = entry.get("total_length", 0) + rec["length"]
        entry["mime_counter"] = entry.get("mime_counter", Counter())
        entry["mime_counter"][rec["mime"]] += 1
    successful = sum(1 for row in rows if (row.get("status") or "").strip() == "200")
    total_length = sum(_safe_int(row.get("length")) for row in rows)
    result: list[dict] = []
    for extension, entry in parts.items():
        capture_count = entry["capture_count"]
        total_digests = len(entry["distinct_digests"])
        result.append(
            {
                "extension": extension,
                "capture_count": capture_count,
                "share_of_successful": (capture_count / successful)
                if successful
                else 0.0,
                "distinct_urls": len(entry["distinct_urls"] - {""}),
                "distinct_digests": total_digests,
                "total_length": entry["total_length"],
                "share_of_total_length": (entry["total_length"] / total_length)
                if total_length
                else 0.0,
                "avg_length": (entry["total_length"] / capture_count)
                if capture_count
                else 0.0,
                "median_length": 0.0,
                "most_common_mime": entry["mime_counter"].most_common(3),
            }
        )
    result.sort(
        key=lambda item: (
            -item["capture_count"],
            -item["total_length"],
            item["extension"],
        )
    )
    return result


def build_mime_rows(rows: list[dict]) -> list[dict]:
    buckets: defaultdict[str, dict] = defaultdict(dict)
    for row in rows:
        rec = classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
            length=row.get("length"),
        )
        mime = rec["mime"]
        entry = buckets[mime]
        entry["capture_count"] = entry.get("capture_count", 0) + 1
        entry["distinct_urls"] = entry.get("distinct_urls", set())
        entry["distinct_urls"].add(rec["url"] or "")
        entry["distinct_digests"] = entry.get("distinct_digests", set())
        if rec["digest"]:
            entry["distinct_digests"].add(rec["digest"])
        entry["total_length"] = entry.get("total_length", 0) + rec["length"]
        entry["ext_counter"] = entry.get("ext_counter", Counter())
        entry["ext_counter"][rec["extension"]] += 1
    successful = sum(1 for row in rows if (row.get("status") or "").strip() == "200")
    total_length = sum(_safe_int(row.get("length")) for row in rows)
    result: list[dict] = []
    for mime, entry in buckets.items():
        capture_count = entry["capture_count"]
        result.append(
            {
                "mime": mime,
                "capture_count": capture_count,
                "share_of_successful": (capture_count / successful)
                if successful
                else 0.0,
                "distinct_urls": len(entry["distinct_urls"] - {""}),
                "distinct_digests": len(entry["distinct_digests"]),
                "total_length": entry["total_length"],
                "share_of_total_length": (entry["total_length"] / total_length)
                if total_length
                else 0.0,
                "most_common_extensions": entry["ext_counter"].most_common(5),
            }
        )
    result.sort(
        key=lambda item: (-item["capture_count"], -item["total_length"], item["mime"])
    )
    return result


def build_extension_mime_rows(rows: list[dict]) -> list[dict]:
    buckets: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        rec = classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
            length=row.get("length"),
        )
        buckets[(rec["extension"], rec["mime"])] += 1
    return [
        {"extension": ext, "mime": mime, "capture_count": count}
        for (ext, mime), count in sorted(
            buckets.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]


def build_format_rollup(rows: list[dict]) -> list[dict]:
    extension_rows = build_extension_rows(rows)
    rollup = {
        "document": 0,
        "data": 0,
        "presentation": 0,
        "archive": 0,
        "image": 0,
        "audio": 0,
        "video": 0,
        "web": 0,
        "unknown": 0,
    }
    for item in extension_rows:
        ext = item["extension"]
        category = summarize_category(ext)
        if category in rollup:
            rollup[category] += item["capture_count"]
    return [
        {"category": category, "capture_count": count}
        for category, count in rollup.items()
    ]


def build_domain_rows(rows: list[dict]) -> list[dict]:
    domain_rows: defaultdict[str, dict] = defaultdict(dict)
    for row in rows:
        rec = classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
        )
        url = row.get("url") or ""
        hostname = ""
        if url:
            hostname = urlsplit(url).netloc.lower().split(":", 1)[0]
        key = (hostname, rec["category"])
        entry = domain_rows[key]
        entry["hostname"] = hostname
        entry["category"] = rec["category"]
        entry["capture_count"] = entry.get("capture_count", 0) + 1
        entry["distinct_urls"] = entry.get("distinct_urls", set())
        entry["distinct_urls"].add(url)
        entry["distinct_digests"] = entry.get("distinct_digests", set())
        if rec["digest"]:
            entry["distinct_digests"].add(rec["digest"])
    result = [
        {
            "hostname": item["hostname"],
            "category": item["category"],
            "capture_count": item["capture_count"],
            "distinct_urls": len(item["distinct_urls"] - {""}),
            "distinct_digests": len(item["distinct_digests"]),
        }
        for item in domain_rows.values()
    ]
    result.sort(
        key=lambda item: (
            -item["capture_count"],
            -item["distinct_digests"],
            item["hostname"],
            item["category"],
        )
    )
    return result


def build_duplication_rows(rows: list[dict]) -> list[dict]:
    by_extension: defaultdict[str, dict] = defaultdict(dict)
    for row in rows:
        rec = classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
            length=row.get("length"),
        )
        ext = rec["extension"]
        entry = by_extension[ext]
        entry["captures"] = entry.get("captures", 0) + 1
        entry["urls"] = entry.get("urls", set())
        entry["urls"].add(rec["url"] or "")
        entry["digests"] = entry.get("digests", set())
        if rec["digest"]:
            entry["digests"].add(rec["digest"])
    result = []
    for extension, entry in by_extension.items():
        captures = entry["captures"]
        digests = entry["digests"]
        urls = entry["urls"]
        result.append(
            {
                "extension": extension,
                "captures": captures,
                "distinct_urls": len(urls - {""}),
                "distinct_digests": len(digests),
                "captures_per_url": (captures / len(urls - {""})) if urls else 0.0,
                "captures_per_digest": (captures / len(digests)) if digests else 0.0,
                "duplicate_capture_fraction": 1.0 - (len(digests) / captures)
                if captures
                else 0.0,
            }
        )
    result.sort(
        key=lambda item: (
            -item["captures"],
            -item["distinct_digests"],
            item["extension"],
        )
    )
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary(path: Path, summary: str) -> None:
    path.write_text(summary, encoding="utf-8")


def run_analysis(
    input_spec: str,
    output_dir: Path,
    *,
    max_rows: int | None = None,
    collection: str | None = None,
    allow_pdf_filtered: bool = False,
    offset: int = 0,
) -> dict:
    logger.info("Inspecting input: %s", input_spec)
    blocker = determine_raw_data_blocker(input_spec)
    if blocker is not None and not allow_pdf_filtered:
        raise ValueError(
            f"{blocker} Use a raw/unfiltered CDX dataset or pass "
            "--allow-pdf-filtered for local validation only."
        )

    records = query_cdx_rows(
        input_spec,
        max_rows=max_rows,
        collection=collection,
        offset=offset,
    )
    if not records:
        raise ValueError(
            f"No rows were returned from {input_spec}. Check the input path or schema."
        )

    # Quick sanity check to catch obvious PDF-only filtered inputs.
    if not allow_pdf_filtered:
        sample = records[: min(len(records), 20000)]
        ext_counter = Counter(
            classify_record(
                url=row.get("url"),
                mime=row.get("mime"),
                status=row.get("status"),
                digest=row.get("digest"),
            )["extension"]
            for row in sample
        )
        pdf_ext = ext_counter.get("pdf", 0) + ext_counter.get("application/pdf", 0)
        non_pdf = sum(ext_counter.values()) - pdf_ext
        if (
            sample
            and pdf_ext
            and (pdf_ext / sum(ext_counter.values())) >= 0.95
            and non_pdf == 0
        ):
            raise ValueError(
                "The selected input appears to be PDF-only and has already "
                "passed through Govscape's PDF filter. Use the raw/unfiltered "
                "EOT CDX, not complete_cdx.parquet."
            )
    total_rows = len(records)
    successful_rows = sum(
        1 for row in records if (row.get("status") or "").strip() == "200"
    )
    non_success_rows = total_rows - successful_rows
    statuses = Counter(
        (row.get("status") or "unknown").strip() or "unknown" for row in records
    )
    unique_urls = {row.get("url") for row in records if row.get("url")}
    unique_digests = {row.get("digest") for row in records if row.get("digest")}
    total_length = sum(_safe_int(row.get("length")) for row in records)
    extension_rows = build_extension_rows(records)
    mime_rows = build_mime_rows(records)
    ext_mime_rows = build_extension_mime_rows(records)
    format_rows = build_format_rollup(records)
    domain_rows = build_domain_rows(records)
    duplication_rows = build_duplication_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    distinct_mime_types = {
        classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
        )["mime"]
        for row in records
    }
    distinct_url_extensions = {
        classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
        )["extension"]
        for row in records
    }
    captures_without_extension = sum(
        1
        for row in records
        if classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
        )["extension"]
        == "unknown"
    )
    captures_without_mime = sum(
        1
        for row in records
        if classify_record(
            url=row.get("url"),
            mime=row.get("mime"),
            status=row.get("status"),
            digest=row.get("digest"),
        )["mime"]
        == "unknown"
    )

    summary_lines = [
        "EOT CDX file-type census",
        "========================",
        f"input: {input_spec}",
        f"collection: {infer_collection_name(input_spec, collection)}",
        f"limited: {max_rows is not None}",
        f"scanned_records: {total_rows}",
        f"successful_captures: {successful_rows}",
        f"status_excluded: {non_success_rows}",
        f"distinct_urls: {len(unique_urls)}",
        f"distinct_digests: {len(unique_digests)}",
        f"distinct_mime_types: {len(distinct_mime_types)}",
        f"distinct_url_extensions: {len(distinct_url_extensions)}",
        f"captures_without_extension: {captures_without_extension}",
        f"captures_without_mime: {captures_without_mime}",
        f"sum_reported_length: {total_length}",
        "",
        "Status breakdown:",
    ]
    for status, count in statuses.most_common():
        summary_lines.append(f"  {status}: {count}")
    summary_lines.extend(["", "Top extensions:"])
    summary_lines.extend(
        f"  {item['extension']}: count={item['capture_count']} "
        f"total_length={item['total_length']}"
        for item in extension_rows[:10]
    )
    summary_lines.extend(["", "Top MIME types:"])
    summary_lines.extend(
        f"  {item['mime']}: count={item['capture_count']} "
        f"total_length={item['total_length']}"
        for item in mime_rows[:10]
    )
    summary = "\n".join(summary_lines) + "\n"
    write_summary(output_dir / "summary.txt", summary)
    write_csv(
        output_dir / "extensions.csv",
        [
            "extension",
            "capture_count",
            "share_of_successful",
            "distinct_urls",
            "distinct_digests",
            "total_length",
            "share_of_total_length",
            "avg_length",
            "median_length",
        ],
        extension_rows,
    )
    write_csv(
        output_dir / "mime_types.csv",
        [
            "mime",
            "capture_count",
            "share_of_successful",
            "distinct_urls",
            "distinct_digests",
            "total_length",
            "share_of_total_length",
        ],
        mime_rows,
    )
    write_csv(
        output_dir / "extension_mime.csv",
        [
            "extension",
            "mime",
            "capture_count",
        ],
        ext_mime_rows,
    )
    write_csv(
        output_dir / "format_rollup.csv",
        [
            "category",
            "capture_count",
        ],
        format_rows,
    )
    write_csv(
        output_dir / "domains_by_format.csv",
        [
            "hostname",
            "category",
            "capture_count",
            "distinct_urls",
            "distinct_digests",
        ],
        domain_rows,
    )
    write_csv(
        output_dir / "duplication.csv",
        [
            "extension",
            "captures",
            "distinct_urls",
            "distinct_digests",
            "captures_per_url",
            "captures_per_digest",
            "duplicate_capture_fraction",
        ],
        duplication_rows,
    )

    metadata = {
        "source_url": input_spec,
        "source_dataset": "Source.coop public bucket",
        "collection": infer_collection_name(input_spec, collection),
        "is_limited_run": max_rows is not None,
        "max_rows": max_rows,
        "input_is_pdf_selected": blocker is not None,
        "blocker_message": blocker,
        "total_cdx_records_scanned": total_rows,
        "successful_captures": successful_rows,
        "status_breakdown": dict(statuses),
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "classification_version": "eot-file-types-v1",
        "status_filter": (
            "HTTP status == 200 treated as successful capture; all other "
            "statuses retained in the census and status breakdown."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Census file types and MIME usage in raw End of Term CDX data."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Path to a raw CDX parquet or gzipped JSON-line CDX file. If "
            "omitted, the script defaults to the raw EOT S3 index path for "
            "the requested EOT collection."
        ),
    )
    parser.add_argument(
        "--collection", default="EOT-2024", help="Collection name such as EOT-2024."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="EOT year to use when composing the default raw S3 path.",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Where to write the analysis outputs."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit records processed for development runs.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N records before processing the slice.",
    )
    parser.add_argument(
        "--allow-pdf-filtered",
        action="store_true",
        help=(
            "Allow analysis of a PDF-selected dataset for validation only; "
            "default is to reject such inputs."
        ),
    )
    parser.add_argument(
        "--source-coop-confirmation",
        action="store_true",
        help="Print the Source.coop dataset discovery note and exit.",
    )
    parser.add_argument(
        "--source",
        choices=["s3", "source.coop", "local"],
        default="s3",
        help=(
            "Input source selection. The raw public Source.coop path is not "
            "currently available, so the default uses the raw EOT S3 index "
            "files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.source_coop_confirmation:
        print(
            "Source.coop discovery: the public govscape bucket currently exposes "
            "only the PDF-focused eota-pdf-archive dataset and eota-ocr prefix; "
            "no raw/unfiltered EOT CDX is visible there. The raw archived CDX "
            "is therefore not available through the public Source.coop path used "
            "in this repo, so the analysis falls back to the raw EOT S3 index "
            "files when a local or explicit input is not supplied."
        )
        return
    if not args.input:
        if args.source == "source.coop":
            raise SystemExit(
                "No raw Source.coop CDX is currently exposed; the public bucket "
                "contains only the PDF-selected Govscape archive. Use the raw "
                "EOT S3 input with --source s3 or pass --input explicitly."
            )
        if args.source == "local":
            raise SystemExit(
                "--source local requires --input to point at a raw local CDX file."
            )
        args.input = (
            f"s3://eotarchive/eot-index/collections/{args.collection}/indexes/*.gz"
        )
        if args.collection == "EOT-2024":
            args.input = (
                f"s3://eotarchive/eot-index/collections/EOT-{args.year}/indexes/*.gz"
            )
    output_dir = Path(args.output_dir)
    run_analysis(
        args.input,
        output_dir,
        max_rows=args.max_rows,
        collection=args.collection,
        allow_pdf_filtered=args.allow_pdf_filtered,
        offset=args.offset,
    )


if __name__ == "__main__":
    main()
