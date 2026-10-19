#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from scripts.analyze_eot_file_types import (
    extract_url_extension,
    parse_cdxj_line,
    summarize_category,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    shard TEXT NOT NULL,
    start_record INTEGER NOT NULL,
    end_record INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    committed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS totals (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS status_counts (
    status TEXT PRIMARY KEY,
    capture_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS mime_counts (
    mime TEXT PRIMARY KEY,
    capture_count INTEGER NOT NULL,
    total_length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS extension_counts (
    extension TEXT PRIMARY KEY,
    capture_count INTEGER NOT NULL,
    total_length INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS extension_mime_counts (
    extension TEXT NOT NULL,
    mime TEXT NOT NULL,
    capture_count INTEGER NOT NULL,
    PRIMARY KEY (extension, mime)
);
CREATE TABLE IF NOT EXISTS format_counts (
    category TEXT PRIMARY KEY,
    capture_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS domain_category_counts (
    hostname TEXT NOT NULL,
    category TEXT NOT NULL,
    capture_count INTEGER NOT NULL,
    PRIMARY KEY (hostname, category)
);
CREATE TABLE IF NOT EXISTS seen_urls (url TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS seen_digests (digest TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS extension_urls (
    extension TEXT NOT NULL,
    url TEXT NOT NULL,
    PRIMARY KEY (extension, url)
);
CREATE TABLE IF NOT EXISTS extension_digests (
    extension TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (extension, digest)
);
CREATE TABLE IF NOT EXISTS mime_urls (
    mime TEXT NOT NULL,
    url TEXT NOT NULL,
    PRIMARY KEY (mime, url)
);
CREATE TABLE IF NOT EXISTS mime_digests (
    mime TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY (mime, digest)
);
"""


def _safe_int(value: object) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _hostname(url: str | None) -> str:
    if not url:
        return ""
    return urlsplit(url).netloc.lower().split(":", 1)[0]


def iter_cdxj_batches(
    input_path: Path, batch_size: int
) -> Iterator[tuple[int, int, list[dict]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict] = []
    start_record = 0
    chunk_index = 0
    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            record = parse_cdxj_line(raw_line)
            if record is None:
                continue
            if not batch:
                start_record = chunk_index * batch_size
            batch.append(record)
            if len(batch) == batch_size:
                yield chunk_index, start_record, batch
                chunk_index += 1
                batch = []
    if batch:
        yield chunk_index, start_record, batch


class StreamingState:
    def __init__(
        self,
        state_path: Path,
        *,
        exact_distinct: bool = False,
        batch_size: int = 100_000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(state_path)
        self.exact_distinct = exact_distinct
        self.batch_size = batch_size
        self.connection.executescript(SCHEMA)
        self.connection.commit()
        self._validate_or_set_config()

    def _validate_or_set_config(self) -> None:
        expected = {
            "batch_size": str(self.batch_size),
            "exact_distinct": str(self.exact_distinct),
        }
        existing = dict(self.connection.execute("SELECT key, value FROM run_config"))
        has_chunks = self.connection.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
        if has_chunks and not existing:
            raise ValueError(
                "State database predates run configuration; use a new state path "
                "or migrate it before resuming."
            )
        if existing and existing != expected:
            raise ValueError(
                f"State configuration {existing} does not match requested {expected}."
            )
        if not existing:
            self.connection.executemany(
                "INSERT INTO run_config VALUES (?, ?)", expected.items()
            )
            self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def completed_chunks(self, shard: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT chunk_id FROM chunks WHERE shard = ?", (shard,)
        )
        return {row[0] for row in rows}

    def commit_chunk(
        self,
        shard: str,
        chunk_index: int,
        start_record: int,
        records: list[dict],
    ) -> None:
        chunk_id = f"{shard}:{chunk_index:08d}"
        if self.connection.execute(
            "SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone():
            return

        status_counts: Counter[str] = Counter()
        mime_counts: Counter[str] = Counter()
        extension_counts: Counter[str] = Counter()
        extension_mime_counts: Counter[tuple[str, str]] = Counter()
        format_counts: Counter[str] = Counter()
        domain_counts: Counter[tuple[str, str]] = Counter()
        mime_lengths: Counter[str] = Counter()
        extension_lengths: Counter[str] = Counter()

        self.connection.execute("BEGIN")
        try:
            for record in records:
                url = record.get("url")
                digest = (record.get("digest") or "").strip()
                mime = (record.get("mime") or "unknown").strip().lower() or "unknown"
                extension = extract_url_extension(url)
                category = summarize_category(extension)
                status = (record.get("status") or "unknown").strip() or "unknown"
                length = _safe_int(record.get("length"))
                hostname = _hostname(url)

                status_counts[status] += 1
                mime_counts[mime] += 1
                mime_lengths[mime] += length
                extension_counts[extension] += 1
                extension_lengths[extension] += length
                extension_mime_counts[(extension, mime)] += 1
                format_counts[category] += 1
                domain_counts[(hostname, category)] += 1

                if url and self.exact_distinct:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO seen_urls VALUES (?)", (url,)
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO mime_urls VALUES (?, ?)", (mime, url)
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO extension_urls VALUES (?, ?)",
                        (extension, url),
                    )
                if digest and self.exact_distinct:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO seen_digests VALUES (?)", (digest,)
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO mime_digests VALUES (?, ?)",
                        (mime, digest),
                    )
                    self.connection.execute(
                        "INSERT OR IGNORE INTO extension_digests VALUES (?, ?)",
                        (extension, digest),
                    )

            for status, count in status_counts.items():
                self.connection.execute(
                    "INSERT INTO status_counts VALUES (?, ?) "
                    "ON CONFLICT(status) DO UPDATE SET capture_count = "
                    "capture_count + excluded.capture_count",
                    (status, count),
                )
            for mime, count in mime_counts.items():
                self.connection.execute(
                    "INSERT INTO mime_counts VALUES (?, ?, ?) ON CONFLICT(mime) DO UPDATE SET capture_count = capture_count + excluded.capture_count, total_length = total_length + excluded.total_length",
                    (mime, count, mime_lengths[mime]),
                )
            for extension, count in extension_counts.items():
                self.connection.execute(
                    "INSERT INTO extension_counts VALUES (?, ?, ?) ON CONFLICT(extension) DO UPDATE SET capture_count = capture_count + excluded.capture_count, total_length = total_length + excluded.total_length",
                    (extension, count, extension_lengths[extension]),
                )
            for (extension, mime), count in extension_mime_counts.items():
                self.connection.execute(
                    "INSERT INTO extension_mime_counts VALUES (?, ?, ?) ON CONFLICT(extension, mime) DO UPDATE SET capture_count = capture_count + excluded.capture_count",
                    (extension, mime, count),
                )
            for category, count in format_counts.items():
                self.connection.execute(
                    "INSERT INTO format_counts VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET capture_count = capture_count + excluded.capture_count",
                    (category, count),
                )
            for (hostname, category), count in domain_counts.items():
                self.connection.execute(
                    "INSERT INTO domain_category_counts VALUES (?, ?, ?) ON CONFLICT(hostname, category) DO UPDATE SET capture_count = capture_count + excluded.capture_count",
                    (hostname, category, count),
                )

            total_length = sum(_safe_int(record.get("length")) for record in records)
            totals = {
                "scanned_records": len(records),
                "total_length": total_length,
                "successful_captures": sum(
                    1
                    for record in records
                    if str(record.get("status") or "").strip() == "200"
                ),
            }
            for key, value in totals.items():
                self.connection.execute(
                    "INSERT INTO totals VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = value + excluded.value",
                    (key, value),
                )
            end_record = start_record + len(records)
            self.connection.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                (
                    chunk_id,
                    shard,
                    start_record,
                    end_record,
                    len(records),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise


def export_core_csvs(
    state_path: Path, output_dir: Path, *, exact_distinct: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_path)
    try:
        total_length = connection.execute(
            "SELECT COALESCE(value, 0) FROM totals WHERE key = 'total_length'"
        ).fetchone()[0]
        successful = connection.execute(
            "SELECT COALESCE(value, 0) FROM totals WHERE key = 'successful_captures'"
        ).fetchone()[0]
        mime_url_counts = dict(
            connection.execute("SELECT mime, COUNT(*) FROM mime_urls GROUP BY mime")
        )
        mime_digest_counts = dict(
            connection.execute("SELECT mime, COUNT(*) FROM mime_digests GROUP BY mime")
        )
        extension_url_counts = dict(
            connection.execute(
                "SELECT extension, COUNT(*) FROM extension_urls GROUP BY extension"
            )
        )
        extension_digest_counts = dict(
            connection.execute(
                "SELECT extension, COUNT(*) FROM extension_digests GROUP BY extension"
            )
        )
        queries = {
            "status_counts.csv": ("status, capture_count", "status_counts"),
            "format_rollup.csv": ("category, capture_count", "format_counts"),
            "extension_mime.csv": (
                "extension, mime, capture_count",
                "extension_mime_counts",
            ),
            "domains_by_format.csv": (
                "hostname, category, capture_count",
                "domain_category_counts",
            ),
        }
        for filename, (fields, table) in queries.items():
            fieldnames = [field.strip() for field in fields.split(",")]
            rows = connection.execute(f"SELECT {fields} FROM {table}")
            with (output_dir / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(fieldnames)
                writer.writerows(rows)
        for filename, table, key_name, url_counts, digest_counts in (
            (
                "mime_types.csv",
                "mime_counts",
                "mime",
                mime_url_counts,
                mime_digest_counts,
            ),
            (
                "extensions.csv",
                "extension_counts",
                "extension",
                extension_url_counts,
                extension_digest_counts,
            ),
        ):
            rows = connection.execute(
                f"SELECT {key_name}, capture_count, total_length FROM {table} "
                "ORDER BY capture_count DESC, total_length DESC, 1"
            )
            with (output_dir / filename).open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        key_name,
                        "capture_count",
                        "share_of_successful",
                        "distinct_urls",
                        "distinct_digests",
                        "total_length",
                        "share_of_total_length",
                    ]
                )
                for key, count, length in rows:
                    writer.writerow(
                        [
                            key,
                            count,
                            count / successful if successful else 0.0,
                            url_counts.get(key, "") if exact_distinct else "",
                            digest_counts.get(key, "") if exact_distinct else "",
                            length,
                            length / total_length if total_length else 0.0,
                        ]
                    )
        with (output_dir / "duplication.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "extension",
                    "captures",
                    "distinct_urls",
                    "distinct_digests",
                    "captures_per_url",
                    "captures_per_digest",
                    "duplicate_capture_fraction",
                ]
            )
            for extension, captures, _total_length in connection.execute(
                "SELECT extension, capture_count, total_length FROM extension_counts "
                "ORDER BY capture_count DESC, total_length DESC, extension"
            ):
                distinct_urls = extension_url_counts.get(extension, 0)
                distinct_digests = extension_digest_counts.get(extension, 0)
                writer.writerow(
                    [
                        extension,
                        captures,
                        distinct_urls if exact_distinct else "",
                        distinct_digests if exact_distinct else "",
                        captures / distinct_urls if distinct_urls else "",
                        captures / distinct_digests if distinct_digests else "",
                        (
                            1.0 - (distinct_digests / captures)
                            if exact_distinct and captures
                            else ""
                        ),
                    ]
                )
        metadata = {
            "state_path": str(state_path),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "totals": dict(connection.execute("SELECT key, value FROM totals")),
            "exact_distinct_tracking": exact_distinct,
            "distinct_metrics_note": (
                "Exact URL and digest metrics are included."
                if exact_distinct
                else "Distinct URL and digest metrics are not tracked in compact mode."
            ),
            "distinct_urls": (
                connection.execute("SELECT COUNT(*) FROM seen_urls").fetchone()[0]
                if exact_distinct
                else None
            ),
            "distinct_digests": (
                connection.execute("SELECT COUNT(*) FROM seen_digests").fetchone()[0]
                if exact_distinct
                else None
            ),
            "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        connection.close()


def run_shard(
    input_path: Path,
    state_path: Path,
    batch_size: int,
    *,
    exact_distinct: bool = False,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    state = StreamingState(
        state_path,
        exact_distinct=exact_distinct,
        batch_size=batch_size,
    )
    shard = input_path.name
    completed = state.completed_chunks(shard)
    try:
        for chunk_index, start_record, records in iter_cdxj_batches(
            input_path, batch_size
        ):
            chunk_id = f"{shard}:{chunk_index:08d}"
            if chunk_id in completed:
                continue
            state.commit_chunk(shard, chunk_index, start_record, records)
            logger.info("Committed %s records=%d", chunk_id, len(records))
    finally:
        state.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream raw EOT CDXJ data into a resumable SQLite state database."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--exact-distinct",
        action="store_true",
        help="Track exact URL/digest distinctness; substantially increases disk use.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_shard(
        args.input,
        args.state,
        args.batch_size,
        exact_distinct=args.exact_distinct,
    )
    if args.output_dir:
        export_core_csvs(
            args.state,
            args.output_dir,
            exact_distinct=args.exact_distinct,
        )


if __name__ == "__main__":
    main()
