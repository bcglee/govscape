import gzip
from pathlib import Path

from scripts.analyze_eot_file_types import (
    classify_record,
    extract_url_extension,
    normalize_mime,
    parse_cdxj_line,
    query_cdx_rows,
    summarize_category,
)


def test_normalize_mime_removes_parameters_and_downcases() -> None:
    assert normalize_mime("Text/CSV; charset=UTF-8") == "text/csv"
    assert normalize_mime("application/octet-stream") == "application/octet-stream"
    assert normalize_mime(None) == "unknown"
    assert normalize_mime("") == "unknown"


def test_extract_url_extension_handles_common_cases() -> None:
    assert (
        extract_url_extension("https://example.gov/reports/report.PDF?download=true")
        == "pdf"
    )
    assert extract_url_extension("https://example.gov/data/table.csv#preview") == "csv"
    assert extract_url_extension("https://example.gov/foo/archive.tar.gz") == "tar.gz"
    assert extract_url_extension("https://example.gov/foo/bar") == "unknown"
    assert extract_url_extension("https://example.gov/download?id=123") == "unknown"
    assert (
        extract_url_extension("https://example.gov/directory.with.dot/file")
        == "unknown"
    )
    assert extract_url_extension("https://example.gov/file.") == "unknown"


def test_classify_record_detects_mime_extension_mismatches() -> None:
    info = classify_record(
        url="https://example.gov/file.csv",
        mime="text/plain",
        status="200",
        digest="abc",
    )
    assert info["extension"] == "csv"
    assert info["mime"] == "text/plain"
    assert info["mime_category"] == "text"
    assert info["category"] == "data"
    assert info["extension_matches_mime"] is False


def test_summarize_category_groups_formats() -> None:
    assert summarize_category("pdf") == "document"
    assert summarize_category("csv") == "data"
    assert summarize_category("pptx") == "presentation"
    assert summarize_category("mp3") == "audio"
    assert summarize_category("mp4") == "video"
    assert summarize_category("html") == "web"
    assert summarize_category("unknown") == "unknown"


def test_parse_cdxj_line_extracts_json_payload() -> None:
    raw = (
        '1,0,0,1)/ 20240927134430 {"url": "https://example.gov/report.pdf", '
        '"mime": "application/pdf", "status": "200", '
        '"digest": "sha1:abc123", "length": "1234", '
        '"offset": "987", "filename": "warc.gz"}'
    )
    record = parse_cdxj_line(raw)
    assert record is not None
    assert record["url"] == "https://example.gov/report.pdf"
    assert record["mime"] == "application/pdf"
    assert record["status"] == "200"
    assert record["digest"] == "sha1:abc123"
    assert record["length"] == "1234"
    assert record["offset"] == "987"
    assert record["filename"] == "warc.gz"


def test_query_cdx_rows_handles_raw_cdxj_gzip(tmp_path: Path) -> None:
    path = tmp_path / "sample.cdxj.gz"
    payload = (
        '1,0,0,1)/ 20240927134430 {"url": "https://example.gov/report.pdf", '
        '"mime": "application/pdf", "status": "200", '
        '"digest": "sha1:abc123", "length": "1234", '
        '"offset": "987", "filename": "warc.gz"}\n'
        '1,0,0,1)/ 20241217000826 {"url": "https://example.gov/data.csv", '
        '"mime": "text/csv", "status": "404", '
        '"digest": "sha1:def456", "length": "4321", '
        '"offset": "432", "filename": "other.warc.gz"}\n'
    )
    with gzip.open(path, "wb") as handle:
        handle.write(payload.encode("utf-8"))

    rows = query_cdx_rows(str(path), max_rows=10)
    assert len(rows) == 2
    assert rows[0]["url"] == "https://example.gov/report.pdf"
    assert rows[1]["mime"] == "text/csv"
    assert rows[1]["status"] == "404"
