import gzip
import json
import sqlite3
from pathlib import Path

from scripts.eot_streaming_analysis import (
    export_core_csvs,
    iter_cdxj_batches,
    run_shard,
)
from scripts.run_eot_census import parse_s5cmd_listing, parse_s5cmd_sizes


def _write_sample(path: Path, count: int = 5) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for index in range(count):
            payload = {
                "url": f"https://example.gov/file-{index % 2}.pdf",
                "mime": "application/pdf",
                "status": "200" if index % 2 == 0 else "301",
                "digest": f"sha1:{index % 3}",
                "length": str(index + 1),
            }
            handle.write(f"1,0,0,1)/ 20240927134430 {json.dumps(payload)}\n")


def test_iter_cdxj_batches_has_bounded_ranges(tmp_path: Path) -> None:
    path = tmp_path / "sample.cdxj.gz"
    _write_sample(path, count=5)

    batches = list(iter_cdxj_batches(path, batch_size=2))

    assert [(index, start, len(rows)) for index, start, rows in batches] == [
        (0, 0, 2),
        (1, 2, 2),
        (2, 4, 1),
    ]


def test_parse_s5cmd_listing_ignores_non_cdx_files() -> None:
    listing = """2026/04/16 16:56:33 104608068 cluster.summary
2026/04/16 16:56:33 3320756784 eot24-cdx-00000.cdxj.gz
2026/04/16 16:56:33 11120029006 eot24-cdx-00001.cdxj.gz
"""

    assert parse_s5cmd_listing(listing) == [
        "eot24-cdx-00000.cdxj.gz",
        "eot24-cdx-00001.cdxj.gz",
    ]
    assert parse_s5cmd_sizes(listing) == {
        "eot24-cdx-00000.cdxj.gz": 3320756784,
        "eot24-cdx-00001.cdxj.gz": 11120029006,
    }


def test_invalid_batch_size_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sample.cdxj.gz"
    _write_sample(path, count=1)

    try:
        list(iter_cdxj_batches(path, batch_size=0))
    except ValueError as error:
        assert str(error) == "batch_size must be positive"
    else:
        raise AssertionError("zero batch size was accepted")


def test_run_shard_is_resumable_without_double_counting(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cdxj.gz"
    state_path = tmp_path / "state.sqlite"
    output_dir = tmp_path / "output"
    _write_sample(input_path, count=5)

    run_shard(input_path, state_path, batch_size=2, exact_distinct=True)
    run_shard(input_path, state_path, batch_size=2, exact_distinct=True)
    export_core_csvs(state_path, output_dir, exact_distinct=True)

    connection = sqlite3.connect(state_path)
    try:
        assert connection.execute(
            "SELECT value FROM totals WHERE key = 'scanned_records'"
        ).fetchone() == (5,)
        assert connection.execute(
            "SELECT value FROM totals WHERE key = 'successful_captures'"
        ).fetchone() == (3,)
        assert connection.execute(
            "SELECT capture_count FROM status_counts WHERE status = '200'"
        ).fetchone() == (3,)
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone() == (3,)
    finally:
        connection.close()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["distinct_urls"] == 2
    assert metadata["distinct_digests"] == 3


def test_resume_rejects_changed_batch_configuration(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cdxj.gz"
    state_path = tmp_path / "state.sqlite"
    _write_sample(input_path, count=5)

    run_shard(input_path, state_path, batch_size=2)

    try:
        run_shard(input_path, state_path, batch_size=3)
    except ValueError as error:
        assert "does not match requested" in str(error)
    else:
        raise AssertionError("changed batch size was accepted")


def test_resume_rejects_changed_distinct_configuration(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.cdxj.gz"
    state_path = tmp_path / "state.sqlite"
    _write_sample(input_path, count=5)

    run_shard(input_path, state_path, batch_size=2)

    try:
        run_shard(input_path, state_path, batch_size=2, exact_distinct=True)
    except ValueError as error:
        assert "does not match requested" in str(error)
    else:
        raise AssertionError("changed distinct mode was accepted")
