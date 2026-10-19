from pathlib import Path

from scripts.eot_batch_analysis import build_output_dir, parse_s5cmd_listing


def test_parse_s5cmd_listing_filters_summary_and_keeps_archive_files() -> None:
    listing = """2026/04/16 16:56:33 104608068 cluster.summary
2026/04/16 16:56:33 3320756784 eot24-cdx-00000.cdxj.gz
2026/04/16 16:56:33 11120029006 eot24-cdx-00001.cdxj.gz
"""

    assert parse_s5cmd_listing(listing) == [
        "eot24-cdx-00000.cdxj.gz",
        "eot24-cdx-00001.cdxj.gz",
    ]


def test_build_output_dir_uses_archive_basename_without_suffix() -> None:
    root = Path("/tmp/eot-output-root")
    out_dir = build_output_dir(root, Path("/tmp/eot-full-safe/eot24-cdx-00000.cdxj.gz"))

    assert out_dir == root / "eot24-cdx-00000"
