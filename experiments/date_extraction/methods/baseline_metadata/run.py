# AI modified: 2026-07-13 9572ec45
"""Baseline A: the embedded PDF CreationDate metadata field.

This is what govscape currently records — for scanned documents it is the
scan/digitization date, which is exactly the failure mode this experiment
measures.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import (  # noqa: E402
    load_manifest,
    parse_pdf_creation_date,
    result_row,
    write_results,
)


def main():
    rows = []
    for record in load_manifest():
        start = time.perf_counter()
        raw = record["embedded_creation_date"]
        date = parse_pdf_creation_date(raw)
        rows.append(
            result_row(
                record["digest"],
                date,
                evidence=f"PDF metadata CreationDate: {raw}",
                wall_ms=(time.perf_counter() - start) * 1000,
            )
        )
    out = write_results("baseline_metadata", rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
