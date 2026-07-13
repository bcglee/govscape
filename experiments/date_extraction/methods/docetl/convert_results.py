# AI modified: 2026-07-13 9572ec45
"""Convert DocETL output to the common results format."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import result_row, write_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall_ms_total", type=float, default=0)
    args = parser.parse_args()

    out_path = os.path.join(os.path.dirname(__file__), "docetl_output.json")
    with open(out_path, encoding="utf-8") as f:
        outputs = json.load(f)

    rows = [
        result_row(
            o["digest"],
            o.get("creation_date"),
            evidence=str(o.get("evidence", "")),
            wall_ms=args.wall_ms_total / max(1, len(outputs)),
        )
        for o in outputs
    ]
    out = write_results("docetl", rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
