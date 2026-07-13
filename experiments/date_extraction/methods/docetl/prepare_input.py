# AI modified: 2026-07-13 9572ec45
"""Build docs.json (digest + shared excerpt) for the DocETL pipeline."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import doc_excerpt, load_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = load_manifest()
    if args.limit:
        records = records[: args.limit]
    docs = [{"digest": r["digest"], "text": doc_excerpt(r)} for r in records]
    out_path = os.path.join(os.path.dirname(__file__), "docs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f)
    print(f"Wrote {len(docs)} docs to {out_path}")


if __name__ == "__main__":
    main()
