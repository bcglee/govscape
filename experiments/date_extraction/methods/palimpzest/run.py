# AI modified: 2026-07-13 9572ec45
"""Palimpzest (MIT, CIDR'25): sem_map over the shared document excerpts with
the single local vLLM model registered as the only available model.

Run inside the palimpzest venv:
  /home/ubuntu/venvs/palimpzest/bin/python methods/palimpzest/run.py
"""

import argparse
import json
import os
import sys
import time
import urllib.request

import palimpzest as pz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import (  # noqa: E402
    doc_excerpt,
    load_manifest,
    result_row,
    write_results,
)

BASE_URL = "http://127.0.0.1:8000/v1"


def served_model_id() -> str:
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_name = served_model_id()
    model = pz.Model(f"hosted_vllm/{model_name}", api_base=BASE_URL)
    print(f"model: {model_name}")

    records = load_manifest()
    if args.limit:
        records = records[: args.limit]
    vals = [{"digest": r["digest"], "text": doc_excerpt(r)} for r in records]

    dataset = pz.MemoryDataset(id="gov-pdfs", vals=vals)
    dataset = dataset.sem_map(
        [
            {
                "name": "creation_date",
                "type": str,
                "desc": (
                    "The date this document was actually written or created "
                    "— NOT the date it was scanned, digitized, published "
                    "online, or crawled. For a letter, the date the letter "
                    "was written; for a report, the date on the report "
                    "itself. Format as YYYY-MM-DD, YYYY-MM, or YYYY "
                    "depending on how precisely the document states it, or "
                    "'N/A' if the document does not reveal when it was "
                    "written."
                ),
            },
            {
                "name": "evidence",
                "type": str,
                "desc": "Short quote from the document supporting the date.",
            },
        ]
    )

    config = pz.QueryProcessorConfig(
        available_models=[model],
        policy=pz.MaxQuality(),
        progress=True,
    )
    start = time.perf_counter()
    output = dataset.run(config)
    wall_ms = (time.perf_counter() - start) * 1000

    df = output.to_df()
    by_digest = {row["digest"]: row for _, row in df.iterrows() if "digest" in row}
    rows = []
    for record in records:
        out = by_digest.get(record["digest"])
        rows.append(
            result_row(
                record["digest"],
                None if out is None else str(out.get("creation_date", "")),
                evidence="" if out is None else str(out.get("evidence", "")),
                wall_ms=wall_ms / len(records),
            )
        )
    out_path = write_results("palimpzest", rows)
    stats = getattr(output, "execution_stats", None)
    if stats is not None:
        cost = getattr(stats, "total_execution_cost", None)
        print(f"execution cost: {cost}")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
