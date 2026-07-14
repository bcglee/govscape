# AI modified: 2026-07-13 9572ec45
"""Baseline C: direct chat completion against the local endpoint on the
page-1 + last-page excerpt. The reference point the frameworks wrap."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import (  # noqa: E402
    EXTRACTION_PROMPT,
    chat_completion,
    doc_excerpt,
    load_manifest,
    parse_extraction_json,
    result_row,
    served_model,
    write_results,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--method_name",
        default="baseline_llm",
        help="results/<method_name>.jsonl (used by the model-size sweep)",
    )
    args = parser.parse_args()

    model = served_model()
    print(f"model: {model}")
    records = load_manifest()
    if args.limit:
        records = records[: args.limit]

    rows = []
    for i, record in enumerate(records):
        start = time.perf_counter()
        prompt = EXTRACTION_PROMPT.format(text=doc_excerpt(record))
        try:
            resp = chat_completion(
                [{"role": "user", "content": prompt}],
                model=model,
                max_tokens=300,
                json_mode=True,
            )
            content = resp["choices"][0]["message"]["content"]
            date, evidence = parse_extraction_json(content)
            usage = resp.get("usage", {})
        except Exception as e:
            date, evidence, usage = None, f"request failed: {e}", {}
        rows.append(
            result_row(
                record["digest"],
                date,
                evidence=evidence,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                wall_ms=(time.perf_counter() - start) * 1000,
            )
        )
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(records)}")

    out = write_results(args.method_name, rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
