# AI modified: 2026-07-13 9572ec45
"""Generate silver ground-truth labels with the served model reading MORE of
the document than any method under test sees (first 5 pages + last 2 pages,
up to ~20k chars). Writes eval/labels.jsonl with provenance "silver"; a later
human pass can upgrade records to provenance "gold" in place.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import (  # noqa: E402
    EXPERIMENT_DIR,
    EXTRACTION_PROMPT,
    chat_completion,
    load_manifest,
    normalize_date,
    page_text,
    parse_extraction_json,
    served_model,
)

LABELS_PATH = os.path.join(EXPERIMENT_DIR, "eval", "labels.jsonl")
MAX_CHARS = 20000


def annotator_excerpt(record: dict) -> str:
    digest, num_pages = record["digest"], record["num_pages"]
    parts = []
    for pg in range(min(5, num_pages)):
        text = page_text(digest, pg).strip()
        if text:
            parts.append(f"[PAGE {pg + 1}]\n{text[:5000]}")
    for pg in range(max(5, num_pages - 2), num_pages):
        text = page_text(digest, pg).strip()
        if text:
            parts.append(f"[PAGE {pg + 1} (near end)]\n{text[:2500]}")
    return "\n\n".join(parts)[:MAX_CHARS]


def granularity(normalized: str) -> str:
    if normalized == "N/A":
        return "na"
    return {4: "year", 7: "month", 10: "day"}.get(len(normalized), "other")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = served_model()
    print(f"annotator model: {model}")
    records = load_manifest()
    if args.limit:
        records = records[: args.limit]

    with open(LABELS_PATH, "w", encoding="utf-8") as out:
        for i, record in enumerate(records):
            start = time.perf_counter()
            prompt = EXTRACTION_PROMPT.format(text=annotator_excerpt(record))
            try:
                resp = chat_completion(
                    [{"role": "user", "content": prompt}], model=model
                )
                content = resp["choices"][0]["message"]["content"]
                date, evidence = parse_extraction_json(content)
            except Exception as e:
                date, evidence = None, f"annotation failed: {e}"
            normalized = normalize_date(date)
            out.write(
                json.dumps(
                    {
                        "digest": record["digest"],
                        "date": normalized,
                        "granularity": granularity(normalized),
                        "evidence": evidence,
                        "provenance": "silver",
                        "annotator": model,
                        "wall_ms": round((time.perf_counter() - start) * 1000, 1),
                    }
                )
                + "\n"
            )
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(records)}")

    print(f"Wrote {len(records)} labels to {LABELS_PATH}")


if __name__ == "__main__":
    main()
