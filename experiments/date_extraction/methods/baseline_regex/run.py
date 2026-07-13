# AI modified: 2026-07-13 9572ec45
"""Baseline B: regex scan for explicit dates in page-1 (+ last page) text.

Takes the earliest-appearing full date on page 1; falls back to month-year,
then to the last page. Zero model cost. This also captures the
cheap-strategy-first idea from Doctopus.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import (  # noqa: E402
    load_manifest,
    normalize_date,
    page_text,
    result_row,
    write_results,
)

MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)
FULL_DATE_PATTERNS = [
    rf"{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}",
    rf"\d{{1,2}}\s+{MONTH}\.?\s+(?:19|20)\d{{2}}",
    r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}",
    r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}",
]
MONTH_YEAR_PATTERNS = [
    rf"{MONTH}\.?,?\s+(?:19|20)\d{{2}}",
    r"\d{1,2}/(?:19|20)\d{2}",
]


def find_date(text: str) -> tuple[str, str] | None:
    """Return (raw_match, context) for the best date in text, or None."""
    for patterns in (FULL_DATE_PATTERNS, MONTH_YEAR_PATTERNS):
        best = None
        for pattern in patterns:
            m = re.search(pattern, text)
            if m and (best is None or m.start() < best.start()):
                best = m
        if best:
            ctx = text[max(0, best.start() - 40) : best.end() + 40]
            return best.group(0), ctx.replace("\n", " ")
    return None


def main():
    rows = []
    for record in load_manifest():
        start = time.perf_counter()
        digest = record["digest"]
        found = find_date(page_text(digest, 0))
        if not found and record["num_pages"] > 1:
            found = find_date(page_text(digest, record["num_pages"] - 1))
        raw, evidence = found if found else (None, "no date pattern matched")
        if raw and normalize_date(raw) == "N/A":
            raw, evidence = None, f"unnormalizable match: {raw}"
        rows.append(
            result_row(
                digest,
                raw,
                evidence=evidence,
                wall_ms=(time.perf_counter() - start) * 1000,
            )
        )
    out = write_results("baseline_regex", rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
