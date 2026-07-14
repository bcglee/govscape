# AI modified: 2026-07-13 9572ec45
"""Score method results against eval/labels.jsonl.

Metrics per method:
- date accuracy on dated docs (label != N/A), at year / year-month / label
  granularity ("exact" truncates the prediction to the label's precision)
- N/A precision and recall (the label says the doc doesn't reveal its date)
- coverage and cost (tokens, wall time)
- breakdown for digital-text vs OCR'd-scan documents

Writes results/report.md and results/report.csv.
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import EXPERIMENT_DIR, RESULTS_DIR, load_manifest  # noqa: E402

LABELS_PATH = os.path.join(EXPERIMENT_DIR, "eval", "labels.jsonl")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def truncate(date: str, granularity: str) -> str:
    if date == "N/A":
        return date
    return date[: {"year": 4, "month": 7, "day": 10}[granularity]]


def score_method(rows: list[dict], labels: dict, scanned: set) -> dict:
    stats = {
        "n": 0,
        "dated_n": 0,
        "year_hits": 0,
        "ym_n": 0,
        "ym_hits": 0,
        "exact_hits": 0,
        "na_pred": 0,
        "na_true": 0,
        "na_both": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "wall_ms": 0.0,
        "scan_dated_n": 0,
        "scan_year_hits": 0,
        "dig_dated_n": 0,
        "dig_year_hits": 0,
    }
    for row in rows:
        label = labels.get(row["digest"])
        if label is None:
            continue
        stats["n"] += 1
        stats["tokens_in"] += row.get("tokens_in", 0)
        stats["tokens_out"] += row.get("tokens_out", 0)
        stats["wall_ms"] += row.get("wall_ms", 0)
        pred = row["normalized_date"]
        gold = label["date"]
        if pred == "N/A":
            stats["na_pred"] += 1
        if gold == "N/A":
            stats["na_true"] += 1
            if pred == "N/A":
                stats["na_both"] += 1
            continue
        stats["dated_n"] += 1
        is_scan = row["digest"] in scanned
        year_hit = pred != "N/A" and pred[:4] == gold[:4]
        stats["year_hits"] += year_hit
        stats["scan_dated_n" if is_scan else "dig_dated_n"] += 1
        stats["scan_year_hits" if is_scan else "dig_year_hits"] += year_hit
        if label["granularity"] in ("month", "day"):
            stats["ym_n"] += 1
            stats["ym_hits"] += pred != "N/A" and pred[:7] == gold[:7]
        stats["exact_hits"] += truncate(pred, label["granularity"]) == gold
    return stats


def ratio(num: int, den: int) -> str:
    return f"{num / den:.2f}" if den else "-"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=LABELS_PATH)
    parser.add_argument(
        "--gold",
        default=os.path.join(EXPERIMENT_DIR, "eval", "labels_gold.jsonl"),
        help="Overlay these gold labels onto --labels by digest (if present)",
    )
    args = parser.parse_args()

    labels = {r["digest"]: r for r in load_jsonl(args.labels)}
    gold = {r["digest"]: r for r in load_jsonl(args.gold)}
    labels.update(gold)  # gold takes precedence per digest
    if gold:
        print(f"Overlaid {len(gold)} gold labels onto {len(labels)} total\n")
    scanned = {
        r["digest"] for r in load_manifest() if r["ocr_pages"] or r["empty_pages"]
    }

    header = [
        "method",
        "n",
        "year_acc",
        "ym_acc",
        "exact_acc",
        "na_precision",
        "na_recall",
        "year_acc_scan",
        "year_acc_digital",
        "tokens/doc",
        "ms/doc",
    ]
    table = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.jsonl"))):
        method = os.path.splitext(os.path.basename(path))[0]
        s = score_method(load_jsonl(path), labels, scanned)
        table.append(
            [
                method,
                s["n"],
                ratio(s["year_hits"], s["dated_n"]),
                ratio(s["ym_hits"], s["ym_n"]),
                ratio(s["exact_hits"], s["dated_n"]),
                ratio(s["na_both"], s["na_pred"]),
                ratio(s["na_both"], s["na_true"]),
                ratio(s["scan_year_hits"], s["scan_dated_n"]),
                ratio(s["dig_year_hits"], s["dig_dated_n"]),
                round((s["tokens_in"] + s["tokens_out"]) / max(1, s["n"])),
                round(s["wall_ms"] / max(1, s["n"])),
            ]
        )

    n_dated = sum(1 for v in labels.values() if v["date"] != "N/A")
    lines = [
        "# Date-extraction results",
        "",
        f"Corpus: {len(labels)} docs ({n_dated} dated, "
        f"{len(labels) - n_dated} N/A per labels; {len(scanned)} with "
        "scanned/OCR pages)",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS_DIR, "report.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)
    print(f"\nWrote {RESULTS_DIR}/report.md and report.csv")


if __name__ == "__main__":
    main()
