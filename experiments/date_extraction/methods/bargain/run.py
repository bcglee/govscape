# AI modified: 2026-07-13 9572ec45
"""BARGAIN (Zeighami et al., PACMMOD'26) on local models.

Proxy and oracle share the served model (only one fits the T4 at a time), so
this is the task-cascade flavor of BARGAIN: the cheap proxy reads a short
page-1 prefix while the expensive oracle reads the full multi-page excerpt.
BARGAIN_A picks the proxy-confidence threshold that guarantees the output
matches the oracle on >= target fraction of records (probability 1 - delta).

Run inside the bargain venv:
  /home/ubuntu/venvs/bargain/bin/python methods/bargain/run.py
"""

import argparse
import os
import sys
import time

import numpy as np

from BARGAIN import BARGAIN_A, Oracle, Proxy
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common import (  # noqa: E402
    EXTRACTION_PROMPT,
    load_manifest,
    normalize_date,
    page_text,
    parse_extraction_json,
    result_row,
    write_results,
)

BASE_URL = "http://127.0.0.1:8000/v1"
PROXY_CHARS = 1200
ORACLE_MAX_CHARS = 20000


def oracle_excerpt(record: dict) -> str:
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
    return "\n\n".join(parts)[:ORACLE_MAX_CHARS]


class LocalProxy(Proxy):
    def __init__(self, client: OpenAI, model: str):
        super().__init__(verbose=True, max_workers=8)
        self.client = client
        self.model = model
        self.tokens_in = 0
        self.tokens_out = 0

    def proxy_func(self, input: str):
        prompt = EXTRACTION_PROMPT.format(text=input[:PROXY_CHARS])
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
            logprobs=True,
        )
        self.tokens_in += resp.usage.prompt_tokens
        self.tokens_out += resp.usage.completion_tokens
        choice = resp.choices[0]
        date, _ = parse_extraction_json(choice.message.content)
        logprobs = choice.logprobs.content or []
        score = (
            float(np.exp(np.mean([t.logprob for t in logprobs]))) if logprobs else 0.0
        )
        return normalize_date(date), score


class LocalOracle(Oracle):
    def __init__(self, client: OpenAI, model: str):
        super().__init__(verbose=True, max_workers=8)
        self.client = client
        self.model = model
        self.tokens_in = 0
        self.tokens_out = 0

    def oracle_func(self, input: str, proxy_output):
        prompt = EXTRACTION_PROMPT.format(text=input)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        self.tokens_in += resp.usage.prompt_tokens
        self.tokens_out += resp.usage.completion_tokens
        date, _ = parse_extraction_json(resp.choices[0].message.content)
        oracle_date = normalize_date(date)
        return oracle_date == proxy_output, oracle_date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    model = client.models.list().data[0].id
    print(f"model: {model}")

    records = load_manifest()
    if args.limit:
        records = records[: args.limit]
    digests = [r["digest"] for r in records]
    data_records = [oracle_excerpt(r) for r in records]

    proxy = LocalProxy(client, model)
    oracle = LocalOracle(client, model)
    bargain = BARGAIN_A(proxy, oracle, target=args.target, delta=args.delta)

    start = time.perf_counter()
    outputs, oracle_used = bargain.process(data_records, return_oracle_usage=True)
    wall_ms = (time.perf_counter() - start) * 1000

    rows = [
        result_row(
            digest,
            output,
            evidence="oracle" if used else "proxy",
            wall_ms=wall_ms / len(digests),
        )
        for digest, output, used in zip(digests, outputs, oracle_used, strict=True)
    ]
    tokens_in = proxy.tokens_in + oracle.tokens_in
    tokens_out = proxy.tokens_out + oracle.tokens_out
    for row in rows:
        row["tokens_in"] = tokens_in // len(rows)
        row["tokens_out"] = tokens_out // len(rows)

    out = write_results("bargain", rows)
    n_oracle = int(sum(oracle_used))
    print(
        f"Wrote {len(rows)} rows to {out}\n"
        f"oracle used on {n_oracle}/{len(rows)} records "
        f"({100 * n_oracle / len(rows):.0f}%)\n"
        f"tokens: proxy {proxy.tokens_in}+{proxy.tokens_out}, "
        f"oracle {oracle.tokens_in}+{oracle.tokens_out}, "
        f"wall {wall_ms / 1000:.0f}s"
    )


if __name__ == "__main__":
    main()
