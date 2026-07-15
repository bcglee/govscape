# AI modified: 2026-07-13 9572ec45
# AI modified: 2026-07-14 8d36a86c
"""Baseline C: direct chat completion against the local endpoint on the
page-1 + last-page excerpt. The reference point the frameworks wrap.

--thinking on|off targets Qwen3.5-style hybrid reasoning models: "on" lets the
model emit its <think> block first (needs the server started with
REASONING_PARSER=qwen3 so the final answer arrives in message.content), "off"
disables it via chat_template_kwargs. Sampling follows the Qwen3.5 model card
per mode; the default (no flag) keeps the original greedy behavior used for
the Qwen2.5 runs.
"""

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

# Qwen3.5 model-card recommended sampling (greedy decoding is advised against)
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 1.5}
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}


def request_config(thinking: str | None) -> tuple[float, int, dict, bool]:
    """Returns (temperature, max_tokens, extra_body, json_mode) per mode.

    Thinking mode must NOT use guided JSON: vLLM applies the grammar from the
    first token, which suppresses the think block entirely (verified on
    Qwen3.5-0.8B) — the answer is parsed from free-form output instead.
    """
    if thinking == "on":
        extra = dict(THINKING_SAMPLING)
        return extra.pop("temperature"), 4096, extra, False
    if thinking == "off":
        extra = dict(INSTRUCT_SAMPLING)
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        return extra.pop("temperature"), 300, extra, True
    return 0.0, 300, {}, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--method_name",
        default="baseline_llm",
        help="results/<method_name>.jsonl (used by the model-size sweep)",
    )
    parser.add_argument(
        "--thinking",
        choices=["on", "off"],
        default=None,
        help="Hybrid-reasoning toggle for Qwen3.5-style models",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None, help="Override the mode default"
    )
    parser.add_argument(
        "--digests_file",
        default=None,
        help="Only process digests listed in this file (one per line)",
    )
    args = parser.parse_args()

    model = served_model()
    print(f"model: {model}  thinking: {args.thinking or 'n/a'}")
    records = load_manifest()
    if args.digests_file:
        with open(args.digests_file, encoding="utf-8") as f:
            wanted = {line.strip() for line in f if line.strip()}
        records = [r for r in records if r["digest"] in wanted]
    if args.limit:
        records = records[: args.limit]

    temperature, max_tokens, extra_body, json_mode = request_config(args.thinking)
    if args.max_tokens:
        max_tokens = args.max_tokens

    rows = []
    for i, record in enumerate(records):
        start = time.perf_counter()
        prompt = EXTRACTION_PROMPT.format(text=doc_excerpt(record))
        try:
            resp = chat_completion(
                [{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                extra_body=extra_body or None,
            )
            msg = resp["choices"][0]["message"]
            # Reasoning models that never emit </think> leave content null
            # and the whole response in the parser's "reasoning" field
            content = msg.get("content") or msg.get("reasoning") or ""
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
