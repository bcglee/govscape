# AI modified: 2026-07-13 9572ec45
"""Smoke-test the local OpenAI-compatible endpoint: completion + logprobs."""

import argparse
import json
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.base_url}/models", timeout=10) as r:
        models = json.load(r)["data"]
    model = models[0]["id"]
    print(f"model: {model}")

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly one word: what is the capital "
                "of France?",
            }
        ],
        "max_tokens": 10,
        "temperature": 0,
        "logprobs": True,
    }
    req = urllib.request.Request(
        f"{args.base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)

    choice = resp["choices"][0]
    content = choice["message"]["content"].strip()
    logprobs = choice.get("logprobs", {}).get("content")
    print(f"completion: {content!r}")
    print(f"logprobs present: {bool(logprobs)}")
    if "paris" not in content.lower():
        print("FAIL: unexpected completion")
        sys.exit(1)
    if not logprobs:
        print("FAIL: no logprobs (BARGAIN needs them)")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
