# AI modified: 2026-07-13 9572ec45
"""Shared helpers for date-extraction methods: manifest access, page text,
date normalization, and the common results format."""

import json
import os
import re
import urllib.request

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(EXPERIMENT_DIR, "corpus", "manifest.jsonl")
CORPUS_DATA_DIR = os.path.join(EXPERIMENT_DIR, "corpus", "data")
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")

MONTHS = {
    m.lower(): i + 1
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    )
}
MONTHS.update({m[:3]: v for m, v in list(MONTHS.items())})


def load_manifest(manifest_path: str = MANIFEST_PATH) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def page_text(digest: str, pg_no: int, data_dir: str = CORPUS_DATA_DIR) -> str:
    path = os.path.join(data_dir, "txt", digest, f"{digest}_{pg_no}.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def doc_excerpt(record: dict, max_chars: int = 6000) -> str:
    """Page-1 + last-page text, the truncated input all methods share."""
    first = page_text(record["digest"], 0)
    parts = [f"[PAGE 1]\n{first[: max_chars * 3 // 4]}"]
    last_pg = record["num_pages"] - 1
    if last_pg > 0:
        last = page_text(record["digest"], last_pg)
        if last.strip():
            parts.append(f"[LAST PAGE]\n{last[: max_chars // 4]}")
    return "\n\n".join(parts)[:max_chars]


def normalize_date(raw: str | None) -> str:
    """Normalize a date string to YYYY[-MM[-DD]] or 'N/A'."""
    if not raw:
        return "N/A"
    s = raw.strip().strip(".").strip()
    if not s or s.lower() in {"n/a", "na", "none", "unknown", "null", ""}:
        return "N/A"

    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return m.group(1)
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
    if m and m.group(1).lower()[:3] in MONTHS:
        month = MONTHS[m.group(1).lower()[:3]]
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})", s)
    if m and m.group(2).lower()[:3] in MONTHS:
        month = MONTHS[m.group(2).lower()[:3]]
        return f"{m.group(3)}-{month:02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"([A-Za-z]+)\.?,?\s+(\d{4})", s)
    if m and m.group(1).lower()[:3] in MONTHS:
        return f"{m.group(2)}-{MONTHS[m.group(1).lower()[:3]]:02d}"
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    return "N/A"


def parse_pdf_creation_date(raw: str) -> str:
    """Parse a PDF metadata date like D:20150612143019-04'00' to YYYY-MM-DD."""
    m = re.match(r"D:(\d{4})(\d{2})?(\d{2})?", raw or "")
    if not m:
        return "N/A"
    parts = [p for p in m.groups() if p]
    return "-".join(parts)


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

EXTRACTION_PROMPT = """\
Below is text from a U.S. government PDF. Determine the date the document \
was actually written or created — NOT the date it was scanned, digitized, \
published online, or crawled. For a letter this is the date the letter was \
written; for a report, the date on the report itself.

Answer in JSON: {{"creation_date": "<date>", "evidence": "<short quote>"}}
Use YYYY-MM-DD, YYYY-MM, or YYYY depending on how precisely the document \
states it. If the document does not reveal when it was written, use "N/A".

Document text:
{text}"""


def served_model(base_url: str = DEFAULT_BASE_URL) -> str:
    with urllib.request.urlopen(f"{base_url}/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def chat_completion(
    messages: list[dict],
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    max_tokens: int = 200,
    temperature: float = 0.0,
    logprobs: bool = False,
    json_mode: bool = False,
    extra_body: dict | None = None,
) -> dict:
    """Call the local OpenAI-compatible endpoint; returns the raw response.

    json_mode uses vLLM guided decoding to force valid JSON output — small
    models otherwise ramble a preamble and hit max_tokens before any JSON.
    extra_body merges extra request fields (e.g. top_p, presence_penalty,
    chat_template_kwargs={"enable_thinking": False} for Qwen3.5).
    """
    body = {
        "model": model or served_model(base_url),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if logprobs:
        body["logprobs"] = True
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def vllm_token_counters(base_url: str = DEFAULT_BASE_URL) -> tuple[int, int]:
    """Read vLLM's cumulative (prompt, generation) token counters.

    Snapshot before/after a run to attribute tokens to frameworks that don't
    expose usage themselves.
    """
    url = base_url.removesuffix("/v1") + "/metrics"
    with urllib.request.urlopen(url, timeout=10) as r:
        text = r.read().decode()
    totals = {"prompt": 0, "generation": 0}
    for line in text.splitlines():
        for kind in totals:
            if line.startswith(f"vllm:{kind}_tokens_total"):
                totals[kind] += int(float(line.split()[-1]))
    return totals["prompt"], totals["generation"]


def parse_extraction_json(content: str) -> tuple[str | None, str]:
    """Parse {"creation_date": ..., "evidence": ...} from model output.

    Reasoning models may bury the JSON in surrounding prose — prefer the last
    well-formed object that mentions creation_date before falling back to the
    outermost brace span.
    """
    candidates = re.findall(r'\{[^{}]*"creation_date"[^{}]*\}', content, re.DOTALL)
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        return data.get("creation_date"), str(data.get("evidence", ""))
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None, f"unparseable output: {content[:100]}"
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, f"invalid JSON: {content[:100]}"
    return data.get("creation_date"), str(data.get("evidence", ""))


def write_results(method: str, rows: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{method}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out_path


def result_row(
    digest: str,
    predicted_date: str | None,
    evidence: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    wall_ms: float = 0,
) -> dict:
    return {
        "digest": digest,
        "predicted_date": predicted_date,
        "normalized_date": normalize_date(predicted_date),
        "evidence": evidence,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_ms": round(wall_ms, 1),
    }
