# AI modified: 2026-07-14 9572ec45
"""Local web app for human gold-labeling of document creation dates.

Renders each PDF's page images alongside candidate dates (embedded metadata,
regex, silver annotator, and every model result), and records the human's
final call to eval/labels_gold.jsonl (provenance "gold"). Non-destructive:
the silver labels.jsonl is never modified, and gold labeling is resumable.

Run on the server, then port-forward 5055 to view in a browser:
  poetry run python experiments/date_extraction/eval/label_app.py
  # from your laptop: ssh -L 5055:127.0.0.1:5055 <server>
"""

import argparse
import datetime
import glob
import json
import os
import sys

from flask import Flask, redirect, request, send_from_directory, url_for

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import (  # noqa: E402
    CORPUS_DATA_DIR,
    EXPERIMENT_DIR,
    RESULTS_DIR,
    load_manifest,
    normalize_date,
    parse_pdf_creation_date,
)

GOLD_PATH = os.path.join(EXPERIMENT_DIR, "eval", "labels_gold.jsonl")
SILVER_PATH = os.path.join(EXPERIMENT_DIR, "eval", "labels.jsonl")
PDF_DIR = os.path.join(EXPERIMENT_DIR, "corpus", "s3_sample", "pdfs")

# Friendly source labels for candidate chips
SOURCE_LABELS = {
    "baseline_llm": "Qwen-7B",
    "direct_qwen3b": "Qwen-3B",
    "direct_qwen1.5b": "Qwen-1.5B",
    "direct_qwen0.5b": "Qwen-0.5B",
    "direct_phi3.5mini": "Phi-3.5",
    "docetl": "DocETL",
    "palimpzest": "Palimpzest",
    "bargain": "BARGAIN",
    "bargain_year": "BARGAIN-yr",
}


def granularity(normalized: str) -> str:
    if normalized == "N/A":
        return "na"
    return {4: "year", 7: "month", 10: "day"}.get(len(normalized), "other")


def load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class LabelStore:
    """Digest-keyed gold labels persisted to a JSONL file (upsert)."""

    def __init__(self, path: str):
        self.path = path
        self.by_digest = {r["digest"]: r for r in load_jsonl(path)}

    def get(self, digest: str) -> dict | None:
        return self.by_digest.get(digest)

    def save(self, record: dict) -> None:
        self.by_digest[record["digest"]] = record
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in self.by_digest.values():
                f.write(json.dumps(rec) + "\n")


def build_candidates(digest: str, manifest_rec: dict, results: dict) -> list[dict]:
    """Group candidate dates by normalized value → list of source labels."""
    raw_sources: list[tuple[str, str]] = []
    meta = parse_pdf_creation_date(manifest_rec["embedded_creation_date"])
    if meta != "N/A":
        raw_sources.append((meta, "PDF metadata"))
    regex = results.get("baseline_regex", {}).get(digest)
    if regex and regex != "N/A":
        raw_sources.append((regex, "regex"))
    silver = results.get("__silver__", {}).get(digest)
    if silver and silver != "N/A":
        raw_sources.append((silver, "silver (7B annotator)"))
    for method, label in SOURCE_LABELS.items():
        val = results.get(method, {}).get(digest)
        if val and val != "N/A":
            raw_sources.append((val, label))

    grouped: dict[str, list[str]] = {}
    for value, source in raw_sources:
        grouped.setdefault(value, []).append(source)
    # Sort chips by how many sources agree (consensus first)
    return [
        {"value": v, "sources": srcs, "n": len(srcs)}
        for v, srcs in sorted(grouped.items(), key=lambda kv: -len(kv[1]))
    ]


def create_app(args) -> Flask:
    app = Flask(__name__)
    manifest = load_manifest(args.manifest)
    order = [r["digest"] for r in manifest]
    by_digest = {r["digest"]: r for r in manifest}

    # Candidate sources: all result files keyed method→digest→normalized_date
    results: dict[str, dict[str, str]] = {}
    for path in glob.glob(os.path.join(RESULTS_DIR, "*.jsonl")):
        method = os.path.splitext(os.path.basename(path))[0]
        results[method] = {r["digest"]: r["normalized_date"] for r in load_jsonl(path)}
    results["__silver__"] = {r["digest"]: r["date"] for r in load_jsonl(SILVER_PATH)}
    silver_by_digest = {r["digest"]: r for r in load_jsonl(SILVER_PATH)}

    store = LabelStore(args.gold_path)

    @app.route("/")
    def index():
        # Jump to first unlabeled doc, else the first doc
        for i, digest in enumerate(order):
            if digest not in store.by_digest:
                return redirect(url_for("doc", i=i))
        return redirect(url_for("doc", i=0))

    @app.route("/img/<digest>/<int:pg>")
    def img(digest, pg):
        return send_from_directory(
            os.path.join(args.data_dir, "img", digest),
            f"{digest}_{pg}.jpeg",
        )

    @app.route("/pdf/<digest>")
    def pdf(digest):
        return send_from_directory(PDF_DIR, f"{digest}.pdf")

    @app.route("/doc/<int:i>")
    def doc(i):
        i = max(0, min(i, len(order) - 1))
        digest = order[i]
        rec = by_digest[digest]
        candidates = build_candidates(digest, rec, results)
        existing = store.get(digest)
        silver = silver_by_digest.get(digest, {})
        return render_doc(
            i, len(order), rec, candidates, existing, silver, len(store.by_digest)
        )

    @app.route("/save", methods=["POST"])
    def save():
        i = int(request.form["i"])
        digest = request.form["digest"]
        action = request.form["action"]
        if action == "na":
            date = "N/A"
        else:
            date = normalize_date(request.form.get("date", "").strip())
        record = {
            "digest": digest,
            "date": date,
            "granularity": granularity(date),
            "provenance": "gold",
            "labeler": args.labeler,
            "labeled_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "uncertain": request.form.get("uncertain") == "on",
            "notes": request.form.get("notes", "").strip(),
            "silver_date": silver_by_digest.get(digest, {}).get("date"),
        }
        store.save(record)
        return redirect(url_for("doc", i=i + 1))

    return app


def render_doc(i, total, rec, candidates, existing, silver, n_done) -> str:
    digest = rec["digest"]
    num_pages = rec["num_pages"]
    crawl = rec.get("first_crawl_date")
    chips = "".join(
        f'<button type="button" class="chip{" consensus" if c["n"] > 1 else ""}" '
        f"onclick=\"pick('{c['value']}')\">{c['value']} "
        f'<span class="src">{", ".join(c["sources"])}</span></button>'
        for c in candidates
    )
    prefill = ""
    existing_note = ""
    if existing:
        d = existing["date"]
        prefill = "" if d == "N/A" else d
        flag = " ⚠ uncertain" if existing.get("uncertain") else ""
        existing_note = (
            f'<div class="done">already labeled: <b>{d}</b>{flag} '
            f"— re-saving overwrites</div>"
        )
    pages_js = json.dumps(
        [url_for("img", digest=digest, pg=p) for p in range(num_pages)]
    )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>gold labeling {i + 1}/{total}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #1a1a1a;
         color: #eee; display: flex; height: 100vh; }}
  #left {{ flex: 2; overflow: auto; background: #000; text-align: center; }}
  #left img {{ max-width: 100%; }}
  #right {{ flex: 1; padding: 20px; overflow: auto; max-width: 520px; }}
  .bar {{ height: 4px; background: #2a7; width: {100 * n_done / total:.1f}%; }}
  h2 {{ margin: 6px 0; font-size: 15px; color: #9cf; word-break: break-all; }}
  .ctx {{ color: #aaa; font-size: 13px; margin: 8px 0; }}
  .chip {{ display: block; width: 100%; text-align: left; margin: 4px 0;
           padding: 8px; background: #333; color: #eee; border: 1px solid #555;
           border-radius: 5px; cursor: pointer; font-size: 14px; }}
  .chip.consensus {{ border-color: #2a7; }}
  .chip:hover {{ background: #444; }}
  .src {{ color: #888; font-size: 11px; }}
  input[type=text] {{ width: 100%; padding: 10px; font-size: 16px;
           font-family: monospace; background: #222; color: #fff;
           border: 1px solid #666; border-radius: 5px; box-sizing: border-box; }}
  .btns {{ display: flex; gap: 8px; margin-top: 12px; }}
  button.act {{ flex: 1; padding: 12px; font-size: 15px; border: none;
           border-radius: 5px; cursor: pointer; }}
  .save {{ background: #2a7; color: #000; font-weight: bold; }}
  .na {{ background: #a55; color: #fff; }}
  .nav a {{ color: #9cf; margin-right: 14px; }}
  .done {{ background: #443; padding: 6px; border-radius: 4px; font-size: 13px; }}
  textarea {{ width: 100%; background: #222; color: #eee; border: 1px solid #666;
           border-radius: 5px; box-sizing: border-box; }}
  .pg {{ color: #888; }}
</style></head><body>
<div id=left>
  <div style="padding:8px">
    <button type=button onclick="prev()">◀ page</button>
    <span class=pg id=pgind>1 / {num_pages}</span>
    <button type=button onclick="next()">page ▶</button>
    &nbsp; <a href="{url_for("pdf", digest=digest)}" target=_blank
       style="color:#9cf">open raw PDF ↗</a>
  </div>
  <img id=pageimg src="{url_for("img", digest=digest, pg=0)}">
</div>
<div id=right>
  <div class=bar></div>
  <div class=nav style="margin:8px 0">
    <a href="{url_for("doc", i=i - 1)}">◀ prev</a>
    <b>{i + 1} / {total}</b> ({n_done} labeled)
    <a href="{url_for("doc", i=i + 1)}">next ▶</a>
  </div>
  <h2>{rec["pretty_name"] or "(untitled)"}</h2>
  <div class=ctx>digest {digest}<br>
    pages: {num_pages} &nbsp; crawled: {crawl or "?"}
    (true date must be ≤ this)<br>
    <a href="{rec.get("url") or "#"}" target=_blank style="color:#78a">source url</a>
  </div>
  {existing_note}
  <p style="font-size:13px;color:#ccc">Read the document, then choose or type
    the date it was <b>written</b> (YYYY, YYYY-MM, or YYYY-MM-DD). Candidates
    (green = multiple methods agree):</p>
  {chips or '<i style="color:#888">no candidate dates proposed</i>'}
  <form method=post action="{url_for("save")}" style="margin-top:14px">
    <input type=hidden name=i value="{i}">
    <input type=hidden name=digest value="{digest}">
    <input type=hidden name=action value="date" id=action>
    <input type=text name=date id=date placeholder="YYYY-MM-DD" value="{prefill}"
       autofocus autocomplete=off>
    <label style="font-size:13px;display:block;margin-top:8px">
      <input type=checkbox name=uncertain> hard / uncertain</label>
    <textarea name=notes rows=2 placeholder="notes (optional)"></textarea>
    <div class=btns>
      <button class="act save" type=submit>Save &amp; next ▶</button>
      <button class="act na" type=submit
        onclick="document.getElementById('action').value='na'">N/A</button>
    </div>
  </form>
</div>
<script>
  const pages = {pages_js};
  let pg = 0;
  function show() {{
    document.getElementById('pageimg').src = pages[pg];
    document.getElementById('pgind').textContent = (pg + 1) + ' / ' + pages.length;
  }}
  function next() {{ if (pg < pages.length - 1) {{ pg++; show(); }} }}
  function prev() {{ if (pg > 0) {{ pg--; show(); }} }}
  function pick(v) {{ document.getElementById('date').value = v;
                      document.getElementById('action').value = 'date'; }}
  document.getElementById('date').addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{ document.getElementById('action').value = 'date'; }}
  }});
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data_dir", default=CORPUS_DATA_DIR)
    parser.add_argument("--gold_path", default=GOLD_PATH)
    parser.add_argument("--labeler", default=os.environ.get("USER", "human"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    args = parser.parse_args()
    if args.manifest is None:
        from common import MANIFEST_PATH

        args.manifest = MANIFEST_PATH

    app = create_app(args)
    print(f"Gold labeling at http://{args.host}:{args.port}  → {args.gold_path}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
