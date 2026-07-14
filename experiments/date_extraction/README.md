<!-- AI modified: 2026-07-13 9572ec45 -->

# Document Creation-Date Extraction Experiments

Compare off-the-shelf LLM-powered document-processing frameworks for extracting the
*true* creation date of govscape PDFs — the date the document was actually written
(e.g., the date a scanned letter was authored), not the scan/digitization date stored
in the embedded PDF `CreationDate` metadata. "N/A" is a valid answer when the
document does not reveal its date.

All methods run against **open-source models only**, served locally on this machine's
single NVIDIA T4 (16 GB, SM75/fp16).

## Prior work evaluated

| System | Verdict | Link |
|--------|---------|------|
| DocETL (VLDB'25) | Run — `methods/docetl/` | https://github.com/ucbepic/docetl |
| Palimpzest (CIDR'25) | Run — `methods/palimpzest/` | https://github.com/mitdbg/palimpzest |
| BARGAIN, "Cut Costs, Not Accuracy" (PACMMOD'26) | Run — `methods/bargain/` | https://github.com/ucbepic/BARGAIN |
| Task Cascades (PACMMOD'26) | Skipped: repo is a GPT-4o-hardcoded experiment artifact, not a library. Its idea (cheap prefix/simplified-task pass, escalate on low confidence) is what `methods/bargain/` exercises with local models. | https://github.com/ucbepic/task-cascades |
| TWIX (arXiv 2501.06659) | Skipped: requires ≥2 docs sharing one visual template; a heterogeneous .gov corpus breaks the core assumption, and the authored date is prose/letterhead metadata rather than a template field. | https://github.com/ucbepic/TWIX |
| QUEST (VLDB'25) | Skipped: multi-predicate query optimizer degenerates for single-attribute extraction; unlicensed research artifact, OpenAI hardcoded. Its segment-retrieval insight survives here as the page-1 + last-page input truncation all methods use. | https://github.com/qiyandeng/QUEST |
| Doctopus (VLDB'25) | Skipped: unlicensed benchmark-hardcoded artifact; budget allocation across attributes/strategies is meaningless for one attribute. Its cheap-strategy-first insight survives as `methods/baseline_regex/`. | https://github.com/mutong184/Doctopus |

## Layout

- `corpus/` — eval corpus prep: ~150 PDFs randomly sampled from the public archive,
  per-page text/images via `PDFExtractionStage`, OCR fallback for scanned pages,
  `manifest.jsonl` (one record per document).
- `serving/` — local model server: vLLM (primary; OpenAI-compatible + logprobs)
  serving Qwen2.5-7B-Instruct-AWQ, Ollama fallback. Smoke test included.
- `methods/` — one subdir per method; frameworks get their own venv (their deps
  conflict with the repo poetry env). Every method writes
  `results/<method>.jsonl`: `{digest, predicted_date, normalized_date, evidence,
  tokens_in, tokens_out, wall_ms}`.
- `eval/` — `labels.jsonl` ground truth (silver labels from a full-document 7B
  annotation pass initially; `provenance` field supports later human gold upgrade)
  and `score.py` (exact/year-month/year accuracy, N/A precision & recall, cost).
- `results/` — per-method outputs + final report.

## Running

All commands from the repo root. Framework venvs live under `/home/ubuntu/venvs/`
(`vllm`, `docetl`, `palimpzest`, `bargain`); the repo poetry env additionally needs
`pip install pytesseract ocrmypdf` (system `tesseract` required) for corpus OCR.

```bash
# 1. Corpus: download ~150 random PDFs, extract text/images, OCR scans, manifest
poetry run python scripts/data_prep/download_sample_pdfs.py \
  --bucket_name us-west-2.opendata.source.coop/govscape/eota-pdf-archive/ \
  --local_base_dir experiments/date_extraction/corpus/s3_sample --num_pdfs 150
poetry run python experiments/date_extraction/corpus/build_corpus.py \
  --pdf_dir experiments/date_extraction/corpus/s3_sample/pdfs \
  --data_dir experiments/date_extraction/corpus/data \
  --cdx_parquet experiments/date_extraction/corpus/s3_sample/cdx/complete_cdx.parquet \
  --manifest experiments/date_extraction/corpus/manifest.jsonl

# 2. Serving: vLLM + Qwen2.5-7B-Instruct-AWQ on the T4 (OpenAI API on :8000)
nohup bash experiments/date_extraction/serving/serve_vllm.sh > /tmp/vllm.log 2>&1 &
poetry run python experiments/date_extraction/serving/smoke_test.py

# 3. Silver ground-truth labels (annotator reads far more text than any method)
poetry run python experiments/date_extraction/eval/annotate_silver.py

# 4. Methods (each writes results/<method>.jsonl; all accept --limit N)
poetry run python experiments/date_extraction/methods/baseline_metadata/run.py
poetry run python experiments/date_extraction/methods/baseline_regex/run.py
poetry run python experiments/date_extraction/methods/baseline_llm/run.py
bash experiments/date_extraction/methods/docetl/run.sh
/home/ubuntu/venvs/palimpzest/bin/python experiments/date_extraction/methods/palimpzest/run.py
/home/ubuntu/venvs/bargain/bin/python experiments/date_extraction/methods/bargain/run.py

# 5. Score everything against the labels
poetry run python experiments/date_extraction/eval/score.py
```

### Gold labeling (human)

Silver labels (7B annotator) are the default ground truth; upgrade any subset to
human gold with the browser labeling app. It shows each PDF's pages plus candidate
dates (metadata / regex / silver / every model, grouped by agreement) and records
your call to `eval/labels_gold.jsonl` (non-destructive, resumable):

```bash
# on the server
poetry run python experiments/date_extraction/eval/label_app.py
# from your laptop, tunnel the port, then open http://127.0.0.1:5055
ssh -L 5055:127.0.0.1:5055 <server>
```

Per document: read the pages (page-nav + "open raw PDF" link; the crawl date is
shown as an upper bound), then click a candidate chip or type the date the document
was *written* (any of YYYY / YYYY-MM / YYYY-MM-DD; free text is normalized on save),
or hit **N/A**. Mark **hard/uncertain** and add notes for tricky cases. It opens at
the first unlabeled doc and you can stop anytime.

Score against gold (overlays gold onto silver per digest, so partial labeling still
counts — un-labeled docs fall back to their silver label):

```bash
poetry run python experiments/date_extraction/eval/score.py   # auto-uses labels_gold.jsonl
```

Each gold record keeps `silver_date` alongside the human `date`, so you can measure
how far the 7B annotator drifted from ground truth once labeling is done.

Serving notes (T4 / SM75): fp16 only; `--enforce-eager` because torch.compile is
pathologically slow on Turing; `--tool-call-parser hermes` because DocETL requests
schema output via tool-calling. The AWQ int4 7B (~5.5 GB weights) leaves room for
an 8k context.

## Results (2026-07-13, silver labels)

See `results/report.md` for the full table. Headline numbers (year-level accuracy
on the 138 dated docs / N/A precision / tokens per doc):

| method | year_acc | na_precision | tokens/doc | notes |
|---|---|---|---|---|
| baseline_metadata | 0.43 | 0.12 | 0 | current govscape behavior |
| baseline_regex | 0.54 | 0.20 | 0 | |
| palimpzest | 0.67 | 0.27 | 1653 | |
| docetl | 0.76 | 0.37 | 1943 | |
| baseline_llm | 0.85 | 0.65 | 928 | direct call, same model+excerpt |
| bargain / bargain_year | 0.98 | 0.92 | ~2800 | inflated — see caveat |

Findings:

1. **The problem is real.** The embedded PDF `CreationDate` (what govscape stores
   today) agrees with the document's own stated date on only 43% of dated docs,
   and on scans it drops to 33% — the scan-date-vs-written-date gap this branch
   set out to measure.
2. **A direct local-LLM call is the strongest practical method.** Qwen2.5-7B-AWQ
   on page-1 + last-page text: 85% year accuracy at ~930 tokens and ~1.7 s/doc
   (~50k docs/day on this T4). Accuracy drops to 0.70 on OCR'd scans vs 0.88 on
   digital text.
3. **The declarative frameworks add overhead, not accuracy, at this task shape.**
   DocETL (0.76) and Palimpzest (0.67) wrap the same model and excerpt but score
   below the direct call — their generic prompt scaffolding + tool-call output
   path loses more than it gains for single-field extraction, costs ~2x the
   tokens, and DocETL intermittently aborts when the model emits a malformed
   tool call (retry loop in `run.sh`). Their value (optimizers, operator
   pipelines) doesn't engage on a one-op pipeline.
4. **BARGAIN's guarantee machinery works, and its verdict is instructive: it
   refused the cheap path.** At target=0.9/δ=0.1 it routed 100% (exact
   validation; 95% with year-level validation) of records to the expensive
   oracle — the page-1-prefix proxy is not certifiably reliable, so with only
   local models the cost-savings story mostly collapses. Its near-perfect
   accuracy row is **circular**: its oracle is the same model+excerpt as the
   silver-label annotator.
5. **Caveat on all rows:** ground truth is silver (7B annotator reading ~3x more
   text than any method). Human gold labeling (`provenance` upgrade in
   `eval/labels.jsonl`) is the planned next step; treat absolute numbers as
   agreement-with-annotator until then.

## Model-size sweep (2026-07-14)

`serving/run_model_sweep.sh` reruns the direct baseline with smaller models
(all with vLLM guided-JSON decoding, `json_mode` in `common.chat_completion`):

| model | year_acc | ym_acc | na_precision | na_recall | ms/doc |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.69 | 0.62 | 0.00 | 0.00 | 903 |
| Qwen2.5-1.5B | 0.65 | 0.46 | 0.09 | 0.17 | 1299 |
| Qwen2.5-3B | 0.74 | 0.74 | 0.30 | 0.58 | 1963 |
| Phi-3.5-mini (3.8B) | 0.71 | 0.64 | 0.34 | 0.92 | 3040 |
| Qwen2.5-7B-AWQ | 0.85 | 0.80 | 0.62 | 0.83 | 1749 |

Sweep findings:

1. **Guided JSON is mandatory below 7B.** Without `response_format` enforcement,
   Qwen2.5-3B rambled a reasoning preamble past the token cap on 34/150 docs and
   scored 0.31 year accuracy; with guided decoding it scored 0.74. The 7B was
   format-reliable either way. Format compliance, not extraction ability, is the
   first thing that breaks with model size.
2. **Year-level accuracy degrades gracefully; abstention collapses.** Even 0.5B
   gets 0.69 year accuracy (most docs state a prominent date — grabbing it is
   easy, cf. the regex baseline at 0.54). But N/A handling scales steeply:
   0.5B never abstains correctly (0.00 precision), 3B reaches 0.30, 7B 0.62.
   Distinguishing "this document does not reveal its date" from "I found some
   date-like string" is the capability that requires scale.
3. **Precision scales too:** month-level accuracy is 0.46-0.74 below 7B vs 0.80
   at 7B — small models more often return a year when the doc states a full date.
4. Caveat: labels are silver from Qwen2.5-7B, so the sweep measures agreement
   with the 7B annotator; same-family models may be slightly favored. The
   qualitative gaps (abstention, granularity) are large enough to survive that.

## Gold-label results (2026-07-14, 79 human labels)

79/150 docs now have human gold labels (`eval/labels_gold.jsonl`, incl. all 12
formerly-N/A docs and 17/30 scans; 5 flagged uncertain). `score.py` overlays gold
onto silver by default; pass `--labels eval/labels_gold.jsonl` for the gold-only
view. Gold-only (68 dated, 11 N/A):

| method | year_acc | ym_acc | exact_acc | na_precision | na_recall |
|---|---|---|---|---|---|
| silver annotator (7B, full doc) | 0.85 | — | 0.51 | 1.00 | 0.36 |
| bargain / bargain_year | 0.82–0.85 | 0.75 | ~0.51 | 1.00 | 0.45 |
| baseline_llm (7B direct) | 0.81 | 0.66 | 0.44 | 1.00 | 0.36 |
| docetl | 0.79 | 0.70 | 0.56 | 0.64 | 0.64 |
| direct_qwen3b / phi3.5 / qwen0.5b | 0.72 | 0.57–0.67 | 0.50–0.63 | — | — |
| baseline_metadata | 0.43 | 0.30 | 0.25 | 0.00 | 0.00 |

Findings vs the silver-only evaluation:

1. **The silver annotator was measurably wrong.** It matches human gold exactly
   on only 38/79 docs (48%); 85% at year level. Failure modes: hallucinating
   dates on undatable docs (gold has 19 N/A across 150 vs silver's 12), century
   misreads on old scans (1592→1892, 1666→1866 style), and coarser granularity
   than the document supports.
2. **BARGAIN's near-perfect silver scores were indeed circular.** Against gold it
   falls from 0.98 to 0.82–0.85 year accuracy — statistically indistinguishable
   from the direct 7B call (0.81) and DocETL (0.79) at n=68 (±~9%). Reading 3x
   more context buys granularity (ym/exact), not year-level accuracy.
3. **Abstention is the open problem for every LLM method.** All 7B-based methods
   have perfect N/A precision but recall <= 0.45: when they answer N/A it's
   right, but they hallucinate a date on more than half the docs a human judges
   undatable. DocETL's laxer prompt scaffolding trades precision for the best
   recall (0.64).
4. **Rankings among the frameworks are unchanged** (metadata << regex < small
   models < Palimpzest < DocETL <= direct 7B <= BARGAIN), so the silver-label
   methodology was directionally sound — it inflated absolute numbers, not the
   ordering.
