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
