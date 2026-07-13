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

See per-directory READMEs. Order: corpus → serving → silver labels → methods → score.
