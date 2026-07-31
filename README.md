# README.md for generation and evaluation of GovText: `scripts/data_prep`

Three files produce the GovText data product and every number reported in the
paper.

| File | Role |
|---|---|
| `ocr_common.py` | Shared constants, S3 clients, tar helpers, checkpointing |
| `build_data_product.py` | Assembles the four released artifacts per PDF |
| `analyze_ocr_and_metadata.py` | Computes per-page comparison metrics |

---

## `ocr_common.py`

Support code imported by both pipeline scripts. Nothing here runs on its own.

- **Source Cooperative constants** — bucket names and key prefixes for both the
  public HTTPS endpoint and the native `us-east-1` bucket used for writes.
- **`CoopWriter`** — upload, put, and server-side copy with exponential backoff
  on transient errors and automatic credential reload on expiry. Every write
  path either succeeds or raises; there is no silent fall-through.
- **`make_coop_anon_client`** — unsigned client, the same credential-free access
  path described in the paper's Listing 1.
- **`tar_pages` / `untar_pages`** — the released text format. Page-level text is
  written as `{pdf_id}_{page}.txt` inside `{pdf_id}.tar.gz`, **1-indexed**.
- **`JsonCheckpoint`** — a JSON dict persisted after each batch, so both
  pipelines resume without reprocessing.
- **`DIGEST_RE`** — base32 SHA-1, the identifier inherited from the End of Term
  Web Archive CDX index.

## `build_data_product.py`

Builds the release in three independently checkpointed phases.

**`--phase text`** — walks the OCR-text keyspace, and for each PDF copies the
source PDF into the product and extracts its embedded text layer with
pypdfium2 (`_extract_pdf_pages`), uploading it as a gzipped tar. This is the
`extracted_text/` artifact and the pypdfium2 side of the paper's comparison.

**`--phase copy`** — server-side `CopyObject` of `ocr_text/` and
`ocr_metadata/` from the upstream archive into the product. No bytes transit
the instance.

**`--phase cdx`** — DuckDB semi-join filtering `complete_cdx.parquet` to the
released set, producing the web-provenance artifact (URL, crawl date, and
source WARC location for every capture). The hosting-URL statistics in Table I
come from this file.

Documents are released only when all four artifacts are present, so any
keyspace can be enumerated to obtain the same document set.

## `analyze_ocr_and_metadata.py`

Produces every number in Section IV. For each document it fetches
both text renderings from S3 in memory, aligns them by page, and writes
per-page metrics to Parquet shards.

Each page is classified into one of four buckets — `both`, `ocr_only`,
`pdf_text_only`, `neither` — which is Table II. Comparison metrics are computed
only on `both` pages.

**Normalization** (`normalize_l1`, `normalize_l2`), paper §IV-D:

- **L1** — NFKC, control characters replaced with spaces, whitespace collapsed.
- **L2** — L1 plus HTML tag and Markdown artifact removal, then lower-case.

**Metrics per page** (`_pair_metrics`, `compute_page_metrics`), computed at both
normalization levels:

| Column | Paper |
|---|---|
| `rouge_l_*` | LCS-based F1 over words, §IV-D |
| `jaccard_*` | Word-set overlap, §IV-D |
| `norm_edit_distance_char_*` | Character error rate, §IV-D |
| `norm_edit_distance_word_*` | Word error rate, §IV-D |
| `containment_ocr_*`, `containment_pdf_*` | §IV-E |
| `tokens_{ocr,pdf}_{raw,l1,l2}` | `o200k_base` counts, §IV-A and Table I |
| `garbage_ratio_{ocr,pdf}_raw` | §IV-C |

`garbage_ratio` counts characters that are Unicode replacement, control,
format, surrogate, private-use, or unassigned, over all non-whitespace
characters. Undefined ratios are stored as the sentinel `-1.0`.

Execution is a hybrid of processes and threads (`--procs` × `--threads`): the
workload is I/O-bound on S3 fetches, so many concurrent requests per worker
process substantially outperform either alone.

Versions used: pypdfium2 4.30, tiktoken 0.13.0, olmOCR 0.4.
