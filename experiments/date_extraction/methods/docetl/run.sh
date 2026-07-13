# AI modified: 2026-07-13 9572ec45
#!/usr/bin/env bash
# Run the DocETL date-extraction pipeline against the local vLLM server.
# DocETL occasionally aborts when the model emits a malformed tool call
# (InvalidOutputError); its cache makes retries cheap, so retry up to 3x.
# Token counts come from vLLM's /metrics counters (crash-proof), not
# DocETL's console report.
set -euo pipefail
cd "$(dirname "$0")"

VENV=/home/ubuntu/venvs/docetl
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

vllm_counter() {
  curl -s "${HOSTED_VLLM_API_BASE%/v1}/metrics" |
    grep -E "^vllm:${1}_tokens_total" | awk '{s+=$2} END {printf "%.0f", s}'
}

"$VENV/bin/python" prepare_input.py "$@"

IN_BEFORE=$(vllm_counter prompt)
OUT_BEFORE=$(vllm_counter generation)
START=$(date +%s%3N)
for attempt in 1 2 3; do
  if "$VENV/bin/docetl" run pipeline.yaml 2>&1 | tee "docetl_run_${attempt}.log"; then
    break
  elif [ "$attempt" = 3 ]; then
    echo "docetl failed after 3 attempts" >&2
    exit 1
  fi
  echo "docetl attempt $attempt failed; retrying (cache resumes)..." >&2
done
END=$(date +%s%3N)
IN_AFTER=$(vllm_counter prompt)
OUT_AFTER=$(vllm_counter generation)

"$VENV/bin/python" convert_results.py --wall_ms_total "$((END - START))" \
  --tokens_in_total "$((IN_AFTER - IN_BEFORE))" \
  --tokens_out_total "$((OUT_AFTER - OUT_BEFORE))"
