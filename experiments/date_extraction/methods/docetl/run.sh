# AI modified: 2026-07-13 9572ec45
#!/usr/bin/env bash
# Run the DocETL date-extraction pipeline against the local vLLM server.
set -euo pipefail
cd "$(dirname "$0")"

VENV=/home/ubuntu/venvs/docetl
export HOSTED_VLLM_API_BASE="${HOSTED_VLLM_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

"$VENV/bin/python" prepare_input.py "$@"
START=$(date +%s%3N)
"$VENV/bin/docetl" run pipeline.yaml
END=$(date +%s%3N)
"$VENV/bin/python" convert_results.py --wall_ms_total "$((END - START))"
