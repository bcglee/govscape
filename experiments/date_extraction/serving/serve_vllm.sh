# AI modified: 2026-07-13 9572ec45
# AI modified: 2026-07-14 8d36a86c
#!/usr/bin/env bash
# Serve a model on the local T4 via vLLM's OpenAI-compatible API.
# Usage: serve_vllm.sh [model] (default Qwen2.5-7B-Instruct-AWQ)
# Env knobs:
#   VLLM_VENV        venv to use (default /home/ubuntu/venvs/vllm; use
#                    /home/ubuntu/venvs/vllm-q35 for Qwen3.5 models)
#   REASONING_PARSER e.g. "qwen3" for Qwen3.5 thinking models (default none)
#   EXTRA_FLAGS      extra vllm serve flags (e.g. --language-model-only)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
VENV="${VLLM_VENV:-/home/ubuntu/venvs/vllm}"

ARGS=()
if [ -n "${REASONING_PARSER:-}" ]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

# --enforce-eager: torch.compile / CUDA-graph capture is extremely slow on
# the T4 (SM75); eager mode trades some throughput for fast, reliable startup.
exec "$VENV/bin/vllm" serve "$MODEL" \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-hermes}" \
  --host 127.0.0.1 \
  --port "$PORT" \
  "${ARGS[@]}" ${EXTRA_FLAGS:-}
