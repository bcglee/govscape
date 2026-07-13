# AI modified: 2026-07-13 9572ec45
#!/usr/bin/env bash
# Serve a model on the local T4 via vLLM's OpenAI-compatible API.
# Usage: serve_vllm.sh [model] (default Qwen2.5-7B-Instruct-AWQ)
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"

# --enforce-eager: torch.compile / CUDA-graph capture is extremely slow on
# the T4 (SM75); eager mode trades some throughput for fast, reliable startup.
exec /home/ubuntu/venvs/vllm/bin/vllm serve "$MODEL" \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 127.0.0.1 \
  --port "$PORT"
