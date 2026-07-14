# AI modified: 2026-07-14 8d36a86c
#!/usr/bin/env bash
# Qwen3.5 hybrid-reasoning sweep: each dense model that fits the T4, run in
# thinking and non-thinking mode. Results land in
# results/direct_<name>_{think,nothink}.jsonl. Restores the 7B server at the end.
set -uo pipefail
cd "$(dirname "$0")/../../.."

MODELS="Qwen/Qwen3.5-0.8B=q35_0.8b
Qwen/Qwen3.5-2B=q35_2b
Qwen/Qwen3.5-4B=q35_4b"

stop_server() {
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 5
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
  sleep 3
}

start_and_wait() {
  local parser="${3:-qwen3}"
  [ "$parser" = none ] && parser=""
  VLLM_VENV="${2:-/home/ubuntu/venvs/vllm-q35}" \
    REASONING_PARSER="$parser" MAX_LEN="${4:-16384}" \
    nohup bash experiments/date_extraction/serving/serve_vllm.sh "$1" > /tmp/vllm.log 2>&1 &
  for _ in $(seq 1 240); do
    if curl -s -m 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      return 0
    fi
    if ! pgrep -f "vllm serve" >/dev/null; then
      echo "SERVER DIED for $1"
      tail -15 /tmp/vllm.log
      return 1
    fi
    sleep 5
  done
  echo "TIMEOUT waiting for $1"
  return 1
}

for entry in $MODELS; do
  model="${entry%%=*}"
  name="${entry##*=}"
  echo "=== $model ==="
  stop_server
  if ! start_and_wait "$model"; then
    echo "SKIPPING $name"
    continue
  fi
  for mode in nothink think; do
    flag=off; [ "$mode" = think ] && flag=on
    echo "--- $name $mode ---"
    poetry run python experiments/date_extraction/methods/baseline_llm/run.py \
      --thinking "$flag" --method_name "direct_${name}_${mode}" 2>&1 | tail -2
  done
done

echo "=== restoring 7B server ==="
stop_server
start_and_wait "Qwen/Qwen2.5-7B-Instruct-AWQ" /home/ubuntu/venvs/vllm none 8192 \
  && echo "7B restored"
echo "=== SWEEP DONE ==="
