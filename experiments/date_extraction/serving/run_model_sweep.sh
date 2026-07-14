# AI modified: 2026-07-14 9572ec45
#!/usr/bin/env bash
# Model-size sweep: serve each model on the T4 in turn and run the direct
# extraction baseline against it. Results land in results/direct_<name>.jsonl
# (the 7B reference is the existing baseline_llm row). Restores the 7B server
# when done.
set -uo pipefail
cd "$(dirname "$0")/../../.."

SWEEP="Qwen/Qwen2.5-0.5B-Instruct=direct_qwen0.5b
Qwen/Qwen2.5-1.5B-Instruct=direct_qwen1.5b
Qwen/Qwen2.5-3B-Instruct=direct_qwen3b
microsoft/Phi-3.5-mini-instruct=direct_phi3.5mini"

stop_server() {
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 5
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
  sleep 3
}

start_and_wait() {
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

for entry in $SWEEP; do
  model="${entry%%=*}"
  name="${entry##*=}"
  echo "=== $model -> $name ==="
  stop_server
  if ! start_and_wait "$model"; then
    echo "SKIPPING $name"
    continue
  fi
  poetry run python experiments/date_extraction/methods/baseline_llm/run.py \
    --method_name "$name" 2>&1 | tail -2
done

echo "=== restoring 7B server ==="
stop_server
start_and_wait "Qwen/Qwen2.5-7B-Instruct-AWQ" && echo "7B restored"
echo "=== SWEEP DONE ==="
