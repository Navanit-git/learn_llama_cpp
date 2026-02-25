#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/nav_wsl/.cache/llama.cpp/unsloth_Qwen3.5-27B-GGUF_Qwen3.5-27B-Q2_K.gguf}"
PORT="${PORT:-8080}"
THREADS="${THREADS:-10}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH="${BATCH:-128}"
UBATCH="${UBATCH:-128}"
NGL_VALUES="${NGL_VALUES:-0 2 4 6 8}"
OUT_DIR="${OUT_DIR:-$PWD/../archieve/qwen_bench_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"
RESULTS_CSV="$OUT_DIR/results.csv"

echo "ngl,eval_tps,prompt_tps,log_file" > "$RESULTS_CSV"

echo "Benchmark output dir: $OUT_DIR"
echo "Model: $MODEL_PATH"
echo "NGL values: $NGL_VALUES"

run_one() {
  local ngl="$1"
  local log_file="$OUT_DIR/ngl_${ngl}.log"

  echo
  echo "=== Testing -ngl $ngl ==="

  ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    --ctx-size "$CTX_SIZE" --jinja \
    -b "$BATCH" -ub "$UBATCH" \
    -t "$THREADS" -tb "$THREADS" -np 1 \
    -ngl "$ngl" --fit off \
    -fa on \
    -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port "$PORT" \
    > "$log_file" 2>&1 &
  local server_pid=$!

  cleanup() {
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  }

  trap cleanup RETURN

  local ready=0
  for _ in $(seq 1 180); do
    if grep -q "server is listening on" "$log_file"; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if [[ "$ready" -ne 1 ]]; then
    echo "Failed to start at -ngl $ngl (see $log_file)"
    echo "$ngl,0,0,$log_file" >> "$RESULTS_CSV"
    return 0
  fi

  curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen35-27b-local",
      "messages": [{"role": "user", "content": "Write a short 120-word paragraph about Linux terminal productivity."}],
      "max_tokens": 160,
      "temperature": 0.2,
      "top_p": 0.9,
      "stream": false,
      "chat_template_kwargs": {"enable_thinking": false}
    }' >/dev/null

  sleep 1

  local eval_tps
  local prompt_tps

  eval_tps=$(sed -nE 's/.*eval time =.*\([[:space:]]*[0-9.]+ ms per token,[[:space:]]*([0-9.]+) tokens per second\).*/\1/p' "$log_file" | tail -n1)
  prompt_tps=$(sed -nE 's/.*prompt eval time =.*\([[:space:]]*[0-9.]+ ms per token,[[:space:]]*([0-9.]+) tokens per second\).*/\1/p' "$log_file" | tail -n1)

  eval_tps="${eval_tps:-0}"
  prompt_tps="${prompt_tps:-0}"

  echo "-ngl $ngl => eval_tps=$eval_tps, prompt_tps=$prompt_tps"
  echo "$ngl,$eval_tps,$prompt_tps,$log_file" >> "$RESULTS_CSV"
}

for ngl in $NGL_VALUES; do
  run_one "$ngl"
done

echo
printf "%-8s %-12s %-12s\n" "ngl" "eval_tps" "prompt_tps"
tail -n +2 "$RESULTS_CSV" | while IFS=, read -r ngl eval_tps prompt_tps _; do
  printf "%-8s %-12s %-12s\n" "$ngl" "$eval_tps" "$prompt_tps"
done

best_line=$(tail -n +2 "$RESULTS_CSV" | awk -F, 'BEGIN{best=-1; line=""} {val=$2+0; if (val>best) {best=val; line=$0}} END{print line}')
if [[ -n "$best_line" ]]; then
  IFS=, read -r best_ngl best_eval best_prompt best_log <<< "$best_line"
  echo
  echo "Best setting: -ngl $best_ngl (eval_tps=$best_eval, prompt_tps=$best_prompt)"
  echo "Best log: $best_log"
fi

echo "CSV: $RESULTS_CSV"
