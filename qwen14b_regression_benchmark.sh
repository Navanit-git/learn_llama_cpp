#!/usr/bin/env bash
set -euo pipefail

# Qwen3-14B regression benchmark sweep for llama.cpp.
#
# Usage:
#   ./qwen14b_regression_benchmark.sh
#   MODE=smoke ./qwen14b_regression_benchmark.sh
#   MODE=exhaustive MAX_CASES=120 ./qwen14b_regression_benchmark.sh
#   MODEL_PATH=/path/to/model.gguf OUT_DIR=/tmp/run ./qwen14b_regression_benchmark.sh
#
# Optional overrides:
#   THREAD_VALUES="8 10" NGL_VALUES="0 4 8 12" CTX_VALUES="1024 2048"
#   BATCH_VALUES="64 128 256" UBATCH_VALUES="64 128 256"
#   KV_TYPES="q8_0:q8_0 f16:f16" FA_VALUES="on auto"

ROOT_DIR="${ROOT_DIR:-$PWD}"
LLAMA_DIR="${LLAMA_DIR:-$ROOT_DIR/llama.cpp}"
SERVER_BIN="${SERVER_BIN:-$LLAMA_DIR/build/bin/llama-server}"
BENCH_BIN="${BENCH_BIN:-$LLAMA_DIR/build/bin/llama-bench}"
CLI_BIN="${CLI_BIN:-$LLAMA_DIR/build/bin/llama-cli}"

MODEL_PATH="${MODEL_PATH:-$HOME/.cache/llama.cpp/unsloth_Qwen3-14B-GGUF_Qwen3-14B-Q4_K_M.gguf}"
MODE="${MODE:-full}" # smoke | full | exhaustive
PORT="${PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
MAX_TOKENS="${MAX_TOKENS:-180}"
MAX_CASES="${MAX_CASES:-0}" # 0 = no cap
RUN_LLAMA_BENCH="${RUN_LLAMA_BENCH:-1}"
RESUME="${RESUME:-1}" # 1 = resume from existing results.csv in OUT_DIR

OUT_DIR="${OUT_DIR:-$ROOT_DIR/archieve/qwen14b_regression_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

RESULTS_CSV="$OUT_DIR/results.csv"
BEST_CSV="$OUT_DIR/best_configs.csv"
META_TXT="$OUT_DIR/meta.txt"
LLAMA_BENCH_LOG="$OUT_DIR/llama_bench.log"

require_bin() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: missing executable: $path"
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path"
    exit 1
  fi
}

setup_arrays() {
  if [[ -n "${THREAD_VALUES:-}" ]]; then
    IFS=' ' read -r -a THREAD_ARR <<< "$THREAD_VALUES"
  else
    case "$MODE" in
      smoke) THREAD_ARR=(10) ;;
      full) THREAD_ARR=(8 10) ;;
      exhaustive) THREAD_ARR=(8 10 12) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${NGL_VALUES:-}" ]]; then
    IFS=' ' read -r -a NGL_ARR <<< "$NGL_VALUES"
  else
    case "$MODE" in
      smoke) NGL_ARR=(0 4 8 12) ;;
      full) NGL_ARR=(0 2 4 6 8 10 12 14) ;;
      exhaustive) NGL_ARR=(0 2 4 6 8 10 12 14 16 18 20) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${CTX_VALUES:-}" ]]; then
    IFS=' ' read -r -a CTX_ARR <<< "$CTX_VALUES"
  else
    case "$MODE" in
      smoke) CTX_ARR=(2048) ;;
      full) CTX_ARR=(1024 2048 3072) ;;
      exhaustive) CTX_ARR=(1024 1536 2048 3072 4096) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${BATCH_VALUES:-}" ]]; then
    IFS=' ' read -r -a BATCH_ARR <<< "$BATCH_VALUES"
  else
    case "$MODE" in
      smoke) BATCH_ARR=(128) ;;
      full) BATCH_ARR=(64 128 256) ;;
      exhaustive) BATCH_ARR=(64 128 192 256 384) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${UBATCH_VALUES:-}" ]]; then
    IFS=' ' read -r -a UBATCH_ARR <<< "$UBATCH_VALUES"
  else
    case "$MODE" in
      smoke) UBATCH_ARR=(128) ;;
      full) UBATCH_ARR=(64 128 256) ;;
      exhaustive) UBATCH_ARR=(64 128 192 256) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${KV_TYPES:-}" ]]; then
    IFS=' ' read -r -a KV_ARR <<< "$KV_TYPES"
  else
    case "$MODE" in
      smoke) KV_ARR=("q8_0:q8_0") ;;
      full) KV_ARR=("q8_0:q8_0" "f16:f16") ;;
      exhaustive) KV_ARR=("q8_0:q8_0" "f16:f16") ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi

  if [[ -n "${FA_VALUES:-}" ]]; then
    IFS=' ' read -r -a FA_ARR <<< "$FA_VALUES"
  else
    case "$MODE" in
      smoke) FA_ARR=(on) ;;
      full) FA_ARR=(on auto) ;;
      exhaustive) FA_ARR=(on auto off) ;;
      *) echo "ERROR: unknown MODE=$MODE"; exit 1 ;;
    esac
  fi
}

wait_for_server_ready() {
  local pid="$1"
  local log_file="$2"

  for _ in $(seq 1 "$STARTUP_TIMEOUT"); do
    if grep -q "server is listening on" "$log_file"; then
      return 0
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

stop_server() {
  local pid="$1"
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

extract_case_metrics() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys
path = sys.argv[1]
text = open(path, 'r', encoding='utf-8', errors='ignore').read().splitlines()

prompt = []
evals = []
load_sec = 'NA'

for line in text:
    m = re.search(r'prompt eval time =\s*([0-9.]+)\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[0-9.]+\s*ms per token,\s*([0-9.]+)\s*tokens per second\)', line)
    if m:
        prompt.append(m.group(2))
        continue
    m = re.search(r'\beval time =\s*([0-9.]+)\s*ms\s*/\s*\d+\s*tokens\s*\(\s*[0-9.]+\s*ms per token,\s*([0-9.]+)\s*tokens per second\)', line)
    if m:
        evals.append(m.group(2))
        continue
    m = re.search(r'fitting params to free memory took\s*([0-9.]+)\s*seconds', line)
    if m and load_sec == 'NA':
        load_sec = m.group(1)

p1 = prompt[0] if len(prompt) >= 1 else '0'
e1 = evals[0] if len(evals) >= 1 else '0'
p2 = prompt[1] if len(prompt) >= 2 else (prompt[-1] if prompt else '0')
e2 = evals[1] if len(evals) >= 2 else (evals[-1] if evals else '0')
print(p1, e1, p2, e2, load_sec)
PY
}

load_resume_state() {
  local csv_file="$1"
  local keys_out_file="$2"
  python3 - "$csv_file" "$keys_out_file" <<'PY'
import csv
import sys

csv_file = sys.argv[1]
keys_out = sys.argv[2]

max_case = 0
rows = 0
keys = []

with open(csv_file, newline='', encoding='utf-8') as f:
  reader = csv.DictReader(f)
  for row in reader:
    rows += 1
    try:
      max_case = max(max_case, int(row.get('case_id') or 0))
    except Exception:
      pass

    key = "|".join([
      row.get('threads', ''),
      row.get('ngl', ''),
      row.get('ctx', ''),
      row.get('batch', ''),
      row.get('ubatch', ''),
      row.get('cache_k', ''),
      row.get('cache_v', ''),
      row.get('fa', ''),
    ])
    keys.append(key)

with open(keys_out, 'w', encoding='utf-8') as f:
  for k in keys:
    f.write(k + "\n")

print(max_case, rows)
PY
}

run_case() {
  local case_id="$1"
  local threads="$2"
  local ngl="$3"
  local ctx="$4"
  local batch="$5"
  local ubatch="$6"
  local cache_k="$7"
  local cache_v="$8"
  local fa="$9"

  local log_file="$OUT_DIR/logs/case_${case_id}.log"
  local status="ok"
  local error=""

  "$SERVER_BIN" \
    -m "$MODEL_PATH" \
    --ctx-size "$ctx" --jinja \
    -b "$batch" -ub "$ubatch" \
    -t "$threads" -tb "$threads" -np 1 \
    -ngl "$ngl" --fit off \
    -fa "$fa" \
    -ctk "$cache_k" -ctv "$cache_v" \
    --host 127.0.0.1 --port "$PORT" \
    > "$log_file" 2>&1 &
  local pid=$!

  if ! wait_for_server_ready "$pid" "$log_file"; then
    status="start_failed"
    error="server_did_not_listen"
    stop_server "$pid"
    echo "$case_id,$status,$error,$threads,$ngl,$ctx,$batch,$ubatch,$cache_k,$cache_v,$fa,0,0,0,0,NA,$log_file" >> "$RESULTS_CSV"
    return 0
  fi

  local payload_1
  payload_1=$(python3 - "$MAX_TOKENS" <<'PY'
import json, sys
max_tokens = int(sys.argv[1])
obj = {
  "model": "qwen3-14b-local",
  "messages": [{"role": "user", "content": "Give 5 concise Linux terminal productivity tips."}],
  "max_tokens": max_tokens,
  "temperature": 0.2,
  "top_p": 0.9,
  "stream": False,
  "chat_template_kwargs": {"enable_thinking": False}
}
print(json.dumps(obj))
PY
)

  local payload_2
  payload_2=$(python3 - "$MAX_TOKENS" <<'PY'
import json, sys
max_tokens = int(sys.argv[1])
obj = {
  "model": "qwen3-14b-local",
  "messages": [{"role": "user", "content": "Now give 3 shell aliases for faster navigation."}],
  "max_tokens": max_tokens,
  "temperature": 0.2,
  "top_p": 0.9,
  "stream": False,
  "chat_template_kwargs": {"enable_thinking": False}
}
print(json.dumps(obj))
PY
)

  if ! curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$payload_1" >/dev/null; then
    status="request_failed"
    error="post_1_failed"
  fi

  if [[ "$status" == "ok" ]]; then
    if ! curl -sS -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$payload_2" >/dev/null; then
      status="request_failed"
      error="post_2_failed"
    fi
  fi

  sleep 1
  stop_server "$pid"

  read -r p1 e1 p2 e2 load_sec < <(extract_case_metrics "$log_file")

  echo "$case_id,$status,$error,$threads,$ngl,$ctx,$batch,$ubatch,$cache_k,$cache_v,$fa,$p1,$e1,$p2,$e2,$load_sec,$log_file" >> "$RESULTS_CSV"
}

# --- main ---
require_bin "$SERVER_BIN"
require_file "$MODEL_PATH"
setup_arrays

echo "Qwen3-14B regression benchmark" | tee "$META_TXT"
echo "Date: $(date -Iseconds)" | tee -a "$META_TXT"
echo "Mode: $MODE" | tee -a "$META_TXT"
echo "Model: $MODEL_PATH" | tee -a "$META_TXT"
echo "Server: $SERVER_BIN" | tee -a "$META_TXT"

echo >> "$META_TXT"
echo "--- llama.cpp checks ---" | tee -a "$META_TXT"
"$SERVER_BIN" --version >> "$META_TXT" 2>&1 || true
if [[ -x "$CLI_BIN" ]]; then
  "$CLI_BIN" --version >> "$META_TXT" 2>&1 || true
fi
if [[ -d "$LLAMA_DIR/.git" ]]; then
  git -C "$LLAMA_DIR" rev-parse --short HEAD >> "$META_TXT" 2>&1 || true
  git -C "$LLAMA_DIR" status --short >> "$META_TXT" 2>&1 || true
fi

if [[ "$RUN_LLAMA_BENCH" == "1" && -x "$BENCH_BIN" ]]; then
  {
    echo "=== llama-bench sanity run ==="
    "$BENCH_BIN" -m "$MODEL_PATH" -t 10 -ngl 0 -p 256 -n 64 -r 1
  } > "$LLAMA_BENCH_LOG" 2>&1 || true
fi

echo "Output dir: $OUT_DIR"
echo "Results CSV: $RESULTS_CSV"

declare -A DONE_KEYS=()
declare -i case_id=0
declare -i executed_cases=0
declare -i resume_skipped=0

if [[ "$RESUME" == "1" && -f "$RESULTS_CSV" ]]; then
  resume_keys_file="$OUT_DIR/.resume_keys.txt"
  read -r resume_max_case resume_rows < <(load_resume_state "$RESULTS_CSV" "$resume_keys_file")

  case_id="$resume_max_case"
  executed_cases="$resume_rows"

  while IFS= read -r k; do
    [[ -n "$k" ]] && DONE_KEYS["$k"]=1
  done < "$resume_keys_file"
  rm -f "$resume_keys_file"

  echo "Resume mode: found $executed_cases existing cases (max case_id=$case_id), skipping completed combos."
else
  cat > "$RESULTS_CSV" <<'CSV'
case_id,status,error,threads,ngl,ctx,batch,ubatch,cache_k,cache_v,fa,prompt_tps_1,eval_tps_1,prompt_tps_2,eval_tps_2,fit_load_sec,log_file
CSV
fi

auto_skip=0
for threads in "${THREAD_ARR[@]}"; do
  for ngl in "${NGL_ARR[@]}"; do
    for ctx in "${CTX_ARR[@]}"; do
      for batch in "${BATCH_ARR[@]}"; do
        for ubatch in "${UBATCH_ARR[@]}"; do
          if (( ubatch > batch )); then
            continue
          fi
          for kv in "${KV_ARR[@]}"; do
            IFS=':' read -r cache_k cache_v <<< "$kv"
            for fa in "${FA_ARR[@]}"; do
              combo_key="${threads}|${ngl}|${ctx}|${batch}|${ubatch}|${cache_k}|${cache_v}|${fa}"

              if [[ -n "${DONE_KEYS[$combo_key]+x}" ]]; then
                resume_skipped=$((resume_skipped + 1))
                continue
              fi

              # Known constraint in llama.cpp: quantized V cache requires flash attention.
              if [[ "$fa" == "off" && "$cache_v" != "f16" && "$cache_v" != "f32" && "$cache_v" != "bf16" ]]; then
                auto_skip=$((auto_skip + 1))
                continue
              fi

              if (( MAX_CASES > 0 && executed_cases >= MAX_CASES )); then
                echo "Reached MAX_CASES=$MAX_CASES, stopping."
                break 7
              fi

              case_id=$((case_id + 1))
              executed_cases=$((executed_cases + 1))

              echo "[case $case_id] t=$threads ngl=$ngl ctx=$ctx b=$batch ub=$ubatch kv=$cache_k/$cache_v fa=$fa"
              run_case "$case_id" "$threads" "$ngl" "$ctx" "$batch" "$ubatch" "$cache_k" "$cache_v" "$fa"
              DONE_KEYS["$combo_key"]=1
            done
          done
        done
      done
    done
  done
done

echo
echo "Completed cases: $executed_cases"
echo "Auto-skipped invalid combos: $auto_skip"
echo "Resume-skipped completed combos: $resume_skipped"

echo
echo "Top 15 by eval_tps_2"
{
  head -n1 "$RESULTS_CSV"
  tail -n +2 "$RESULTS_CSV" | awk -F, '$2=="ok" {print}' | sort -t, -k15,15nr | head -n 15
} | column -s, -t

python3 - "$RESULTS_CSV" "$BEST_CSV" <<'PY'
import csv
import sys

results_csv = sys.argv[1]
best_csv = sys.argv[2]

rows = []
with open(results_csv, newline='', encoding='utf-8') as f:
  reader = csv.DictReader(f)
  for row in reader:
    if row.get('status') != 'ok':
      continue
    try:
      row['_eval2'] = float(row.get('eval_tps_2', '0') or 0)
      row['_eval1'] = float(row.get('eval_tps_1', '0') or 0)
      row['_prompt2'] = float(row.get('prompt_tps_2', '0') or 0)
      row['_ngl'] = int(row.get('ngl', '0') or 0)
      row['_ctx'] = int(row.get('ctx', '0') or 0)
      row['_batch'] = int(row.get('batch', '0') or 0)
      row['_ubatch'] = int(row.get('ubatch', '0') or 0)
    except ValueError:
      continue
    if row['_eval2'] <= 0:
      continue
    rows.append(row)

rows.sort(key=lambda r: (r['_eval2'], r['_prompt2'], r['_eval1']), reverse=True)

low_vram = [
  r for r in rows
  if r['_ngl'] <= 8 and r.get('cache_k') == 'q8_0' and r.get('cache_v') == 'q8_0'
]

with open(best_csv, 'w', newline='', encoding='utf-8') as f:
  fields = [
    'rank_group', 'rank', 'case_id', 'eval_tps_2', 'prompt_tps_2',
    'eval_tps_1', 'prompt_tps_1', 'threads', 'ngl', 'ctx', 'batch',
    'ubatch', 'cache_k', 'cache_v', 'fa', 'fit_load_sec', 'log_file'
  ]
  w = csv.DictWriter(f, fieldnames=fields)
  w.writeheader()

  for i, r in enumerate(rows[:20], 1):
    w.writerow({
      'rank_group': 'overall',
      'rank': i,
      'case_id': r.get('case_id'),
      'eval_tps_2': r.get('eval_tps_2'),
      'prompt_tps_2': r.get('prompt_tps_2'),
      'eval_tps_1': r.get('eval_tps_1'),
      'prompt_tps_1': r.get('prompt_tps_1'),
      'threads': r.get('threads'),
      'ngl': r.get('ngl'),
      'ctx': r.get('ctx'),
      'batch': r.get('batch'),
      'ubatch': r.get('ubatch'),
      'cache_k': r.get('cache_k'),
      'cache_v': r.get('cache_v'),
      'fa': r.get('fa'),
      'fit_load_sec': r.get('fit_load_sec'),
      'log_file': r.get('log_file'),
    })

  for i, r in enumerate(low_vram[:10], 1):
    w.writerow({
      'rank_group': 'low_vram',
      'rank': i,
      'case_id': r.get('case_id'),
      'eval_tps_2': r.get('eval_tps_2'),
      'prompt_tps_2': r.get('prompt_tps_2'),
      'eval_tps_1': r.get('eval_tps_1'),
      'prompt_tps_1': r.get('prompt_tps_1'),
      'threads': r.get('threads'),
      'ngl': r.get('ngl'),
      'ctx': r.get('ctx'),
      'batch': r.get('batch'),
      'ubatch': r.get('ubatch'),
      'cache_k': r.get('cache_k'),
      'cache_v': r.get('cache_v'),
      'fa': r.get('fa'),
      'fit_load_sec': r.get('fit_load_sec'),
      'log_file': r.get('log_file'),
    })

print('NO_VALID_ROWS' if not rows else f'OVERALL_BEST_CASE={rows[0].get("case_id")} OVERALL_BEST_EVAL2={rows[0].get("eval_tps_2")}')
print('NO_LOW_VRAM_ROWS' if not low_vram else f'LOW_VRAM_BEST_CASE={low_vram[0].get("case_id")} LOW_VRAM_BEST_EVAL2={low_vram[0].get("eval_tps_2")}')
PY

echo
echo "Artifacts:"
echo "- $META_TXT"
echo "- $RESULTS_CSV"
echo "- $BEST_CSV"
if [[ -f "$LLAMA_BENCH_LOG" ]]; then
  echo "- $LLAMA_BENCH_LOG"
fi

#  MODE=exhaustive OUT_DIR=/home/nav_wsl/code/learn_llama_cpp/archieve/qwen14b_regression_20260226_013044 RESUME=1  ./qwen14b_regression_benchmark.sh