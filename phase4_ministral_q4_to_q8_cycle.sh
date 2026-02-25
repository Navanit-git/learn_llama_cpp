#!/usr/bin/env bash
# Phase 4 query-driven quant benchmark for:
#   unsloth/Qwen3.5-27B-GGUF
#
# NEW in this version:
#   - CPU usage % snapshot (before/after per query)
#   - GPU temperature & power draw (nvidia-smi)
#   - Thermal throttling detection (clock speed drop)
#   - Time-to-first-token (TTFT) parsed from llama-cli log
#   - Model load time measured via bash timing
#   - Peak vs average GPU memory delta during inference
#   - Prefill t/s and generation t/s tracked separately (also in query CSV)
#   - Tool call test via llama-server /v1/chat/completions with tool schema
#   - Tool call test via llama-cli with crafted prompt
#   - Tool call success: both function name + arguments validated
#
# Quick start:
#   ./phase4_ministral_q4_to_q8_cycle.sh
#
#   VARIANTS_CSV="Q4_K_M,Q5_K_M" ./phase4_ministral_q4_to_q8_cycle.sh
#   KEEP_MODELS=1 INCLUDE_EXTRA_QUERIES=1 ./phase4_ministral_q4_to_q8_cycle.sh

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$SCRIPT_DIR/llama.cpp}"
BIN_DIR="${BIN_DIR:-$LLAMA_DIR/build/bin}"

LLAMA_CLI_BIN="${LLAMA_CLI_BIN:-$BIN_DIR/llama-cli}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$BIN_DIR/llama-server}"

# ─── Model identity ───────────────────────────────────────────────────────────
# Mistral defaults (kept for quick switch-back):
# MODEL_REPO="${MODEL_REPO:-unsloth/Ministral-3-14B-Instruct-2512-GGUF}"
# MODEL_PREFIX="${MODEL_PREFIX:-Ministral-3-14B-Instruct-2512}"

# Qwen3.5 defaults:
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-27B-GGUF}"
MODEL_PREFIX="${MODEL_PREFIX:-Qwen3.5-27B}"

# ─── Runtime defaults (tuned for 4GB VRAM) ────────────────────────────────────
THREADS="${THREADS:-10}"
N_GPU_LAYERS="${N_GPU_LAYERS:-15}"
N_CPU_MOE="${N_CPU_MOE:-0}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

N_PREDICT="${N_PREDICT:-64}"
INCLUDE_EXTRA_QUERIES="${INCLUDE_EXTRA_QUERIES:-0}"
QUERY_TIMEOUT_SEC="${QUERY_TIMEOUT_SEC:-6000}"

CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
RESPONSE_TEXT_SOURCE="${RESPONSE_TEXT_SOURCE:-server}"

# ─── Server config ────────────────────────────────────────────────────────────
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-19080}"
SERVER_BOOT_WAIT_SEC="${SERVER_BOOT_WAIT_SEC:-12}"

# ─── Cache / output ───────────────────────────────────────────────────────────
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llama.cpp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/archieve/phase4/ministral_q4_to_q8}"
KEEP_MODELS="${KEEP_MODELS:-0}"

# ─── GPU polling interval for peak memory tracking (seconds) ─────────────────
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-1}"

# ─── Variants ─────────────────────────────────────────────────────────────────
if [[ -n "${VARIANTS_CSV:-}" ]]; then
  IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
else
  VARIANTS=(
    "Q2_K"
    "Q3_K_M"
    "Q3_K_S"
    "Q4_0"
    "Q4_1"
    "Q4_K_M"
    "Q4_K_S"
    "Q5_K_M"
    "Q5_K_S"
    "Q6_K"
    "Q8_0"
    "UD-Q4_K_XL"
    "UD-Q5_K_XL"
    "UD-Q6_K_XL"
    "UD-Q8_K_XL"
  )
fi

# ─── Core queries ─────────────────────────────────────────────────────────────
QUERIES=(
  "Hi"
  "How many r are there in strawberries?"
)

if [[ "$INCLUDE_EXTRA_QUERIES" == "1" ]]; then
  QUERIES+=(
    "Give a one-line Linux tip for monitoring memory while running local LLM inference."
  )
fi

# ─── Tool-call test definition ────────────────────────────────────────────────
# Expected: model calls get_weather(location="Paris", unit="celsius")
TOOL_CALL_USER_MSG="What is the current weather in Paris in celsius?"
TOOL_EXPECTED_NAME="get_weather"
TOOL_EXPECTED_ARGS='{"location":"Paris","unit":"celsius"}'  # canonical lowercase keys

TOOL_SCHEMA='{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {"type": "string", "description": "City name"},
        "unit":     {"type": "string", "enum": ["celsius","fahrenheit"]}
      },
      "required": ["location","unit"]
    }
  }
}'

# llama-cli tool-call prompt (system + user merged for -p)
TOOL_CLI_PROMPT='[TOOL_CALLS] You have access to the following tool:
get_weather(location: string, unit: "celsius"|"fahrenheit") -> returns current weather.
Respond ONLY with a raw JSON object on one line like: {"name":"get_weather","arguments":{"location":"...","unit":"..."}}
User: What is the current weather in Paris in celsius?
Assistant:'

# ─── Helpers ──────────────────────────────────────────────────────────────────

require_bin() {
  local p="$1"
  if [[ ! -x "$p" ]]; then
    echo "ERROR: binary not found or not executable: $p"
    exit 1
  fi
}

hf_cache_model_path() {
  local repo="$1" model_file="$2"
  local repo_key="${repo//\//_}"
  echo "$CACHE_DIR/${repo_key}_${model_file}"
}

bytes_to_gib() {
  python3 -c "b=int('$1'); print(f'{b/(1024**3):.2f}')"
}

csv_escape() {
  local s="${1:-}"
  s="${s//$'\r'/ }"
  s="${s//$'\n'/\\n}"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

# ── Speed & timing parsers ────────────────────────────────────────────────────

extract_speed_from_cli_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
pp = tg = None
for line in text:
    m = re.search(r'\[\s*Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\|\s*Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s\s*\]', line)
    if m:
        pp, tg = m.group(1), m.group(2)
if pp is None or tg is None:
    for line in text:
        if 'prompt eval time' in line.lower():
            m = re.search(r'\(([0-9]+(?:\.[0-9]+)?)\s+tokens per second\)', line)
            if m: pp = m.group(1)
        elif re.search(r'(^|\s)eval time', line.lower()) and 'prompt eval time' not in line.lower():
            m = re.search(r'\(([0-9]+(?:\.[0-9]+)?)\s+tokens per second\)', line)
            if m: tg = m.group(1)
print((pp or 'NA') + ' ' + (tg or 'NA'))
PY
}

# Returns TTFT in milliseconds (time from start to first generated token)
extract_ttft_from_cli_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
ttft = None
for line in text:
    # llama.cpp logs: "llama_perf_context_print: ... time to first token = X ms"
    m = re.search(r'time.{0,10}first.{0,10}token\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*ms', line, re.IGNORECASE)
    if m:
        ttft = m.group(1)
        break
    # alternative: prompt eval time in ms
    m2 = re.search(r'prompt eval time\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*ms', line, re.IGNORECASE)
    if m2:
        ttft = m2.group(1)
print(ttft or 'NA')
PY
}

# Returns model load time in seconds from a log
extract_load_time_from_cli_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
load_ms = None
for line in text:
    m = re.search(r'load time\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*ms', line, re.IGNORECASE)
    if m:
        load_ms = float(m.group(1))
        break
if load_ms is not None:
    print(f"{load_ms/1000:.3f}")
else:
    print('NA')
PY
}

extract_cuda_memory_from_cli_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys
text = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
line = None
for row in text:
  # Newer logs may not include the words "memory breakdown" on the CUDA row,
  # so match any CUDA0 breakdown line that carries the numeric decomposition.
  if 'CUDA0' in row and '=' in row:
    line = row
if not line:
    print('NA NA NA NA'); raise SystemExit(0)
m = re.search(r'\((\d+)\s*=\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\)', line)
if not m:
    print('NA NA NA NA'); raise SystemExit(0)
# Output order: cuda_total_mb, cuda_model_mb, cuda_context_mb, cuda_compute_mb
total_m = re.search(r'\|\s*(\d+)\s*=\s*\d+\s*\+\s*\(', line)
total = total_m.group(1) if total_m else 'NA'
model, context, compute = m.group(2), m.group(3), m.group(4)
print(total, model, context, compute)
PY
}

# Extracts a readable assistant response from llama-cli logs.
# Returns a single-line string (or NA) suitable for CSV storage.
extract_llm_response_from_cli_log() {
  local log_file="$1"
  python3 - "$log_file" <<'PY'
import re, sys

try:
  lines = open(sys.argv[1], 'r', encoding='utf-8', errors='ignore').read().splitlines()
except Exception:
  print('NA')
  raise SystemExit(0)

out = []
for raw in lines:
  line = raw.strip()
  if not line:
    continue

  # Remove most infrastructure/noise lines
  if re.match(r'^(llama|ggml|main|build|system_info|sampling|generate|common)\b', line, re.IGNORECASE):
    continue
  if line.startswith(('Device ', 'Consider compiling with ', 'model      :', 'modalities :', 'available commands:')):
    continue
  if line.startswith('/exit ') or line.startswith('/regen') or line.startswith('/clear') or line.startswith('/read'):
    continue
  if line in ('Loading model...', 'Exiting...'):
    continue
  if 'memory breakdown' in line:
    continue
  if re.match(r'^\[\s*Prompt:.*\|\s*Generation:.*\]$', line):
    continue

  # Strip REPL-style markers
  if line.startswith('> '):
    line = line[2:].strip()

  # Keep actual content, drop chat role labels
  if line.startswith('User:') or line.startswith('Assistant:'):
    continue

  out.append(line)

if not out:
  print('NA')
else:
  text = ' '.join(out)
  # keep CSV readable
  print(text[:4000])
PY
}

# Extract server assistant text content (if present) from OpenAI-style response JSON.
extract_server_response_text() {
  local response_json="$1"
  python3 - "$response_json" <<'PY'
import json, sys
try:
  data = json.load(open(sys.argv[1], 'r', encoding='utf-8', errors='ignore'))
  choices = data.get('choices', [])
  msg = choices[0].get('message', {}) if choices else {}
  content = msg.get('content', '')
  if isinstance(content, list):
    content = ' '.join(str(x) for x in content)
  content = str(content).strip()
  print(content[:4000] if content else 'NA')
except Exception:
  print('NA')
PY
}

# Query llama-server once and return assistant response text.
# Returns: response text or NA
fetch_server_response_for_query() {
  local model_path="$1"
  local query_text="$2"
  local server_log="$3"
  local port="$4"
  local response_file="$5"

  set +e
  "$LLAMA_SERVER_BIN" \
    -m "$model_path" \
    --host "$SERVER_HOST" \
    --port "$port" \
    -t "$THREADS" \
    -ngl "$N_GPU_LAYERS" \
    -ncmoe "$N_CPU_MOE" \
    --ctx-size "$CTX_SIZE" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    --log-prefix \
    --log-file "$server_log" \
    >/dev/null 2>&1 &
  local server_pid=$!

  sleep "$SERVER_BOOT_WAIT_SEC"

  local smoke_rc=1
  local http_rc=1

  curl -s "http://${SERVER_HOST}:${port}/v1/models" > /dev/null
  smoke_rc=$?

  if [[ $smoke_rc -eq 0 ]]; then
    local payload
    payload="$(python3 - "$query_text" "$N_PREDICT" <<'PY'
import json, sys
q = sys.argv[1]
n_predict = int(sys.argv[2])
obj = {
  'model': 'local',
  'messages': [{'role': 'user', 'content': q}],
  'max_tokens': n_predict,
  'temperature': 0.2,
  'top_p': 0.95,
}
print(json.dumps(obj))
PY
)"

    curl -s -X POST \
      "http://${SERVER_HOST}:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      -o "$response_file"
    http_rc=$?
  fi

  kill "$server_pid" >/dev/null 2>&1
  wait "$server_pid" >/dev/null 2>&1
  set -e

  if [[ $smoke_rc -eq 0 && $http_rc -eq 0 && -f "$response_file" ]]; then
    extract_server_response_text "$response_file"
  else
    echo "NA"
  fi
}

# ── System snapshots ──────────────────────────────────────────────────────────

# Returns: used_mb,total_mb,util_pct,temp_c,power_w,clock_mhz
gpu_snapshot_full() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.current.graphics \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
  else
    echo "NA,NA,NA,NA,NA,NA"
  fi
}

ram_available_mb() {
  free -m | awk '/^Mem:/ {print $7}'
}

# CPU usage: average across all cores over a 1-second window
cpu_usage_pct() {
  if command -v mpstat >/dev/null 2>&1; then
    mpstat 1 1 2>/dev/null | awk '/Average/ && /all/ {printf "%.1f", 100 - $NF}'
  elif [[ -f /proc/stat ]]; then
    python3 - <<'PY'
import time
def read_cpu():
    with open('/proc/stat') as f:
        vals = list(map(int, f.readline().split()[1:]))
    idle = vals[3]
    total = sum(vals)
    return idle, total
i1, t1 = read_cpu(); time.sleep(1); i2, t2 = read_cpu()
dt = t2 - t1; di = i2 - i1
print(f"{100*(dt-di)/dt:.1f}" if dt>0 else "NA")
PY
  else
    echo "NA"
  fi
}

# Background GPU memory poller: writes samples to a tmp file, killed after inference
# NOTE: must be called directly (not via $(...)) so the background job is a real
# child of the current shell and can be waited on / killed reliably.
start_gpu_mem_poll() {
  local poll_file="$1"
  (
    while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 | tr -d ' ' >> "$poll_file"
      sleep "$GPU_POLL_INTERVAL"
    done
  ) &
  # do NOT echo $! here; caller captures $! directly after this call
}

stop_gpu_mem_poll() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

# Given a file of GPU mem samples (one int per line), returns: peak_mb avg_mb delta_mb
compute_gpu_mem_stats() {
  local poll_file="$1"
  python3 - "$poll_file" <<'PY'
import sys, statistics
try:
    vals = [int(l.strip()) for l in open(sys.argv[1]) if l.strip().isdigit()]
    if not vals:
        print("NA NA NA"); raise SystemExit
    peak = max(vals)
    avg  = statistics.mean(vals)
    delta = peak - min(vals)
    print(f"{peak} {avg:.1f} {delta}")
except Exception:
    print("NA NA NA")
PY
}

# Thermal throttle detection: compare baseline clock to post-inference clock
# Returns "yes" / "no" / "NA"
detect_thermal_throttle() {
  local clock_before="$1"
  local clock_after="$2"
  python3 - "$clock_before" "$clock_after" <<'PY'
import sys
try:
    b = float(sys.argv[1]); a = float(sys.argv[2])
    # >15% drop in GPU clock is considered throttling
    print("yes" if b > 0 and (b - a) / b > 0.15 else "no")
except Exception:
    print("NA")
PY
}

# ── Tool call validators ───────────────────────────────────────────────────────

# Validates llama-server tool call response (OpenAI-style JSON)
# Returns: name_ok(0/1) args_ok(0/1) tool_call_status(pass/fail/error)
validate_server_tool_call() {
  local response_json="$1"
  python3 - "$response_json" "$TOOL_EXPECTED_NAME" "$TOOL_EXPECTED_ARGS" <<'PY'
import json, sys

resp_file   = sys.argv[1]
exp_name    = sys.argv[2]
exp_args_s  = sys.argv[3]

try:
    with open(resp_file) as f:
        data = json.load(f)
    exp_args = json.loads(exp_args_s)

    choices   = data.get("choices", [])
    msg       = choices[0].get("message", {}) if choices else {}
    tool_calls = msg.get("tool_calls", [])

    if not tool_calls:
        print("0 0 fail"); raise SystemExit

    tc         = tool_calls[0]
    got_name   = tc.get("function", {}).get("name", "")
    got_args_s = tc.get("function", {}).get("arguments", "{}")
    got_args   = json.loads(got_args_s)

    name_ok = int(got_name.strip().lower() == exp_name.strip().lower())

    # Normalize string values to lowercase for comparison
    def norm(d):
        return {k: v.lower() if isinstance(v, str) else v for k, v in d.items()}

    args_ok = int(norm(got_args) == norm(exp_args))
    status  = "pass" if name_ok and args_ok else "fail"
    print(f"{name_ok} {args_ok} {status}")
except Exception as e:
    print(f"0 0 error:{e}")
PY
}

# Validates llama-cli raw text output for a JSON tool call
validate_cli_tool_call() {
  local raw_text="$1"
  python3 - "$raw_text" "$TOOL_EXPECTED_NAME" "$TOOL_EXPECTED_ARGS" <<'PY'
import json, sys, re

raw_file   = sys.argv[1]
exp_name   = sys.argv[2]
exp_args_s = sys.argv[3]

try:
    text     = open(raw_file, encoding='utf-8', errors='ignore').read()
    exp_args = json.loads(exp_args_s)

    # Focus on content after assistant marker when present
    marker = 'Assistant:'
    start_text = text.split(marker, 1)[1] if marker in text else text

    # Extract first valid JSON object using a decoder to avoid greedy regex issues
    dec = json.JSONDecoder()
    obj = None
    for m in re.finditer(r'\{', start_text):
        i = m.start()
        try:
            candidate, _ = dec.raw_decode(start_text[i:])
            if isinstance(candidate, dict):
                obj = candidate
                break
        except Exception:
            continue

    if obj is None:
        print("0 0 fail:no_json"); raise SystemExit

    got_name = obj.get("name", "")
    got_args = obj.get("arguments", {})
    if isinstance(got_args, str):
        got_args = json.loads(got_args)

    name_ok = int(got_name.strip().lower() == exp_name.strip().lower())

    def norm(d):
        return {k: v.lower() if isinstance(v, str) else v for k, v in d.items()}

    args_ok = int(norm(got_args) == norm(exp_args))
    status  = "pass" if name_ok and args_ok else "fail"
    print(f"{name_ok} {args_ok} {status}")
except Exception as e:
    print(f"0 0 error:{e}")
PY
}

# ── Download helpers ───────────────────────────────────────────────────────────

find_model_path() {
  local model_file="$1"
  find "$CACHE_DIR" -type f \
    \( -name "$model_file" -o -name "*_${model_file}" \) \
    ! -name "*.downloadInProgress" | head -n 1
}

download_model_file() {
  local repo="$1" model_file="$2" out_log="$3"
  mkdir -p "$CACHE_DIR"
  local final_path; final_path="$(hf_cache_model_path "$repo" "$model_file")"
  local tmp_path="${final_path}.downloadInProgress"
  local url="https://huggingface.co/${repo}/resolve/main/${model_file}"
  if [[ -f "$final_path" ]]; then
    echo "Already present: $final_path" > "$out_log"; return 0
  fi
  { echo "Downloading: $url"; echo "Target: $final_path"
    curl -L --fail --retry 3 --retry-delay 2 -C - -o "$tmp_path" "$url"
    mv -f "$tmp_path" "$final_path"
  } > "$out_log" 2>&1
}

# ── Server smoke + tool call via server ───────────────────────────────────────

run_server_and_tool_test() {
  local model_path="$1"
  local server_log="$2"
  local port="$3"
  local tool_resp_file="$4"

  set +e
  "$LLAMA_SERVER_BIN" \
    -m "$model_path" \
    --host "$SERVER_HOST" \
    --port "$port" \
    -t "$THREADS" \
    -ngl "$N_GPU_LAYERS" \
    -ncmoe "$N_CPU_MOE" \
    --ctx-size "$CTX_SIZE" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    --log-prefix \
    --log-file "$server_log" \
    >/dev/null 2>&1 &
  local server_pid=$!

  sleep "$SERVER_BOOT_WAIT_SEC"

  # Basic smoke check
  local smoke_rc
  curl -s "http://${SERVER_HOST}:${port}/v1/models" > /dev/null
  smoke_rc=$?

  # Tool call via /v1/chat/completions
  local tool_http_rc=1
  if [[ $smoke_rc -eq 0 ]]; then
    local payload
    payload="$(python3 -c "
import json
tool = json.loads('''$TOOL_SCHEMA''')
msg  = {'role':'user','content':'$TOOL_CALL_USER_MSG'}
obj  = {'model':'local','messages':[msg],'tools':[tool],'tool_choice':'auto','max_tokens':128,'temperature':0}
print(json.dumps(obj))
")"
    curl -s -X POST \
      "http://${SERVER_HOST}:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      -o "$tool_resp_file"
    tool_http_rc=$?
  fi

  kill "$server_pid" >/dev/null 2>&1
  wait "$server_pid" >/dev/null 2>&1
  set -e

  local server_status="failed"
  [[ $smoke_rc -eq 0 ]] && server_status="ok"

  local tool_server_name_ok=0 tool_server_args_ok=0 tool_server_status="skipped"
  if [[ $smoke_rc -eq 0 && $tool_http_rc -eq 0 && -f "$tool_resp_file" ]]; then
    read -r tool_server_name_ok tool_server_args_ok tool_server_status \
      < <(validate_server_tool_call "$tool_resp_file")
  fi

  echo "$server_status $tool_server_name_ok $tool_server_args_ok $tool_server_status"
}

# ─── Pre-flight ───────────────────────────────────────────────────────────────
require_bin "$LLAMA_CLI_BIN"
require_bin "$LLAMA_SERVER_BIN"

# Guard: cap N_GPU_LAYERS to a safe value for 4GB VRAM cards
# 7.67 GiB Q4_K_M at ~40 layers → ~200 MB/layer; 15 layers ≈ 3.0 GB leaving headroom for KV cache
MAX_SAFE_GPU_LAYERS=15
if [[ "$N_GPU_LAYERS" -gt "$MAX_SAFE_GPU_LAYERS" ]]; then
  echo "WARN: N_GPU_LAYERS=$N_GPU_LAYERS exceeds safe limit for 4 GB VRAM. Capping to $MAX_SAFE_GPU_LAYERS." >&2
  N_GPU_LAYERS=$MAX_SAFE_GPU_LAYERS
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p \
  "$OUTPUT_ROOT/bench_logs" \
  "$OUTPUT_ROOT/server_logs" \
  "$OUTPUT_ROOT/results" \
  "$OUTPUT_ROOT/query_logs" \
  "$OUTPUT_ROOT/tool_logs"

SUMMARY_CSV="$OUTPUT_ROOT/results/summary_${TIMESTAMP}.csv"
QUERY_CSV="$OUTPUT_ROOT/results/query_details_${TIMESTAMP}.csv"
TOOL_CSV="$OUTPUT_ROOT/results/tool_call_${TIMESTAMP}.csv"
RUN_LOG="$OUTPUT_ROOT/results/run_${TIMESTAMP}.log"
TERMINAL_LOG="$OUTPUT_ROOT/results/terminal_${TIMESTAMP}.log"

# Mirror ALL stdout+stderr to TERMINAL_LOG (the full "what you'd see in the terminal")
exec > >(tee -a "$TERMINAL_LOG") 2>&1
echo "=== Terminal log started: $TERMINAL_LOG ==="

# ─── CSV headers ──────────────────────────────────────────────────────────────
cat > "$SUMMARY_CSV" <<'CSV'
variant,model_file,size_bytes,size_gib,download_status,model_load_time_s,queries_ok,avg_prefill_tps,avg_generation_tps,server_status,server_log,tool_server_name_ok,tool_server_args_ok,tool_server_status,tool_cli_name_ok,tool_cli_args_ok,tool_cli_status
CSV

cat > "$QUERY_CSV" <<'CSV'
variant,query_id,query_text,cli_status,prefill_tps,generation_tps,ttft_ms,cuda_total_mb,cuda_model_mb,cuda_context_mb,cuda_compute_mb,gpu_mem_peak_mb,gpu_mem_avg_mb,gpu_mem_delta_mb,gpu_before_mb,gpu_total_mb,gpu_util_before_pct,gpu_temp_before_c,gpu_power_before_w,gpu_clock_before_mhz,gpu_after_mb,gpu_util_after_pct,gpu_temp_after_c,gpu_power_after_w,gpu_clock_after_mhz,thermal_throttle,cpu_usage_before_pct,cpu_usage_after_pct,ram_avail_before_mb,ram_avail_after_mb,response_text,query_log
CSV

cat > "$TOOL_CSV" <<'CSV'
variant,method,tool_call_user_msg,expected_name,expected_args,name_ok,args_ok,status,response_text,raw_log
CSV

# ─── Main loop ────────────────────────────────────────────────────────────────
echo "=== Phase 4 Ministral Q4->Q8 cycle (enhanced) ===" | tee -a "$RUN_LOG"
echo "Model repo : $MODEL_REPO"  | tee -a "$RUN_LOG"
echo "Output root: $OUTPUT_ROOT" | tee -a "$RUN_LOG"
echo "Timestamp  : $TIMESTAMP"   | tee -a "$RUN_LOG"

variant_index=0
for variant in "${VARIANTS[@]}"; do
  variant_index=$((variant_index + 1))
  model_file="${MODEL_PREFIX}-${variant}.gguf"
  server_log="$OUTPUT_ROOT/server_logs/${variant}.log"
  tool_server_resp="$OUTPUT_ROOT/tool_logs/${variant}_server_resp.json"
  tool_cli_log="$OUTPUT_ROOT/tool_logs/${variant}_cli.log"

  echo | tee -a "$RUN_LOG"
  echo "--- [$variant_index/${#VARIANTS[@]}] $variant ---" | tee -a "$RUN_LOG"
  echo "Model file: $model_file" | tee -a "$RUN_LOG"

  # ── Download ──────────────────────────────────────────────────────────────
  download_status="ok"
  if ! download_model_file \
      "$MODEL_REPO" "$model_file" \
      "$OUTPUT_ROOT/bench_logs/${variant}_download.log"; then
    download_status="failed"
  fi

  model_path="$(hf_cache_model_path "$MODEL_REPO" "$model_file")"
  if [[ ! -f "$model_path" ]]; then
    model_path="$(find_model_path "$model_file" || true)"
  fi
  if [[ -z "$model_path" ]]; then
    echo "WARN: could not locate model in $CACHE_DIR" | tee -a "$RUN_LOG"
    download_status="failed"
    echo "${variant},${model_file},0,0.00,${download_status},NA,0,NA,NA,skipped,${server_log},0,0,skipped,0,0,skipped" >> "$SUMMARY_CSV"
    continue
  fi

  size_bytes="$(stat -c%s "$model_path")"
  size_gib="$(bytes_to_gib "$size_bytes")"
  echo "Model path : $model_path" | tee -a "$RUN_LOG"
  echo "Size       : ${size_bytes} bytes (${size_gib} GiB)" | tee -a "$RUN_LOG"

  # ── Measure model load time with a minimal dry-run ────────────────────────
  # We run llama-cli with -n 1 and parse load time from its log
  # load_log="$OUTPUT_ROOT/query_logs/${variant}_loadtime.log"
  # load_time_start=$SECONDS
  # timeout 12000 "$LLAMA_CLI_BIN" \
  #   -m "$model_path" \
  #   -p "hi" \
  #   -n 1 \
  #   -t "$THREADS" \
  #   -ngl "$N_GPU_LAYERS" \
  #   -ncmoe "$N_CPU_MOE" \
  #   --ctx-size 512 \
  #   -b 512 -ub 512 \
  #   --no-display-prompt \
  #   > "$load_log" 2>&1 || true
  # load_time_end=$SECONDS
  # model_load_time_s="$(extract_load_time_from_cli_log "$load_log")"
  # # Fallback to wall-clock if log doesn't contain load time
  # if [[ "$model_load_time_s" == "NA" ]]; then
  #   model_load_time_s="$((load_time_end - load_time_start))"
  # fi
  # echo "Model load time: ${model_load_time_s}s" | tee -a "$RUN_LOG"
  model_load_time_s="NA"

  # ── Per-query inference ───────────────────────────────────────────────────
  queries_ok=0
  pp_sum=0 tg_sum=0 pp_count=0 tg_count=0

  query_idx=0
  for q in "${QUERIES[@]}"; do
    query_idx=$((query_idx + 1))
    query_log="$OUTPUT_ROOT/query_logs/${variant}_q${query_idx}.log"
    query_resp_server_log="$OUTPUT_ROOT/server_logs/${variant}_q${query_idx}_response.log"
    query_resp_server_json="$OUTPUT_ROOT/query_logs/${variant}_q${query_idx}_server_response.json"
    gpu_poll_file="$(mktemp /tmp/gpu_poll_XXXXXX.txt)"

    # -- Before-snapshots
    cpu_before="$(cpu_usage_pct)"
    gpu_before_raw="$(gpu_snapshot_full)"
    ram_before="$(ram_available_mb)"

    gpu_before_mb="$(echo    "$gpu_before_raw" | cut -d',' -f1)"
    gpu_total_mb="$(echo     "$gpu_before_raw" | cut -d',' -f2)"
    gpu_util_before="$(echo  "$gpu_before_raw" | cut -d',' -f3)"
    gpu_temp_before="$(echo  "$gpu_before_raw" | cut -d',' -f4)"
    gpu_power_before="$(echo "$gpu_before_raw" | cut -d',' -f5)"
    gpu_clock_before="$(echo "$gpu_before_raw" | cut -d',' -f6)"

    # Start background GPU memory poller (direct call, NOT $(...), so $! is a real child)
    start_gpu_mem_poll "$gpu_poll_file"
    poll_pid=$!

    # -- Run inference
    cli_status="ok"
    if ! timeout "$QUERY_TIMEOUT_SEC" "$LLAMA_CLI_BIN" \
        -m "$model_path" \
        -p "$q" \
        -st \
        --simple-io \
        -n "$N_PREDICT" \
        -t "$THREADS" \
        -ngl "$N_GPU_LAYERS" \
        -ncmoe "$N_CPU_MOE" \
        --ctx-size "$CTX_SIZE" \
        -b "$BATCH_SIZE" \
        -ub "$UBATCH_SIZE" \
        -ctk "$CACHE_TYPE_K" \
        -ctv "$CACHE_TYPE_V" \
        -fa auto \
        --temp 0.2 \
        --top-p 0.95 \
        --no-display-prompt \
        < /dev/null > "$query_log" 2>&1; then
      cli_status="failed"
    fi

    stop_gpu_mem_poll "$poll_pid"

    # -- After-snapshots
    cpu_after="$(cpu_usage_pct)"
    gpu_after_raw="$(gpu_snapshot_full)"
    ram_after="$(ram_available_mb)"

    gpu_after_mb="$(echo    "$gpu_after_raw" | cut -d',' -f1)"
    gpu_util_after="$(echo  "$gpu_after_raw" | cut -d',' -f3)"
    gpu_temp_after="$(echo  "$gpu_after_raw" | cut -d',' -f4)"
    gpu_power_after="$(echo "$gpu_after_raw" | cut -d',' -f5)"
    gpu_clock_after="$(echo "$gpu_after_raw" | cut -d',' -f6)"

    # -- Parse metrics from log
    read -r prefill_tps generation_tps  < <(extract_speed_from_cli_log     "$query_log")
    ttft_ms="$(extract_ttft_from_cli_log "$query_log")"
    read -r cuda_total_mb cuda_model_mb cuda_context_mb cuda_compute_mb \
                                          < <(extract_cuda_memory_from_cli_log "$query_log")
    read -r gpu_peak_mb gpu_avg_mb gpu_delta_mb \
                                          < <(compute_gpu_mem_stats           "$gpu_poll_file")
    response_text="NA"
    if [[ "$RESPONSE_TEXT_SOURCE" == "server" ]]; then
      query_response_port="$((SERVER_PORT_BASE + 1000 + variant_index * 100 + query_idx))"
      response_text="$(fetch_server_response_for_query "$model_path" "$q" "$query_resp_server_log" "$query_response_port" "$query_resp_server_json")"
    fi
    if [[ -z "$response_text" || "$response_text" == "NA" ]]; then
      response_text="$(extract_llm_response_from_cli_log "$query_log")"
    fi

    # Thermal throttle: compare clock before vs after
    thermal_throttle="$(detect_thermal_throttle "$gpu_clock_before" "$gpu_clock_after")"

    rm -f "$gpu_poll_file"

    [[ "$cli_status" == "ok" ]] && queries_ok=$((queries_ok + 1))

    if [[ "$prefill_tps" != "NA" ]]; then
      pp_sum="$(python3 -c "print(float('$pp_sum') + float('$prefill_tps'))")"
      pp_count=$((pp_count + 1))
    fi
    if [[ "$generation_tps" != "NA" ]]; then
      tg_sum="$(python3 -c "print(float('$tg_sum') + float('$generation_tps'))")"
      tg_count=$((tg_count + 1))
    fi

    # Write query CSV row
    echo "${variant},${query_idx},$(csv_escape "$q"),${cli_status},${prefill_tps},${generation_tps},${ttft_ms},${cuda_total_mb},${cuda_model_mb},${cuda_context_mb},${cuda_compute_mb},${gpu_peak_mb},${gpu_avg_mb},${gpu_delta_mb},${gpu_before_mb},${gpu_total_mb},${gpu_util_before},${gpu_temp_before},${gpu_power_before},${gpu_clock_before},${gpu_after_mb},${gpu_util_after},${gpu_temp_after},${gpu_power_after},${gpu_clock_after},${thermal_throttle},${cpu_before},${cpu_after},${ram_before},${ram_after},$(csv_escape "$response_text"),$(csv_escape "$query_log")" \
      >> "$QUERY_CSV"

    echo "  Q${query_idx} status=${cli_status} | prefill=${prefill_tps} t/s | gen=${generation_tps} t/s | TTFT=${ttft_ms}ms | throttle=${thermal_throttle} | GPU peak=${gpu_peak_mb}MB delta=${gpu_delta_mb}MB | temp=${gpu_temp_after}°C | power=${gpu_power_after}W" \
      | tee -a "$RUN_LOG"
  done

  # ── Averages ──────────────────────────────────────────────────────────────
  avg_prefill_tps="NA"
  avg_generation_tps="NA"
  [[ $pp_count -gt 0 ]] && avg_prefill_tps="$(python3 -c "print(f'{float(\"$pp_sum\")/int(\"$pp_count\"):.2f}')")"
  [[ $tg_count -gt 0 ]] && avg_generation_tps="$(python3 -c "print(f'{float(\"$tg_sum\")/int(\"$tg_count\"):.2f}')")"

  # ── Server smoke + server tool call ───────────────────────────────────────
  port="$((SERVER_PORT_BASE + variant_index))"
  read -r server_status tool_server_name_ok tool_server_args_ok tool_server_status \
    < <(run_server_and_tool_test "$model_path" "$server_log" "$port" "$tool_server_resp") || true

  tool_server_response_text="$(extract_server_response_text "$tool_server_resp")"
  echo "${variant},server,$(csv_escape "$TOOL_CALL_USER_MSG"),$(csv_escape "$TOOL_EXPECTED_NAME"),$(csv_escape "$TOOL_EXPECTED_ARGS"),${tool_server_name_ok},${tool_server_args_ok},${tool_server_status},$(csv_escape "$tool_server_response_text"),$(csv_escape "$tool_server_resp")" \
    >> "$TOOL_CSV"

  # ── Show server tool call raw response in terminal
  echo "  --- Tool call (server) raw response ---" | tee -a "$RUN_LOG"
  if [[ -f "$tool_server_resp" ]]; then
    python3 -c "
import json, sys
try:
    d = json.load(open('$tool_server_resp'))
    print(json.dumps(d, indent=2))
except Exception:
    print(open('$tool_server_resp').read())
" | tee -a "$RUN_LOG"
  else
    echo "  (no response file)" | tee -a "$RUN_LOG"
  fi

  # ── CLI tool call ─────────────────────────────────────────────────────────
  tool_cli_status="failed"
  tool_cli_name_ok=0
  tool_cli_args_ok=0

  set +e
  timeout --kill-after=10s "$QUERY_TIMEOUT_SEC" "$LLAMA_CLI_BIN" \
    -m "$model_path" \
    -p "$TOOL_CLI_PROMPT" \
    -st \
    --simple-io \
    -n 128 \
    -t "$THREADS" \
    -ngl "$N_GPU_LAYERS" \
    -ncmoe "$N_CPU_MOE" \
    --ctx-size "$CTX_SIZE" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -ctk "$CACHE_TYPE_K" \
    -ctv "$CACHE_TYPE_V" \
    --temp 0.0 \
    --no-display-prompt \
    < /dev/null > "$tool_cli_log" 2>&1
  tool_cli_rc=$?
  set -e

  if [[ $tool_cli_rc -eq 0 ]]; then
    tool_cli_status="ok"
  elif [[ $tool_cli_rc -eq 124 || $tool_cli_rc -eq 137 ]]; then
    tool_cli_status="timeout"
  else
    tool_cli_status="failed"
  fi

  echo "  CLI tool call exit: rc=${tool_cli_rc} status=${tool_cli_status}" | tee -a "$RUN_LOG"

  # ── Show CLI tool call raw output in terminal
  echo "  --- Tool call (cli) raw model output ---" | tee -a "$RUN_LOG"
  grep -v '^llama\|^ggml\|^main\|^build\|^system_info\|^sampling\|^generate\|^[[:space:]]*$' "$tool_cli_log" 2>/dev/null \
    | tee -a "$RUN_LOG" || true
  echo "  ---" | tee -a "$RUN_LOG"

  if [[ "$tool_cli_status" == "ok" ]]; then
    validator_out="$(validate_cli_tool_call "$tool_cli_log" 2>/dev/null || true)"
    if [[ -n "$validator_out" ]]; then
      read -r tool_cli_name_ok tool_cli_args_ok tool_cli_status <<< "$validator_out"
    else
      tool_cli_name_ok=0
      tool_cli_args_ok=0
      tool_cli_status="error:validator_no_output"
    fi
  fi

  tool_cli_response_text="$(extract_llm_response_from_cli_log "$tool_cli_log")"
  echo "${variant},cli,$(csv_escape "$TOOL_CALL_USER_MSG"),$(csv_escape "$TOOL_EXPECTED_NAME"),$(csv_escape "$TOOL_EXPECTED_ARGS"),${tool_cli_name_ok},${tool_cli_args_ok},${tool_cli_status},$(csv_escape "$tool_cli_response_text"),$(csv_escape "$tool_cli_log")" \
    >> "$TOOL_CSV"

  echo "  Server: ${server_status} | Tool (server): name=${tool_server_name_ok} args=${tool_server_args_ok} -> ${tool_server_status}" | tee -a "$RUN_LOG"
  echo "  Tool (cli): name=${tool_cli_name_ok} args=${tool_cli_args_ok} -> ${tool_cli_status}" | tee -a "$RUN_LOG"

  # ── Summary row ───────────────────────────────────────────────────────────
  echo "${variant},${model_file},${size_bytes},${size_gib},${download_status},${model_load_time_s},${queries_ok},${avg_prefill_tps},${avg_generation_tps},${server_status},${server_log},${tool_server_name_ok},${tool_server_args_ok},${tool_server_status},${tool_cli_name_ok},${tool_cli_args_ok},${tool_cli_status}" \
    >> "$SUMMARY_CSV"

  # ── Cleanup ───────────────────────────────────────────────────────────────
  if [[ "$KEEP_MODELS" != "1" ]]; then
    rm -f "$model_path"
    echo "  Removed model from cache: $model_path" | tee -a "$RUN_LOG"
  else
    echo "  KEEP_MODELS=1 -> model retained: $model_path" | tee -a "$RUN_LOG"
  fi

done

echo | tee -a "$RUN_LOG"
echo "=== Completed ===" | tee -a "$RUN_LOG"
echo "Summary CSV    : $SUMMARY_CSV"  | tee -a "$RUN_LOG"
echo "Query CSV      : $QUERY_CSV"    | tee -a "$RUN_LOG"
echo "Tool call CSV  : $TOOL_CSV"     | tee -a "$RUN_LOG"
echo "Run log        : $RUN_LOG"      | tee -a "$RUN_LOG"