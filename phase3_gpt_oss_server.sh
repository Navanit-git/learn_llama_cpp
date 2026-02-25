#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER_BIN_DEFAULT="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"

MODEL_REPO="ggml-org/gpt-oss-20b-GGUF"
CTX_SIZE="${CTX_SIZE:-8192}"
N_CPU_MOE="${N_CPU_MOE:-24}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-1024}"
THREADS="${THREADS:-12}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_SERVER_BIN_DEFAULT}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "llama-server binary not found or not executable: $LLAMA_SERVER_BIN"
  echo "Set LLAMA_SERVER_BIN to the correct path, for example:"
  echo "  LLAMA_SERVER_BIN=$SCRIPT_DIR/llama.cpp/build/bin/llama-server ./phase3_gpt_oss_server.sh"
  exit 1
fi

exec "$LLAMA_SERVER_BIN" \
  -hf "$MODEL_REPO" \
  --ctx-size "$CTX_SIZE" \
  --n-cpu-moe "$N_CPU_MOE" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  -t "$THREADS" \
  --host "$HOST" \
  --port "$PORT" \
  --jinja \
  -fa auto \
  --reasoning-format auto \
  --temp 1.0 \
  --top-p 1.0 \
  $EXTRA_ARGS \
  "$@"
