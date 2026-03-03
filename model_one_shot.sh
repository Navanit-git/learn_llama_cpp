#!/usr/bin/env bash
set -euo pipefail

# One script for any GGUF model: download, metadata check, and optional smoke benchmark.
#
# Quick start:
#   1) Edit MODEL_REPO and MODEL_FILE below.
#   2) Run: ./model_one_shot.sh
#
# Actions:
#   ACTION=download ./model_one_shot.sh
#   ACTION=check    ./model_one_shot.sh
#   ACTION=smoke    ./model_one_shot.sh
#   ACTION=all      ./model_one_shot.sh   # default

ROOT_DIR="${ROOT_DIR:-/home/nav_wsl/code/learn_llama_cpp}"

# Change these 2 lines for any model:
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-4B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-Q8_0.gguf}"

ACTION="${ACTION:-all}" # download | check | smoke | all

CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llama.cpp}"
CACHE_PATH="$CACHE_DIR/${MODEL_REPO//\//_}_${MODEL_FILE}"

GGUF_BIN="${GGUF_BIN:-$ROOT_DIR/llama.cpp/build/bin/llama-gguf}"

BASE_NAME="${MODEL_FILE%.gguf}"
META_OUT="${META_OUT:-$ROOT_DIR/${BASE_NAME}_gguf_meta.txt}"
FILTER_OUT="${FILTER_OUT:-$ROOT_DIR/${BASE_NAME}_gguf_meta_filtered.txt}"

# Smoke benchmark settings (uses generic regression runner)
RUNNER="${RUNNER:-$ROOT_DIR/model_regression_benchmark.sh}"
MODE="${MODE:-smoke}"
MAX_CASES="${MAX_CASES:-20}"
RESUME="${RESUME:-1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/archieve/${BASE_NAME}_regression_smoke_$(date +%Y%m%d_%H%M%S)}"

download_model() {
  mkdir -p "$CACHE_DIR"
  if [[ -f "$CACHE_PATH" ]]; then
    echo "Using cached model: $CACHE_PATH"
    return 0
  fi

  echo "Downloading model"
  echo "- repo: $MODEL_REPO"
  echo "- file: $MODEL_FILE"
  echo "- output: $CACHE_PATH"

  curl -L --fail --retry 3 --retry-delay 2 -C - \
    -o "$CACHE_PATH" \
    "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
}

check_model() {
  if [[ ! -x "$GGUF_BIN" ]]; then
    echo "ERROR: missing executable: $GGUF_BIN"
    echo "Build llama.cpp first, or set GGUF_BIN=/path/to/llama-gguf"
    exit 1
  fi

  echo "Dumping GGUF metadata to: $META_OUT"
  "$GGUF_BIN" "$CACHE_PATH" r | tee "$META_OUT"

  echo
  echo "Writing filtered metadata to: $FILTER_OUT"
  grep -Ei 'layer|block|expert|moe|ffn|head|rope|context|vocab|embedding' "$META_OUT" \
    | tee "$FILTER_OUT" \
    | sed -n '1,200p'
}

run_smoke() {
  if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: benchmark runner not executable: $RUNNER"
    echo "Set RUNNER=/path/to/*_regression_benchmark.sh"
    exit 1
  fi

  echo "Running smoke benchmark"
  echo "- runner: $RUNNER"
  echo "- mode: $MODE"
  echo "- max cases: $MAX_CASES"
  echo "- out dir: $OUT_DIR"

  MODEL_PATH="$CACHE_PATH" \
  MODE="$MODE" \
  MAX_CASES="$MAX_CASES" \
  RESUME="$RESUME" \
  OUT_DIR="$OUT_DIR" \
  "$RUNNER"
}

case "$ACTION" in
  download)
    download_model
    ;;
  check)
    download_model
    check_model
    ;;
  smoke)
    download_model
    run_smoke
    ;;
  all)
    download_model
    check_model
    run_smoke
    ;;
  *)
    echo "ERROR: unknown ACTION=$ACTION"
    echo "Use: download | check | smoke | all"
    exit 1
    ;;
esac

echo
echo "Done."
echo "- Model: $CACHE_PATH"
if [[ "$ACTION" == "check" || "$ACTION" == "all" ]]; then
  echo "- Metadata: $META_OUT"
  echo "- Filtered: $FILTER_OUT"
fi
if [[ "$ACTION" == "smoke" || "$ACTION" == "all" ]]; then
  echo "- Benchmark out: $OUT_DIR"
fi
