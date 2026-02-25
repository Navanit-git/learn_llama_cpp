#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$WORKSPACE_DIR/llama.cpp"
PY_BIN="/home/nav_wsl/code/gen_env/bin/python"

if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "ERROR: llama.cpp directory not found at: $LLAMA_DIR"
  exit 1
fi

if [[ ! -x "$PY_BIN" ]]; then
  echo "ERROR: Python executable not found at: $PY_BIN"
  exit 1
fi

cd "$LLAMA_DIR"

echo "=== Phase 2 Check: Python Environment ==="
"$PY_BIN" --version
"$PY_BIN" -m pip --version

echo
echo "=== Phase 2 Check: Core Conversion Dependencies ==="
"$PY_BIN" -m pip show numpy sentencepiece transformers gguf protobuf torch >/dev/null
echo "OK: Required packages are installed"

echo
echo "=== Phase 2 Check: Import Smoke Test ==="
"$PY_BIN" -c "import numpy, sentencepiece, transformers, gguf, google.protobuf, torch; print('OK: imports')"

echo
echo "=== Phase 2 Check: Conversion Script Smoke Tests ==="
"$PY_BIN" convert_hf_to_gguf.py --help >/dev/null
echo "OK: convert_hf_to_gguf.py --help"
"$PY_BIN" convert_hf_to_gguf_update.py --help >/dev/null
echo "OK: convert_hf_to_gguf_update.py --help"
"$PY_BIN" convert_llama_ggml_to_gguf.py --help >/dev/null
echo "OK: convert_llama_ggml_to_gguf.py --help"
"$PY_BIN" convert_lora_to_gguf.py --help >/dev/null
echo "OK: convert_lora_to_gguf.py --help"

echo
echo "PASS: Phase 2 environment and conversion dependencies look good."