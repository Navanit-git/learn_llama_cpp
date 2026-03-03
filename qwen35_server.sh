#!/usr/bin/env bash
set -euo pipefail

# ── Qwen3.5-4B-Q8_0 server — optimised for GTX 1650 Ti (4 GB VRAM) ──────────
#
# Architecture: hybrid Gated-DeltaNet (Mamba) + Gated-Attention, 32 layers
#   Layout:  8 × (3 × DeltaNet-FFN  +  1 × Attention-FFN)
#   → only 8/32 layers use KV cache; 24 use recurrent state (RS)
#
# VRAM budget (3182 MiB free):
#   Q8_0  weights (all 33 layers)       = 4264 MiB  ← DOES NOT FIT → 14/33 GPU → 4 t/s
#   Q4_K_M weights (all 33 layers)      ≈ 2805 MiB  ← fits ~26/33 GPU → 7-9 t/s ✓
#   KV cache q8_0 @ 20k ctx (8 layers)  ≈  340 MiB
#   RS buffer (np=1, 1 seq)             ≈   50 MiB
#   Compute buffer                      ≈  250 MiB
#
# Strategy: let -fit auto-allocate GPU layers (no explicit -ngl) so that
#   model+KV+RS+compute truly fits in VRAM without VMM paging.
#   Use -ctk/-ctv q8_0 and -np 1 to free ~450 MiB for extra GPU layers.
#   Use -fitt 256 (WSL2, headless) so auto-fit packs the GPU tighter.
#
# Override any default via env:
#   NGL=99 CTX_SIZE=4096 ./qwen35_server.sh
#
# For a bigger speed win, recompile llama.cpp for your exact GPU:
#   cd llama.cpp/build && cmake .. \
#     -DGGML_CUDA=ON \
#     -DCMAKE_CUDA_ARCHITECTURES="75" \
#     -DGGML_CUDA_FORCE_MMQ=ON \
#     -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
#   && cmake --build . --config Release -j$(nproc)
# FORCE_MMQ uses hand-tuned kernels that are faster on Turing (no tensor cores).
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-4B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-Q4_K_M.gguf}"  # Q4_K_M fits ~26/33 layers on GTX 1650 Ti
# To revert:  MODEL_FILE=Qwen3.5-4B-Q8_0.gguf ./qwen35_server.sh

# ── tunables (override via env) ──────────────────────────────────────────────
CTX_SIZE="${CTX_SIZE:-20480}"        # 20 k context
THREADS="${THREADS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"      # prefill batch
UBATCH_SIZE="${UBATCH_SIZE:-512}"    # match batch for fewer dispatches
NP="${NP:-1}"                        # 1 slot → saves ~150 MiB RS on GPU
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}" # q8_0 KV → saves ~300 MiB vs f16
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
FIT_TARGET="${FIT_TARGET:-256}"      # MiB to leave free on GPU (WSL2 headless OK)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-1.0}"
# Set NGL to override auto-fit (e.g. NGL=99 forces all layers, NGL=15 manual)
NGL="${NGL:-}"

LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/qwenterminal_v2.log}"

LLAMA_SERVER_BIN_DEFAULT="$SCRIPT_DIR/llama.cpp/build/bin/llama-server"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_SERVER_BIN_DEFAULT}"

# ── download model to cache ──────────────────────────────────────────────────
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/llama.cpp}"
CACHE_PATH="$CACHE_DIR/${MODEL_REPO//\//_}_${MODEL_FILE}"

if [[ ! -f "$CACHE_PATH" ]]; then
  mkdir -p "$CACHE_DIR"
  echo "Downloading $MODEL_REPO/$MODEL_FILE → $CACHE_PATH"
  curl -L --fail --retry 3 --retry-delay 2 -C - \
    -o "$CACHE_PATH" \
    "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
fi

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "llama-server binary not found or not executable: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

# ── build command ────────────────────────────────────────────────────────────
CMD=(
  "$LLAMA_SERVER_BIN"
  -m "$CACHE_PATH"
  --ctx-size "$CTX_SIZE"
  -np "$NP"
  -t "$THREADS"
  -b "$BATCH_SIZE"
  -ub "$UBATCH_SIZE"
  -ctk "$CACHE_TYPE_K"
  -ctv "$CACHE_TYPE_V"
  -fitt "$FIT_TARGET"
  --jinja
  --reasoning-format auto
  -fa auto
  --temp "$TEMP"
  --top-p "$TOP_P"
  --host "$HOST"
  --port "$PORT"
)

# only pass -ngl when the user explicitly sets it
[[ -n "$NGL" ]] && CMD+=( -ngl "$NGL" )

echo "┌─────────────────────────────────────────────────"
echo "│ Model : $MODEL_REPO / $MODEL_FILE"
echo "│ Cache : $CACHE_PATH"
echo "│ Ctx   : $CTX_SIZE  Threads: $THREADS  Slots: $NP"
echo "│ KV    : K=$CACHE_TYPE_K  V=$CACHE_TYPE_V"
echo "│ Fit   : target=${FIT_TARGET} MiB free  NGL=${NGL:-auto}"
echo "│ Log   : $LOG_FILE"
echo "│ Listen: http://$HOST:$PORT"
echo "└─────────────────────────────────────────────────"
echo ""
echo "Starting server … (Ctrl-C or 'kill \$!' to stop)"
echo "Logs → $LOG_FILE"
echo ""

exec "${CMD[@]}" ${EXTRA_ARGS:+$EXTRA_ARGS} 2>&1 | tee "$LOG_FILE"
