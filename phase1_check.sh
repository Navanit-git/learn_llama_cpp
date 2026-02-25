#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$WORKSPACE_DIR/llama.cpp"

if [[ ! -d "$LLAMA_DIR" ]]; then
  echo "ERROR: llama.cpp directory not found at: $LLAMA_DIR"
  exit 1
fi

cd "$LLAMA_DIR"

echo "=== Phase 1 Check: Toolchain ==="
cmake --version | head -n 1
nvcc --version | sed -n '1,4p'
g++-10 --version | head -n 1

echo
echo "=== Phase 1 Check: RAM ==="
mem_total_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
mem_avail_kb="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
swap_total_kb="$(awk '/^SwapTotal:/ { print $2 }' /proc/meminfo)"
swap_free_kb="$(awk '/^SwapFree:/ { print $2 }' /proc/meminfo)"

mem_total_gb="$(awk -v kb="$mem_total_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')"
mem_avail_gb="$(awk -v kb="$mem_avail_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')"
swap_total_gb="$(awk -v kb="$swap_total_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')"
swap_free_gb="$(awk -v kb="$swap_free_kb" 'BEGIN { printf "%.2f", kb/1024/1024 }')"

echo "MemTotal: ${mem_total_gb} GiB"
echo "MemAvailable: ${mem_avail_gb} GiB"
echo "SwapTotal: ${swap_total_gb} GiB"
echo "SwapFree: ${swap_free_gb} GiB"

if (( mem_total_kb < 12000000 )); then
  echo "WARN: Total RAM is below ~11.4 GiB; larger quantized models may be constrained."
fi

if (( mem_avail_kb < 6000000 )); then
  echo "WARN: Available RAM is below ~5.7 GiB; close other memory-heavy apps before inference/builds."
fi

echo
echo "=== Phase 1 Check: CMake Cache ==="
cmake -N -L build | grep -E 'GGML_CUDA|CMAKE_CUDA_HOST_COMPILER' || true

if [[ -f build/CMakeCache.txt ]]; then
  host_compiler_line="$(grep '^CMAKE_CUDA_HOST_COMPILER:FILEPATH=' build/CMakeCache.txt || true)"
  if [[ -n "$host_compiler_line" ]]; then
    echo "$host_compiler_line"
  else
    compiler_file="build/CMakeFiles/3.22.1/CMakeCUDACompiler.cmake"
    if [[ -f "$compiler_file" ]]; then
      host_compiler_cmake="$(grep '^set(CMAKE_CUDA_HOST_COMPILER ' "$compiler_file" || true)"
      if [[ -n "$host_compiler_cmake" ]]; then
        echo "$host_compiler_cmake"
      else
        echo "WARN: CMAKE_CUDA_HOST_COMPILER not found in CMake cache or compiler metadata"
      fi
    else
      echo "WARN: CMAKE_CUDA_HOST_COMPILER not set in build/CMakeCache.txt"
    fi
  fi
else
  echo "ERROR: build/CMakeCache.txt not found"
  exit 1
fi

echo
echo "=== Phase 1 Check: Required Binaries ==="
for bin in llama-cli llama-server llama-quantize; do
  if [[ -x "build/bin/$bin" ]]; then
    echo "OK: build/bin/$bin"
  else
    echo "MISSING: build/bin/$bin"
    exit 1
  fi
done

echo
echo "=== Phase 1 Check: Smoke Tests ==="
./build/bin/llama-cli --help >/dev/null
./build/bin/llama-server --help >/dev/null

echo "PASS: Phase 1 environment and artifacts look good."
