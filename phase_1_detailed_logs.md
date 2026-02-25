# Phase 1 Detailed Logs and Methods


## Objective
Complete **Phase 1: Core Environment Setup** by:
1. Cloning `llama.cpp`
2. Building with CUDA support

## Detailed Logs

### 1) Repository cloned
- Command run:
  - `git clone https://github.com/ggml-org/llama.cpp.git`
- Result:
  - Repository cloned successfully to `./llama.cpp`
  - Checked revision: `8c2c0108d` on branch `master`

### 2) CUDA build configured with CMake
- Command run:
  - `cmake -S . -B build -DGGML_CUDA=ON`
- Result:
  - Configuration succeeded
  - CUDA backend detected and enabled
  - `nvcc` detected (`Cuda compilation tools, release 11.5, V11.5.119`)
  - Build files generated in `./llama.cpp/build`

### 3) Initial build attempted
- Command run:
  - `cmake --build build -j12`
- Result:
  - Build started and compiled CPU components successfully
  - Build failed in CUDA compilation stage

### 4) Root cause identified
- CUDA 11.5 is incompatible with GCC 11.4 as a host compiler in this setup.
- Error observed:
  - `error: parameter packs not expanded with '...'`

### 5) Compatibility fix applied
- Used GCC 10 as CUDA host compiler.
- Commands run:
  1. `sudo apt-get update && sudo apt-get install -y gcc-10 g++-10`
  2. `rm -rf build`
  3. `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10`
  4. `cmake --build build -j12`

### 6) Build result after fix
- Build completed successfully to 100%.
- CUDA backend built successfully (`libggml-cuda.so`).
- Required binaries confirmed in `./llama.cpp/build/bin`:
  - `llama-cli`
  - `llama-server`
  - `llama-quantize`

## Methods Used (Phase 1)

1. **Git-based source acquisition**
   - Pulled upstream `llama.cpp` directly from the official repository.

2. **CMake out-of-source build**
   - Used `cmake -S . -B build` to keep build artifacts isolated.

3. **CUDA backend enablement**
   - Enabled NVIDIA acceleration through `-DGGML_CUDA=ON`.

4. **Host compiler pinning for CUDA 11.5**
   - Forced `nvcc` host compiler to `g++-10` with:
     - `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10`

5. **Clean rebuild strategy**
   - Removed old build cache (`rm -rf build`) before reconfiguration to prevent stale toolchain settings.

6. **Artifact-based verification**
   - Verified success by checking generated binaries instead of relying only on configure output.

### 7) First optimized rebuild attempt — FORCE_MMQ (v3, FAILED to improve)
- **Motivation:** Runtime logs showed `ggml_cuda_graph_set_enabled: disabling CUDA graphs due to GPU architecture`. Hypothesis: `DGGML_CUDA_FORCE_MMQ` would improve GPU performance on Turing without tensor cores.
- Commands run:
  1. `rm -rf build`
  2. `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -DGGML_CUDA_FORCE_MMQ=ON -DCMAKE_CUDA_ARCHITECTURES="75"`
  3. `cmake --build build -j10`
- **Result: Regression.** Prefill dropped from 7.96 → 2.21 t/s. `FORCE_MMQ` replaces cuBLAS GEMM with integer MMQ kernels that are tuned for single-token decoding, not large-batch prefill.
- **Conclusion:** `FORCE_MMQ` is harmful for this model. Do NOT use it.

### 8) Final optimal rebuild — ARCHS=75 only, no FORCE_MMQ (v4, BEST)
- **Motivation:** Keep `ARCHS=75` (which avoids dead kernel code for other GPU generations) but drop `FORCE_MMQ`. Also add `-ctk q8_0 -ctv q8_0` KV cache quantization at runtime.
- Commands run:
  1. `rm -rf build`
  2. `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -DCMAKE_CUDA_ARCHITECTURES="75"`
  3. `cmake --build build -j10`
- **Result: Best performance achieved.**
  - Prefill: 7.96 t/s (same as v2)
  - Generation: **11.33 t/s** (+61% over v2's 7.05 t/s)
  - Cached prefill: **18.52 t/s** (+29% over v2's 14.39 t/s)
  - KV cache VRAM: 261 MiB (halved from 492 MiB by `-ctk q8_0 -ctv q8_0`)
  - VRAM free: 1078 MiB (vs 850 MiB in v2)

---

## Final Status
- **Phase 1 Step 1 (Clone):** ✅ Completed
- **Phase 1 Step 2 (Build with CUDA):** ✅ Completed
- **Phase 1 Step 3 (FORCE_MMQ rebuild, v3):** ❌ Regressed — reverted
- **Phase 1 Step 4 (ARCHS=75 optimal rebuild, v4):** ✅ Completed — current production build

## Reproducibility Note
On this machine (CUDA 11.5), the confirmed optimal CMake configure command is:
```bash
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -DCMAKE_CUDA_ARCHITECTURES="75"
```

- `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10` — required; CUDA 11.5 is incompatible with GCC 11.4
- `-DCMAKE_CUDA_ARCHITECTURES="75"` — target only GTX 1650 Ti (cc7.5); leaner binary, correct kernel dispatch
- **Do NOT use** `-DGGML_CUDA_FORCE_MMQ=ON` — proven to regress prefill speed on this model

## Phase 1 Verification Script (Personal Workspace)

To keep personal checks separate from the upstream `llama.cpp` repository, a workspace-level script is used:
- `./phase1_check.sh` (path: `/home/nav_wsl/code/learn_llama_cpp/phase1_check.sh`)

### What it validates
1. Toolchain versions (`cmake`, `nvcc`, `g++-10`)
2. RAM and swap status (`MemTotal`, `MemAvailable`, `SwapTotal`, `SwapFree`)
3. CUDA build flags in CMake cache (`GGML_CUDA=ON`)
4. CUDA host compiler metadata (`/usr/bin/g++-10`)
5. Required binaries:
  - `build/bin/llama-cli`
  - `build/bin/llama-server`
  - `build/bin/llama-quantize`
6. Smoke tests:
  - `./build/bin/llama-cli --help`
  - `./build/bin/llama-server --help`

### How to run
```bash
cd /home/nav_wsl/code/learn_llama_cpp
./phase1_check.sh
```

### Current result
- Script run status: `PASS`
- RAM observed during validation:
  - `MemTotal: 15.47 GiB`
  - `MemAvailable: ~14 GiB`
  - `SwapTotal: 4.00 GiB`



