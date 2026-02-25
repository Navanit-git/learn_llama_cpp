# llama.cpp Requirements and Machine Capabilities

## 1. `llama.cpp` General Requirements
*   **Hardware:**
    *   **CPU:** Modern CPU with AVX2 or AVX-512 instructions for decent CPU inference.
    *   **RAM:** Depends entirely on the model size. A 7B/8B parameter model quantized to 4-bit needs ~4.5GB to 5GB of RAM.
    *   **GPU (Optional):** VRAM dictates how much of the model can be offloaded for faster generation.
*   **Software:**
    *   C/C++ compiler (`gcc` or `clang`).
    *   Build tools (`make` or `cmake`).
    *   For NVIDIA GPU acceleration: CUDA Toolkit (`nvcc`).

---

## 2. Your Machine's Capabilities

### CPU
*   **Model:** Intel Core i7-9750H @ 2.60 GHz
*   **Cores / Threads:** 6 cores / 12 threads (all allocated to WSL2)
*   **L3 Cache:** 12 MB
*   **Instruction sets:** SSE3, SSSE3, AVX, AVX2, F16C, FMA, BMI2
*   **WSL2 thread usage:** 10 threads assigned to llama.cpp (`-t 10 -tb 10`), leaving 2 for OS/HTTP
*   **Role in inference:** Handles all MoE expert layers (`ffn_*_exps`) since they don't fit in 4GB VRAM. AVX2 is critical here.
*   **Verdict:** Solid. AVX2 + 10 threads achieved ~8 t/s prefill on a 20B MoE model.

### RAM
*   **Total (WSL2):** 15.47 GiB
*   **Available (typical):** ~11 GiB
*   **Swap:** 4.0 GiB (50 MiB used)
*   **Configured via:** `.wslconfig` (max RAM allocation)
*   **Current model RAM usage:** ~10.95 GiB (CPU_Mapped MoE expert weights for gpt-oss-20b)
*   **Verdict:** Sufficient for 20B MXFP4 MoE models. Leaves ~4 GiB headroom.

### GPU
*   **Model:** NVIDIA GeForce GTX 1650 Ti
*   **VRAM:** 4096 MiB total / ~3935 MiB free (idle)
*   **Architecture:** Turing (compute capability 7.5)
*   **Tensor cores:** None (Turing without tensor cores — affects CUDA kernel selection)
*   **Driver version:** 571.96
*   **CUDA graphs:** Disabled at runtime due to MoE dynamic branching
*   **Current VRAM usage (gpt-oss-20b):**
    *   Model weights on GPU: 1,242 MiB
    *   KV cache (q8_0): 261 MiB  ← halved vs f16 (was 492 MiB)
    *   Compute buffer: 398 MiB
    *   Free: ~1,078 MiB
*   **Role in inference:** Handles all attention layers (25/25 layers offloaded). Flash Attention enabled.
*   **Verdict:** Handles attention well at ~6.5–7 t/s generation. MoE experts overflow to CPU RAM.

### Disk
*   **Total:** 1007 GiB
*   **Used:** 44 GiB
*   **Available:** 913 GiB
*   **Verdict:** Ample space for large model files.

### OS / Environment
*   **OS:** Ubuntu 22.04 (WSL2)
*   **Kernel:** Linux 6.6.87.2-microsoft-standard-WSL2
*   **WSL2 host:** Windows with GPU passthrough via CUDA for WSL

---

## 3. Software Dependencies

| Tool | Version | Notes |
|---|---|---|
| `cmake` | 3.22.1 | Build system for llama.cpp |
| `nvcc` (CUDA Toolkit) | 11.5 (V11.5.119) | GPU kernel compiler |
| `gcc` | 11.4.0 | System default C/C++ compiler |
| `gcc-10` / `g++-10` | 10.5.0 | Required as CUDA host compiler (CUDA 11.5 incompatible with GCC 11) |

### Key Compatibility Note
CUDA 11.5 is incompatible with GCC 11.4 as a host compiler. Always use:
```
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
```

---

## 4. Observed Performance (gpt-oss-20b MXFP4 MoE)

| Version | Build flags | Prefill | Generation | Notes |
|---|---|---|---|---|
| v1 | default (6t, np4) | 1.21 t/s | 5.86 t/s | Baseline — wrong defaults |
| v2 | 10t, np1, b512 | **7.96 t/s** | **7.05 t/s** | Best prefill achieved |
| v3 | +FORCE_MMQ, +q8_0 KV | 2.21 t/s | 6.61 t/s | MMQ hurt prefill; q8_0 KV saved 231 MiB VRAM |

**Current optimal command:**
```bash
./build/bin/llama-server \
  -m <model.gguf> \
  --ctx-size 20000 --jinja \
  -b 512 -ub 512 \
  -t 10 -tb 10 \
  --n-cpu-moe 32 -np 1 \
  -fa auto -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --host 0.0.0.0 --port 8080
```
