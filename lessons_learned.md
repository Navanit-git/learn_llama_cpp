# Lessons Learned

Findings and hard-won knowledge from completed phases. This is a living document — add a new section when each phase completes.

---

## Phase 1 — Build Flags (CUDA Compilation)

### CUDA 11.5 requires GCC 10 as host compiler

GCC 11.4 (Ubuntu 22.04 default) causes parameter pack errors when compiling CUDA kernels with CUDA 11.5. The fix is to pass the older compiler explicitly:

```cmake
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10
```

Install if missing: `sudo apt install g++-10`

### Target only the machine's GPU architecture

The default build targets all CUDA architectures (6 variants), wasting compile time and producing a larger binary. Targeting only cc7.5 (GTX 1650 Ti):

```cmake
-DCMAKE_CUDA_ARCHITECTURES="75"
```

Effect: ~30% faster compile, smaller binary, better kernel selection at runtime since the dispatcher has no dead paths.

### `GGML_CUDA_FORCE_MMQ=ON` is harmful for this model

Enabling `FORCE_MMQ` replaces cuBLAS GEMM kernels with integer MMQ (matrix multiply quantized) kernels. MMQ is tuned for single-token generation — it is inefficient for large batch prefill. With this flag, prefill on gpt-oss-20b dropped from 7.96 → 2.21 t/s (+259% regression).

**Rule of thumb:** Only enable `FORCE_MMQ` if VRAM is critically constrained and prefill speed is not important.

### Final build command

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

---

## Phase 3 — Runtime Flags (gpt-oss-20b MXFP4 MoE)

### Threads are the single biggest lever on this machine

All 32 MoE expert layers overflow to CPU RAM because they don't fit in 4 GB VRAM. This means CPU throughput is critical for the expert computation.

| Threads (`-t`) | Prefill | Improvement |
|---|---|---|
| 6 (default) | 1.21 t/s | baseline |
| 10 | 7.96 t/s | +558% |

**Command:** `-t 10 -tb 10` (inference threads + batch threads)

### Parallel slots waste KV cache VRAM

Each parallel slot (`-np N`) pre-allocates a full KV cache block. With a 20K context window at f16, each slot costs ~492 MiB. For a personal single-user chatbot, `-np 1` is always correct.

| `-np` | KV VRAM |
|---|---|
| 4 (default) | 4 × 492 = 1968 MiB |
| 1 | 492 MiB |

### Quantized KV cache halves VRAM with negligible quality loss

`-ctk q8_0 -ctv q8_0` quantizes the KV cache tensors to 8-bit. This is not the attention weights — it's the cached key/value activations. Impact on output quality is negligible at 8-bit.

| KV dtype | KV VRAM | Quality |
|---|---|---|
| f16 (default) | 492 MiB | Reference |
| q8_0 | 261 MiB | Negligible loss |

The freed 231 MiB VRAM headroom allowed the GPU to better schedule compute buffers, improving generation speed from 7.05 → 11.33 t/s.

### Use `-ngl 99` instead of a hard-coded layer count

`-ngl 99` (or any number larger than the model's layer count) offloads all layers to GPU. This is cleaner than counting layers manually and still works correctly even if the model doesn't have 99 layers — llama.cpp clamps to the actual count.

### `--n-cpu-moe` for MoE models

For Mixture-of-Experts models, this flag forces expert layers to CPU regardless of `-ngl`. With 4 GB VRAM, all 32 experts of gpt-oss-20b overflow to host RAM anyway, but setting `--n-cpu-moe 32` makes this explicit and avoids VRAM OOM errors on context fill.

### Flash Attention is always safe to enable

`-fa auto` enables Flash Attention when the backend supports it (it does on CUDA). Flash Attention reduces memory bandwidth for the attention computation. Enabling it unconditionally is safe — it falls back gracefully when unsupported.

### Batch size `-b 512` maximises CPU throughput

Larger batches allow AVX2 SIMD to process more tokens in parallel during prefill. `-b 512 -ub 512` (logical batch + physical micro-batch) is the sweet spot for this CPU. Larger values don't help further due to memory bandwidth limits.

---

## Performance Comparison Table — gpt-oss-20b MXFP4 MoE

| Run | Build Flags | Runtime Flags | Prefill (t/s) | Generation (t/s) | Cached Prefill (t/s) | KV VRAM |
|---|---|---|---|---|---|---|
| v1 | default (6 archs) | 6t · np4 · b256 | 1.21 | 5.86 | — | 492 MiB |
| v2 | default (6 archs) | 10t · np1 · b512 | 7.96 | 7.05 | 14.39 | 492 MiB |
| v3 | ARCHS=75 + FORCE_MMQ | 10t · np1 · b512 · q8_0 KV | 2.21 | 6.61 | 13.16 | 261 MiB |
| **v4 ✅** | **ARCHS=75** | **10t · np1 · b512 · q8_0 KV** | **7.96** | **11.33** | **18.52** | **261 MiB** |

**Key takeaways from the comparison:**
- v2 vs v1: threads alone gave 6.6× prefill speedup
- v3 vs v2: FORCE_MMQ destroyed prefill (−72%) while saving 231 MiB VRAM
- v4 vs v2: reverting FORCE_MMQ + adding q8_0 KV gave free VRAM savings AND better generation speed (+61%)

---

## Current Optimal Server Command (gpt-oss-20b)

```bash
./build/bin/llama-server \
  -m /home/nav_wsl/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf \
  --ctx-size 20000 --jinja \
  -b 512 -ub 512 \
  -t 10 -tb 10 \
  --n-cpu-moe 32 -np 1 \
  -fa auto -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --host 0.0.0.0 --port 8080
```

**Memory footprint at steady state:**
- Model weights on GPU: 1,242 MiB
- KV cache (q8_0): 261 MiB
- Compute buffer: 398 MiB
- GPU free: ~1,078 MiB
- CPU RAM (MoE experts): ~10.95 GiB

---

## Flag Quick-Reference Card

| Flag | Value | Effect |
|---|---|---|
| `-t` | 10 | CPU inference threads (all cores minus 2 for OS) |
| `-tb` | 10 | Batch processing threads |
| `-np` | 1 | 1 parallel KV slot (single-user) |
| `-b` | 512 | Logical batch size |
| `-ub` | 512 | Physical micro-batch size |
| `-ngl` | 99 | Offload all layers to GPU |
| `--n-cpu-moe` | 32 | Force all MoE experts to CPU |
| `-fa` | auto | Enable Flash Attention |
| `-ctk` | q8_0 | 8-bit quantized KV cache keys |
| `-ctv` | q8_0 | 8-bit quantized KV cache values |
| `--jinja` | — | Use Jinja2 chat template (required for instruct models) |
| `--ctx-size` | 20000 | Context window in tokens |
