# Phase 3 Runbook: GPT-OSS on this machine

Date completed: 2026-02-25

This runbook adapts Phase 3 for `gpt-oss` on Linux with 4GB VRAM and ~15GB RAM.

## Status: ✅ Completed

**Achieved performance (v4 build, optimal flags):**
- Prefill: 7.96 t/s
- Generation: 11.33 t/s
- Cached prefill (subsequent turns): 18.52 t/s
- KV cache VRAM: 261 MiB
- VRAM free: 1,078 MiB

---

## Why this differs from the original plan

For `gpt-oss`, `llama.cpp` can load a ready GGUF directly from Hugging Face using `-hf`.
You do not need to convert raw model weights or quantize before first inference.

---

## Current Optimal Command (v4)

Run from inside `llama.cpp/`:

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

Then open: `http://127.0.0.1:8080`

---

## Flag Rationale

| Flag | Value | Why |
|---|---|---|
| `-t 10 -tb 10` | 10 threads | All MoE experts run on CPU — more threads = faster prefill. Leave 2 for OS/HTTP. |
| `--n-cpu-moe 32` | 32 | All 32 MoE experts overflow to host RAM (4GB VRAM is insufficient to hold them). |
| `-np 1` | 1 slot | Single-user setup. Eliminates 3× wasted KV cache from auto-selected np=4. |
| `-ngl 99` | all layers | Forces all 25 layers to GPU. Cleaner than hard-coding layer count. |
| `-ctk q8_0 -ctv q8_0` | 8-bit KV | Halves KV VRAM (492 → 261 MiB). Freed VRAM improved generation speed +61%. |
| `-fa auto` | Flash Attn | Auto-enables Flash Attention. Confirmed enabled in logs. |
| `-b 512 -ub 512` | batch 512 | Larger batch improves CPU AVX2 throughput for MoE matmuls. |

---

## VRAM Budget (with optimal flags)

| Component | MiB |
|---|---|
| Model weights (GPU) | 1,242 |
| KV cache (q8_0) | 261 |
| Compute buffer | 398 |
| **Total used** | **1,901** |
| **Free** | **1,078** |
| GPU total | 4,095 |

---

## Performance History

| Run | Prefill | Generation | Cached Prefill | Notes |
|---|---|---|---|---|
| v1 | 1.21 t/s | 5.86 t/s | — | Wrong defaults (6t, np4, b256) |
| v2 | 7.96 t/s | 7.05 t/s | 14.39 t/s | Fixed threads + batch |
| v3 | 2.21 t/s | 6.61 t/s | 13.16 t/s | FORCE_MMQ regression |
| **v4** | **7.96 t/s** | **11.33 t/s** | **18.52 t/s** | **ARCHS=75 + q8_0 KV — current best** |

---

## OOM / Slowdown Overrides

If you hit memory pressure, reduce in this order:

```bash
# Step 1 — reduce context
--ctx-size 8192

# Step 2 — ensure all MoE on CPU
--n-cpu-moe 32

# Step 3 — reduce batch
-b 256 -ub 256
```

---

## Phase 3 Completion Checks

- [x] Server starts without OOM
- [x] `GET /v1/models` responds
- [x] Basic chat completion works from Web UI and API
- [x] Performance benchmarked across 4 runs
- [x] Optimal build and runtime flags identified and documented
