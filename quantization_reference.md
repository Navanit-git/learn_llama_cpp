# Quantization Reference

Everything needed for Phase 4: understanding GGUF quantization variants, choosing a model, and running benchmarks on this machine.

---

## How GGUF Quantization Naming Works

Quantization reduces model weight precision from 16-bit floats down to integers. The GGUF filename encodes the quantization tier used.

### Bit-width suffixes

| Name | Bits/weight | RAM (7B) | RAM (8B) | Quality vs F16 | Recommended for |
|---|---|---|---|---|---|
| `F16` | 16 | ~14 GB | ~16 GB | Reference | Not useful on this machine |
| `Q8_0` | 8 | ~7.0 GB | ~8.0 GB | Negligible loss | Best quality that fits in RAM |
| `Q6_K` | 6 | ~5.5 GB | ~6.2 GB | Minimal | Great balance |
| `Q5_K_M` | 5 | ~4.6 GB | ~5.2 GB | Very low | Good default; fits fully in VRAM |
| `Q5_K_S` | 5 | ~4.4 GB | ~5.0 GB | Very low | Slightly smaller than `_M` |
| `Q4_K_M` | 4 | ~4.0 GB | ~4.6 GB | Low | Most popular; 8B fits in 4 GB VRAM |
| `Q4_K_S` | 4 | ~3.8 GB | ~4.3 GB | Low | Slightly smaller than `_M` |
| `Q3_K_M` | 3 | ~3.3 GB | ~3.7 GB | Moderate | Space-constrained only |
| `Q2_K` | 2 | ~2.7 GB | ~3.0 GB | Poor | Avoid unless critically space-constrained |
| `IQ4_XS` | ~4 | ~3.8 GB | ~4.3 GB | Low | Importance-quantized; same size as Q4_K_S but better quality |
| `IQ4_NL` | ~4 | ~4.0 GB | ~4.5 GB | Low | Non-linear importance quantization |
| `IQ3_XXS` | ~3 | ~3.0 GB | ~3.4 GB | Moderate | Very compact; better than Q3_K at same size |
| `IQ2_XXS` | ~2 | ~2.5 GB | ~2.8 GB | Poor | Edge-case only |

### Suffix guide: `_K`, `_M`, `_S`, `IQ`

| Suffix | Meaning |
|---|---|
| `_K` | K-quant — uses a mixed-precision block scheme; better quality than uniform quantization |
| `_M` / `_S` | Medium / Small variant within a tier — `_M` has slightly higher quality and larger size |
| `IQ` prefix | Importance-quantized — assigns higher precision to "important" weights; same size as equivalent K-quant but better quality |

**Rule of thumb for this machine (15 GB RAM, 4 GB VRAM):**
- 7B/8B model fully in VRAM: `Q4_K_M` (just fits) or `Q5_K_M` (better quality, slight overflow to RAM)
- 7B/8B model RAM-assisted: `Q6_K` or `Q8_0` — attention on GPU, rest in RAM
- 3B model: `Q8_0` — fits comfortably

---

## Reading a Multi-GGUF Repo on Hugging Face

Many repos (especially `bartowski/*`) host all quant tiers for a single model in one place. Example: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

```
Meta-Llama-3.1-8B-Instruct-Q2_K.gguf          ← 2-bit, low quality
Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf        ← 3-bit, moderate
Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf        ← 4-bit medium, most popular
Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf        ← 5-bit medium, recommended
Meta-Llama-3.1-8B-Instruct-Q6_K.gguf          ← 6-bit, high quality
Meta-Llama-3.1-8B-Instruct-Q8_0.gguf          ← 8-bit, near-lossless
Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf        ← ~4-bit importance-quantized
```

The `-hf` / `--hf-file` flags let you pick exactly one file from the repo without downloading the others:

```bash
./build/bin/llama-cli \
  --hf-repo bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --hf-file Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -t 10 -ngl 99 -p "What is quantization?"
```

Models download to `~/.cache/llama.cpp/` by default.

---

## Recommended Model Candidates for Phase 4

These models have multiple GGUF variants in a single repo and are well-suited to this machine.

| Repo | Model size | # Quant variants | Notes |
|---|---|---|---|
| `bartowski/Llama-3.2-3B-Instruct-GGUF` | 3B | 8+ | Very fast; best for learning quant trade-offs; fits entirely in 4 GB VRAM even at Q8_0 |
| `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | 8B | 10+ | Excellent instruct model; Q4_K_M just fits in 4 GB VRAM |
| `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | 7B | 10+ | Strong coding + reasoning; good baseline for chatbot |
| `ggml-org/Qwen2.5-7B-Instruct-GGUF` | 7B | 6+ | Official GGUF release; strong multilingual and coding; good for DevOps chatbot |
| `ggml-org/Qwen2.5-Coder-7B-Instruct-GGUF` | 7B | 6+ | Coding specialist; best choice if Phase 7/8 is a DevOps/coding chatbot |

**Suggested start:** `bartowski/Llama-3.2-3B-Instruct-GGUF` — small enough that even `Q8_0` fits in VRAM, so you can benchmark all tiers cleanly without RAM spill complicating results.

---

## Benchmark Commands

### Speed benchmark with `llama-bench`

```bash
./build/bin/llama-bench \
  -m /home/nav_wsl/.cache/llama.cpp/<model-variant>.gguf \
  -p 512 \      # prompt tokens (prefill measure)
  -n 128 \      # generation tokens
  -t 10 \       # CPU threads
  --no-mmap 0 \ # allow mmap (normal operation)
  -ngl 99       # all layers to GPU
```

Outputs a table with `pp` (prompt processing = prefill) and `tg` (token generation) in t/s.

### Monitor VRAM during inference

```bash
# Continuous 1-second polling
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
  --format=csv -l 1
```

### Monitor RAM during inference

```bash
# Check before, during, and after loading
watch -n 1 'free -h'

# OR for more detail:
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable|Cached"
```

### Full benchmark loop script

Run this to benchmark all variants in a repo automatically:

```bash
#!/bin/bash
MODEL_DIR="$HOME/.cache/llama.cpp"
BINARY="./build/bin/llama-bench"

for gguf in "$MODEL_DIR"/*Llama-3.2-3B*.gguf; do
  echo "=== Benchmarking: $(basename $gguf) ==="
  $BINARY -m "$gguf" -p 512 -n 128 -t 10 -ngl 99
  echo ""
done
```

---

## Benchmark Results Template

Fill this in as each variant is tested.

**Model family:** ___________  
**Date:** ___________

| Variant | RAM used | VRAM used | Prefill (t/s) | Generation (t/s) | Load time | Notes |
|---|---|---|---|---|---|---|
| Q2_K | | | | | | |
| Q3_K_M | | | | | | |
| Q4_K_M | | | | | | |
| IQ4_XS | | | | | | |
| Q5_K_M | | | | | | |
| Q6_K | | | | | | |
| Q8_0 | | | | | | |

---

## Expected Results on This Hardware

Based on Phase 3 experience with the gpt-oss-20b model and general knowledge of the GTX 1650 Ti + i7-9750H combo:

- **3B Q8_0:** ~20–30 t/s generation (fits entirely in VRAM)
- **3B Q4_K_M:** ~30–40 t/s generation (tiny model, very fast)
- **7B/8B Q4_K_M:** ~10–15 t/s generation (fits in 4 GB VRAM)
- **7B/8B Q8_0:** ~5–8 t/s generation (overflows to RAM; CPU bottleneck)
- **7B/8B Q5_K_M:** ~8–12 t/s generation (slight spill, good balance)

These are estimates. Actual results will vary by model architecture (dense vs MoE, attention type, etc.).

---

## Why This Will Inform Later Phases

The choice of model + quant tier from Phase 4 becomes the **deployment model** for Phases 5–9. Choosing well here means:
- Phase 6 (systemd service): the model path and flags go into the service unit
- Phase 7 (FastAPI chatbot): the model needs to give good-quality responses at the chosen quant level
- Phase 8 (RAG): context window size affects how much retrieved text can be injected; higher quant = more context budget for RAG chunks
