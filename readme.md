# Llama.cpp Learning Roadmap

## Overview

A hands-on, machine-based learning path for running and building on top of `llama.cpp` on local hardware. The focus is practical: understand the inference stack, experiment with model variants, deploy like a real service, and build a working chatbot with RAG.

**Machine:** Intel i7-9750H · 15 GB RAM · GTX 1650 Ti 4 GB VRAM · Ubuntu 22.04 WSL2

---

## Phase Status

| Phase | Description | Status | Detail |
|---|---|---|---|
| 1 | Core Environment Setup | ✅ Done | [phase_1_detailed_logs.md](phase_1_detailed_logs.md) |
| 2 | Python Environment & Dependencies | ✅ Done | [phase_2_detailed_logs.md](phase_2_detailed_logs.md) |
| 3 | GPT-OSS First Inference | ✅ Done | [phase_3_gpt_oss_runbook.md](phase_3_gpt_oss_runbook.md) |
| 4 | GGUF Quantization Variant Exploration | 🔄 In Progress | [phase_4_detailed_logs.md](phase_4_detailed_logs.md) |
| 5 | Native Inference Deep Dive | 🔲 Planned | — |
| 6 | DevOps — `llama-server` as a System Service | 🔲 Planned | — |
| 7 | API Integration — FastAPI Chatbot | 🔲 Planned | — |
| 8 | Domain-Specific Chatbot + RAG | 🔲 Planned | — |
| 9 | Python Bindings (`llama-cpp-python`) | 🔲 Planned | — |
| 10 | Web UI | 🔲 Optional | — |
| Extra | Raw HF → GGUF Conversion | ⏸ Deferred | — |

---

## Phase Descriptions

### Phase 1 — Core Environment Setup ✅
Clone `llama.cpp` and compile with CUDA support for the GTX 1650 Ti (compute capability 7.5). Requires GCC 10 as the CUDA host compiler due to CUDA 11.5 incompatibility with GCC 11.

### Phase 2 — Python Environment ✅
Set up a `venv` and install `llama.cpp/requirements/`. Enables Python tooling and conversion scripts used in later phases.

### Phase 3 — First Inference ✅
Pull `ggml-org/gpt-oss-20b-GGUF` directly from Hugging Face using the `-hf` flag. Tune flags for this machine's 4 GB VRAM + 15 GB RAM constraint. Achieved ~8 t/s prefill, ~11 t/s generation.

### Phase 4 — GGUF Quantization Variants
Hugging Face repos often ship one model in multiple quantization tiers (`Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`, etc.) as separate `.gguf` files. This phase benchmarks each variant on this hardware to find the practical quality-vs-speed sweet spot. See [quantization_reference.md](quantization_reference.md) for the full guide, model candidates, and benchmark commands.

### Phase 5 — Native Inference Deep Dive
Master `llama-cli` (interactive mode, sampling parameters, grammar-constrained output) and the full `llama-server` feature set (REST endpoints, health/metrics, slot system). Stress-test context limits. A runbook will be created when this phase starts.

### Phase 6 — DevOps: Service Management
Move from running the server manually to a proper `systemd` service with auto-restart, startup/teardown scripts, and log rotation. Treat the inference engine as a managed service the way a real deployment would. A runbook will be created when this phase starts.

### Phase 7 — FastAPI Chatbot Layer
Wrap `llama-server`'s OpenAI-compatible endpoint in a Python `FastAPI` service that adds per-session conversation memory and system persona management. A runbook will be created when this phase starts.

### Phase 8 — Domain-Specific Chatbot + RAG
Specialise the model for a specific domain using system prompt engineering and Retrieval-Augmented Generation — embed local documents into ChromaDB, retrieve relevant chunks at query time, inject as context. No fine-tuning required. A runbook will be created when this phase starts.

### Phase 9 — Python Bindings
Install `llama-cpp-python` with CUDA and drive the model directly from Python (no HTTP layer). Compare overhead vs the binary server. Experiment with LangChain. A runbook will be created when this phase starts.

### Phase 10 — Web UI (Optional)
Add a chat UI for daily use. Best options: Open WebUI (zero code, points at the local OpenAI endpoint) or Gradio `ChatInterface`.

### Extra — Raw HF → GGUF Conversion (Deferred)
Download raw `safetensors` weights, convert with `convert_hf_to_gguf.py`, quantize with `llama-quantize`. A learning exercise about GGUF internals, deferred until disk space is not a concern (needs >100 GB free for a 7B+ model pipeline).

---

## Supporting Documents

| File | Contents |
|---|---|
| [machine_capabilities.md](machine_capabilities.md) | Hardware specs, VRAM/RAM breakdown, observed performance |
| [lessons_learned.md](lessons_learned.md) | Build flag findings, runtime tuning, and performance tables from Phases 1–3 |
| [reference_guides.md](reference_guides.md) | Article summaries and external resource notes |
| [quantization_reference.md](quantization_reference.md) | GGUF quant naming, model candidates, and benchmark commands for Phase 4 |
| [phase_4_detailed_logs.md](phase_4_detailed_logs.md) | Phase 4 execution log |
| [phase_1_detailed_logs.md](phase_1_detailed_logs.md) | Phase 1 execution log |
| [phase_2_detailed_logs.md](phase_2_detailed_logs.md) | Phase 2 execution log |
| [phase_3_gpt_oss_runbook.md](phase_3_gpt_oss_runbook.md) | Phase 3 execution log |

