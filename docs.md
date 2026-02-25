# Llama.cpp Project Documentation

## 1. Reference Summaries

### Article 1: Llama.cpp Guide (SteelPh0enix Blog)
Url : https://blog.steelph0enix.dev/posts/llama-cpp-guide/

*   **Core Focus:** Building `llama.cpp` from scratch, understanding its components, and running it locally on any hardware.
*   **Building:** Recommends using `cmake` and `ninja` to build the project. It emphasizes enabling specific backends (like Vulkan or CUDA) for GPU acceleration.
*   **Model Preparation:** Explains how to download raw HuggingFace models (like `safetensors`), convert them to the `GGUF` format using the included Python script (`convert_hf_to_gguf.py`), and then quantize them using the `llama-quantize` tool.
*   **Quantization:** Discusses the trade-offs of different quantization levels (e.g., `Q8_0`, `Q4_K_M`). Lower bits mean less RAM/VRAM usage but slightly lower accuracy (higher perplexity).
*   **Execution:** Details how to use `llama-server` (which provides an OpenAI-compatible API and a Web UI) and `llama-cli` (for terminal-based chat). It also explains various generation parameters like Temperature, Top-K, Top-P, and Context Size.

### Article 2: The Ultimate Guide to Efficient LLM Inference (PyImageSearch)
URL: https://pyimagesearch.com/2024/08/26/llama-cpp-the-ultimate-guide-to-efficient-llm-inference-and-applications/
*   **Core Focus:** The broader ecosystem, the GGUF format, and Python bindings (`llama-cpp-python`).
*   **GGUF Format:** Explains that GGUF is a single-file format containing the model architecture, weights, and tokenizer, making it highly portable and efficient.
*   **Python Integration:** Deep dives into `llama-cpp-python`. It shows how to install it with hardware acceleration (e.g., `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`).
*   **High-Level API:** Demonstrates how to use the Python `Llama` class for text completion, chat completion, and pulling models directly from the Hugging Face Hub.
*   **Server API:** Shows the `llama-server` OpenAI-compatible endpoint pattern — `openai.OpenAI(base_url="http://localhost:8080/v1")` — which is the foundation of the Phase 7 chatbot.
*   **UI integrations:** Covers Ollama, LM Studio, Jan.ai, and GPT4All — all of which use `llama.cpp` as their backend. Open WebUI (Phase 10 Option B) follows the same pattern.
*   **Applications:** Shows how to integrate `llama.cpp` with LangChain and how to build a multimodal chat UI using Gradio and Vision Language Models.

### Article 3: llama-server OpenAI API Documentation
URL: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
*   **Core Focus:** Complete reference for `llama-server` flags, REST endpoints, and configuration.
*   **Relevant endpoints:** `POST /v1/chat/completions` (chat), `GET /health`, `GET /metrics` (Prometheus), `POST /tokenize`, `POST /detokenize`.
*   **Slot system:** Each `-np` slot is an independent KV cache. With `-np 1` (our config), all requests are serialized — simpler for a personal chatbot.
*   **Grammar support:** `llama-server` supports `grammar` field in requests for JSON-constrained output — useful for building structured chatbot APIs.

### Article 4: ChromaDB Getting Started (RAG vector store)
URL: https://docs.trychroma.com/getting-started
*   **Core Focus:** Local vector database for embedding-based retrieval. Runs fully in-process (no server needed for development).
*   **Relevance:** Phase 8 (RAG) uses ChromaDB to store document embeddings and retrieve relevant context at query time.
*   **Integration pattern:** Embed with `sentence-transformers`, store in ChromaDB `PersistentClient`, query at inference time and inject results into the system prompt.

---

## 2. Revised Learning Path & Philosophy

### Why This Path Changed

After completing Phases 1–3, it became clear that the original plan had a bottleneck in Phase 4 (raw HF → GGUF conversion). That pipeline requires:
- Downloading 20–40 GB of raw `safetensors` weights (disk and time cost)
- Significant RAM headroom during conversion (often 2× model size)
- No tangible learning benefit until after conversion succeeds

This machine (15 GB RAM, 4 GB VRAM) can run many excellent pre-quantized models directly from Hugging Face. The more productive learning direction is:

1. **Explore the GGUF quantization landscape** — try multiple pre-built variants of the same model, measure real trade-offs on this hardware
2. **Master inference tooling** — `llama-cli` for exploration, `llama-server` as a proper daemon
3. **DevOps-first deployment** — run the server as a `systemd` service, manage processes, expose it safely
4. **API integration** — wrap the OpenAI-compatible endpoint in a FastAPI chatbot with conversation memory and system persona
5. **Domain-specific chatbot** — use system prompts and RAG to specialize the model
6. **Python bindings** — `llama-cpp-python` for direct programmatic control
7. **GGUF creation (Extra track)** — when disk and time are available, learn the full HF → GGUF pipeline as a bonus exercise

---

## 3. Detailed Project Plan

Based on the machine capabilities (15 GB RAM, 10 CPU threads, 4 GB VRAM GTX 1650 Ti) and the reference guides, here is the revised exhaustive step-by-step plan:

---

### Phase 1: Core Environment Setup ✅
1.  **Clone `llama.cpp` Repository:** Fetch the latest source code from GitHub.
2.  **Build with CUDA Support:** Configure with `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, and `CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10`. Compile a Release binary.

---

### Phase 2: Python Environment & Dependencies ✅
3.  **Set up Python Virtual Environment:** Create an isolated `venv` to avoid polluting the system Python.
4.  **Install llama.cpp Python dependencies:** Install packages from `llama.cpp/requirements/` to enable conversion and tooling scripts.

---

### Phase 3: GPT-OSS Quick Path (Direct GGUF Inference) ✅
5.  **Pull pre-built GGUF from Hugging Face:** Use `ggml-org/gpt-oss-20b-GGUF` with the `-hf` flag — no local conversion needed.
6.  **Tune for this hardware:** `--n-cpu-moe 32`, `-ngl 99`, `-ctk q8_0`, `-fa auto`, `-np 1`, `-b 512`, `-t 10`.
7.  **Stabilize and benchmark:** Measure prefill, generation, and cached-prefill t/s; document optimal flags.

---

### Phase 4: GGUF Quantization Variant Exploration
**Goal:** Understand the real-world memory vs quality trade-off for different quantization levels by running actual benchmarks on this machine. Hugging Face hosts many models with multiple GGUF files (e.g., `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`, `IQ3_XXS`) in a single repo.

**How quantization naming works:**
| Name | Bits/weight | RAM (7B model) | Quality loss | Use case |
|---|---|---|---|---|
| `F16` | 16 | ~14 GB | None | Reference quality |
| `Q8_0` | 8 | ~7 GB | Negligible | Best quality that fits |
| `Q6_K` | 6 | ~5.5 GB | Minimal | Great quality/size balance |
| `Q5_K_M` | 5 | ~4.6 GB | Very low | Good default for 8 GB systems |
| `Q4_K_M` | 4 | ~4 GB | Low | Most popular; fits 4 GB VRAM |
| `Q3_K_M` | 3 | ~3.3 GB | Moderate | Emergency space saving |
| `IQ4_XS` | ~4 | ~3.8 GB | Low | Importance-quantized; better than Q4_K_M |
| `IQ3_XXS` | ~3 | ~3 GB | Moderate | Very small footprint |

**Steps:**
8.  **Choose a smaller model with multiple GGUF variants.** Good candidates for 15 GB RAM / 4 GB VRAM:
    - `Qwen/Qwen2.5-7B-Instruct-GGUF` (ggml-org mirror)
    - `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
    - `bartowski/Mistral-7B-Instruct-v0.3-GGUF`
    - `lmstudio-community/Llama-3.2-3B-Instruct-GGUF` (lighter, good for experimenting)

    All of these repos contain multiple `.gguf` files, one per quantization level.

9.  **Run each variant with the same prompt and measure:**
    ```bash
    ./build/bin/llama-bench \
      -m <model.gguf> \
      -p 512 -n 128 \
      -t 10 --no-mmap 0
    ```
    Record: model load time, prefill t/s, generation t/s, RAM used (`/proc/meminfo`), VRAM used (`nvidia-smi`).

10. **Use `llama-cli` for a quick qualitative check** — ask the same 3–5 test questions to each variant and assess answer quality vs the `Q8_0` reference.

11. **Build a personal benchmark sheet** documenting which quant tier is the practical sweet spot for this machine.

**Key insight to internalize:** The `-hf` flag in `llama-server` / `llama-cli` lets you specify the repo and the exact filename, so switching variants is a single flag change:
```bash
./build/bin/llama-cli \
  --hf-repo bartowski/Llama-3.2-3B-Instruct-GGUF \
  --hf-file Llama-3.2-3B-Instruct-Q5_K_M.gguf \
  -t 10 -p "Tell me about GGUF quantization"
```

---

### Phase 5: Native Inference Deep Dive (`llama-cli` & `llama-server`)
**Goal:** Become fluent in the two primary inference tools: `llama-cli` for interactive/scripted use, and `llama-server` as a production-grade local API server.

12. **`llama-cli` mastery:**
    - Interactive chat mode (`-i`, `-r`, `--in-prefix`, `--in-suffix`)
    - Chat templates with `--jinja` (important for instruct models)
    - Sampling parameters: `--temp`, `--top-p`, `--top-k`, `--repeat-penalty`, `--min-p`
    - Context management: `--ctx-size`, `--keep` (tokens to keep on context overflow), `--rope-scaling`
    - Grammar-constrained generation (`--grammar`, `--grammar-file`) to force JSON/structured output

13. **`llama-server` full feature walkthrough:**
    - Built-in web UI at `http://localhost:8080` — test the chat interface
    - OpenAI-compatible REST API: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`
    - Health / metrics endpoint: `GET /health`, `GET /metrics` (Prometheus format)
    - Slot management with `-np` (parallel users) and understanding KV cache sharing
    - System prompt injection via `--system-prompt` flag

14. **Benchmark and document** the chosen model+quant combo's limits: max reliable context size, degradation threshold, OOM conditions.

---

### Phase 6: DevOps Deployment — `llama-server` as a System Service
**Goal:** Stop running the server manually in a terminal. Set it up as a proper, auto-restarting background service the way a real deployment would work.

15. **Create a `systemd` service unit file:**
    ```ini
    # /etc/systemd/system/llama-server.service
    [Unit]
    Description=llama.cpp Inference Server
    After=network.target

    [Service]
    Type=simple
    User=nav_wsl
    WorkingDirectory=/home/nav_wsl/code/learn_llama_cpp/llama.cpp
    ExecStart=/home/nav_wsl/code/learn_llama_cpp/llama.cpp/build/bin/llama-server \
      -m /home/nav_wsl/.cache/llama.cpp/<model.gguf> \
      --ctx-size 8192 --jinja \
      -b 512 -ub 512 \
      -t 10 -tb 10 \
      -np 1 -fa auto -ngl 99 \
      -ctk q8_0 -ctv q8_0 \
      --host 127.0.0.1 --port 8080
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    ```
    Commands: `sudo systemctl enable llama-server`, `sudo systemctl start llama-server`, `journalctl -u llama-server -f`.

    > **WSL2 note:** `systemd` must be enabled in `/etc/wsl.conf` (`[boot] systemd=true`). Alternatively, use a startup script with `nohup` or `tmux`.

16. **Write a startup/teardown shell script** (`start_llm.sh`, `stop_llm.sh`) that wraps the service commands and shows live status — a minimal DevOps runbook.

17. **Set up log rotation** so `journald` or a custom log file doesn't grow unbounded during long inference sessions.

18. **Test resilience:** Kill the process mid-inference; confirm `systemd` auto-restarts it. Test the API is reachable after restart.

---

### Phase 7: API Integration — FastAPI Chatbot Layer
**Goal:** Build a Python FastAPI service that sits in front of `llama-server`, adds conversation memory, persona management, and a clean REST API for a personal chatbot.

19. **Install FastAPI + dependencies** in the venv:
    ```bash
    pip install fastapi uvicorn httpx openai python-dotenv
    ```

20. **Create `chatbot/main.py`** with:
    - `POST /chat` endpoint — accepts `{ "message": "...", "session_id": "..." }`
    - In-memory conversation history keyed by `session_id` (dict of message lists)
    - Forwards conversation to `llama-server` via the OpenAI-compatible client (`openai` SDK, `base_url=http://127.0.0.1:8080/v1`)
    - Returns the assistant reply

    Core pattern:
    ```python
    import openai

    client = openai.OpenAI(
        base_url="http://127.0.0.1:8080/v1",
        api_key="no-key-needed"
    )

    def chat(history: list[dict], user_msg: str, system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}] + history
        messages.append({"role": "user", "content": user_msg})
        resp = client.chat.completions.create(
            model="local",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    ```

21. **Add a `/reset/{session_id}` endpoint** to clear conversation memory for a session.

22. **Run both services together** — `llama-server` as the backend, `uvicorn chatbot.main:app --port 8000` as the API layer. Verify with `curl` or HTTPie.

23. **Write a simple test script** (`test_chat.py`) that sends a multi-turn conversation and prints the full exchange — this proves session memory is working.

---

### Phase 8: Domain-Specific Chatbot — System Prompts & RAG
**Goal:** Specialize the general-purpose model for a specific domain (e.g., a personal knowledge assistant, coding helper, or DevOps assistant) without any fine-tuning.

24. **System prompt engineering:**
    - Write a detailed, role-defining system prompt that constrains the model's persona, knowledge scope, and response format
    - Test with edge cases: questions outside the domain, ambiguous queries, adversarial prompts
    - Learn how the instruct model's chat template affects system prompt behavior (especially with `--jinja`)

25. **RAG (Retrieval-Augmented Generation) setup:**
    - Install a local vector store: `pip install chromadb sentence-transformers`
    - Use a small embedding model (e.g., `all-MiniLM-L6-v2` via `sentence-transformers`) to embed document chunks
    - Build a simple document ingestion script: read `.txt`/`.md` files → chunk → embed → store in ChromaDB
    - At query time: embed the user question → retrieve top-K relevant chunks → inject into the system/user prompt as context

    Minimal RAG retrieval pattern:
    ```python
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("my_docs")

    def retrieve(query: str, k: int = 3) -> str:
        embedding = embedder.encode(query).tolist()
        results = collection.query(query_embeddings=[embedding], n_results=k)
        return "\n\n".join(results["documents"][0])
    ```

26. **Integrate RAG into the FastAPI chatbot** — before forwarding to `llama-server`, retrieve relevant context and prepend it to the system message.

27. **Test end-to-end** with a real document corpus (e.g., your project documentation files, a set of markdown notes, or a technical manual).

---

### Phase 9: Python Bindings — `llama-cpp-python`
**Goal:** Move from using `llama-server` over HTTP to controlling the model directly from Python — useful for embedding the model into scripts, pipelines, or tools that can't easily talk to an HTTP server.

28. **Install `llama-cpp-python` with CUDA** in the venv:
    ```bash
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10" \
      pip install llama-cpp-python
    ```

29. **Write a basic inference script** using the `Llama` high-level API:
    ```python
    from llama_cpp import Llama

    llm = Llama(
        model_path="/home/nav_wsl/.cache/llama.cpp/<model.gguf>",
        n_gpu_layers=-1,
        n_ctx=4096,
        n_threads=10,
        flash_attn=True,
        verbose=False,
    )
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is quantization in LLMs?"},
        ]
    )
    print(output["choices"][0]["message"]["content"])
    ```

30. **Explore `llama-cpp-python`'s built-in OpenAI-compatible server:**
    ```bash
    python3 -m llama_cpp.server \
      --model <model.gguf> \
      --n_gpu_layers -1 \
      --n_ctx 4096
    ```
    Compare the behavior and overhead vs `llama-server` binary.

31. **LangChain integration experiment:**
    ```python
    from langchain_community.llms import LlamaCpp
    from langchain_core.prompts import ChatPromptTemplate

    llm = LlamaCpp(model_path="<model.gguf>", n_gpu_layers=-1, n_ctx=4096)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a DevOps expert."),
        ("human", "{question}")
    ])
    chain = prompt | llm
    print(chain.invoke({"question": "How does systemd manage service restarts?"}))
    ```

---

### Phase 10: Advanced Application — Web UI (Optional)
**Goal:** Replace the raw API with a proper chat UI suitable for day-to-day personal use.

32. **Option A — Gradio UI:**
    - Build a `gr.ChatInterface`-based app that calls the Phase 7 FastAPI chatbot
    - Add a system prompt input, model selector dropdown, and parameter sliders (temp, top-p)

33. **Option B — Open WebUI (recommended for personal use):**
    - Install Open WebUI (Docker or pip): it provides a full ChatGPT-like UI that connects directly to any OpenAI-compatible endpoint
    - Point it at `http://localhost:8080/v1` — zero custom code required
    - Supports multiple model slots, conversation history, RAG document uploads, and custom personas

34. **Option C — Continue using the llama-server web UI:**
    - `llama-server` ships with a built-in web UI at `http://localhost:8080` that is already functional for personal use

---

### Extra Track: Raw HF → GGUF Conversion Pipeline (Deferred)
**Why deferred:** This track requires downloading 20–40 GB of raw `safetensors` weights and enough RAM headroom during conversion. It is a learning exercise about the GGUF format internals, not required for running or building on top of models.

**When to do this:** When disk space is abundant (>100 GB free) and there is a specific model not yet available in GGUF format on Hugging Face.

E1. **Download a raw model from Hugging Face:**
    ```bash
    huggingface-cli download openai/gpt-oss-20b --local-dir ./models/gpt-oss-20b-raw
    # or a smaller model for learning:
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen-1.5b-raw
    ```

E2. **Convert to GGUF:**
    ```bash
    python llama.cpp/convert_hf_to_gguf.py \
      ./models/qwen-1.5b-raw \
      --outfile ./models/qwen-1.5b-f16.gguf \
      --outtype f16
    ```

E3. **Quantize with `llama-quantize`:**
    ```bash
    ./build/bin/llama-quantize \
      ./models/qwen-1.5b-f16.gguf \
      ./models/qwen-1.5b-Q4_K_M.gguf \
      Q4_K_M
    ```

E4. **Compare the self-converted GGUF** against the pre-built HF-hosted version using `llama-bench` to verify numerical equivalence and understand what the conversion pipeline does.

---

## 4. Phase Completion Status

| Phase | Description | Status | Log |
|---|---|---|---|
| Phase 1 | Core Environment Setup (clone + CUDA build) | ✅ Completed | `phase_1_detailed_logs.md` |
| Phase 2 | Python Environment & Dependencies | ✅ Completed | `phase_2_detailed_logs.md` |
| Phase 3 | GPT-OSS Quick Path (Direct GGUF inference) | ✅ Completed | `phase_3_gpt_oss_runbook.md` |
| Phase 4 | GGUF Quantization Variant Exploration | 🔲 Not started | — |
| Phase 5 | Native Inference Deep Dive (`llama-cli` & `llama-server`) | 🔲 Not started | — |
| Phase 6 | DevOps Deployment (`systemd` service, process mgmt) | 🔲 Not started | — |
| Phase 7 | API Integration — FastAPI Chatbot Layer | 🔲 Not started | — |
| Phase 8 | Domain-Specific Chatbot (System Prompts + RAG) | 🔲 Not started | — |
| Phase 9 | Python Bindings (`llama-cpp-python`) | 🔲 Not started | — |
| Phase 10 | Advanced UI (Gradio / Open WebUI) | 🔲 Not started — Optional | — |
| Extra | Raw HF → GGUF Conversion Pipeline | ⏸ Deferred | — |

---

## 5. Key Findings & Lessons Learned

### Build Flags (Phase 1)
- **CUDA 11.5 requires GCC 10** as host compiler (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10`). GCC 11.4 causes parameter pack errors.
- **`-DCMAKE_CUDA_ARCHITECTURES="75"`** (GTX 1650 Ti = cc7.5 only) reduces compile time, binary size, and improves runtime kernel dispatch by eliminating dead code for other GPU generations.
- **`-DGGML_CUDA_FORCE_MMQ=ON` is harmful** for this model. It replaces cuBLAS GEMM with integer MMQ kernels tuned for single-token decode, destroying large-batch prefill speed (7.96 → 2.21 t/s).

### Runtime Flags (Phase 3)
- **Threads (`-t 10 -tb 10`)** was the single biggest win: 6 → 10 threads improved prefill 1.21 → 7.96 t/s (+558%) because all MoE expert layers run on CPU.
- **`-np 1` (single parallel slot)** eliminates wasted KV cache VRAM (4 slots × 492 MiB → 1 slot × 261 MiB).
- **`-ctk q8_0 -ctv q8_0`** halves KV cache VRAM (492 → 261 MiB) with negligible quality impact, freeing VRAM headroom that improved generation speed.
- **`-ngl 99`** (all layers to GPU) is cleaner than hard-coding layer counts.
- **`--n-cpu-moe 32`** (all 32 experts → CPU) correctly reflects that all MoE experts overflow to host RAM with 4GB VRAM.
- **`FORCE_MMQ` revert + ARCHS=75 + q8_0 KV** together achieved: prefill 7.96 t/s, generation **11.33 t/s**, cached prefill **18.52 t/s**.

### Performance Summary (gpt-oss-20b MXFP4 MoE)

| Run | Build | Key Flags | Prefill | Generation | Cached Prefill | KV VRAM |
|---|---|---|---|---|---|---|
| v1 | default (6 archs) | 6t, np4, b256 | 1.21 t/s | 5.86 t/s | — | 492 MiB |
| v2 | default (6 archs) | 10t, np1, b512 | 7.96 t/s | 7.05 t/s | 14.39 t/s | 492 MiB |
| v3 | ARCHS=75 + FORCE_MMQ | 10t, np1, b512, q8_0 KV | 2.21 t/s | 6.61 t/s | 13.16 t/s | 261 MiB |
| **v4** | **ARCHS=75** | **10t, np1, b512, q8_0 KV** | **7.96 t/s** | **11.33 t/s** | **18.52 t/s** | **261 MiB** |

### Current Optimal Server Command
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

---

## 6. Execution Logs

Detailed implementation logs and methods are maintained in:
- `phase_1_detailed_logs.md`
- `phase_2_detailed_logs.md`
- `phase_3_gpt_oss_runbook.md`

---

## 7. Next Immediate Actions (Phase 4 Kickoff)

### Recommended Model Candidates for Phase 4 Benchmarking

All of these repos host multiple `.gguf` files with different quantization levels.
Pick one repo and run every variant through `llama-bench`.

| Repo | Size | Notes |
|---|---|---|
| `bartowski/Llama-3.2-3B-Instruct-GGUF` | 3B | Very fast; good for learning quant trade-offs |
| `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | 8B | Solid quality; Q4_K_M fits entirely in 4 GB VRAM |
| `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | 7B | Popular instruct model; many quant tiers |
| `Qwen/Qwen2.5-7B-Instruct-GGUF` | 7B | Strong multilingual + coding; official GGUF release |
| `ggml-org/Qwen2.5-Coder-7B-Instruct-GGUF` | 7B | Coding specialist; good for DevOps chatbot |

### Phase 4 Benchmark Command Template

```bash
# One-shot generation benchmark (adjust -m for each quant variant)
./build/bin/llama-bench \
  -m /home/nav_wsl/.cache/llama.cpp/<model-variant>.gguf \
  -p 512 -n 128 \
  -t 10 --no-mmap 0 \
  -ngl 99

# Check VRAM usage during inference
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

### Phase 6 Prerequisites Checklist (DevOps)

Before starting Phase 6, confirm:
- [ ] `systemd` is enabled in WSL2: `/etc/wsl.conf` has `[boot]\nsystemd=true`
- [ ] Chosen model is cached locally (not downloaded on every restart)
- [ ] Port 8080 is free and not blocked by Windows firewall
- [ ] `wsl --update` run recently to ensure stable WSL2 kernel
