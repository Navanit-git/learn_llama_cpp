# Reference Guides

External articles and documentation that inform this project's approach.

---

## Article 1: Llama.cpp Guide — SteelPh0enix Blog

**URL:** https://blog.steelph0enix.dev/posts/llama-cpp-guide/

**What it covers:**
- Building `llama.cpp` from scratch using `cmake`/`ninja` with CUDA or Vulkan backends
- Downloading raw HuggingFace models (`safetensors`), converting to GGUF via `convert_hf_to_gguf.py`, and quantizing with `llama-quantize`
- Quantization trade-offs: lower bit-width = less RAM but higher perplexity
- Using `llama-server` (OpenAI-compatible API + built-in web UI) and `llama-cli` (terminal chat)
- Generation parameters: Temperature, Top-K, Top-P, Context Size, Repeat Penalty

**Most relevant to:** Build setup (Phase 1), conversion concepts (Extra Track), inference flags (Phase 5)

---

## Article 2: The Ultimate Guide to Efficient LLM Inference — PyImageSearch

**URL:** https://pyimagesearch.com/2024/08/26/llama-cpp-the-ultimate-guide-to-efficient-llm-inference-and-applications/

**What it covers:**
- GGUF format internals: single-file format containing weights + tokenizer + metadata; successor to GGML format
- Quantization types supported: FP32, FP16, BF16, 8-bit, 6-bit, 5-bit, 4-bit, 3-bit, 2-bit, 1.5-bit
- `llama-cpp-python` bindings: installation with CUDA, `Llama` class high-level API, `from_pretrained` for HF Hub, `create_chat_completion`
- **OpenAI-compatible server pattern** (key for Phase 7):
  ```python
  import openai
  client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="no-key")
  resp = client.chat.completions.create(model="local", messages=[...])
  ```
- LangChain integration via `LlamaCpp` class
- UI frontends that use `llama.cpp` as their backend: Ollama, LM Studio, Jan.ai, GPT4All, Oobabooga
- Multimodal (LLaVA / Moondream) via Gradio — image upload + question answering

**Most relevant to:** Phase 7 (FastAPI chatbot), Phase 9 (Python bindings), Phase 10 (UI)

---

## Article 3: llama-server REST API Reference

**URL:** https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

**What it covers:**

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | List loaded models |
| `GET /health` | Service health check |
| `GET /metrics` | Prometheus-format metrics |
| `POST /tokenize` | Tokenize text |
| `POST /detokenize` | Decode token IDs |

**Key concepts:**
- `-np N` creates N independent KV-cache slots (parallel requests). With `-np 1`, all requests are serialized — correct for a single-user personal chatbot.
- `grammar` field in request body forces constrained output (JSON schema-based) — useful for building structured API layers.
- `--system-prompt` flag pre-fills a system message for all sessions.
- Flash Attention (`-fa auto`) is only beneficial when prompt batches are large; safe to keep always-on.

**Most relevant to:** Phase 5 (server feature walkthrough), Phase 6 (DevOps), Phase 7 (API integration)

---

## Article 4: ChromaDB Documentation — Getting Started

**URL:** https://docs.trychroma.com/getting-started

**What it covers:**
- Fully local vector database — runs in-process, no separate server required for development
- `PersistentClient(path="./chroma_db")` for disk-backed storage across restarts
- Collections store documents + embeddings + metadata
- Query by embedding vector: `collection.query(query_embeddings=[...], n_results=k)`

**Setup pattern for RAG (Phase 8):**
```python
import chromadb
from sentence_transformers import SentenceTransformer

# One-time setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("docs")

# Ingest documents
for i, chunk in enumerate(text_chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embedder.encode(chunk).tolist()],
        ids=[f"chunk_{i}"]
    )

# Retrieve at query time
def retrieve(question: str, k: int = 3) -> str:
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return "\n\n".join(results["documents"][0])
```

**Most relevant to:** Phase 8 (RAG domain chatbot)

---

## Article 5: LangChain + LlamaCpp Integration

**URL:** https://python.langchain.com/docs/integrations/llms/llamacpp/

**What it covers:**
- `LlamaCpp` LLM class works directly with a local `.gguf` file path
- Supports `n_gpu_layers`, `n_ctx`, `temperature`, `top_p` parameters
- Plugs into any LangChain chain, agent, or RAG pipeline the same way as any other LLM provider
- Can be used alongside `ChatPromptTemplate`, LCEL (`|` operator), and `RetrievalQA`

**Most relevant to:** Phase 9 (Python bindings, LangChain experiment)
