# LM Forge

LM Forge is a native local LLM workstation for running GGUF models on-device. It has a Rust inference/daemon/API core, an optional egui desktop UI, and a Tauri + React workstation UI.

## What it does

- Loads GGUF models with `llama-cpp-2` / llama.cpp and memory maps model files.
- Runs local token generation through a Unix-socket JSON-RPC daemon.
- Provides OpenAI-compatible chat, completion, and embedding endpoints plus an Anthropic-compatible messages endpoint.
- Supports native GGUF chat templates, token streaming, sampling controls, stop sequences, and optional JSON-schema constraints.
- Searches Hugging Face model repositories, lists GGUF files, downloads with resume/cancellation, and manages local model files.
- Detects local CPU/RAM capabilities and estimates model fit before download.
- Provides local conversation persistence, model loading controls, generation presets, telemetry, and configurable inference settings.
- Supports local document ingestion, chunking, vector storage, and offline RAG search.
- Includes MCP-related module support and structured application logging.

## Interfaces

| Interface | Purpose |
| --- | --- |
| `lm-studio --daemon` | Headless JSON-RPC daemon on a Unix socket. |
| `lms` | CLI controller for loading, chatting, inspecting, downloading, checking model fit, server control, and RAG. |
| `lm-studio --server` | Local HTTP API server, default `127.0.0.1:1234`. |
| `lm-studio --gui` | Lightweight native egui desktop UI. |
| Tauri UI | LM Forge desktop workstation, in `app-gui/local-ai-workstation`. |

## CLI quick start

```bash
# Terminal 1: start daemon
cargo run -- --daemon

# Terminal 2: load a model
cargo run --bin lms -- load models/model.gguf --threads 4 --ctx 2048

# Generate text from the loaded model
cargo run --bin lms -- chat "Reply with exactly: LM Forge is running." --stream false

# Inspect daemon/model state
cargo run --bin lms -- ps

# Unload when finished
cargo run --bin lms -- unload
```

## HTTP API

The local server binds to `127.0.0.1:1234` and exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/messages`

Start it with a preloaded model:

```bash
cargo run -- --server --model models/model.gguf --port 1234
```

## Desktop UI

The Tauri application offers:

- Chat sessions with assistant-response copy support.
- GGUF model library and model load/unload actions.
- Hugging Face discovery and direct downloads.
- Per-machine RAM/CPU display and file-fit guidance before download.
- CPU thread, context length, FlashAttention, mmap, presets, and sampling controls.
- Throughput and system telemetry panels.
- LM Forge application branding and native app icons.

Start frontend development from its directory only:

```bash
cd app-gui/local-ai-workstation
pnpm run dev
```

## Architecture

```text
Tauri + React UI                  egui UI / CLI
       |                              |
       +------------ Rust core --------+
                    |
      llama.cpp via llama-cpp-2
                    |
                GGUF model
```

Core modules live in `src/`:

- `engine.rs`: llama.cpp model loading and inference.
- `daemon.rs`: JSON-RPC Unix-socket daemon and client.
- `server.rs`: OpenAI/Anthropic-compatible HTTP API.
- `discovery.rs`: GGUF inspection, hardware telemetry, model-fit estimation, downloads.
- `rag.rs`: offline document parsing, chunking, and vector retrieval.
- `mcp.rs`: MCP functionality.
- `ui.rs`: native egui UI.
- `logger.rs`: structured logging.

## Build and validation

```bash
# Rust core
cargo check
cargo build

# Tauri backend
cd app-gui/local-ai-workstation/src-tauri
CARGO_HOME=/home/leywin/.cargo cargo check

# Frontend
cd app-gui/local-ai-workstation
pnpm run build
```

## Current potential

- A private, offline local-AI desktop workstation that works without cloud inference.
- A local API runtime for OpenAI- and Anthropic-style clients.
- A model manager that can recommend downloadable GGUF files based on the current machine.
- A foundation for RAG, tool/MCP workflows, model benchmarking, and multiple optional inference engines.

## Known limits

- CPU-only speed depends heavily on model size, quantization, available RAM, context length, thermal limits, and thread configuration.
- The root test suite has one known pre-existing failing grammar test: `test_gbnf_grammar_from_json_schema` references unavailable `engine::GbnfGrammar`.
- The root engine and the Tauri embedded backend are separate implementations; benchmark each if comparing UI versus headless throughput.
- Do not move/delete a mapped GGUF while it is loaded.
