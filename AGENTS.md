# lm-studio

Native Local LLM Workstation: a Rust engine (inference, JSON-RPC Unix-socket daemon, OpenAI/Anthropic-compatible HTTP API, RAG, MCP) with a Tauri 2 + Vite + React frontend. GGUF models live in `models/` and are zero-copy mmap'd on load.

## Dev environment

- Rust (edition 2021, cargo/rustc 1.99.1) for the engine; Node 22 + pnpm 10.34.3 for the frontend (see `app-gui/local-ai-workstation/.mise.toml`).
- Frontend node_modules are preinstalled (`app-gui/local-ai-workstation/node_modules`). Root `package.json` and `pnpm-lock.yaml` are empty stubs — ignore them.

## Build & test

### Rust engine
- `cargo check` — compile-check the crate.
- `cargo build` — full debug build (binaries `lm-studio` and `lms`).
- `cargo run --bin lms -- --help` — exercise the CLI daemon controller.
- `cargo test` — unit tests; **currently red**: `test_gbnf_grammar_from_json_schema` in src/lib.rs fails with E0433 (`engine::GbnfGrammar` does not exist). Pre-existing; 4 of 5 tests pass.

### Frontend
- `cd app-gui/local-ai-workstation && pnpm run dev` — Vite dev server (port 8443 per tauri.conf.json).
- `cd app-gui/local-ai-workstation && pnpm run build` — verified: exit 0 (~3s), output to `dist/`.
- `cd app-gui/local-ai-workstation && pnpm run tauri:dev` — Tauri desktop with the Vite dev server.

## Layout

- `src/` — Rust library (`lib.rs`) + `main.rs` (`lm-studio`) + `bin/lms.rs` (`lms`). Modules: discovery, engine, daemon, server, rag, mcp, ui, logger.
- `app-gui/local-ai-workstation/` — Tauri 2 desktop app (`src-tauri/` crate) + Vite+React+TS frontend.
- `models/` — GGUF cache, e.g. `qwen2.5-3b-instruct-q2_k.gguf` (~1.3 GB).
- `.claude/settings.local.json` — local Claude permissions; not a build artifact.

## Conventions

- CLI: `clap` derive (`#[derive(Parser)]`, `#[command]`, `#[arg])`.
- Rust errors: `Result<..., String>`, client-facing errors via `eprintln!`, structured logs via `log_info!/log_warn!/log_error!/log_debug!` macros (src/logger.rs). Tests in `#[cfg(test)] mod tests` inside src/lib.rs.
- Daemon: JSON-RPC over Unix socket at `/tmp/llm-daemon.sock` (override with `--socket`). Methods: load, unload, ps, chat, server_start/stop, pull, status. `InferenceEngine` managed as `Arc<RwLock<Option<>>>`.
- API: `src/server.rs` binds `127.0.0.1:1234` with `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages` (OpenAI + Anthropic-compatible). `lms server start/stop` controls the daemon, not HTTP.
- Frontend: React 19 + TS 5.7 + Tailwind CSS v4 (`@tailwindcss/vite`) + oxfmt, `@` alias for src, `src/main.tsx` → `src/App.tsx`.

## Pitfalls

- Never `npm/pnpm` at the repo root — that package.json is a stub. All frontend work is in `app-gui/local-ai-workstation/`.
- `cargo test` is red until `GbnfGrammar` exists; pre-existing, not an environment issue.
- The daemon idle-TTL reaper auto-unloads models after `idle_ttl` seconds of inactivity (default 300); `lms ps` shows `idle_seconds`.
- `settings_use_mmap` is true by default: `memmap2` + `LlamaModel::load_from_file` — do not delete or move `models/` files while loaded.
- `cargo build` at root does not touch the Tauri Rust crate (`app-gui/src-tauri/` is a separate crate with its own Cargo.toml, same dependency versions, not linked to the root crate).
