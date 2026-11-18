# LM Forge Engineering TODO

Status labels: `now` blocks correctness or data safety; `next` improves reliability; `later` is a product enhancement. Each item needs a focused implementation and validation before moving on.

## Now — correctness and source hygiene

- [x] Verify the JSON-schema grammar test target exists. `engine::GbnfGrammar` is present; run the full suite.
- [x] Make all root tests pass and eliminate warnings.
- [x] Add correct repository ignores for models, dependencies, build output, archives, and dated backups.
- [x] Decide what source assets belong in the first Git commit; keep generated data untracked.
- [x] Add CI: root `cargo check`/`cargo test`, Tauri `cargo check`, frontend build.
- [x] Add a deterministic `lms bench` command with machine/runtime metadata and JSON output.
- [x] Store benchmark records locally, keyed by model checksum, quant, context, threads, backend, and machine fingerprint.

## Now — inference consolidation (resolved)

A process boundary would add daemon supervision, socket ownership, lifecycle failure modes, and deployment complexity. The lean solution is one shared Rust engine crate consumed by both binaries. This needs a workspace/package migration, so it is scheduled after the release-stability work.

- [x] Choose architecture: extract the root inference engine into a shared crate used by Tauri and the root binaries.
- [ ] Implement the shared crate migration and feature-parity tests.
- [ ] Remove duplicate inference paths after feature parity is tested.
- [ ] Ensure UI and CLI report the same model metadata, sampling options, thread policy, and benchmark results.
- [ ] Add explicit generation cancellation through the shared engine contract.
- [ ] Serialize model load/unload/generate operations to avoid engine-state races.

## Next — model management and safety

- [ ] Parse complete GGUF metadata before download/load: architecture, parameter count, quant, layers, context, chat template, and tokenizer.
- [ ] Calculate KV-cache and GPU/RAM budgets from metadata and selected context instead of file size alone.
- [ ] Present compatibility status for CPU capabilities, backend, context, RAM, and GPU offload.
- [ ] Retrieve and verify SHA-256 where a trusted model source supplies one.
- [ ] Mark downloads usable only after size/checksum validation.
- [ ] Create model details cards with metadata, compatibility, last load error, and benchmark history.

## Next — persistence and validation

- [ ] Replace WebView `localStorage` state with a versioned JSON app-data file.
- [ ] Add migration from existing localStorage sessions, settings, and custom presets.
- [ ] Validate persisted and UI-provided settings at frontend and Rust boundaries.
- [x] Replace frontend `as any` select casts with typed parsers and safe fallbacks.
- [ ] Add backup/export/import for app state.

## Next — error handling and observability

- [ ] Replace production serialization and lock `unwrap()` calls with errors or recovery logs.
- [ ] Add structured generation/load/download failure records with actionable UI messages.
- [ ] Add crash-safe log rotation and an in-app diagnostics export.
- [x] Add tests for corrupt GGUF headers.
- [ ] Add tests for malformed JSON-RPC input, cancellation, download failure, and persistence migration.

## Later — UI maintainability and product work

- [ ] Split `App.tsx` by existing boundaries: `ChatView`, `ModelsView`, `DiscoverView`, `ActivityRail`, and `DownloadsPopover`.
- [ ] Keep shared types/hooks close to their consumers; do not add a generic component framework.
- [ ] Add generation queue visibility and one-active-generation policy.
- [ ] Add benchmark comparison UI using recorded local results only.
- [ ] Add release versioning, changelog, package signing strategy, and upgrade migration tests.
- [ ] Evaluate optional external engines only after the shared engine contract is stable.

## Repository cleanup — completed

- [x] Permanently deleted `app-gui/local-ai-workstation.backup_2026-11-15_173204` by explicit confirmation.
- [x] Permanently deleted generated `dist`, `target`, duplicate `node_modules`, and package-store data by explicit confirmation.
