# Changelog

All notable changes to AetherKit. Versions follow semantic versioning; 0.0.x releases are experimental previews with no API compatibility promise.

## 0.0.1 — 2026-10-26

First experimental developer preview. Qualified on one Apple M5 Max running macOS 27.0.1 with Xcode 27.1 / Swift 6.4. See the [release checklist](docs/release-0.0.1-checklist.md) for scope and remaining gates.

### Added

- **AetherKit:**
  - owned float32/fp16/int32 tensors and typed errors
  - `GeneratedImage` with reference float32 scaling and PNG output
  - `LanguageModelManifest` registration and integrity checks
- **AetherCoreAI:**
  - local Core AI model loading and introspection
  - a stateless typed reference executor (`ModelSession`) with one-call admission, cancellation and an awaited `unload()`
  - native functions are resolved once per model session
- **AetherKrea:** bundle validation with provenance pins, trusted shape-specialization registration, prompt and conditioning contracts, rotary inputs, the flow schedule and seeded noise.
- **AetherKreaTokenizers:** a local Hugging Face tokenizer adapter that never downloads files.
- **AetherKreaCoreAI:**
  - `KreaGenerationSession`: text encoding, four denoiser segments per Euler step and VAE decoding
  - balanced and resident residency
  - prepared prompts for encode-once, many-seed rendering
  - cancellation and terminal unload
- **AetherLanguageCoreAI (experimental):**
  - `LanguageGenerationSession` with a growable fp16 KV cache, chunked prefill and greedy decoding
  - `onToken` streaming, delivered while the next step runs
  - explicit `unload()`
- **AetherH3 / AetherH3CoreAI (experimental, qualification only):**
  - H3 CPU geometry, packing, the Turbo8 schedule and export identity
  - BF16 native sessions for block, chain, outer-phase and linked eight-step denoising qualification
- **aether-probe:** device diagnostics, model inspection, Krea render/batch/parity/benchmark commands and H3 qualification commands. Benchmark plans accept an optional `residency`.

### Performance

- Model-stream verification drains each read chunk and hashes Krea streams concurrently. Peak memory during `verify-krea` fell from about 19.4 GiB to 29 MiB, and wall time from about 7 s to about 3 s.
- Language token delivery overlaps the next decode step.
- Measured Krea 512×512 performance is in the [benchmark notes](docs/krea-benchmarks.md#residency-and-load-path-2026-10-26).

### Known limitations

See [Known limitations](README.md#known-limitations).
