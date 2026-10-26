# Changelog

All notable changes to AetherKit. Versions follow semantic versioning; 0.0.x releases are experimental previews with no API compatibility promise.

## 0.0.1 — 2026-10-26

First experimental developer preview. Qualified on one Apple M5 Max running macOS 27.0.1 with Xcode 27.1 / Swift 6.4. See the [release checklist](docs/release-0.0.1-checklist.md) for scope and remaining gates.

### Added

- **AetherKit:** owned float32/fp16/int32 tensors and typed errors, `GeneratedImage` with reference float32 scaling and PNG output, and `LanguageModelManifest` registration and integrity checks.
- **AetherAssets:** immutable local asset installation (staged SHA-256-verified import, installation IDs, protected reader leases, lock-coordinated abandoned-stage recovery).
- **AetherCoreAI:** local Core AI model loading and introspection, and a stateless typed reference executor (`ModelSession`) with one-call admission, cancellation and awaited `unload()`. Native functions are resolved once per model session. Sessions can open from installed assets. `ResourceCoordinator` and `ModelResourceAdmission` provide optional transactional reservations.
- **AetherKrea:**
  - bundle validation with provenance pins and trusted shape-specialization registration
  - prompt, conditioning and reference-image contracts
  - image-to-image refine schedules, including an off-grid start sigma
  - rotary inputs, the flow schedule and seeded noise
  - acceptance of the vanilla Krea 2 Turbo export
- **AetherKreaTokenizers:** a local Hugging Face tokenizer adapter that never downloads files.
- **AetherKreaCoreAI:**
  - `KreaGenerationSession` for text-to-image, reference images (vision encoder) and refine
  - balanced and resident residency, prepared prompts for encode-once rendering
  - cancellation and terminal unload
  - installed-asset sessions and `KreaResourceAdmission`
- **AetherLanguageCoreAI (experimental):**
  - `LanguageGenerationSession` with a growable fp16 KV cache, chunked prefill and greedy decoding
  - `onToken` streaming, delivered while the next step runs
  - explicit `unload()`
- **AetherH3 / AetherH3Tokenizers / AetherH3CoreAI (experimental):**
  - H3 CPU contracts, profile validation, generated Turbo schedules, stitching and a local tokenizer
  - native profile stages behind `H3VideoSession`: prompt conditioner, keyframes, denoising with double-buffered weights, video and audio decoders
  - H3 checkpoint leases and stage admission
- **aether-probe:**
  - device diagnostics and model inspection
  - Krea render, batch (`--steps`, `--ref`), parity and benchmark commands; benchmark plans accept an optional `residency`
  - H3 qualification and profile commands

### Performance

- Model-stream verification drains each read chunk and hashes Krea streams concurrently, keeping memory bounded and taking `verify-krea` from about 7 s to about 3 s.
- Contiguous native tensor outputs are copied in bulk, with validation.
- Language token delivery overlaps the next decode step.

### Known limitations

See [Known limitations](README.md#known-limitations).
