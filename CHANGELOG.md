# Changelog

All notable changes to AetherKit. Versions follow semantic versioning; 0.0.x releases are experimental previews with no API compatibility promise.

## Unreleased (0.0.2)

### Added

- **AetherFlux2, AetherFlux2Tokenizers, AetherFlux2CoreAI** (experimental): FLUX.2 [klein] 4B text-to-image and multi-reference editing. Includes bundle validation with provenance and revision pins, portable installation, the empirical-shift schedule, 4-axis rotary tables, the padded Qwen3 prompt, reference-image policy and preparation, and `Flux2GenerationSession`. The session supports prepared prompts and references, balanced/resident residency, cancellation, terminal unload and optional resource admission. Native parity against independent float32 Diffusers references passes at 12 cases, and the native lifecycle suite passes 44/44 checks: cancellation, unload during work, resident reuse and admission ([guide](docs/flux2-klein.md)).
- **Shared:** `AetherKit.SeededGaussianNoise` (Krea noise delegates to it, with identical values), `AetherAssets.DeclaredModelAsset`/`BundleFiles`, and a package-internal staged-model admission decorator in `AetherCoreAI`.
- **AetherKit:**
  - `SharedGPULease`, a cross-process GPU lease wire-compatible with Atelier's `/tmp/atelier-gpu.lock` protocol. It provides fail-fast and waiting acquisition with timeouts, cancellation and busy reports (including pid-less leases). Release is owner-checked, idempotent and done by renaming the lease aside, so it never leaves a pid-less lease. It also provides signal release, `.reclaim`-serialized stale reclaim that never replaces a newer lease, and inheritance only on an explicit hand-off from a live ancestor (`AETHER_GPU_LEASE_HOLDER`).
  - `SharedGPULease.ToolPolicy`, which reads the `AETHER_GPU_LEASE`, `AETHER_GPU_LEASE_TIMEOUT` (bounded to one year), `AETHER_GPU_LEASE_PATH` and `AETHER_GPU_LEASE_HOLDER` overrides.
  - `GPUAdmission`, `GPUAdmissionOwner` and `GPUAdmissionGrant`: explicitly shared in-process admission that queues fairly or fails fast. The lease is an optional backend for unsandboxed hosts, with bounded turns between processes.
  - `FairInferenceQueue`, adopted from an earlier transport-independent admission design.
- **Sessions:** `KreaGenerationSession`, `Flux2GenerationSession`, `H3VideoSession` and `LanguageGenerationSession` accept an optional `gpu: GPUAdmissionOwner`. With it, operation reservations are taken after the grant and `unload()` withdraws an operation still waiting for the GPU. Sessions without it are unchanged.
- **AetherLanguageCoreAI:** `LanguageStepContract`, detected from the asset's function signature. The new `bucketedPositions` contract (`cache_positions`, host-written cache rows, last-row logits, power-of-two prefill chunks, caches grown in 256-row steps) keeps input shapes in a bounded set, so it needs no native model recycling and its native shape cache stays bounded (one capacity per 256 tokens of context). It trades step cost for that: on Qwen3-0.6B fp16 a bucketed decode step measured 0.88× an unrecycled running step (0.84× at 8K), and time to first token is 2–3× longer, while against default recycling it decodes 1.37–1.43× faster because no reloads occur ([evidence](docs/evidence/language-bucketed-decode-2026-10-26.json)). `runningPositions` (0.0.1) assets keep working unchanged. `CacheStatistics.distinctStepShapes` reports the submitted (q, capacity) pairs.
- **AetherKit:** `LanguageExportBounds` and the optional `LanguageModelManifest.exportBounds` (exported prefill chunk and cache capacity). Bucketed assets must declare them; Kit refuses contexts beyond the exported capacity and keeps prefill chunks within the bound.
- **Forge/language:** development-only Qwen3 exporter for both contracts (bucketed by default; `--contract running` for the 0.0.1 contract), an upstream-Transformers CPU reference, and an evidence recorder.
- **AetherSpeech, AetherSpeechCoreAI** (experimental): Kokoro-82M speech synthesis from phoneme strings to 24 kHz float32 PCM. Includes typed voices for all 54 v1.0 voices, phoneme validation against the bundle vocabulary, a `SpeechPhonemizer` protocol for app-supplied G2P (no bundled G2P or espeak-ng), `kokoro.json` bundle validation with SHA-256 pins, portable installation, duration rounding and alignment, seeded vocoder noise, and `KokoroSpeechSession`. The session supports segment-by-segment delivery, cancellation, terminal unload and optional resource admission. Native parity against independent float32 upstream PyTorch references passes at four cases (exact frames, waveform within the model's float32 precision floor), with bit-exact cancellation recovery and active unload ([guide](docs/kokoro-speech.md)).
- **AetherKit:** `GeneratedAudio`, planar float32 PCM with 16-bit and float WAV output.

### Changed

- **aether-probe:** GPU commands (`inspect`, `smoke`, Krea render, batch and parity, `h3-profile-run`, `h3-profile-decode`, and the `qualify-h3*` and `benchmark-krea` modes other than `--prepare-only`) now take the shared GPU lease themselves when well formed, and wait while another holder has it. Malformed commands print their usage error without waiting. `Scripts/benchmark-krea.sh` relies on this.
- **Native suites:** the opt-in native test suites, FLUX.2 included, now take the lease through a test-local admission and run one GPU test at a time.
- `LanguageGenerationSession.open(at:...)`'s `maximumSubmissionsPerModel` is optional; nil selects the contract default (128 for running, none for bucketed). `open(at:)` and `init(model:)` take `exportBounds`.

See [GPU lease and admission](docs/gpu-lease.md).

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
