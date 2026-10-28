# AetherKit

Generative AI on Apple Core AI, with model-specific execution for Apple silicon.

**Status: 0.0.2 experimental developer preview.** APIs may change in any 0.0.x release, and there is no compatibility promise yet. No model weights are distributed. See the [changelog](CHANGELOG.md) and [known limitations](#known-limitations).

| Product | Maturity | What it provides |
| --- | --- | --- |
| `AetherKit` | Preview | Owned float32/fp16/int32 tensors, typed errors, decoded RGB/PNG images, PCM audio with WAV output, language model registration, the shared GPU lease and opt-in GPU admission. No Core AI dependency. |
| `AetherAssets` | Preview | Immutable local asset installation: staged SHA-256-verified import, installation IDs, reader leases, abandoned-stage recovery. No Core AI dependency. |
| `AetherCoreAI` | Preview | Local Core AI model loading and introspection, a stateless typed reference executor, installed-asset model sessions and optional shared resource reservations. |
| `AetherKrea` | Preview | Krea bundle validation, prompt/conditioning and reference-image contracts, refine schedules, rotary inputs and seeded noise. |
| `AetherKreaTokenizers` | Preview | Optional local tokenizer adapter (Hugging Face `Tokenizers`); never downloads files. |
| `AetherKreaCoreAI` | Preview | Krea 2 Turbo text-to-image, with reference images and image-to-image refine: Qwen text encoder → four denoiser segments → Euler sampling → VAE decode. |
| `AetherLanguageTokenizers` | **Experimental** (0.0.2) | A language model's local tokenizer and own chat template: chat messages (with image placeholders) to rendered token IDs, and tokens back to text. Never downloads. |
| `AetherLanguageCoreAI` | **Experimental** | Local dense-language generation with a growable KV cache, streaming tokens and explicit unload; Qwen3-VL image + text prompts (`VisionLanguageSession`). |
| `AetherVision` | **Experimental** | Core-AI-independent image decoding and Qwen3-VL preprocessing. |
| `AetherH3`, `AetherH3Tokenizers` | **Experimental** | MiniMax H3 CPU contracts: geometry, packing, schedules, profile validation, stitching, video request types and a local tokenizer. |
| `AetherH3CoreAI` | **Experimental** | H3 native profile stages behind `H3VideoSession`: prompt conditioning, keyframes, denoising, video/audio decoding. Graph, trajectory and memory gates pass at 960×544 under an 18 GiB bar; media audit is pending ([guide](docs/h3-integration.md)). |
| `AetherFlux2`, `AetherFlux2Tokenizers` | **Experimental** (0.0.2) | FLUX.2 [klein] 4B CPU contracts: bundle validation, padded Qwen3 prompt, empirical-shift schedule, 4-axis rotary, reference-image policy, and a local tokenizer. |
| `AetherFlux2CoreAI` | **Experimental** (0.0.2) | FLUX.2 [klein] 4B text-to-image and multi-reference editing: Qwen3 encoder → three denoiser segments → Euler → VAE ([guide](docs/flux2-klein.md)). |
| `AetherSpeech` | **Experimental** (0.0.2) | Kokoro-82M CPU contracts: typed voices, phoneme validation against the bundle vocabulary, an app-supplied `SpeechPhonemizer` protocol (no bundled G2P), bundle pins, duration alignment and seeded noise. |
| `AetherSpeechCoreAI` | **Experimental** (0.0.2) | Kokoro-82M speech synthesis: phonemes → duration graph → host alignment → iSTFTNet vocoder → 24 kHz float32 PCM, with segment-by-segment delivery ([guide](docs/kokoro-speech.md)). |
| `aether-probe` | Dev tool | Qualification, rendering and benchmark CLI; not SDK surface. |

## Requirements

- Package floor: macOS 14 and iOS 17, so apps can link the language facade there; it reports "unsupported OS" below 27. Everything that runs a model (Core AI, the `*CoreAI` products, the FoundationModels adapter) needs Apple silicon and macOS 27 or iOS 27 at runtime, and its declarations are `@available(macOS 27, iOS 27, *)` ([deployment floor](docs/deployment-floor.md)).
- Xcode 27.1 / Swift 6.4. Several targets enable Swift's experimental `Lifetimes` feature, which Core AI's span-based APIs require, so older toolchains are unsupported.
- **Qualified on one Apple M5 Max (64 GB) running macOS 27.0.1 (26A434).** Other Macs are expected to work but are unqualified. iOS is unqualified in this repository, except Kokoro speech on one iPhone.
- Model assets: your own local Core AI exports (see [Models and assets](#models-and-assets)).

## Installation

```swift
.package(url: "https://github.com/bubstackdev/aether-kit", exact: "0.0.2")
```

```swift
.product(name: "AetherKreaCoreAI", package: "aether-kit"),
.product(name: "AetherKreaTokenizers", package: "aether-kit"),
```

## Krea text-to-image

```swift
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import Foundation

let assets = try KreaAssets.open(at: bundleURL)            // reads headers, not model streams
let tokenizer = try await KreaTokenizer.open(assets: assets)
let session = KreaGenerationSession(assets: assets)       // preferGPU, balanced residency
let configuration = try KreaConfiguration(width: 512, height: 512, seed: 17017)
let image = try await session.generate(configuration: configuration,
                                       prompt: tokenizer.prepare("a red fox in fresh snow"))
try image.savePNG(to: outputURL)
await session.unload()
```

- **Repeated images of one prompt:** call `prepare(prompt:)` once, then `generate(configuration:preparedPrompt:)` for each seed.
- **Reference images:** `prepare(referencePrompt:references:)` runs reference pictures through the bundle's vision encoder ([guide](docs/krea-reference-images.md)).
- **Image-to-image refine:** `refine(...)` runs a partial schedule from encoded start latents ([generation guide](docs/krea-generation.md)).
- **Installed assets:** to bind native models to verified, lease-protected local installations, open sessions from an `AetherAssets` installation with `KreaGenerationSession(installed:)`.
- **Shared memory budgets:** `KreaResourceAdmission` reserves host-qualified estimates against a `ResourceCoordinator` ([Krea foundations](docs/krea-foundations.md)).
- **Residency:** `balanced` (the default) releases the encoder before loading denoisers. `.resident` keeps everything loaded, so later fresh prompts are much faster at the cost of more GPU memory (see measured performance below).
- **Concurrency:** one request runs per session at a time. Overlapping calls fail with `sessionBusy`. Cancel the calling task to stop between native calls. `unload()` waits for active work, then closes the session permanently.

### Measured performance

Measured 2026-10-26 on the M5 Max at 512×512, eight steps. Every image matched the independent references. See the [release-line record](docs/evidence/krea-release-benchmarks-2026-10-26.json) and [benchmark methodology](docs/krea-benchmarks.md).

| Scenario | Quiet machine | Release line, heavy concurrent load |
| --- | --- | --- |
| Fresh prompt each request, `resident` | 5.0 s | 5.0 s median (4.8–5.5 s) |
| Repeat image, conditioning already prepared | 4.6 s (about 0.535 s per step) | 6.5 s median |
| Fresh prompt each request, `balanced` (reloads about 19 GB of models) | 14.1 s median | 54 s median (37–85 s), reload-bound |

The quiet-machine column was measured the same day on a parallel development line that shares this sampling path. The resident results agree across the two runs. A quiet-window re-measure of balanced mode on this line is pending. Balanced mode's fresh-prompt latency depends heavily on memory pressure, because every request reloads the encoder and denoisers. GPU memory peaks at about 15 GB balanced and about 23 GB resident.

## FLUX.2 [klein] text-to-image and reference editing (experimental)

`Flux2GenerationSession` runs a locally exported FLUX.2 [klein] 4B bundle (Apache-2.0 checkpoint). It generates images in four steps without guidance, optionally conditioned on up to two reference pictures. Prompt and reference preparation, residency, cancellation and unload follow the Krea session's contract. Export the bundle with `Forge/flux2` in this repository. The qualified 0.0.2 bundle passes native parity against independent float32 Diffusers references computed on the CPU at 12 cases from 512² to 1344×768 (113/113 checks), and the native lifecycle suite (44/44). See the [FLUX.2 klein guide](docs/flux2-klein.md) for the export, API and the scope of that evidence.

## Language generation (experimental)

```swift
import AetherLanguageCoreAI

let session = try await LanguageGenerationSession.open(bundle: bundleURL, platform: .macos)
let reply = try await session.generate(prompt: renderedTokenIDs, maximumTokens: 512,
                                       prefillChunkSize: 64, onToken: { token in /* stream */ })
await session.unload()
```

Bundles are registered with a `language-model.json` manifest that Kit verifies before loading. Tokens stream while the next decode step runs on the GPU. `AetherLanguageTokenizers` renders chat messages through the model's own chat template into the token IDs the session takes.

New exports use the 0.0.2 bucketed step contract: input shapes stay in a fixed set (one 64-token prefill width, caches grown in 256-row steps). Core AI still retains Metal memory each time a model's input shape changes, so the session reloads the model once the memory retained since load passes a budget (2 GiB by default). It reloads at idle when it can; for the Qwen3-4B writer that is about once every 35 long prompts. 0.0.1-contract assets still reload every 128 submissions. Qualified models: Qwen3-0.6B, an abliterated Qwen3-4B-Instruct-2507 (Atelier's writer), and Qwen3-VL-2B and an abliterated Qwen3-VL-4B (Atelier's describer) with image input through `VisionLanguageSession` ([Qwen3-VL guide](docs/qwen3-vl.md)). See [language generation](docs/language-generation.md).

## H3 video with audio (experimental)

`H3VideoSession` takes an `H3VideoRequest` and returns frames and audio. The request carries the prompt, geometry, seed, optional first and last keyframes, and whether to generate audio. It runs over a validated schema-3 H3 profile. At 960×544 for text-to-video+audio and first-frame-to-video+audio, the denoiser graphs (blocks 0 and 49 at real step-0 contexts, prepare and heads), the decoders and the first-frame encoder pass against a CPU oracle, and all 457 block-boundary hashes of a capture replay. With measured admission (`H3ProfileEngine.measuredAdmission`), a render is refused before any stage loads unless its largest stage fits; the worst case is 15.3 GiB, under an 18 GiB bar. Measured admission knows one profile on one Mac (`c49d8dbf`, Turbo8, M5 Max) and refuses others. A 124-frame, eight-step render took about 6–7 minutes on the qualified Mac (busy-host timings). Media audit and per-stage lifecycle qualification are pending, so treat outputs and APIs as experimental. See [H3 integration](docs/h3-integration.md) and [H3 foundations](docs/h3-foundations.md).

## Kokoro speech synthesis (experimental)

```swift
import AetherSpeech
import AetherSpeechCoreAI

let session = KokoroSpeechSession(assets: try KokoroAssets.open(at: bundleURL))
let speech = try await session.synthesize(try session.phonemes("həlˈO wˈɜɹld."),
                                          configuration: try KokoroSpeechConfiguration(voice: .afHeart))
try speech.audio.saveWAV(to: outputURL)
await session.unload()
```

`KokoroSpeechSession` runs a locally exported [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) bundle (Apache-2.0 weights, 54 voices). It takes Kokoro phoneme strings, not text. The SDK contains no grapheme-to-phoneme code; apps supply it through `SpeechPhonemizer`, which keeps espeak-ng (GPL-3.0) out of the dependency graph. `synthesize(segments:)` delivers each utterance as soon as it is decoded. Cancellation and unload follow the Krea session's contract. Optional admission reserves model residency, per-call scratch and per-request operation storage against estimates checked to match the bundle. Native output matches the upstream PyTorch reference at four fixed cases within the model's own float32 precision floor. See the [Kokoro guide](docs/kokoro-speech.md) for the export recipe, licensing decision, compute placement and tolerances, and `Examples/Speech` for a consumer command.

Measured 2026-10-26 on the M5 Max, float32, durations on CPU and decoder on GPU ([record](docs/evidence/kokoro-qualification-2026-10-26.json)). These are shared-host observations, not quiet-window benchmarks.

| Scenario | Result |
| --- | --- |
| Warm synthesis, unbucketed decoder (2.5–22.8 s utterances) | about 32× faster than real time (0.08 s for 2.45 s; 0.70 s for 22.8 s) |
| Warm synthesis, bucketed decoder (default export) | same speed as unbucketed at the same shape (paired ratio 0.95–1.17); padding costs at most the bucket ratio |
| New utterance length | unbucketed: 2–6 s specialization every time; bucketed: once per bucket, then 0.1–0.2 s |
| Memory over 50 distinct lengths | unbucketed: 19 GB footprint, growing; bucketed: about 4 GB through bucket 1024 (bucket 2048 adds about 8 GB of Metal) |
| CPU-only placement | about 1.3× real time |

## Models and assets

AetherKit loads local Core AI exports that you supply and never downloads weights.

- **Krea:** a bundle directory containing `aether.json` (the recipe), `metadata.json`, six or seven `.aimodel` assets with `main.hash` declarations, and `tokenizer/`. It can optionally be installed through `AetherAssets`. A full written bundle specification isn't published yet.
- **Language:** a registered bundle with `language-model.json`.
- **H3:** a schema-3 profile that references checkpoint ranges in place.
- **Kokoro:** a bundle exported by `Forge/kokoro/export_kokoro.py`: `kokoro.json` (contract, vocabulary and SHA-256 pins), two `.aimodel` assets and `voices/*.f32`. It can optionally be installed through `AetherAssets`.

Assets keep their own licenses.

## Development

```sh
sh Scripts/check-cpu.sh                         # weight-free Release suite (clears AETHER_* opt-ins)
sh Scripts/check-floor.sh                       # every product at the macOS 14 / iOS 17 floor, and the floor consumer
swift run -c release aether-probe doctor
```

Tests build at macOS 27: run them through `Scripts/check.sh` or `Scripts/check-cpu.sh`, or pass `--triple arm64-apple-macosx27.0` to `swift test` yourself. A bare `swift test` fails with `'RunTestsWithTheOS27Triple' is only available in macOS 27 or newer` ([why](docs/deployment-floor.md#tests-build-at-os-27)).

`Scripts/check.sh` uses SwiftPM's default build engine. Set `AETHER_BUILD_SYSTEM=native` in restricted sandboxes where the default engine's `dsymutil` step can't run. Both check scripts forward arguments to `swift test` (for example, `--filter` or `--xunit-output`). [GitHub CI](docs/ci.md) runs the weight-free Release suite and both independent consumers on GitHub's Apple silicon `xcode-27` runner with Xcode 27.1, for PRs targeting `main` and pushes to `main`.

Real-weight suites are opt-in and skipped by default:

- `AETHER_NATIVE_QUALIFICATION`
- `AETHER_NATIVE_DYNAMIC_QUALIFICATION`
- `AETHER_NATIVE_DECODER_QUALIFICATION`
- `AETHER_NATIVE_DECODER_PIPELINE_QUALIFICATION`
- the H3 profile tests
- `AETHER_NATIVE_KOKORO_QUALIFICATION` (via `Scripts/qualify-kokoro-native.sh`)
- `AETHER_NATIVE_FLUX2_QUALIFICATION` and `AETHER_NATIVE_FLUX2_LIFECYCLE`
- `AETHER_NATIVE_LANGUAGE_QUALIFICATION`, `AETHER_NATIVE_LANGUAGE_MEMORY_BUDGET`, `AETHER_NATIVE_LANGUAGE_OVERLAP` and the other `AETHER_NATIVE_LANGUAGE_*` suites
- `AETHER_NATIVE_QWEN3VL`, `AETHER_NATIVE_QWEN3VL_SESSION` and `AETHER_NATIVE_VL_MEMORY`

Run one at a time; [Krea generation](docs/krea-generation.md) has the commands. GPU-using probe commands, the benchmark script and the native suites take the shared GPU lease (`/tmp/atelier-gpu.lock`) themselves and wait while Atelier or another tool holds it; see [GPU lease and admission](docs/gpu-lease.md) for the `AETHER_GPU_LEASE*` overrides and the opt-in session admission. Python export and reference tooling runs in Forge's locked environments: `uv sync --frozen --project Forge` (language, Qwen3-VL, Kokoro and FLUX.2) and `uv sync --frozen --project Forge/h3-oracle` (H3). Engineering records and evidence are indexed in [docs](docs/README.md); release status is in the [0.0.2 release notes](docs/release-notes/0.0.2.md) and the [0.0.2 roadmap](docs/roadmap-0.0.2.md).

## Known limitations

- **Single qualified device:** one M5 Max running macOS 27.0.1. iOS is unqualified here, except Kokoro speech, which passed on one iPhone (iPhone19,4, iOS 27.1).
- **No Krea bundle specification**, and no hosted assets or download transport.
- **Slow first image:** model first-load dominates the first image in a process.
- **Krea progress callbacks are synchronous:** a slow observer stalls sampling.
- **Resource admission reserves caller estimates;** it is not measured footprint enforcement, and qualified estimates aren't shipped (except H3's measured admission entry).
- **Language:** Core AI retains Metal memory on every input-shape change until a model is released. Sessions bound it by reloading the model once a growth budget (2 GiB by default) is spent, at idle where possible; this is a workaround, not a fix. Bucketed decode runs at about 0.88× the speed of an unrecycled running-contract step (Qwen3-0.6B fp16). 0.0.1-contract assets reload every 128 submissions.
- **Qwen3-VL:** Core AI wires a separate copy of the decoder's text weights for text-only and image prompts, so a one-image request on the 4B wires about 16 GiB, and opening the decoder reads its 22.5 GiB compiled file into the page cache. Images snap to five fixed grids (stretching by up to about 15%, more beyond 16:9); one image per prompt; qualified on macOS only ([limits](docs/qwen3-vl.md#limits-and-follow-ups)).
- **H3:** media audit and per-stage lifecycle qualification are pending. Measured admission knows one profile on one Mac (`c49d8dbf`, Turbo8, M5 Max) and refuses others. The qualified profile is a local export outside this repository. The 2026-10-18 gates are historical, because their assets were lost.
- **Kokoro:** phoneme input only; the app supplies G2P. The decoder specializes once per length bucket on the GPU (about 1–6 s each), and larger buckets hold several GB of Metal memory until unload; bucket 2048 alone adds about 8 GB. Sessions therefore cap utterances at 1024 frames (about 25.6 s) by default on macOS and 512 (about 12.8 s) on iOS, where one iPhone was qualified; `frameCap: .extended` raises the cap to 2048, and longer segments are refused with their length and the cap. The duration graph runs on the CPU because its GPU specialization fails on macOS 27.0.1. 0.0.2 ships the float32 decoder only; the float16/int8 export flags are experimental and fail qualification.
- **CI scope:** PR checks cover weight-free contracts and consumer integration. Real-weight GPU qualification remains an explicit local operation.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE), which includes third-party notices. External model assets keep their own terms.
