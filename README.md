# AetherKit

Generative AI on Apple Core AI, with model-specific execution for Apple silicon.

**Status: 0.0.1 experimental developer preview.** APIs may change in any 0.0.x release, and there is no compatibility promise yet. No model weights are distributed. See the [changelog](CHANGELOG.md) and [known limitations](#known-limitations).

| Product | Maturity | What it provides |
| --- | --- | --- |
| `AetherKit` | Preview | Owned float32/fp16/int32 tensors, typed errors, decoded RGB/PNG images, language model registration. No Core AI dependency. |
| `AetherCoreAI` | Preview | Local Core AI model loading, introspection and a stateless typed reference executor. |
| `AetherKrea` | Preview | Krea bundle validation, prompt/conditioning contracts, rotary inputs, schedule and seeded noise. |
| `AetherKreaTokenizers` | Preview | Optional local tokenizer adapter (Hugging Face `Tokenizers`); never downloads files. |
| `AetherKreaCoreAI` | Preview | Krea 2 Turbo text-to-image: Qwen text encoder → four denoiser segments → Euler sampling → VAE decode. |
| `AetherLanguageCoreAI` | **Experimental** | Local dense-language generation with a growable KV cache, streaming tokens and explicit unload. |
| `AetherH3` | **Experimental** | MiniMax H3 CPU contracts: geometry, packing, Turbo8 schedule, export identity and probe plans. |
| `AetherH3CoreAI` | **Experimental, qualification only** | BF16 native sessions for H3 block/chain/phase/denoising qualification. No conditioner, decoders or media output. |
| `aether-probe` | Dev tool | Qualification, rendering and benchmark CLI; not SDK surface. |

## Requirements

- Apple silicon, macOS 27.
- Xcode 27.1 / Swift 6.4. Three targets enable Swift's experimental `Lifetimes` feature, which Core AI's span-based APIs require, so older toolchains are unsupported.
- **Qualified on one Apple M5 Max (64 GB) running macOS 27.0.1 (26A434).** Other Macs are expected to work but are unqualified. The package declares iOS 27, but iOS is unqualified in this repository.
- Model assets: your own local Core AI exports (see [Models and assets](#models-and-assets)).

## Installation

Add the package and the products you need:

```swift
.package(url: "https://github.com/atelier-private-dev/aether-kit", exact: "0.0.1")
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
- **Residency:** `balanced` (the default) releases the encoder before loading denoisers, using about 15 GB of GPU memory at 512×512. `.resident` keeps everything loaded (about 23 GB), making later fresh prompts about 2.8× faster.
- **Concurrency:** one request runs per session at a time. Overlapping calls fail with `sessionBusy`. Cancel the calling task to stop between native calls. `unload()` waits for active work, then closes the session permanently.
- **Integrity:** `assets.verify()` streams every model file through SHA-256. It's explicit because it reads about 19 GB.

See [Krea generation](docs/krea-generation.md) for shape specialization, parity commands and the CLI (`aether-probe render-krea`, `render-krea-batch`).

### Measured performance

Measured 2026-10-26 on the M5 Max at 512×512, eight steps. Every image matched the independent references. See the [record](docs/evidence/krea-residency-load-path-2026-10-26.json) and [benchmark methodology](docs/krea-benchmarks.md).

| Scenario | Time per image |
| --- | --- |
| Repeat image, conditioning already prepared | about 4.6 s (about 0.535 s per step) |
| Fresh prompt each request, balanced | 14.1 s median |
| Fresh prompt each request, resident | 5.0 s median |
| First image in a new process | 23–79 s, mostly first model loads |

## Language generation (experimental)

```swift
import AetherLanguageCoreAI

let session = try await LanguageGenerationSession.open(bundle: bundleURL, platform: .macos)
let reply = try await session.generate(prompt: renderedTokenIDs, maximumTokens: 512,
                                       prefillChunkSize: 64, onToken: { token in /* stream */ })
await session.unload()
```

Bundles are registered with a `language-model.json` manifest that Kit verifies before loading. Tokens stream while the next decode step runs on the GPU. The current export contract reloads the native model every 128 submissions. See [language generation](docs/language-generation.md).

## H3 (experimental, qualification only)

`AetherH3` and `AetherH3CoreAI` expose the plans and native sessions used to qualify MiniMax H3 stages on the GPU, including a linked eight-step provided-conditioning trajectory. They don't generate video or audio. See [H3 integration](docs/h3-integration.md).

## Models and assets

AetherKit loads local Core AI exports that you supply and never downloads weights.

- **Krea:** a bundle directory containing `aether.json` (the recipe), `metadata.json`, six `.aimodel` assets with `main.hash` declarations, and `tokenizer/`. A full written bundle specification isn't published yet.
- **Language:** a registered bundle with `language-model.json`.
- **H3:** prototype exports referenced by qualification manifests.

Assets keep their own licenses.

## Development

```sh
Scripts/check.sh                                # Release build + CPU test suite
swift run -c release aether-probe doctor
```

`Scripts/check.sh` uses SwiftPM's default build engine. Set `AETHER_BUILD_SYSTEM=native` in restricted sandboxes where the default engine's `dsymutil` step can't run. The two engines place products in different directories (`.build/out/Products/Release` versus `.build/arm64-apple-macosx/release`). The H3 qualification wrappers expect a native-engine build.

Real-weight suites are opt-in and skipped by default:

- `AETHER_NATIVE_QUALIFICATION`
- `AETHER_NATIVE_DYNAMIC_QUALIFICATION`
- `AETHER_NATIVE_DECODER_QUALIFICATION`
- `AETHER_NATIVE_DECODER_PIPELINE_QUALIFICATION`

Run one at a time; [Krea generation](docs/krea-generation.md) has the commands. The benchmark harness is `Scripts/benchmark-krea.sh` ([workflow](docs/krea-benchmarks.md)). Engineering records, decisions and qualification evidence are indexed in [docs](docs/README.md); release status is tracked in the [0.0.1 checklist](docs/release-0.0.1-checklist.md).

## Known limitations

- **Single qualified device:** one M5 Max running macOS 27.0.1. iOS is unqualified here.
- **No Krea bundle specification**, and no hosted assets.
- **Slow first image:** the first image in a process takes 23–79 s.
- **Synchronous progress callbacks:** a slow observer stalls sampling.
- **No global memory admission:** several sessions can exceed device memory together.
- **Language:** the export contract reloads the native model every 128 submissions. On-device qualification is tracked by the consuming app.
- **H3:** qualification sessions only. The whole-run lifecycle is unqualified.
- **No CI;** checks run locally.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE), which includes third-party notices. External model assets keep their own terms.
