# Aether-Kit

On-device models for Apple silicon, built directly on Apple's Core AI framework. Requires macOS 27 / iOS 27, Xcode 27.2
and Swift 6.4.

**Status: 0.1.0 in development.** This is a from-scratch re-architecture of AetherKit 0.0.x, which lives on at the `0.0.7`
tag and the `legacy/0.0.x` branch (see [Migrating from 0.0.x](#migrating-from-00x)). Start with
[`docs/DESIGN.md`](docs/DESIGN.md) (the architecture) and [`docs/MANIFEST.md`](docs/MANIFEST.md) (the model bundle contract).
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) tracks what is being carried over from 0.0.x.

## What's here

| Module | What it provides |
| --- | --- |
| `AetherCore` | Model identity and the pinned catalog, with per-file SHA-256 and no runtime catalog fetch. Also the model store: atomic staged installs, resume, joined downloads, reader leases, stage recovery and iOS background downloads. And the `aether.json` manifest schema with the zoo-bundle adapter and its override table. No Core AI dependency. |
| `AetherRuntime` | Core AI graphs with manifest-driven specialization, a tensor arena, bindings and snapshot/restore state, plus a driver registry. `DecoderEngine` runs causal LMs with memory-sized KV, the manifest's prefill schedule, O(1) rewind or hybrid snapshots, and token-exact pipelined greedy decode guarded by a load-time canary. |
| `AetherText` | Tokenizer adapter (template-owned BOS, pinned sources), the Generator (prefix reuse, stop and tag parsing, a seeded sampler, JSON-schema structured output, Hermes tool calls) and `ChatSession`. |
| `AetherFoundationModels` | `AetherLanguageModel`: any supported catalog model behind Apple's `LanguageModelSession`, with guided generation (`@Generable`) and tool calling. |
| `AetherHost` | The shared GPU lease (wire-compatible with Atelier's lock), GPU admission (enforced by default, with per-job priorities), `ResourceCoordinator` (an explicitly shared memory budget), the generative pipelines' `PipelineOptions`, and `ModelHost`'s model lifecycle: idle unload and memory-pressure release for models loaded through it. |
| `AetherImage` | Image generation from a seed, with measured residency planning: `LatentDiffusionPipeline` runs SD-Turbo (`latent-diffusion@1`), `Flux2KleinPipeline` runs FLUX.2 [klein] 4B (`flux2-klein@1`, macOS only), `Krea2Pipeline` runs Krea 2 Turbo, vanilla and realism (`krea2@1`, macOS only). SD-Turbo is under the Stability AI Community License; check its terms before shipping it. FLUX.2 [klein] is Apache-2.0. Krea 2 is under the Krea 2 Community License, which requires a content filter: a Krea pipeline is only built with your `ContentFilter`, and what it allows is your responsibility. |
| `AetherSpeech` | Speech from phonemes: `KokoroSpeechSession` runs Kokoro-82M (`kokoro@1`, 54 voices, 24 kHz), on macOS and iOS. Text-to-phoneme is the app's (`SpeechPhonemizer`): no GPL G2P ships. Apache-2.0. |
| `AetherVideo` | Video with audio: `MiniMaxH3Pipeline` runs MiniMax H3 (`minimax-h3@1`, 960×544×124 at 24 fps with 32 kHz stereo audio), macOS only. MiniMax H3 Community License: its territory excludes the EU, UK, South Korea and USA, and the bundle is not publicly hosted: it runs from a bundle you export yourself (`Forge/h3/`). |
| `AetherVision` | Vision models on static graphs (M4): `DepthEstimator` runs Depth Anything 3 small (`depth-estimator@1`, relative depth and confidence, resized back to the source image); `ObjectDetector` runs YOLOX-S (`detector-dense@1`, the 80 COCO classes, boxes in the source image) and RF-DETR nano and seg-nano (`detector-detr@1`, COCO, seg-nano with instance masks at the source's resolution). macOS only so far. More drivers (CLIP, super-resolution) follow. |
| `AetherEmbeddings` | Embeddings on static graphs (M4): `TextEmbedder` runs EmbeddingGemma 300M (`text-embedder@1`, 768-dimensional unit vectors, query and document prompts, 256 tokens), macOS only so far. The reranker and classifiers follow. Gemma is provided under and subject to the Gemma Terms of Use found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms). |
| `Aether` | The umbrella: re-exports AetherText, AetherVision, AetherImage, AetherSpeech, AetherVideo, AetherEmbeddings and the placeholder. |
| `AetherDecide` | Planned (milestones M5–M6). |

### Models

| Catalog id | Driver | macOS | iOS | License |
| --- | --- | --- | --- | --- |
| `sd-turbo` | `latent-diffusion@1` | float32 | mixed precision (separate bundle) | Stability AI Community License: commercial use only under USD 1M revenue, with registration; "Powered by Stability AI" |
| `flux2-klein-4b` | `flux2-klein@1` | ✓ | refused (`unsupportedDevice`) | Apache-2.0 |
| `krea-2-turbo` | `krea2@1` | ✓ | refused | Krea 2 Community License: name begins with "Krea", NOTICE line, commercial use only under USD 1M revenue, **content filter required (§4.2)** |
| `krea-2-turbo-realism` | `krea2@1` | ✓ | refused | as Krea 2, plus the merged adapter's Civitai permissions. **Mature:** trained for explicit adult content; a gated Hugging Face repo (accept its terms, then install with a token) |
| `kokoro-82m` | `kokoro@1` | ✓ | ✓ (512-frame cap) | Apache-2.0 |
| `depth-anything-3-small` | `depth-estimator@1` | ✓ | refused | Apache-2.0 |
| `yolox-s` | `detector-dense@1` | ✓ | refused | Apache-2.0 |
| `rf-detr-nano` | `detector-detr@1` | ✓ | refused | Apache-2.0 |
| `rf-detr-seg-nano` | `detector-detr@1` | ✓ | refused | Apache-2.0 |
| `embeddinggemma-300m` | `text-embedder@1` | ✓ | refused | Gemma Terms of Use and the Gemma Prohibited Use Policy, passed on to your users. A gated Hugging Face repo with automatic approval (accept the terms, then install with a token) |
| none: MiniMax H3, a locally exported bundle | `minimax-h3@1` | ✓ | refused | MiniMax H3 Community License: **no rights in the EU, UK, South Korea or USA**; "MiniMax H3" shown in commercial products. Not in the catalog and not publicly hosted |

Every row but MiniMax H3 is in the built-in catalog, hosted under `aether-models` on Hugging Face. H3 has no catalog
entry: export the bundle with `Forge/h3/` where its license allows, and load it from disk (below). Check each license
before you ship a model.

## Quick start

```swift
.package(url: "https://github.com/bubstackdev/aether-kit", branch: "main")   // until 0.1.0 is tagged
```

A one-call `ChatSession(catalog:)` arrives with `ModelHost` in M6. Today, a session is built from an installed bundle:

1. Resolve a catalog entry for the device (`ModelRef.resolve`).
2. Install it (`ModelStore.install`).
3. Load or synthesize its manifest (`BundleManifest.loadOrSynthesize`).
4. Call `ChatSession.load(bundle:manifest:device:)`, then `respond(to:)`, which streams `GenerationEvent`s: `.text`,
   `.thinking`, `.toolCall` and `.finished`.

[`Tests/AetherTextTests/TextGateTests.swift`](Tests/AetherTextTests/TextGateTests.swift) is the complete, gate-verified
flow.

Image, speech and vision bundles carry their manifest (`.aether/aether.json`), so they load directly once installed:

```swift
import AetherCore
import AetherEmbeddings
import AetherImage
import AetherRuntime
import AetherSpeech
import AetherVision
import Foundation

let device = Device.current
let store = ModelStore(root: try ModelStore.defaultRoot())

// SD-Turbo (macOS and iOS)
let turbo = try await store.install("sd-turbo", for: device)
let sd = try LatentDiffusionPipeline(bundle: turbo, manifest: try Manifest.load(fromBundle: turbo.url), device: device)
let image = try await sd.generate(ImageRequest(prompt: "a lighthouse at dusk", seed: 7, steps: 4))
let png = try image.pngData()            // or image.cgImage()

// Krea 2 (macOS): a pipeline exists only with your ContentFilter. What it allows is your responsibility.
struct MyFilter: ContentFilter {
    func review(prompt: String) async throws -> ContentVerdict { .allow }   // call your moderation here
    func review(image: GeneratedImage, prompt: String) async throws -> ContentVerdict { .allow }
}
let kreaBundle = try await store.install("krea-2-turbo", for: device)
let krea = try await Registry.image.load(
    kreaBundle, as: Krea2Driver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default))
let kreaImage = try await krea.pipeline(filter: MyFilter()).generate(ImageRequest(prompt: "a fox in a birch forest"))

// Kokoro (macOS and iOS): text-to-phoneme is yours (no GPL G2P ships).
struct MyPhonemizer: SpeechPhonemizer {
    func phonemize(_ text: String, language: KokoroVoice.Language) async throws -> [String] { /* your G2P */ [] }
}
let kokoroBundle = try await store.install("kokoro-82m", for: device)
let speech = try KokoroSpeechSession(
    bundle: kokoroBundle, manifest: try Manifest.load(fromBundle: kokoroBundle.url), device: device)
let segments = try await speech.synthesize(text: "Hello there.", phonemizer: MyPhonemizer(), SpeechRequest(voice: .afHeart))
let wav = try segments[0].audio.wavData()

// Depth Anything 3 small (macOS): relative depth, larger is farther, at the source image's size.
let depthBundle = try await store.install("depth-anything-3-small", for: device)
let depth = try await Registry.vision.load(
    depthBundle, as: DepthEstimatorDriver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default))
let map = try await depth.estimator().estimate(ImageInput(contentsOf: URL(filePath: "/path/to/photo.jpg")))   // map.values, map.confidence

// YOLOX-S (macOS): COCO objects, boxes normalized in the source image.
let yoloxBundle = try await store.install("yolox-s", for: device)
let yolox = try await Registry.vision.load(
    yoloxBundle, as: DenseDetectorDriver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default))
let objects = try await yolox.detector().detect(ImageInput(contentsOf: URL(filePath: "/path/to/photo.jpg")),
                                                DetectionRequest(scoreThreshold: 0.4))   // [Detection]: label, score, box

// RF-DETR seg-nano (macOS): COCO objects with instance masks at the photo's size (rf-detr-nano: boxes only).
let segBundle = try await store.install("rf-detr-seg-nano", for: device)
let seg = try await Registry.vision.load(
    segBundle, as: DetrDetectorDriver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default))
let instances = try await seg.detector().detect(ImageInput(contentsOf: URL(filePath: "/path/to/photo.jpg")))
let alpha = instances.first?.mask?.alpha()   // DetectionMask: bounds, bits, foreground, alpha()
// EmbeddingGemma 300M (macOS): queries and documents are embedded with their own prompts, then compared by cosine.
// Its repo is gated: accept the Gemma terms on Hugging Face and give the store a token (see below).
let gemmaBundle = try await store.install("embeddinggemma-300m", for: device)
let embedder = try await Registry.embeddings.load(
    gemmaBundle, as: TextEmbedderDriver.self, options: LoadOptions(device: device, cachePolicy: .default, cache: .default)
).embedder()
let query = try await embedder.embed("Which planet is known as the Red Planet?", role: .query)
let documents = try await embedder.embed(["Mars is often called the Red Planet.", "Bread goes stale."], role: .document)
let best = documents.indices.max { query.similarity(documents[$0]) < query.similarity(documents[$1]) }
```

MiniMax H3 (macOS) has no catalog entry: its license grants no rights in the EU, UK, South Korea or USA, and its
bundle is not hosted. Export it yourself with `Forge/h3/` where the license allows, then load it from disk:

```swift
import AetherCore
import AetherVideo
import Foundation

let device = Device.current
let url = URL(filePath: "/path/to/minimax-h3/bundle")      // your Forge/h3 export
let manifest = try Manifest.load(fromBundle: url)
let local = ModelRef(
    catalogID: manifest.id, repo: "local/\(manifest.id)", revision: "local", variant: "macos", driver: manifest.driver,
    assets: [.init(name: "main", repo: "local/\(manifest.id)", revision: "local", subtree: "", files: [])])
let h3 = try await MiniMaxH3Pipeline(
    bundle: InstalledBundle(ref: local, url: url, isStale: false), manifest: manifest, device: device)
let clip = try await h3.generate(VideoRequest(prompt: "waves on a pebble beach at dawn", seed: 3))
// clip.frames: 124 RGB8 frames at 24 fps; clip.audio: the 32 kHz stereo soundtrack (`wavData()`)
```

The image, speech and video pipelines are not managed by `ModelHost`: call `unload()` to release their graphs.

`FLUX.2` works like SD-Turbo (`Flux2KleinPipeline(bundle:manifest:device:)`, which is `async`); its requests need a
`width` and `height`. A gated repo needs a token: `ModelStore(root:headers: ModelStore.huggingFaceAuthorization(token:))`.

Models download from pinned Hugging Face revisions when installed, and are verified file by file. Core AI isn't available
in the iOS Simulator: run on a device or a Mac.

## Migrating from 0.0.x

0.1.0 is a **breaking** release: new modules, new APIs and a macOS 27 / iOS 27 floor (0.0.x linked down to macOS 14 /
iOS 17).

- **To stay on 0.0.x,** depend on `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")`. **`from: "0.0.7"` will resolve to
  0.1.0 and break,** so pin it. Fixes for 0.0.x go to the `legacy/0.0.x` branch.
- **Product names changed:**
  - `AetherLanguage*` → `AetherText` + `AetherFoundationModels`;
  - `AetherAssets` → `AetherCore`;
  - `AetherKit`'s lifecycle and GPU lease → `AetherHost`;
  - `AetherCoreAI` → `AetherRuntime`.
- **The `@AetherGenerable` macros are gone.** Use FoundationModels' `@Generable` through `AetherLanguageModel`, or
  `StructuredOutput` with `ChatSession.respond(generating:)`.
- **Model bundles:** 0.1.0 reads `aether.json` per [MANIFEST.md](docs/MANIFEST.md). The 0.0.x prototype manifests, which
  are also named `aether.json`, are rejected as `unknownDriver`.

## Development

```bash
scripts/ci.sh                    # hermetic: catalog, warnings-as-errors build, public API, tests
AETHER_GATES=1 scripts/ci.sh     # plus real-model gates (needs downloaded models and a GPU)
docs/RELEASE.md                  # the release steps and the pre-tag checklist
scripts/check-api.sh --update    # record an intended public-API change in api/ (after a build)
```

The public-API check compares against `api/` and only reports until 0.1.0, when `"enforce": true` in
`scripts/api-freeze.json` makes a breaking change fail CI.

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
