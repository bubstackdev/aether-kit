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
| `AetherHost` | The shared GPU lease (wire-compatible with Atelier's lock), GPU admission, and model lifecycle: idle unload and memory-pressure release. |
| `AetherImage` | Image generation. `LatentDiffusionPipeline` runs SD-Turbo (`latent-diffusion@1`) from a seed, with measured residency planning. SD-Turbo is under the Stability AI Community License; check its terms before shipping it. |
| `AetherVision`, `AetherEmbeddings`, `AetherDecide`, `Aether` | Planned (milestones M4–M6). |

More image, video and speech generation (Krea 2 Turbo vanilla and realism, MiniMax H3, FLUX.2 [klein], Kokoro-82M)
is being ported onto this architecture before 0.1.0 is tagged.

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
scripts/ci.sh                    # hermetic: catalog, warnings-as-errors build, tests
AETHER_GATES=1 scripts/ci.sh     # plus real-model gates (needs downloaded models and a GPU)
```

Licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
