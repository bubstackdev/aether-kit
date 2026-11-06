# Changelog

## 0.1.0 — unreleased

A from-scratch re-architecture of AetherKit on Apple's Core AI. 0.0.x continues at the `0.0.7` tag and on
`legacy/0.0.x`.

### Breaking

- **Platform floor:** macOS 27 / iOS 27 (0.0.x linked at macOS 14 / iOS 17).
- **Products renamed or replaced:** AetherCore, AetherRuntime, AetherText, AetherFoundationModels, AetherHost,
  AetherImage, AetherSpeech (AetherVision, AetherEmbeddings, AetherDecide and Aether are placeholders until M4–M6).
  The 0.0.x frozen APIs (the `AetherLanguage` facade, `AetherAssets` preparation and readiness, `@AetherGenerable`)
  are not carried over; see the README's migration notes.
- **Model bundles** are described by the `aether.json` manifest (docs/MANIFEST.md, schema v1). 0.0.x prototype
  manifests are rejected (`unknownDriver`).
- **`from: "0.0.7"` resolves to 0.1.0.** Pin `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")` to stay on 0.0.x.

### New

- **Catalog and store** (AetherCore): a compiled-in catalog pinned to Hugging Face revisions with per-file SHA-256
  (no runtime catalog fetch); manifest schema v1 with a zoo-bundle adapter and a reviewed override table.
- **Runtime** (AetherRuntime):
  - manifest-driven Core AI graphs, a tensor arena, bindings, snapshot/restore state and a driver registry;
  - `DecoderEngine`: memory-sized KV, chunked prefill (≤ 512 tokens), O(1) rewind or hybrid snapshots, and
    token-exact pipelined greedy decode guarded by a load-time canary;
  - the generative foundation: seeded noise, flow-matching and epsilon schedules, `GeneratedImage` /
    `GeneratedAudio`, and residency planning from measured phase footprints (`GraphResidency`, `FootprintLedger`,
    `ResidencyPlan`);
  - device chains for block-by-block models: `GraphChain` (each call's output feeds the next on the GPU),
    `WeightBanks` (per-block weights streamed from bundle files into two banks), `ShapeProfile`, `SerialGate`.
- **Text** (AetherText): the Generator (prefix reuse, stop and tag parsing, a seeded sampler, JSON-schema structured
  output, Hermes tool calls) and `ChatSession`.
- **FoundationModels** (AetherFoundationModels): `AetherLanguageModel` behind `LanguageModelSession`, with
  `@Generable` guided generation and tools.
- **Image** (AetherImage): `LatentDiffusionPipeline`, `Flux2KleinPipeline` and `Krea2Pipeline`, sharing
  `ImagePipelineOptions` / `ImageRequest`, measured admission (macOS default budget: three quarters of RAM) and
  cancellation between graph calls. **`ContentFilter`**: a Krea pipeline exists only with the app's filter, which
  reviews the prompt and the output; a refusal is `AetherError.GenerationFailure.contentRefused(stage:reason:)`.
- **Speech** (AetherSpeech): `KokoroSpeechSession` from phonemes or from text through an app-supplied
  `SpeechPhonemizer` (no GPL G2P ships); per-platform frame caps, prewarm, multi-segment output.
- **Forge exporters** (`Forge/`): reproducible, pinned exports for SD-Turbo, FLUX.2 [klein], Krea 2 (vanilla and
  realism, with the adapter merged before export), Kokoro and MiniMax H3, each with a CPU reference and a GPU gate.
- **Public-API freeze check** (`scripts/check-api.sh`, in `scripts/ci.sh`): the frozen modules are compared with
  `api/`; report-only until the 0.1.0 freeze (docs/RELEASE.md).
- **Device qualification** (`Tools/DeviceGate`, docs/DEVICE.md): the SD-Turbo and Kokoro gates on a physical iPhone.

### Models

| Catalog id | Platforms | Gate (against the upstream reference) |
| --- | --- | --- |
| `sd-turbo` | macOS (float32), iOS (mixed precision: fp32 text encoder, fp16 U-Net and VAE) | from the seed vs CPU Diffusers; also on an iPhone |
| `flux2-klein-4b` | macOS | per graph vs CPU Diffusers; image from the reference's rows SSIM 0.991 |
| `krea-2-turbo`, `krea-2-turbo-realism` | macOS | from the seed vs CPU Diffusers: SSIM 0.999 / 0.988 |
| `kokoro-82m` | macOS, iOS (512-frame cap) | vs unmodified upstream on CPU: frames exact, rel. L2 ≤ 0.032; also on an iPhone |
| the language models (Qwen3, Qwen3.5, Gemma 3/4, MiniCPM, LFM2.5, Granite, Nemotron, Mistral, …) | macOS, iOS | greedy output token-exact against a same-device reference |

Bundles are hosted under `aether-models` on Hugging Face; each carries its license and NOTICE. The Krea realism repo
is gated (mature). The licenses differ per model; the README's table lists their terms.

### Carried over from 0.0.x (re-implemented, validated against 0.0.x's tests and fixtures)

- **Store:** staged SHA-256 installs and resume, reader leases, stage recovery, cross-process install locks, a
  free-space check, token and redirect safety, iOS background downloads.
- **Host** (AetherHost): the GPU lease (Atelier-compatible) and admission, the fair queue, model lifecycle (idle
  unload, memory-pressure release, iOS easing) and `ResourceCoordinator` reservations.
- **Text:** the sampler (matching Transformers' recorded processors), the structured-output constraint, the
  text-stream rules and the tokenizer oracle cases.
- **Generative recipes:** the seeded noise stream (bit-identical to 0.0.x), the schedule arithmetic, and the SD-Turbo,
  FLUX.2, Krea and Kokoro host recipes.

### Known limitations

- FLUX.2 and Krea are macOS-only; reference images (FLUX.2) and reference images or refine (Krea) are not built yet.
- Text-to-phoneme for Kokoro is the app's.
- The image and speech pipelines don't yet take part in AetherHost's idle and memory-pressure unload policy.
- Generative outputs are not bit-identical across devices (fp16 arithmetic differs); gates compare against the
  reference on each device.
- One bundle per platform where precision differs (SD-Turbo on iOS); the manifest's `variant.platform` is not
  enforced at load.

### In progress before the 0.1.0 tag

- **MiniMax H3** (video and audio): the exporter and its oracle gate, then the `minimax-h3@1` driver. Its license
  grants no rights in the EU, UK, South Korea or USA, so its bundle is not publicly hosted.
