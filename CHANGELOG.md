# Changelog

## 0.1.0 — unreleased

A from-scratch re-architecture of AetherKit. 0.0.x continues at the `0.0.7` tag and on `legacy/0.0.x`.

### Breaking

- **Platform floor:** macOS 27 / iOS 27 (0.0.x linked at macOS 14 / iOS 17 and reported `unsupportedOS` below 27).
- **All products renamed or replaced:** AetherCore, AetherRuntime, AetherText, AetherFoundationModels, AetherHost
  (and the planned AetherVision, AetherEmbeddings, AetherDecide, Aether). The 0.0.x frozen APIs (the `AetherLanguage`
  facade, `AetherAssets` preparation and readiness, `@AetherGenerable`) are not carried over. See the README's
  migration notes.
- **Model bundles** are described by the `aether.json` manifest in docs/MANIFEST.md. The 0.0.x prototype manifests are
  rejected.
- **`from: "0.0.7"` dependencies will resolve to 0.1.0.** Pin `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")` to
  stay.

### New

- **Pinned catalog and manifest:** a catalog with per-file SHA-256 and no runtime catalog fetch. Manifest schema v1,
  with a zoo-bundle adapter driven by a reviewed override table.
- **Owned decode engine** on Core AI (`DecoderEngine`):
  - memory-sized KV;
  - manifest-driven prefill (prefill phases, ≤ 512-token chunks);
  - O(1) rewind, and snapshots for hybrid models;
  - token-exact depth-3 pipelined greedy decode (about 500 tok/s on qwen3-0.6b, M5 Max) guarded by a load-time canary.
- **Generator and ChatSession:** prefix reuse across turns, stop and tag parsing, a seeded sampler,
  JSON-schema structured output, Hermes tool calls.
- **`AetherLanguageModel`** for FoundationModels' `LanguageModelSession`, with guided generation and tools.

### Carried over from 0.0.x (re-implemented, validated against 0.0.x's tests and fixtures)

- **Store:** staged SHA-256 installs and resume, plus reader leases, stage recovery, cross-process install locks, a
  free-space check, token and redirect safety, and iOS background downloads.
- **AetherHost:** the GPU lease (Atelier-compatible) and admission, plus lifecycle (idle unload, memory-pressure
  release, iOS easing).
- **Text:** the sampler (matching HF Transformers' recorded processors), the structured-output constraint, and the
  text-stream behaviour, plus the tokenizer oracle cases.
- **Generative foundation (G0):**
  - the seeded noise stream (bit-identical to 0.0.x);
  - the Krea and FLUX.2 flow-matching schedule arithmetic;
  - `GeneratedImage` and `GeneratedAudio`;
  - `ResourceCoordinator` reservations;
  - new on top: typed `schedule`, `latent` and `residency` manifest blocks (MANIFEST.md §5), and `GraphResidency`
    with Metal-measured phase footprints.
- **SD-Turbo (G1),** the first generative port, in the new **`AetherImage`** product:
  - the `latent-diffusion@1` driver (`LatentDiffusionPipeline`): CLIP text encoder → U-Net steps on the host Euler
    epsilon update → VAE decoder, with measured residency planning, an optional shared `ResourceCoordinator` and GPU
    admission, cancellation between graph calls, and its own CLIP BPE tokenizer;
  - `epsilonEuler` schedules (`EpsilonEulerSchedule`, tables captured from the pinned Diffusers scheduler) and
    spatial latents in the manifest;
  - a reproducible Forge exporter (`Forge/sdturbo/`, float32), a CPU Diffusers reference and a GPU graph check. The
    gate runs the driver from the seed alone: RMSE 1.0e-5 (1 step) and 3.3e-5 (4 steps) against Diffusers.
  - SD-Turbo is under the Stability AI Community License (commercial use only under USD 1M revenue, with
    registration; "Powered by Stability AI" attribution). The exported bundle carries the license and NOTICE line.
- **Public-API freeze check** (`scripts/check-api.sh`, run by `scripts/ci.sh`): `swift-api-digester` dumps of
  AetherCore, AetherRuntime, AetherHost, AetherText and AetherFoundationModels compared with `api/`. Report-only until
  the 0.1.0 freeze sets `"enforce": true`.

- **FLUX.2 [klein] 4B (G2)** in `AetherImage`:
  - the `flux2-klein@1` driver (`Flux2KleinPipeline`): the bundle's Qwen3 chat template, padded to 512 rows → text
    encoder → three denoiser segments chained by their carries, per step, on the host Euler update with the empirical
    shift → VAE decoder; macOS only in 0.1.0 (`unsupportedDevice` on iOS before anything loads);
  - the typed `flux2` manifest block (MANIFEST.md §5.7, now normative) and optional graph roles in driver contracts
    (the reference-image VAE encoder);
  - a reproducible Forge exporter (`Forge/flux2/`): int8 per-channel weights on the text encoder and denoisers, both
    VAE graphs; its GPU gate against CPU Diffusers passes 31/31;
  - the Swift gate against CPU Diffusers from the seed: prompt ids exact, text real-row cosine ≥ 0.99921, the image
    from the reference's text rows SSIM 0.991 / PSNR 34.0 dB; the whole pipeline is recorded at SSIM 0.908 (0.0.x's
    bars; padding rows condition the denoiser, so the full path moves under any precision change).
- `ImagePipelineOptions` and `ImageRequest` are shared by the image pipelines, and take a `GPUMemoryProbe` (tests pass
  a fixed one). `TokenizerAdapter` can load without stop markers (`forGeneration: false`) for drivers that only encode
  prompts; `PromptRenderers.chatTemplate` reads a bundle's verified template.

- **H3 runtime prerequisites (G4 step 1)** (`AetherRuntime/Chain/`): `GraphChain` binds each call's output as the
  next call's input on the GPU (two slots, one stream, waits at the end or at sync points; byte-identical to the
  unchained path), `WeightBanks` streams per-block weights from bundle files into two banks with the sync rule
  enforced, `ShapeProfile` pins the shapes a measured footprint holds for, and `SerialGate` queues busy callers.
  MANIFEST.md §5.8 proposes the `weights` table.

- **H3 exporter (G4 step 2a)** (`Forge/h3/`): every graph the 0.0.x prototype qualified, with the transformer block,
  the Qwen3-VL decoder layer, the text refiner and the vision tower taking their weights as inputs from per-block
  bundle files (MANIFEST.md §5.8, now normative, decoded and validated as `Manifest.GraphWeights`); the per-graph
  oracle gate `check_h3.py`; and a GPU gate running the export's own block graph through `GraphChain` and
  `WeightBanks` (4 blocks × 4096 rows: output within 0.0051 relative L2 of the CPU, chained bytes equal to the
  unchained path's). The real-weight export (94 GB, run outside the licence's Excluded Territories per the owner)
  passes `check_h3.py` at 960×544×124: every graph within 0.0095 relative L2 of the original modules (block 49 at
  0.0067). It is not publicly hosted; the catalog entry waits for a compliant hosting route.

### In progress before 0.1.0 is tagged

Krea 2 Turbo (vanilla and realism), MiniMax H3 and Kokoro-82M on the new runtime; the SD-Turbo and FLUX.2 hosted
bundles and catalog entries, SD-Turbo's iPhone validation, and FLUX.2's reference images.
