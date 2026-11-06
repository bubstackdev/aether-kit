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
- **An enforced GPU lock with priorities** (AetherHost):
  - `GPUAdmission.shared` is the process's default admission: every image, speech and video pipeline and the text
    `Generator` (so `ChatSession` and `AetherLanguageModel`) admit through it unless given another
    (`PipelineOptions.gpu`, `Generator(gpu:)`, `ChatSession.load(gpu:)`), one grant per generation, clip, utterance
    request or chat reply. On unsandboxed macOS it also holds the machine's `SharedGPULease` and accepts a lease handed
    off through `AETHER_GPU_LEASE_HOLDER`; in sandboxed apps and on iOS it is in-process only. Tests opt out with
    `@_spi(Testing) GPUAdmissionOwner.unadmitted`.
  - `GPUPriority` (`.low`, `.medium`, `.high`) through an optional `GPUJob` on each request (`ImageRequest.job`,
    `SpeechRequest.job`, `VideoRequest.job`, `GenerationParameters.job`; absent: `.medium`). `.high` goes ahead of
    every waiting request, `.low` waits until nothing else is waiting, FIFO within a level; running work is never
    preempted. Setting `job.priority` moves a waiting request; it has no effect once the request runs. There is no
    queue-management API; cancelling the task leaves the queue.
  - `PriorityInferenceQueue` replaces `FairInferenceQueue` (round-robin across owners is gone; the owner is a label for
    the per-owner limit). Pipelines and the Generator now take the GPU before their own turn, so priority also orders
    one pipeline's requests.

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

- **Krea 2 Turbo, vanilla and realism (G3)** in `AetherImage`:
  - the `krea2@1` driver (`Krea2Pipeline`): the fixed Krea prompt, unpadded → text encoder (12 tapped layers, the
    34-row prefix dropped) → four denoiser segments chained by their carries, per step, on the host Euler update with
    the exponential shift → VAE decoder; macOS only in 0.1.0;
  - **`ContentFilter`**, required by the Krea 2 Community License (§4.2): `Krea2Driver.load` gives a `Krea2Model`, and a
    pipeline is only made with the app's filter (no default). It reviews the prompt before any work and the image
    before it is returned; a refusal is the new `AetherError.GenerationFailure.contentRefused(stage:reason:)`;
  - the typed `krea2` manifest block (MANIFEST.md §5.5–§5.6, now normative); a merged adapter's `scale` may be a
    number or a list;
  - the exporter writes the installed-revision layout (`.aether/aether.json`); the two local bundles' manifests were
    moved and regenerated in place (no re-export);
  - the Swift gate per variant against CPU Diffusers from the seed: rendered ids exact, text row cosine 0.999999, the
    whole pipeline SSIM 0.9989 / PSNR 49.9 dB (vanilla) and 0.988 / 39.6 dB (realism), repeats bit-identical;
  - the denoiser segments and VAE specialize without `expectFrequentReshapes` (with it their carries are non-finite);
  - image pipelines default to `MemoryBudget.imageGeneration`: three quarters of RAM on macOS (Krea plans about 34 GiB
    a generation), `platformDefault` on iOS.
- **H3 runtime prerequisites (G4 step 1)** (`AetherRuntime/Chain/`): `GraphChain` binds each call's output as the
  next call's input on the GPU (two slots, one stream, waits at the end or at sync points; byte-identical to the
  unchained path), `WeightBanks` streams per-block weights from bundle files into two banks with the sync rule
  enforced, `ShapeProfile` pins the shapes a measured footprint holds for, and `SerialGate` queues busy callers.
  MANIFEST.md §5.8 proposes the `weights` table.
- **Catalog:** `sd-turbo` (macOS and iOS) and `flux2-klein-4b` (macOS), from `aether-models/aether-kit-sd-turbo` and
  `aether-models/aether-kit-flux2-klein-4b` at pinned revisions, every file's SHA-256 recorded; `kokoro-82m` (macOS and iOS) from
  `aether-models/aether-kit-kokoro-82m`; `sd-turbo` on iOS takes the mixed-precision export from
  `aether-models/aether-kit-sd-turbo-ios`; `krea-2-turbo` and `krea-2-turbo-realism` (macOS) from
  `aether-models/krea-2-turbo-aether-kit` and the gated `aether-models/krea-2-turbo-realism-aether-kit`; catalog entries
  carry `mature` (the realism merge), and `scripts/gen-catalog` sends the Hugging Face token for gated repos.

- **Kokoro-82M speech (G5)**, in the new **`AetherSpeech`** product:
  - the `kokoro@1` driver (`KokoroSpeechSession`): phonemes (`KokoroPhonemes`, Unicode scalars, unknown symbols
    refused) → durations → host rounding and alignment → the bucketed decoder with seeded harmonic noise →
    `GeneratedAudio`; multi-segment requests, frame caps per platform, prewarm, measured admission, cancellation;
  - text through an app-supplied `SpeechPhonemizer` (no G2P ships: misaki falls back to GPL espeak-ng);
  - the typed `kokoro` block (MANIFEST.md §5.9, normative) and a reproducible Forge exporter (`Forge/kokoro/`, float32,
    54 voices). The Swift gate against unmodified upstream on the CPU: frames exact, rel. L2 0.0076–0.032 (bars
    min(0.05, 4× the fp32-vs-fp64 floor)), log-mel ≤ 0.015, repeats bit-identical.
  - Both graphs run on the GPU: Swift's Core AI runtime traps loading the durations graph for the CPU on macOS 27.2.

- **On-device qualification** (`docs/DEVICE.md`, `Tools/DeviceGate`): SD-Turbo and Kokoro pass their gates on a
  physical iPhone (iPhone Duo, `h19p`, iOS 27.2). SD-Turbo's iOS bundle is mixed precision (text encoder fp32, U-Net and
  VAE fp16; the Forge exporter gained per-graph `--precision` options): 4 steps RMSE 0.012 on device (bar 0.03), 1.7 GiB
  peak. Kokoro runs at its 512-frame iOS cap (longer utterances refused before decoding).

- **H3 exporter (G4 step 2a)** (`Forge/h3/`): every graph the 0.0.x prototype qualified, with the transformer block,
  the Qwen3-VL decoder layer, the text refiner and the vision tower taking their weights as inputs from per-block
  bundle files (MANIFEST.md §5.8, now normative, decoded and validated as `Manifest.GraphWeights`); the per-graph
  oracle gate `check_h3.py`; and a GPU gate running the export's own block graph through `GraphChain` and
  `WeightBanks` (4 blocks × 4096 rows: output within 0.0051 relative L2 of the CPU, chained bytes equal to the
  unchained path's). The real-weight export (94 GB, run outside the licence's Excluded Territories per the owner)
  passes `check_h3.py` at 960×544×124: every graph within 0.0095 relative L2 of the original modules (block 49 at
  0.0067). It is not publicly hosted; the catalog entry waits for a compliant hosting route.

- **MiniMax H3 (G4)** in the new **`AetherVideo`** product: `minimax-h3@1` (`MiniMaxH3Pipeline`), text to 24 fps video with
  32 kHz stereo audio at the qualified 960×544×124 profile, macOS only. The conditioner and the 50-block transformer
  chain on the GPU with their weights streamed from the bundle (`GraphChain`, `WeightBanks`); the host recipe is
  Diffusers' layout, row timesteps and rotary, bit-identical to the graph reference for the first two steps; a clip and
  its repeat bit-identical; phase peaks 4.2 / 9.7 / 1.4 GiB within the 18 GiB bar. The typed `h3` block (MANIFEST.md
  §5.10 normative); `TensorBuffer.readBytes`/`writeBytes`/`copyBytes` and `WeightBanks.load(block:into:)` for bf16
  tensors between host and Metal arenas. The exporter's manifest fixes: `video.noise.order` is `sourceVideoThenAudio`
  (0.0.x's draw order), the pack's text input is named `text`, and no graph takes the reshape hint. FL2VA keyframes are
  deferred; the bundle isn't publicly hosted (license).
- **Second pre-release review fixes:** store references are validated and contained before any file operation
  (`AetherError.invalidReference`); an install is reused only for the same installation; tokenizer paths are
  contained and stop ids bounded; released bindings no longer validate; structured replies must complete their
  schema; exact integer schema bounds, and constraints beside `$ref`/`anyOf` refused; ModelHost leaks no pin on a type
  mismatch and cancelled callers load nothing; the GPU-admission timeout race; cancellation between prefill chunks;
  canary off means depth one; a top-k crash at tiny temperatures; cumulative resident footprints keyed by revision
  and device; float32 logits reads 377 → 6 µs; bounded constraint cache; download backpressure; the device gate
  script fails on failed checks; `Aether` re-exports `AetherSpeech`.
- **Pre-release review fixes:** `ResidentAdmission` (AetherHost) keeps a pipeline's resident graphs reserved on a
  shared coordinator and releases them on refusal; measured footprints are used for manifests without residency
  (Kokoro) and after late releases; canvases need `maxImageTokens`; drivers load model values with settable options;
  `KokoroVoice.Language` is open; `GeneratedAudio.sampleCount` (was `frameCount`).

### Known limitations

- **Decoder capacity** is sized at load from the model's load-time Metal delta (at least its files), but memory
  arrives with the first run (G0: about 3× the files held). Re-sizing after the canary needs the engine to lower its
  capacity limit after construction.
- **Per-phase plans** assume only the phase before a phase is still releasing (≈5 s late return). Phases shorter than
  that (SD-Turbo's are sub-second) can overlap three deep; per-phase mode is the memory-short fallback, and Krea's
  phases, the case it exists for, run longer.
- **Kokoro** is admitted against the measurement for its frame cap's bucket; until a cap-bucket run (or `prewarm`)
  is measured, a smaller bucket's measurement stands in and under-counts.
- **Waiting turns:** five actor-owned FIFO queues (Generator, ChatSession, the image pipelines, Kokoro, `SerialGate`)
  keep a cancelled waiter until its turn comes (it then checks cancellation and leaves); one cancellation-aware
  primitive is planned. A cancelled `ModelHost` caller joined to another caller's load waits for that load to end.
- **The FoundationModels adapter** renders the prompt again for its context check (the Generator's duplicate render
  for attention-only models is gone).

### In progress before 0.1.0 is tagged

H3's FL2VA keyframes; FLUX.2's and Krea's reference images.
