# Changelog

## Unreleased

M4 (AetherVision and AetherEmbeddings, docs/M4.md), after the 0.1.0 tag.

### New

- **Vision foundation (M4a):**
  - `ImageInput` (AetherRuntime): a `CGImage` or an image file, decoded with ImageIO, EXIF orientation applied by
    moving pixels, alpha dropped as Pillow's `convert("RGB")` or composited (ported from 0.0.x's `VisionRGBImage`).
  - `ImagePreprocessor` and `PreparedImage`: the typed `preprocess.image` block (MANIFEST.md §2.5) as a transform:
    `stretch`, `letterbox`, `centerCrop` and `fitLongSide` resizing with `vImage` or the bit-exact `pillowBicubic` and
    `torchUint8Bicubic` filters (0.0.x's antialiased bicubic in each reference's fixed-point arithmetic), scale, mean
    and std, RGB or BGR, `NCHW`, `NVCHW` or patch rows (Qwen-VL block-major, SigLIP 2 row-major), fp32 or fp16 host
    values converted at bind to the graph input's type, and an `ImageGeometry` to map results back to the source.
    Known answers from Pillow and torch (`Forge/vision/fixtures_preprocess.py`).
  - Typed `postprocess` (`depth`, `denseNms`, `detrSigmoid`, `embedding`, `gliner`, `yesNo`, `maxSim`, `superRes`) and
    `labels` (`{set: "coco-91"}` or `{file}`) blocks, decoded and validated (MANIFEST.md §2.8).
  - `StatelessPipeline` (AetherHost): the stateless drivers' shared core: static graphs checked against the driver's
    contract, residency and memory admission, `PipelineOptions`, GPU admission at the request's `GPUJob` priority,
    FIFO turns and cancellation between graph calls (`GraphResidency.run`).
  - `AetherError.invalidInput` for an image that doesn't decode or lay out.
  - `GraphResidency.validating(_:manifest:contract:)` replaces the image and speech drivers' own copies.
- **Depth (M4b):** AetherVision's first driver, `depth-estimator@1` (`DriverContract.depthEstimator`,
  `DriverID.depthEstimator`), for Depth Anything 3 small from our Forge export (`Forge/depth/`):
  - `DepthEstimator` (actor on `StatelessPipeline`), loaded through `Registry.vision` (`DepthEstimatorDriver` →
    `DepthEstimatorModel.estimator()`); `estimate(_: ImageInput)` or, for real time, `prepare(_:)` then
    `estimate(_: PreparedImage)`, with a `DepthRequest` (`resizeBack` override, `GPUJob`).
  - `DepthMap`: row-major values, optional confidence, the manifest's polarity, the `ImageGeometry`; `range` and a
    subscript.
  - The `depth` postprocess: `resizeBack` resamples to the source image bilinearly through the geometry, never reading
    letterbox padding.
  - Gate `DepthEstimatorGateTests`: rel L2 0.0002–0.0011 against the CPU float32 upstream reference (bar 0.02).
  - Catalog: `depth-anything-3-small` (macOS), `aether-models/aether-kit-depth-anything-3-small`, Apache-2.0.
- **Detection (M4e):** `detector-dense@1` (`DriverContract.detectorDense`, `DriverID.detectorDense`), for YOLOX-S from
  our Forge export (`Forge/yolox/`):
  - The detectors' shared types: `Detection` (label, score, box normalized in the source image and clipped to it,
    optional `DetectionMask`), `DetectionRequest` (`scoreThreshold`, `maxDetections`, `GPUJob`, over the manifest's
    defaults) and `DetectionLabels` (contiguous or COCO-id columns; the built-in `coco-91` table).
  - `ObjectDetector` (actor on `StatelessPipeline`), loaded through `Registry.vision` (`DenseDetectorDriver` →
    `ObjectDetectorModel.detector()`); `detect(_: ImageInput)` or `prepare(_:)` then `detect(_: PreparedImage)`.
  - The `denseNms` postprocess: objectness × best class, threshold, greedy per-class or class-agnostic NMS, the top
    `maxDetections`, exactly upstream YOLOX's `postprocess`; boxes mapped back through the letterbox geometry.
  - Gate `ObjectDetectorDenseGateTests`: the host NMS reproduces the reference's detections exactly from its preds; on
    the GPU, the same detections on all six cases (IoU ≥ 0.99999, scores within 2e-6; bar IoU 0.95, 0.01).
  - Catalog: `yolox-s` (macOS), `aether-models/aether-kit-yolox-s`, Apache-2.0.
- **DETR detection and instance masks (M4f):** `detector-detr@1` (`DriverContract.detectorDetr`, `DriverID.detectorDetr`),
  for RF-DETR nano and seg-nano from our Forge export (`Forge/rfdetr/`):
  - `ObjectDetector` runs it too (its head is picked by the manifest's driver and `postprocess` kind), loaded through
    `Registry.vision` (`DetrDetectorDriver` → `ObjectDetectorModel.detector()`).
  - The `detrSigmoid` postprocess: sigmoid scores, the best `maxDetections` labelled (query, class) pairs strictly
    above the threshold (ties by flat index), no NMS, exactly upstream RF-DETR's `PostProcess`; boxes through the
    geometry; with `masks`, each detection's query plane resampled bilinearly to the source and thresholded, only over
    the region that can exceed it.
  - `DetectionMask` is compact: the source's size, the foreground's `bounds` and one bit per pixel; `foreground`,
    `alpha()`, a subscript, `area` and `intersectionOverUnion(_:)`.
  - Gate `ObjectDetectorDetrGateTests` (both variants): the head reproduces upstream's detections from the reference's
    raw outputs; on the GPU, the same detections on all six cases per variant (IoU ≥ 0.99999, scores within 4.4e-6, mask IoU 1.0; bar IoU 0.95, 0.01, mask 0.95).
  - Catalog: `rf-detr-nano` and `rf-detr-seg-nano` (macOS), `aether-models/aether-kit-rf-detr-nano` and
    `-seg-nano`, Apache-2.0.

### Changed

- `postprocess` `detrSigmoid` has a typed `selection` (`topKPairs`, the default; the RF-DETR export's `x-selection`
  note before), and its `masks` block must be present exactly when the graph binds `masks`.
- `preprocess.image.resize.padValue` also takes a bare number (`114` is `[114]`), as the YOLOX export writes it.

- `Manifest.Postprocess.Depth.Polarity`'s documentation: `largerIsNearer` is disparity (Depth Anything V1/V2),
  `largerIsFarther` depth or distance, relative or metric (Depth Anything 3); it had the two the wrong way round.
- `Manifest.Postprocess` is an enum of typed variants (was `kind` plus loose parameters), `Manifest.Labels` is
  `.set` or `.file` (was `set`/`table` strings), and `Manifest.Preprocess.image` is `Manifest.ImagePreprocess`.

## 0.1.0 — unreleased

A from-scratch re-architecture of AetherKit on Apple's Core AI. 0.0.x continues at the `0.0.7` tag and on
`legacy/0.0.x`.

### Breaking

- **Platform floor:** macOS 27 / iOS 27 (0.0.x linked at macOS 14 / iOS 17 and reported `unsupportedOS` below 27).
- **Products renamed or replaced:** AetherCore, AetherRuntime, AetherText, AetherFoundationModels, AetherHost,
  AetherImage, AetherSpeech, AetherVideo, and the `Aether` umbrella, which re-exports the task modules (AetherVision,
  AetherEmbeddings and AetherDecide are placeholders until M4–M6). The 0.0.x frozen APIs (the `AetherLanguage` facade,
  `AetherAssets` preparation and readiness, `@AetherGenerable`) are not carried over; see the README's migration notes.
- **Model bundles** are described by the `aether.json` manifest (docs/MANIFEST.md, schema v1). 0.0.x prototype
  manifests are rejected (`unknownDriver`).
- **`from: "0.0.7"` resolves to 0.1.0.** Pin `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")` to stay on 0.0.x.

### New

- **Catalog and store** (AetherCore): a compiled-in catalog pinned to Hugging Face revisions with per-file SHA-256
  (no runtime catalog fetch); manifest schema v1 with a zoo-bundle adapter and a reviewed override table.
- **Runtime** (AetherRuntime):
  - manifest-driven Core AI graphs, a tensor arena, bindings, snapshot/restore state and a driver registry;
  - `DecoderEngine`: memory-sized KV, chunked prefill (≤ 512 tokens), O(1) rewind or hybrid snapshots, and
    token-exact pipelined greedy decode (about 500 tok/s on qwen3-0.6b, M5 Max) guarded by a load-time canary;
  - the generative foundation: seeded noise, flow-matching and epsilon schedules, `GeneratedImage` /
    `GeneratedAudio`, and residency planning from measured phase footprints (`GraphResidency`, `FootprintLedger`,
    `ResidencyPlan`);
  - device chains for block-by-block models: `GraphChain` (each call's output feeds the next on the GPU),
    `WeightBanks` (per-block weights streamed from bundle files into two banks), `ShapeProfile`, `SerialGate`.
- **Text** (AetherText): the Generator (prefix reuse, stop and tag parsing, a seeded sampler, JSON-schema structured
  output, Hermes tool calls) and `ChatSession`.
- **FoundationModels** (AetherFoundationModels): `AetherLanguageModel` behind `LanguageModelSession`, with
  `@Generable` guided generation and tools.
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
- **Generative pipelines** share one `PipelineOptions` (AetherHost; `ImagePipelineOptions`,
  `KokoroSpeechSession.Options` and `VideoPipelineOptions` are its aliases): graph cache, memory budget, an optional
  shared `ResourceCoordinator`, GPU admission and a footprint ledger. Each driver's `load` gives a model value
  (`LatentDiffusionModel`, `Flux2KleinModel`, `Krea2Model`, `KokoroModel`, `MiniMaxH3Model`) with settable options,
  from which the pipeline is made.
- **Image** (AetherImage): `LatentDiffusionPipeline`, `Flux2KleinPipeline` and `Krea2Pipeline`, sharing
  `ImageRequest`, measured admission (macOS default budget: three quarters of RAM) and cancellation between graph
  calls. **`ContentFilter`**: a Krea pipeline exists only with the app's filter, which reviews the prompt and the
  output; a refusal is `AetherError.GenerationFailure.contentRefused(stage:reason:)`.
- **Speech** (AetherSpeech): `KokoroSpeechSession` from phonemes or from text through an app-supplied
  `SpeechPhonemizer` (no GPL G2P ships); per-platform frame caps, prewarm, multi-segment output.
- **Video** (AetherVideo): `MiniMaxH3Pipeline` (`minimax-h3@1`), text to 24 fps video with 32 kHz stereo audio at
  960×544×124, macOS only, its 50-block transformer chained on the GPU with weights streamed from the bundle. It needs
  a bundle you export yourself with `Forge/h3/`: MiniMax H3 is not in the catalog and its bundle is not publicly
  hosted (license, below).
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
| MiniMax H3 (not in the catalog: a locally exported bundle) | macOS | per graph vs the original modules (≤ 0.0095 rel. L2); the driver's first steps bit-identical to the graph reference; phase peaks within the 18 GiB bar |
| the language models (Qwen3, Qwen3.5, Gemma 3/4, MiniCPM, LFM2.5, Granite, Nemotron, Mistral, …) | macOS, iOS | greedy output token-exact against a same-device reference |

The catalog's bundles are hosted under `aether-models` on Hugging Face; each carries its license and NOTICE. The Krea
realism repo is gated (mature). MiniMax H3's license grants no rights in the EU, UK, South Korea or USA, so its bundle
is not hosted: export it with `Forge/h3/` where the license allows and load it from disk. The licenses differ per
model; the README's table lists their terms.

### Carried over from 0.0.x (re-implemented, validated against 0.0.x's tests and fixtures)

- **Store:** staged SHA-256 installs and resume, reader leases, stage recovery, cross-process install locks, a
  free-space check, token and redirect safety, iOS background downloads.
- **Host** (AetherHost): the GPU lease (Atelier-compatible) and admission, the fair queue, model lifecycle (idle
  unload, memory-pressure release, iOS easing) and `ResourceCoordinator` reservations.
- **Text:** the sampler (matching Transformers' recorded processors), the structured-output constraint, the
  text-stream rules and the tokenizer oracle cases.
- **Generative recipes:** the seeded noise stream (bit-identical to 0.0.x), the schedule arithmetic, and the SD-Turbo,
  FLUX.2, Krea, Kokoro and MiniMax H3 host recipes.

### Known limitations

- FLUX.2, Krea and MiniMax H3 are macOS-only.
- Text-to-phoneme for Kokoro is the app's.
- The image, speech and video pipelines don't take part in `ModelHost`'s idle and memory-pressure unload; an app
  unloads them itself (`unload()`).
- Generative outputs are not bit-identical across devices (fp16 arithmetic differs); gates compare against the
  reference on each device.
- One bundle per platform where precision differs (SD-Turbo on iOS). The catalog picks the device's variant; a
  manifest's `variant.platform` is not checked at load.
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

H3's FL2VA keyframes; FLUX.2's and Krea's reference images (and Krea's refine).

### Changes by milestone

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
  0.0067). It is not publicly hosted and not in the catalog.

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
