# Changelog

## Unreleased

### Fixed

- **Describer memory growth (G5).** The 4B Qwen3-VL describer grew 5.6 GiB of Metal over 12 requests after warm-up.
  It was not the image path: Core AI retains memory on every decoder call wider than about 32 tokens, and on every
  decode step after one, even when the shapes repeat (text prompts of the same length grew alike). Two changes:
  - **Prefill width.** A Core AI decoder specialized with `expectFrequentReshapes` is now fed in calls of at most 32
    tokens (`DecoderEngine.Options.prefillWidth`, default `DecoderEngine.reshapeSafePrefillWidth`; the manifest's
    chunk restores the old schedule). The growth fell to 1,139 MiB. Prefill is about twice as slow (one picture 0.51 s
    instead of 0.24 s); the owner accepted that trade (2026-11-10). Applies to `decoder-lm@1` exports too.
  - **Reload budget (docs/VLM.md, V6).** What the narrow calls still retain is released by reloading the decoder's
    graphs once the model retains more than `DecoderEngine.Options.reloadBudgetBytes` (default
    `DecoderEngine.defaultReloadBudgetBytes`, 2 GiB, as 0.0.x's) past its level after the first request. The
    `Generator` checks at the end of every request, inside its turn and GPU grant, before the request's stream ends;
    `DecoderEngine.reloadIfRetainedPastBudget()` does it for other callers, and `DecoderEngine.reloads` counts them.
    KV state, position and images survive a reload, so prefix reuse continues. G5 passes: 880 MiB after warm-up. Over
    48 requests the engine reloaded twice (after requests 16 and 37), 2.7 s each (one run measured 9.9 s), and Metal
    stayed under 12.9 GiB.

### New

- **A live resident-bytes reading for kept graphs.** `Krea2Pipeline.liveResidentBytes` (and
  `ResidentAdmission.liveResidentBytes`, `GraphResidency.liveResidentBytes`) read the pipeline's probe at the call: the
  process's level above where it stood before the graphs loaded, Metal included, 0 when nothing is resident. It follows
  what the kept graphs retain after the plan, where `Krea2Pipeline.plannedResidentBytes` (and
  `ResidentAdmission.residentBytes`) stays the last plan's held bytes (what remains reserved on a coordinator). The
  probe reads the whole process, so models loaded after the pipeline count too. `DecoderEngine.Memory` names its live
  reading `liveResidentBytes` too. Additive; the API baselines record it.
- **Measured and declared model memory for the decoder engine** (`decoder-lm@1` and `vlm-static-inject@1`;
  docs/VLM.md §5, V5). Loading allocates almost nothing, so load charged the KV budget with the graph files' size
  (8.28 GiB for the 4B Qwen3-VL export, against 9.4–13.6 GiB of Metal measured over a session).
  - `DecoderEngine.memory` reads the engine's memory at the call: the level before load, now and at its highest, the
    engine's own buffers, the charge, `measuredModelBytes` (the model's memory above the engine's buffers) and
    `liveResidentBytes` (the level now above the level before load).
  - `DecoderEngine.Options.modelCharge` `.declared(bytes:)` declares the model's memory (an earlier session's
    measurement, or a gate's); load charges the larger of it and what it measured, and refuses with `capacity` when
    the KV no longer fits.
  - Without a declaration, the charge is as before. Additive; the API baselines record it.
  - The G5 retention gate (`DescriberRetentionGateTests`, opt-in): 20 mixed requests, the level after warm-up must
    grow by less than 1 GiB; then a second load charges the measurement. Not run yet.
- **Qualified model bytes for known decoder exports.** The kit keeps a table of measured model memory (package
  only), keyed by the export's `Manifest.contentDigest` and the chip's Core AI architecture, as H3's qualified
  footprints are; `DecoderEngine.qualifiedModelBytes(for:device:)` looks one up. A `decoder-lm@1` or
  `vlm-static-inject@1` bundle load with `Options.modelCharge` `.qualified` (the default) declares its export's entry,
  so the first load charges the measured memory without the host persisting `measuredModelBytes`; `.measured` or
  `.declared(bytes:)` wins, and an export or chip without an entry charges what load measures, as before.
  `DecoderEngine.declaredModelBytes` reports the figure load was given. The table ships empty: the G5 retention gate and the
  Qwen3 export gate now print the entry to add (`GATE qualified model bytes entry:`) and check an existing entry still
  covers what they measure. Additive; the API baselines record it.

- **Qwen3-VL references on synthetic pictures:** `Forge/qwen3vl/reference_qwen3vl.py --pictures <dir>` takes
  `synthetic_pictures.py`'s one-picture and text-only cases in place of the photographs, so pins that carry none (the
  4B pins) get CPU float32 references for `check_qwen3vl.py`.

- **Krea 2 at a fixed text length (`textMask`).** `Forge/krea2` exports the denoisers with upstream Diffusers' text
  key-padding mask, applied in the text-fusion refiner and all 28 joint blocks (`--text-mask`, the default;
  `--no-text-mask` keeps the exact-length graphs). The text axis is static at 1,116 rows, the most a reference prompt
  has, for text-only and reference prompts alike: one function per segment file and one shape for every prompt.
  - **Why one length:** on macOS 27.2, a second, shorter function in the same file slowed the first's steps about
    1.7×, and Core AI kept a separate copy of the weights for each function it ran.
  - **Contract:** `krea2@1` gains the optional `textMask` input, on all four segments or none.
  - **Driver:** `Krea2Pipeline` writes the real conditioning rows first and zero rows after, passes the mask, runs at
    the segments' static length (refusing rows past it) and keys admission on it.
  - **Rows reported:** `prepare(prompt:)`, the metrics and `Krea2Conditioning` still report the real rows.
  - **Older exports:** exact-length exports load and run as before.
- **Krea 2 reference conditioning beside resident denoisers.** A measured all-resident peak is now keyed by every
  role resident during the generation (see the admission entry below).
  - A Krea 2 reference encode releases the text-only text encoder, loads the vision tower and the text encoder's
    references function for that encode alone, and releases both after, so the denoisers stay loaded.
  - These are `package` APIs only (`GraphResidency.releasePhase`); there is no public API change.
- **Measured** on the masked realism export (1024², 8 steps, M5 Max 64 GB), all in one fresh process: prompts of 27,
  27, 8 and 258 rows, then a four-reference prompt of 621 rows.
  - Nothing reloaded after the first render. Resident memory stayed at 47.0–48.3 GB under the 48 GiB budget.
  - The first render took 118 s, with a 38.5 s first step that each process pays once. Every later render took
    35.6–36.9 s, the reference one 36.5 s.
  - Later steps took 4.1–4.3 s. Exact-length steps took 2.6–2.9 s at the prompts' own lengths; the 1,116 text rows
    cost the difference.
- **Gates against the Diffusers oracle** (SSIM ≥ 0.94; the comparison with the exact-length export is a report only):
  - 258-row prompt at 1024²: masked 0.9777 / 33.6 dB, exact length 0.9821 / 34.5 dB. The two are identical per step
    (teacher-forced cosine 0.999999); the gap builds up only over the closed loop.
  - Text-only prompt at 512²: 0.9927 / 41.6 dB. Refine: 0.99994. References: 0.9791.
  - Against the exact-length export: 43.7 dB / SSIM 0.995 at 512², and 39.3 dB / SSIM 0.991 on the 258-row prompt at
    1024².
- **Tools:** `check_krea2.py` pads and masks for masked bundles and saves its closed-loop image for
  `--compare-image`. `bench_krea2_{steps,segment,file,attention}.py` measure step and attention time.

- **Krea 2 progress and metrics** for hosts: `Krea2Pipeline.generate(_:observing:)` reports
  `ImageProgress` in order (conditioning ready with its row count, denoisers ready, each step with a copy of its packed
  latents, decoded) and returns an `ImageGeneration`: the image and its `ImageGenerationMetrics`. The metrics cover the
  wait for the GPU and the pipeline, both filter reviews, each phase, each step, the shapes, the graph roles the
  generation had to load, and the highest Metal allocation the probe read. A handler that throws stops the generation
  between graph calls and releases its graphs, which gives a host its own checks at these boundaries. `generate(_:)`
  is unchanged and renders the same image. The change is additive; the `AetherImage` API baseline records it.
- **Krea 2 prompt preparation:** `Krea2Pipeline.prepare(prompt:)` renders a prompt as `generate` does, on the CPU
  with no graph loaded and no filter call. It returns a `Krea2PreparedPrompt`: the ids the text encoder receives, the
  rendered count before truncation, the conditioning rows, whether the rendering was cut, and the export's
  `maxTokens`. A host that must not render a cut prompt can preflight it. A prompt `prepare` refuses, `generate`
  refuses too.
- **Krea 2 residency controls:** `setRetention(_:)` chooses `ImageRetention.keepLoaded` (the default, as before) or
  `.releaseAfterEachGeneration`. `releaseIfIdle()` releases every graph only while no generation is running or
  waiting: it never waits and never interrupts one, and returns `false` otherwise. `plannedResidentBytes` is the measured
  (Metal-inclusive) footprint the kept graphs hold, as the last generation planned it.
- **An opt-in memory-pressure gate for pipelines** (`PipelineOptions.pressure`, absent by default): with a
  `MemoryPressureMonitor` set, a Krea 2 generation that would have to load graphs throws `AetherError.memoryPressure`
  while the monitor reads critical, and graphs already resident still run.
- **Bundle identity and verification** for bundles opened from any folder: `InstalledBundle.contentIdentity()` is
  a location-free SHA-256 over the driver, the variant, every declared file (path, size, SHA-256) and the manifest's
  `contentDigest`. The store root, folder, repo and revision don't enter it, so a restored copy keeps its identity
  and a changed file or manifest gets a new one. `contentIdentity(capability:)` binds it to one capability of the
  bundle. `verify(progress:)` reads every declared file against its size and SHA-256 with progress and cancellation,
  as an install does, then decodes the manifest. Additive; the `AetherCore` API baseline records it.
- **Compiled-cache status:** `CompiledCache.status(of:manifest:device:roles:cache:)` asks Core AI's compile cache
  (`AIModelCache.model(for:options:)`) whether each graph already has a compiled copy for this device, using the
  specialization options a load from the manifest uses. It compiles nothing (the lookup's own cost on a large model is
  not yet measured), and returns each graph's
  declared source bytes, so a host can reserve disk before a first load (`uncompiledSourceBytes`). Additive; the
  `AetherRuntime` API baseline records it.
- **Bundle status** a host can show, for every driver's bundles: `InstalledBundle.status(supportedDrivers:)` returns
  `BundleStatus` `.notPrepared` (the folder is absent or empty), `.ready`, or `.blocked` with a `BundleBlockReason`:
  `incomplete` (files but no manifest), `missingFiles` (the declared files absent, with their total bytes),
  `unsupportedSchema`, `unknownDriver` (including a driver version this build doesn't know), `unsupportedDriver` (not
  among the caller's drivers, for example `Set(registry.ids)`), `invalidManifest` (naming the field) and `corruptFile`.
  It reads the manifest and stats every declared file, hashing nothing (52 ms for a 94 GB, 1,375-file local export).
  `verifiedStatus(supportedDrivers:progress:)` adds every declared file's SHA-256, with progress and cancellation,
  and reports a changed byte as `corruptFile`. Declared files are the reference's and the manifest's
  `assets.*.files`, each once, so a local export opened with a reference that lists no files is checked in full; a file
  both declare differently is `invalidManifest`. `InstalledBundle.status(ofLocalBundleAt:supportedDrivers:)` and
  `verifiedStatus(ofLocalBundleAt:supportedDrivers:progress:)` take a folder before there is a reference for it.
  Additive; the `AetherCore` API baseline records it.
- **Krea 2 conditioning reuse:** `Krea2Pipeline.encode(prompt:job:)` runs the text encoder alone (after the filter's
  prompt review) and returns a `Krea2Conditioning`, which the host owns. `generate(_:reusing:observing:)` renders from it
  at any seed and size without loading or running the text encoder, pixel for pixel what `generate(_:)` renders. It is
  refused for another export's conditioning or another prompt. An encode plans memory against the last generation's
  canvas, so the denoisers that generation kept stay loaded. Additive; the `AetherImage` API baseline records it.
- **Krea 2 refine (image-to-image)** for exports with the optional `vaeEncoder` graph (`DriverContract.krea2` now
  lists it; `supportsRefine` says whether an export has it). `encode(image:job:)` packs a planar RGB image on [-1, 1]
  into the decoder's `[rows, columns, 64]` latents. `generate(_:refining:reusing:noise:observing:)` runs from a
  `Krea2RefineStart`: a start step (`(1 − σ) · x0 + σ · noise`, then the remaining steps) or an off-grid last-step
  sigma. It optionally takes reused conditioning (one encode for every tile of a picture) and caller noise (tiles cut
  from one canvas-wide field). `FlowMatchSchedule` gains `replacingLastSigma(_:trainTimesteps:)` and
  `refineStart(x0:noise:step:)`. Additive; the `AetherRuntime` and `AetherImage` API baselines record it.
- **Krea 2 reference images** for exports with a `krea2.references` block, a `vision` graph and the text encoder's
  `references` function (0.0.x's `visionTokens` contract, ported exactly). `Krea2ReferenceImage` takes RGB bytes or an
  encoded image. `prepare(prompt:references:)` gives the expanded prompt's ids and conditioning rows on the CPU.
  `encode(prompt:references:job:)` runs the vision graph per image, then the references function: image rows as
  extension ids, M-RoPE positions, slot-indexed image embeddings and deepstack levels. It returns `Krea2Conditioning`
  with the images' digests in order (`referenceDigests`), for `generate(_:reusing:observing:)`. Preprocessing is
  bit-exact with 0.0.7, and the pixels and the pad expansion and positions are exact against the 0.0.x oracle's
  fixtures. For the second function: `DriverContract.GraphContract.inputsByPhase`, and multi-function graphs load as
  one specialization (`PhasedGraphBackend`, `GraphFunctionSet`). Additive; the API baselines record it.
- **GPU waiters (`AetherHost`)**: a waiting `SharedGPULease.acquire` and `scripts/gpu-lease` hold a shared flock on the
  lease's `<name>.waiters` file while they wait (this user's single-link, mode-0600 sibling, checked as `<name>.reclaim`
  is). A holder that keeps the GPU between its own jobs can ask `SharedGPULease.hasWaiters(directory:)`
  (or `othersWaiting` on its lease) and step aside. The answer also covers the lab wrapper's exclusive lock on
  `~/.cache/aether-kit/gpu.lock`; `hasWaiters(directory:legacyWrapper:)` names another wrapper lock, or none (`nil`). Never blocks. The lease protocol is otherwise unchanged. Additive; the `AetherHost`
  API baseline records it.
- **Compile-cache tooling:** `scripts/coreai-cache list [--entries]` shows Core AI's compile cache by OS build and
  bucket (one bucket per process name), with sizes, entry counts and dates. `scripts/coreai-cache prune` deletes what
  its selectors pick (`--bucket`, `--older-than DAYS`, `--other-os-builds`, `--stale --source DIR`), never a
  `--keep` bucket. It is a dry run without `--yes`, skips entries a process holds, never follows symlinks and refuses
  a root that isn't a Core AI cache. `AETHER_GATES=1 scripts/ci.sh` now prunes the test runners' buckets
  (`swiftpm-testing-helper`, `com.apple.dt.xctest.tool`) after the gates, under the GPU lease, which used to leave tens of GB behind;
  `AETHER_CI_KEEP_COMPILE_CACHE=1` keeps them. docs/RELEASE.md records what the cache keys on (the export's
  `main.hash`, the specialization options and the process's bundle identifier or executable name). It also records
  that `GraphOptions.Cache.appGroup` has no cache to open for an unsigned or ad-hoc signed tool or under `swift test`.
- **MiniMax H3 keyframes (FL2VA):** `VideoRequest.keyframes` anchors a clip to a picture at its first frame, its last
  frame or both (`VideoKeyframes.first`, `.last`, `.firstAndLast`; each a `VideoKeyframe`, RGB8 at the profile's
  canvas, not resized), with the new `VideoRequest(prompt:keyframes:…)`. The driver runs the export's FL2VA graphs:
  the tiled first-frame encoder (`videoEncoderTile`, `videoEncoderTail<W>x<H>`) for each keyframe's condition rows,
  noised to t = 0.999 from Kit's stream before the generated rows and held through every step, and Diffusers'
  conditioner presentation (a `"<Picture i>: "` label and a vision block per keyframe from `conditionerVision<W>x<H>`,
  M-RoPE positions, DeepStack after layers 0–2), with the `fl2va` recipe. Layout, presentation, the processor's
  pixels and tower tables, the M-RoPE gather and the noising are exact against Diffusers' own code
  (`Forge/h3/keyframe_fixture_h3.py`). A last frame alone is new: it is presented as one keyframe, anchored 5/3 before
  the end of the generated span. `check_h3.py --only condition` runs the keyframe-encoder check alone. `WeightBanks` now
  fills a weight input whose preferred strides pad its rows (the tower's MLP width is off Core AI's alignment) from
  the row-major extent, row by row; an extent must hold the input's logical bytes. Additive; the `AetherVideo` API
  baseline records it.
- **Our own Qwen3 language-model exports:** `Forge/qwen3/` exports any pinned Qwen3 dense decoder checkpoint in
  Hugging Face format (`Qwen3ForCausalLM`; the architecture comes from its config.json, and sliding windows, rope
  scaling, attention bias and MoE are refused) as a local `decoder-lm@1` bundle that `ChatSession` and `Generator` open
  from its folder through `InstalledBundle(ref:url:isStale:)`. The graph runs the running contract (in-place KV
  states, last-row logits, prefill chunks up to 512 tokens, up to 32,768 positions) with the 0.0.x Qwen3 recipe:
  float16 weights, KV, residual stream and head, float32 attention, norm statistics and rotary angles. The first pin
  is `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated@c9bd464` (Apache-2.0; 8.06 GB bundle, exported in 26 s, peak
  RSS 27 GB). `selftest_qwen3.py` checks the graph against transformers' own Qwen3 on small random checkpoints (tied
  and untied heads; rel. L2 ≤ 7e-7 one-shot and over chunks). The gate (`DecoderLMExportGateTests`) compares greedy
  replies with the 0.0.7 export of the same checkpoint on nine synthetic chat turns (system + user and user only,
  `maxTokens` limits, prompts up to 719 tokens, several requests on one opened model, an open-only pre-warm): 7 of 9
  token-exact, and the other two fork where the 0.0.7 reference itself is tied in float16 (logit margins 0 and one
  ulp; the near-tie rule). Teacher-forced on the reference's replies, 541 of 543 positions take its token, and the
  two that don't are those ties. No kit API changes.
- **Vision-language models (`vlm-static-inject@1`)**, Qwen3-VL first (docs/VLM.md): `VLMStaticInjectDriver` loads
  a decoder and a grid-static vision tower into one `DecoderEngine`, which conforms to the new
  `ImageConditionedEngine` (`imageInput`, `boundImages`, `bindImages(_:prepared:)`). A user turn carries images
  (`ChatMessage.user(_:images:)` with `ChatImage`, rendered as content parts before its text),
  `ChatSession.respond(to:images:)` sends one, and `Generator.generate(prompt:images:parameters:)` takes a prompt
  already rendered with one placeholder per image. Several images per turn: their tokens are extension ids over
  slots packed in prompt order, and `ImagePromptPlan` writes the decoder's rotary inputs by upstream's rule (gated
  against 0.0.7's M-RoPE positions). Images are prepared on the CPU before the reply's single GPU grant; a later turn
  reuses the state past images it already holds and skips their encode, and never past an image that changed (two
  same-sized pictures render the same ids). `DecoderEngine` binds `binding: static` inputs and accepts
  `inputEncodings.tokens.extensionIds`. Manifest: `preprocess.image.resize` mode `smartResize` (`factor`,
  `minPixels`, `maxPixels`; `Resize.smartSize(width:height:)`), `graphs.<g>.kind`, `image.slots`,
  `ImageBlock.typedTokens()`, `DriverContract.vlmStaticInject`, `DriverID.vlmStaticInject` and `.visionTower`.
  `bindImages` holds the engine as its operation across the tower calls (a feed, decode, rewind or second bind
  meanwhile is refused). After a failed bind on a state that still holds earlier images' tokens, feeding, decoding and
  snapshots are refused until a rewind before them, a reset or a restore; the Generator resets on such a failure. A
  conversation's images stay in the `ChatSession` history and count against one context's limits (16 images and
  4,096 image tokens on the Qwen3-VL exports): past them every turn is refused until `clearHistory()`.
  `Forge/qwen3vl` exports any pinned Qwen3-VL checkpoint (`--pins`; 4B: `source-pins-4b-abliterated.json`) with packed
  image slots (4,096 slots, 16 images), fp16 image statics and an fp16 tied head (`--head-compute`; float32 logits).
  Adding the `smartResize` case to
  `Manifest.ImagePreprocess.Resize.Mode` breaks an exhaustive `switch` over it; the API baselines record the change.
  The FoundationModels bridge stays text-only for now.

### Changed

- **API freeze review (#180 findings 5 and 8, approved by the owner).** Renamed or hidden before 0.2.0; each was public
  only in the 0.2.0 alphas, so 0.1.0 is unaffected:
  - `Krea2Pipeline.residentBytes` is now `plannedResidentBytes`: it is the last generation's plan, and the live
    reading is a different figure. (`ResidentAdmission.residentBytes`, frozen in 0.1.0, keeps its name.)
  - `SharedGPULease.legacyWrapperLock` is no longer public. `hasWaiters(directory:)` checks the lab wrapper's lock
    itself; `hasWaiters(directory:legacyWrapper:)` stays for another lock, or `nil` for none, with no default.
- **API freeze review (#180 finding 7, approved by the owner): the VLM plumbing is mostly package-only.** All of it
  was public only in the 0.2.0 alphas.
  - Now `package`: `PhasedGraphBackend` and `GraphFunctionSet`; the 9-argument `DecoderEngine.init(step:prefill:graph:
    vision:visionGraph:image:preprocess:device:options:)` (hosts load a bundle); `ImagePromptPlan.init(rendered:images:
    spec:)`, `placeholderTokens`, `imagePositions(slots:)`, `ropeShifts(entries:)`, `reusableLimit(bound:)`,
    `unusedShiftStart` and `Image.shiftAmount`; `ImageInputSpec.prepare(_:)`; `PreparedPromptImage`'s `init`,
    `values`, `rowLength`, `merge` and `geometry`.
  - `DecoderEngine.imageEncodes` is `@_spi(Diagnostics)`.
  - Still public, so a host can stand in its own engine for the describer (Atelier's tests do):
    `ImageConditionedEngine`, `ImageInputSpec` (and its init), `ImagePromptPlan` with `tokens`, `images`, `idBase` and
    `ImagePromptPlan.Image`, and `PreparedPromptImage` with its grid, `digest` and `tokens`.
  - Those that stay are renamed to the Swift-side convention: `gridH`/`gridW` → `gridHeight`/`gridWidth`
    (`PreparedPromptImage`, `ImagePromptPlan.Image`) and `padId`/`openId`/`closeId` → `padID`/`openID`/`closeID`
    (`ImageInputSpec`, `ImagePromptPlan.padID`). `idBase` keeps its name, which already follows the convention.
    Manifest `Codable` fields (`image.tokens.padId`, …) keep their JSON spelling.

### Fixed

- **Bundle verification in bounded memory.** `InstalledBundle.verify(progress:)` and the verified bundle status read
  each file in 8 MiB chunks, but on Apple platforms every `FileHandle` read is an autoreleased buffer, and the loop
  kept them all until it returned. On a 94 GB local export the process reached 13.5 GB resident and was killed before
  it finished. Each read and its hashing now run in their own autorelease pool, and memory stays flat (a 256 MiB file
  grew the footprint by 257 MiB before, under 64 MiB now; `FileChunksTests`).

- **Admission counts each shape specialization a pipeline keeps.** Graphs loaded without the reshape hint keep one
  specialization per shape (Krea 2's denoisers per prompt row count: about 27 GB each for the realism export, measured
  on a 64 GB Mac). The ledger kept one footprint per phase whatever the number of shapes behind it, so a second shape
  was planned at one shape's size and ran past the budget into swap. After that, every plan at that canvas counted
  both shapes and was refused even with one. Footprints are now keyed by the specializations held
  (`FootprintLedger.Key.specializations`, `estimate(_:)` for counts not measured, a full extra copy each when only one
  is). A new shape that doesn't fit beside the others with every graph resident releases the graphs and reloads for it
  alone; only a shape that doesn't fit alone is refused.
- **Working memory that outlives its phase no longer pushes resident graphs to per-phase release.** A run's working
  memory can stay allocated after its phase ends and come back during a later phase (Krea 2's decoder at 1024²: about
  11.5 GB of activations for 52 MB of weights, returned while the next prompt's text encoder loaded). The all-resident
  accounting counted it as held, added it again on the next run although nothing had loaded, and measured nothing
  for the encoder that loaded while it came back. On a 64 GB Mac the phases' sum then exceeded the budget by 54 MB, and
  the plan fell to per phase and never returned: every render loaded every graph again. Now:
  - A phase occurrence that loads nothing at the same specialization count keeps the phase's held bytes; its rise
    above what the residency already holds is its transient.
  - Memory that comes back during a phase is taken off the held bytes of the phases before it (the one that ran last
    first), and a load is measured from the lowest level its phase reached.
  - Admission records the peak an all-resident generation actually reached, per profile, specialization count and
    the graphs resident (an export's graphs a request never uses aren't loaded). When the phases' sum doesn't fit but
    such a measured peak does, the graphs stay resident, or a per-phase residency returns to all resident. A
    measurement counts only if it held at least the graphs resident now and a graph of every phase the plan counts (an
    encode alone doesn't).
  No public API changes.
- **Krea 2 metrics and the pressure gate see admission's release.** `loadedRoles` and the `PipelineOptions.pressure`
  check now read the resident graphs after admission plans. When it released them, the generation reports, and gates
  on, loading them all.

## 0.1.0 — 2026-11-07

A from-scratch re-architecture of AetherKit on Apple's Core AI. 0.0.x continues at the `0.0.7` tag and on
`legacy/0.0.x`.

### Breaking

- **Platform floor:** macOS 27 / iOS 27 (0.0.x linked at macOS 14 / iOS 17 and reported `unsupportedOS` below 27).
- **Products renamed or replaced:** AetherCore, AetherRuntime, AetherText, AetherFoundationModels, AetherHost,
  AetherImage, AetherSpeech, AetherVideo, AetherVision, AetherEmbeddings, and the `Aether` umbrella, which re-exports
  the task modules (AetherDecide is a placeholder until M5). The 0.0.x frozen APIs (the `AetherLanguage` facade,
  `AetherAssets` preparation and readiness, `@AetherGenerable`) are not carried over; see the README's migration notes.
- **Model bundles** are described by the `aether.json` manifest (docs/MANIFEST.md, schema v1). 0.0.x prototype
  manifests are rejected (`unknownDriver`).
- **`from: "0.0.7"` resolves to 0.1.0.** Pin `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")` to stay on 0.0.x.

### New

- **Catalog and store** (AetherCore): a compiled-in catalog pinned to Hugging Face revisions with per-file SHA-256
  (no runtime catalog fetch); manifest schema v1 with a zoo-bundle adapter and a reviewed override table. A bundle from
  `ModelStore` holds its reader lease while a model reads it (`InstalledBundle.readLease()`), so `ModelStore.remove`
  refuses with `inUse` meanwhile.
- **Runtime** (AetherRuntime):
  - manifest-driven Core AI graphs, a tensor arena, bindings, snapshot/restore state and a driver registry;
  - `DecoderEngine`: memory-sized KV (covering growth copies and hybrid snapshots), chunked prefill (≤ 512 tokens),
    O(1) rewind or hybrid snapshots, and token-exact pipelined greedy decode (about 500 tok/s on qwen3-0.6b, M5 Max)
    guarded by a load-time canary;
  - **`LanguageEngine`** (B1, docs/BACKENDS.md): the token-level seam the text stack runs on, an `Actor` protocol with
    an opaque `EngineSnapshot`. `DecoderEngine` conforms; `Generator.engine` is `any LanguageEngine`, so another
    backend's engine can run under the Generator, `ChatSession` and `AetherLanguageModel`. Core AI specifics
    (`canary`, `decodeMode`, `sizing`) are on `DecoderEngine` (`generator.engine as? DecoderEngine`);
    `ChatSession.generator(engine:bundle:manifest:gpu:)` wraps an engine you load yourself;
  - the generative foundation: seeded noise, flow-matching and epsilon schedules, `GeneratedImage` /
    `GeneratedAudio`, and residency planning from measured phase footprints (`GraphResidency`, `FootprintLedger`,
    `ResidencyPlan`, and a `ReleaseLog` that holds a per-phase phase until earlier releases have returned);
  - device chains for block-by-block models: `GraphChain` (each call's output feeds the next on the GPU),
    `WeightBanks` (per-block weights streamed from bundle files into two banks), `ShapeProfile`;
  - `SerialGate`: the one cancellation-aware, optionally bounded FIFO every request queue uses (`Generator`,
    `ChatSession`, the image, speech, video and stateless pipelines, `GraphChain`). A request cancelled while it waits
    leaves the queue at once;
  - image input for the vision and embedding drivers: `ImageInput` (a `CGImage` or a file, ImageIO decode, EXIF
    orientation, alpha policy, a `maximumPixels` limit checked before decoding), `ImagePreprocessor` and
    `PreparedImage` (the typed `preprocess.image` block: `stretch`, `letterbox`, `centerCrop`, `fitLongSide`; `vImage`
    or the bit-exact `pillowBicubic`, `pillowBilinear`, `torchUint8Bicubic` and `torchBilinear` filters; `NCHW`,
    `NVCHW` or patch layouts; an `ImageGeometry` to map results back to the source);
  - `Embedding` (`vector`, `dimension`, `dot`, `norm`, `normalized`, cosine `similarity(_:)`, Matryoshka
    `truncated(to:)`) and `LabelScore`, shared by AetherVision and AetherEmbeddings; `CLIPTokenizer`, whose added
    tokens and pad come from the tokenizer's files.
- **Text** (AetherText): the Generator (prefix reuse, stop and tag parsing, a seeded sampler, JSON-schema structured
  output, Hermes tool calls) and `ChatSession`.
- **FoundationModels** (AetherFoundationModels): `AetherLanguageModel` behind `LanguageModelSession`, with
  `@Generable` guided generation and tools.
- **An enforced GPU lock with priorities** (AetherHost):
  - `GPUAdmission.shared` is the process's default admission: every image, speech, video, vision and embedding
    pipeline and the text `Generator` (so `ChatSession` and `AetherLanguageModel`) admit through it unless given
    another (`PipelineOptions.gpu`, `Generator(gpu:)`, `ChatSession.load(gpu:)`), one grant per generation, clip,
    utterance request, chat reply or vision/embedding request. On unsandboxed macOS it also holds the machine's
    `SharedGPULease` and accepts a lease handed off through `AETHER_GPU_LEASE_HOLDER`; in sandboxed apps and on iOS it
    is in-process only. Tests opt out with `@_spi(Testing) GPUAdmissionOwner.unadmitted`.
  - `GPUPriority` (`.low`, `.medium`, `.high`) through an optional `GPUJob` on each request (`ImageRequest.job`,
    `SpeechRequest.job`, `VideoRequest.job`, `GenerationParameters.job`, and the vision and embedding requests; absent:
    `.medium`). `.high` goes ahead of every waiting request, `.low` waits until nothing else is waiting, FIFO within a
    level; running work is never preempted. Setting `job.priority` moves a waiting request; it has no effect once the
    request runs. There is no queue-management API; cancelling the task leaves the queue.
  - `PriorityInferenceQueue` replaces `FairInferenceQueue` (round-robin across owners is gone; the owner is a label for
    the per-owner limit). Pipelines and the Generator take the GPU before their own turn, so priority also orders one
    pipeline's requests.
- **Pipelines** share one `PipelineOptions` (AetherHost; `ImagePipelineOptions`, `KokoroSpeechSession.Options` and
  `VideoPipelineOptions` are its aliases): graph cache, memory budget, an optional shared `ResourceCoordinator`, GPU
  admission and a footprint ledger. Each driver's `load` gives a model value (`LatentDiffusionModel`,
  `Flux2KleinModel`, `Krea2Model`, `KokoroModel`, `MiniMaxH3Model`, `DepthEstimatorModel`, `ObjectDetectorModel`,
  `ImageTextEncoderModel`, `TextEmbedderModel`) with settable options, from which the pipeline is made. The stateless
  drivers share `StatelessPipeline` (static graphs checked against the driver's contract, residency and memory
  admission, cancellation between graph calls).
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
- **Vision** (AetherVision), on macOS and iOS, loaded through `Registry.vision`; each takes an `ImageInput` or a
  `prepare(_:)`d `PreparedImage` (real time: prepare the next frame while the last runs):
  - `DepthEstimator` (`depth-estimator@1`, Depth Anything 3 small): `estimate(_:)` → `DepthMap` (values, optional
    confidence, polarity, geometry), resized back to the source by default (`DepthRequest.resizeBack`).
  - `ObjectDetector`: YOLOX-S (`detector-dense@1`, upstream's NMS) and RF-DETR nano and seg-nano (`detector-detr@1`,
    sigmoid top-k, no NMS). `detect(_:_:)` → `[Detection]` (label, score, box normalized in the source, optional
    compact `DetectionMask` at the source's resolution), with a `DetectionRequest` (`scoreThreshold`,
    `maxDetections`) and `DetectionLabels` (`coco-91` built in).
  - `ImageTextEncoder` (`clip-joint@1`, OpenAI CLIP ViT-B/32): `embed(image:)`, `embed(texts:)` (in the text graph's
    batch buckets), `classify(_:labels:template:)` → `[LabelScore]` (template `"a photo of a {}."` by default),
    `scores(image:labels:texts:)`.
- **Embeddings** (AetherEmbeddings), on macOS and iOS, loaded through `Registry.embeddings`: `TextEmbedder`
  (`text-embedder@1`, EmbeddingGemma 300M): `embed(_:role:_:)` for one text or a batch, as a `.query` or `.document`
  (`EmbeddingRole`), with the manifest's prompts, padding and truncation (`tokenizer.text`); `tokenize(_:role:)` then
  `embed(_: [TokenizedText])` splits the CPU and GPU halves.
- **Forge exporters** (`Forge/`): reproducible, pinned exports, each with a CPU reference and a GPU gate:
  - for the kit's drivers: SD-Turbo, FLUX.2 [klein], Krea 2 (vanilla and realism, with the adapter merged before
    export), Kokoro, MiniMax H3, Depth Anything 3 small, YOLOX-S, RF-DETR nano and seg-nano, CLIP ViT-B/32 and
    EmbeddingGemma 300M; `Forge/ios_manifest.py` derives a published export's iOS manifest;
  - **Forge tooling only** (no kit driver in 0.1.0, not in the catalog): Qwen3-VL 2B (`Forge/qwen3vl/`),
    GLiNER2.5-Decide (`Forge/gliner/`), ColModernVBERT (`Forge/colmodernvbert/`), Shieldstral 1.0 3B
    (`Forge/shieldstral/`) and AdcSR ×4 (`Forge/adcsr/`, with its OpenRAIL++-M deployer notice). See "Models" and
    "After 0.1.0".
- **Public-API freeze** (`scripts/check-api.sh`, in `scripts/ci.sh`): every module with public API (AetherCore,
  AetherRuntime, AetherHost, AetherText, AetherFoundationModels, AetherImage, AetherSpeech, AetherVideo, AetherVision,
  AetherEmbeddings) is compared with the committed `api/` baseline, enforced from 0.1.0: a breaking change, or a
  compatible one not recorded with `--update`, fails CI (docs/RELEASE.md).
- **Tooling:** `scripts/gpu-lease [--timeout S] [--quiet] -- <command>` runs a command while holding the cross-process
  GPU lease (for Forge checks and other GPU work outside the kit; `AETHER_GATES=1 scripts/ci.sh` runs its gate step
  under it). `scripts/ci.sh` also builds in release with warnings as errors, and `scripts/ci-ios.sh` (CI's iOS job;
  `AETHER_CI_IOS=1 scripts/ci.sh` locally) builds every library for iOS.
- **Device qualification** (`Tools/DeviceGate`, docs/DEVICE.md): SD-Turbo, Kokoro and the six vision and embedding
  models pass their gates on a physical iPhone.

### Models

| Catalog id | Platforms | Gate (against the upstream reference) |
| --- | --- | --- |
| `sd-turbo` | macOS (float32), iOS (mixed precision: fp32 text encoder, fp16 U-Net and VAE) | from the seed vs CPU Diffusers; also on an iPhone |
| `flux2-klein-4b` | macOS | per graph vs CPU Diffusers; image from the reference's rows SSIM 0.991 |
| `krea-2-turbo`, `krea-2-turbo-realism` | macOS | from the seed vs CPU Diffusers: SSIM 0.999 / 0.988 |
| `kokoro-82m` | macOS, iOS (512-frame cap) | vs unmodified upstream on CPU: frames exact, rel. L2 ≤ 0.032; also on an iPhone |
| MiniMax H3 (not in the catalog: a locally exported bundle) | macOS | per graph vs the original modules (≤ 0.0095 rel. L2); the driver's first steps bit-identical to the graph reference; phase peaks within the 18 GiB bar |
| `depth-anything-3-small` | macOS, iOS | vs CPU float32 upstream on the same tensor: rel. L2 ≤ 0.0011 (bar 0.02); end to end from Pillow-identical decodes ≤ 0.00074; also on an iPhone |
| `yolox-s` | macOS, iOS | the same detections on all six cases: IoU ≥ 0.99999, scores within 2e-6 (bar IoU 0.95, 0.01); also on an iPhone |
| `rf-detr-nano`, `rf-detr-seg-nano` | macOS, iOS | the same detections on all six cases per variant: IoU ≥ 0.99999, scores within 4.4e-6, mask IoU 1.0 (bar 0.95, 0.01, mask 0.95); also on an iPhone |
| `clip-vit-b32` | macOS, iOS | vs transformers' `CLIPModel` on CPU in float32: image cosine ≥ 0.9999983, text ≥ 0.9999993, tokenizer 17/17 id-exact, zero-shot argmax 7/7; also on an iPhone |
| `embeddinggemma-300m` | macOS, iOS | token ids exact (7/7), cosine ≥ 0.9999999 vs CPU float32 upstream (bar 0.999); also on an iPhone |
| the language models (Qwen3, Qwen3.5, Gemma 3/4, MiniCPM, LFM2.5, Granite, Nemotron, Mistral, …) | macOS; iOS for most (the catalog lists each entry's variants) | greedy output token-exact against a same-device reference |

The catalog's bundles are hosted under `aether-models` on Hugging Face; each carries its license and NOTICE. The Krea
realism repo is gated (mature), and the EmbeddingGemma repo is gated with automatic approval (Gemma terms; install
with a token). MiniMax H3's license grants no rights in the EU, UK, South Korea or USA, so its bundle is not hosted:
export it with `Forge/h3/` where the license allows and load it from disk. The licenses differ per model; the README's
table lists their terms.

**Forge-only exporters.** These export, check against their CPU reference and gate on the GPU in Forge, but 0.1.0 has
no driver to run them and no catalog entry:

| Exporter | Model | For the driver (after 0.1.0) | License |
| --- | --- | --- | --- |
| `Forge/qwen3vl/` | Qwen3-VL 2B Instruct (vision tower + static-inject decoder) | `vlm-static-inject@1` | Apache-2.0 |
| `Forge/gliner/` | GLiNER2.5-Decide (s256/s512 buckets) | `gliner-classifier@1` | Apache-2.0 |
| `Forge/colmodernvbert/` | ColModernVBERT (query and document towers) | `late-interaction-retriever@1` | MIT |
| `Forge/shieldstral/` | Shieldstral 1.0 3B (yes/no classifier) | `yes-no-classifier@1` | Apache-2.0 |
| `Forge/adcsr/` | AdcSR ×4 super-resolution (tiled) | `image-to-image-tiled@1` | CreativeML Open RAIL++-M (weights), Apache-2.0 (code); deployer notice |

### Carried over from 0.0.x (re-implemented, validated against 0.0.x's tests and fixtures)

- **Store:** staged SHA-256 installs and resume, reader leases, stage recovery, cross-process install locks, a
  free-space check, token and redirect safety, iOS background downloads.
- **Host** (AetherHost): the GPU lease (Atelier-compatible) and admission, model lifecycle (idle unload,
  memory-pressure release, iOS easing) and `ResourceCoordinator` reservations.
- **Text:** the sampler (matching Transformers' recorded processors), the structured-output constraint, the
  text-stream rules and the tokenizer oracle cases.
- **Generative recipes:** the seeded noise stream (bit-identical to 0.0.x), the schedule arithmetic, and the SD-Turbo,
  FLUX.2, Krea, Kokoro and MiniMax H3 host recipes.
- **Image input:** 0.0.x's `VisionRGBImage` decode (EXIF orientation, alpha) and its antialiased bicubic, as
  `ImageInput` and the `pillowBicubic` / `torchUint8Bicubic` filters.

### Known limitations

- FLUX.2, Krea and MiniMax H3 are macOS-only.
- Text-to-phoneme for Kokoro is the app's.
- The image, speech, video, vision and embedding pipelines don't take part in `ModelHost`'s idle and memory-pressure
  unload; an app unloads them itself (`unload()`).
- Generative outputs are not bit-identical across devices (fp16 arithmetic differs); gates compare against the
  reference on each device.
- One bundle per platform where precision differs (SD-Turbo on iOS). The catalog picks the device's variant; a
  manifest's `variant.platform` is not checked at load.
- **Decoder capacity** is sized at load from the model's load-time Metal delta (at least its files), but memory
  arrives with the first run (G0: about 3× the files held). Re-sizing after the canary needs the engine to lower its
  capacity limit after construction.
- **Kokoro** is admitted against the measurement for its frame cap's bucket; until a cap-bucket run (or `prewarm`)
  is measured, a smaller bucket's measurement stands in and under-counts.
- **The FoundationModels adapter** renders the prompt again for its context check.
- **Depth Anything 3** takes a 504 × 504 stretch (its reference's), not upstream's aspect-preserving cv2 resize: the
  depth matches the reference on the same input, but on non-square photos it isn't upstream's end-to-end result.
- **JPEG inputs:** ImageIO decodes JPEGs differently from Pillow's libjpeg (up to tens of levels), so detections and
  embeddings from a JPEG are close to, but not exactly, the Pillow-based references. The preprocessing is gated
  bit-exact through Pillow's decode, and end to end where both decoders give the same bytes (PNGs).

### After 0.1.0

Not in this release; roadmap in docs/M4.md and docs/BACKENDS.md:

- **VLMs:** the `vlm-static-inject@1` driver (Qwen3-VL first; its exporter is in Forge), then OCR (`vlm-embeds@1`).
- **The optional M4 drivers** for the Forge-only exporters: GLiNER (`gliner-classifier@1`), Shieldstral
  (`yes-no-classifier@1`), ColModernVBERT (`late-interaction-retriever@1`) and AdcSR (`image-to-image-tiled@1`), plus
  `CVPixelBuffer` input and `LiveVision`.
- **True DA3 parity:** upstream's cv2 preprocessing (long side 504, multiples of 14, aspect kept) and a non-square
  graph.
- **RF-DETR's split chain** (`backbone.features` through `source.from`).
- **Our own language-model exports:** Forge `decoder-lm` exporters for the catalog's language models, re-hosted under
  `aether-models` one at a time (today they are pinned public Core AI conversions).
- **MLX:** the B0 lab spike, then (if it is a go) `mlx-lm@1` in an `AetherMLX` sub-package (B2–B5).
- **Generative extras:** H3's FL2VA keyframes; FLUX.2's and Krea's reference images, and Krea's refine.
- **AetherDecide** (M5) and the one-call `Aether` ops (M6).

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
- **FLUX.2 [klein] 4B (G2)** in `AetherImage`:
  - the `flux2-klein@1` driver (`Flux2KleinPipeline`): the bundle's Qwen3 chat template, padded to 512 rows → text
    encoder → three denoiser segments chained by their carries, per step, on the host Euler update with the empirical
    shift → VAE decoder; macOS only in 0.1.0 (`unsupportedDevice` on iOS before anything loads);
  - the typed `flux2` manifest block (MANIFEST.md §5.7, now normative) and optional graph roles in driver contracts
    (the reference-image VAE encoder);
  - `ImagePipelineOptions` and `ImageRequest` shared by the image pipelines, with a `GPUMemoryProbe` (tests pass a
    fixed one); `TokenizerAdapter` can load without stop markers (`forGeneration: false`) for drivers that only
    encode prompts; `PromptRenderers.chatTemplate` reads a bundle's verified template;
  - a reproducible Forge exporter (`Forge/flux2/`): int8 per-channel weights on the text encoder and denoisers, both
    VAE graphs; its GPU gate against CPU Diffusers passes 31/31;
  - the Swift gate against CPU Diffusers from the seed: prompt ids exact, text real-row cosine ≥ 0.99921, the image
    from the reference's text rows SSIM 0.991 / PSNR 34.0 dB; the whole pipeline is recorded at SSIM 0.908 (0.0.x's
    bars; padding rows condition the denoiser, so the full path moves under any precision change).
- **Krea 2 Turbo, vanilla and realism (G3)** in `AetherImage`:
  - the `krea2@1` driver (`Krea2Pipeline`): the fixed Krea prompt, unpadded → text encoder (12 tapped layers, the
    34-row prefix dropped) → four denoiser segments chained by their carries, per step, on the host Euler update with
    the exponential shift → VAE decoder; macOS only in 0.1.0;
  - **`ContentFilter`**, required by the Krea 2 Community License (§4.2): `Krea2Driver.load` gives a `Krea2Model`, and
    a pipeline is only made with the app's filter (no default). It reviews the prompt before any work and the image
    before it is returned; a refusal is the new `AetherError.GenerationFailure.contentRefused(stage:reason:)`;
  - the typed `krea2` manifest block (MANIFEST.md §5.5–§5.6, now normative); a merged adapter's `scale` may be a
    number or a list;
  - the exporter writes the installed-revision layout (`.aether/aether.json`);
  - the Swift gate per variant against CPU Diffusers from the seed: rendered ids exact, text row cosine 0.999999, the
    whole pipeline SSIM 0.9989 / PSNR 49.9 dB (vanilla) and 0.988 / 39.6 dB (realism), repeats bit-identical;
  - the denoiser segments and VAE specialize without `expectFrequentReshapes` (with it their carries are non-finite);
  - image pipelines default to `MemoryBudget.imageGeneration`: three quarters of RAM on macOS (Krea plans about 34 GiB
    a generation), `platformDefault` on iOS.
- **H3 runtime prerequisites (G4 step 1)** (`AetherRuntime/Chain/`): `GraphChain` binds each call's output as the
  next call's input on the GPU (two slots, one stream, waits at the end or at sync points; byte-identical to the
  unchained path), `WeightBanks` streams per-block weights from bundle files into two banks with the sync rule
  enforced, `ShapeProfile` pins the shapes a measured footprint holds for, and `SerialGate` queues busy callers.
- **H3 exporter (G4 step 2a)** (`Forge/h3/`): every graph the 0.0.x prototype qualified, with the transformer block,
  the Qwen3-VL decoder layer, the text refiner and the vision tower taking their weights as inputs from per-block
  bundle files (MANIFEST.md §5.8, now normative, decoded and validated as `Manifest.GraphWeights`); the per-graph
  oracle gate `check_h3.py`; and a GPU gate running the export's own block graph through `GraphChain` and
  `WeightBanks` (4 blocks × 4096 rows: output within 0.0051 relative L2 of the CPU, chained bytes equal to the
  unchained path's). The real-weight export (94 GB, run outside the licence's Excluded Territories per the owner)
  passes `check_h3.py` at 960×544×124: every graph within 0.0095 relative L2 of the original modules (block 49 at
  0.0067). It is not publicly hosted and not in the catalog.
- **MiniMax H3 (G4)** in the new **`AetherVideo`** product: `minimax-h3@1` (`MiniMaxH3Pipeline`), text to 24 fps video
  with 32 kHz stereo audio at the qualified 960×544×124 profile, macOS only. The conditioner and the 50-block
  transformer chain on the GPU with their weights streamed from the bundle (`GraphChain`, `WeightBanks`); the host
  recipe is Diffusers' layout, row timesteps and rotary, bit-identical to the graph reference for the first two steps;
  a clip and its repeat bit-identical; phase peaks 4.2 / 9.7 / 1.4 GiB within the 18 GiB bar. The typed `h3` block
  (MANIFEST.md §5.10 normative); `TensorBuffer.readBytes`/`writeBytes`/`copyBytes` and
  `WeightBanks.load(block:into:)` for bf16 tensors between host and Metal arenas. The exporter's manifest fixes:
  `video.noise.order` is `sourceVideoThenAudio` (0.0.x's draw order), the pack's text input is named `text`, and no
  graph takes the reshape hint. FL2VA keyframes are deferred; the bundle isn't publicly hosted (license).
- **Kokoro-82M speech (G5)**, in the new **`AetherSpeech`** product:
  - the `kokoro@1` driver (`KokoroSpeechSession`): phonemes (`KokoroPhonemes`, Unicode scalars, unknown symbols
    refused) → durations → host rounding and alignment → the bucketed decoder with seeded harmonic noise →
    `GeneratedAudio`; multi-segment requests, frame caps per platform, prewarm, measured admission, cancellation;
  - text through an app-supplied `SpeechPhonemizer` (no G2P ships: misaki falls back to GPL espeak-ng);
  - the typed `kokoro` block (MANIFEST.md §5.9, normative) and a reproducible Forge exporter (`Forge/kokoro/`, float32,
    54 voices). The Swift gate against unmodified upstream on the CPU: frames exact, rel. L2 0.0076–0.032 (bars
    min(0.05, 4× the fp32-vs-fp64 floor)), log-mel ≤ 0.015, repeats bit-identical.
  - Both graphs run on the GPU: Swift's Core AI runtime traps loading the durations graph for the CPU on macOS 27.2.
- **Generative catalog:** `sd-turbo` (macOS and iOS) and `flux2-klein-4b` (macOS), from
  `aether-models/aether-kit-sd-turbo` and `aether-models/aether-kit-flux2-klein-4b` at pinned revisions, every file's
  SHA-256 recorded; `kokoro-82m` (macOS and iOS) from `aether-models/aether-kit-kokoro-82m`; `sd-turbo` on iOS takes
  the mixed-precision export from `aether-models/aether-kit-sd-turbo-ios`; `krea-2-turbo` and `krea-2-turbo-realism`
  (macOS) from `aether-models/krea-2-turbo-aether-kit` and the gated `aether-models/krea-2-turbo-realism-aether-kit`;
  catalog entries carry `mature` (the realism merge), and `scripts/gen-catalog` sends the Hugging Face token for gated
  repos.
- **On-device qualification** (`docs/DEVICE.md`, `Tools/DeviceGate`): SD-Turbo and Kokoro pass their gates on a
  physical iPhone (iPhone Duo, `h19p`, iOS 27.2). SD-Turbo's iOS bundle is mixed precision (text encoder fp32, U-Net and
  VAE fp16; the Forge exporter gained per-graph `--precision` options): 4 steps RMSE 0.012 on device (bar 0.03), 1.7 GiB
  peak. Kokoro runs at its 512-frame iOS cap (longer utterances refused before decoding).
- **The `LanguageEngine` seam (B1, docs/BACKENDS.md):** for code that tracked `main` before it, the migration:
  `generator.engine.description.maxCapacity` → `generator.engine.maxCapacity` (likewise `vocab`, `hasRecurrentState`);
  `DecoderEngine.snapshot()` returns an `EngineSnapshot` and `restore(_:)` takes one (`DecoderEngine.Snapshot` is no
  longer public; a snapshot of another engine type is refused with `unsupportedOperation`); `ChatSession.load`,
  `ChatSession.generator(bundle:…)` and `AetherLanguageModel.load` keep their `DecoderEngine.Options` parameter. No
  behaviour change.
- **The GPU lock and priorities:** `GPUAdmission.shared` by default everywhere, `GPUPriority` through `GPUJob`, and
  `PriorityInferenceQueue` replacing the fair queue (see "New").
- **Vision foundation (M4a):** `ImageInput`, `ImagePreprocessor` and `PreparedImage` (above), with known answers from
  Pillow and torch (`Forge/vision/fixtures_preprocess.py`); typed `postprocess` (`depth`, `denseNms`, `detrSigmoid`,
  `embedding`, `gliner`, `yesNo`, `maxSim`, `superRes`; an enum of typed variants) and `labels` (`{set: "coco-91"}` or
  `{file}`) blocks, decoded and validated (MANIFEST.md §2.8); `Manifest.Preprocess.image` is
  `Manifest.ImagePreprocess`; `StatelessPipeline` (AetherHost); `AetherError.invalidInput` for an image that doesn't
  decode or lay out; `GraphResidency.validating(_:manifest:contract:)` replaces the image and speech drivers' copies.
- **Depth (M4b):** AetherVision's first driver, `depth-estimator@1` (`DriverContract.depthEstimator`), for Depth
  Anything 3 small from our Forge export (`Forge/depth/`): `DepthEstimator`, `DepthMap` and the `depth` postprocess
  (`resizeBack` resamples bilinearly through the geometry, never reading letterbox padding; polarity `largerIsFarther`
  is depth, `largerIsNearer` disparity). The bundle stretches with `pillowBicubic`, its reference's resize. Gate
  `DepthEstimatorGateTests`: rel L2 0.0002–0.0011 against the CPU float32 upstream reference (bar 0.02); Pillow's
  decode through `preprocess.image` gives the reference tensor bit for bit, and the SOH frames end to end, resized
  back, are gated at 0.02 (0.00074 and 0.00041). Catalog: `depth-anything-3-small`,
  `aether-models/aether-kit-depth-anything-3-small` at `4c1fae82`, Apache-2.0.
- **Text embedding (M4c):** AetherEmbeddings' first driver, `text-embedder@1` (`DriverContract.textEmbedder`), for
  EmbeddingGemma 300M from our Forge export (`Forge/embeddinggemma/`): `TextEmbedder`, `Embedding`, and
  `tokenizer.text` (MANIFEST.md §2.4) as the typed `Manifest.Tokenizer.Text` (`padId`, `padSide`, `padTo`,
  `truncation`, `prompts`, `maskRule`): the role's prompt plus the text, stripped, encoded with the tokenizer's special
  tokens, truncated keeping them, right-padded with a mask of real ids; the `embedding` postprocess (pooling in the
  graph; `normalized: host` L2-normalizes on the CPU). AetherEmbeddings depends on AetherText (its tokenizer adapter),
  as AetherImage and AetherVideo do. Gate `TextEmbedderGateTests`: token ids and masks exact on 7 cases (one
  truncated), cosine ≥ 0.9999999 against the CPU float32 upstream reference (bar 0.999), every query's best document
  as the reference's. Catalog: `embeddinggemma-300m`, `aether-models/aether-kit-embeddinggemma-300m` at `d5016965`,
  license `gemma`; the repo is gated with automatic approval, so installing needs a Hugging Face token.
- **CLIP (M4d):** `clip-joint@1` (`DriverContract.clipJoint`) for OpenAI CLIP ViT-B/32 from our Forge export
  (`Forge/clip/`): `ImageTextEncoder`, with `Embedding` (gaining `dot`, `norm`, `normalized`) and `LabelScore` in
  AetherRuntime so AetherVision and AetherEmbeddings share both, and `CLIPTokenizer` moved there from AetherImage
  (AetherVision must not depend on AetherImage). Its added tokens and pad come from the tokenizer's files instead of a
  hard-coded `!` → 0 (SD 2.x's still pads with `!`, 0; OpenAI CLIP's encodes `hello!` as 3306, 256);
  `init(directory:length:pad:)` takes `pad: Int32?`. Graph buckets on axis `batch` (`Manifest.Graph.bucket(_:)`,
  `Buckets.values`/`smallestFitting(_:)`, `GraphResidency.graph(_:bucket:)`); manifest fields
  `preprocess.image.resize.shortestSide`, `postprocess.embedding.dimension` and `logitScale`,
  `tokenizer.text.unusedBatchRows` and the mask rules `nonPad`, `throughFirstEos` and `none` beside `realTokens`. Gate
  `ImageTextEncoderGateTests` against transformers' `CLIPModel` in float32 on the CPU. Catalog: `clip-vit-b32`,
  `aether-models/aether-kit-clip-vit-b32`, MIT.
- **Dense detection (M4e):** `detector-dense@1` (`DriverContract.detectorDense`) for YOLOX-S from our Forge export
  (`Forge/yolox/`): `ObjectDetector` and the shared `Detection`, `DetectionRequest` and `DetectionLabels`; the
  `denseNms` postprocess (objectness × best class, threshold, greedy per-class or class-agnostic NMS, the top
  `maxDetections`, exactly upstream YOLOX's `postprocess`; boxes mapped back through the letterbox geometry).
  `letterbox` sizes are YOLOX's `int(side · min(H / h, W / w))` in Double (the 2443 × 424 demo strip letterboxes to
  639 × 111, as upstream does); `padValue` also takes a bare number. The bundle letterboxes with `pillowBilinear`. Gate
  `ObjectDetectorDenseGateTests`: the host NMS reproduces the reference's detections exactly from its preds; on the
  GPU, the same detections on all six cases (IoU ≥ 0.99999, scores within 2e-6; bar IoU 0.95, 0.01); end to end from
  `ImageInput` on the PNG cases at the same bars (IoU 1.0, scores within 2e-6). Catalog: `yolox-s`,
  `aether-models/aether-kit-yolox-s` at `e57fcca8`, Apache-2.0.
- **DETR detection and instance masks (M4f):** `detector-detr@1` (`DriverContract.detectorDetr`) for RF-DETR nano
  and seg-nano from our Forge export (`Forge/rfdetr/`), as a second head of `ObjectDetector`: the `detrSigmoid`
  postprocess (sigmoid scores, the best `maxDetections` labelled (query, class) pairs strictly above the threshold, ties
  by flat index, no NMS, exactly upstream RF-DETR's `PostProcess`; a typed `selection`, `topKPairs` by default; a
  `masks` block exactly when the graph binds `masks`), with each detection's query plane resampled bilinearly to the
  source and thresholded into a compact `DetectionMask` (bounds and one bit per pixel; `foreground`, `alpha()`, `area`,
  `intersectionOverUnion(_:)`). The bundles stretch with `torchBilinear`. Gate `ObjectDetectorDetrGateTests` (both
  variants): the head reproduces upstream's detections from the reference's raw outputs; on the GPU, the same
  detections on all six cases per variant (IoU ≥ 0.99999, scores within 4.4e-6, mask IoU 1.0; bar IoU 0.95, 0.01, mask
  0.95). Catalog: `rf-detr-nano` and `rf-detr-seg-nano`, `aether-models/aether-kit-rf-detr-nano` and `-seg-nano` at
  `49e797b1` and `b7408d29`, Apache-2.0.
- **Parity resize filters:** `pillowBilinear` (Pillow's `Image.resize(BILINEAR)`, bit-exact on 8-bit output) and
  `torchBilinear` (torch's float32 `interpolate(bilinear, antialias=False)`, bit-exact in float32; `stretch` with
  `NCHW`/`NVCHW` only). The DA3, YOLOX-S and RF-DETR bundles were re-exported to name their reference's filter (graphs
  unchanged) and re-pinned; each gate checks that Pillow's decoded pixels (`decoded.png`) through `preprocess.image`
  give the reference tensor bit for bit.
- **iOS variants for the M4 models:** `depth-anything-3-small`, `embeddinggemma-300m`, `clip-vit-b32`, `yolox-s`,
  `rf-detr-nano` and `rf-detr-seg-nano` resolve on iOS with the same drivers. They passed the Mac gate bars on an
  iPhone Duo (`iPhone19,4`, `h19p`; docs/DEVICE.md). Each `ios` variant is the macOS export's files unchanged (JIT
  `.aimodel` packages, nothing per architecture) plus a manifest with `variant.platform: ios`, at `.aether/aether.json`
  on the repo's `ios` branch, which the variant's main asset pins: DA3 `43390fd4`, EmbeddingGemma `462d7f90`, CLIP
  `be0c479c`, YOLOX-S `1ff82066`, RF-DETR nano `383f4c4e`, seg-nano `1c2aba91`. `Forge/ios_manifest.py` derives the
  manifest from the published export. Covered by `CatalogTests` and the hermetic iOS load tests
  (`VisionIOSVariantTests`, `TextEmbedderIOSVariantTests`).
- **Forge-only exporters (M4g, M4i–M4l groundwork):** Qwen3-VL 2B, GLiNER2.5-Decide, ColModernVBERT, Shieldstral 1.0
  3B and AdcSR ×4, each with its pinned sources, CPU reference and GPU gate; their drivers come after 0.1.0.
- **First pre-release review fixes:**
  - `ResidentAdmission` (AetherHost) keeps a pipeline's resident graphs reserved on a shared coordinator and releases
    them on refusal; measured footprints are used for manifests without residency (Kokoro) and after late releases;
    canvases need `maxImageTokens`; drivers load model values with settable options; `KokoroVoice.Language` is open;
    `GeneratedAudio.sampleCount` (was `frameCount`).
  - Store references are validated and contained before any file operation (`AetherError.invalidReference`); an
    install is reused only for the same installation; tokenizer paths are contained and stop ids bounded; released
    bindings no longer validate; structured replies must complete their schema; exact integer schema bounds, and
    constraints beside `$ref`/`anyOf` refused; ModelHost leaks no pin on a type mismatch and cancelled callers load
    nothing; the GPU-admission timeout race; cancellation between prefill chunks; canary off means depth one; a top-k
    crash at tiny temperatures; cumulative resident footprints keyed by revision and device; float32 logits reads 377
    → 6 µs; bounded constraint cache; download backpressure; the device gate script fails on failed checks; `Aether`
    re-exports `AetherSpeech`.
- **Second pre-release review fixes:**
  - **Store, host and validation:** download flow control suspends and resumes the data task under the bookkeeping's
    lock (no stalled download); `ModelRef` validates file sizes and hashes and totals without trapping;
    `ImageInput.maximumPixels` (default 64 Mi pixels) refuses oversized images before decoding; `FlowMatchSchedule`
    refuses a non-normal shift; a cancelled `ModelHost` caller stops waiting at once and an unwanted load is dropped;
    `MemoryPressureMonitor` counts a repeated warning; footprint-ledger keys include the export's manifest digest (H3's
    seeded measurements apply only to the qualified export); `TensorBuffer.copyBytes` within one arena goes through
    the arena.
  - **Memory and pipelines:** per-phase residency frees each phase's graphs at its end (SD-Turbo, FLUX.2, Krea); a
    cancellation during the last graph call is honoured; `ResidentAdmission.finish` releases the residency when the
    measured bytes can't stay reserved.
  - **Structured output:** optional properties are omitted rather than allowed to be `null`, and constraints the
    grammar can't hold (a required name with no property, a `const` outside its `enum`, a keyword on the wrong type)
    are refused at compile time.
  - **Decoder and chat:** KV sizing covers the strategy's peak (growth copies, hybrid snapshots, every logits buffer;
    `Options.reservedSnapshots`, `Generator.maxSnapshots`), allocating upfront when growing on demand would fit fewer
    tokens (`KVSizing.initialAllocation(capacity:budget:)`); an explicit capacity limits a static sequence axis; one
    invalid-logit policy for every host path (`Logits.argmax` throws `invalidLogits` on NaN or an all `-inf` row); GPU
    completions report failure; a reply running across `ChatSession.clearHistory()` doesn't put its turn back.
  - **Reader leases in normal loading:** a bundle from `ModelStore` remembers its store and every pipeline built on
    `ResidentAdmission` (and `DecoderEngine`) holds its reader lease from creation until `unload()` or deinit; the next
    request after `unload()` takes it again (refusing with `inUse` or `unknownModel` before loading).
    `InstalledBundle` equality ignores the store. `ResidentAdmission.init` and `PipelineOptions.admission` take
    `bundle:` and throw; `StatelessPipeline.init(manifest:…)` takes `bundle:`.
  - **Per-phase overlap is enforced:** a `ReleaseLog` (one per `ResidentAdmission`) records each release, and a
    per-phase phase waits until no more than the plan's overlap (`ResidencyPlan.overlap`) was released within the
    window (6 s), so short back-to-back phases (SD-Turbo's) can't overlap more memory than planned.
    `GraphResidency.beginPhase` is `async throws`; `GraphResidency.init` takes `releases:`, and
    `ResidentAdmission.Settings` takes `releaseWindow:` and `releaseClock:`.
  - **Diffusion host steps reuse their buffers:** SD-Turbo writes `x / inputScale` straight into the U-Net's input,
    and SD-Turbo, FLUX.2 and Krea apply the Euler update in one validating pass into a reused scratch array
    (`EpsilonEulerSchedule.inputScale(step:)`, `advance(step:latents:epsilon:into:)` /
    `advance(step:latents:epsilon:scratch:)`, `velocity:` on `FlowMatchSchedule`). Images are bit-identical; host work
    per step (release build, `HostStepBenchmarkTests`): SD-Turbo 512² 822 → 76 µs, FLUX.2/Krea 1024² 2.34 →
    2.02 ms.
  - **Resident footprints no longer accumulate a phantom:** a phase's held bytes and peak are capped by the level
    above where the process stood before any graph loaded, less the other phases' held bytes (Depth Anything 3's gate:
    1137 / 1122 MiB peak / held → 374 / 359 MiB).
  - **One serialization primitive:** `SerialGate` is `Sendable`, cancellation-aware and optionally bounded, and every
    request queue uses it instead of its own busy flags. `withTurn(_:)` throws (cancellation, or
    `AetherError.AdmissionFailure.queueFull(limit:)` past `SerialGate(limit:)`); `withTurnIgnoringCancellation(_:)` is
    what `unload()` uses. GPU priority stays in `GPUAdmission`: pipelines take the GPU first, then their own turn.
  - **CI:** a release build with warnings as errors in `scripts/ci.sh`, and the iOS compile check (`scripts/ci-ios.sh`).
