# Kokoro speech synthesis — 2026-10-26

Status: experimental, 0.0.2 checkpoint. Qualified on one Mac and one iPhone (see [iPhone](#iphone)). AetherSpeech turns validated Kokoro phoneme strings into 24 kHz mono float32 PCM using two exported Core AI graphs. Native parity with the upstream PyTorch model, timing, cancellation and unload passed on one Mac (M5 Max, macOS 27.0.1), recorded in the [qualification record](evidence/kokoro-qualification-2026-10-26.json). Not yet covered: iPhone parity at bucket 512 and on other iPhone models, quantized or fp16 exports, sub-utterance streaming, and listening review.

## Source and license

| Item | Value |
| --- | --- |
| Model | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) v1.0, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, not gated |
| Weights | `kokoro-v1_0.pth`, SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`; this matches the hash published on the model card |
| License | **Apache-2.0**: model card front matter, card text and Hub metadata (`license:apache-2.0`), confirmed 2026-10-26 |
| Training-data attribution | The card lists CC BY 3.0 (Koniwa `tnc`) and CC BY 4.0 (SIWIS) audio in the training set. Keep that attribution wherever the weights are redistributed |
| Base model | `yl4579/StyleTTS2-LJSpeech` (StyleTTS 2 with an iSTFTNet vocoder) |
| Reference code | `kokoro` 0.9.4 wheel (Apache-2.0, SHA-256 `a129dc63…5789efd`), used read-only by `Forge/kokoro` for export and references |

AetherKit ships no Kokoro weights. Users export a bundle themselves (see below), and those files keep their upstream license. (Kit's only published weights are the Qwen3-0.6B sample bundle, [sample app](sample-app.md).)

## Phonemization: the licensing decision

The upstream pipeline turns text into phonemes with `misaki`, which is Apache-2.0. For words missing from its dictionaries, misaki can fall back to **espeak-ng (GPL-3.0)** through `phonemizer-fork` and `espeakng-loader`. A GPL dependency is incompatible with this Apache-2.0 SDK.

**Decision (v1): AetherSpeech accepts phonemes, not text, and contains no grapheme-to-phoneme (G2P) code.**

- `KokoroPhonemes` validates a string against the bundle's vocabulary: 114 single Unicode scalars, token IDs up to 177. Upstream silently drops unknown symbols; AetherSpeech refuses them so that a G2P mismatch shows up immediately. The string is not normalized: Kokoro uses `a` + U+0303 for nasal vowels, but precomposed `ç` (U+00E7).
- The app supplies G2P through the `SpeechPhonemizer` protocol (`phonemize(_:language:) -> [String]`). It returns one or more utterances of at most 510 scalars each, ideally split at sentence or clause boundaries. `session.synthesize(text:phonemizer:configuration:)` wires it in.
- The export and reference harness import only `kokoro.model`, `kokoro.istftnet` and `kokoro.modules`. They do not run kokoro's package `__init__`, which imports misaki, and they refuse to run if a real kokoro package has already been imported. No code in this repository imports or links misaki, phonemizer or espeak-ng. Whether those packages happen to be installed in a given Forge environment is outside the repository's control.

License-clean G2P routes for apps:

1. **Dictionary lookup plus a permissive fallback.** CMUdict is BSD-licensed. Map ARPAbet to Kokoro's symbols (diphthongs `A I O W Y`, stress marks `ˈ ˌ`, and so on), and handle out-of-vocabulary words with a permissively licensed or self-trained seq2seq model. Do not ship espeak-ng.
- misaki's own `us_gold`/`us_silver`/`gb_*` lexicons ship in its Apache-2.0 wheel, but the wheel does not document where they came from. **We do not recommend bundling them until their provenance has been checked.** Upstream lists replacing the espeak fallback with seq2seq models trained on these dictionaries as future work.
2. **Server-side or offline G2P.** Precompute phonemes for fixed prompts and ship them as data.
3. **Hand-authored phonemes** for short, fixed UI strings.

Language follows the voice prefix (`a` American English, `b` British English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese, `p` Brazilian Portuguese, `z` Mandarin). Kokoro is language-blind past the phonemes, so the app's G2P decides the language.

## Graphs and host steps

`Forge/kokoro/export_kokoro.py` splits upstream `KModel.forward_with_tokens` at its data-dependent point:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `durations` (`durations.aimodel`, 74 MiB) | `input_ids` int32 [1,T] (boundary 0, phonemes, 0); `style` [1,256] | `duration` [1,T] (per-token sums before speed); `prosody` [1,T,640]; `text` [1,512,T] |
| host | | frames = max(1, round-half-even(sum / speed)); gather token columns into F frames |
| `decode` (`decoder.aimodel`, 235 MiB) | `prosody` [1,640,F]; `text` [1,512,F]; `style` [1,256]; `noise` [1,F,600,9] standard normal | `audio` [1,600F] float32, 24 kHz |

Both graphs are float32 with dynamic T (3…512) and F (2…4096, about 102 s). The bundle also carries all 54 v1.0 voices as `voices/<id>.f32` (510 × 256 float32). As upstream does, the style row is chosen by phoneme count (`pack[len(phonemes) − 1]`).

Primitives that Core AI does not lower are replaced by exact or numerically equivalent forms. `Forge/kokoro/kokoro_graphs.py` documents each one:

- **LSTMs** (five over tokens, one over frames) become a `while_loop` recurrence that runs both directions as a batch of 2. Upstream's pack/unpack is the identity for batch 1 without padding.
- **STFT/iSTFT**: `torch.stft`/`istft` use complex numbers, so they become real convolutions. The forward pass uses reflect padding with exact twiddles (zeros and ones at quarter turns, so `atan2` never flips between +π and −π on DC/Nyquist). The inverse uses irfft weighting and divides by the overlap-added squared-window envelope. Upstream's `CustomSTFT` is only an approximation and is not used.
- **Upsampling**: nearest ×2/×300 becomes reshape/expand. Linear ×300 reproduces PyTorch's CPU sampling coordinates. `x % 1` becomes `x − floor(x)`. `rsqrt(tensor(2))` (an int tensor) becomes a float constant.
- **SineGen randomness becomes an input.** Its random initial phase has no effect upstream: the following linear downsampling by 300 never samples index 0. The harmonic noise draw is generated by `KokoroNoise` (SplitMix64 plus Box–Muller, identical to `KreaNoise`) from `KokoroSpeechConfiguration.seed`, so output is reproducible. Upstream draws from PyTorch's unseeded generator.

## API

```swift
import AetherKit
import AetherSpeech
import AetherSpeechCoreAI

let assets = try KokoroAssets.open(at: bundleURL)          // validates kokoro.json and pins; no model bytes read
let session = KokoroSpeechSession(assets: assets)           // durations on CPU, decoder on GPU (see Compute)
do {
    let phonemes = try session.phonemes("həlˈO wˈɜɹld, ðɪs ɪz kˈOkəɹO.")
    let speech = try await session.synthesize(phonemes, configuration: try KokoroSpeechConfiguration(voice: .afHeart))
    try speech.audio.saveWAV(to: outputURL)                 // 16-bit PCM; .float32 keeps unclipped samples
    await session.unload()
} catch {
    await session.unload()
    throw error
}
```

- **Typed voices.** `KokoroVoice` has static members for all 54 v1.0 voices (`.afHeart`, `.bmGeorge`, …) with `language` and `gender`. `KokoroVoice("af_heart")` parses an identifier, and `assets.availableVoices` lists what the bundle carries.
- **Configuration.** `speed` must be in 0.25…4 and divides durations before rounding. `seed` seeds the vocoder noise.
- **Chunked delivery.** `synthesize(segments:configuration:progress:)` runs utterances in order and delivers each one as a `.chunk` event as soon as it is decoded. Segment *i* uses seed `seed &+ i`. This is streaming at segment granularity: a player can start after the first sentence. `GeneratedAudio.concatenating` joins the parts. Sub-utterance streaming would need a windowed decoder and is not implemented; the vocoder's instance norms and the frame-level LSTM see the whole utterance.
- **`GeneratedAudio`** (AetherKit) is planar float32 at any sample rate, with WAV writing (`pcm16` clips and rounds ties to even; `float32` keeps samples unclipped). It is generic, so other audio producers can use the same delivery type.
- **Lifecycle** follows Krea. Models load on first use and stay resident. One request runs at a time; overlap fails with `sessionBusy`. Cancelling the caller task stops between native calls after awaiting the submitted call, and the session returns to `ready` with models retained. `unload()` rejects new work, waits for the active request, then closes permanently.
- **Installation.** `assets.installationManifest()` describes `kokoro.json`, both model directories and every voice for `LocalAssetStore`. `KokoroSpeechSession(installed:)` holds the read lease until joined unload.
- **Admission.** `KokoroResourceAdmission` is opt-in. Its two model estimates are keyed by stream and metadata hash and are checked against the bundle when you construct it, so estimates from another export are refused before any request runs.
  - A decorator around the model factory, the same pattern Krea uses, reserves each model's residency before opening it. Failed opens roll back, and the residency is held until joined unload.
  - It also reserves per-call scratch around each native call. Operation bytes are held for the whole request and must cover the retained segment audio.
  - These are caller estimates, not measured memory. Qualified estimates for the real models are not yet published.
- **Frame cap.** `frameCap: KokoroFrameCap` limits utterance length. The default is `.platformDefault`: `.standard` (1024 frames, about 25.6 s) on macOS and `.compact` (512 frames, about 12.8 s) on iOS. `.extended` (2048 frames) is an explicit opt-in that costs about 8 GB more GPU memory. Lower caps are allowed for devices with less memory. An utterance above the cap is refused with an error that gives its frame count and the cap; split such text into shorter segments. The check runs after the durations model has predicted frames and before any decoder work. In a multi-segment request, earlier segments may already have been delivered when a later one is refused.

  Memory follows the decoder bucket, not the cap. A cap rounds up to the next bucket (`KokoroFrameCap.decoderFrames(in:)`), so a 600-frame cap runs at bucket 1024, the size the system killed on the qualified iPhone. On iOS, use `.compact` or a cap of 512 or less. Only iOS builds, including Mac Catalyst, default to `.compact`; macOS, visionOS and tvOS default to `.standard`. `prewarm()` stops at the cap's bucket.
- **Shared GPU admission.** `gpu: GPUAdmissionOwner` opts a session into a host-shared `GPUAdmission`, following the Krea and FLUX pattern. Each request is granted before any model work or reservation, and `unload()` withdraws a request still waiting. The opt-in native suites take the shared lease through `nativeGPUGrant()`.
- **Segment results** are retained in the returned array, at 2.4 KB per frame. A request is limited to `maximumSegments` (256).

## Compute placement

The default `KokoroComputePlacement.recommended` runs `durations` with `cpuOnly` and `decode` with `preferGPU`. Framework findings on macOS 27.0.1 / coreai-torch 0.4.2 force this:

- **`durations` on the GPU aborts** in MPSGraph's `ANERegionFormationPass` ("Region input 1 was not rank-4 or rank-5"), reproduced with `Forge/kokoro/probe_coreai_graph.py`. On the CPU it takes about 30 ms per utterance.
- **A decoder with float16 compute aborts on the GPU.** MPSGraph places its float16 regions on the Neural Engine, and ANE inference then fails with an assertion. `SpecializationOptions.allowedComputeUnitKinds` is read-only, so an app cannot require the GPU and exclude the Neural Engine. See [Reduced-precision decoders](#reduced-precision-decoders).
- **The decoder opened with `expectFrequentReshapes: true` on the GPU aborts the process** in CoreAIRuntime ("Cannot make a Tensor.Layout from an unresolved TensorRequirements"); see [this log](evidence/kokoro-gpu-reshape-abort-2026-10-26.log). AetherSpeech therefore sets the reshape hint only for CPU placement.

Without the hint, **each new GPU frame count specializes once**. On an unbucketed decoder that means a 2–6 s stall the first time each utterance length appears, plus memory that never comes back (see the next section). MPSGraph also logs a non-fatal ANE compile failure during each specialization and then falls back. `.uniform(.cpuOnly)` avoids GPU specialization but runs at about 1.3× real time.

## Length buckets

Bundles exported since the bucket work (`export_kokoro.py --decoder bucketed`, the default) declare `decoderBuckets` (64, 128, 256, 512, 1024, 2048, 4096 frames) and a decoder that takes an extra `frames` input. For each utterance, the session pads the aligned features to the smallest bucket that fits and extends the seeded noise to the bucket length. It passes the valid frame count and trims the padded tail from the output. The GPU therefore specializes at most once per bucket. `prewarm(maximumFrames:)` specializes buckets ahead of time, and the session decodes old unbucketed bundles unpadded as before.

**Exactness.** The padded decoder confines everything that depends on sequence length to the valid prefix:

- InstanceNorm statistics: one pass of masked sums, with the affine terms folded into a per-channel gain and shift, and the output masked to exact zeros;
- the reverse LSTM direction;
- the F0 phase interpolation edge, with the last valid value held over the padding;
- the harmonic STFT's end reflection, with ten samples written at the valid end;
- the iSTFT frames and window envelope.

LeakyReLU and Snake both map 0 to 0. Conv inputs that follow a masked norm therefore already match zero padding, and only transposed-conv outputs and generator-stage inputs are masked again.

In float64, padded and unpadded valid prefixes agree to 1e-12, so the masking is exact. In float32, reduction-order rounding is amplified exactly as the model's own float32-vs-float64 difference is:

- Padded output is bit-identical when an utterance exactly fills its bucket.
- Otherwise it differs by 0.4–1.8% relative L2, measured on the four parity cases and on lengths 63/64/65 through 511/512/513 ([eager check](evidence/kokoro-buckets-lean-eager-check-2026-10-26.json)).
- Against upstream, the bucketed native run stays within the unchanged parity limits: frames exact, rel. L2 0.008–0.033, log-mel mean 0.004–0.013 ([report](evidence/kokoro-buckets-lean-native-2026-10-26.json)).

**Warm speed**, paired and interleaved: one process per shape, all three decoders warm, nine rounds, median seconds ([log](evidence/kokoro-buckets-paired-timing-2026-10-26.log)):

| Shape (frames) | Unbucketed | First masking (`f771a0b`) | Lean masking (current) |
| ---: | ---: | ---: | ---: |
| 128 | 0.088 | 0.196 (2.2×) | 0.103 (1.17×) |
| 256 | 0.157 | 0.405 (2.6×) | 0.161 (1.03×) |
| 1024 | 0.775 | 1.910 (2.5×) | 0.736 (0.95×) |

The first masking's cost came from the decomposed two-pass norm statistics and from extra masking around every norm and conv. Together these multiplied the number of large-tensor kernels. The lean form above runs at parity with the unbucketed decoder at the same shape. Padding then costs at most the ratio of the bucket size to the utterance length: a 913-frame utterance padded to 1024 took a median 0.70 s, against 0.94 s unpadded, in the same rounds. These are paired ratios on a loaded host (load average 26–36); absolute times there are not benchmarks.

**Memory** ([summary](evidence/kokoro-buckets-2026-10-26.json)). Both decoders ran over the same 50 distinct lengths, 41–1877 frames:

| | Unbucketed decoder | Bucketed decoder |
| --- | --- | --- |
| First use of a new length | 1.9–5.8 s, every new length | Once per bucket; later lengths in a used bucket need no specialization |
| Process footprint | 19.2 GB after 50 lengths, growing with each | about 4.2 GB through bucket 1024; 6–12 GB while bucket 2048 is in use |
| Metal allocation | 33.9 GB, growing with each length | 13.5 GB with buckets 64–2048, flat within each bucket |
| After `unload()` | 14.0 GB footprint, 17.6 GB Metal | with bucket 2048 used: footprint 0.33 GB or 8.5 GB (two runs; the 8.5 GB reading was taken before the release finished, see [Memory after unload](#memory-after-unload-003)), Metal about 8.2 GB; at or under the 1024 cap: 2.2 GB footprint and 2.1 GB Metal in both of two runs |

Metal allocation per bucket grows faster than the bucket size: 0.3, 0.4, 0.6, 1.6 and 2.3 GB for buckets 64 through 1024, and 8.3 GB for bucket 2048 alone. The session therefore caps utterances at 1024 frames by default on macOS and at 512 on iOS (`KokoroFrameCap.platformDefault`). Bucket 2048 is used only with `frameCap: .extended`, and bucket 4096 never. Size admission estimates for the largest bucket the cap allows, and prewarm only the buckets your utterances need. Right after unload, the framework's allocator caches are not returned deterministically, and MPSGraph keeps some per-process Metal allocation past model release. The cycling measurement below separates the two.

**Cold first synthesis.** The qualification ran on a bundle exported minutes earlier that had never been specialized, so each bucket's first use was genuinely cold. No shared cache was cleared; Core AI exposes no per-process cache directory. Results:

- 1.8 s for the first utterance, which also loads the models;
- 1.3 s and 2.5 s for the first use of buckets 256 and 1024;
- 0.12–0.20 s for later unseen lengths (66, 103 and 146 frames) in warmed buckets.

### Memory after unload (0.0.3)

`kokoroUnloadCycling` (opt-in, in `AetherNativeTests`) runs five cycles in one process. Each cycle opens a session,
synthesizes a short utterance, calls `unload()` and drops the session. With `AETHER_NATIVE_UNLOAD_KOKORO_CAP=extended`
the session uses `frameCap: .extended` and also prewarms through bucket 2048. It is the same harness as FLUX.2's
([FLUX.2 guide](flux2-klein.md#memory-after-unload-003)).

- **Standard cap:** 70 MiB of footprint remains after the first cycle, then within 3 MiB over five cycles; Metal 13 MiB.
- **Bucket 2048:** at the moment `unload()` returns, the footprint is still 8.3–8.6 GB. 2 s later it is 0.09–0.28 GB,
  in every cycle. The release completes shortly after `unload()` returns, so the earlier "8.5 GB in one of two runs"
  was a reading taken too early, not memory that stays. The Metal device's `currentAllocatedSize` stays at 8,312 MiB
  from the second cycle (10,312 after the first) and does not grow. At the same time `footprint` shows about 2 MiB of
  IOAccelerator memory and system wired memory is flat. The counter does not mean the memory is held. Kit allocates no
  Metal memory in this path; the counter belongs to Core AI's MPSGraph execution. The footprint noise of up to
  0.28 GB is the macOS allocator's large-block cache described in the FLUX.2 guide.

See the [summary](evidence/unload-cycling-2026-10-27.json).

```sh
AETHER_NATIVE_UNLOAD_CYCLING=kokoro AETHER_NATIVE_UNLOAD_KOKORO_CAP=extended AETHER_NATIVE_UNLOAD_OUTPUT=/new/output/dir \
AETHER_KOKORO_BUNDLE=~/.cache/huggingface/aether-kit/kokoro-82m/coreai-fp32-buckets-2026-10-26 \
~/.cache/aether-kit/with-gpu-lock.sh "kokoro unload cycling" swift test --triple arm64-apple-macosx27.0 -c release --filter kokoroUnloadCycling
```

## iPhone

`Examples/SpeechDevice` is an XcodeGen app, KokoroDeviceProbe. It synthesizes the parity cases on the device and prints one JSON report. To run it:

1. `Forge/kokoro/stage_device_assets.py` copies a trimmed bundle (models plus the four reference voices) and the references into `Assets/`, which is git-ignored.
2. Build with automatic signing, using only existing provisioning.
3. `xcrun devicectl device process launch --console` runs the app, and the app is uninstalled afterwards.

The run on 2026-10-26 used an iPhone19,4 (11.3 GB RAM, iOS 27.1, app memory budget about 3.4 GB) with the float32 bundle and the recommended placement ([record](evidence/kokoro-iphone-2026-10-26.json)):

| Case | Bucket | Rel. L2 (bound) | Log-mel mean | Warm synthesis | Warm RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-af_heart | 128 | 0.0081 (0.029) | 0.0038 | 0.19 s | 0.078 |
| speed-bf_emma | 256 | 0.0233 (0.050) | 0.0065 | 0.38 s | 0.084 |
| sentence-am_michael | 256 | 0.0277 (0.050) | 0.0109 | 0.38 s | 0.061 |
| long-af_bella (913 frames) | 1024 | killed by the system with the 1024 cap; refused before decoding with the 512 cap | | | |
| 463-frame memory probe | 512 | not compared (no reference) | | 0.75 s | 0.065 |

- **Parity and speed.** Frames match exactly and parity holds within the Mac limits, running 12–16× faster than real time. The first utterance takes 5.5 s, including model load and the first specialization; a later bucket's first use takes 0.7–1.1 s.
- **Memory.** Peak footprint was 2.1 GB with bucket 512 in use, leaving 1.3 GB of the budget. Bucket 1024 exhausted the budget: the app had 2.7 GB available before that case and was killed.
- **iOS default.** iOS sessions therefore default to `KokoroFrameCap.compact` (512 frames, about 12.8 s), and longer text must be segmented.

**Second iPhone, bucket 512 (2026-10-28).** An iPhone18,2 (iPhone 17 Pro Max, iOS 27.2, 11.4 GiB; 3,368 MiB available
to the app at launch, the same as the iPhone19,4) ran the same float32 bundle with the bucket-512 reference set and the
512 cap ([record](evidence/kokoro-iphone18-bucket512-2026-10-28.json),
[batch summary](evidence/iphone-batch-2026-10-28.json)). All cases pass with frames and tokens exact:

| Case | Bucket | Rel. L2 (bound) | Log-mel mean | Warm synthesis | Warm RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-af_heart | 128 | 0.0102 (0.029) | 0.0040 | 0.24 s | 0.097 |
| speed-bf_emma | 256 | 0.0331 (0.050) | 0.0070 | 0.50 s | 0.110 |
| sentence-am_michael | 256 | 0.0268 (0.050) | 0.0110 | 0.49 s | 0.080 |
| medium-af_bella | 512 | 0.0254 (0.050) | 0.0150 | 1.01 s | 0.091 |
| long-af_bella (913 frames) | — | refused above the 512 cap before decoding | | | |

- **Bucket 512 on a device:** `medium-af_bella` passes at 0.0254 (Mac: 0.027), closing the 0.0.2 gap.
- **Memory:** peak footprint 2,144 MiB, with at least 1,292 MiB still available; 1,094 MiB of footprint remained after
  `unload()` in the same process.
- **First utterance:** 7.0 s, including load and the first specialization.
- **The float16 Neural Engine retest still aborts on iOS 27.2.** The decoder-only mode ran the float32 decoder at
  bucket 256, then opened the float16-compute decoder, whose first call aborted the process in MPSGraph:
  `ANERegion.mm:445: failed assertion 'ANE inference operation failed'` (`Code=-19`), the same failure the Mac hit at
  `ANERegion.mm:414` on macOS 27.0.1 ([log](evidence/kokoro-iphone18-fp16-ane-abort-2026-10-28.log)). No float16 timing
  or accuracy was obtained, and the float32 decoder stays the only shipped one.

**Prepared for that run (0.0.3).**

- **A bucket-512 reference case.** `reference_kokoro.py` adds `medium-af_bella`: two sentences, 442 frames, float64
  floor rel L2 0.022. The four earlier cases regenerate with byte-identical audio: each `audio.f32` hashes the same as
  before ([audio hashes](evidence/kokoro-references-bucket512-audio-2026-10-27.json)). On the Mac, native parity passes all
  five cases, with the new one at rel L2 0.027 and log-mel 0.015
  ([references](evidence/kokoro-references-bucket512-2026-10-27.json),
  [report](evidence/kokoro-native-bucket512-2026-10-27.json)). A first choice, `medium-am_adam` (419 frames), was
  rejected. Kit measured 0.058, about 1.07× that case's own float32-versus-float64 upstream floor (0.054), so it was within
  the floor-multiple rule. But the absolute 0.05 bound is out of reach on that voice even for upstream float32 against
  float64, so the case can't be judged by it ([report](evidence/kokoro-native-bucket512-am_adam-rejected-2026-10-27.json)). Staged with `--frame-cap 512`, the
  device run covers the four cases up to bucket 512 and checks that the 913-frame case is refused before decoding.
- **The float16 Neural Engine retest.** `stage_device_assets.py BUNDLE REFERENCES --decoder-only FP16_BUNDLE` stages
  only two decoders: the float32 one and the reduced-precision one from an `export_kokoro.py --decoder-precision
  float16` bundle (30 s and 1.5 GiB on the CPU). KokoroDeviceProbe then runs both through `ModelSession` with
  `preferGPU` on the same synthetic inputs at bucket 256, bypassing the session's float32-only rule. It records
  timings and the float16 output's distance from float32. If iOS still places the float16 regions on the Neural
  Engine and aborts, the process dies, and its last progress line shows where.

## Reduced-precision decoders

**0.0.2 ships the float32 decoder only.** The `--decoder-precision` export flags are experimental, and every reduced-precision decoder fails qualification. `export_kokoro.py --decoder-precision` produces three alternative decoders, recorded in the bundle contract as `decoderPrecision` (AetherSpeech refuses to open any bundle whose decoder is not float32):

- `float16`: float16 weights and float16 conv/linear compute;
- `float16-storage`: float16 weights, widened to float32 for compute;
- `int8`: weight-only, symmetric, per-channel int8 via coreai-opt.

In all three, the norms, LSTMs, harmonic source and STFT/iSTFT stay float32, and graph inputs and outputs stay float32. **None passes the limits stated before the runs**, so only the float32 decoder ships ([record](evidence/kokoro-precision-2026-10-26.json)).

| Decoder | Size | Accuracy vs upstream (rel. L2 / log-mel mean) | Warm decode, paired (256 / 1024 frames) | Metal after bucket 1024 |
| --- | ---: | --- | --- | ---: |
| float32 (shipped) | 236 MiB | 0.008–0.033 / 0.004–0.013, passes | 0.125 / 0.526 s | 5.1 GB |
| float16-storage | 140 MiB | 0.045–0.415 / 0.017–0.071, fails 2 of 4 cases | 0.131 / 0.531 s | 5.6 GB |
| int8 | 65 MiB | 0.63–1.37 / 0.076–0.251, fails all cases | 0.125 / 0.527 s | 5.1 GB |
| float16 compute | 121 MiB | 0.91–1.35 / 0.13–0.24 (CPU-only run), fails all cases | aborts on the GPU | — |

- The float16-compute decoder cannot run on the GPU. MPSGraph places its float16 regions on the Neural Engine, where inference fails with an assertion ([log](evidence/kokoro-precision-fp16-compute-gpu-abort-2026-10-26.log)), and `SpecializationOptions.allowedComputeUnitKinds` is read-only, so the GPU cannot be required. Its accuracy was therefore measured with `cpuOnly`, at about 2.5× real time.
- Neither weight format changes decode time or run-time memory. Weights are widened to float32 for compute, and activations dominate memory. Per-bucket Metal allocation for each variant is in the record.
- The cause is the same sensitivity that sets the float32 floor. A 1e-6 relative perturbation of the harmonic source already moves the waveform by 1.5e-4, so weight rounding at about 1e-3 relative error, as in float16, shows up clearly in both metrics.

## Qualification

Run `Scripts/qualify-kokoro-native.sh BUNDLE REFERENCES NEW_OUTPUT [recommended|preferGPU|automatic|cpuOnly]` under the shared GPU lock. It runs the opt-in `kokoroNativeParityTimingAndLifecycle` test, which is skipped unless `AETHER_NATIVE_KOKORO_QUALIFICATION=1`.

References come from `Forge/kokoro/reference_kokoro.py`. It runs the unmodified upstream `KModel` in CPU float32 on four fixed phoneme cases (29–335 phonemes; voices af_heart, am_michael, bf_emma at speed 1.2, and af_bella at speed 0.9), with the same seeded noise wrapped into upstream's istftnet module. For each case it also records two floors:

- **Sensitivity floor:** upstream compared with itself after a 1e-6 relative perturbation of the harmonic source.
- **Precision floor:** upstream float32 compared with the same upstream code in float64.

### Tolerances

These were stated in the test before the first passing native run:

- token IDs and integer frames per token must equal upstream exactly;
- waveform relative L2 ≤ 0.05 **and** ≤ 4 × that case's float32-vs-float64 upstream difference;
- log-mel mean absolute difference ≤ 0.03 nats. The log-mel uses an 80-band HTK mel scale over 0–12 kHz, `n_fft` 1024, hop 256, ln with a 1e-5 floor, and is implemented identically in Python and Swift; the test cross-checks the two on the sensitivity pair.

**Reduced-precision decoders** (float16 and weight-only int8 exports) use the following bounds, stated before their first native run on 2026-10-26:

- frames exact; the duration graph stays float32;
- waveform relative L2 ≤ min(0.25, 10 × the case's float32-vs-float64 upstream rel. L2);
- log-mel mean ≤ min(0.08, 5 × that case's float32-vs-float64 log-mel mean).

Both bounds scale with the model's measured float32 sensitivity for that case. The waveform bound is the looser of the two because phase error dominates waveform L2 and is perceptually weak, while the log-mel bound tracks the spectral envelope.

The reason for a relative waveform bound: the vocoder's harmonic source accumulates phase over the utterance, which amplifies float32 rounding. A 1e-6 perturbation of the source moves the output by 1.5e-4 to 7e-4 relative L2. Upstream's own float32 run differs from its float64 run by 0.7–2.5%, the same order as the export.

### Results (2026-10-26)

All checks passed: [native report](evidence/kokoro-native-2026-10-26.json), [test log](evidence/kokoro-native-tests-2026-10-26.log).

| Case | Seconds | Frames exact | Rel. L2 (bound) | fp32↔fp64 upstream | Log-mel mean | Warm RTF |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| short-af_heart | 2.45 | ✓ | 0.0098 (0.029) | 0.0073 | 0.0040 | 0.033 |
| sentence-am_michael | 6.18 | ✓ | 0.0248 (0.050) | 0.0253 | 0.0106 | 0.031 |
| speed-bf_emma (×1.2) | 4.50 | ✓ | 0.0305 (0.050) | 0.0215 | 0.0070 | 0.033 |
| long-af_bella (×0.9) | 22.83 | ✓ | 0.0268 (0.050) | 0.0229 | 0.0144 | 0.031 |

- **Real-time factor.** Warm synthesis time divided by audio duration is about 0.031, roughly 32× faster than real time on an M5 Max: 0.08 s for 2.45 s of audio and 0.70 s for 22.8 s. Utterance lengths this session had not seen before took 1.80–1.90 s on first synthesis (GPU specialization) and 0.06–0.12 s when repeated. These timings come from a shared host without a quiet-window protocol; they are observations, not benchmarks.
- **Repeats.** Repeated synthesis was bit-identical for every case.
- **Lifecycle.** Cancelling after durations were predicted (just before or during the decoder call) surfaced `CancellationError`, and the session returned to `ready`. Recovery reproduced the baseline exactly. `unload()` during an active request waited, the request completed with baseline-identical audio, the state became `closed`, and later requests failed with `sessionClosed`.
- **CPU-only run.** An earlier run with `.uniform(.cpuOnly)` also passed every gate, with rel. L2 0.010–0.040 and warm RTF 1.25–1.34; see the [CPU-only report](evidence/kokoro-native-cpu-only-2026-10-26.json).
- **Eager export graph.** On the CPU, before Core AI conversion, the export graph sits at the same precision floor (0.7–2.5%) against upstream, as recorded in the [references](evidence/kokoro-references-2026-10-26.json).

**Recheck after rebasing onto 0.0.1 (`128573d`).** The four parity cases and lifecycle checks passed again with bit-identical audio and identical metrics; see the [recheck report](evidence/kokoro-native-recheck-0.0.1-base-2026-10-26.json). The CPU suite with the default `swiftbuild` engine registers 263 tests and passes, with 15 opt-in native suites skipped ([log](evidence/kokoro-cpu-tests-0.0.1-base-2026-10-26.log)).

The pre-rebase [CPU suite](evidence/kokoro-cpu-tests-2026-10-26.log) passed 249 tests. Weight-free coverage includes: tokenization against upstream fixtures, noise parity with the Python harness, torch-round frame parity, alignment, WAV encoding, bundle validation (corruption, traversal, symlink escape, duplicate voices, contract drift), `LocalAssetStore` round trip, chunk ordering and seeds, busy rejection, cancellation recovery, unload ordering, contract violations, frame limits, and the phonemizer path. The pre-rebase run covered only operation-byte admission; model residency and scratch admission, rollback, and bundle-matched estimates are covered from the review fixes onward.

## Reproduce

```sh
uv sync --frozen --project Forge; FORGE=Forge/.venv/bin/python   # torch 2.11 + coreai-torch 0.4.2
SNAP=~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987
UP=~/.cache/aether-kit/kokoro/upstream/kokoro-0.9.4   # unpacked kokoro-0.9.4 wheel
$FORGE -I Forge/kokoro/export_kokoro.py ~/.cache/huggingface/aether-kit/kokoro-82m/NEW --snapshot $SNAP --upstream $UP
$FORGE -I Forge/kokoro/reference_kokoro.py ~/.cache/huggingface/aether-kit/kokoro-82m/NEW-references --snapshot $SNAP --upstream $UP
~/.cache/aether-kit/with-gpu-lock.sh "kokoro: native" sh Scripts/qualify-kokoro-native.sh BUNDLE REFERENCES NEW_OUTPUT
```

Core AI exports are not byte-reproducible, so a new export gets new model hashes in `kokoro.json`. Qualification relies on the reference comparison, not on hash continuity. `Examples/Speech` is a consumer command, `aether-speak BUNDLE OUT.wav VOICE 'PHONEMES|PHONEMES'`, that prints each segment as it arrives.

## Next steps

- iPhone parity at bucket 512 (with a reference case in that range), other device models, and whether the increased-memory-limit entitlement admits bucket 1024.
- A reduced-precision decoder that passes: for example, float16 storage limited to the large generator convolutions, or quantization-aware calibration against the references.
- Reduce bucket 2048's Metal allocation, or drop that bucket by capping segment length; consider a fixed-window streaming decoder for long utterances.
- Retest GPU `durations` and the reshape hint on later OS/coreai-torch releases.
- A recorded listening review, and a license-clean G2P sample adapter (CMUdict-based), kept outside the SDK targets.
