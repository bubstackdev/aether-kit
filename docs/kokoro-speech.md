# Kokoro speech synthesis — 2026-10-26

Status: experimental, 0.0.2 checkpoint. AetherSpeech turns validated Kokoro phoneme strings into 24 kHz mono float32 PCM using two exported Core AI graphs. Native parity with the upstream PyTorch model, timing, cancellation and unload passed on one Mac (M5 Max, macOS 27.0.1), recorded in the [qualification record](evidence/kokoro-qualification-2026-10-26.json). Not yet covered: iOS, quantized or fp16 exports, sub-utterance streaming, and listening review.

## Source and license

| Item | Value |
| --- | --- |
| Model | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) v1.0, revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, not gated |
| Weights | `kokoro-v1_0.pth`, SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`; this matches the hash published on the model card |
| License | **Apache-2.0**: model card front matter, card text and Hub metadata (`license:apache-2.0`), confirmed 2026-10-26 |
| Training-data attribution | The card lists CC BY 3.0 (Koniwa `tnc`) and CC BY 4.0 (SIWIS) audio in the training set. Keep that attribution wherever the weights are redistributed |
| Base model | `yl4579/StyleTTS2-LJSpeech` (StyleTTS 2 with an iSTFTNet vocoder) |
| Reference code | `kokoro` 0.9.4 wheel (Apache-2.0, SHA-256 `a129dc63…5789efd`), used read-only by `Forge/kokoro` for export and references |

AetherKit ships no weights. Users export a bundle themselves (see below), and those files keep their upstream license.

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
- **Segment results** are retained in the returned array, at 2.4 KB per frame. A request is limited to `maximumSegments` (256).

## Compute placement

The default `KokoroComputePlacement.recommended` runs `durations` with `cpuOnly` and `decode` with `preferGPU`. Two framework failures on macOS 27.0.1 / coreai-torch 0.4.2 force this:

- **`durations` on the GPU aborts** in MPSGraph's `ANERegionFormationPass` ("Region input 1 was not rank-4 or rank-5"), reproduced with `Forge/kokoro/probe_coreai_graph.py`. On the CPU it takes about 30 ms per utterance.
- **The decoder opened with `expectFrequentReshapes: true` on the GPU aborts the process** in CoreAIRuntime ("Cannot make a Tensor.Layout from an unresolved TensorRequirements"); see [this log](evidence/kokoro-gpu-reshape-abort-2026-10-26.log). AetherSpeech therefore sets the reshape hint only for CPU placement.

Without the hint, **each new GPU frame count specializes once**. On an unbucketed decoder that means a 2–6 s stall the first time each utterance length appears, plus memory that never comes back (see the next section). MPSGraph also logs a non-fatal ANE compile failure during each specialization and then falls back. `.uniform(.cpuOnly)` avoids GPU specialization but runs at about 1.3× real time.

## Length buckets

Bundles exported since the bucket work (`export_kokoro.py --decoder bucketed`, the default) declare `decoderBuckets` (64, 128, 256, 512, 1024, 2048, 4096 frames) and a decoder that takes an extra `frames` input. For each utterance, the session pads the aligned features to the smallest bucket that fits and extends the seeded noise to the bucket length. It passes the valid frame count and trims the padded tail from the output. The GPU therefore specializes at most once per bucket. `prewarm(maximumFrames:)` specializes buckets ahead of time, and the session decodes old unbucketed bundles unpadded as before.

**Exactness.** The padded decoder confines everything that depends on sequence length to the valid prefix:

- InstanceNorm statistics;
- every conv input, masked to look like zero padding;
- the reverse LSTM direction;
- the F0 phase interpolation edge, with the last valid value held over the padding;
- the harmonic STFT's end reflection, with ten samples written at the valid end;
- the iSTFT frames and window envelope.

In float64, padded and unpadded valid prefixes agree to 1e-12. The masking is therefore exact; the only difference is reduction order. In float32 that rounding is amplified exactly as the model's own float32-vs-float64 difference is. Padded output is bit-identical when an utterance exactly fills its bucket, and differs by 0.3–1.5% relative L2 otherwise, measured on the four parity cases and on lengths 63/64/65 through 511/512/513 ([eager check](evidence/kokoro-buckets-eager-check-2026-10-26.json)). Against upstream, the bucketed native run stays within the unchanged parity limits: frames exact, rel. L2 0.010–0.031, log-mel mean 0.004–0.014 ([report](evidence/kokoro-buckets-native-2026-10-26.json)).

**Measured trade-off** ([summary](evidence/kokoro-buckets-2026-10-26.json)). Both decoders ran over the same 50 distinct lengths, 41–1877 frames:

| | Unbucketed decoder | Bucketed decoder |
| --- | --- | --- |
| First use of a new length | 1.9–5.8 s, every new length | Once per bucket; later lengths in a used bucket need no specialization |
| Process footprint after 50 lengths | 19.2 GB, growing with every length | about 4.2 GB, flat within each bucket |
| Footprint after `unload()` | 14.0 GB, not returned | 0.33 GB |
| Metal allocation after 50 lengths | 33.9 GB | 13.4 GB, with buckets 64–2048 used |

Metal allocation per bucket grows faster than the bucket size: 0.3, 0.7, 1.3, 2.9, 5.1 and 13.4 GB cumulative through buckets 64, 128, 256, 512, 1024 and 2048. Size admission estimates for the largest bucket a session can reach. Prewarm only the buckets your utterances need, and split text so that segments stay at or below 1024 frames (about 25 s) where memory matters.

**Cold first synthesis.** The qualification ran on a bundle that had been exported minutes earlier and never specialized, so each bucket's first use was genuinely cold. No shared cache was cleared; Core AI exposes no per-process cache directory. Results:

- 2.9 s for the first utterance, which also loads the models;
- 1.8 s and 5.6 s for the first use of buckets 256 and 1024;
- 0.3–0.6 s for later unseen lengths (66, 103 and 146 frames) that fell into warmed buckets.

**Cost.** In this run the bucketed decoder was slower when warm, with a real-time factor of 0.12–0.19 on the parity cases. In interleaved runs in one process, at 1024 frames and regardless of padding, it took about 1.4–2.5× the unbucketed decoder's time.

- The cause is the masking itself, not padding: the bucketed graph is just as slow at an exact bucket fit.
- Every timing run shared the host with load averages of 50–100.
- The unbucketed figure from a quieter part of the day (0.031) is not directly comparable, and a quiet-window remeasure is pending.
- Buckets bound compile stalls and memory. They currently cost warm throughput.

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
FORGE=/path/to/forge/.venv/bin/python   # torch 2.11 + coreai-torch 0.4.2
SNAP=~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987
UP=~/.cache/aether-kit/kokoro/upstream/kokoro-0.9.4   # unpacked kokoro-0.9.4 wheel
$FORGE -I Forge/kokoro/export_kokoro.py ~/.cache/huggingface/aether-kit/kokoro-82m/NEW --snapshot $SNAP --upstream $UP
$FORGE -I Forge/kokoro/reference_kokoro.py ~/.cache/huggingface/aether-kit/kokoro-82m/NEW-references --snapshot $SNAP --upstream $UP
~/.cache/aether-kit/with-gpu-lock.sh "kokoro: native" sh Scripts/qualify-kokoro-native.sh BUNDLE REFERENCES NEW_OUTPUT
```

Core AI exports are not byte-reproducible, so a new export gets new model hashes in `kokoro.json`. Qualification relies on the reference comparison, not on hash continuity. `Examples/Speech` is a consumer command, `aether-speak BUNDLE OUT.wav VOICE 'PHONEMES|PHONEMES'`, that prints each segment as it arrives.

## Next steps

- fp16 or int8 decoder export, with new tolerances and size/RTF measurements; an iOS build and device run.
- Remeasure bucketed vs unbucketed warm throughput in a quiet window. If the masking overhead holds, consider fewer masked stages, for example padding only the generator stages, or a fixed-window streaming decoder.
- Retest GPU `durations` and the reshape hint on later OS/coreai-torch releases.
- A recorded listening review, and a license-clean G2P sample adapter (CMUdict-based), kept outside the SDK targets.
