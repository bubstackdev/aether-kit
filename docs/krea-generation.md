# Krea reference generation — 2026-10-17

## Implemented behavior

`AetherKreaCoreAI.KreaGenerationSession` now implements the complete reference path: prepared prompt → Qwen text encoder → selected conditioning → four denoiser segments at each scheduler step → separately rounded CPU Euler updates → VAE decode → owned float32 planar RGB. Callers may instead supply conditioning and exact initial noise for fixture replay. `GeneratedImage` retains unclipped float32 pixels and writes PNG using the reference float32 scaling and ties-to-even quantization.

This implementation copies tensors through host memory between model calls. It does not yet preserve device-resident carries or implement a Metal sampling kernel, so it makes no throughput or speedup claim. Native provided-conditioning and full encoded runs now produce images and match their independent reference hashes for the retained 768×1344 tower fixture.

The four segment descriptors are checked before denoiser execution for scalar type, shape, statelessness, upstream carries, required root inputs, non-overwritten outputs and a final-segment velocity. Carries remain available through the entire step, including nonadjacent consumers of `cond`. Runtime results are checked again. The decoder must consume fp16 `[height/16,width/16,64]` latents and produce float32 `[3,height,width]` pixels.

## Specialization registration

`assets.registeringSpecialization(at:)` reads the prototype's existing JSON registration format. It requires four original source hashes matching the opened bundle, four specialized hashes, four distinct specialized assets and both shape dimensions. It streams the original and specialized source bytes through SHA-256 and verifies the small model metadata before publishing a new immutable asset set. Registration can read roughly 25 GB for the retained tower export.

Registration files are trusted offline provenance supplied by the caller; hashes establish content linkage, not the semantic correctness or publisher authenticity of a newly exported model. Unknown fields, incompatible source identities, duplicate shapes, malformed counts and corruption fail. At most eight registrations are accepted. Other shapes select the original dynamic bundle. Keep external assets and registration provenance immutable during use.

Only one denoiser set is retained by a session. Changing its selection releases the previous set before loading the next. Loading is serial in this first implementation; partial-load failure releases already loaded models. Balanced residency now releases any retained denoisers before acquiring an encoder for a new prompt, then releases the encoder after conditioning before loading denoisers. Thus repeated prompt encoding reloads denoisers; provided-conditioning calls still reuse the selected set. Resident mode retains both stages for later requests. The decoder remains available across generation calls.

## Session use

```swift
import AetherKit
import AetherKrea
import AetherKreaTokenizers
import AetherKreaCoreAI

var assets = try KreaAssets.open(at: bundleURL)
assets = try assets.registeringSpecialization(at: shapeRegistrationURL)
let tokenizer = try await KreaTokenizer.open(assets: assets)
let prompt = try tokenizer.prepare("a tall clock tower in a snowy old town at night, lanterns glowing")
let session = KreaGenerationSession(assets: assets)
do {
    let image = try await session.generate(
        configuration: KreaConfiguration(width: 768, height: 1344, seed: 71344),
        prompt: prompt
    )
    await session.unload()
    try image.savePNG(to: outputURL)
} catch {
    await session.unload()
    throw error
}
```

Registration is optional for ordinary dynamic rendering. `assets.verify()` is an explicit full integrity check over all six dynamic models; opening alone checks declarations, small metadata and tokenizer files.

Concurrent generation or prompt preparation on a session fails with `sessionBusy`. Cancel the caller task to stop cooperatively between model calls. The reference session waits for each submitted call before releasing ownership; the one-step native cancellation and unload checks below now verify this ordering at the awaited framework-call boundary. GPU queue draining is not independently instrumented. `unload()` rejects new requests, waits for generation and closes the session permanently. Open a new session to use it again. This does not evict OS caches or promise page eviction.

Progress callbacks are synchronous, serial and throwable. Returning allows generation to continue; throwing aborts it. Latent buffers use Swift value ownership. No unbounded output stream is created. Keep callbacks short and avoid blocking on session lifecycle operations.

## Prepare once, render multiple seeds

Use `session.prepare(prompt:)` to encode a tokenized prompt once. It returns a `KreaPreparedPrompt` containing owned fp16 conditioning and the originating bundle fingerprint. Preparation releases retained denoisers before encoding and unloads the encoder before returning, including when the session uses resident policy. The decoder remains available. Repeated calls to `prepare` explicitly encode again; there is no hidden prompt cache.

```swift
// assets, tokenizer and outputDirectory are supplied by the caller.
let session = KreaGenerationSession(assets: assets)
do {
    let tokens = try tokenizer.prepare("a red fox in fresh snow")
    let prepared = try await session.prepare(prompt: tokens)
    print("Conditioning bytes:", prepared.conditioningByteCount)
    for seed: UInt64 in [17017, 17018] {
        let image = try await session.generate(
            configuration: KreaConfiguration(width: 512, height: 512, seed: seed),
            preparedPrompt: prepared
        )
        try image.savePNG(to: outputDirectory.appendingPathComponent("fox-\(seed).png"))
    }
    await session.unload()
} catch {
    await session.unload()
    throw error
}
```

Keep the same session open between images to retain the selected denoisers/functions and decoder. Configuration, seed and optional exact noise remain per-image inputs. A changed denoiser selection replaces the previous set; preparing another prompt releases denoisers before encoding. The supplied output directory must exist.

Prepared values retain no native models or raw prompt text. Their conditioning payload is at most 30 MiB (512 rows × 30,720 fp16 values); `conditioningByteCount` excludes Swift overhead and model allocations. Callers control how many values they retain and release them by dropping references. Swift value copies share storage until mutation. Values remain valid after session unload and can be used by another session with the same bundle fingerprint. A different bundle fails before model acquisition. Keep external model assets immutable, as with other session APIs; the fingerprint describes declared asset identities and is not a new stream-integrity check.

Preparation and generation share one operation slot: overlap fails with `sessionBusy`. Cancellation waits for submitted work before releasing that slot, and `unload()` waits for active preparation or generation before permanently closing the session. The existing generation events and provided-conditioning path are used for prepared rendering.

The prepared-prompt API passes CPU ownership, reuse, seed, identity, failure, cancellation and unload tests. The opt-in dynamic qualification suite now includes preparation plus two repeated prepared generations per shape against its independently generated eight-step references, checking encoder release and retained model reuse. The [native prepared-prompt qualification](evidence/krea-prepared-prompt-qualification-2026-10-18.json) now passes all 65 checks, including exact prepared repeats at both sizes. The earlier matched benchmark remains separate performance evidence; this qualification run establishes correctness and ownership, not fresh-prompt latency. Preparation-specific cancellation and cross-session prepared ownership remain CPU-qualified.

## Commands

```sh
# Render with an explicit seed and optional trusted specialization.
.build/arm64-apple-macosx/release/aether-probe render-krea \
  /path/to/macos-any-gpu /path/to/image.png \
  'a tall clock tower in a snowy old town at night, lanterns glowing' \
  768 1344 71344 /path/to/tower-20.json

# Native acceptance against the independent Swift provided-conditioning baseline.
.build/arm64-apple-macosx/release/aether-probe parity-krea \
  /path/to/macos-any-gpu /path/to/krea-lab \
  /path/to/fixtures/4c12ae019488ff1c /path/to/tower-20.json \
  /path/to/new-parity-output provided
```

### Batch rendering from the CLI

```sh
source .local/native-assets.env
.build/arm64-apple-macosx/release/aether-probe render-krea-batch \
  "$AETHER_KREA_BUNDLE" "$PWD/.local/fox-batch-01" \
  'a red fox in fresh snow' 512 512 17017,17018,17019
```

The command accepts 1–64 comma-separated unsigned 64-bit decimal seeds and renders eight steps per image unless `--steps=N` (1–50) asks for another count; the report records the count used. It prepares the prompt once, unloads the encoder, then serially reuses conditioning and the same generation session. Append an optional trusted `SHAPE.json` registration as with `render-krea`. Duplicate seeds are allowed; filenames include a one-based index, such as `001-seed-17017.png`, so repeats have distinct paths.

The output directory must be new. `report.json` is written atomically before setup and as each image progresses; it records the prompt, requested seeds and dimensions, bundle fingerprint, conditioning hash/payload size, optional registration hash, OS/RAM, setup duration, per-image generation durations, float32 pixel hashes and PNG byte hashes. A successful command prints the complete report as JSON to stdout; progress goes to stderr. `completed: true` means all PNGs were saved and the session unloaded, rather than an independent numerical parity assertion.

Setup timing includes asset/tokenizer preparation, optional specialization verification, encoding, conditioning hashing and encoder unload. Per-image generation includes model acquisition and progress logging, while final pixel hashing, PNG/report IO and session unload are outside that duration. There are no warm-ups or benchmark statistics. The first image can include initial denoiser/decoder acquisition; use the separate benchmark protocol for performance comparisons.

Dynamic model opening checks declarations, metadata and tokenizer files; this rendering command does not stream-verify all six model weights (`modelStreamsVerified: false`). Optional specialization registration performs its existing source/specialization verification. Assets must stay immutable during use.

On a thrown setup, generation, cancellation or save error, the command waits for active work and unloads the session, exits nonzero and attempts to save `stage: failed`, `completed: false` and the error. Completed images remain in place, and later seeds are not attempted. PNGs are written through temporary files before publication. If the destination itself becomes unwritable, failure-report persistence may also fail; the original error is retained on stderr. A killed process can leave an incomplete report. Existing output directories are never reused or resumed.

Run `encoded` in a separate new output directory to check tokenization, native conditioning and its independent full-encoding baseline. The two modes deliberately use different reference records. The gate verifies dynamic model streams and source-linked specialization before execution, checks token IDs and masks, requires all eight ordered latent hashes and final decoded float32 pixel hash, and writes PNG plus `report.json`. Encoded mode also requires the selected text hash. Mismatches fail immediately and incomplete or preparation-failed runs retain `passed: false`. Reports include reference/fixture/registration hashes, bundle and selection identities, stage, events and elapsed time. Existing output directories are rejected to preserve evidence.

The opt-in native session test uses the same native backend and generation runner, with an internal observer that records model-session identities and brackets the awaited native call itself, before output copying. It does not gate or artificially delay native work. To run it alongside the CPU fixture suite:

```sh
AETHER_KREA_BUNDLE=/path/to/macos-any-gpu \
AETHER_KREA_FIXTURES=/path/to/fixtures/4c12ae019488ff1c \
AETHER_KREA_LAB=/path/to/krea-lab \
AETHER_KREA_SHAPE=/path/to/tower-20.json \
AETHER_NATIVE_QUALIFICATION=1 \
AETHER_NATIVE_OUTPUT=/path/to/new-session-output Scripts/check.sh
```

Without `AETHER_NATIVE_QUALIFICATION=1`, the native test is skipped. It uses one-step generations to isolate session behavior, retaining the eight-step independent parity gates above as separate evidence. Its output directory must be new; it retains `report.json` and images, including failure details if qualification fails. Cancellation and active-unload checks require observing a native call still in progress; finishing before that observation fails the check rather than claiming an unobserved lifecycle property.

The separate full dynamic parity suite creates common input fixtures, invokes the existing unmodified KreaLab executable serially for two independent encoded references, and checks this project's complete eight-step pipeline and cancellation recovery against them:

```sh
source .local/native-assets.env
AETHER_KREA_REFERENCE_PROBE="$AETHER_KREA_LAB/bin/round2-final" \
AETHER_NATIVE_DYNAMIC_QUALIFICATION=1 \
AETHER_NATIVE_DYNAMIC_OUTPUT="$PWD/.local/new-dynamic-qualification" \
AETHER_NATIVE_QUALIFICATION=0 Scripts/check.sh
```

This suite requires the standalone reference executable and a new output directory. It saves generated input bytes, independent reference logs/records, native traces and PNGs under that directory. Native suites are separate opt-ins; run them separately to avoid overlapping real-weight jobs on the shared host. The independent child process is joined and terminated on cancellation or timeout. Neither command downloads models.

The isolated decoder probe consumes the independent reference's saved final latents:

```sh
source .local/native-assets.env
AETHER_NATIVE_DECODER_QUALIFICATION=1 \
AETHER_NATIVE_DECODER_REFERENCE="$PWD/.local/native-dynamic-balanced-20261017/dynamic384-reference" \
AETHER_NATIVE_DECODER_OUTPUT="$PWD/.local/new-isolated-decoder-check" Scripts/check.sh
```

The integrated decoder probe uses retained fixture conditioning, four real dynamic denoisers and the real decoder. It defaults to one step, matching the earlier lifecycle qualification; set `AETHER_NATIVE_DECODER_PIPELINE_STEPS=8` for a focused full-schedule repetition check. Its output is compared with its own pre-cancellation baseline, not an independent image baseline for that conditioning/geometry combination:

```sh
source .local/native-assets.env
AETHER_NATIVE_DECODER_PIPELINE_QUALIFICATION=1 \
AETHER_NATIVE_DECODER_PIPELINE_STEPS=1 \
AETHER_NATIVE_DECODER_PIPELINE_OUTPUT="$PWD/.local/new-pipeline-decoder-check" Scripts/check.sh
```

For a smaller eight-step recovery reproducer, set `AETHER_NATIVE_DECODER_PIPELINE_WIDTH=128`, `AETHER_NATIVE_DECODER_PIPELINE_HEIGHT=128`, and `AETHER_NATIVE_DECODER_PIPELINE_STEPS=8`. Compare `AETHER_NATIVE_DECODER_PIPELINE_RECOVERY=retain` (default, reuse all five sessions) with `reopen` (unload the cancelled pipeline and create a fresh one). Run the modes serially with different new output directories. Width and height default to 384×512; configuration validation still applies.

The pipeline probe writes `events.jsonl` directly on every event. Each event includes a monotonic system-uptime timestamp, phase, model path, instance and sequence; native begin/end events share a call identifier. Submitted calls include input shapes. Internal optional backend observers distinguish function loading, input copies, awaited native execution and output copies; model acquisition and close boundaries are also recorded. `tail -n 1 OUTPUT/events.jsonl` shows the most recent boundary while `report.json` retains completed assertions. Journaling errors fail qualification. Observation polls at 1 ms for this probe to catch short native decodes; it still fails if decoding finishes before cancellation is observed, and never gates or sleeps inside a native call. The observers default to nil in production. Native functions now load lazily once per model session and remain owned until unload. `AETHER_NATIVE_FUNCTION_LIFETIME=call` selects the previous per-invocation loading policy in the tracked test factory; `session` is the default. This diagnostic control does not change the public loader. Diagnostic writes affect execution timing, so these runs do not establish performance.


For this checkout, source `.local/native-assets.env` first. Its model paths now point directly into `~/.cache/huggingface/aether-kit/`:

| Asset | Cache path relative to `~/.cache/huggingface/aether-kit/` |
| --- | --- |
| Krea bundle and tokenizer | `krea-2-turbo-realism/macos-any-gpu/` |
| Tower specialized denoisers | `krea-2-turbo-realism/tower-20/` |
| Registration referencing cached denoisers | `krea-2-turbo-realism/tower-20.json` |
| Affine smoke fixture | `smoke-v1/affine.aimodel` |

The original export directories are compatibility symlinks. All files were hash-verified during the move, including a verified cross-volume copy of the tower specialization before removing its original directory. The registration's content identities remain unchanged; only its segment paths changed. The [migration record](evidence/model-cache-migration-2026-10-17.json) records every file identity and location. These local Core AI exports live under the Hugging Face cache root; Apple's derived specialization cache remains separate.

The full 34-test suite and all 23 native session checks were repeated successfully using these cache paths. The trace confirms every model load came from the cache, and all seven completed generations match their pre-move latent and pixel hashes exactly. Affine smoke also passed from its cached path on all three policies. See the [cache validation manifest](evidence/huggingface-cache-validation-2026-10-17.json), [cached session trace](evidence/krea-hf-cache-session-2026-10-17.json) and [test log](evidence/krea-hf-cache-tests-2026-10-17.log). Cached-run images are under `.local/native-session-hf-cache-20261017`.

## Memory after unload (0.0.3)

`kreaUnloadCycling` (opt-in, in `AetherNativeTests`) runs five cycles in one process. Each cycle opens a session,
generates one 256² image in two steps from a prompt with dynamic denoisers, calls `unload()` and drops the session. It
is the same harness as FLUX.2's ([FLUX.2 guide](flux2-klein.md#memory-after-unload-003)). Footprint 2 s after each
unload, against a 58 MiB baseline: 398, 477, 572, 565 and 562 MiB. Metal allocation stays at 110 MiB.

As with FLUX.2, the residue grows for three cycles and then stops, while live malloc memory stays within 12 MiB of the
baseline. It is freed large allocations that the macOS allocator keeps dirty in its large-block cache. This is macOS
allocator behaviour, bounded and reused by the next load, not a Kit or Core AI leak. As a diagnostic only, with that
cache disabled (`MallocLargeCache=0`) the run settles 36 MiB above the baseline. After two cycles, allocation histories
attribute 35 of the 128 MiB in those regions to the ANE compiler during Core AI's MPSGraph specialization and about
8 MiB to Kit's own tensor copies and tokenizer configuration. The rest has no recorded allocation. Kit makes no change.
See the [summary](evidence/unload-cycling-2026-10-27.json).

```sh
AETHER_NATIVE_UNLOAD_CYCLING=krea AETHER_NATIVE_UNLOAD_OUTPUT=/new/output/dir \
AETHER_KREA_BUNDLE=~/.cache/huggingface/aether-kit/krea-2-turbo-realism/macos-any-gpu \
~/.cache/aether-kit/with-gpu-lock.sh "krea unload cycling" swift test --triple arm64-apple-macosx27.0 -c release --filter kreaUnloadCycling
```

## Verification and limits

The release suite with all external CPU fixtures enabled passes 33 tests. New coverage exercises specialized selection and fallback, descriptor rejection, nonadjacent carry routing, repeated model reuse, balanced/resident encoder lifetime, partial-load cleanup and retry, shape-switch release, cancellation/unload ordering, PNG roundtrip and parity-report failure handling.

The CPU Euler replay matches every retained tower update and final latent byte hash using saved reference velocities. Forge's `stepLatents` files are **pre-update** inputs; the separate lab parity journal contains **post-update** hashes. The test adapter checks that distinction explicitly. This replay qualifies sampling arithmetic, not native model outputs.

The earlier small native affine attempt failed before inference because the sandbox could not open Core AI's cache lock under `~/Library/Caches/coreai-cache`. Follow-up execution outside the agent sandbox passed affine smoke with `cpuOnly`, `automatic` and `preferGPU`, including two executions and retained output after unload. The sandbox itself currently fails to start because its configured cache writable root is a symlink; native runs used approved execution outside it.

The follow-up release suite passed all 33 tests with `AETHER_KREA_BUNDLE`, `AETHER_KREA_FIXTURES` and `AETHER_KREA_LAB` set; none were skipped. Both real-weight parity modes then passed with `preferGPU`, balanced encoder residency and the source-linked tower specialization:

- `provided`: token IDs, all six dynamic model streams and the specialization verified; all eight post-update latent hashes and final decoded float32 pixel hash matched the independent provided-conditioning baseline.
- `encoded`: the same integrity checks passed; native Qwen selected conditioning, all eight post-update latent hashes and final decoded float32 pixel hash matched the separate full-encoding baseline.

Both runs wrote a 768×1344 PNG. The provided image was visually inspected and shows the expected snowy clock tower. Local outputs are `.local/native-parity-provided-20261017/image.png` and `.local/native-parity-encoded-20261017/image.png`. The reports retain the exact reference, registration, fixture and selection identities. The [validation manifest](evidence/krea-native-validation-2026-10-17.json) records commands, source/binary hashes and PNG hashes; the [provided report](evidence/krea-native-provided-2026-10-17.json), [encoded report](evidence/krea-native-encoded-2026-10-17.json) and [complete test log](evidence/krea-native-tests-2026-10-17.log) are preserved in the repository.

These are correctness checks on a shared host under significant concurrent load, as confirmed by the user. Report elapsed times include preparation, verification and native execution, with uncontrolled OS cache state; they are not benchmarks. Latency and throughput qualification awaits a quiet window.

Eight-step independent pixel/latent parity is qualified for one retained 768×1344, seed-71344 fixture and its specialization. The follow-up session checks below extend native coverage to two dynamic shapes and additional prompt lengths, using one-step generations.

## Native session qualification follow-up

The opt-in test passed alongside the full external CPU fixture suite: **34 tests passed, none skipped**. Its 23 native checks recorded 38 native-call pairs, 15 model-session instances and seven completed images. All 15 instances closed exactly once. The [session report](evidence/krea-native-session-2026-10-17.json) preserves the complete ordered trace and output hashes; the [validation manifest](evidence/krea-native-session-validation-2026-10-17.json) records commands, toolchain, source/test-binary hashes and image identities; the [full test log](evidence/krea-native-session-tests-2026-10-17.log) records the suite result.

- Repeated specialized 768×1344 generation retained all four denoisers and the decoder: no additional model opens or closes, with identical latent and float32 pixel values.
- Cancellation was requested after observing the first native segment still in progress. An overlapping request was rejected, native return preceded cancellation return, no second segment ran, and session state returned to ready. A subsequent generation reused the models and reproduced the original latent and pixel values exactly.
- Switching to 512×512 with a 45-token prompt (11 conditioning rows) selected the original dynamic denoisers. All four specialized instances closed before the first dynamic denoiser opened. Repeating with the captured conditioning reused the dynamic models and reproduced both hashes exactly.
- Switching again to 384×512 with a 75-token prompt (41 conditioning rows) exercised dynamic image and text dimensions. The same decoder session handled all three image shapes. Balanced residency released each of the two native encoder instances after conditioning.
- During another 384×512 generation, unload entered closing while the first native call remained active and rejected new work. Generation completed with identical pixels, all native returns preceded model release, unload finished closed, the owned output remained valid, and subsequent generation was rejected. Repeated unload was accepted.

These checks used `preferGPU`, balanced residency and one sampling step. The local artifacts are under `.local/native-session-20261017-native-boundary`; the one-step images are lifecycle/shape test artifacts, not full-schedule image-quality acceptance. Dynamic outputs were checked for finite values, expected shape and repeat stability without an independent dynamic-image baseline.

The observer surrounds the awaited `function.run` itself, reporting native return before owned-output copying, and never blocks or gates native work. The trace establishes ordering at that API boundary; it does not observe GPU command queues or prove immediate physical memory reclamation. Model reuse refers to the same `ModelSession` instances, not OS cache state. These runs also shared the busy host and establish no performance result.

Further qualification beyond the follow-up below includes additional prompts/shapes/seeds, real resident-encoder reuse, broader failure cases, physical iOS support, device-resident optimization and benchmarking during the quiet window.

During the next qualification pass, the 512×512 eight-step run matched the independent reference and encoder cancellation returned cleanly, but re-encoding for recovery stalled in MPSGraph's GPU memory-residency path. The process was stopped and the [incomplete attempt and diagnostic identities](evidence/krea-dynamic-residency-stall-2026-10-17.json) were preserved. It is not a passing recovery result, and shared host load prevents attributing the stall to a single cause. The prior balanced path acquired an encoder alongside the preceding denoiser set; the updated policy above removes that ownership overlap.

The new controlled regression checks close-before-acquire ordering and decoder reuse. With that change, the existing one-step native lifecycle suite again passed all 23 checks and reproduced all seven pre-change latent/pixel hashes; see the [balanced-policy session report](evidence/krea-balanced-session-2026-10-17.json) and [test log](evidence/krea-balanced-session-tests-2026-10-17.log). That run passed all 34 CPU tests plus the existing native suite; the full dynamic suite was disabled so heavy native jobs ran serially.

## Full dynamic numerical parity and stage cancellation (2026-10-17)

The [October 17 qualification record](evidence/krea-dynamic-qualification-2026-10-17.json) separates completed checks from the remaining full-schedule repetition limit. Both cases use seed 17017 and eight steps with the original dynamic assets from the Hugging Face cache. The existing unmodified standalone KreaLab `round2-final` executable generated encoded references in separate child processes, using common token IDs and initial-noise bytes. Each runtime independently executed its encoder, rotary/schedule preparation, denoisers and decoder. Tokenization and initial noise are common inputs, rather than independent implementation comparisons in these new cases.

| Case | Prompt tokens / conditioning rows | Result |
| --- | --- | --- |
| 512×512 snow fox | 45 / 11 | Native conditioning, all eight ordered latent hashes and decoded float32 pixel hash match exactly |
| 384×512 watercolor forest | 75 / 41 | Native conditioning, all eight ordered latent hashes and decoded float32 pixel hash match exactly |

Both PNGs were visually inspected. They are `.local/native-dynamic-balanced-20261017/dynamic512.png` and `dynamic384.png`. The [numerical record](evidence/krea-dynamic-numerical-parity-2026-10-17.json) preserves inputs and actual/expected hashes, linked to the [512 reference](evidence/krea-dynamic512-independent-reference-2026-10-17.json) and [384 reference](evidence/krea-dynamic384-independent-reference-2026-10-17.json). The repeated 512 reference reproduced the initial attempt's exact output. Independent source files and executable bytes remained unchanged, and the numerical run's library source hashes matched the library at that checkpoint. The subsequent function-lifetime change is qualified separately below.

Encoder cancellation was observed during the native encoder call. The revised balanced policy closed all four preceding denoiser sessions before acquiring the encoder. Cancellation returned after native return and encoder close, no denoising began, and session state returned to ready. A fresh native re-encoding in the same generation session then reproduced conditioning, all eight latent hashes and pixels exactly.

Decoder cancellation has two passing native checks, supplemented by a controlled generation-pipeline regression:

- The [isolated native decoder](evidence/krea-isolated-decoder-cancellation-2026-10-17.json) consumed the independent 384×512 reference's exact final latents. Its initial and post-cancellation pixels matched that independent reference exactly. Cancellation was requested while the awaited native call remained active; return preceded cancellation delivery, and the same decoder session recovered and unloaded successfully.
- The [integrated native decoder check](evidence/krea-native-pipeline-decoder-cancellation-2026-10-17.json) used one sampling step with four real dynamic denoisers plus the decoder retained in one generation session. Cancellation occurred during native decoding, an overlapping request was rejected, no image was emitted, and state returned to ready. Recovery reused all five model sessions and reproduced the baseline latent/pixel values exactly. All five sessions closed once.
- `decoderCancellationRestoresPipelineAndCanRecover` checks generation-state recovery and suppression of the decoded event while a controlled decoder is suspended. The final [test log](evidence/krea-native-pipeline-decoder-tests-2026-10-17.log) passes 35 CPU tests and this native suite: 36 passed, with the other three native suites intentionally skipped to serialize GPU work. The isolated native suite also [passed separately](evidence/krea-isolated-decoder-tests-2026-10-17.log).

**Limit at that checkpoint:** the combined full-dynamic suite timed out after ten minutes waiting to observe the decoder call on a second eight-step generation with retained denoisers. A stack sample during the wait was in MPSGraph GPU residency. Both initial eight-step parity cases and encoder recovery had passed, but decoder cancellation was not recorded in that full-schedule run. Its [failure status](evidence/krea-integrated-decoder-status-2026-10-17.json), [partial trace](evidence/krea-integrated-decoder-incomplete-2026-10-17.json), [test log](evidence/krea-integrated-decoder-tests-2026-10-17.log) and [sample](evidence/krea-dynamic-decoder-wait-2026-10-17.sample.txt) are preserved. The timed-out process cleaned up and exited before an operator stop command arrived; it was not a passing suite or an operator-terminated decoder call.

Repeated full-schedule execution with retained models therefore remained unqualified at that checkpoint. The next investigation targeted host contention and native resource lifetime. Passing isolated and one-step decoder checks did not resolve that limit. No performance result is claimed from any of these shared-host runs.

The [original slice evidence](evidence/krea-generation-2026-10-17.json) preserves the earlier cache failure and pre-native status; the follow-up manifest supersedes those status fields. The independent Rust/lab implementation is unchanged. The repository's initial commit is `1e7048d`; this slice and validation are subsequent working-tree changes.

The [focused GPU retry](evidence/krea-gpu-retry-2026-10-17-status.json) used `AETHER_NATIVE_DECODER_PIPELINE_STEPS=8` at 384×512, seed 17017, with retained 20-row fixture conditioning. Its first generation completed all 33 native calls. The repeated generation also completed all eight sampling steps, reached the native decoder, rejected overlap, cancelled without emitting an image and returned to ready after native return. Recovery then remained incomplete for more than ten minutes; operator termination was requested for only the test helper. The raw report is preserved at its last recovery checkpoint, rather than rewritten as a test failure or pass. Stack samples during repetition and recovery show GPU-residency work; roughly 9.5 GiB of swap was in use, and quiet conditions were not verified. This establishes eight-step decoder cancellation for that profile, but does not qualify recovery or rerun the original 41-row full-dynamic case. No library or test code changed for this retry. That prompted the function-lifetime investigation with live per-call checkpoints below.


## Native function lifetime and recovery (2026-10-18)

The [current qualification](evidence/krea-function-reuse-qualification-2026-10-18.json) supersedes the earlier full-schedule recovery limit for the tested cases. `CoreAIBackend` now resolves each requested native function lazily once per model session, reuses it across calls, and releases its ownership with the backend at unload. Invalid inputs and stateful functions are still rejected before resolution; failed function loads are not cached. A synchronized cache keeps backend copies consistent. No public API changed.

Live journals reproduced the previous policy's slowdown inside native denoising before cancellation was requested, rather than locating it solely in cancellation state handling. The bounded per-call control was stopped after 43 completed calls; its last recorded awaited call was denoiser2 during the repeat. Its [journal](evidence/krea-function-reuse-control-2026-10-18.jsonl), [stack sample](evidence/krea-function-reuse-control-2026-10-18.sample.txt), [memory summary](evidence/krea-function-reuse-control-2026-10-18.vmmap.txt) and [stop record](evidence/krea-function-reuse-control-2026-10-18.stop.json) are preserved. An exploratory session-retention arm completed the same 384×512 workload and reproduced the control baseline exactly. This supports changing application-level function lifetime; it does not isolate the framework allocator/driver mechanism. Memory snapshots and event timestamps are diagnostic, with uncontrolled host/cache conditions, and establish no benchmark result.

The final lazy implementation passed these serial native runs, each alongside all 35 CPU tests:

| Run | Native checks passed | Native calls completed | Functions loaded / reused | Model sessions opened and closed |
| --- | ---: | ---: | ---: | ---: |
| [Independent full dynamic suite](evidence/krea-function-reuse-full-2026-10-18.json) | 39 | 169 | 18 / 151 | 18 |
| [384×512, eight steps, retained recovery](evidence/krea-function-reuse-retained-2026-10-18.json) | 10 | 99 | 5 / 94 | 5 |
| [128×128, eight steps, reopened recovery](evidence/krea-function-reuse-reopened-2026-10-18.json) | 11 | 99 | 10 / 89 | 10 |
| [Existing shape-switch and active-unload suite](evidence/krea-function-reuse-lifecycle-2026-10-18.json) | 23 | 38 | 15 / 23 | 15 |

The full suite matched independent conditioning, all eight latent hashes and float32 pixels at both 512×512 (11 conditioning rows) and 384×512 (41 rows), seed 17017. Encoder cancellation recovered through fresh encoding and all eight steps. The previously blocked repeated 384×512 decoder cancellation and full recovery now pass against the independent reference. Both original pixel hashes remain unchanged. The independent executable and source files were unchanged.

The retained/reopened focused probes use the same 20-row fixture conditioning and compare with their own pre-cancellation baseline; the 128×128 case is a lifecycle reproducer, not image-quality acceptance. Their journals exactly match their final report events, and the tests assert one function load per model session. The first 128×128 attempt missed a roughly 10 ms decode at the prior observation interval; the [missed observation](evidence/krea-function-reuse-observation-miss-2026-10-18.json) is retained, and subsequent probes observe real in-flight cancellation at 1 ms without gating native execution.

The existing lifecycle suite also verified cached decoder reuse across three image sizes and unload during active work. All seven of its latent/pixel outputs match the pre-change balanced-policy report. The affine smoke passed CPU-only, automatic and GPU preferences, including repeated calls and output validity after unload. Each final Swift test run registered 39 tests: 36 passed and the other three native suites were intentionally skipped to serialize GPU work. Reports, complete logs, source identities, artifact digests and smoke results are linked from the current qualification. Broader prompts/seeds/devices, resident-encoder reuse and independent GPU queue-drain observation remain outside these claims.

The opt-in [benchmark workflow](krea-benchmarks.md) uses these qualified dynamic references for correctness while separating warm-up, measured iteration timings and memory counters. The [first user-run baseline](evidence/krea-benchmark-baseline-2026-10-18.json) now records performance observations with exact outputs; the 512×512 case still shows conditioning warm-up drift.

The subsequent [matched encoded/provided benchmark](evidence/krea-benchmark-comparison-2026-10-18.json) passes all 32 image checks with matching provenance. Reusing conditioning and generation models gives medians of 16.72 s at 512×512 and 13.26 s at 384×512, versus 38.09 s and 25.70 s when encoding each request. One-time conditioning preparation (28.49 s and 14.93 s) and warm-ups are excluded from provided medians. The first sampling step approaches later-step latency with retention; later steps remain similar across profiles. See the [benchmark analysis](krea-benchmarks.md#matched-comparison-2026-10-18) for profile order, variability, thermal observations and timing scope.

## Prepared-prompt native qualification: 2026-10-18

With the user-confirmed quiet GPU, the extended dynamic suite passed all 65 checks. The Release run passed 54 CPU tests and one native suite (58 registered, three other native suites skipped). All six dynamic model streams were verified, and the source hashes remained unchanged throughout the run.

At both 512×512 and 384×512, eight-step generations with seed 17017 matched fresh independent conditioning, every latent step and decoded pixels. Fresh reference hashes also match the previously qualified references. Each shape completed ordinary generation, cancellation recovery, preparation and two prepared-prompt repeats: eight completed library images in total, including four prepared images. Encoder cancellation with fresh re-encoding and decoder cancellation with full-schedule recovery both passed.

Preparation matched the independent conditioning and released its encoder. Each prepared generation made 33 native calls without encoding. The second repeat at each shape reused all generation models and all 33 native function invocations without model or function loads during generation. Terminal unload follows the second repeat under the same trace phase label; its close events occur after that generation's final returned call.

Across the suite, 303 native begin/end pairs completed and all 28 native model sessions closed exactly once, after their submitted calls returned. Native functions loaded 28 times and were reused 275 times. Preparation-specific cancellation, independent prepared-value ownership and seed variation retain their CPU coverage; broader native devices, seeds and failure cases are outside this run.

The [qualification summary](evidence/krea-prepared-prompt-qualification-2026-10-18.json) pins source, executable and artifact hashes. The [full native report](evidence/krea-prepared-prompt-native-2026-10-18.json) preserves all checks and boundary events; the [test log](evidence/krea-prepared-prompt-native-tests-2026-10-18.log) records the passing suite. Fresh [512×512](evidence/krea-prepared-prompt-reference-dynamic512-2026-10-18.json) and [384×512](evidence/krea-prepared-prompt-reference-dynamic384-2026-10-18.json) independent records are archived alongside their logs. The native test elapsed 510.303 seconds including verification, independent references, diagnostics and cancellation; this is not a prepared-prompt latency benchmark. The earlier readiness record remains a historical CPU checkpoint.

## Batch CLI validation: 2026-10-18

The [batch validation record](evidence/krea-batch-render-validation-2026-10-18.json) records 60 passing CPU tests (64 registered, four opt-in native suites skipped). Six new tests cover argument bounds, ordered/duplicate seeds, one-time preparation, output hashes, refusal to reuse directories, setup/generation/save errors, partial results and cancellation cleanup. Cancellation waits for submitted generation to return before unload and prevents publication of its cancelled image.

The real CLI completed three eight-step 512×512 images for `a red fox in fresh snow`, using seeds `17017,17018,17017`. Conditioning and both seed-17017 pixel hashes match the independent reference. Their PNG bytes are also identical; seed 17018 produces different pixels, without a new independent reference claim for that seed. All three PNG dimensions and byte hashes were verified, and stdout JSON matches the saved report. A second invocation against the existing directory exits nonzero without modifying the report; malformed seeds fail before creating an output directory.

The [native batch report](evidence/krea-batch-render-native-2026-10-18.json), [progress log](evidence/krea-batch-render-native-2026-10-18.log), [CPU test log](evidence/krea-batch-render-tests-2026-10-18.log) and [source/executable provenance](evidence/krea-batch-render-native-provenance-2026-10-18.json) are preserved. This is a CLI correctness check at one shape, without specialization; the preceding library qualification separately covers native prepared reuse at both dynamic shapes. Rendering timings have no warm-up or controlled performance protocol and are not benchmark results.
