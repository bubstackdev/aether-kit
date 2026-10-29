# MiniMax H3 integration

**Status (0.0.3): Preview.** The H3 products left the experimental label on 2026-10-28, once repeated native runs, the
user's subjective audio and video review, memory, cancellation and timings were accepted
([release checklist](qualification-and-release.md), [0.0.3 release notes](release-notes/0.0.3.md#h3-promotion-to-preview)).
They are qualified on one profile and one Mac (`5af39db5…`, Turbo8, M5 Max); Turbo4 stays outside measured admission.
The sections below are dated records and keep the wording of their time.

The `AetherH3` SwiftPM product (experimental until 0.0.3) starts with CPU preparation contracts for the corrected Turbo8 text-to-video/audio path. It depends on `AetherKit`, with no Core AI, Metal, tokenizer or model-weight dependency. This slice performs no native inference, model loading, asset download or checkpoint conversion.

## Status: 2026-10-26

**The 2026-10-18 records below are historical.** They cover block 0, lifecycle, the three-block chain, the fifty-block backbone, the outer phases and the eight-step provided-conditioning trajectory. Each describes exactly what ran against its pinned prototype bundles. Those bundles, oracles and the preparation cache lived under `/private/tmp`, and macOS temporary-file cleanup removed them. Only empty directory skeletons remain. The commands below that cite `/private/tmp/aether-h3-*` can no longer run. The assets were not regenerated: Core AI exports aren't byte-reproducible, so every pinned capture would have to be redone anyway. [The lost-asset record](evidence/h3-legacy-assets-lost-2026-10-26.json) lists the affected roots, the 64 citing records, the original identities and the regeneration route, should it ever be needed.

**H3 work for 0.0.2 qualifies the schema-3 profile path instead.** Its CPU oracle and export tooling (`prepare-h3-profile-oracle.py`, `export-h3-profile.py`, `h3_export_lib.py`) run in Forge's H3 oracle environment, `uv sync --frozen --project Forge/h3-oracle`, then `Forge/h3-oracle/.venv/bin/python -I -B Scripts/...`. It is pinned to exactly the environment that produced the 2026-10-26 evidence (torch 2.13.0, transformers 5.20.1, peft 0.21.2, diffusers 0.41.0.dev0 at public commit `7ecb1e1b`); re-running the FL2VA keyframe condition in it reproduces the recorded hash ([record](evidence/forge-environment-2026-10-27.json)). That path is `Scripts/export-h3-profile.py`, `H3Profile`, `H3ProfileEngine.stages` and `H3VideoSession`. It uses none of the lost assets. Its complete profile is referenced in place under Atelier's team staging folder, `~/Code/atelier/.local/team/staging/h3/exports/h3-fl2va`. Weights are read as byte ranges of the pinned checkpoint snapshot and Turbo adapter. [The profile asset record](evidence/h3-profile-assets-2026-10-26.json) covers several checks:

- both bundle hashes;
- every model stream hash;
- the adapter hash, which matches the pinned `e16ac208…`;
- that no referenced path is missing or under a temporary directory;
- the passing opt-in `AETHER_H3_PROFILE`/`AETHER_H3_PROFILE_PARAMETERS` test, which hashes every table, model stream and in-place weight binding.

aether-kit does not own that folder: moving or deleting it invalidates the pins.

**Kit-owned home (2026-10-27).** The referenced staging files (17.6 GiB, 1,266 files) are now APFS clones under `~/.cache/aether-kit/h3/profile-c49d8dbf/`, with the tiled first-frame encoder overlay beside them; the checkpoint and adapter blobs stay in the Hugging Face cache. The new `bundle.json` differs from the old one only in two path prefixes, and every content hash is unchanged, but Kit identifies a profile by its manifest bytes and requires absolute paths, so the profile identity is now `5af39db5…` (was `c49d8dbf…`). Every cloned file verifies against its recorded hash, regenerated dumps match the gate dumps tensor for tensor, and a replay of the qualified T2VA capture under the new identity reproduces all 457 boundary hashes ([record](evidence/h3-profile-home-2026-10-27.json)). The 2026-10-28 memory gate measures the new identity, and the bundled measured entry is keyed to it. Paths relative to `bundle.json`, which would make the identity independent of location, are a promotion follow-up.

## Available contracts

- `H3Geometry` validates positive multiples of 32 within the prototype's 1,032,192-pixel budget, and explicitly snaps requested frames upward to `17n+5` at 24 fps. The default 120-frame request resolves to 124 frames, 37 latent frames, 18,870 target video rows and 207 audio latents per stereo channel at 960×544. Production geometry requires a snapped duration within 5–15 seconds; requests whose next grid point exceeds 360 frames fail. `diagnostic: true` allows shorter geometry without asserting native support.
- `H3Layout` computes text, optional condition-video, stereo-audio and target-video row positions, tags and indices. Its per-step plans preserve the actual unique timestep cardinality/order and AdaLN mapping. Keyframe row layout is available; image encoding and keyframe generation are not implemented.
- `H3Geometry.packVideo` preserves the source `[24,T,H,W]` to `[T*H/2*W/2,96]` patch ordering using owned float32 tensors.
- `H3Schedule` uses the captured eight-evaluation Turbo8 video/audio 12/3 grid. Its data-ward x0 blend preserves separately rounded float32 products. It does not substitute Krea's Euler update or regenerate the grid with a different linspace. Base50 and custom schedules are not exposed.
- `H3ExportHeader` inspects only identity fields in the prototype JSON manifest: schema/completeness declarations, pinned model snapshot and Diffusers revision, corrected per-head QKV marker, and exact dynamic-LoRA Turbo8 adapter identity. It hashes the supplied manifest bytes. It does not open referenced files, verify weights, validate tensor bindings or establish that declared conversion/completeness claims are true. A complete asset validator remains future work.

```swift
import AetherH3

let geometry = try H3Geometry(width: 960, height: 544, requestedFrames: 120)
// textTags must come from the eventual H3 processor; these are illustrative.
let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 49))
let schedule = try H3Schedule()
let first = layout.timesteps(for: schedule.steps[0])
print(geometry.frames, layout.rows, first.values.count) // 124, 19333, 1
let second = layout.timesteps(for: schedule.steps[1])
print(second.values.count) // 2, not the synthetic three-timestep lab plan
```

The existing prototype uses model snapshot `42ed227ee7df40d41602854ae760620d6eb651fe`, Diffusers commit `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941` and layout marker `raw_per_head_qkv_to_contiguous_qkv_v1`. Turbo8 requires the pinned `lightx2v/Minimax-h3-Turbo` dynamic adapter at revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`, SHA-256 `e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f`, rank 128 and alpha 8. Merged adapters and earlier layouts are rejected by header inspection.

## CPU validation

[Readiness evidence](evidence/h3-cpu-readiness-2026-10-18.json) records 69 passing CPU tests (73 registered, four native suites disabled), including nine H3 tests. Every one of the eight video and audio scheduler-update hashes matches the retained independent upstream CPU oracle. Video packing matches the original upstream patchify hash. All coordinates, token tags and video/audio/text index bytes match the captured original pipeline for text-only, first-keyframe, last-keyframe and both-keyframe layouts. Position fixtures are narrowed once from source float64 to float32, while indices/tags are narrowed from int64 to int32. These checked-in fixtures are small metadata/numeric records, not weights or inference outputs from this Swift pipeline.

The original prototype's CPU QKV sentinel regression was rerun unchanged and passed for body and text-refiner attention. It uses distinct head/projection sentinel rows and demonstrates why calling the state-dict converter directly on raw interleaved checkpoint rows is incorrect. No checkpoint conversion was run here. The Swift header tests reject stale/missing layout markers and incompatible adapter fields, and accept identity fields extracted from the retained corrected export.

Keyframe packing captures used stub encoder features to inspect layout; those tests do not establish keyframe encoding or fidelity. Scheduler fixtures use supplied finite vectors/velocities; they do not qualify a full denoising trajectory. The earlier corrected Rust pilot remains separate [prototype evidence](prototype-evidence.md#h3-evidence); its native timings and quality acceptance do not qualify this new Swift target.

## Next implementation boundary

The native block binding checkpoint below adds selected-file verification and a BF16-preserving adapter; full-pipeline asset validation and execution remain future work. H3's source-precision backbone and direct shared Metal weight binding must not be silently routed through Krea's fp16 host-copy executor. Port the pinned conditioner presentation and unnormalized 50-layer tap, outer projections/refinement, timestep/modulation preparation, complete 50-block evaluations, heads, both VAEs and temporal/spatial stitching in independently testable stages. The retained lab's explicit-weight backbone is useful evidence for that work, not a complete selected end-to-end implementation.

Native probes, model exports, full renders and benchmarks require a separately available GPU window. No automatic native H3 workload is attached to `Scripts/check.sh`. GPU time becoming available does not itself launch inference. Full asset verification, cancellation/recovery, repeated generation, memory limits, media audit and subjective audio/video acceptance remain necessary before promoting any H3 profile.

## Native block binding checkpoint

`AetherH3CoreAI` now contains a compile-tested, experimental fixed-input block session. The first [GPU block probe](evidence/h3-block-qualification-2026-10-18.json) now passes both captured timestep cases and exact full-output repeats. This is a qualification adapter for corrected block 0, not the full H3 generator.

`H3BlockProbePlan` ties the two captured body cases to a corrected Turbo8 bundle, the pinned scheduler, both actual modulation-table cardinalities, all 22 block-0 base/adapter bindings, rotary identities and selected independent CPU reference rows. It checks declarations for all 50 blocks but verifies bytes only for the selected models, inputs and references. The manifests are trusted local data and may name external absolute paths; this is not a confined asset-download format. Files must remain immutable throughout preparation and use.

`H3TensorSpec`, `H3TensorData` and `H3TensorFile` preserve little-endian BF16/F32/int32 bytes with checked extents, finite floating-point values and SHA-256 identities. BF16 widens exactly for numerical comparison and is never narrowed to fp16. Tensors are limited to 1 GiB each; the selected input/output buffers together must fit the 2 GiB admission limit. These are byte-admission limits, not process/GPU memory guarantees. Modulation indices are range-checked before native acquisition.

The native adapter validates descriptors before allocating inputs, retains one function and immutable directly owned shared Metal input buffers, and uses a distinct output buffer. Calls are serialized; cancellation keeps the operation slot until submitted stream work returns. Terminal unload waits for active work and releases backend ownership. Encode errors also join the stream before releasing buffers. The CPU tests exercise these lifecycle rules with controlled backends. The explicit lifecycle command below checks submitted-work cancellation and bounded memory stability; long-run retention and full generation remain separate gates.

The installed SDK's opaque `ComputeStream` is not `Sendable`. The compatibility import is restricted to the native adapter, with actor-owned resources and the session's one-call admission boundary. Swift still emits two concurrency diagnostics at stream completion under that import. This interop boundary remains experimental; no unchecked `Sendable` conformance or process-global native state was added.

In 0.0.3 those diagnostics are gone. Every native adapter, and the language sessions, now hold their stream in AetherCoreAI's `SerialComputeStream`, which only encodes and joins. It is the single `@unchecked Sendable` boundary, resting on the same one-call admission, and it is a workaround for the SDK's missing annotation: it goes once Core AI annotates the stream. The compatibility imports were removed.

### CPU preparation

The retained corrected local exports are available on this machine:

```sh
.build/arm64-apple-macosx/release/aether-probe qualify-h3 \
  /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  "$PWD/.local/h3-block-preparation-new" --prepare-only
```

This reads roughly 2 GB across the selected input sets, hashes model streams and tensors, validates finite values/tags and verifies reference files. It does not acquire a Core AI model, create a Metal device/buffer or execute inference. A passing report says `stage: ready`, `selectedFilesVerified: true`, `nativeExecuted: false`, with no native runs or resource snapshots. Existing output directories are rejected. The two manifests are rechecked before publishing readiness.

[Binding readiness evidence](evidence/h3-binding-readiness-2026-10-18.json) records the successful real prepare-only check and the expanded CPU suite. Metadata fixtures rewrite external paths into a local test namespace; ordinary tests do not need model weights. The earlier CPU-only readiness record remains a historical checkpoint.

### GPU block probe

After an explicit available GPU window, the next command is:

```sh
.build/arm64-apple-macosx/release/aether-probe qualify-h3 \
  /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  "$PWD/.local/h3-block-native-02" --run
```

It verifies the selected files again, then serially loads each of the two body graphs with corrected block-0 weights and runs it twice. All output values must be finite. The 64 captured rows per case must pass the existing predefined 0.02 relative-L2 bound against independent CPU references; complete output bytes must repeat exactly. Reports retain failures and completed comparisons, plus before-load/after-run/after-unload physical-footprint, resident and Metal counters. These counters overlap; they are boundary observations, not allocation peaks. Cases unload before the next graph is acquired. Final publication rechecks both manifest identities.

The first GPU check should be run as an owned, monitored process with a five-minute stop/join budget; the CLI does not itself enforce a timeout. This probe does not test native cancellation, all 50 blocks, unsampled numerical rows, the complete schedule, conditioner, decoders, audio/video quality or performance. The first numerical/repeat boundary passes; the separate lifecycle probe below supplies a subsequent bounded gate. No job is scheduled to start when the estimated GPU-busy interval ends.

## First GPU block qualification: 2026-10-18

The [native qualification record](evidence/h3-block-qualification-2026-10-18.json) passes four calls: corrected block 0 at the two captured T2VA timestep cardinalities, twice each. The independent 64-row CPU comparisons have relative L2 0.0110056 and 0.00430384 (about 1.10% and 0.43%), both below the existing predefined 0.02 limit. CPU/native bytes differ. Full outputs are finite, and complete output hashes repeat exactly within each case. This is a bounded module gate, not complete H3 generation or full-output numerical parity.

Both terminal session closes completed. Metal allocations fell from approximately 11.21 GiB after each case's runs to 1.03 GiB immediately after unload. Physical footprint remained approximately 11.81 GiB at those unload snapshots; resident, footprint and Metal counters overlap. Neither page reclamation nor long-run memory stability is established. The first and second post-unload Metal observations match, but two observations do not qualify a retention policy. The subsequent bounded lifecycle probe below addresses this before scaling to additional blocks.

The [owned-process controller](evidence/h3-block-controller-2026-10-18.json) records exit 0 and a joined process in 7.1154 seconds, within its 300-second budget and without timeout. This time includes CPU verification/setup and is not a benchmark. Source/executable/manifest hashes match the prepared checkpoint. The [raw report](evidence/h3-block-native-2026-10-18.json) preserves comparisons and memory observations. Existing CPU readiness and compiler interop notes remain separate historical evidence. The lifecycle probe below is a subsequent experiment.


## Bounded block lifecycle probe

`qualify-h3-lifecycle` reuses the selected-file and descriptor gates, with explicit `--prepare-only` and `--run` modes and a new output directory. Ordinary CPU tests do not launch this workload. Preparation verifies files without acquiring a Metal device or native model. For an authorized GPU window:

```sh
.build/arm64-apple-macosx/release/aether-probe qualify-h3-lifecycle \
  /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  "$PWD/.local/h3-lifecycle-native-new" --run
```

The fixed plan opens each cardinality four times and checks two normal calls per session. Cycle 2 cancels submitted work and checks recovery in the retained session. Cycle 3 cancels submitted work while concurrently unloading; cycle 4 checks reopened recovery. All 18 successful outputs must pass the existing sampled CPU numerical gate and the full-output SHA-256 from the first block qualification. Four cancelled calls bring the total to 22 native submissions.

The synchronous diagnostic observer records successful encode submission, joined stream completion, operation-slot release and terminal unload. Cancellation is requested directly from the submission callback, before entering the stream-completion await. The probe requires completion before slot release and unload completion; concurrent unload must start while submitted work is still pending. A missed boundary fails qualification. This checks cancellation of submitted work and draining ownership; it does not demonstrate GPU preemption or identify the instant hardware execution ends. No artificial delay or GPU gate is inserted.

Memory is sampled every 20 ms plus before/after load, after calls and at 0, 1 and 5 seconds after unload. The declared sampled bounds are 20 GiB physical footprint (including the process lifetime peak) and 18 GiB Metal allocations. After the warmup cycles, each cardinality's cycle-3 and cycle-4 five-second footprint, resident and Metal observations may grow by at most 512 MiB relative to cycle 2. These are a short experiment's acceptance limits, not product memory guarantees. Counters overlap and cannot be added. Samples can miss allocation peaks; only the OS lifetime counters capture their respective process peaks. Violating observations and completed comparisons are persisted before failure. The command needs an external owned-process timeout, as in the first block check.


### Lifecycle qualification: 2026-10-18

The [qualification record](evidence/h3-lifecycle-qualification-2026-10-18.json) passes after changing the native backend to retain one `ComputeStream` per session. Calls remain serialized and joined before stream reuse, including cancellation/error paths. No delay was added between retained-session calls, and the declared limits were unchanged.

The initial [fresh-stream control](evidence/h3-lifecycle-fresh-stream-native-2026-10-18.json) stopped at its predefined memory limit after eight submissions: seven numerically passing outputs and one correctly drained cancellation. Its sampled physical-footprint peak reached approximately 21.00 GiB and Metal reached 20.25 GiB. The [source change](evidence/h3-lifecycle-stream-reuse-2026-10-18.diff) and pinned control/fixed source manifests show that only the backend source and rebuilt executable differ among shared pins. Keeping one stream removed the observed growth in this experiment; this does not identify undocumented Core AI allocator behavior.

The fixed [raw report](evidence/h3-lifecycle-native-2026-10-18.json) passes all 22 native submissions across eight session opens. All 18 successful outputs match the earlier complete-output hashes, and sampled CPU relative L2 remains 0.0110056/0.00430384, below 0.02. Four cancellation traces observe submission, cancellation request, joined completion and then slot release. Both concurrent unloads start before joined completion and finish afterward. Recovery passes in retained sessions and after reopening.

The OS lifetime physical-footprint peak is 7.4562 GiB; sampled Metal allocation peaks at 6.6961 GiB. After unload, physical footprint returns near 0.59 GiB by the one-second samples; by five seconds Metal is 720,896 bytes in every case. Cycle 4 versus cycle 2 five-second footprint growth is 81,920 bytes for body-0 and zero for body-1; Metal growth is zero, and resident counters decrease. All observations pass the unchanged limits. Delayed reclamation is observed here, while long-running retention and full-pipeline memory behavior remain unqualified.

The [controller](evidence/h3-lifecycle-controller-2026-10-18.json) records exit 0, joined ownership, unchanged source pins and 84.5343 seconds within its 300-second budget; this elapsed time is not a benchmark. [Release CPU tests](evidence/h3-lifecycle-tests-2026-10-18.log) pass 85 tests (89 registered, four native suites disabled). The existing two [SDK stream interop diagnostics](evidence/h3-lifecycle-compiler-interop-2026-10-18.log) remain explicit. The chain probe below is a subsequent experiment.

This lifecycle result enabled the bounded three-block chain below, with verified weights and independent per-block references. Full 50-block evaluations, outer preparation/projections, the complete Turbo8 trajectory, conditioner, decoders and audio/video acceptance remain outstanding.


## Bounded three-block chain

`H3ChainProbePlan` and the experimental `H3ChainSession` execute corrected blocks 0, 1 and 2 at the real 19,333-row geometry, separately for both captured T2VA timestep cardinalities. The plan inherits the block-0 bundle/header/scheduler/descriptor gates and selects three distinct sets of 22 source-precision base/adapter bindings. Block-specific modulation tables preserve each case's cardinality. All selected files are hashed and floating values validated before native acquisition. The directly owned Metal buffer admission bound is 4 GiB (about 3.03 GB of declared bindings per case); this is separate from native runtime allocations.

The session retains one model, function and compute stream. An immutable initial hidden buffer restarts every call. Two distinct output buffers alternate through the three blocks, so a block consumes its predecessor's GPU output without host conversion or in-place aliasing. Each step joins the stream before the next step can reuse storage. The API returns three owned BF16 CPU snapshots for qualification. Cancellation after block 1 submission drains that block and prevents block 2 submission; terminal unload retains ownership until submitted work returns.

### Independent references and scope

There was no retained independent CPU reference for this corrected multi-block chain. `Scripts/prepare-h3-chain-oracle.py` calls the pinned original Diffusers modules, canonical raw-shard converter and dynamic PEFT adapter on CPU. It checks all 66 exported weight tensors against independently loaded original parameters, prepares the three original modulation tables per cardinality, and verifies that its sampled-query instrumentation is byte-identical to full CPU execution on 128 rows for all six block/case combinations. All source projects, checkpoints and exports remain read-only; outputs go into a new local directory. No graph export, checkpoint conversion output or GPU API is invoked by the helper.

For the production geometry, the independent CPU calculation uses the complete native input context for each block, retaining every attention key/value and comparing 64 selected output rows. The block-0 context is the original preparation fixture; blocks 1 and 2 use captured predecessor outputs. The final qualification pins complete output hashes from that capture, ensuring the CPU references are applied to exactly the contexts used to compute them. This establishes independent per-block checks and native chain continuity. It does not establish an independently computed CPU chain trajectory, unsampled CPU numerical parity, all 50 blocks or a complete Turbo8 render.

### Explicit capture and qualification

The command requires explicit modes and new output directories:

```sh
aether-probe qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY --prepare-only
aether-probe qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY --capture
aether-probe qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY --run
```

Preparation performs no native acquisition. It reports whether all per-block CPU references and captured output hashes are present. `--capture` runs six native submissions, verifies block 0 against its prior CPU gate and exact full-output hash, saves finite intermediate outputs, and ends with `captureComplete: true`, `passed: false`, `stage: capturedAwaitingCPUReferences`. Capture is explicitly unqualified. `--run` rejects missing CPU references or expected output hashes before acquiring native resources.

**Legacy, prototype-only.** The chain, backbone and denoising CPU oracles (`prepare-h3-chain-oracle.py`, `prepare-h3-backbone-oracle.py`, `prepare-h3-denoising-oracle.py`) load the superseded aether-prototype's forge and the lab's vendored Diffusers from `--prototype` and `--lab`. They run in neither Forge environment, and their `/private/tmp` inputs are gone. The commands below are the record of the 2026-10-18 workflow, not a supported one:

```sh
H3_PYTHON=<the 2026-10-18 prototype forge interpreter; not provided>
H3_LAB=<minimax-h3-lab checkout>
H3_PROTOTYPE=<aether-prototype checkout>
H3_BUNDLE=/private/tmp/aether-h3-qkv-ready/render/bundle.json
H3_PHASES=/private/tmp/aether-h3-qkv-ready/phase/bundle.json
H3_PROBE="$PWD/.build/arm64-apple-macosx/release/aether-probe"

PYTHONDONTWRITEBYTECODE=1 "$H3_PYTHON" -B Scripts/prepare-h3-chain-oracle.py \
  --lab "$H3_LAB" --prototype "$H3_PROTOTYPE" --bundle "$H3_BUNDLE" --phases "$H3_PHASES" \
  --output "$PWD/.local/h3-chain-cpu-new" --prepare
"$H3_PROBE" qualify-h3-chain "$H3_BUNDLE" "$H3_PHASES" \
  "$PWD/.local/h3-chain-cpu-new/chain.json" "$PWD/.local/h3-chain-capture-prepared-new" --prepare-only
# Execute the next GPU command under an owned 300-second stop/join controller.
"$H3_PROBE" qualify-h3-chain "$H3_BUNDLE" "$H3_PHASES" \
  "$PWD/.local/h3-chain-cpu-new/chain.json" "$PWD/.local/h3-chain-capture-new" --capture
PYTHONDONTWRITEBYTECODE=1 "$H3_PYTHON" -B Scripts/prepare-h3-chain-oracle.py \
  --lab "$H3_LAB" --prototype "$H3_PROTOTYPE" --bundle "$H3_BUNDLE" --phases "$H3_PHASES" \
  --output "$PWD/.local/h3-chain-references-new" --references "$PWD/.local/h3-chain-capture-new"
"$H3_PROBE" qualify-h3-chain "$H3_BUNDLE" "$H3_PHASES" \
  "$PWD/.local/h3-chain-references-new/chain.json" "$PWD/.local/h3-chain-run-prepared-new" --prepare-only
# Execute the next GPU command under the same bounded controller policy.
"$H3_PROBE" qualify-h3-chain "$H3_BUNDLE" "$H3_PHASES" \
  "$PWD/.local/h3-chain-references-new/chain.json" "$PWD/.local/h3-chain-run-new" --run
```

The qualified plan performs 38 native submissions: ten complete chains (30 successful block outputs) and four calls cancelled after the second block (eight submitted blocks). Each cardinality has repeated calls, retained recovery, reopened recovery, cancellation with concurrent unload, and recovery after that close. Every successful block output must pass the unchanged 0.02 sampled CPU relative-L2 bound and its captured complete-output hash. Cancellation traces require block 0 completion, block 1 submission/cancellation/joined completion and then slot release, with no block 2 submission. Concurrent unload must start before joined completion and finish afterward. Cancellation observes submitted work and draining ownership; it does not prove GPU preemption or the exact hardware-completion instant.

The existing 20 GiB process footprint/lifetime-peak, 18 GiB sampled Metal and 512 MiB settled-growth bounds remain unchanged. Memory is sampled every 20 ms plus at load/run boundaries and 0, 1 and 5 seconds after each unload. Cycle-2 and cycle-3 settled samples are compared with cycle 1 for each cardinality. Counters overlap; sampling can miss Metal peaks. The CLI requires an external owned-process timeout. The existing SDK `ComputeStream` Sendable diagnostics remain under confined compatibility imports in both experimental adapters; no unchecked Sendable conformance is added.


### Chain qualification: 2026-10-18

The [qualification record](evidence/h3-chain-qualification-2026-10-18.json) passes corrected blocks 0–2 at both captured production cardinalities. The [six-call capture](evidence/h3-chain-capture-2026-10-18.json) remains explicitly unqualified; the following [independent CPU references](evidence/h3-chain-cpu-references-2026-10-18.json) and [38-submission native qualification](evidence/h3-chain-native-2026-10-18.json) establish the bounded result. Capture and qualification share identical source/executable pins. No extra model weights or graphs were exported or copied; intermediate capture tensors stay in `.local` and are not checked into the repository.

All 30 successful block outputs across ten chains pass the existing independent CPU 0.02 relative-L2 gate and match complete capture hashes exactly. The per-block relative errors are:

| Captured timestep case | Block 0 | Block 1 | Block 2 |
| --- | ---: | ---: | ---: |
| First cardinality | 1.1006% | 0.1288% | 0.0492% |
| Second cardinality | 0.4304% | 0.1585% | 0.0893% |

All four cancellation traces observe blocks 0 and 1 submitted, then block 1 joined before slot release, with no block 2 submission. Both concurrent unloads start before joined completion and finish afterward. Retained recovery, reopened recovery and recovery after concurrent close preserve all three hashes for both cardinalities. These are per-block CPU comparisons at identical captured contexts, not independently computed CPU chain parity.

The process lifetime physical-footprint peak is 9.3072 GiB; sampled Metal peaks at 8.3755 GiB. Five-second post-unload footprint settles near 0.79 GiB, with 720,896 Metal bytes in every settled sample. Cycle-3 versus cycle-1 footprint growth is 2,342,912 bytes for body-0 and 131,072 bytes for body-1, below the unchanged 512 MiB bound; Metal growth is zero. Counters overlap, and this short probe does not establish long-run or full-pipeline memory behavior.

[CPU preparation](evidence/h3-chain-cpu-preparation-2026-10-18.json) records all 66 independently verified exported bindings and six exact full-versus-sampled CPU method checks at 128 rows. [Release tests](evidence/h3-chain-tests-2026-10-18.log) pass 91 CPU tests (95 registered, four native tests disabled), including stale-source/reference rejection, block ordering/cardinality gates, session ownership and cancellation traces. [SDK interop diagnostics](evidence/h3-chain-compiler-interop-2026-10-18.log) remain explicit for both native adapters.

The [GPU controller](evidence/h3-chain-controller-2026-10-18.json) exits 0 and joins in 125.2723 seconds within its 300-second budget. The separate [CPU reference controller](evidence/h3-chain-cpu-reference-controller-2026-10-18.json) exits 0 and joins within its 600-second CPU-only budget. These elapsed times include verification/setup and settling on a shared machine and are not benchmarks. The fifty-block probe below is a subsequent experiment.

This three-block result enabled the fifty-block gate below. Complete Turbo8 preparation, the full denoising trajectory, conditioning, decoders and audio/video acceptance remain outstanding.


## Fifty-block backbone with bounded weight residency

Schema 2 of `H3ChainProbePlan` admits exactly blocks 0–49 in order, with `weight_residency: one_block` and required CPU numerical checkpoints 0, 1, 2, 24 and 49. Schema 1 remains the resident three-block profile. Every delivered block must be finite; qualification requires a pinned complete-output hash for all 50 blocks, plus independent sampled CPU references at the five checkpoints. This is a backbone execution gate, not all-block CPU numerical parity, an independent CPU trajectory or a complete H3 render.

The native backend owns one set of 22 shared Metal weight buffers. It verifies, copies and hashes each block's immutable source files into those slots only after the preceding stream work has joined. A cancelled or failed reload submits no new work, and the next call reloads block 0. The two alternating hidden/output buffers, immutable initial state, fixed tags/rotary data, all small modulation tables, model, function and retained compute stream are shared across the call. Native bound-buffer accounting uses one resident weight set, independent of the total source-weight bytes. The measured process/GPU limits remain separate from this admission estimate.

`H3ChainSession.run(onOutput:)` delivers one owned CPU block snapshot at a time and awaits the consumer before advancing. The qualifier retains only metadata, numerical metrics and hashes; capture writes only the contexts/checkpoints needed by CPU references (outputs 0, 1, 2, 23, 24, 48 and 49). There is no internal array of fifty full CPU snapshots. Callers can retain delivered data themselves; that caller-owned memory is outside the native buffer bound. The array-returning `run()` remains available for the three-block profile and rejects the fifty-block profile. Consumer errors and cancellation preserve the operation slot until callback/native work has returned.

`Scripts/prepare-h3-backbone-oracle.py` independently checks all 1,100 base/adapter bindings against the original checkpoint/converter/adapter one block at a time. It verifies 100 retained original CPU modulation tables and reuses the prior six qualified CPU references for blocks 0–2; their capture contexts must retain exact hashes. After native capture, it calls the original CPU modules for blocks 24 and 49 at both complete captured input contexts, first validating sampled-query instrumentation against full CPU execution on 128 rows. These add four numerical references without claiming a complete CPU backbone trajectory. The checkpoint, lab, prototype and existing exports remain read-only; no model weights or graphs are copied or exported.

Use `qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY` with the existing explicit `--prepare-only`, `--capture` and `--run` modes. Prepare the schema-2 manifest using the new CPU helper with `--prepare`, then capture under an owned 600-second process budget. Capture executes 100 block submissions and remains explicitly unqualified. Run the CPU helper with `--references CAPTURE_DIRECTORY`, prepare the complete reference manifest, then qualify under an owned 1,200-second budget. The CPU helper also needs `--lab`, `--prototype`, `--bundle`, `--phases`, `--phase-cpu` (the retained original CPU phase checks), `--prior-chain` (the earlier qualified three-block reference manifest), and a new `--output` directory.

The fixed native qualification performs four complete chains and two calls cancelled after block 24 submission: 200 complete block outputs plus 50 cancelled-call submissions, for 250 total. The first cardinality checks retained-session recovery; the second checks cancellation with concurrent unload and reopened recovery. Every delivered block's hash must match capture, and checkpoint CPU errors must remain below the predefined 0.02 relative-L2 bound. Cancellation traces require all submissions/completions through block 24, joined completion before slot release/unload completion, and no block 25 submission. `progress.json` atomically persists completed output metadata during each chain, including failures, without retaining full tensors.

The existing 20 GiB footprint/lifetime-peak, 18 GiB sampled Metal and 512 MiB settled-growth bounds remain unchanged. The first five-second post-unload sample anchors subsequent settled footprint, resident and Metal checks. Sampling and counters retain the limitations described above; the workload is qualification on a shared machine, not a benchmark. CPU tests exercise the full-profile admission/hash/checkpoint requirements, middle-block cancellation traces, callback backpressure, unload and retry after consumer failure. The existing SDK stream interop diagnostics remain explicit.


### Backbone qualification: 2026-10-18

The [qualification record](evidence/h3-backbone-qualification-2026-10-18.json) passes all 50 corrected blocks at both captured production cardinalities with one resident weight set. [CPU preparation](evidence/h3-backbone-cpu-preparation-2026-10-18.json) independently verifies all 1,100 bindings and 100 original CPU modulation tables. The [100-submission capture](evidence/h3-backbone-capture-2026-10-18.json) remains explicitly unqualified; the completed [CPU references](evidence/h3-backbone-cpu-references-2026-10-18.json) and [250-submission qualification](evidence/h3-backbone-native-2026-10-18.json) establish the bounded result. Capture and qualification share identical source/executable pins. Large intermediate tensors remain in `.local`; no model weights or graphs were copied or exported.

All 200 outputs from four complete backbone passes are finite and match capture hashes exactly. Twenty sampled CPU comparisons at blocks 0, 1, 2, 24 and 49 pass the unchanged 0.02 relative-L2 limit. The largest error is 1.1006%; the new middle/final checkpoints reach at most 0.5629%. These references use 64 selected output rows with complete captured native predecessor contexts, not an independent CPU backbone trajectory or CPU numerical checks at every block. Four new 128-row full-versus-sampled CPU method checks pass before creating the middle/final references.

Both midpoint cancellations submit through block 24, join that work before releasing ownership, and prevent block 25 submission. Retained-session recovery, concurrent unload and reopened recovery preserve every block hash. The run opens three sessions; it does not establish long-run retention behavior.

The source weights total 37.13 GiB, while directly owned bound buffers admit about 1.35 GiB per case. Process lifetime physical footprint peaks at 7.7250 GiB and sampled Metal at 6.9076 GiB. Five-second post-unload footprint settles near 0.61 GiB, with 720,896 Metal bytes. Maximum settled footprint growth from the first baseline is 5,144,600 bytes, below the unchanged 512 MiB bound; settled Metal growth is zero. These counters overlap, and 20 ms sampling can miss Metal peaks.

[Release tests](evidence/h3-backbone-tests-2026-10-18.log) pass 95 CPU tests (99 registered, four native tests disabled). [SDK stream interop diagnostics](evidence/h3-backbone-compiler-interop-2026-10-18.log) remain explicit in both experimental native adapters. The [owned GPU controller](evidence/h3-backbone-controller-2026-10-18.json) exits zero, joins without timeout in 542.7543 seconds within its 1,200-second budget, and verifies unchanged source pins. This includes integrity checks, file IO and settling on a shared machine; it is not a benchmark.

The next gate is Turbo8 outer preparation, timestep handling and output heads, followed by the complete eight-step video/audio trajectory. Decoders, conditioning coverage and media acceptance remain outstanding. No GPU inference was launched after the successful backbone qualification.

## Outer-phase binding readiness

`H3PhaseProbePlan` selects five fixed boundaries from the corrected export: input preparation (including the embedded text refiner), timestep embeddings for the first two Turbo8 steps, and video/audio output heads at those two cardinalities. It requires a passing original CPU record tied to the exact bundle hash. Every selected graph, input and complete reference must match that record, including path identity, shape, precision and SHA-256. Preparation output must link to the qualified backbone's initial hidden context; heads use that same fixed context and their corresponding original CPU timestep embedding. This is an isolated module gate, not heads applied to the final backbone output or a linked denoising step.

The plan also checks timestep bytes against `H3Schedule` and every head row index against `H3Layout.timesteps`. In-range indices from the wrong modality are rejected. Prepare preserves BF16 text/hidden and F32 video/audio; time and head embeddings and both velocity outputs remain F32. All selected CPU output references cover complete tensors. The CPU record is trusted local provenance; this gate reuses its retained original-module results rather than rerunning those Python modules.

`H3PhaseSession` owns one model, loaded function, retained stream, immutable input buffers and distinct named output buffers. It binds BF16/F32/int32 shared Metal storage directly and returns owned snapshots for every output. One-call admission remains held through cancellation and joined completion; unload waits for active work, and partial encode errors join before ownership is released. Its confined SDK compatibility import emits two stream Sendable diagnostics, as do the two existing experimental adapters. This remains a manually confined interop boundary, without unchecked Sendable conformance.

The explicit `qualify-h3-phases` command has CPU-only `--prepare-only` and native `--run` modes. The native plan fixes 30 submissions: for each of five cases, two normal calls, a retained-session cancellation and recovery, a cancellation with concurrent unload, then reopened recovery. All 20 successful calls must pass the unchanged 0.02 full-tensor relative-L2 bound (28 named output comparisons) and exact repeat/recovery hashes. Cancellation must observe submitted work and joined completion before slot release. Physical footprint/lifetime peak remain bounded by 20 GiB, sampled Metal by 18 GiB, and five-second post-unload growth by 512 MiB between the two opens of each case. Sampling is every 20 ms; counters overlap and sampled peaks can be missed. These limits and short repetition counts do not establish full-pipeline or long-run retention behavior.

### CPU readiness: 2026-10-18

[Readiness evidence](evidence/h3-phases-readiness-2026-10-18.json) records 105 passing CPU tests (109 registered, four native tests disabled) and a passing real selected-file preparation check. Tests cover CPU identity rejection, linked contexts, wrong timestep/index bytes, full-tensor numerical comparisons, both output descriptors and cancellation/unload/retry ownership. The actual preparation hashes the three selected model streams and all selected tensors, validates finite values and schedule/index contents, and returns `stage: ready`, `nativeExecuted: false`, no native runs and no memory samples. Its largest admitted input/output buffer set is 215,669,248 bytes, excluding embedded model weights and native workspace. No GPU inference, model export, asset download or weight copy ran for this checkpoint.

`Scripts/qualify-h3-phases.py` supplies an owned process with a 300-second preparation budget or a 600-second native budget, terminating and joining on timeout. Before native launch it requires a passing, joined preparation controller and identical executable/source/fixture/manifest pins. It preserves stdout, stderr, source provenance and controller status beside the new output directory. The command used for the subsequent GPU qualification below was:

```sh
python3 -B Scripts/qualify-h3-phases.py \
  --bundle /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  --phases /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  --cpu-record /private/tmp/aether-h3-qkv-phase-oracles/cpu-checks.json \
  --prepared .local/h3-phases-prepared-02 \
  --output .local/h3-phases-native-01 --run
```

For a fresh CPU preparation, replace the final two lines with `--output NEW_PREPARATION_DIRECTORY --prepare-only`, then use that directory with `--prepared` for the native run. Output directories and controller artifacts must be new. Source or executable changes require fresh preparation. Ordinary tests and `Scripts/check.sh` do not launch this GPU workload.

The subsequent native qualification below passes this module gate. The next implementation boundary is linked preparation → all 50 blocks → output heads and the complete eight-step video/audio trajectory, including per-step timestep/modulation preparation. Conditioner presentation and the unnormalized 50-layer tap, broader conditioning, both decoders and media acceptance remain outstanding.


### Outer-phase qualification: 2026-10-18

The [qualification record](evidence/h3-phases-qualification-2026-10-18.json) passes the fixed 30-submission plan: 20 successful calls and ten observed cancellations across ten session opens. All 28 complete-tensor CPU comparisons pass the unchanged 0.02 relative-L2 bound. Every output is finite, and full-output hashes repeat exactly across normal calls, retained cancellation recovery and reopened recovery. CPU/native bytes differ. Per-output errors are:

| Boundary | Output | Relative L2 |
| --- | --- | ---: |
| Prepare | Hidden | 0.034101% |
| Time 0 | Embedding | 0.000395% |
| Heads 0 | Video | 0.267017% |
| Heads 0 | Audio | 0.229319% |
| Time 1 | Embedding | 0.001401% |
| Heads 1 | Video | 0.268450% |
| Heads 1 | Audio | 0.240560% |

Every cancellation trace observes one submission, cancellation request, joined completion, slot release and cancelled return in that order. All five concurrent unloads start while submitted work is pending and finish after joined completion. The retained sessions recover after cancellation; reopened sessions preserve both named head-output hashes.

The [raw report](evidence/h3-phases-native-2026-10-18.json) records a 3.7557 GiB process lifetime physical-footprint peak, 2.1357 GiB sampled Metal peak and 2,340 memory samples. Final five-second post-unload footprint is 0.5791 GiB, with 720,896 Metal bytes. Maximum positive settled footprint/resident growth between the two opens of each case is 245,760 bytes, below the unchanged 512 MiB gate; settled Metal growth is zero. Resident memory has a separate 8.2170 GiB lifetime peak; these counters overlap and cannot be added. The short probe and sampled peaks do not establish general retention or full-pipeline memory behavior.

The [owned controller](evidence/h3-phases-native-controller-2026-10-18.json) exits zero and joins without timeout in 58.0224 seconds within its 600-second budget. Preparation and native execution share all 55 executable/source/fixture/manifest pins, rechecked after exit; stderr is empty. The existing 105 passing CPU tests and six confined SDK stream diagnostics remain the implementation checkpoint. Elapsed time includes file verification, CPU comparisons and settling; it is not a benchmark. No further GPU inference was launched after this successful qualification.

This qualifies isolated outer modules at the captured contexts. Native prepare produces a numerically passing hidden state whose bytes differ from the CPU hidden state used by the earlier backbone and current head probes. The timestep probes likewise compare native embeddings separately while heads receive CPU embeddings. Linking these boundaries requires new context-aware numerical checks; the present hashes do not establish a native denoising step. Next are native prepare → all 50 blocks → both heads, coupled video/audio latent updates and the complete eight-step trajectory. Conditioner presentation/taps, broader conditioning, both decoders and media acceptance remain separate gates.

## Linked eight-step denoising

`H3DenoisingPlan` links the corrected outer-phase and fifty-block profiles to 50 original modulation graphs. It uses provided BF16 conditioning and F32 initial video/audio latents at the captured T2VA geometry. The retained bundle's modulation paths predate a prototype directory move; callers supply an explicit modulation directory, while every graph must still match its original declared stream hash. No graph export or model-weight copy is involved.

`H3DenoisingSession` computes eight native timestep embeddings and 400 native BF16 modulation tables, loading one modulation graph at a time. For each Turbo8 step it passes the current latents through native preparation, all 50 corrected blocks and both native output heads, then applies both F32 scheduler updates. It selects the one-timestep body at step 0 and the two-timestep body for steps 1–7. Switching releases the previous body backend after joined work; each backend owns only one resident set of block weights and distinct hidden-state buffers. The later body profile, time, prepare and head functions retain their streams across calls.

Runtime bindings validate actual descriptor names, states, dimensions, precision and a 2 GiB module buffer-admission limit before copying inputs. The body route checks runtime hidden/table/tag extents and index bounds; a subsequent fixed-probe call restores the original immutable fixture contents. Hidden states remain BF16 throughout. Boundary values are CPU-owned snapshots copied into the next stage's shared buffers, not a zero-copy pipeline or a performance optimization. The awaited consumer controls any snapshots it retains. Session admission remains held through native work and consumer callbacks, including cancellation, failure and concurrent unload.

The shared CPU-tested control loop carries both modalities through all eight independent scheduler-update fixtures and observes all 400 ordered block boundaries. Additional tests check input/descriptor rejection, original-state retry after consumer failure and cancellation/unload ownership. The suite passes 112 CPU tests (116 registered, four native tests disabled). The generic runtime function adds two confined SDK stream Sendable diagnostics; the four experimental native adapters now each retain that documented compatibility boundary.

`qualify-h3-denoising` separates CPU-only preparation, explicitly unqualified capture, and context-aware qualification:

```sh
H3_PROTOTYPE=<aether-prototype checkout>
python3 -B Scripts/qualify-h3-denoising.py \
  --bundle /private/tmp/aether-h3-qkv-ready/render/bundle.json \
  --phases /private/tmp/aether-h3-qkv-ready/phase/bundle.json \
  --cpu-record /private/tmp/aether-h3-qkv-phase-oracles/cpu-checks.json \
  --chain .local/h3-backbone-cpu-references-01/chain.json \
  --modulations "$H3_PROTOTYPE/.local/h3-diagnostic/models" \
  --output NEW_PREPARATION_DIRECTORY --prepare-only
```

A native capture adds `--prepared NEW_PREPARATION_DIRECTORY`, chooses a new `--output` directory and replaces `--prepare-only` with `--capture`. It performs 824 native submissions (8 time, 400 modulation, 8 prepare, 400 body, 8 heads) plus both CPU latent updates at each step. The recorder validates every boundary's order, shape, finite values and named outputs, persists 864 tensor identities, and saves only the contexts needed for references. It does not retain all 400 full hidden states. Capture remains `passed: false`, `captureComplete: true` until independent CPU references are available. Both preparation and capture run under owned stop/join controllers; preparation has a 300-second budget and native work a 1,800-second budget. Footprint/lifetime peak remain capped at 20 GiB and sampled Metal at 18 GiB.

`Scripts/prepare-h3-denoising-oracle.py` is CPU-only and calls the pinned original modules at captured native contexts: complete time/modulation/prepare/head/scheduler references, plus sampled body blocks 0 and 49 at all eight steps. It verifies sampled-query instrumentation against full 128-row CPU execution, loads one CPU backbone block at a time, and preserves original dynamic adapter/QKV identities. Qualification requires 464 numerical references and all 864 captured hashes, including explicit input-context identities for every CPU reference. The original CPU functions are not substituted with the new Swift formulas. This is per-module numerical validation along a native trajectory; it does not establish an independently computed CPU trajectory, all-block CPU parity, conditioner presentation/taps, keyframe coverage, decoders, media acceptance or long-run cancellation/retention behavior. Native elapsed times include hashing, copying, file IO, CPU comparisons and settling and are not benchmarks.

### Model-stream verification memory control

The [first linked capture](evidence/h3-denoising-memory-control-report-2026-10-18.json) stopped at the unchanged 20 GiB footprint bound immediately after its first timestep output. Lifetime footprint reached 52.7278 GiB while sampled Metal was only 63.0156 MiB; it did not execute a denoising step. Its [controller](evidence/h3-denoising-memory-control-controller-2026-10-18.json) exits 1 and joins without timeout, preserving the failed report and first output identity. The run remains unqualified.

The synchronous model-stream verifier read roughly 26 GB of embedded modulation graphs per pass through Foundation `FileHandle`. Its autoreleased chunk storage could survive until the next asynchronous suspension. The [reader change](evidence/h3-denoising-memory-reader-fix-2026-10-18.diff) drains each 8 MiB chunk inside an explicit autorelease pool while preserving SHA-256 checks. The integrity test now spans two complete chunks and a partial tail and rejects corruption in the second chunk. All [112 CPU tests](evidence/h3-denoising-memory-fixed-tests-2026-10-18.log) still pass. The CLI also checks CPU verification footprint/lifetime peak before acquiring native resources.

A fresh [CPU preparation](evidence/h3-denoising-memory-fixed-prepared-2026-10-18.json) verifies the same assets with a 0.5804 GiB lifetime footprint peak and no native acquisition. The [controller](evidence/h3-denoising-memory-fixed-controller-2026-10-18.json) passes and joins under the same 300-second budget. Shared pin differences are confined to the reader, the pre-native memory guard, the chunk-integrity test and rebuilt executable. Graphs, weights, geometry, numerical tolerances and the 20/18 GiB native limits are unchanged. This CPU result does not by itself qualify the linked GPU trajectory; a new capture follows it.

### Linked capture and original CPU references: 2026-10-18

The [bounded capture](evidence/h3-denoising-capture-summary-2026-10-18.json) completes all eight linked steps: 824 native submissions and 864 finite tensor identities. Its controller exits zero and joins without timeout in 882.7007 seconds within the 1,800-second budget. Capture remains explicitly unqualified. All 480 saved tensor files (5,309,332,992 bytes, retained in `.local`) pass independent extent/hash checks; every input latent hash after step 0 equals the preceding update hash. Eight timestep and 400 modulation outputs also match the previously retained native preparation-cache identities exactly.

Capture's process lifetime footprint peaks at 10.5409 GiB and sampled Metal at 11.1119 GiB. Five seconds after unload, footprint is 0.9359 GiB and Metal is 720,896 bytes. The same 20/18 GiB bounds apply, and the corrected reader's pre-native verification peaks near 0.58 GiB. Counters overlap, sampled peaks can be missed, and this single trajectory does not establish long-run retention or native whole-trajectory cancellation/recovery.

The [CPU reference result](evidence/h3-denoising-cpu-reference-summary-2026-10-18.json) completes 464 references: 448 complete tensors and 16 sampled body checkpoints (blocks 0 and 49 at all eight steps). Sixteen full-128-row versus selected-32-row method checks are byte-identical. The helper runs only on four CPU threads; its owned controller exits zero and joins within its 1,800-second budget. Every reference file and recorded source pin is verified.

All 456 comparisons possible from saved capture outputs pass the unchanged 0.02 relative-L2 limit; their largest error is 0.548004%. The eight block-0 outputs were not saved, so their sampled comparisons require the subsequent native replay. The CPU references use actual captured native embeddings/hidden inputs, including complete keys/values for sampled attention queries. All 16 original CPU scheduler updates match the captured Swift update hashes exactly. These checks do not establish an independently computed CPU denoising trajectory or CPU parity at every backbone block.


### Denoising qualification: 2026-10-18

The [qualification record](evidence/h3-denoising-qualification-2026-10-18.json) passes all eight linked Turbo8 steps at the fixed 960×544, 124-frame, 19,333-row T2VA geometry with provided conditioning. The final run completes 824 native submissions: eight time embeddings, 400 modulation calls, eight preparations, 400 backbone blocks and eight two-output head calls. Both coupled F32 latent updates execute at every step. All 864 finite tensor hashes reproduce the independent capture exactly, including every intermediate block output and both final latents.

All 464 original-CPU numerical comparisons pass the unchanged 0.02 relative-L2 bound. Their scope and maximum errors are:

| Stage | Comparisons | CPU reference scope | Maximum relative L2 |
| --- | ---: | --- | ---: |
| Time embeddings | 8 | Complete tensor | 0.001401% |
| Modulation tables | 400 | Complete tensor | 0.010095% |
| Preparation | 8 | Complete tensor | 0.041223% |
| Backbone | 16 | 64 rows at blocks 0 and 49, all eight steps | 1.096185% |
| Video/audio heads | 16 | Complete tensor | 0.232990% |
| Video/audio updates | 16 | Complete tensor; byte-identical | 0.000000% |

This includes the eight block-0 comparisons unavailable from saved capture files. These are 448 complete tensors and 16 sampled body checkpoints at actual captured native contexts, rather than an independently computed CPU trajectory or numerical checks at every backbone block. Native/CPU values differ where the numerical gates allow; all 16 CPU scheduler updates are byte-identical.

The [raw native report](evidence/h3-denoising-native-2026-10-18.json) records a 10.5723 GiB lifetime physical-footprint peak and 11.1119 GiB sampled Metal peak. Five seconds after unload, footprint settles to 0.9808 GiB and Metal to 720,896 bytes. The 20/18 GiB limits remain unchanged. The [controller](evidence/h3-denoising-native-controller-2026-10-18.json) exits zero and joins without timeout in 1,048.0039 seconds within its 1,800-second budget. Stderr is empty. All 166 capture source/model/fixture/manifest pins remain identical in the final run; its additional pin is the complete CPU-reference manifest. The separate original-CPU reference job and all native controllers have joined. No GPU inference was launched after this successful qualification.

The verifier memory fix remains necessary: the earlier one-submission control exceeded 52 GiB before its guard stopped it, while the corrected reader completes both eight-step passes inside the original memory limits. The controller logs and reader diff preserve that failed control. The Release suite remains at 112 passing CPU tests (116 registered, four native tests disabled), including multi-chunk integrity, eight-step latent carry and cancellation/backpressure tests. Eight confined SDK stream Sendable diagnostics remain explicit. The 882.7007-second capture and 1,048.0039-second qualification include file verification, CPU copies/comparisons, IO and settling on a shared machine; they are not benchmarks.

For another qualification, supply the completed `references.json` with the wrapper's `--references` option during a fresh `--prepare-only` run, then use the same reference manifest and new preparation directory with `--prepared ... --run`. Existing output/controller directories are rejected; changed source/executable/reference pins require new preparation.

The result is a bounded experimental provided-conditioning trajectory. Native whole-trajectory cancellation/recovery, retained repetition and long-run retention remain separate gates; the earlier per-module native lifecycle tests and current CPU session tests do not establish those properties for the complete run. Conditioner presentation and its unnormalized 50-layer tap, keyframe coverage, both decoders/stitching and media acceptance remain outstanding. Next is the full-run lifecycle gate, followed by real conditioning and decoding.

### Denoising lifecycle protocol

`qualify-h3-denoising` also accepts `--lifecycle` with a completed CPU-reference manifest. Run `Scripts/qualify-h3-denoising.py --prepare-only` with the same bundle, phases, CPU record, chain, modulation directory and references first, then supply that new preparation directory with `--prepared ... --lifecycle`. Both output paths must be new. The wrapper pins the executable, sources, reference manifest and selected modulation models before execution, checks those pins against preparation, and rechecks them after joining the child process.

The predefined gate permits 3,340 native submissions within a 5,400-second owned process budget: an initial full trajectory, cancellation on submitted step-0/block-24 work, a full recovery on the retained session, another cancellation with concurrent unload, and a full recovery on a reopened session. Each complete run must reproduce all 864 previously qualified hashes and pass 464 original-CPU comparisons at the same contexts. Each cancellation must deliver precisely the preceding 435 tensor outputs and 410 CPU comparisons, join the submitted block before releasing the operation slot, and submit no later block. Concurrent unload must start while GPU work is pending and finish after its joined completion; a closed session must reject another request.

The process samples memory every 20 ms and retains the 20 GiB physical-footprint/lifetime-peak and 18 GiB Metal limits. Each session is measured at 0, 1 and 5 seconds after unload. The reopened session's final footprint, resident memory and Metal allocation may each grow by at most 512 MiB relative to the first unloaded session. Per-role `progress.json` files retain output identities and numerical comparisons without retaining tensor arrays. The final report records ordered native and lifecycle events. These bounded checks qualify the provided-conditioning trajectory's lifecycle; they do not establish long-running soak behavior, real conditioning, decoder correctness, media quality or benchmark performance.

The first lifecycle attempt was [stopped at the user's request](handoffs/h3-denoising-lifecycle-2026-10-18.md). Initial and retained full runs, retained cancellation and concurrent-unload cancellation passed; reopened recovery was interrupted near the end of its fourth step. The overall lifecycle gate remains incomplete. The handoff preserves the controller's joined termination, partial numerical results, completed ownership traces, outstanding memory-growth check and repeat instructions. GPU work is stopped pending a request to resume.

## Schema-3 graph gate at 960×544: 2026-10-26

`Scripts/qualify-h3-profile-graphs.py` runs a fixed sequence of stages. Each stage runs as an owned process group with a wall budget and is joined before the next one starts:

1. CPU dumps for T2VA and FL2VA (the FL2VA dump adds a first frame).
2. The original-module CPU text oracle.
3. `aether-probe h3-profile-run --graphs` for both dumps, each under `~/.cache/aether-kit/with-gpu-lock.sh`. That lock now also takes Atelier's `/tmp/atelier-gpu.lock` lease.
4. The original-module CPU graph oracles.
5. `--part compare`.
6. A gate.

The gate requires every listed tensor to be present, finite, the same shape and within the predefined 0.02 relative-L2 limit. It also requires the oracle's tokenizer and layout checks to pass, and every source, executable, input and profile pin to be unchanged. CPU stages never hold the GPU lock. Each one waits for at least 40% free memory and is stopped if its resident memory exceeds 20 GiB.

Both runs used profile `00669ed7…` at 960×544 and 124 frames (19,333 rows, 49 prompt tokens), seed 1043, Turbo8, with the [prototype production prompt](evidence/h3-profile-graphs-prompt-2026-10-26.txt). Each native graph after the conditioner tap receives the CPU oracle's own input bytes, so each comparison isolates one graph.

**The qualifying run passes.** In [that gate record](evidence/h3-profile-graphs-960-contexts-2026-10-26.json), blocks read their real step-0 contexts. The native run chains all 50 blocks from the native packed sequence and saves block 0's output, block 49's input (the native block-48 output) and block 49's output. Heads read the native block-49 output. The CPU oracle evaluates the original bf16 modules on exactly those bytes, and the gate checks that it did. Every check passes:

| Graph | Relative L2 | Cosine |
| --- | ---: | ---: |
| Conditioner tap (native vs CPU, independent) | 0.851% | 0.999998 |
| Prepare text (refined) | 0.660% | 0.999978 |
| Prepare pack | 0.0014% | 1.000000 |
| Block 0, real step-0 context | 0.701% | 0.999982 |
| Block 49, real step-0 context | 0.621% | 0.999981 |
| Heads on block-49 output, video | 0.170% | 0.999999 |
| Heads on block-49 output, audio | 0.180% | 0.999998 |
| Tile decoder (bf16), one window | 0.631% | 0.999980 |
| Audio decoder (fixed Na 207) | 0.0022% | 1.000000 |
| First-frame encoder 960×544 (FL2VA) | 0.022% | 1.000000 |

The tokenizer ids equal Transformers, and the layout and step-plan checks are all true. This covers step 0 only, with one timestep cardinality. Blocks at later steps, at the two-timestep cardinality, belong to the trajectory gate. The [CPU tests](evidence/h3-profile-graphs-contexts-tests-2026-10-26.log) pass with the context change: 232 run, 219 pass and the rest are opt-in skips.

**The first run failed and is superseded.** [Its gate record](evidence/h3-profile-graphs-960-2026-10-26.json) fed both blocks and the heads the dump's synthetic unit-variance `probe_hidden`. Block 49 measured 2.640% on that input, beyond the limit; everything else passed. [A CPU float32 diagnostic](evidence/h3-profile-graphs-960-block49-diagnostic-2026-10-26.json) explains why. The native output is closer to the float32 module (2.43%) than the original bf16 CPU module itself is (2.63%), so that synthetic input exposed bf16 arithmetic noise, not a native defect. The float32 result is kept as a diagnostic only. The acceptance criterion stays at 0.02 relative L2 against the original-precision CPU module, and the fix was the realistic contexts above.

**Memory item, since resolved.** On profile 00669ed7 the FL2VA first-frame encoder sampled **27.3 GiB of Metal** in both runs (10.2–11.8 GiB footprint), well above 18 GiB. The gate recorded it under `openItems`. The tiled encoder below brings it to 2.05 GiB, and the [rerun on the overlay profile](evidence/h3-profile-graphs-960-tiled-2026-10-26.json) passes every comparison with no open items. The native T2VA graphs peaked at 13.5 GiB footprint and 12.6 GiB sampled Metal.

**Resources for the qualifying run.** GPU-lock hold times were 105 s for T2VA (the 50-block chain) and 23 s for FL2VA, after waits of about 4.4 and 4.6 minutes for other agents. The CPU graph oracle took 14 minutes, about 7 minutes for each bf16 block at 19,333 rows, with a sampled process-group resident peak of 12.4 GiB. This gate checks isolated graphs only. It is not a denoising-trajectory, decoded-clip, lifecycle, memory-limit or performance qualification.

## The 18 GiB memory bar: tiled first-frame encoder and measured admission (2026-10-26)

H3 is held to an 18 GiB bar: the Metal allocation peak over a complete render (`MTLDevice.currentAllocatedSize`, sampled every 20 ms) must not exceed 18 GiB for either task. The bar is checked by a gate and enforced at admission.

### Diagnosis

[The memory record](evidence/h3-memory-2026-10-26.json) breaks the original FL2VA peak down with phase timelines (`aether-probe h3-memory-encoder` / `h3-memory-render`). The whole-frame first-frame encoder holds 0.20 GiB after load and specialization, so there is no load or compile spike. Its first call then raises Metal to 27.26 GiB, and that stays reserved while the model is loaded. It is reclaimed only 1–6 s after release. The graph unrolls the pipeline's own tiled encode, and its scratch grows about 1.8 GiB per 256 px tile: 7.46 GiB for the 4-tile 288×288 canvas, 27.26 GiB for 15 tiles at 960×544. `expectFrequentReshapes` changes neither memory nor output. Nothing else is resident at that point, so reordering residency could not help.

### Tiled first-frame encoder

`export-h3-profile.py --stage video-encoder-first-tiled` splits `encode_vae_condition` where the pipeline already splits it:

- **One tile graph**, shared by every canvas: ImageNet normalization, encoder, `quant_conv` → moments `[1,48,1,16,16]`.
- **One tail graph per canvas:** the source `_stitch_tiles` blend in latent space, the posterior with the fixed seed-42 draw baked in, float16 rounding and latent normalization.

On CPU, tile graph plus tail is byte-identical to `encode_vae_condition` at 960×544, 544×960 and 288×288 ([export record](evidence/h3-encoder-tiled-export-2026-10-26.json)). The stage writes an overlay bundle that adds `video_encoder_first_tiled` plus a `lineage` record of the base bundle and its hash. Every other entry stays the base 00669ed7's, and the base export directory is never written. The overlay, `c49d8dbf…`, lives on `/Volumes/Scratch/aether/aether-kit-h3-2026-10-26/profile-encoder-tiled`.

`H3Profile.keyframeEncoding(width:height:)` selects the tiled graphs when the manifest declares them for the canvas, and otherwise the whole-frame graph; there is no flag. `H3ProfileKeyframeEncoder` runs the tile graph once per source tile (row-major over `H3SpatialTiling.source`) and then the tail. CPU tests cover both paths and the manifest selection.

Natively, the tiled encoder's output is **byte-identical** to the whole-frame graph's: the same condition SHA-256 at 960×544 and at 288×288, so relative L2 is 0. The encoder gate still passes at 0.022% against the CPU oracle. Encoder Metal falls from 27.26 to 2.05 GiB, and footprint from 17.4 to 1.58 GiB, at both canvases. All 457 T2VA trajectory boundary hashes are identical before and after the change ([identity record](evidence/h3-trajectory-t2va-identity-2026-10-26.json)).

### Memory gate

`Scripts/qualify-h3-memory.py` renders T2VA and FL2VA through `H3VideoSession` in both orientations, at 124 frames with Turbo8, under 20 ms timelines. Each sample is attributed to its stage; a stage includes memory the previous stage has not yet returned. [The gate](evidence/h3-memory-gate-2026-10-26.json) states `memoryBarBytes` and fails if any render's Metal peak exceeds it. It passes. Peaks, as Metal / footprint in GiB, at 960×544 (544×960 matches):

| Stage | T2VA | FL2VA |
| --- | ---: | ---: |
| Conditioner | 1.83 / 2.01 | 1.83 / 2.01 |
| First-frame encoder (tiled) | — | 2.05 / 1.89 |
| Denoising | 13.49 / 14.28 | 13.80 / 14.63 |
| Video decode | 5.64 / 9.81 | 5.78 / 12.63 |
| Audio decode | 1.03 / 1.89 | 1.03 / 1.92 |
| **Whole render, Metal** | **13.49** | **13.80** |

The video-decode footprint is mostly the denoiser's memory that has not yet been reclaimed. Denoising rises by about 2.7 GiB from step 1, when the two-timestep body specialization joins the one-timestep one. That costs memory but not time (see below). Since 2026-10-27 a render uses one specialization, and the current gate is in [one modulation shape per render](#denoiser-memory-one-modulation-shape-per-render-2026-10-27).

**Diagnostic: spike capture.** `aether-probe h3-memory-render PROFILE DUMP NEW_DIR --spike-capture=GiB` (and `qualify-h3-memory.py --spike-capture GIB`, which passes it to every render and lists the triggers in the gate record) watches the 20 ms Metal samples. The first time a denoising sample exceeds the given level, it writes `trigger.json` (phase, time, Metal and footprint), a 1 s `sample` of every thread's stack, `vmmap --summary` and `footprint` for the process into `NEW_DIR.spike/`, about 120 KB. It is off by default and fires at most once per render. `sample` suspends the process's threads while it reads them, so step timing inside a captured window is not representative; memory is. Instruments is not used: even a Blank template with only Metal Resource Events writes GPU-counter files at about 2 GB per minute.

`--carry-peaks RECORD…` makes the measured entry also cover the per-stage peaks of earlier gate records, each cited in the new record (`carriedStagePeaks`), so a transient one gate saw is not dropped by the next.

### Measured admission

The gate writes a measured entry: per stage, the largest sample of max(Metal, footprint), plus the session's carried CPU state. The carried state is sized from the clip's tensors (192 MiB). The entry is bundled as `AetherH3/Resources/h3-measured-resources.json`. `H3ProfileEngine.measuredAdmission(profile:recipe:coordinator:)` turns it into an `H3ResourceAdmission`:

- Every stage reserves its measured peak plus 256 MiB.
- The operation reserves the carried state.
- Pixel and frame limits are those measured, at most 124 frames. Admission is keyed by pixel count, so any canvas up to 522,240 px (960×544) is admitted on the 960×544 numbers. That is conservative: the measured canvas is the largest allowed, and both measured orientations peak identically.

`H3MeasuredResources` refuses any entry whose largest stage plus margin and operation exceeds 18 GiB. The worst case was 15.07 GiB at first and 15.26 GiB on the chained build; the padded build's entry must also cover a step-0 transient, about 16.6 GiB (see [one modulation shape per render](#denoiser-memory-one-modulation-shape-per-render-2026-10-27)). Before a run reserves anything, admission checks that the operation plus the largest stage the request needs fits the coordinator's capacity. Once the operation is reserved, it checks that the largest stage fits what is still available. Either check refuses the run before any stage loads. A request needing a stage the entry lacks is also refused up front: the first-frame encoder for FL2VA, the audio decoder when audio is produced. Stages are still reserved one at a time, so if other holders take the budget mid-run, a later stage can still be refused (CPU-tested).

Pass the admission to `H3VideoSession(stages:resources:gpu:)` together with the host's `GPUAdmission` owner. The operation and every stage reservation are then taken only after the GPU grant, and a generation waiting for the GPU holds no reservation (CPU-tested).

The bar binds callers that use measured admission. It is opt-in: `H3VideoSession(stages:)` without `resources:` reserves nothing and is not checked against the bar.

The gate record pins the files a render depends on (`filesSHA256`): the probe binary, `Package.swift`, the profile, prompt and keyframes, and the `AetherKit`, `AetherCoreAI`, `AetherH3`, `AetherH3CoreAI` and `AetherProbe` sources. Nothing enforces them after the gate; the release checklist ([qualification and release](qualification-and-release.md)) requires a re-gate when any pinned path has changed since the bundled entry's record.

Measured admission covers only the profile/device pairs present in `h3-measured-resources.json`. Today that is the Kit-owned clone of the qualified profile (`5af39db5…`, the c49d8dbf overlay with only its paths changed) with Turbo8 on Core AI architecture `h17c`; the in-place c49d8dbf identity no longer has an entry. Any other Mac or profile is refused, not estimated. To add one, run `Scripts/qualify-h3-memory.py --record NAME` on that device against that profile. It must pass. It writes its gate record to `docs/evidence/NAME`, and its `measuredResources` entry cites that file; the entry, `stage_peak_bytes` and `operation_bytes` included, is then appended to the resource file. The trajectory gate also fails when its capture or replay render exceeds 18 GiB.

## Schema-3 Turbo8 trajectory gate (2026-10-26)

`H3ProfileDenoiser` takes an opt-in `H3DenoiserObserver` through `H3ProfileEngine.stages(..., observer:)`. It receives tensors the trajectory already holds on the CPU: the refined text, each step's input latents, the packed sequence, every block output, both heads and the updated latents. Without an observer nothing is constructed. A CPU test shows identical kernel calls and latents with and without one.

`Scripts/qualify-h3-profile-trajectory.py` owns one task's gate, in order:

1. **Capture:** a complete `H3VideoSession` render (`aether-probe h3-profile-run --render --capture=STEPS`) that hashes every boundary. That is 457 hashes for 8 steps: refined text, then per step the input latents, packed sequence, 50 block outputs, heads and updated latents. It also saves the selected steps' reference contexts.
2. **Replay:** a fresh render that must reproduce every hash (`--replay`). A difference stops the gate.
3. **CPU oracle:** `prepare-h3-profile-oracle.py --part trajectory`, which runs off the GPU lock on 8 threads with one block resident. At each selected step it computes original-precision references at the captured contexts, on full tensors:
   - the original packing;
   - blocks 0 and 49, on their captured inputs, with live AdaLN from the live time embedding;
   - both heads, on the captured block-49 output;
   - both schedulers' `step`.
4. **Float32 diagnostic:** an optional stage, never an acceptance criterion.
5. **Gate:** 0.02 relative L2 on every required comparison, exact replay, unchanged pins, and the 18 GiB bar on both renders. Captured tensors are deleted once the gate passes.

**T2VA passes ([gate record](evidence/h3-profile-trajectory-t2va-2026-10-26.json)).** It ran on overlay profile c49d8dbf at 960×544 and 124 frames, seed 1043, with the production prompt.

- **Replay:** all 457 boundary hashes reproduce exactly. They also match a capture taken before the tiled-encoder change and a replay built on the rebased tree.
- **Comparisons:** 56 across all eight steps; maximum relative L2 per graph:

  | Graph | Maximum relative L2 |
  | --- | ---: |
  | Packing | 0.0016% |
  | Block 0 | 0.838% |
  | Block 49 | 0.713% |
  | Heads, video / audio | 0.170% / 0.214% |
  | Scheduler updates | 0, byte-identical (all 16) |

- **Float32 diagnostic:** native blocks are 0.38–0.62% from the same modules in float32.
- **Memory:** both renders peak at 13.49 GiB Metal.
- **Run time:** the CPU oracle took 2 h 49 min on a busy machine, about 10–20 min per bf16 block at 19,333 rows.

The first oracle attempt was stopped by its 20 GiB resident guard ([controller record](evidence/h3-profile-trajectory-t2va-oracle-rss-stop-2026-10-26.json)) because it kept both blocks and their float32 copies resident. The oracle now keeps one block resident, and float32 runs as its own stage.

A replay built from the landed tree on e4e41a6 also reproduces all 457 T2VA hashes ([record](evidence/h3-trajectory-t2va-replay-landed-2026-10-26.json)).

**FL2VA passes ([gate record](evidence/h3-profile-trajectory-fl2va-2026-10-26.json)).** It ran on the tiled first-frame encoder, overlay c49d8dbf, at 960×544 and 124 frames, seed 1043, with the crane keyframe, on the landed code.

- **Replay:** all 457 boundary hashes reproduce exactly.
- **CPU references:** steps 0, 1 and 7, which cover both timestep cardinalities and the last step, give 21 comparisons. Maximum relative L2 per graph:

  | Graph | Maximum relative L2 |
  | --- | ---: |
  | Packing | 0.0015% |
  | Block 0 | 0.602% |
  | Block 49 | 1.120% |
  | Heads, video / audio | 0.169% / 0.203% |
  | Scheduler updates | 0, byte-identical; keyframe anchor rows never stepped |

- **Float32 diagnostic:** native blocks are 0.33–1.03% from float32.
- **Memory:** both renders peak at 13.80 GiB Metal, under the gate's 18 GiB bar.
- **Run time:** the CPU oracle took 58 min.

## Denoiser speed: device-chained blocks (2026-10-26)

[A phase profile of one denoising step](evidence/h3-speed-profile-2026-10-26.json) covers 50 block calls at 19,333 rows, in interleaved sessions within one process. It shows the denoiser is GPU-bound at about 680 ms of GPU work per block:

- **Weight streaming is already hidden.** The next block's weights are prefetched into the spare bank while the GPU runs: 150–230 ms warm, about 560 ms on the first pass, which also hashes each file. Keeping more weights resident would therefore buy nothing.
- **No guidance pass.** Each step runs 50 blocks and one heads call.
- **The second body specialization costs memory, not time.** From step 1 the two-timestep shape joins the one-timestep one; a first call at the other cardinality takes under 0.3 s extra, once per render. It is avoided since 2026-10-27 (see below).
- **`expectFrequentReshapes = false` is rejected.** It gave no speedup, changed the output bytes and added 2.07 GiB of Metal.
- **The removable cost was the host round trip per block:** about 100 ms copying each 208 MB output to the host, and about 5 ms copying it back.

`H3DenoiserKernels.blocks(...)` now runs a step's blocks as one chain. Block 0 copies the packed sequence in. Every later block binds its predecessor's output on the device, through `NativeH3Graph`'s `.chained` input and two alternating output buffers. Only the last output is read back, or every output when an observer is attached.

Outputs stay byte-identical:

- A chained build replays all 457 boundary hashes of both the T2VA and the FL2VA captures ([replays](evidence/h3-speed-chaining-replays-2026-10-26.json)).
- After the chain bookkeeping moved into `H3ChainState`, a native replay at 5cdb5f3 again reproduces all 457 T2VA hashes ([record](evidence/h3-speed-chaining-replay-t2va-5cdb5f3-2026-10-27.json)).
- Four paired renders without an observer all end on the capture's final latents.

[Paired, interleaved timings](evidence/h3-speed-chaining-paired-t2va-2026-10-26.json) for T2VA at 960×544, ABBA order, n = 2 each, load average 9–20:

| | Current | Chained | Change |
| --- | ---: | ---: | ---: |
| Denoising, median | 350.7 s | 326.3 s | −7.0% |
| Render, median | 406.0 s | 381.6 s | −6.0% |

The two device chain buffers raise the Metal peak by about 0.2 GiB. [The memory gate](evidence/h3-memory-gate-chained-2026-10-26.json), rerun on the chained build, passes: T2VA peaks at 13.69 GiB, FL2VA at 14.00 GiB, and the admission worst case is 15.26 GiB. The bundled measured entry is updated to match.

### Timed renders on this Mac: busy host

**These are busy-host timings, not a quiet reference.** Other sessions kept the host loaded all night, so after the agreed capped wait both renders ran anyway. Each is one complete `H3VideoSession` render (`aether-probe h3-memory-render`, chained build): 960×544, 124 frames, Turbo8 (8 steps), M5 Max, Core AI architecture h17c ([record](evidence/h3-timed-renders-2026-10-26.json)). The T2VA render started at a 1-minute load of 2.8 (5-minute 20.5), and load rose to 126 during the run. The FL2VA render ran at a load of about 68–84 throughout.

| Stage, wall seconds | T2VA (busy host, load 3 → 126) | FL2VA (busy host, load 68–84) |
| --- | ---: | ---: |
| Conditioner (graph load + 50 streamed layers) | 19.2 | 26.1 |
| First-frame encoder, tiled | — | 5.6 |
| Denoising, 8 steps | 299.4 | 352.0 |
| Video decode (tile graph load + 105 tiles) | 35.8 | 35.6 |
| Audio decode | 0.75 | 0.72 |
| **Total** | **355.1** | **420.0** |

Per-step denoising times:

- T2VA: 33.8, 35.2, 35.3, 35.5, 36.8, 37.4, 42.4 and 42.8 s, slowing as load rose.
- FL2VA: 43.6–44.3 s.

Graphs open lazily inside their stage, so each stage's time includes its own load and specialization. Under steady load, the FL2VA step 0 (44.0 s) equals the later steps (median 44.1 s). So inside denoising, the body graph's load and specialization, text refinement and first-pass weight hashing add no measurable time: denoising is compute. The conditioner and video-decode rows include their graph loads. A quiet-host reference remains to be measured.

### Quiet-host timing reference (2026-10-28)

The reference for this Mac (M5 Max, Core AI architecture h17c): one complete `H3VideoSession` render per task through `aether-probe h3-memory-render`, at 960×544 and 124 frames, Turbo8, seed 1043, profile `5af39db5…`, build `bd048e4`. The user's transcode was paused. No other GPU work overlapped a render (the lock serialized it; other agents' short GPU jobs ran between renders). A script refused to start a render unless the one-minute load was under 10, memorystatus at least 70 and at least 40 GiB free ([record](evidence/h3-quiet-timing-2026-10-28.json)).

The two references come from different windows: FL2VA ran 07:30–07:36, right after a first T2VA run, and started beside one compile at 100% CPU; the T2VA reference is a re-run at 07:41–07:47, after the FLUX benchmark.

| Stage, wall seconds | T2VA (re-run, load 0.9–2.6) | FL2VA (load 1.3–3.9) |
| --- | ---: | ---: |
| Conditioner (graph load + 50 streamed layers) | 18.8 | 19.0 |
| First-frame encoder, tiled | — | 4.8 |
| Denoising, 8 steps | 307.9 | 288.5 |
| Video decode | 30.8 | 29.8 |
| Audio decode | 0.7 | 0.7 |
| **Total of the stages** | **358.2** | **342.8** |

Times are sums of the session's raw phase durations, rounded at the end. Steps took 36.3–39.9 s for T2VA and 35.4–36.3 s for FL2VA. T2VA's steps are slower although FL2VA denoises more rows; the comparison crosses windows, and with one run per task the cause is not known. This is one run per task, not a benchmark distribution.

The first T2VA run is kept in the record as a side note only. Its load was 9.3 at the start, so it passed the guard, and peaked at 10.7 at 2.4 min, while another session's TripoSR job, Atelier Studio and an index build were still running; its steps took 38.2–44.6 s (total 380.1 s). The load guard alone does not make a run quiet.

## Denoiser memory: one modulation shape per render (2026-10-27)

**Cause.** The body graph's `temb` input is the block's AdaLN table, `[6, 3k, 5376]`, and the heads graph's is the timestep embedding, `[k, 2688]`. Both are exported with k dynamic in 1…3, where k is the step's number of distinct row timesteps. T2VA has k = 1 at step 0, where video and audio both start at t = 1.0, and k = 2 from step 1, where the shift-12 and shift-3 schedules diverge. FL2VA has 2 and then 3, because keyframe rows use max(t, 0.999). Core AI specializes a reshape-enabled function once per input shape and keeps every specialization until the model is released, so the change at step 1 left a second body specialization resident: the 2.57 GiB rise in the chained gate's T2VA render.

**Workaround.** `H3ModulationPadding` pads every step's tables and embedding to the trajectory's largest k by repeating existing rows, so a render uses one body and one heads specialization. This works around Core AI's per-shape retention; it does not fix it. The padding is exact: the block reads its table only through `index_select` on the AdaLN indices (below 3k) and the heads read embedding rows only through the timestep indices (below k), so the appended rows are never selected. No re-export is needed. Two alternatives were rejected: reopening the body after step 0 gave the same plateau but cost 12 s of re-specialization at step 1 even at 288×288, and exporting a fixed cardinality is equivalent but needs a re-export.

**Exactness.** Replays of both qualified captures on the padded build reproduce all 457 boundary hashes each (`replay_mismatches` 0 for T2VA and FL2VA), from dumps regenerated identical to the gate dumps.

**Memory.** Denoising Metal, plateau / peak in GiB, T2VA 960×544×124 unless named:

| Render | Current build | Padded |
| --- | ---: | ---: |
| T2VA 960×544 (standalone) | 13.49 / 13.69 | 10.93 / 11.13 |
| T2VA 544×960 (gate) | — | 10.93 / 11.13 |
| FL2VA, both orientations (gate) | — / 14.00 (chained gate, 2026-10-26; no unpadded plateau recorded) | 11.17 / 11.84 |
| 960×544×22, 3,693 rows | 3.83 / 3.87 | 3.35 / 3.38 |
| 288×288×22, 690 rows | 1.99 / 2.00 | 1.90 / 1.90 |

The saving scales at about 0.133 MiB per packed row. FL2VA's 11.84 GiB peak is at step 0, as before. The video-decode stage's footprint peak, mostly denoiser memory not yet reclaimed, falls from 11.07 GiB in the
2026-10-26 chained gate ([record](evidence/h3-memory-gate-chained-2026-10-26.json)) to 8.16 GiB in the 2026-10-27
padded gate ([record](evidence/h3-memory-gate-padded-2026-10-27.json)), and is 8.45 GiB in the 2026-10-28 gate of
record ([record](evidence/h3-memory-gate-review-2026-10-28.json)).

**Step time.** The extra table rows cost nothing measurable. Single T2VA 960×544 renders on a shared host: the current build took 384.0 s of denoising (median of steps 1–7: 47.7 s) and the padded build 380.8 s (47.4 s). Steps 1–7 use the same specialization in both builds, and the padded build no longer specializes again at step 1.

**Memory gate and an unexplained step-0 transient.** [The gate](evidence/h3-memory-gate-padded-2026-10-27.json) passes: T2VA 960×544 peaks at 14.05 GiB and the other three renders at 11.13–11.84 GiB. The 14.05 GiB is a transient: both gate renders of T2VA 960×544 showed a rise in step 0 that was released within about 5 s. [The first](evidence/h3-memory-gate-padded-transient-2026-10-27.json) reached 16.12 GiB twice, at 12 s and 28.6 s into step 0, for about 5.5 and 6.3 s. Each time the footprint rose with it within about 0.3 s, to 15.96 and 15.71 GiB, stayed above 12.5 GiB for about 1 s, and fell back to about 11.1 GiB while Metal stayed at 16.12 GiB for the rest of the window. The second reached 14.05 GiB once (footprint 13.89 GiB, above 12.5 GiB for about 0.8 s). So the memory is touched, briefly; admission counts the Metal peak, which is the larger. It did not appear in a standalone repeat of the same render, in a render with a fresh, empty Core AI compiled cache, in the gate's other three renders or in the current-build render, so it is not cold specialization, and its cause is unknown. In all, it appeared in 2 of 9 full-size padded renders, both of them gate renders on a busy host; the unpadded build was never measured under the same conditions, so whether padding causes it is unknown.

**Admission covers the transient.** That gate's own entry kept only the second render (worst case 14.49 GiB), which does not cover the first render's 16.12 GiB. **The 2026-10-28 gate** ([record](evidence/h3-memory-gate-review-2026-10-28.json)) measures the final build (merged main with the deployment floor, and the #55 review fixes) on the Kit-owned profile `5af39db5…` with `--spike-capture 12.5` and `--carry-peaks`. It passes. All four renders peak at 11.13 GiB (T2VA) and 11.84 GiB (FL2VA, step 0); the transient did not recur, so no capture fired. The same gate before the review fixes ([earlier record](evidence/h3-memory-gate-2026-10-28.json)) gave the same result. The carried record must have passed on the same device, and its SHA-256 is recorded (`carriedRecords`). The entry carries the first padded gate's 16.12 GiB denoising peak, citing its record (`carriedStagePeaks`), so the bundled measured entry's admitted worst case is 16.56 GiB: higher than the unpadded build's 15.26 GiB, and under the 18 GiB bar. Padding lowers the steady plateaus and the video-decode footprint, not the admission. The gate ran on a busy host (one-minute load 59–142, 51–150 in the earlier run; the CPU saturated by an unrelated transcode), so its stage timings are recorded but are not a timing reference; a quiet-host reference is still open. See the [record](evidence/h3-second-specialization-2026-10-27.json) for every render's step times and peaks.

## Session lifecycle qualification (2026-10-28)

`H3ProfileEngine.stages(..., nativeObserver:)` takes an opt-in `H3NativeCallObserver` for qualification. Every native submission of every stage's graphs reports `begin` right before its GPU work is encoded and `end` once that work has joined (`failed` if encoding threw, after the join), with the graph's role (`conditioner`, `text`, `pack`, `body`, `heads`, `keyframe`, `videoTile`, `audio`) and a per-graph call count. Without an observer nothing is reported.

`aether-probe qualify-h3-session-lifecycle PROFILE DUMP NEW_DIR` uses it to qualify `H3VideoSession` on the native stages, in one process:

- **Baseline:** a render whose denoiser boundaries are all hashed, with its frames and PCM.
- **Cancellation during each stage:** requested while a native call of that stage is in flight. T2VA: conditioner layer 10, body call 60 (step 1, block 10), video tile 1 and the audio decoder. FL2VA: the first keyframe-encoder call. The in-flight call must be that stage's, must join before the cancellation error returns, and no native call may begin afterwards; the session returns to ready, and a recovery render in the same session must reproduce the baseline: every trajectory hash, the frames and the PCM.
- **Unload during an active denoiser call:** the active render completes and matches the baseline, every call joins before `unload()` returns, none begins afterwards, and a later `generate` is refused.
- **Reopened session:** a new session renders the baseline again, and memory 0, 1 and 5 s after its unload stays within 512 MiB of the first session's.

Validation runs at 288×288×22 on profile `5af39db5…` pass every check: FL2VA 15 of 15, and T2VA 34 of 34, run as two shorter GPU holds with `--scenarios` (the conditioner and denoiser cancellations; then the video-tile and audio cancellations, unload and reopen). Every recovery, the active render under unload and the reopened render reproduce the baseline's 457 trajectory hashes, frames and PCM. After each unload Metal settles to about 2 MiB within 5 s. The qualifying runs (profile `5af39db5…`, on the build pinned by the first 2026-10-28 memory gate; the final build differs from it only in the probe's GPU-lease parsing for `h3-memory-render`, an empty-table guard in the padding, `H3ModulationPadding` becoming internal, and the memory-gate script's carried-record checks, none of them on the lifecycle path) pass all 49 checks: T2VA 13 and 21 in two holds, FL2VA 15 ([record](evidence/h3-session-lifecycle-2026-10-28.json)). The host was busy during them, which does not affect these checks.

**Media audit.** `Scripts/audit-h3-media.py` passes on the Turbo8 T2VA and FL2VA 960×544×124 renders: 124 frames at 24 fps with no constant, black, white or frozen frames, and PCM of the right length with no clipping beyond a few samples in a million. The T2VA render is the qualified replay under `5af39db5…`; the FL2VA render is the qualified replay on the padded build under the in-place profile `c49d8dbf…`, whose content is identical (the identities differ only in paths). The T2VA audio is very quiet, as its prompt asks. The 544×960 gate renders keep no media, so the audit covers 960×544 only. The user's subjective audio and video review on 2026-10-28 approved the renders ([record](evidence/h3-media-audit-2026-10-28.json)).

## Turbo4 and video-decode tiling measurements (0.0.3)

**Turbo4 stays experimental in 0.0.3, outside measured admission.** The user's side-by-side review on 2026-10-28 judged it worth qualifying properly, which is planned for 0.0.4. The profile already carries Turbo4 tables, so a Turbo4 render needs no export: `h3-profile-dump ... turbo4` and `h3-profile-run --render`. At 960×544×124 a Turbo4 render takes about 227 s for T2VA and 242 s for FL2VA. Both pass the media audit. Against Turbo8 on the same seed and prompt, frame PSNR is 12–15 dB and luma SSIM 0.41–0.49, but the two schedules give different clips (the same scene and bird in a different pose), so those numbers measure difference, not quality; quality is a subjective review of the side-by-side clips. Turbo4 has no measured-admission entry (its execution identity ends in `turbo4`) and no CPU trajectory gate, so in 0.0.3 it is outside measured admission ([record](evidence/h3-turbo4-and-decode-tiling-2026-10-28.json)).

**Video decode is GPU- and per-call-bound.** `h3-profile-decode` on the T2VA 960×544×124 render takes 34.2 s: the first tile 5.2 s (graph load and specialization), 104 warm tiles at 0.244 s each, stitching 3.1 s and RGB8 conversion 0.4 s. A thread sample during the tile loop shows the decoding task in Kit host work about 11% of the time and otherwise waiting on the GPU, with MPSGraph encoding each tile call on several threads. Overlapping host stitching with the next tile could save at most about 4 s (about 1% of a render), so 0.0.3 makes no host change; cutting the per-call cost (fewer, larger tiles or several tiles per call, which needs an export) and the first-tile load is 0.0.4 work.


## Last-frame and first+last keyframes (2026-10-29)

**Status: implemented, qualification pending.** The 960×544 trajectory gate has references for step 0 only; steps 1 and 7 are pending a rerun. Every record below is labelled "text-only presentation (known deviation, PR #76)": see the presentation paragraph.

`H3VideoSession` accepts `H3VideoRequest.lastFrame`, alone or with `firstFrame`. A clip can therefore end on a chosen picture, or start and end on chosen pictures. The model card lists the FL2VA checkpoint as taking zero, one or two images. The pinned Diffusers pipeline runs all of these as the single `fl2va` task:

- **Layout:** there is one condition block per keyframe, in packed order (first, then last), between the text and the audio. Upstream places the last anchor at `n + Σspans − 5/3`, 5/3 before the end of the generated span. `H3Layout` already built this, and its fixtures match the pipeline for both layouts.
- **Noise:** each condition is noised to `t = 0.999` by its own draw (`MiniMaxH3PrepareConditionLatentsStep`). The draws come in packed order, before the noise for the generated rows. `H3Geometry.initialLatents(seed:conditions:)` draws Kit's stream in that order, and `initialLatents(seed:firstFrameCondition:lastFrameCondition:)` is its convenience form. A first frame alone draws exactly the stream the first-frame path always drew; a test will pin its hashes against the pre-change stream.
- **Timesteps:** every condition row is held at `max(t, 0.999)`, so a last-only or first+last request has the same distinct row timesteps as a first-only one. The profile's `fl2va` step tables apply unchanged, and `H3Profile.steps` still checks them bit for bit against the layout.
- **Encoding:** both keyframes use the same single-frame encode (`encode_vae_condition`, seed 42), so the tiled first-frame graphs encode the last frame too. The session runs one `encodingKeyframe` stage for both, first then last, and observes cancellation between the two encodes.

No export, step table or download is new, and the work stays inside profile `5af39db5…`. A last-only request has the same row count as a first-only one, so the latents now declare their anchors (`H3Latents.anchors`) and the denoiser builds its layout from them, not from the row count. A hand-built first-frame `H3Latents` needs `anchors: [.first]`.

**Conditioner presentation (known deviation, [PR #76](https://github.com/bubstackdev/aether-kit/pull/76)).** Upstream's `MiniMaxH3FL2VATextEncoderStep` puts a `"<Picture i>: "` label and a Qwen3-VL vision block (`vision_start`, N × `image_pad`, `vision_end`, with the keyframe's pixels) in front of the prompt for each keyframe, tagged as video. Kit's conditioner presents the prompt alone for every FL2VA request: first, last and both. The graph and trajectory gates did not catch this, because both the native run and the CPU oracle were fed the same text-only ids. Last-frame conditioning keeps the text-only presentation, consistent with the first-frame path. The fix is tracked separately.

**Memory and time** ([record](evidence/h3-last-frame-memory-2026-10-29.json)). These are `h3-memory-render` runs on profile `5af39db5…`, Turbo8, seed 1043. The first frame is the crane keyframe, and the last frame is a synthetic camera push-in: the centre 60% of the same picture. The host was busy, so the times are not a quiet reference.

| Stage, Metal / footprint GiB (wall s) | 288×288×22 | 960×544×124 |
| --- | --- | --- |
| Conditioner | 1.83 / 2.02 (20.1) | 1.83 / 2.01 (20.6) |
| Keyframe encoder, both frames | 2.05 / 1.99 (2.9) | 2.05 / 1.89 (8.6) |
| Denoising | 2.77 / 2.77 (49.8) | **12.08** / 12.55 (413.2) |
| Video decode | 5.04 / 1.85 (8.2) | 5.93 / 8.04 (43.2) |
| Audio decode | 0.72 / 0.78 (4.5) | 0.88 / 1.71 (5.5) |
| Render | 85.5 s | 491 s |

At 960×544, a first+last render peaks at 12.08 GiB Metal, against 11.84 GiB for a first frame alone (2026-10-28 gate). Each keyframe adds 510 rows: first-only has 19,843 rows, first+last 20,353, and T2VA 19,333. The bundled measured entry's denoising reservation (16.12 GiB) covers this. The keyframe-encoder stage, encoding two frames, peaked at 2,203,959,296 B, which is 2.34 MiB over the entry's previous 2,201,501,696 B. The entry's `encodingKeyframe` peak is raised to the measured value and cites this record. The entry stays keyed to `5af39db5…`.

**Keyframe adherence.** In the 960×544 first+last render, the first frame matches the first keyframe at 31.9 dB PSNR and the last frame matches the last keyframe at 34.8 dB. The last frame scores 10.3 dB against the first keyframe. At 288×288 the figures are 32.5, 30.8 and 13.8 dB.

**Trajectory gates.** `qualify-h3-profile-trajectory.py` takes `--last-keyframe`. For FL2VA it also compares the captured step-0 condition rows with the pipeline's own VAE encode and noising, in anchor order (`step0/conditions`).

- **960×544×124 first+last** (20,353 rows; steps 0, 1 and 7 captured): the capture and its replay agree on all 457 boundary hashes ([record](evidence/h3-last-frame-trajectory-960-capture-replay-2026-10-29.json)), and the layout and step-plan checks against the pipeline pass. Step 0 passes every comparison against the bf16 CPU modules ([record](evidence/h3-last-frame-oracle-960-step0-interrupted-2026-10-29.json)):

  | Comparison, step 0 | Relative L2 |
  | --- | --- |
  | Condition rows | 0.021% |
  | Packing | 0.0015% |
  | Block 0 | 0.59% |
  | Block 49 | 1.15% |
  | Video heads | 0.17% |
  | Audio heads | 0.19% |
  | Scheduler updates | byte-identical |

  First-only at the same step scored 0.60% and 1.12% for blocks 0 and 49 (bf16; 0.70% and 1.03% against float32), so first+last is on par. The oracle run was stopped for a host pause after step 0. **Steps 1 and 7 are pending**, and the feature is not qualified until they pass the same 0.02 bar.
- **288×288×22** (all eight steps): the replay is exact, the condition rows measure 0.023%, and packing, block 0, heads and scheduler updates pass. Block 49 exceeds the 0.02 bar.
  - First+last: 3.4–4.3% against bf16, 3.8–4.3% against float32.
  - First-only control: 3.9–4.8% against bf16, 4.4–5.0% against float32. It ran on the identical harness, steps, seed and bar, without the last frame.
  - The excess is therefore a property of this small canvas (771 and 852 rows), not of last-frame conditioning. The bar is qualified at 960×544. Records: [first+last](evidence/h3-last-frame-trajectory-288-first-last-2026-10-29.json), [first-only control](evidence/h3-last-frame-trajectory-288-first-only-control-2026-10-29.json).
