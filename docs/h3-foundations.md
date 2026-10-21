# H3 external weight leases and resource admission

This slice starts from merged main `ac544a1` after PRs #10 and #11. It adds optional ownership/admission to the schema-3 profile path. Existing unprotected/unbudgeted entry points remain available. Atelier and its SDK pin are unchanged.

## External checkpoint ownership

`AetherAssets.ExternalFileLease` opens read-only regular-file descriptors and shared advisory locks. It keeps those descriptors rather than reopening by path for each read. A cache tool replacing a path, moving a snapshot symlink or unlinking a source cannot redirect subsequent leased reads: the original inode remains readable until its last lease owner releases it. File descriptors are deduplicated across the same resolved source. No checkpoint is copied, registered or deleted.

`read(into:from:offset:)` uses bounded `pread` directly into caller storage. It rejects unleased paths, out-of-range extents, changed size/mtime, unsupported sources and lock contention. Partial acquisition failure closes already-opened descriptors. Advisory locks coordinate only with cooperating writers; leases cannot prevent arbitrary in-place mutation by other software. Content hashes and the caller's immutable-file contract still apply. Restoring a timestamp is not a content proof. Unlink changes are intentionally permitted because they do not change the original descriptor's bytes. Keeping a descriptor open also delays reclamation of an unlinked file's disk space until the lease is dropped.

`H3TensorFile.holding(_:)` and `H3EmbeddingTable.holding(_:)` retain a lease and validate complete range coverage before use. Tensor range concatenation, BF16 bits and existing SHA-256 checks stay unchanged. Native weight-slot reads still go directly into Metal buffers with no extra full-weight Data copy. The embedding lookup reads pinned rows as before.

`H3Profile.protectingWeights()` returns a value with protected backbone, conditioner, refiner and embedding references. It preserves profile/model identity and source paths. It opens descriptors but does not hash the entire checkpoint or load Core AI. Actual tensor reads retain the existing integrity gates; explicit full verification remains available. Profile graphs, tokenizer and schedule tables remain under their existing caller-owned immutability contract. This is external weight ownership, not a portable H3 installation or a general cache manager. Descriptor-limit failures refuse and roll back rather than changing process-global limits.

A protected profile/stage value owns its leases. Retain it for the session lifetime; release caller-owned copies when no longer needed. `H3VideoSession.unload()` drops its own stages after joining, but cannot release other owners' copies.

## Shared resource admission

`H3ProfileEngine.stages` now reports a resource identity containing the exact profile manifest hash and selected recipe. Custom injected stages can supply a stable identity explicitly. The new throwing `H3VideoSession(stages:resources:)` refuses mismatched identities.

`H3ResourceAdmission` supplies host-qualified operation and per-stage peak estimates, plus pixel, snapped-frame and prompt-byte limits. Required stages are checked before generation; sound and first-frame requests require their corresponding estimates. There are no guessed byte defaults. The same explicit ResourceCoordinator can be shared with Krea or other sessions.

Operation storage is reserved before CPU preparation begins and held through the entire call. A stage peak is reserved before its start event or model acquisition, then returned only after that stage's awaited release. The peak must cover loading/specialization, native scratch and reusable buffers, including both conditioner/denoiser weight banks and any concurrently held graphs inside the stage. Reservations are not released just because the caller cancels. Failure cleanup joins stage release before returning capacity. Concurrent terminal unload callers join the same closing task.

```swift
let pinned = try profile.protectingWeights()
let stages = try H3ProfileEngine.stages(profile: pinned, recipe: recipe, tokenize: tokenize)
let admission = try H3ResourceAdmission(
    coordinator: appResources,
    executionIdentity: H3ProfileEngine.resourceIdentity(profile: pinned, recipe: recipe),
    operationBytes: qualified.operationBytes,
    stageBytes: qualified.stageBytes,
    maximumPixels: qualified.maximumPixels,
    maximumFrames: qualified.maximumFrames,
    maximumPromptUTF8Bytes: qualified.maximumPromptUTF8Bytes)
let session = try H3VideoSession(stages: stages, resources: admission)
// Use the existing generate(request, onEvent:) API, then await session.unload().
```

`qualified` denotes the host's own exact-profile/device measurements, not an SDK-provided estimate service. These reservations admit estimates; they do not intercept each native allocation or guarantee process footprint. CPU intermediates overlapping stage transitions belong in the operation estimate. Caller-retained frames, audio and input/output values outside the operation lifetime are not automatically charged. Native profile/device/memory qualification remains necessary before advertising support.

## Validation scope

Tiny weight-free fixtures cover descriptor pinning across replacement/unlink/symlink changes, shared-lock release, partial-acquisition rollback, in-place change detection, range bounds, tensor order, embedding rows, complete profile weight propagation and unchanged identity. Session fakes cover reservation ordering, stage refusal before acquisition, profile/shape/stage rejection, shared-budget contention and recovery, cancellation during held work or held release, callback failure and concurrent terminal unload.

Synthetic graph declarations are not runnable native profiles and logical reservation counts are not real model byte estimates. No model/GPU workload or native performance claim follows from these checks. The earlier native qualification records remain scoped to their exact sources/assets.

## Executed checkpoint — 2026-10-20

`AETHER_BUILD_SYSTEM=swiftbuild sh Scripts/check-cpu.sh` passed with 207 tests passing and 14 native/fixture opt-ins skipped (221 registered). The [source-pinned record](evidence/h3-foundations-2026-10-20.json), [test output](evidence/h3-foundations-tests-2026-10-20.log) and [build output](evidence/h3-foundations-build-2026-10-20.log) record this CPU-only run. Hosted CI still needs a runner equipped with macOS 27, Xcode 27.1 and Swift 6.4.

The independent `Examples/Foundations` consumer also passed `swift run -c release`, checking verified install, reader protection, removal and reservation release.
