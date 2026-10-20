# SDK foundations

This document records the foundation milestone at `9f33d86` (PR #10). Subsequent [Krea integration](krea-foundations.md) adds model-family adoption; the historical evidence below remains scoped to the foundation checkpoint.

Implementation starts from `8cad5911c072281c9fcd7caaa254cf61d7d4a43f`, the remote main revision pinned by Atelier on 2026-10-20. Work is isolated on `codex/sdk-foundations`; neither the older dirty checkout nor Atelier's pinned checkout is changed.

## Reconciled baseline

| Capability | Source at the baseline | Evidence boundary |
| --- | --- | --- |
| Krea text generation and prepared conditioning | Implemented | Existing native parity/recovery records cover their named assets and shapes |
| Krea references and refinement | Implemented, including reference prompt preparation and refine start sigma | Reference conditioning has scoped native evidence; this is not general end-to-end reference/refine certification |
| H3 prompt conditioner, keyframes, denoising, video/audio decoders and video session | Implemented in H3ProfileStages/H3VideoSession; double-buffered weights merged | The older interrupted denoising lifecycle record is historical, not the qualification status of every newer profile. Current profile/geometry and complete media/lifecycle records must govern promotion |
| Atelier adapters, local selection, single worker/pool and resource observations | Implemented in the separate Atelier repositories | App gates and app policy remain Atelier responsibilities; no app or physical-device qualification is claimed by this SDK change |
| Generic asset installation and SDK reservations | Added by this slice | CPU/weight-free behavior only; no new native memory or latency claim |
| Stateful language generation | Not implemented | The shared executor continues to reject stateful functions |

Existing README/status narratives precede several merged APIs. This matrix distinguishes source presence from certification; immutable historical evidence is preserved.

## AetherAssets

`AetherAssets` is a separately selectable Foundation/CryptoKit/Darwin product with no Core AI, Metal or tokenizer dependency. `AssetManifest` identifies the asset set separately from its execution contract. Its installation ID is SHA-256 of normalized, sorted-key schema-1 JSON, including relative file names, byte counts and content hashes. Location, receipt time and human-friendly names do not participate. Unknown schema versions or required features, traversal, duplicate names and file/parent collisions refuse before installation.

`LocalAssetStore` streams declared portable files into private staging, hashes the copied bytes, writes bounded metadata and atomically renames the complete directory into place. Source symlinks are refused. Retry verifies existing content and never overwrites it. Failure/cancellation cleans its own stage; previous installed identities survive. Cancellation observed before publication refuses; after the atomic commit the result is success. A process crash can leave an unreferenced stage; automatic crash-stage collection is deferred. Atomic visibility is implemented, not a power-loss durability guarantee.

Acquisition takes a shared advisory lock before metadata/content verification and returns an `AssetReadLease`. Last-owner destruction closes the lock. Install/removal need an exclusive lock; busy fails immediately. Explicit removal can discard a corrupt installation without trusting its metadata, so a verified reinstall remains possible. Locks are separate from removable directories and are never unlinked. The root is app-owned; advisory locks protect cooperating processes, not arbitrary external edits. Acquisition hashes all bytes again and can be expensive.

Use `ModelSession.open(modelPath:in:...)` to bind the reader lease to native model ownership automatically. The model path must be declared in the installation. Higher-level Krea/H3 adapters have not yet adopted the store: callers using them must retain the reader lease through joined unload. There is no automatic Atelier migration or asset duplication.

This first store manages whole portable files. H3's in-place external weight ranges remain under the existing H3 verification contracts; they are not silently copied/imported or claimed to be managed by this store. Shared blob/range leases, device/profile qualification, download transport and catalog trust are subsequent slices.

## Resource admission

`ResourceCoordinator` is an explicitly shared fail-fast ledger. `reserve`, `resize` and `release` are synchronous mutex transactions with overflow-safe arithmetic. The host selects capacity; no global singleton, eviction or competing job queue is introduced. A reservation releases idempotently or on destruction. Reserve overlapping old/new storage before replacement; shrink only after old storage is released.

`ModelResourceAdmission` supplies known caller estimates for model residency (including load peak) and peak run scratch. `ModelSession.open` reserves before loading and rolls back on load failure. `runTensors` reserves before dispatch. The task holds scratch until backend completion; unload waits for it before releasing backend/model residency and installed assets. Multiple sessions share one coordinator. Existing calls without admission preserve their behavior.

These are reservations of estimates, not measured footprint or allocator enforcement. Returned tensors remain caller-owned and their lifetime after a call is not charged by this executor. Krea/H3 sessions and tensor/state growth are not automatically budgeted yet. Atelier keeps scheduling, budget choice, checkpoints, user refusal and content gates.

## Validation and rollout

Run `sh Scripts/check-cpu.sh` for the weight-free Release suite; it clears inherited AETHER opt-ins and external fixture paths. `Examples/Foundations` is an independent consumer requiring no Core AI linkage. CI performs these checks on a host with the declared toolchain; `AETHER_MAC_RUNNER` selects an equipped runner. Native/model qualification remains separately operated.

Before adoption, qualify the exact model estimates and integrate reservations into Krea/H3 stage acquisition, including overlapping buffers and both H3 weight banks. Until then this is opt-in infrastructure, not a cross-modality memory guarantee.

## Executed checkpoint — 2026-10-20

The full weight-free Release suite passes **182 tests**, with **14 opt-in native/external tests skipped** (196 registered). The independent foundation consumer builds and runs with the default Swift build engine; its executable has no Core AI framework dependency. `git diff --check` passes. Existing H3 stream Sendable diagnostics and the native test build-engine deprecation remain. No model/GPU workload ran. See [source-pinned evidence](evidence/sdk-foundations-2026-10-20.json) and the [archived CPU log](evidence/sdk-foundations-cpu-tests-2026-10-20.log). Remote CI and physical iOS validation remain pending.
