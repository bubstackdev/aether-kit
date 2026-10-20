# Krea asset and resource integration

This slice builds on foundation commit `9f33d86` (PR #10). It is SDK-only: Atelier's pinned checkout, model list, queue, policies and files are unchanged.

## Portable installation

`KreaAssets.installationManifest()` describes the exact portable bundle using the generic asset contract. It includes the six model directories (stream, declaration and metadata), bundle declarations, tokenizer files and the optional vision encoder when declared. Optional `config.json`, `chat_template.jinja` and `chat_template.json` under the tokenizer directory are included because swift-transformers' local loader can consume them. Model stream hashes come from existing declarations; the store verifies copied bytes rather than trusting those declarations. Header/tokenizer drift after opening is refused.

The new installation ID is separate from generation identity. Reopening via `KreaAssets.open(in:)` checks the complete portable installation contract and preserves the original text/reference fingerprints. No source metadata is rewritten. Portable installation refuses source symlinks/aliases that would require rewriting paths. Registered external specializations and external refine encoders are not imported.

```swift
let source = try KreaAssets.open(at: sourceBundle)
let store = try LocalAssetStore(root: appAssetsDirectory)
let id = try await store.install(source.installationManifest(), from: source.root)
let session = try KreaGenerationSession(installed: try await store.acquire(id))
// Use the existing prepare/generate API.
await session.unload()
```

The installed initializer is lazy: it does not load native models. Its private factory holds the read lease until terminal joined unload; every opened ModelSession also retains installation protection. Keeping a separate lease alive continues protecting the installation after session close, as intended. Undeclared external model paths refuse. Existing caller-managed local-bundle initialization remains supported.

Import explicitly duplicates portable files into managed storage and can require tens of GB. It is never an automatic migration. H3 in-place checkpoint ranges remain outside this store's managed-file scope.

## Resource profiles

`KreaResourceAdmission` combines one host-selected shared ResourceCoordinator, an operation-storage estimate, exact model estimates and image-token/conditioning-row limits. Model estimates are keyed by stream plus metadata hash, never by display name or path. Unknown model bytes and shapes outside the profile refuse. There are no invented default byte estimates: the caller must supply qualified maxima.

```swift
let modelEstimates = try ([assets.textEncoder, assets.decoder] + assets.denoisers).map { asset in
    // Look up this exact stream/metadata pair in the host's qualified estimate table.
    try KreaModelResourceEstimate(asset: asset,
                                  residentBytes: measuredProfile.residentBytes(for: asset),
                                  runScratchBytes: measuredProfile.scratchBytes(for: asset))
}
let admission = try KreaResourceAdmission(
    coordinator: appResources, operationBytes: measuredProfile.operationBytes,
    maximumImageTokens: measuredProfile.maximumImageTokens,
    maximumConditioningRows: measuredProfile.maximumConditioningRows,
    models: modelEstimates)
let session = KreaGenerationSession(assets: assets, resources: admission)
```

The profile lookup above is host pseudocode, not another SDK service. Include vision, external refine and specialized models in the estimate set when using those paths. Each estimate must cover loading/specialization peaks as well as retained weights and native execution/copying, throughout the admitted shape range. The operation estimate must cover overlapping host carries and intermediate tensors.

A factory decorator reserves residency before opening a model and rolls it back on load failure. Its session wrapper reserves per-call scratch before native submission and retains it through completion. Existing stage policies release residency on joined unload; partial denoiser-load failure closes previously opened models. Operation storage is reserved before running preparation/generation/refine and held through callbacks and submitted work. Two Krea sessions can share the same coordinator without introducing another job queue or eviction manager.

These are reservations of estimates, not measured process footprint, OS allocation enforcement or a production memory bound. Inputs and returned/prepared tensors retained by callers are not automatically charged outside the operation lifetime. Core AI cache storage is separate. Native/device qualification and real profile estimates remain required before a host advertises memory support.

## Validation scope

CPU tests cover text/reference installation round trips, optional tokenizer preservation, fingerprint stability, installed-session reader lifetime, declaration drift/copy corruption, incompatible contracts, model/operation/scratch refusal, partial-load rollback, cancellation/unload joins, and contention/recovery across two sessions. Scripted models use tiny logical reservation counts; these are not real model memory estimates or native performance evidence.

The initial foundation PR's remote CI stops at its preflight because the hosted runner has Xcode 26.6 / Swift 6.3.3. Configure `AETHER_MAC_RUNNER` with macOS 27 / Xcode 27.1 / Swift 6.4 before treating remote checks as executed SDK validation.

## Executed checkpoint — 2026-10-20

The complete standard-engine Release suite passes **194 tests**, with **14 opt-in native/external tests skipped** (208 registered). This includes cancellation while opening a model: admission remains charged until the opener joins and the returned model is closed. `git diff --check` passes. [Source-pinned evidence](evidence/krea-foundations-2026-10-20.json) retains separate test/build logs. Existing H3 stream Sendable diagnostics remain; no model/GPU workload or native performance qualification ran.
