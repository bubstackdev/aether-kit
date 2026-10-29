# Model readiness

Status: implemented and qualified 2026-10-29: weight-free CPU tests, and runs on the Mac and an iPhone
([Qualification](#qualification)). This is the second piece of the 0.0.4 release gate, app readiness
([roadmap](roadmap-0.0.4.md#release-gate-app-readiness)), after [model preparation](model-preparation.md). The user decided
the four choices below on 2026-10-29. Type names may still change once the API has run in the sample app.

## Quick start

```swift
import AetherAssets
import AetherLanguage

let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: entryData)   // shipped in the app
let preparation = ModelPreparation(store: try LocalAssetStore(root: modelsDirectory), transport: .urlSession)
let readiness = ModelReadiness(entry: entry, preparation: preparation, check: .language)

for await state in readiness.states {
    switch state {
    case let .notInstalled(present, total): offerDownload(resumingFrom: present, of: total)  // then readiness.prepare()
    case let .installing(progress): show(progress.phase, progress.fraction)
    case .verifying: show("Checking the model")
    case let .ready(lease): model = AetherLanguageModel(installation: lease)
    case let .unsupported(reason): hideFeature(because: reason)
    case let .failed(failure): show(failure, canRetry: failure.isRetryable)                  // retry: readiness.prepare()
    }
}
```

## Decisions (user, 2026-10-29)

1. **Scope:** generic. The state type lives in `AetherAssets` and works for any catalog entry. Each product supplies
   its own check (`ModelRuntimeCheck`). The language facade adopts it in 0.0.4; Kokoro, Krea and FLUX.2 can follow
   without a second design.
2. **Observation:** an async stream. `current` is a snapshot and `states` yields the current state, then each change,
   in the style of model preparation's events. A SwiftUI app wraps it in a few lines of `@Observable`.
3. **Memory:** declared in the catalog entry. An entry may carry a memory requirement, compared with the device before
   anything downloads. Without one, nothing is checked up front.
4. **Ready** means installed, verified (the read lease is held) and runnable here. The first request, or `prewarm()`,
   loads the model. Loading, idle unloading and memory pressure belong to managed lifecycle, the next gate piece, so
   readiness doesn't change when a model unloads.

## States

| State | Meaning | What the app can do |
| --- | --- | --- |
| `notInstalled(bytesPresent:bytesTotal:)` | Not installed; `bytesPresent` are downloaded bytes a preparation resumes from. | `prepare()` |
| `installing(Progress)` | Preparing: `checking`, `downloading` or `installing` (copying and hashing into the store), with bytes done and total for that phase. | `cancel()` |
| `verifying` | Hashing the installation before it's handed out. | Wait. |
| `ready(AssetReadLease)` | Verified and runnable. Open the model from the lease. | Open it; `uninstall()` |
| `unsupported(Unsupported)` | `operatingSystem`, `platform`, or `memory(requiredBytes:availableBytes:)`. | Hide the feature. |
| `failed(Failure)` | `preparation(PreparationError)`, `verification(String)` or `invalidModel(String)`, each with `isRetryable`. | `prepare()` if retryable |

`availability` on `AetherLanguageModel` stays the no-load check of a bundle. Readiness covers the steps before a
model exists: whether it can run here, whether it's installed, and whether its bytes are intact.

## How a state is decided

Creating a `ModelReadiness` never touches the network and never hashes before it knows the model can run:
1. **The product's entry check.** `.language` returns `operatingSystem` below macOS 27 or iOS 27, and `invalidModel`
   for an entry whose installation contract isn't a language bundle.
2. **The entry's platforms.** An entry for another platform is `unsupported(.platform)`.
3. **The entry's memory requirement**, against the device (below).
4. **The store's status.** Installed means `verifying` at once: `LocalAssetStore.acquire` hashes every byte, as it
   always does, and returns the read lease. Otherwise `notInstalled`, with any bytes a previous run left.
5. **The product's installation check**, on the verified installation. `.language` is the facade's own
   `availability` for the installed registration, so ready means `AetherLanguageModel(installation:)` reports it
   available. A missing chat template, for example, fails as `invalidModel`.

Steps 1–3 come first, and `prepare()` checks them again before it starts. So a model ruled out by the product's check
or by the entry's requirements is never downloaded, and never verified if it's already installed. What only the
installation can show (step 5) is found after downloading and verifying. That includes the platform of an entry that
doesn't declare `platforms`: an iOS language entry without them is downloaded on a Mac before it's reported
`unsupported(.platform)`, and `refresh()` verifies it again each time. Entries from `catalog-entry` declare the
platform for language bundles.

## Requirements and memory

```swift
public struct ModelRequirements: Codable, Sendable, Equatable {
    public enum Platform: String, Codable, Sendable { case macos, ios }
    public let platforms: [Platform]?   // nil: any
    public let memoryBytes: UInt64?     // nil: unchecked
}
```

`ModelCatalogEntry` gains an optional `requirements`. Entries written before it have none and decode unchanged; the
installation ID doesn't depend on it. `aether-probe catalog-entry` writes the registration's platform for a language
bundle, and `--memory-bytes BYTES` declares the memory requirement.

**`memoryBytes` is the model's qualified peak physical footprint** while it runs: loaded, and serving a request at its
registered context. It's the figure iOS enforces an app's memory limit against, taken from the model's device
qualification record. For the Qwen3 0.6B iOS bundle that's 220 MiB, the 219.2 MiB peak of the
[0.0.3 iPhone run](evidence/language-iphone18-qwen3-0.6b-2026-10-28.json) rounded up. Weights that Core AI maps from
the file aren't part of that footprint. Refusing an open whose mapped bytes won't fit is the hardening track's iOS
open-time check ([roadmap](roadmap-0.0.4.md#parallel-track-hardening-and-coverage)).

`ModelReadiness.deviceMemoryLimit` is what the requirement is compared with:
- **iOS:** the app's memory limit, which is its current footprint plus `os_proc_available_memory()`. It includes the
  increased-memory-limit entitlement when the app has it. When iOS reports no limit (`os_proc_available_memory()` is 0
  for a process that isn't an app), the limit is nil and memory requirements go unchecked.
- **Mac,** and an iOS app running on a Mac: physical memory.

The limit is read when readiness is created, on `refresh()`, and on `prepare()` from `notInstalled` or `failed`. An app's
limit can change while it runs, but `unsupported(.memory)` stays until the app calls `refresh()`.

Platform names Kit doesn't know (say `visionos`) decode, and match no device, so a newer entry stays readable. An
unknown key inside `requirements` is ignored. A future requirement that older Kits must not ignore needs a new entry
`schemaVersion`, which they refuse.

## Actions

- **`prepare()`**, from `notInstalled` or `failed`: checks steps 1–3 again, then runs `ModelPreparation.prepare` and
  verifies the result. From `failed(.verification)` it replaces the corrupt installation, as preparation does. After
  `failed(.preparation)` it runs again whether or not the error is retryable, for example once the app has freed disk
  space. It does nothing while installing, verifying or ready, when unsupported (`refresh()` decides again), or after
  `failed(.invalidModel)`, where the same model would fail the same way. With `BackgroundTransport`, call it at launch
  to join a transfer that continued while the app wasn't running.
- **`cancel()`** stops a running preparation. Downloaded bytes stay, and the state returns to `notInstalled` with them.
- **`refresh()`**, from `notInstalled`, `unsupported` or `failed`: decides again. For example after another process
  installed the model, or to verify an installation again without downloading.
- **`uninstall()`** removes the installation and any downloaded bytes. Readiness drops the lease it holds first, so the
  state reads `notInstalled` while the removal runs. A `prepare()` in that window starts once the removal ends. If a
  model opened from the lease is still alive, removal fails with `PreparationError.busy` and the installation is
  verified again. It also fails with busy while installing, verifying or already uninstalling. A background transfer
  the system still runs for the entry isn't cancelled, and may leave a partial file when it lands; the next
  `uninstall()` or `prepare()` deals with it.

`states` uses the newest-value buffer: a slow reader sees the latest state, not every progress step. Every reader
sees the current state first. Releasing the readiness cancels its preparation or verification (downloaded bytes stay
for the next one) and ends its streams.

## For other products

```swift
let speech = ModelRuntimeCheck(
    entry: { entry in entry.manifest.executionContract == "kokoro.speech-bundle/1" ? .runnable : .invalid("not Kokoro") },
    installation: { _ in .runnable })
```

`ModelRuntimeCheck.none` checks only the entry's requirements. Both closures must be cheap: no hashing, no loading.
They run outside readiness's lock, so they may read `current`. They must not call `prepare()`, `cancel()`,
`refresh()` or `uninstall()`, which run the checks themselves.

## Qualification

[Record](evidence/model-readiness-2026-10-29.json). Nothing downloaded in any run.

**Mac** (M5 Max, 64 GB, macOS 27.2), with `aether-probe readiness` against the store and entries of the
model-preparation qualification:
- The installed Qwen3 0.6B macOS entry went `verifying` → `ready` in 0.44 s, hashing all 11 files (1,208,370,450 bytes).
- The same entry with a 64 GiB + 1 byte requirement was `unsupported(memory)` against 68,719,476,736 bytes of physical
  memory, decided in 0.46 ms without verifying.
- The iOS entry, rewritten by `catalog-entry` with `platforms: ["ios"]` and 220 MiB, was `unsupported(platform)` with
  `--prepare`, and nothing downloaded.

**iPhone** (iPhone19,4, iOS 27.2, no increased-memory-limit entitlement), with the `readiness` mode of
`Examples/LanguageDevice`: `BackgroundTransport` and `.language`, over the store holding the installation from the
[model-preparation iPhone record](evidence/model-preparation-iphone-2026-10-29.json). The entry was the iOS one above,
with its requirements.
- The app's memory limit read 3,539,992,576 bytes, the same in both launches.
- The installed model went `verifying` → `ready` in 0.49 s, hashing all 1,208,370,448 bytes. The model opened from the
  ready state's lease and answered the hash-function prompt in 4.8 s (greedy, 48 tokens).
- With the requirement replaced by 8 GiB, it was `unsupported(memory(requiredBytes: 8589934592, availableBytes:
  3539992576))` before verifying.

## Open questions

- A fresh installation is hashed twice, once by the install's copy and once by the lease. That costs about 0.4 s per
  1.2 GB on the Mac. Preparation could hand over the lease it already verified.
- Whether `verifying` should report progress. It would matter for multi-GB models on older iPhones.
- Core AI's specialization cache: an installed model that isn't specialized yet pays a first-open compile. That
  belongs to managed lifecycle, which owns loading.
