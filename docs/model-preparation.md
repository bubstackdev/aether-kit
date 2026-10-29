# Model preparation

Status: implemented, 2026-10-28; qualified 2026-10-29. Weight-free CPU tests, a real-network run on the Mac and an
iPhone background transfer pass ([Qualification](#qualification)). This is the first piece of the 0.0.4 release gate, app readiness
([roadmap](roadmap-0.0.4.md#release-gate-app-readiness)). The user decided the four choices below on 2026-10-28. The API
is frozen for 0.0.x ([API stability](api-stability.md)), and the [sample app](sample-app.md) uses it.

## Quick start

```swift
import AetherAssets
import AetherLanguage

// An entry Kit's tools wrote for the bundle (aether-probe catalog-entry), shipped in the app.
let entry = try JSONDecoder().decode(ModelCatalogEntry.self, from: entryData)
let store = try LocalAssetStore(root: applicationSupport.appendingPathComponent("Models"))
let preparation = ModelPreparation(store: store, transport: .urlSession)

if case .installed = try preparation.status(entry) {} else {
    for await event in preparation.prepare(entry) {
        switch event {
        case let .downloading(_, _, _, done, total): show(progress: Double(done) / Double(total))
        case let .failed(error): show(error, canRetry: error.isRetryable); return
        default: break
        }
    }
}
let model = AetherLanguageModel(installation: try await store.acquire(entry.installationID))
```

For downloads that should continue while an iOS app is suspended, use `BackgroundTransport(identifier:)` instead of
`.urlSession`, and forward the app's background-session events (see [Transport](#transport)).

## What exists

`AetherAssets` already does the hard part of installation:
- `AssetManifest` names a model's portable files by path, byte count and SHA-256. Its canonical JSON hashes to an
  `installationID`, so the same files under the same identity always install to the same place.
- `LocalAssetStore.install` copies and hashes into a private stage, then publishes it atomically. A retry verifies the
  existing installation instead of overwriting it.
- `acquire` returns an `AssetReadLease`, a shared lock that keeps an installation from being removed while a model
  uses it.
- `removeAbandonedStages` cleans up after a crash.

Krea, FLUX.2 and Kokoro each build their `AssetManifest` from a local bundle (`installationManifest()`), and their
sessions open from a lease (`init(installed:)`). The language facade doesn't: `AetherLanguageModel(bundle:)` opens a
plain folder.

What's missing is everything before `install`. An app can't fetch a model, can't show progress, can't resume an
interrupted transfer, and can't find out before starting whether the device has room.

## Decisions (user, 2026-10-28)

1. **Download:** a pluggable transport plus built-in resumable HTTPS. That includes **iOS background transfers**, so
   multi-GB downloads continue while the app is suspended.
2. **Source:** an app prepares from a **pinned catalog entry**: identity, family, and every file's path, size and
   SHA-256, plus where to get them. Nothing is trusted unless the entry pins it.
3. **Hand-off:** preparation installs into `LocalAssetStore`. The facade and the sessions open from the installation and
   hold its read lease for their lifetime. Bundle URLs keep working.
4. **Hugging Face:** a resolver for pinned revisions only. Branch names are refused.

## The catalog entry

```swift
public struct ModelCatalogEntry: Codable, Sendable {
    public let schemaVersion: Int                // 1
    public let manifest: AssetManifest           // identity, execution contract, files (path, byteCount, sha256)
    public let source: ModelSource
    public let displayName: String?
}

public enum ModelSource: Codable, Sendable {
    case https(base: URL)                        // file URL = base + path; HTTPS only
    case huggingFace(repository: String, revision: String, subdirectory: String?)  // revision: a 40-hex commit
}
```

- **The entry is the trust root.** Its manifest is exactly what `LocalAssetStore` installs, so the `installationID`
  is known before any byte arrives, and the app can ask whether the model is already installed without touching the
  network.
- **One manifest per family.** The families' existing `installationManifest()` functions define which files count.
  Language gains one (`LanguageModelManifest.installationManifest(in:)`): the registration, the model asset(s), the
  tokenizer files, and the vision asset for Qwen3-VL.
- **Kit writes the entries.** `aether-probe catalog-entry FAMILY BUNDLE --https BASE` (or `--huggingface
  OWNER/NAME@REVISION[/SUBDIRECTORY]`) reads an exported bundle, builds the family's own installation manifest and prints
  the entry. The hashes come from the files the exporter produced, through the same code a local install uses, so the
  entry's installation ID always matches. Families: `language`, `kokoro`, `flux2` and `krea`.
- **Hugging Face resolves to** `https://huggingface.co/<repository>/resolve/<revision>/<subdirectory>/<path>`. A
  revision that isn't a full commit hash is refused when the entry is decoded. The app may supply a bearer token for
  gated or private repositories through `ModelPreparation.huggingFaceAuthorization(token:)`, which sends it only to
  huggingface.co. Kit never stores it. A background task that carries it is held by the system's background-session
  store while the task exists, and Kit keeps no resume data for such a task, since resume data holds the request.

## Preparing

```swift
public struct ModelPreparation: Sendable {
    public init(store: LocalAssetStore, transport: any ModelTransport, headers: @escaping @Sendable (URL) -> [String: String] = { _ in [:] })
    public func status(_ entry: ModelCatalogEntry) throws -> PreparationStatus          // no network, no hashing
    public func prepare(_ entry: ModelCatalogEntry) -> AsyncStream<PreparationEvent>
    public func install(_ entry: ModelCatalogEntry) async throws -> String              // prepare, result only
}

public enum PreparationStatus: Sendable {
    case installed(installationID: String)
    case partial(bytesPresent: UInt64, bytesTotal: UInt64)   // resumable download data on disk
    case notInstalled(bytesTotal: UInt64)
}
```

`prepare` returns a stream of events that ends in exactly one terminal event:
- `.checking`: free space and existing data;
- `.downloading(path:, fileBytes:, fileTotal:, bytesDone:, bytesTotal:)`. Progress is coalesced to at most one event
  per 100 ms, and each file's final count is always delivered;
- `.installing(bytesDone:, bytesTotal:)`: the store's copy and hash pass, then the atomic publish;
- terminal: `.installed(installationID:)`, `.failed(PreparationError)` or `.cancelled`.

`headers` supplies per-request headers, such as a Hugging Face bearer token for a gated repository.

Cancelling the consuming task cancels the run. Already downloaded bytes stay on disk for resumption.

**Order of work:**
1. **Check.** If the installation exists, verify it as `install` does and finish. Otherwise sum the bytes still
   missing and compare them with the volume's available capacity for important usage. Refuse with
   `.insufficientSpace(needed:, available:)` before downloading anything.
2. **Download** each missing file into a per-installation download area in the store (`downloads/<installationID>/`).
   It's kept across launches, unlike install's temporary stage. Each file's size is checked when it completes. A file
   of the wrong size is deleted and reported as `.integrityMismatch(path:)`.
3. **Install** by handing the completed download area to `LocalAssetStore` as the source. The store's own copy and hash
   pass stays the single point where bytes become trusted, and it reports progress as `.installing`. A file that
   doesn't hash to the entry is deleted, so the next run fetches it again, and the run fails with
   `.integrityMismatch(path:)`. Hashing once, at install, avoids a third read of multi-GB files. Installing by rename instead of copy, to avoid doubling
   disk use, is a follow-up once it has its own tests.
4. **Clean up.** The download area is removed after a successful install. `removeAbandonedStages` also removes download
   areas for installations that already exist.

Two runs for the same entry share a per-installation download lock: the second reports `.busy` and never downloads
twice. The exception is a run this process cancelled that is still unwinding: a new run waits for it (up to 30 s), so
cancelling and retrying at once works. After taking the lock, a run checks again whether the installation now exists
(another run may have published it meanwhile) and if so verifies it instead of downloading. An existing installation
that fails verification is removed and prepared again; while readers hold it open, the run reports `.busy`.

## Transport

```swift
public protocol ModelTransport: Sendable {
    /// Writes `request.url` to `destination`, appending from `destination`'s current size when `resume` is true.
    func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
               progress: @Sendable (UInt64) -> Void) async throws
}
```

- **`.urlSession` (foreground)** uses a data-task stream. It resumes with an HTTP `Range` request and checks the
  response: 206 with a matching `Content-Range` appends, 200 or a 206 from byte 0 restarts the file, anything else fails.
- **`BackgroundTransport(identifier:)` (iOS, and macOS apps)** uses a background `URLSession` with download tasks, one
  per file, so the system carries transfers on while the app is suspended or terminated.
  - **Redirects:** a background session follows redirects itself and never asks a delegate, so the app's headers
    would travel with them. The transport resolves redirects first, in the foreground. It follows each hop on the
    original host with the app's headers, and stops at the first other host. Each hop is probed with the GET the
    task will send (a one-byte range, abandoned once the headers arrive), not a HEAD, which a server may answer
    differently. The background task then fetches the final URL, carrying the headers only if it's still on the
    original host. A task keeps that URL, and a CDN's signed link can expire before a long-suspended task reaches it,
    so a `403` or `410` retries once with the redirect resolved again and without resume data.
  - **Resuming:** a partial file resumes with a `Range` request, as in the foreground. A `206` that continues the file
    is appended, a `200` or a `206` from byte 0 replaces it, and a `416` restarts it once. A background task writes
    nothing to the download area until it finishes. A stopped or failed task therefore keeps the system's resume data
    beside the file, and the next task for the same bytes continues from it (except a task carrying the app's
    headers, above). Those bytes don't count in `status`. A
    resumed task's response range starts where it stopped, but its file holds every byte since the record's offset,
    so the transport works out where the file starts from the range's end and the size received.
  - **Where bytes go:** each task records its destination (relative to the home directory, which survives an iOS app
    update) in its description. A task that finishes while the app was relaunched only to receive it still lands in
    the download area, and the next run finds the file complete.
  - **No duplicates:** a `fetch` for a file whose task is still running joins that task instead of starting another.
  - **Lifecycle:** the app connects the transport in one place:
  ```swift
  // UIApplicationDelegate, or SwiftUI's .backgroundTask(.urlSession(identifier))
  BackgroundTransport.handleEvents(identifier: identifier, completionHandler: completionHandler)
  ```
- **The app's own transport:** anything that conforms, for CDNs with signed URLs, a custom authentication scheme, or
  tests.

Every transport refuses non-HTTPS URLs, except `file:` URLs, which the test transport accepts for local fixtures.

## Hand-off to the facade and sessions

```swift
let lease = try await store.acquire(installationID)                // verifies and locks
let model = AetherLanguageModel(installation: lease)               // new; holds the lease
let speech = try KokoroSpeechSession(installed: lease)             // existing
```

- `AetherLanguageModel(installation:)` reports availability from the leased directory, exactly as
  `init(bundle:)` does for a folder. The model keeps the lease while loaded and while any session it created is alive.
- `init(bundle:)` stays for apps that manage their own folders.
- **Readiness**, the next gate piece, builds on `status` and the run's events: [model readiness](model-readiness.md)
  turns them into one state per model, ending in a ready state that holds the lease.

## Errors

`PreparationError` is one enum with these cases: `insufficientSpace(needed:, available:)`, `integrityMismatch(path:)`,
`http(status:, path:)`, `network(underlying:)`, `invalidEntry(String)`, `busy` and `fileSystem(Int32)`. Each says
whether retrying can help.

## Qualification

- **Weight-free CPU tests in CI.** A local test transport and a scripted `URLProtocol` cover:
  - resume after a cut connection (206), restart on 200, refusal of a wrong `Content-Range`, restart after a 416;
  - size and hash mismatch, with the bad file dropped and refetched;
  - insufficient space, refused before any download;
  - cancellation mid-file, then an immediate retry that resumes;
  - two concurrent runs for one entry (the second is busy);
  - the Hugging Face URL, revision and authorization rules, and redirects that drop headers at a new host, including
    the background transport's foreground redirect resolution (GET probes that stop at the headers) and that no
    resume data is kept for a task carrying a token;
  - bodies larger than the pinned size, stopped before they can fill the disk;
  - a corrupt installation, removed and prepared again;
  - an installed entry finishing without network;
  - what a finished background task does with its bytes (append after a matching 206, including a task resumed from
    resume data; replace after a 200 or a 206 from byte 0).
- **Mac, real network (2026-10-28, [record](evidence/model-preparation-network-2026-10-28.json)).** A catalog entry
  for the Qwen3-0.6B Kit bundle (11 files, 1,208,370,450 bytes) in a private Hugging Face repository, pinned to its
  commit:
  - the first run was stopped after 300,009,223 bytes, and status then reported exactly that much kept;
  - a retry resumed from there (its first progress event was at 300,025,598 bytes), downloaded the rest and installed;
  - `AetherLanguageModel(installation:)` opened the installation and gave a coherent greedy reply in 5.96 s.

  This run found and fixed one bug: a retry straight after stopping a run reported busy, because the stopped run hadn't
  yet released its lock. A stopped run is now waited for.
- **iPhone, background transfer (2026-10-29, [record](evidence/model-preparation-iphone-2026-10-29.json)).** The
  LanguageDevice probe app (iPhone19,4, iOS 27.2) prepared the iOS bundle's entry (11 files, 1,208,370,448 bytes) from
  the same private repository with `BackgroundTransport`:
  - suspended for 45 s mid-file, the transfer kept going: the last event before was at 48,789,635 bytes, and within a
    second of the resume the system delivered progress to 176,579,147;
  - killed at 211,017,829 bytes and relaunched 53 s later, the app found the running task by its record and joined it;
    its first event was at 920,230,020 bytes, so about 709 MB arrived with no app process running;
  - the remaining files were fetched by new tasks, the entry installed, and `AetherLanguageModel(installation:)` gave a
    coherent greedy reply in 7.50 s.

  - on the final build, after a fresh install, a preparation stopped at 400,081,850 bytes left the system's resume
    data beside the file; the retry continued from it (first event at 400,085,296 bytes) and installed. This run
    qualified a fix: a resumed task's response starts at the resume point, which the first version refused.

  The relaunch shows the home-relative task record resolving on a device. Not yet exercised: a task in flight across
  an app update, when the container path changes, and the retry after an expired signed link.

## Implementation order

1. The catalog entry, `LanguageModelManifest.installationManifest(in:)`, and `aether-probe catalog-entry`. Done.
2. `ModelPreparation` with the check, the download area, install and cleanup, over the transport protocol and a local
   test transport. Done.
3. The foreground `URLSession` transport (`URLSessionTransport`, `.urlSession`), with range resume. Done.
4. The Hugging Face resolver (`ModelSource.huggingFace`) and `ModelPreparation.huggingFaceAuthorization(token:)`. Done.
5. `BackgroundTransport` and `handleEvents(identifier:completionHandler:)`. Done; qualified on an iPhone, since a
   background session needs an app.
6. `AetherLanguageModel(installation:)` with the lease held. Done.
7. The guide and the qualification evidence. Done: the Mac and iPhone records are in `docs/evidence`.

## Open questions

- Install by rename instead of copy, which halves peak disk use for multi-GB bundles. It needs a proof that the
  download area can be trusted at publish time.
- Whether `status` should also report Core AI's specialization cache. A model that's installed but not yet specialized
  pays a first-open compile. [Readiness](model-readiness.md#open-questions) leaves it to managed lifecycle, which owns
  loading.
- Where catalog entries live for the two apps: shipped in the app, or fetched and pinned by hash.
