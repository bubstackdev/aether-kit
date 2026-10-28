# Model preparation

Status: proposed, 2026-10-28. This is the first piece of the 0.0.4 release gate, app readiness
([roadmap](roadmap-0.0.4.md#release-gate-app-readiness)). The user decided the four choices below on 2026-10-28. Type
names are illustrative until the API has run in the sample app.

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
  gated or private repositories. Kit never stores it.

## Preparing

```swift
public struct ModelPreparation: Sendable {
    public init(store: LocalAssetStore, transport: some ModelTransport = .urlSession)
    public func status(_ entry: ModelCatalogEntry) async throws -> PreparationStatus   // no network
    public func prepare(_ entry: ModelCatalogEntry) -> PreparationRun
}

public enum PreparationStatus: Sendable {
    case installed(installationID: String)
    case partial(bytesPresent: UInt64, bytesTotal: UInt64)   // resumable download data on disk
    case notInstalled(bytesTotal: UInt64)
}
```

A `PreparationRun` is an `AsyncSequence` of events that ends in exactly one terminal event:
- `.checking`: free space and existing data;
- `.downloading(file:, bytesDone:, bytesTotal:)` with an overall total. Progress is coalesced: at most a few events a
  second, and the final count of each file is always delivered;
- `.verifying(file:)`;
- `.installing`, the atomic publish;
- terminal: `.installed(installationID)`, `.failed(PreparationError)` or `.cancelled`.

Cancelling the consuming task cancels the run. Already downloaded bytes stay on disk for resumption.

**Order of work:**
1. **Check.** If the installation exists, verify it as `install` does and finish. Otherwise sum the bytes still
   missing and compare them with the volume's available capacity for important usage. Refuse with
   `.insufficientSpace(needed:, available:)` before downloading anything.
2. **Download** each missing file into a per-installation download area in the store (`downloads/<installationID>/`).
   It's kept across launches, unlike install's temporary stage. Each file is hashed as it streams where the transport
   allows. A file whose size or hash doesn't match the entry is deleted and reported as `.integrityMismatch(path:)`.
3. **Install** by handing the completed download area to `LocalAssetStore` as the source. The store's own copy and hash
   pass stays the single point where bytes become trusted. Installing by rename instead of copy, to avoid doubling
   disk use, is a follow-up once it has its own tests.
4. **Clean up.** The download area is removed after a successful install. `removeAbandonedStages` also removes download
   areas for installations that already exist.

Two runs for the same entry share the per-installation lock: the second waits or reports `.busy`, and never downloads
twice.

## Transport

```swift
public protocol ModelTransport: Sendable {
    /// Writes `request.url` to `destination`, appending from `destination`'s current size when `resume` is true.
    func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
               progress: @Sendable (UInt64) -> Void) async throws
}
```

- **`.urlSession` (foreground)** uses a data-task stream. It resumes with an HTTP `Range` request and checks the
  response: 206 with a matching `Content-Range` appends, 200 restarts the file, anything else fails. It hashes while
  writing.
- **`.background(identifier:)` (iOS, and macOS apps)** uses a background `URLSession` with download tasks, one per file,
  so the system carries transfers on while the app is suspended. Download tasks deliver a finished temporary file, so
  hashing happens after delivery, in step 2's verification. Interrupted tasks resume from the system's resume data.
  The app connects Kit to its lifecycle in one place:
  ```swift
  // UIApplicationDelegate, or SwiftUI's .backgroundTask(.urlSession(identifier))
  ModelPreparation.handleBackgroundEvents(identifier: identifier, completionHandler: completionHandler)
  ```
  On relaunch, `prepare` reattaches to the session's outstanding tasks for the entry instead of starting new ones.
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
- **Readiness**, the next gate piece, builds on `status` and the run's events, so it isn't designed here.

## Errors

`PreparationError` is one enum with these cases: `insufficientSpace(needed:, available:)`, `integrityMismatch(path:)`,
`http(status:, path:)`, `network(underlying:)`, `invalidEntry(String)`, `busy` and `fileSystem(Int32)`. Each says
whether retrying can help.

## Qualification

- **Weight-free CPU tests in CI.** A local file transport and a scripted `URLProtocol` cover:
  - resume after a cut connection (206), restart on 200, refusal of a wrong `Content-Range`;
  - size and hash mismatch;
  - insufficient space;
  - cancellation mid-file, then a resumed run;
  - two concurrent runs for one entry;
  - the Hugging Face URL and revision rules;
  - an installed entry finishing without network;
  - background-session reattachment, driven through the delegate.
- **Opt-in real-network qualification.** Prepare a real Kit bundle over HTTPS on the Mac and an iPhone, the iPhone
  including a background transfer that survives app suspension. Then open it through the facade and run a short reply.
  Record the evidence.

## Implementation order

1. The catalog entry, `LanguageModelManifest.installationManifest(in:)`, and `aether-probe catalog-entry`. Done.
2. `ModelPreparation` with the check, the download area, install and cleanup, over the transport protocol and a local
   test transport.
3. The foreground `URLSession` transport, with range resume and streaming hashes.
4. The Hugging Face resolver.
5. The background transport and `handleBackgroundEvents`.
6. `AetherLanguageModel(installation:)` with the lease held.
7. The guide and the qualification evidence.

## Open questions

- Install by rename instead of copy, which halves peak disk use for multi-GB bundles. It needs a proof that the
  download area can be trusted at publish time.
- Whether `status` should also report Core AI's specialization cache. A model that's installed but not yet specialized
  pays a first-open compile. That probably belongs to readiness.
- Where catalog entries live for the two apps: shipped in the app, or fetched and pinned by hash.
