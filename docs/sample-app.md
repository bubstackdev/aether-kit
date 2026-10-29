# Sample app and API freeze

Status: implemented 2026-10-29; qualification below. This is the last piece of the 0.0.4 release gate, app readiness
([roadmap](roadmap-0.0.4.md#release-gate-app-readiness)), after [model preparation](model-preparation.md),
[model readiness](model-readiness.md) and [managed lifecycle](model-lifecycle.md). The user decided the four choices
below on 2026-10-29.

## Decisions (user, 2026-10-29)

1. **Platforms:** one SwiftUI multiplatform app for iOS and macOS, in `Examples/SampleApp`. The Mac build has App
   Sandbox on; iOS is sandboxed anyway. No increased-memory entitlement: Qwen3-0.6B doesn't need it. Both are qualified.
2. **Model source:** the app downloads Qwen3-0.6B through `ModelPreparation`, from committed catalog entries pinned to
   the public Hugging Face repository `aether-models/aether-kit-qwen3-0.6b` (`mac/` and `ios/` bundles, Apache-2.0,
   with LICENSE and NOTICE). No token. Kit's stance changes from "distributes no weights" to: Kit publishes one sample
   bundle, Qwen3-0.6B, converted from `Qwen/Qwen3-0.6B` under Apache-2.0, and otherwise distributes none.
3. **Scope:** a readiness screen, streaming chat, one structured-output demo, and the lifecycle state shown live. No
   Qwen3-VL.
4. **API freeze:** the language facade plus readiness, preparation (with the catalog entry types) and the lifecycle
   policy and state types, enforced in CI against a committed baseline. Later 0.0.x releases add to them and never
   break them.

## The app

```
Examples/SampleApp/
  project.yml                   XcodeGen: one application target for iOS and macOS (27.0); the .xcodeproj is ignored
  SampleApp-macOS.entitlements  the Mac build: app-sandbox, network.client
  make-catalog.sh               writes the two catalog entries from local copies of the public bundle
  Sources/                      SwiftUI views and the app entry point only
  SampleAppCore/                a local Swift package with the app's logic, its catalog entries and its tests
```

Build it with `xcodegen generate` in `Examples/SampleApp`, then open `SampleApp.xcodeproj`, choose a team, and run
the `SampleApp` scheme on a Mac or an iPhone with macOS 27 or iOS 27.

**SampleAppCore** holds everything that isn't a view, so `swift test` covers it (in CI's contracts job):
- `SampleCatalog`: the shipped entry for the platform, decoded and validated.
- `ModelController` (`@Observable`, main actor): a `ModelReadiness` with `.language` over the store in Application
  Support/Models (excluded from backup), with `BackgroundTransport` on iOS and `.urlSession` on the Mac. It follows
  `states` and `MemoryPressure.levels`, opens `AetherLanguageModel(installation:)` on `ready`, and follows its
  `lifecycleStates`. It never stores a readiness state, because `ready` holds the installation's read lease and any
  holder makes `uninstall()` fail busy. Before a delete it asks the screens to let go of the model, then uninstalls.
- `ReadinessDisplay` and `LifecycleDisplay`: pure mappings from a readiness state to the title, detail, progress and
  actions (download or resume, cancel, retry, delete, check again), and from a lifecycle state and pressure level to
  the strip's text.
- `ChatController`: one `AetherLanguageSession` per conversation, streamed. It reads the finished reply's stop reason
  and usage from `ResponseStream.AsyncIterator.response`. Stop cancels the reply, and New chat starts a new session.
- `TodoController`: the structured-output demo. `@AetherGenerable struct TodoItem { title; priority (low, medium,
  high); dueInDays: Int? }` from one sentence, with the JSON the model produced.
- `QualificationRun`: the `-qualify` launch mode (below).

**Screens** (tabs on iPhone, a sidebar on the Mac and iPad):
1. **Model:** state, phase and byte progress; Download (or Resume), Cancel, Retry, Delete, Check again. Covers
   notInstalled (with resumable bytes), installing (checking, downloading, installing), verifying, ready, unsupported
   (OS, platform, memory) and failed (retry offered only when it can help).
2. **Chat:** streaming reply, stop, new chat; disabled until ready.
3. **Structured:** one sentence in, a decoded `TodoItem` out, with its JSON and the time it took.
4. **Lifecycle,** a strip on every screen: not loaded yet, loading, loaded, running, unloaded (requested, idle or under
   memory pressure), and the process's memory-pressure level.

The iOS app forwards background-session events with `.backgroundTask(.urlSession(identifier))`. The Mac app uses the
foreground transport, since it isn't suspended.

**Catalog entries:** `make-catalog.sh MAC_BUNDLE IOS_BUNDLE` runs `aether-probe catalog-entry language BUNDLE
--huggingface aether-models/aether-kit-qwen3-0.6b@7757d22bff5df7e106f7492a12d7278347aa3fb0/mac` (and `/ios`). Both
entries declare their platform. The iOS entry also declares `memoryBytes` 230,686,720 (220 MiB). That is
`footprintPeakMiB` 219.175 from
[the 0.0.3 iPhone record](evidence/language-iphone18-qwen3-0.6b-2026-10-28.json), rounded up, the value
[readiness](model-readiness.md#requirements-and-memory) uses. The Mac entry declares no memory requirement. A CPU test
checks both committed entries: source, revision, subdirectory, platform, memory, 11 files, and the byte totals the
model-preparation qualification recorded.

**Signing:** the spec names no team or profile. The qualification builds pass `DEVELOPMENT_TEAM` on the `xcodebuild`
command line, with nothing created or registered:
- On iOS, automatic signing picks the team's existing wildcard development profile. The app needs no capability that
  an explicit App ID would carry.
- The Mac build is signed with the existing development identity and carries only the sandbox and network-client
  entitlements, which need no profile.

## The freeze

[API stability](api-stability.md) is the reference: the promise, the frozen declarations, the enums that can grow and
why four stay exhaustive, how CI enforces it, and how to change the baseline intentionally. In short:
- **Frozen:** the whole `AetherLanguage` module and the `AetherLanguageMacros` macros; in `AetherAssets`, preparation,
  readiness and the catalog entry types, plus the store types their signatures expose; in `AetherKit`, the lifecycle
  policy and state types and `LanguageModelManifest.Platform`. `AetherLanguageFoundationModels` is not frozen.
- **CI:** the **Public API (frozen)** job (`Scripts/check-api.sh`) dumps those modules with `swift-api-digester`,
  keeps the frozen declarations, and compares them with the committed baseline in `api/`. It runs whenever the Swift
  jobs run, including every change under `Sources/`, and **CI passed** requires it. The direct digester was chosen over
  `swift package diagnose-api-breaking-changes`, which builds its baseline from a repository revision rather than a
  committed file and can't be limited to a list of declarations.
- **Proof on the runner** (PR #80): a commit that made `AetherLanguageSession.prewarm()` internal failed the job with
  `Func AetherLanguageSession.prewarm() has been removed`
  ([run](https://github.com/bubstackdev/aether-kit/actions/runs/40936256797)), and its revert passed
  ([run](https://github.com/bubstackdev/aether-kit/actions/runs/40937869455)). Locally, a removed method, a changed
  parameter type, and `@nonexhaustive` added to an exhaustive enum each fail. A new method, a change to an unfrozen
  type, and a payload-free case added to a `@nonexhaustive` enum each pass. A payload case added to a payload-free enum
  fails, because it removes the synthesized `Hashable`.

## API changes the app led to (made before the freeze)

1. **Enums that can grow are `@nonexhaustive`** (SE-0487), so a later case is additive. Clients outside the package
   need `@unknown default` (an error in Swift 6 mode without it). The changelog lists this under Breaking, with the
   migration. Inside the package, modules still switch exhaustively. The examples that switch over these enums
   (`FloorConsumer`, `LanguageDevice`) handle unknown cases.
2. **`ResponseStream.AsyncIterator.response` is public:** a streamed reply's stop reason, usage, seed and transcript
   entries once `next()` returns nil. The chat screen's footnote reads it.

## Qualification

[Record](evidence/sample-app-2026-10-29.json). The app's `-qualify PHASES` launch mode (`fresh`, `prepare`, `run`,
`idle`, `delete`) calls the same controller actions the buttons do. `-idleSeconds 10` sets the idle policy, and the
app writes its JSON record to its container. Each run downloaded the bundle anonymously from the public repository at
revision `7757d22b`.

**Mac** (Mac17,6, macOS 27.2, Release build, sandboxed: `codesign -d --entitlements` shows app-sandbox and
network.client). The record has `sandboxed: true` and `storeInsideAppContainer: true`. It took two launches:
- `fresh,prepare`, with no model work: from a new container, `notInstalled(0/1208370450)`, then checking, downloading
  and installing, then `verifying` → `ready` 101.8 s after Download.
- `run,idle,delete`, under the shared GPU lease. The installed model verified again to `ready` at launch.
  - The streamed chat answered the hash-function prompt in 8.20 s, first text at 7.97 s, the cold first load included.
    The footnote read "end of turn · 41 prompt + 24 reply tokens", so stop reason and usage came from
    `AsyncIterator.response`.
  - The structured request decoded `{"dueInDays": 7, "priority": "high", "title": "Renew my passport"}` in 0.74 s.
  - The model was `unloaded(idle)` 10.07 s after the last request. A new chat's reply loaded it again and finished in
    1.74 s.
  - Delete returned the state to `notInstalled(0/1208370450)` and left 0 bytes in the store.

**iPhone** (iPhone19,4, iOS 27.2, no increased-memory entitlement, `BackgroundTransport`, one launch through
`devicectl` with all five phases). The memory limit read 3,539,992,576 bytes, over the iOS entry's 230,686,720-byte
requirement.
- From a new install, it reached `ready` 1,187.7 s after Download. Most of that was the transfer: downloading ran from
  0.7 s to 1,186.9 s.
- The streamed chat finished in 7.30 s (first text at 6.97 s), with the same reply and footnote as the Mac.
- Structured output decoded the same `TodoItem` in 0.97 s.
- The idle unload came 10.07 s after the last request, and the reload reply finished in 1.76 s.
- Delete left `notInstalled(0/1208370448)` and 0 bytes in the store.

Not covered by these runs: a pass through the UI by hand, and memory pressure in the app. The
[lifecycle record](evidence/model-lifecycle-2026-10-29.json) qualifies pressure; the strip shows the level from
`MemoryPressure.levels`.
