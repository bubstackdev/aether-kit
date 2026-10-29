# Sample app and API freeze

Status: plan, 2026-10-29. This is the last piece of the 0.0.4 release gate, app readiness
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
  project.yml                 XcodeGen: one multiplatform application target (iOS 27, macOS 27); .xcodeproj ignored
  SampleApp.entitlements      macOS: app-sandbox, network.client
  Catalog/                    qwen3-0.6b-macos.json, qwen3-0.6b-ios.json (aether-probe catalog-entry), app resources
  Sources/App/                SwiftUI views only
  SampleAppCore/              local Swift package: the app's logic, no SwiftUI, with tests
```

**SampleAppCore** holds everything that isn't a view, so it is tested with `swift test`:
- `ModelController` (`@Observable`, main actor): picks the entry for the platform, builds the store in Application
  Support (excluded from iOS backup), `ModelPreparation` with `BackgroundTransport` on iOS and `.urlSession` on the
  Mac, and a `ModelReadiness` with `.language`. It mirrors `readiness.states` and `MemoryPressure.levels`, opens
  `AetherLanguageModel(installation:)` on `ready`, mirrors its `lifecycleStates`, and drops the model before
  `uninstall()` (which fails busy while a model holds the lease).
- `ReadinessDisplay`: a pure mapping from a readiness state to a title, detail, progress and the actions available
  (download, cancel, retry, delete), so the screen's rules are tested without a view.
- `ChatController`: one `AetherLanguageSession`, streaming turns into a message list; stop cancels; new chat starts a
  new session.
- `TodoExtraction`: the structured-output demo. `@AetherGenerable struct TodoItem { title; priority (low, medium,
  high); dueInDays: Int? }` from a free-text sentence.

**Screens** (a tab view on iOS, a sidebar on the Mac):
1. **Model:** state, byte progress and phase; Download / Cancel / Retry / Delete. Covers notInstalled (with resumable
   bytes), installing (checking, downloading, installing), verifying, ready, unsupported (OS, platform, memory) and
   failed (with whether a retry can help).
2. **Chat:** streaming reply, stop, new chat; disabled until ready.
3. **Structured:** one sentence in, a decoded `TodoItem` out, with its JSON.
4. **Lifecycle** (a status strip on every screen): unloaded (never loaded / requested / idle / memory pressure),
   loading, loaded, running, and the process's memory-pressure level.

The iOS app forwards background-session events with `.backgroundTask(.urlSession(identifier))`. The Mac app uses the
foreground transport, since it isn't suspended.

**Catalog entries:** `aether-probe catalog-entry language <bundle> --huggingface
aether-models/aether-kit-qwen3-0.6b@<revision>/mac` (and `/ios`), revision `7757d22bff5df7e106f7492a12d7278347aa3fb0`.
The iOS entry declares `platforms: ["ios"]` and `--memory-bytes 230686720` (220 MiB, the readiness doc's qualified
peak). The Mac entry declares its platform and no memory requirement. A CPU test decodes both committed entries and
checks their source, revision and platform.

**Signing:** the spec commits no team or profile. Qualification builds pass the existing development identity and an
existing profile on the command line; nothing is created or registered.

## The freeze

What later 0.0.x releases may change: add types, members, overloads with defaults, enum cases on the enums marked
below, and conformances. What they may not: remove or rename anything frozen, change a signature or a type, or add a
case to an exhaustive enum.

**Frozen symbols:**

| Module | Frozen |
| --- | --- |
| `AetherLanguage` | The whole module (it is the facade): `AetherLanguageModel`, `AetherLanguageSession`, `ResponseStream`, `AetherTranscript`, `AetherGenerationOptions`, `AetherGenerationSchema`, `AetherGenerable`, `AetherGenerationGuide`, `AetherGenerableCodingKey`, `AetherLanguageError`, `AetherStopReason`, and its extensions `ModelRuntimeCheck.language` and `LanguageModelManifest.installationManifest(in:)`. |
| `AetherLanguageMacros` | `@AetherGenerable`, `@AetherGuide` (structured output). |
| `AetherAssets` | `ModelCatalogEntry`, `ModelSource`, `ModelRequirements`, `ModelPreparation`, `PreparationStatus`, `PreparationEvent`, `PreparationError`, `ModelTransport`, `TransportRequest`, `URLSessionTransport`, `BackgroundTransport`, `ModelReadiness`, `ModelRuntimeCheck`, `ModelRuntimeVerdict`; and the types their signatures expose: `AssetManifest`, `AssetFile`, `LocalAssetStore`, `AssetReadLease`, `AssetStoreError`. |
| `AetherKit` | `ModelLifecyclePolicy`, `ModelLifecycleState`, `ModelUnloadReason`, `MemoryPressure`, `MemoryPressureLevel`; and `LanguageModelManifest.Platform` (in `AetherLanguageModel.init`). |

Not frozen: `AetherLanguageFoundationModels` (it adapts FoundationModels' own API), and everything else.

**CI:** a new job, **Public API (frozen)**, on the same `xcode-27` runner and Xcode 27.1 as the other Swift jobs. It
runs whenever they do, so any change under `Sources/` runs it; a docs-only PR skips it, and **CI passed** requires it.
`Scripts/check-api.sh`:
1. builds the four modules at `arm64-apple-macosx27.0`;
2. dumps each with `swift-api-digester -dump-sdk` (part of every Xcode toolchain; the runner's default Xcode 27.1 is
   build 27A9275, the same build as the local one used for the baseline);
3. prunes the dump to the frozen symbols above (an allowlist in `Scripts/api-freeze.json`);
4. compares it with the committed baseline in `api/` (`swift-api-digester -diagnose-sdk`) and fails on any breaking
   diagnostic. Additions pass.

`Scripts/check-api.sh --update` regenerates the baseline. Updating it is how an intentional break is made, so a PR that
touches `api/` says why. `docs/api-stability.md` will state the promise and that procedure. I chose the direct
digester over `swift package diagnose-api-breaking-changes` because the latter builds its baseline from a git revision
rather than a committed file, and can't be limited to a symbol list. Before relying on it, the job's first run must show
the tool works on the runner; a deliberately breaking commit on the branch must fail it, then be reverted.

## API gaps found while planning (to fix before the freeze)

1. **Enums that will grow.** These are exhaustive to clients today, so adding a case later breaks every exhaustive
   `switch`. Swift 6.4 has `@nonexhaustive` (SE-0487; checked: a client must then write `@unknown default`, an error in
   Swift 6 mode). Proposed for: `AetherLanguageError`, `AetherLanguageModel.UnavailableReason`, `AetherStopReason`,
   `AetherTranscript.Entry`, `AetherGenerationOptions.Reasoning`, `PreparationError`, `PreparationEvent`,
   `ModelSource`, `ModelReadiness.State`, `.Progress.Phase`, `.Unsupported`, `.Failure`, `ModelLifecycleState`,
   `ModelUnloadReason`. Kept exhaustive: `AetherLanguageModel.Availability`, `PreparationStatus`,
   `MemoryPressureLevel`, `ModelRuntimeVerdict`. Doing it later would itself be a break.
2. **A streamed reply's stop reason and usage are unreachable.** `ResponseStream`'s iterator keeps the finished
   `Response` but not publicly, and `collect()` can't follow an iteration. Proposed: a public read-only
   `AsyncIterator.response` (nil until `next()` returns nil). Additive, but the chat screen wants it now.
3. The guides' "type names may still change once the API has run in the sample app" lines go once the freeze lands.

## Documentation changes

`README.md` (the no-compatibility and no-weights statements), `NOTICE` (the sample bundle), `docs/kokoro-speech.md`
("AetherKit ships no weights"), the roadmap row, and the three gate guides' status lines.

## Qualification

Evidence in `docs/evidence/sample-app-2026-10-*.json`. The app has a qualification mode (a launch argument) that runs
the same controller actions the buttons call, with a 10 s idle policy, and writes a JSON record in its container:
- **Mac** (sandboxed build, checked with `codesign -d --entitlements`): a fresh container; download → verify →
  streamed chat → structured `TodoItem` → `unloaded(.idle)` → a reply after the reload → delete. The store must be
  inside the app's container. Model work runs under the GPU lease.
- **iPhone** (one of the two named devices only): the same sequence with `BackgroundTransport`, launched with
  `devicectl`, the record copied from the container.

Both runs also get a manual pass through the UI.
