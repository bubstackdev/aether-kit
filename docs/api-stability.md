# API stability

From 0.0.4, part of AetherKit's API is frozen for the rest of 0.0.x: later 0.0.x releases add to it and never break it.
Everything else stays experimental and may change in any 0.0.x release. The user decided the scope on 2026-10-29
([sample app and API freeze](sample-app.md)).

## What is frozen

| Module | Frozen declarations |
| --- | --- |
| `AetherLanguage` | The whole module, the app-facing language facade: `AetherLanguageModel`, `AetherLanguageSession`, `ResponseStream`, `AetherTranscript`, `AetherGenerationOptions`, `AetherGenerationSchema`, `AetherGenerable`, `AetherGenerationGuide`, `AetherGenerableCodingKey`, `AetherLanguageError`, `AetherStopReason`, and its extensions (`ModelRuntimeCheck.language`, `LanguageModelManifest.installationManifest(in:)`, and the `AetherGenerable` conformances of `String`, `Int`, `Double`, `Float`, `Bool` and `Array`). |
| `AetherLanguageMacros` | `@AetherGenerable` and `@AetherGuide`. |
| `AetherAssets` | Model preparation and readiness: `ModelCatalogEntry`, `ModelSource`, `ModelRequirements`, `ModelPreparation`, `PreparationStatus`, `PreparationEvent`, `PreparationError`, `ModelTransport`, `TransportRequest`, `URLSessionTransport`, `BackgroundTransport`, `ModelReadiness`, `ModelRuntimeCheck`, `ModelRuntimeVerdict`. Also the store types their signatures expose: `AssetManifest`, `AssetFile`, `LocalAssetStore`, `AssetReadLease`, `AssetStoreError`. |
| `AetherKit` | Managed lifecycle: `ModelLifecyclePolicy`, `ModelLifecycleState`, `ModelUnloadReason`, `MemoryPressure`, `MemoryPressureLevel`. Also `LanguageModelManifest.Platform`, which `AetherLanguageModel`'s initializers take. The rest of `LanguageModelManifest` is not frozen. |

[`Scripts/api-freeze.json`](../Scripts/api-freeze.json) is the machine-readable list. Not frozen:
`AetherLanguageFoundationModels`, because it adapts FoundationModels' own API, and every other module and declaration.

## The promise

A later 0.0.x release may:
- add types, members, initializers, and overloads or parameters with defaults that leave existing calls unchanged;
- add a case to an enum marked `@nonexhaustive` (below);
- add conformances;
- deprecate a declaration, which then stays until 0.1.

It may not:
- remove or rename a frozen declaration;
- change a frozen declaration's type, parameters, effects (`async`, `throws`) or generic requirements;
- add a case to an exhaustive enum, or a case that takes away a synthesized conformance (a payload on a payload-free
  enum loses `Hashable`);
- remove a conformance, or make an exhaustive enum `@nonexhaustive`.

The promise is about source compatibility for code that builds against the package with the supported toolchain. It
isn't binary (ABI) stability: AetherKit is a source package, built without library evolution.

### Enums that can grow

These enums are `@nonexhaustive` (SE-0487), so a `switch` over one from outside the package needs `@unknown default`
(an error in Swift 6 language mode without one, a warning in Swift 5 mode). New cases arrive in later 0.0.x releases
without breaking that code.

| Module | `@nonexhaustive` |
| --- | --- |
| `AetherLanguage` | `AetherLanguageError`, `AetherLanguageModel.UnavailableReason`, `AetherStopReason`, `AetherTranscript.Entry`, `AetherGenerationOptions.Reasoning`, `AetherGenerationOptions.SamplingMode.Kind` |
| `AetherAssets` | `PreparationError`, `PreparationEvent`, `ModelSource`, `ModelReadiness.State`, `ModelReadiness.Progress.Phase`, `ModelReadiness.Unsupported`, `ModelReadiness.Failure`, `AssetStoreError` |
| `AetherKit` | `ModelLifecycleState`, `ModelUnloadReason`, `LanguageModelManifest.Platform` (a registration for another OS, such as visionOS, is plausible) |

The payload-free ones (`AetherStopReason`, `AetherGenerationOptions.Reasoning`, `ModelReadiness.Progress.Phase`,
`ModelUnloadReason`) declare `Hashable` explicitly. Swift synthesizes it for a payload-free enum by itself, and would
silently drop it when a case with a payload arrives; declared, a later case keeps it or doesn't compile.

These stay exhaustive, by choice:
- **`AetherLanguageModel.Availability`**: available or not. Every new reason goes in `UnavailableReason`, which can grow.
- **`PreparationStatus`**: installed, partially downloaded, or not installed partitions what a store holds for an entry;
  there is no fourth answer to "what's on disk".
- **`MemoryPressureLevel`**: the system's three memory-pressure levels (normal, warning, critical), ordered, so apps
  compare levels rather than switch over them.
- **`ModelRuntimeVerdict`**: what a product's check returns: runnable, unsupported (whose reasons can grow), or
  invalid. Apps implement checks that return it more than they switch over it.

## How CI enforces it

The **Public API (frozen)** job runs [`Scripts/check-api.sh`](../Scripts/check-api.sh) on every change that runs the
Swift jobs, which includes every change under `Sources/`. **CI passed** requires it. The script:
1. builds the four modules at `arm64-apple-macosx27.0`;
2. dumps each with `swift-api-digester -dump-sdk`, from the selected Xcode toolchain (Xcode 27.1 in CI);
3. keeps the frozen declarations, without internal or package ones ([`Scripts/api-freeze.py`](../Scripts/api-freeze.py));
4. fails if a frozen declaration exposes an `Aether` type the freeze doesn't cover, in a signature, a generic
   constraint or a protocol it conforms to (a frozen declaration must expose only frozen types, or the unfrozen one
   could break it). Conformances are read from their mangled names; one in a form the script doesn't read fails. An
   extension of another module's type counts only its conformances to the module's own protocols. Two gaps remain:
   the dump doesn't say which module declared a conformance, so a frozen module's own conformance of another module's
   type to a third module's unfrozen protocol isn't seen; and the type a frozen module extends isn't itself checked
   (`AetherLanguage` adds `installationManifest(in:)` to `AetherKit.LanguageModelManifest`, which isn't frozen);
5. compares them with the committed baseline in [`api/`](../api) (`swift-api-digester -diagnose-sdk`), and fails on
   every finding except a case added to a `@nonexhaustive` enum;
6. then fails if the baseline is stale: a compatible change (an addition, or an attribute or conformance the digester
   doesn't report) passes the comparison, but until `api/` records it, a later change could undo it unnoticed. The fix
   is `--update` in the same pull request.

`Scripts/test_api_freeze.py` tests the pruning, exposure and comparison rules, and runs first. The script fails closed:
a missing digester, baseline or frozen declaration, an empty dump, or a comparison with no report fails the job. The job uploads the pruned dumps and the digester's reports as diagnostics.

Run it locally with Xcode 27.1 selected:

```sh
sh Scripts/check-api.sh
```

## Changing the frozen API

- **An addition** must be recorded in the same pull request: `sh Scripts/check-api.sh --update`, and commit `api/`.
  Without that, the job fails with "api/<Module>.json is stale".
- **Freezing more** (a new declaration in an unfrozen module): add it to `Scripts/api-freeze.json`, then `--update`.
- **A break** is a decision, not a fix-up. It needs the user's approval, a `Breaking` entry in the changelog with the
  migration, and `--update`. The pull request says why, and its `api/` diff shows exactly what changed.
- **A toolchain change** can change the dump format without changing the API. Regenerate the baseline with the new
  Xcode in its own commit, and check that `-diagnose-sdk` against the old baseline reports nothing first.
