# Deployment floor: macOS 14 and iOS 17

AetherKit's package floor is macOS 14 and iOS 17 (0.0.3). An app at that floor can link the language facade,
`AetherLanguage`, and launch. Below macOS 27 and iOS 27 the facade reports `.unavailable(.unsupportedOS)`. Everything
that runs a model still needs OS 27 at runtime: Core AI exists only there.

## What the floor means

| Product | Builds at the floor | Runs below OS 27 |
| --- | --- | --- |
| `AetherLanguage` (the facade) | Yes | Yes: `availability` is `.unsupportedOS`, and requests throw that reason |
| `AetherKit`, `AetherAssets`, `AetherVision`, `AetherLanguageRuntime`, `AetherLanguageTokenizers` and the other non-Core AI products | Yes | Yes |
| `AetherCoreAI` and every `*CoreAI` product | Yes | No: every declaration is `@available(macOS 27, iOS 27, *)`, so an app calls them behind `#available` |
| `AetherLanguageFoundationModels` | Yes | No: it is `@available(macOS 27, iOS 27, *)`, like FoundationModels' `LanguageModel` |
| `aether-probe` | Yes (macOS) | No: a developer tool for OS 27 |

The facade checks the OS first. Below 27 no backend runs, so it reports `.unsupportedOS` whatever the bundle holds, and
its `contextLength` is 0. The backend choice takes the OS check as a parameter internally, so the weight-free suite
covers that branch on OS 27 (`belowTheCoreAIFloorEveryBundleIsUnsupportedOS`).

## How the Core AI targets are gated

SwiftPM has no per-target deployment floor, no weak-link setting and no file-level availability. So every top-level
declaration in the targets that use Core AI carries `@available(macOS 27, iOS 27, *)`, and members inherit it from their
type or extension. The targets are `AetherCoreAI`, `AetherLanguageCoreAI`, `AetherH3CoreAI`, `AetherKreaCoreAI`,
`AetherFlux2CoreAI`, `AetherSpeechCoreAI` and `AetherProbe`: 245 declarations in 53 files when the floor dropped.

`Scripts/annotate-floor.py` adds the attribute and is idempotent:

```sh
python3 -I Scripts/annotate-floor.py            # annotate every gated target in place
python3 -I Scripts/annotate-floor.py --check    # list declarations without it; exit 1 if any (CI)
```

Rules for the gated targets:
- **New top-level declarations** need the attribute. Run the script, or CI's `--check` fails.
- **After a merge or rebase conflict** in these files, take either side and rerun the script.
- **Code outside the gated targets** reaches them only inside `if #available(macOS 27, iOS 27, *)`, as
  `AetherLanguage/CoreAIBackend.swift` does.
- **Below the gated targets, avoid OS 15+ APIs ungated.** `AetherAssets` and the lease-test helper use
  `OSAllocatedUnfairLock` (macOS 13, iOS 16) rather than `Synchronization.Mutex` (macOS 15, iOS 18); `AetherKit` uses its
  own `UnfairLock` over it.

## Weak linking

No linker flag is involved. Every Core AI type in the SDK is `@available(macOS 27, iOS 27, …)`, so every reference to it
from code built for macOS 14 is a weak import. When all of a library's references are weak, the linker writes
`LC_LOAD_WEAK_DYLIB` for it. This holds whether the framework comes from `.linkedFramework("CoreAI")` or from
autolinking. At launch on an OS without the framework, dyld binds those symbols to null, and the `#available` checks keep
the code from reaching them.

The same applies to FoundationModels, and to `libswiftSynchronization` where it is used only inside gated code. A
floor-14 binary linking the facade had all 91 of its CoreAI binds marked `[weak-import]`.

**One extra library.** swift-collections (through swift-transformers) uses `OutputSpan`, so a floor binary loads
`@rpath/libswiftCompatibilitySpan.dylib`, Swift's Span back-deployment library. Xcode app targets embed it in the
bundle's `Frameworks` automatically. A bare command-line tool has to ship it next to the binary, as the CI launch check
does.

## Tests build at OS 27

Swift Testing refuses `@Test` and `@Suite` inside an availability-gated type, so the test targets can't be gated like the
sources. They use Core AI ungated and build at macOS 27:

```sh
swift test --triple arm64-apple-macosx27.0 …        # what Scripts/check.sh and the qualify scripts run
xcodebuild … MACOSX_DEPLOYMENT_TARGET=27.0 test      # under Xcode's build system
```

A bare `swift test` fails to compile. Every test target that links AetherKit refers to a 27-only marker, so the failure
names the fix: `'RunTestsWithTheOS27Triple' is only available in macOS 27 or newer`. Native qualification commands
carry the triple too, for example
`~/.cache/aether-kit/with-gpu-lock.sh "flux2 native" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2NativeParity`.

## Checks

`Scripts/check-floor.sh` runs in CI on `xcode-27` runners, as two parallel jobs: `--macos` (steps 1 and 2, and the
macOS half of 4) and `--ios` (the iOS builds in 3 and 4). Without either flag it runs everything. It needs no weights and
does no GPU work:
1. `annotate-floor.py --check`;
2. `swift build` of every product at macOS 14;
3. an iOS 17 build of every library product (`xcodebuild -destination generic/platform=iOS`);
4. `Examples/FloorConsumer`, which links `AetherLanguage`, built in release for macOS 14 and for iOS 17 devices. Each
   binary must report its minimum OS and weak-link CoreAI. Every strong load command must be on an allow-list of
   libraries the floor OS has, so a new dependency fails until it is reviewed. The macOS binary runs.

With a stage directory, `check-floor.sh` also copies the consumer binary and its Span library there. CI's **Launch
without CoreAI (macOS 15)** job runs `Scripts/launch-floor-consumer.sh` on a GitHub-hosted `macos-15` runner, which has
no CoreAI framework, and requires `AetherLanguage availability: unsupportedOS`. That job is part of the **CI passed**
gate.

**No macOS 14 launch.** The job ran on `macos-14` until GitHub began retiring that image (fully retired on
2026-11-02); macOS 15 is the oldest hosted OS left. It still proves the CoreAI link is weak and the facade reports `unsupportedOS` below 27. What
covers macOS 14 itself is the build: every product builds at macOS 14, the consumer reports macOS 14 as its minimum OS,
and every strong load command is on the floor OS's allow-list. A call to an API newer than macOS 14 but present on 15
would still launch here, so the compiler's availability checking at the macOS 14 target is what catches it.

**No iOS 17 launch yet.** The iOS 17 device build and its weak CoreAI link are checked. But CoreAI isn't
in the iOS Simulator SDK, so nothing that links the facade builds for the simulator: `import CoreAI` fails there. That
was already true at the old floor. A simulator launch, and simulator builds for apps, need the Core AI targets to compile
without CoreAI (for example `#if canImport(CoreAI)` around them), which is a follow-up.

## Why this design

Before the change, a phase-1 investigation compared the approaches on scratch packages (Xcode 27.1, macOS 27.0.1):
- **Annotating every member** would have been about 1,610 sites in the library targets; public declarations alone
  aren't enough.
- **`-weak_framework` through `unsafeFlags`** isn't needed. SwiftPM 6.4 did accept it in a remote dependency, but an
  Xcode project consumer wasn't tested.
- **Gating the tests per function** would have touched about 600 declarations and still left hand fixes.
