# Managed lifecycle

Status: implemented and qualified 2026-10-29: CPU tests, a Mac run, and real memory pressure on an iPhone
([Qualification](#qualification)).
This is the third piece of the 0.0.4 release gate, app readiness
([roadmap](roadmap-0.0.4.md#release-gate-app-readiness)), after [model preparation](model-preparation.md) and
[model readiness](model-readiness.md). The user decided the four choices below on 2026-10-29.

Readiness ends at `ready`: the model is installed, verified and runnable. Lifecycle covers what happens after that:
when the model loads, when it unloads, and what it does when the device runs short of memory.

## Decisions (user, 2026-10-29)

1. **Scope:** one policy type, applied by every product.
   - The language facade (Qwen3 and Qwen3-VL) and Kokoro get idle unload and memory-pressure release. Kokoro's release
     keeps its session usable.
   - Krea and FLUX.2 apply it to whatever models a session keeps between operations, in either residency mode.
   - H3 releases every stage as it goes, so it only refuses new loads under critical pressure.
2. **Idle unload is on by default:** after 60 s idle on iOS and 5 min on the Mac. Each model can change or disable it.
3. **Memory pressure:** on a warning or a critical notice, every idle model in the process unloads, once per notice. A
   model that's busy when the notice arrives unloads as soon as its request ends. Running requests are never cancelled;
   the level is reported, so the app can cancel if it wants. While pressure is critical, new loads are refused.
   - **Revised the same day, after the iPhone run:** iOS never reports a return to normal after an app's critical
     notice. So critical eases on whichever comes first:
     - a normal notice;
     - the app's available memory recovering to twice what it was at the notice (iOS);
     - 60 s after the last critical notice (on a Mac, only once the kernel no longer reads critical).
4. **No cross-model eviction** in this piece. A process-wide budget with least-recently-used eviction belongs with
   shared declared residency ([proposal](shared-residency-proposal.md)), which spans processes and Atelier.

## API

```swift
// AetherKit
public struct ModelLifecyclePolicy: Sendable, Equatable {
    public var idleUnloadAfter: Duration?          // nil: stays loaded until asked or pressure
    public var releaseUnderMemoryPressure: Bool    // default true
    public static var platformDefault: Self        // iOS 60 s, Mac 5 min, pressure release on
    public static let manual: Self                 // neither
}

public enum ModelLifecycleState: Sendable, Equatable {
    case unloaded(ModelUnloadReason?)              // nil: not loaded yet
    case loading
    case loaded                                    // loaded and idle
    case running
}
public enum ModelUnloadReason: Sendable, Equatable { case requested, idle, memoryPressure }

public enum MemoryPressureLevel: Sendable, Comparable { case normal, warning, critical }
public enum MemoryPressure {
    public static var level: MemoryPressureLevel { get }
    public static var levels: AsyncStream<MemoryPressureLevel> { get }     // current, then each change
}
```

Each product takes a `lifecycle: ModelLifecyclePolicy = .platformDefault` and exposes `lifecycleState` (a snapshot)
and `lifecycleStates` (a stream: current state, then each change, newest-value buffer), as readiness does.

Internally one package type, `ModelLifecycle`, does the bookkeeping for every product:
- **The state.** The product reports `loading`, `loaded`, `running`, `idle` and `unloaded`.
- **The idle timer.** It restarts when a request ends or a load finishes. When it fires, it asks the product to release
  if still idle.
- **Pressure.** The model is registered weakly with the process-wide monitor, a `DispatchSource` memory-pressure source
  for warning, critical and normal. It starts on first use, from the kernel's current level
  (`kern.memorystatus_vm_pressure_level`). Each warning or critical notice releases each idle model once, if the policy
  allows; a model that was busy releases when it next goes idle. A model that was unloaded at the notice, or created
  after it, has nothing to release, and keeps its next load. While the level stays raised, later requests load and
  keep the model, so a long warning doesn't turn every request into a load and an unload. A process started under
  pressure sees the level for the load gate, but that isn't a notice, so it doesn't release anything.
- **Easing.** iOS doesn't report a return to normal after an app's critical notice, and an app can't read the kernel's
  level. So critical also ends by itself, reported as a normal level:
  - once `os_proc_available_memory()` is back to twice what it was at the notice (iOS only);
  - otherwise 60 s after the last critical notice. A repeated critical notice restarts that clock. Where the kernel's
    level is readable (a Mac), an expiry while it still reads critical keeps the level critical and restarts the clock,
    so a sustained critical isn't eased. An easing decided just as a new critical notice arrives is dropped.

  A later critical notice is a new notice, and releases models again.
- **The load gate.** While pressure is critical, a load throws. Language reports `AetherLanguageError.insufficientMemory`;
  the others report a new `AetherError.memoryPressure`.

The product owns its model and decides "idle" under its own isolation, so a release can never race a request: the
release callback returns false if a request started meanwhile.

## Per product

| Product | Today | With lifecycle |
| --- | --- | --- |
| Language (Qwen3, Qwen3-VL) | Lazy load on the first request or `prewarm()`; `unload()` and later requests load again. | Idle timer and pressure call the same reloadable unload, with reason `idle` or `memoryPressure`. `unload()` reports `requested`. |
| Kokoro | Models open on first use and stay until `unload()`, which closes the session for good. | A new internal release unloads both models and keeps the session `ready`; the next request opens them again. Prewarmed bucket specializations are lost with it, as with any unload. `unload()` stays terminal. |
| Krea, FLUX.2 | Both modes keep the denoisers and decoder between operations; Krea also keeps its image encoder. `balanced` releases the text encoder after encoding, and the denoisers before the next text encoder loads. `resident` keeps the text encoder too. `unload()` is terminal. | Idle and pressure release whatever the session keeps, without finishing it; the next operation opens them again. The state is `loaded` while anything is kept. |
| H3 | Every stage loads and releases within one generation. | Only the load gate: a generation started under critical pressure is refused. |

**The language memory budget stays where it is.** `LanguageGenerationSession` reloads a model once Core AI's
retention has spent its growth allowance, at idle, under a GPU grant. That reload is part of the lifecycle: it keeps a
long-lived loaded model bounded. It doesn't change the facade's state, because the model is loaded before and after it.
Its counters stay in `CacheStatistics`. An idle unload or pressure release also resets the budget, since the next load
starts fresh.

## Qualification

- **CPU tests** with an injected clock and pressure source:
  - every state transition, and the last unload reason kept across a failed or refused load;
  - a sustained warning releasing once per notice;
  - idle release racing a new request;
  - pressure during a request (no release until it ends);
  - critical refusing a load, and normal lifting it;
  - each product's release keeping it usable.
- **Mac** (M5 Max, macOS 27.2; [record](evidence/model-lifecycle-2026-10-29.json)): `aether-probe qualify-lifecycle`
  on the installed Qwen3 0.6B, with a 10 s idle policy. Pressure was simulated through the monitor's delivery seam,
  because raising real pressure on this shared Mac would disturb other work.
  - The first reply loaded the model: 1,484.1 MiB of Metal, 3.92 s.
  - 10.5 s after the reply it was `unloaded(.idle)`, with Metal back to 0.7 MiB.
  - The next reply loaded it again, in 1.97 s.
  - A simulated warning released it, to 12.7 MiB.
  - A simulated critical level refused the next load with `insufficientMemory`.
- **iPhone** (iPhone19,4, iOS 27.2, no increased-memory entitlement; same record): the LanguageDevice `lifecycle`
  mode, with a 10 s idle policy, on the installed Qwen3 0.6B. Real pressure came from ballast allocated in touched
  32 MiB steps.
  - The first reply took 2.56 s. 10 s later the model was `unloaded(.idle)`, and the reload reply took 1.95 s.
  - At 2,624 MiB of ballast (a 2,731.6 MiB footprint, 644.4 MiB available), iOS sent UIKit's memory warning and Kit's
    dispatch notice at the same moment. The notice came as **critical**, with no warning first. The idle model released
    60 ms later.
  - **Finding:** on the build before easing, iOS never sent a return to normal. With the ballast freed, Kit still read
    critical for 60 s and refused the next load, and the kernel's level read -1 from the app. Hence the easing above.
  - With easing, freeing the ballast brought available memory to 3,039.1 MiB, more than twice 644.4. Kit reported
    normal, and the next reply loaded.
