# Qualification and release plan

Status: proposed, 2026-10-17. This describes required work; unchecked items are not completed validations.

## Before committing to the runtime

1. Freeze the available prototype source/assets/fixtures and their provenance. Identify missing scratch assets and remove machine-local path assumptions from the reproduction instructions.
2. Inspect Corsairs' current bindings, API support boundary, compatibility tests and packaging. Record whether a Swift facade can preserve its ownership and cancellation guarantees.
3. Build a small direct-Swift Core AI probe for model load, execution, output ownership and release. Compile exact APIs against Xcode 27.1. Follow with matched Krea stages and then end-to-end fixture comparison.
4. Compare runtime candidates using identical model assets, geometry, conditioning, initial noise, precision and schedule. Separate host overhead, GPU time, preparation and compilation. Select the simplest maintainable implementation meeting measured needs.

The first experiment should produce an updated D02 decision with executable reproduction commands, results and limitations. A tiny tensor probe proves API integration, not end-to-end speed.

## Optimization acceptance

- Retain an independent correctness oracle. Export wrappers and reference code must not share the same unverified weight-layout assumptions.
- Compare stepwise intermediates and final artifacts. Preserve exact fixtures for the unchanged baseline; establish numerical and task-quality tolerances before testing optimizations that alter arithmetic.
- Record model/export/tokenizer/adapter hashes, execution recipe, source commit, executable build, device and RAM, OS/SDK/compiler builds, dimensions/context/steps, seed/noise, compute selection and residency.
- Distinguish empty download cache, absent specialization cache, process-cold with OS cache retained, and warm repeated generation. State which caches can actually be controlled.
- Record preparation, first-result latency, generation rate or total time, decoding, peak physical footprint, disk/cache usage and repeat stability. Report sample count and spread; derive percentiles only from enough samples.
- Compare CPU versus device-resident Krea updates, serial versus batch loads, balanced versus resident models, and dynamic versus shape-specialized exports independently before combining wins.
- Test representative prompts and shapes plus edge cases. A fixed fixture pass is not model-wide quality acceptance.

## SDK behavior gates

Before a profile is marked supported:

- [ ] Local bundle preparation and clean download both work; interrupted/corrupt downloads never publish partial assets.
- [ ] Unsupported models/shapes/dtypes/recipes fail with useful errors before large execution.
- [ ] Simultaneous prepare calls deduplicate safely and every observer receives accurate completion/progress.
- [ ] Pinned identity is preserved offline; explicit fallback reports the actual selected identity.
- [ ] Generation repeats without unbounded memory growth; release waits for native work and permits subsequent reuse.
- [ ] Cancel during preparation, model loading and generation; abandon the output stream; verify completion and recovery without leaked work.
- [ ] Concurrent sessions obey admission policy; stress memory pressure and eviction with active leases.
- [ ] The sample app remains responsive and does not perform generation work on the main actor.
- [ ] A sandboxed consumer app works using only declared entitlements and documented storage locations.
- [ ] Progress and media output remain bounded for a slow consumer.
- [ ] Cache invalidation/rebuild after incompatible model/runtime changes has a recoverable path.

Use pure unit/fixture tests for contracts, real Core AI integration tests for execution, and physical devices for platform qualification. Simulator UI/compile checks must not be reported as model execution evidence.

## Model promotion

Krea is the proposed first supported model. Qualify dynamic shapes beyond the existing exact-shape fixture and establish measured device/memory requirements before publishing its support matrix.

The first language model must exercise multi-turn state, prefill/decode, stop/EOS behavior, context bounds, Unicode streaming, cancellation and optional structured-output capability. Do not imply every engine/model supports every feature.

H3 stays experimental until repeated native runs, subjective audio/video quality, memory behavior, cancellation and cache-hit timings are accepted. Retain the QKV regression and full media audit. Earlier base50 partial runs are not completed benchmarks.

## Release sequence

1. Working local Swift package and sandboxed Krea example, with a chosen runtime and no lab-only runtime paths.
2. Versioned asset schema and hosted immutable assets whose distribution terms have been checked. Supply offline/local-bundle instructions.
3. CI for contracts, package builds, documentation and API compatibility; separately operated Apple-silicon native/model gates. Every supported device row identifies what was actually tested.
4. Clean consumer test from the exact release candidate, including download, first generation, cancellation and offline reuse. Test the intended minimum toolchain and each advertised platform.
5. Root README, license/third-party notices, DocC getting started and model guide, support matrix, benchmark methodology, changelog and issue templates.
6. Publish the repository and semantic `0.1.0` tag, release notes, asset identities and evidence; verify installation from the public URL. Submit to Swift Package Index and host DocC when compatible.

Before public publication, settle repository ownership/URL, SDK license, model hosting, credentials policy, supported platforms/devices and release scope. Those choices are open, not assumptions established by these documents. No external publication occurred during research.
