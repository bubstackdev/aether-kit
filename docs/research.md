# Research findings

Date: 2026-10-17. Scope: bounded source/documentation review for SDK design. No benchmarks, full codebase audit, App Store review, or physical-device tests were run.

## Source identities

| Source | Inspected identity |
| --- | --- |
| john-rocky/coreai-kit | `87838d6efd9f31540377e3764e7baefaefd86472` |
| apple/coreai-models | `3b25d934ccaee2d05edc6e281c555a5a9a13a55c` |
| Local toolchain | Xcode 27.1, build `27A9275`; Swift 6.4, arm64 macOS 27 target |
| Prototype | Local unversioned snapshot; Cargo manifest pins Corsairs `=0.3.0` |

Upstream shallow checkouts were read under `/private/tmp/aetherkit-research-coreai-kit` and `/private/tmp/aetherkit-research-coreai-models`. The links below pin reviewed code; temporary clones are not durable dependencies.

## Apple runtime and upstream reuse

Apple's repository includes language, image diffusion, video diffusion and speech runtime products. Its root manifest requires macOS/iOS 27 and declares tokenizer, grammar and server-related package dependencies. Product target dependencies differ: measure consumer resolution/build/link costs rather than assuming every declared dependency ships in an app. [Manifest](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/Package.swift)

Its pipelined language engine already uses GPU encoding/sampling and rotating owned buffers. Recommendation: compare it before designing another decode engine. [Implementation](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIPipelinedEngine.swift)

The inspected local `CoreAI.swiftinterface` reexports `CoreAIDelegates`; it is not itself a complete API inventory. Do not infer missing APIs from its short contents. Exact callable signatures and ownership rules need compile probes against the selected SDK.

Apple documents per-device specialization and automatic caching, with OS changes invalidating specialized assets. Recommendation: keep downloadable model identity separate from derived cache identity and qualify first use after OS changes. [Specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)

## john-rocky's SDK

The package has separate core, vision, embedding, UI and task-op targets. Its main library depends on a community fork of Apple's runtime, vision and tokenizers, and includes speech resources and custom Metal resources. The rearchitecture should preserve useful behavior while selecting smaller generative dependencies. This is a dependency/design observation, not a performance finding. [Manifest](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Package.swift)

`ModelStore` is an actor and stages complete downloads before final publication. Its offline transport-error path can return a different cached revision. Its cache identity also excludes the configurable endpoint. AetherKit should retain atomic publication, make resolved identity observable, include origin in identity, and require an explicit policy for revision fallback. [ModelStore](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Sources/CoreAIKitCore/ModelStore.swift)

`ChatSession` is already an actor, rejects overlapping generation, and connects stream cancellation to its generation task. A rewrite still needs proof of native drain, state recovery, and consumer-abandonment behavior; adopting actors is not sufficient justification by itself. [ChatSession](https://github.com/john-rocky/coreai-kit/blob/87838d6efd9f31540377e3764e7baefaefd86472/Sources/CoreAIKit/ChatSession.swift)

The fork compatibility notes in its manifest identify recurrent state, chunked prefill, decode-only models, dynamic KV limits and stop behavior as relevant integration areas. Treat these as an investigation checklist, not independently verified bugs in current Apple upstream.

## Publishing

A tagged hosted Git repository is enough for initial SwiftPM distribution. A registry is optional. Ship installation instructions, platform/toolchain requirements, a license and semantic versions. [Apple publishing guide](https://developer.apple.com/documentation/xcode/publishing-a-swift-package-with-xcode)

Swift Package Index accepts public repositories with a root manifest and semantic release tags and can host DocC with configuration. Confirm its build environment supports the required SDK before relying on it for Core AI validation. [Index submission requirements](https://swiftpackageindex.com/add-a-package)

Source SwiftPM is the initial recommendation. Evaluate an XCFramework only if retaining Rust or another binary requirement makes it useful. Signing/notarization requirements depend on whether we distribute a CLI/app or a library; do not conflate them.

## Questions requiring experiments

1. Does the existing Corsairs path use interfaces suitable for a maintained Apple app SDK across OS/SDK updates? Inspect its current binding implementation and published compatibility guarantees; pipeline success alone cannot answer this.
2. Can direct Swift match the existing Krea fixtures and reduce transfers without altering accepted numerics?
3. Does adopting Apple's utilities improve maintenance without unacceptable dependency or lifecycle costs?
4. Which model geometries fit each target device with measured headroom, and what changes under thermal/memory pressure?
5. What portable export and specialization strategy works for distribution across devices and OS updates?
6. Which checkpoint/tokenizer/adapter bundles can be publicly hosted, and under what notices or access terms?

## Limitations

Web documentation is mutable. Upstream code is pinned above; local prototype and toolchain observations are dated. Some Apple reference pages returned JavaScript shells and one guessed encode URL was inaccessible; no exact API signature is asserted from those failed fetches. No claim of a speedup, universal device support, or production readiness follows from this research.
