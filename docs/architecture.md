# Product and architecture

Status: proposed architecture with an initial reference implementation. Date: 2026-10-17. See [implementation status](implementation-status.md).

## Product intent

AetherKit is an independently published Apple developer SDK for generative AI using Core AI. Its distinguishing work is model-specific correctness, execution optimization, and predictable app integration. It is not an Apple product.

The long-term scope includes language, images, speech/audio, and video with audio. Supporting conditioning, embeddings, and media utilities belong where they enable those workflows. General computer vision, hosted APIs, agent orchestration, UI components, and servers are deferred unless a concrete generative use case needs them.

## Initial delivery

Proposed first vertical slice: Krea text-to-image in a sandboxed macOS app, using external assets, reproducing existing correctness fixtures, with repeat generation and explicit resource release. Follow with one small language model to challenge the shared runtime design. H3 remains experimental pending broader qualification. iOS/iPadOS support requires physical-device qualification of each model/profile.

## Ownership boundaries

| Boundary | Owns | Does not own |
| --- | --- | --- |
| Public SDK | Typed requests/results, preparation, capability reporting, events, sessions, errors | Tensor layouts and checkpoint-specific names |
| Model family | Prompt/conditioning semantics, state, schedule, sampling, decoding, execution recipe | Download transport and app UI |
| Shared execution | Core AI calls, completion lifetimes, Metal interop, specialization, admission and residency accounting | A universal generation algorithm |
| Assets | Immutable identities, manifests, download staging, integrity, variants and eviction | Implicit model substitution |
| Offline Forge | Pinned conversion/compression recipes, oracles, qualification and benchmark artifacts | Dependencies required inside the consumer app |

These are code ownership boundaries. Determine actual Swift targets from dependency isolation and compile/link behavior; avoid one product per conceptual box by default. Model-specific code and assets should not make unrelated modalities mandatory dependencies.

## Runtime decision

Swift is the proposed public API language. The inference implementation language remains open.

Compare direct Swift/Core AI, appropriately reused Apple runtime utilities, and a thin Swift facade over the existing Rust/Corsairs path. Measure identical assets and algorithms. Include packaging, supported API usage, ABI compatibility, buffer ownership, cancellation, and sandbox behavior alongside latency/memory. No language has a demonstrated performance advantage in this project comparison yet.

The adjacent lab's Rust-binding research records earlier preferences against Python and Swift wrappers for that binding project. Preserve that context; it is not automatically a requirement for this new Apple SDK. Do not discard the Rust implementation before an explicit runtime decision is recorded.

## Public behavior to specify before freezing APIs

- Explicit preparation reports missing assets, byte counts, unsupported configurations, and specialization work. Support supplied local bundles and network preparation with a caller-selected policy.
- Sessions own mutable generation state. Define whether simultaneous requests queue or fail; actor isolation alone does not prevent interleaving across suspension points.
- Cancellation stops new work cooperatively. Already submitted native/GPU work retains resources until completion. Provide an awaitable completion/release contract; consumer stream abandonment must not strand work or buffers.
- Progress streams are bounded. Coalesce disposable progress/previews; token/media delivery requires a lossless bounded policy or explicit backpressure. Define terminal success/failure/cancellation and partial-result behavior.
- Global resource admission prevents several individually valid sessions from exceeding a shared budget. Preserve explicit ownership and avoid hidden process-global model handles.
- Report estimates separately from measured memory, download bytes, source disk usage, and specialized cache usage. Unknown capacity remains unknown.
- Use typed modality requests. Keep model-specific controls discoverable without forcing chat, diffusion, and joint media generation into a universal enum.
- Diagnostics include model/recipe identity and stage timings. Prompt/content logging is opt-in.

## Asset and execution recipe contract

A manifest should identify schema version, upstream repository and immutable revision, source/export hashes, tokenizer/template, adapters, precision/compression, input/output/state contracts, supported geometry/context, required OS/runtime, and qualification records. Reject unknown required fields/recipes and incompatible bundles before expensive execution.

Stage downloads, verify declared contents, then publish atomically. Identity includes origin, revision, variant and content, preventing custom endpoints from sharing ambiguous cache keys. A pinned request must never silently substitute a different revision; expose explicit fallback policy and the resolved identity.

Keep portable source assets separate from runtime specialization caches. Specialization is dependent on device/OS/options; an OS update may require regeneration. Atomic publication and eviction must respect active model leases. Schema migrations and failed downloads must leave previous complete bundles usable.

## Decision register

| ID | Proposal | Status / evidence needed |
| --- | --- | --- |
| D01 | Swift public SDK, SwiftPM distribution | Package and separate local consumer build/run pass; public release pending |
| D02 | Runtime language and upstream reuse | Open; matched runtime experiment |
| D03 | macOS-first Krea slice | Proposed; assets, distribution rights and minimum memory qualification |
| D04 | Core AI runtime floor 27 | Research-backed; choose package deployment floor separately |
| D05 | Portable assets plus per-device specialization | Proposed; qualify cache invalidation and offline recovery |
| D06 | Small release catalog with explicit experimental entries | Proposed; per-profile evidence governs promotion |
| D07 | Exact identities with opt-in revision fallback | Proposed; test offline and concurrent preparation behavior |

Do not freeze illustrative type names or promise compatibility until the first real app exercises these contracts.
