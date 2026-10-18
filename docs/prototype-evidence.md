# Prototype evidence and migration map

Reviewed: 2026-10-17. Source: `../../lab/aether-prototype`. This is a review of retained records and selected source, not a rerun of inference or an independent audit of every capture.

## Current implementation

The unpublished Rust `aether` 0.1.0 crate requires Rust 1.99 and pins published Corsairs 0.3.0. It forbids unsafe code in the Aether crate. That does not establish the implementation or public-interface status of all transitive native bindings. Models remain external; the crate performs no model downloads.

| Area | Retain | SDK work |
| --- | --- | --- |
| `src/krea/prompt.rs`, `math.rs` | Tokenization, template, noise, rotary and schedule contracts | Port or bridge with independent fixtures |
| `src/krea/assets.rs` | Source identity, specialization matching, recipe checks | Versioned portable asset manifest |
| `src/krea/pipeline.rs` | Model ownership, cache, residency, serial generation | Swift lifecycle, shared admission, native cancellation/reuse validation |
| `src/h3`, `forge/h3_*` | Packing, adapters, QKV normalization, scheduler and tiled decode | Experimental model package and reproducible export tooling |
| Tests and `docs/evidence` | Positive, negative and superseded evidence | Select small redistributable fixtures; remove machine-path dependence |

## Krea evidence

The validation record reports a 27-test release suite, Clippy, formatting and documentation checks. These were not rerun for this research pass.

The retained encoded native report shows token IDs matching, selected text matching, all eight latent hashes matching, and decoded pixels matching an independent Swift baseline. It used parallel loading and the registered 4,032-image-token / 20-conditioning-row specialization. Source streams were verified.

- [Encoded report](../../lab/aether-prototype/docs/evidence/krea-encoded-native.json): `passed: true`, approximately 199.80 seconds for that instrumented run.
- [Provided-conditioning report](../../lab/aether-prototype/docs/evidence/krea-provided-native.json): separate baseline; do not mix its hashes with the encoded path.
- [Validation record](../../lab/aether-prototype/VALIDATION.md): documents ordinary-shell Core AI cache failures and subsequent controlled native success.

The dynamic path accepts multiples of 16 with a 4,096-image-token limit; the retained exact-shape native gate does not qualify every accepted dynamic shape. The resident policy was motivated by a 64 GiB host; it establishes no minimum-memory guarantee.

Latents remain float32 and use separately rounded multiplication and addition before fp16 model boundaries. Moving the update to Metal can change rounding through fusion; retain the exact baseline and declare acceptance criteria before comparing an optimized implementation. The prototype currently returns velocity to the CPU for Euler updates.

Open native coverage: repeated generation, residency transitions, cancellation/drop recovery, explicit unload, warm throughput, peak memory and sandboxed application behavior.

## H3 evidence

[Corrected Turbo result](../../lab/aether-prototype/docs/evidence/h3-qkv-render-summary.json): 960×544, 124 frames, 24 fps, stereo audio, eight complete joint evaluations. Generation took 686.02 seconds; a separate numerical gate plus generation took 708.33 seconds within the 1,200-second window.

Sampled stills were accepted as a usable visual pilot; subjective audio listening was not performed. Audio signal/media audits passed. This was a preparation-cache miss with backend/OS caches retained, not a controlled cold-start or warm-repeat benchmark. Full-trajectory CPU/Swift parity is not claimed.

An earlier export omitted per-head QKV reordering. Earlier numerical agreement shared that omission and did not qualify intended checkpoint semantics. Preserve the independent sentinel regression and require the corrected layout marker. See [correction and scope](../../lab/aether-prototype/docs/h3-turbo.md).

The base50 production attempt did not finish; partial results and projections must not become throughput claims. H3 has no cooperative cancellation-token API in the prototype.

## Portability and provenance gaps

- `.local` is a symlink into an external scratch volume. Retained documentation also references other machine-specific repositories. Inventory available assets and rights before moving anything.
- Freeze source checksums, manifests, recipes, fixture identities and dependency locks before using the prototype as a migration oracle.
- Preserve Apache-2.0 source license and NOTICE provenance. Tokenizers, checkpoints and adapters have separate distribution terms; downloading successfully does not establish permission to redistribute them.
- Adjacent [Rust-binding research](../../lab/docs/coreai-rust-bindings-research.md) is dated feasibility research predating the completed Corsairs-backed pipeline. Do not treat its then-unproven status as a current failure, or the later pipeline success as proof of public-API/App Store suitability.
