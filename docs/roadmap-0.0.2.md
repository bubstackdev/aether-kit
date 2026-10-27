# 0.0.2 roadmap

Status: **released 2026-10-27** ([release notes](release-notes/0.0.2.md)). Scope agreed 2026-10-26; release decisions recorded the same day and extended on 2026-10-27 (below). This replaces the draft on `local-line-2026-10-26`, which planned H3 around prototype bundles; main's schema-3 H3 path makes that unnecessary. Prototype evidence from `aether-engine` lowers porting cost but is not AetherKit support. Every item needs AetherKit's own implementation, independent references and qualification. Confirm licenses against model cards before adoption.

## Scope

| Item | New capability | License | Status at agreement |
| --- | --- | --- | --- |
| FLUX.2 [klein] 4B | Image editing and multi-reference conditioning | Apache-2.0 (card checked) | Export plan approved; engine prototype verified on Mac and iPhone |
| Fixed-shape language decode export | Removes per-token shape growth and the 128-submission model reload | n/a | Implemented on `v0.0.2/language`; Qwen3-0.6B fp16 qualified natively on M5 Max (docs/evidence/language-bucketed-decode-2026-10-26.json) |
| Qwen3-VL-2B | Image input to the language runtime | Apache-2.0 (confirmed, not gated) | Implemented on `v0.0.2/qwen3-vl`; fp16, fixed-shape image grids, Mac-qualified against upstream on synthetic scenes and three photographs (docs/qwen3-vl.md) |
| Kokoro 82M | Speech synthesis (first audio-generation product) | Apache-2.0 (card checked) | Phoneme-input v1 approved: G2P behind a protocol, no GPL dependency |
| **H3 schema-3 qualification** | Promote `H3VideoSession` from experimental, with committed evidence | Apache-2.0 pipeline; confirm checkpoint/adapter terms | Assets present (complete schema-3 profile); evidence missing |
| **Shared GPU lease and admission** | One Kit-owned implementation of GPU coordination, default-on in Kit tools, opt-in for app sessions | n/a | Added 2026-10-26 |

Spikes, with go/no-go reports for 0.0.3: a hybrid recurrent-state language model (Granite 4.0-H; baseline about 112–117 tok/s), ACE-Step music generation and TripoSR image-to-3D.

## Release decisions (2026-10-26)

### In 0.0.2

| Item | Decision | Where |
| --- | --- | --- |
| FLUX.2 [klein] 4B | Text-to-image and multi-reference editing, experimental | On main (#17) |
| Shared GPU lease and admission | `SharedGPULease`, `GPUAdmission` and the session opt-in; default-on in Kit tools and native suites | On main (#18) |
| Bucketed language decode | `bucketedPositions` contract; the default for new exports (`--contract running` keeps the 0.0.1 contract) | On main (#19) |
| Qwen3-VL-2B | Image input on fixed-shape image grids (five patch grids, at most 484 tokens) | On main (#24) |
| Kokoro 82M | Speech synthesis, float32 decoder only; frame cap 1024 on macOS and 512 on iOS | On main (#21) |
| H3 schema-3 | Qualification with an 18 GiB hard memory bar, enforced by measured admission; the bar binds only sessions created with measured admission. Device-chained denoiser blocks cut denoising 7% | On main (#25, #27) |
| `GeneratedImage.pngData()` | In-memory PNG encoding | On main (#20) |
| AetherLanguageTokenizers | A model's own chat template to token IDs | On main (#22) |
| Qwen3-4B writer (Atelier's engine drop) | Bucketed export of the abliterated Qwen3-4B-Instruct-2507 | On main (#23) |
| Qwen3-VL-4B describer (Atelier's engine drop) | Export for Atelier's image describer | On main (#26) |
| Language memory budget | Workaround for Core AI retaining memory on every input-shape change: a fixed prefill width (64) and a 2 GiB growth budget that reloads the model at idle | On main (#28) |
| Forge environments | Two locked Python environments in the repository (`Forge/`, `Forge/h3-oracle`); no tooling depends on the superseded prototype or on Atelier staging | On main (#29) |
| Qwen3-VL memory attribution | Decoder weights wired twice (text and image functions); encoder grids are views of one copy | On main (#31) |
| FLUX.2 CPU references | References computed on the CPU in float32, like every other model's; parity 113/113 and lifecycle 44/44 | On main (#34) |
| FLUX.2 Forge exporter | `Forge/flux2` exports the bundle; its export `a48aaf07` is the qualified 0.0.2 bundle | On main (#35) |
| GitHub CI | Weight-free Release suite and consumer packages on every PR; the repository moved to `bubstackdev/aether-kit` | On main (#32, #33) |

The 0.0.2 cleanup moves duplicated helpers onto the shared ones with no behaviour change (see CHANGELOG).

### Deferred to 0.0.3

- **Spikes:** hybrid recurrent state (Granite 4.0-H), ACE-Step and TripoSR.
- **H3:** Turbo4 and fused attention; video-decode work, and releasing the second specialization. Media audit and per-stage lifecycle qualification before dropping the experimental label, and a durable home for the qualified profile.
- **Qwen3-VL:** iPhone, multi-image input and a quantized decoder.
- **Qwen3-VL memory:** attributed (#31): Core AI wires a separate copy of the decoder's weights for `main` and `withImage` (VL-4B: +8.4 and +7.9 GiB wired, file-backed, so outside the process footprint); the per-grid encoder growth in Metal's accounting is views of one copy, not resident memory ([qwen3-vl.md](qwen3-vl.md#limits-and-follow-ups), [evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)). Fix: one decoder function for text and image prefill, or cross-function weight sharing; the Core AI SDK has no shared-constants option.
- **Language memory budget:** remove the reload workaround once Core AI stops retaining memory per shape change.
- **Kokoro:** bucket-2048 memory (about 8.2 GB of Metal stays allocated after that bucket has been used).
- **GPU lease:** reclaiming pid-less leases. This changes the shared protocol, so it needs Atelier's agreement ([open item](gpu-lease.md#open-item-pid-less-leases)).
- **FLUX.2 tokenizer:** hash the bundle's `tokenizer/chat_template.jinja`, then render `Flux2Tokenizer` prompts through AetherLanguageTokenizers. Today the template is neither verified nor part of the fingerprint, so FLUX.2 keeps its fixed render. Hashing it changes the bundle fingerprint. An opt-in local-files test already shows that the padded output is identical.

## H3 schema-3 qualification

The profile at Atelier's staging export (`h3-fl2va`, 28 GB) reads checkpoint ranges in place and needs no re-export.

1. **Pin assets.** Record bundle and adapter hashes, run the opt-in profile tests, and choose a durable home for the profile.
2. **Graph gate** at 960 for T2VA and FL2VA against the CPU oracle, with a 0.02 relative-L2 limit.
3. **Trajectory gate:** capture natively, then compute CPU references at the captured contexts (tables, prepare, blocks 0/49, heads, scheduler updates) for all eight steps. Whole free-running runs diverge from the CPU oracle, so per-context references are the acceptance method.
4. **Decoders:** bf16 and f32 tile decoders and the fixed audio decoders against CPU.
5. **Lifecycle and memory:** cancellation at each stage, concurrent unload, retained and reopened recovery. New memory limits are needed: FL2VA samples about 29 GB of Metal. These runs also supply qualified `H3ResourceAdmission` estimates.
6. **Media audit and recorded subjective review,** then drop the experimental label.

Estimate: about 2–2.5 GPU-hours, 8–12 CPU-hours and 20–30 GB of new disk.

## Shared GPU lease and admission

Today GPU exclusivity is coordinated three different ways: Atelier's `SharedGPULock` (a `/tmp/atelier-gpu.lock` directory lease with a pid file and `.reclaim`-serialized stale recovery), an external shell wrapper used by development agents, and `aether-probe` usage text asking callers to hold the lock themselves.

- **`SharedGPULease`.** A Kit implementation of the existing directory-lease protocol, wire-compatible with Atelier's: atomic `mkdir`, an atomic pid write, release only while the lease still names its owner, signal-safe release, and stale reclaim that moves the lease aside under the `.reclaim` flock and never replaces a lease created since. Waiting and fail-fast acquisition modes, with timeouts and cancellation. CPU tests use private lock directories and never touch the shared path. Atelier can then depend on this instead of its own copy.
- **Default-on in Kit tools.** `aether-probe` GPU commands (including the H3 profile commands), the benchmark harness and the opt-in native test suites acquire the lease themselves.
- **Opt-in for app sessions.** Krea, language, H3 and future sessions accept a shared GPU admission object alongside `ResourceCoordinator`. It queues or fails fast in-process, with the cross-process lease as an optional backend for unsandboxed hosts. It's not a process-global default: sandboxed apps get a private `/tmp`, and apps may legitimately run models concurrently.
- **Reconcile with `FairInferenceQueue`.** That untracked, transport-independent admission design was found in the older checkout; fold it into this one rather than building a parallel one.
- **Accept when:** the CPU tests cover acquire/release/stale/race/signal paths, `aether-probe` and the native suites hold the lease during GPU work, Atelier and Kit tools exclude each other in a recorded check, and the session APIs document the opt-in.

## Sequencing

1. The language decode export, the H3 asset pinning and graph gate, and the shared GPU lease can start immediately. The lease removes the external wrapper the other workstreams currently depend on.
2. FLUX.2 klein and Kokoro proceed in parallel.
3. Qwen3-VL follows the decode export.
4. H3 steps 3–6 are sequential and GPU-heavy, so schedule them around the other workstreams' GPU needs.
