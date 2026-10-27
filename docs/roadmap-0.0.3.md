# 0.0.3 roadmap

Status: draft, 2026-10-27, written right after the [0.0.2 release](release-notes/0.0.2.md). Its two tracks were decided the same day: the backlog deferred from 0.0.2 gates the 0.0.3 tag, and an app-facing language track runs in parallel. Nothing within the backlog is prioritised yet. As in 0.0.2, every item needs AetherKit's own implementation, independent references and qualification, and model licenses are confirmed against their cards before adoption.

## Release gate: the 0.0.2 backlog

### Memory

| Item | Today | Work |
| --- | --- | --- |
| Qwen3-VL decoder weights | `main` and `withImage` each wire their own copy of the text weights: +8.4 and +7.9 GiB on the 4B, and about 16 GiB for a one-image request ([evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)). Opening the decoder reads its 22.5 GiB compiled file into the page cache. | One decoder function for text and image prefill, or cross-function weight sharing if Core AI adds it. Fewer encoder grids would save nothing resident. |
| Language memory budget | A 2 GiB growth budget reloads the model to bound Core AI's per-shape-change retention ([guide](language-generation.md#native-memory-retention)). | File the Apple Feedback using the guide's reproduction, and remove the workaround once Core AI stops retaining memory. |
| Kokoro bucket 2048 | About 8.3 GB of Metal, and about 8 GB stays allocated after unload. | Release it, or keep 2048 opt-in only. |
| H3 second specialization | Denoising adds about 2.7 GiB from step 1, when the two-timestep body joins the one-timestep one. | Release or avoid it. |
| Qwen3-VL-4B export | The converter needs about 33 GB, because it holds weights per ExportedProgram. | Bound the export's memory. |
| FLUX.2 and Krea after unload | 0.4–0.7 GB of CPU footprint remains. | Add a load/unload cycling test, then fix what it finds. |

### Speed

- **H3:** check Turbo4's quality (it's already exported), fused attention, video-decode tiling (about 40 s today), and a quiet-host timing reference.
- **Bucketed language:** with power-of-two chunks, time to first token was 2–3× the running contract's; the 64-token width reaches 0.21 s on the writer. Re-measure against the running contract and close any remaining gap.

### Coverage and iPhone

- **iPhone:** Qwen3-VL and the language models. Kokoro: parity at bucket 512 on the device, and other iPhone models.
- **Qwen3-VL:**
  - multi-image M-RoPE;
  - parity on rotated photos;
  - alpha handling, which differs from upstream: PIL keeps the stored RGB, while Kit composites over black.
- **Quantization:** quantized Qwen3-VL and language decoders.
- **Kokoro fp16:** retest the Neural Engine placement crash on newer OS releases.
- **H3 promotion:** media audit and per-stage lifecycle qualification before the experimental label is dropped, plus a durable home for the qualified profile. Measured admission covers more profiles and Macs as they're measured.

### Safety and correctness

- **FLUX.2 tokenizer:** hash the bundle's `chat_template.jinja`, which changes the fingerprint, then render prompts through `AetherLanguageTokenizers`.
- **Resource admission** for the Qwen3-VL encoder and decoder.
- **GPU lease:** reclaim pid-less leases. This changes the shared protocol, so it needs Atelier's agreement ([open item](gpu-lease.md#open-item-pid-less-leases)).
- **FLUX.2 evidence:**
  - native reports should record both fixture IDs;
  - move reference-mode regeneration onto the CPU text-to-image pictures at the next intentional regeneration.

### FLUX.2

- Native cancellation of the text encoder.
- iOS.
- A quiet benchmark.
- Prompt adherence.

### Infrastructure

- The `stream` Sendable warnings.

### Spikes

Each spike ends in a go/no-go report:
- a hybrid recurrent-state language model (Granite 4.0-H);
- ACE-Step music generation;
- TripoSR image-to-3D.

## Parallel track: an app-facing language product

Two apps, RenaML and Face Hugger, use FoundationModels `@Generable` today. They asked for an app-facing language product from Kit. The pieces below run alongside the backlog; any that aren't done by the tag move to 0.0.4.

| Piece | Scope in 0.0.3 |
| --- | --- |
| Runtime seam | A sampler or logits hook, and a `LanguageBackend` protocol. The runtime is greedy-only today. |
| Lower deployment floor | Link and launch only. On macOS 14 the facade reports "unsupported OS". An older-OS backend (likely MLX) is a go/no-go spike. The package floor is 27 today. |
| `AetherLanguage` text facade | Owns the tokenizer, template, stop tokens, reasoning, sampling and UTF-8-safe streaming, building on `AetherLanguageTokenizers`. |
| Structured output | Constrained decoding into `Decodable`, with a `@Generable`-style macro. This adds a swift-syntax dependency. |

**Reference models:** each app picks its own, and Kit qualifies whatever they choose. Quantization isn't decided globally.

## Planned for 0.0.4

- **Model preparation:** import, installation, progress, verification and download, shipping together.
- **Typed readiness states.**
- **A managed lifecycle:** lazy loading, idle unload and memory-pressure response.
- **A sandboxed sample app** with published qualification, and an API freeze for the facade.

## Open questions

- The order of the backlog items.
- Whether an older-OS backend goes ahead, after the MLX spike.
- The reference models the two apps choose, and therefore what iPhone and quantization qualification covers.
