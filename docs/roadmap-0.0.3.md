# 0.0.3 roadmap

Status: draft, 2026-10-27, written right after the [0.0.2 release](release-notes/0.0.2.md). Its two tracks were decided the same day: the backlog deferred from 0.0.2 gates the 0.0.3 tag, and an app-facing language track runs in parallel. Nothing within the backlog is prioritised yet. As in 0.0.2, every item needs AetherKit's own implementation, independent references and qualification, and model licenses are confirmed against their cards before adoption.

## Release gate: the 0.0.2 backlog

### Memory

| Item | Today | Work |
| --- | --- | --- |
| Qwen3-VL decoder weights | `main` and `withImage` each wire their own copy of the text weights: +8.4 and +7.9 GiB on the 4B, and about 16 GiB for a one-image request ([evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)). Opening the decoder reads its 22.5 GiB compiled file into the page cache. | Done: one decoder function; the VL-4B wires 8.4 GiB for an image request ([design and outcome](qwen3-vl-single-decoder.md)). |
| Language memory budget | A 2 GiB growth budget reloads the model to bound Core AI's per-shape-change retention ([guide](language-generation.md#native-memory-retention)). | File the Apple Feedback using the guide's reproduction, and remove the workaround once Core AI stops retaining memory. |
| Kokoro bucket 2048 | About 8.3 GB of Metal, and about 8 GB stays allocated after unload. | Done, no change needed: the footprint falls under 0.3 GB within 2 s of unload. The 8.1 GiB Metal counter that remains is not held memory, and 2048 stays opt-in (`.extended`) ([measurement](kokoro-speech.md#memory-after-unload-003)). |
| H3 second specialization | Denoising adds about 2.7 GiB from step 1, when the two-timestep body joins the one-timestep one. | Release or avoid it. |
| Qwen3-VL-4B export | The converter needs about 33 GB. The [design](qwen3-vl-single-decoder.md#export-memory) found the cause is not per-ExportedProgram copies but about five copies at the self-check. | Done: 16.3 GB peak for the VL-4B ([design](qwen3-vl-single-decoder.md#export-memory)). |
| Krea text encoder weights | Krea's text encoder has the Qwen3-VL decoder's two-function layout (`encode` and `encodeWithImages`; a 20.5 GiB compiled cache), so it likely wires two copies of the text weights when reference images are used. | Measure it; if confirmed, one function as in the [Qwen3-VL design](qwen3-vl-single-decoder.md). |
| FLUX.2 and Krea after unload | 0.4–0.7 GB of CPU footprint remains. | Done, no Kit fix: the cycling test shows the macOS allocator's large-block cache, which stops growing after three cycles, not a Kit or Core AI leak ([FLUX.2](flux2-klein.md#memory-after-unload-003), [Krea](krea-generation.md#memory-after-unload-003)). |

### Speed

- **H3:** check Turbo4's quality (it's already exported), fused attention, video-decode tiling (about 40 s today), and a quiet-host timing reference.
- **Bucketed language:** with power-of-two chunks, time to first token was 2–3× the running contract's; the 64-token width reaches 0.21 s on the writer. Re-measure against the running contract and close any remaining gap.

### Coverage and iPhone

- **iPhone:** Qwen3-VL and the language models. Kokoro: parity at bucket 512 on the device, and other iPhone models.
- **Qwen3-VL:**
  - multi-image M-RoPE;
  - parity on rotated photos;
  - alpha handling, which differs from upstream: PIL keeps the stored RGB, while Kit composites over black.
- **Kokoro fp16:** retest the Neural Engine placement crash on newer OS releases.
- **H3 promotion:** media audit and per-stage lifecycle qualification before the experimental label is dropped, plus a durable home for the qualified profile. Measured admission covers more profiles and Macs as they're measured.

### Safety and correctness

- **FLUX.2 tokenizer:** hash the bundle's `chat_template.jinja`, which changes the fingerprint, then render prompts through `AetherLanguageTokenizers`. Done ([guide](flux2-klein.md#chat-template-in-the-fingerprint-003)).
- **Resource admission** for the Qwen3-VL encoder and decoder.
- **GPU lease:** reclaim pid-less leases. This changes the shared protocol, so it needs Atelier's agreement ([open item](gpu-lease.md#open-item-pid-less-leases)).
- **FLUX.2 evidence:**
  - native reports should record both fixture IDs (done);
  - move reference-mode regeneration onto the CPU text-to-image pictures (done: fixture `80a86a51…`, native parity 23/23, [guide](flux2-klein.md#independent-references));
  - native cancellation of the reference encoder. Done: about 0.18–0.21 s, one image's encoder call ([measurement](flux2-klein.md#reference-encoder-cancellation-003)).

### FLUX.2

- Native cancellation of the text encoder. Done: cancel returns in about 0.17 s during the native call, and in up to about 0.9 s if it lands during a balanced session's function load; Core AI interrupts neither ([measurement](flux2-klein.md#text-encoder-cancellation-003)).
- iOS.
- A quiet benchmark.
- Prompt adherence.

### Infrastructure

- The `stream` Sendable warnings.

### Spikes

Each spike ends in a go/no-go report. All three start from a 0.0.2 feasibility report, which recommends going ahead and ends with what it leaves open for 0.0.3:
- a hybrid recurrent-state language model (Granite 4.0-H): [report](spikes/hybrid-state-2026-10.md), go, at about 190 tok/s on the M5 Max with fixed-shape decode; the first open item is how its state contract fits the bucketed decode contract that shipped;
- ACE-Step music generation: [report](spikes/ace-step-2026-10.md), a conditional go for ACE-Step 1.5 text-to-music on Mac, gated on an 8-step fp16 trajectory check from real prompts;
- TripoSR image-to-3D: [report](spikes/triposr-2026-10.md), go for image to OBJ and USDZ at 256³, about 2.5 s per mesh.

The scripts and compact results behind each report are under `Spikes/`.

## Parallel track: an app-facing language product

Two apps, RenaML and Face Hugger, use FoundationModels `@Generable` today. They asked for an app-facing language product from Kit. The pieces below run alongside the backlog; any that aren't done by the tag move to 0.0.4.

| Piece | Scope in 0.0.3 |
| --- | --- |
| Runtime seam | A sampler or logits hook, and a `LanguageBackend` protocol. The runtime is greedy-only today. |
| Lower deployment floor | Link and launch only. On macOS 14 the facade reports "unsupported OS". An older-OS backend (likely MLX) is a go/no-go spike. The package floor is macOS 14 / iOS 17, with a CI launch check on macOS 14 ([deployment floor](deployment-floor.md)). |
| `AetherLanguage` text facade | Owns the tokenizer, template, stop tokens, reasoning, sampling and UTF-8-safe streaming, building on `AetherLanguageTokenizers`. |
| Structured output | Constrained decoding into `Decodable`, with a `@Generable`-style macro. This adds a swift-syntax dependency. |

**Reference models:** each app picks its own, and Kit qualifies whatever they choose. Quantization isn't decided globally.

## Planned for 0.0.4

- **Model preparation:** import, installation, progress, verification and download, shipping together.
- **Typed readiness states.**
- **A managed lifecycle:** lazy loading, idle unload and memory-pressure response.
- **A sandboxed sample app** with published qualification, and an API freeze for the facade.
- **Quantization:** quantized Qwen3-VL and language decoders, moved from 0.0.3. It is the biggest unknown on the list, and was deferred by user decision.

## Open questions

- The order of the backlog items.
- Whether an older-OS backend goes ahead, after the MLX spike.
- The reference models the two apps choose, and therefore what iPhone and quantization (0.0.4) qualification covers.
