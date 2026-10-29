# 0.0.4 roadmap

Status: draft, 2026-10-28, written right after the [0.0.3 release](release-notes/0.0.3.md). The user decided its shape
the same day: **app readiness gates the 0.0.4 tag**, and three 0.0.3 spikes become products alongside it with a
hardening track. Anything in a parallel track that isn't done by the tag moves to 0.0.5. As before, every item needs
AetherKit's own implementation, independent references and qualification, and model licenses are confirmed against their
cards before adoption.

## Release gate: app readiness

RenaML and Face Hugger build on the 0.0.3 `AetherLanguage` facade. What they still lack is everything around a model
that isn't inference: getting it onto the device, knowing whether it can run, and keeping it from costing memory while
idle. The gate is those pieces plus a sample app that proves them together.

| Piece | Scope |
| --- | --- |
| Model preparation | Implemented and qualified 2026-10-29 ([model preparation](model-preparation.md)). Import, installation, progress, verification and download, shipping together. Builds on `AetherAssets`' staged SHA-256-verified installation and reader leases. Download transport is pluggable; Kit still distributes no weights. |
| Typed readiness | One state per model that an app can show: not installed, installing (with progress), verifying, ready, unsupported (OS, platform, memory), failed. `availability` stays the no-load check. Implemented 2026-10-29 ([model readiness](model-readiness.md)). |
| Managed lifecycle | Lazy loading, idle unload and a memory-pressure response, for language, Qwen3-VL and the generative products. It folds in the language memory budget, which reloads a model once Core AI's retention spends its growth allowance. |
| Sample app and API freeze | A sandboxed sample app that prepares, checks and runs a model through the facade, with its qualification published. The facade's API (`AetherLanguageModel`, sessions, structured output) freezes for 0.0.x: later 0.0.x releases add to it without breaking it. |

## Parallel track: products from 0.0.3 spikes

Each was a go in 0.0.3. Product work means a Kit product with its own manifest kind, Forge exporters, independent
references, native qualification and a guide.

- **TripoSR, image to 3D** ([spike](spikes/triposr-2026-10.md)): image to coloured mesh (OBJ and USDZ) at 256³ on Mac,
  with the gather-free grid function. iPhone is a separate decision (about 800 MB of fp16 weights, and the decoder runs
  on every grid point).
- **ACE-Step 1.5, text to music** ([spike](spikes/ace-step-2026-10.md)): without the 5 Hz language model; fp16 DiT and
  encoders, the VAE in fp32 decoded in tiles, and Kit masking the padded condition sequence itself.
- **Granite 4.0-H, hybrid language** ([spike](spikes/hybrid-state-0.0.3.md)): the static-width step contract the spike
  qualified, behind `LanguageBackend`. Re-qualify a dynamic-q contract with positions passed as a `[1, 1]` start, which
  removed the 0.0.3 GPU failures; that would let hybrids use the bucketed planner without padding.

## Parallel track: hardening and coverage

- **iOS open-time memory check:** count mapped asset bytes, not only footprint (`os_proc_available_memory()` measures
  footprint), and refuse an over-budget open instead of failing inside Core AI. Add the extended-virtual-addressing
  entitlement beside the increased memory limit, then run the VL-2B on an iPhone with both
  ([on-device host](language-generation.md#on-device-host-003)). Find out why its assets appear to be mapped more than
  once.
- **FLUX.2 on iOS,** moved from 0.0.3 by user decision.
- **An MLX backend for older OSes,** as an optional separate product, by user decision. The
  [spike](spikes/mlx-backend-2026-10.md) is a conditional go: mlx-swift core only, the Metal toolchain in CI, and its
  own manifest kind for Hugging Face safetensors.
- **H3:** Turbo4 qualification (the user's side-by-side review judged it worth qualifying). **Fused attention is
  done, with no re-export** ([spike](spikes/h3-attention-2026-10.md), 2026-10-29): the qualified body graph already
  exports `F.scaled_dot_product_attention` as Core AI's SDPA composite, and the GPU runs it fused. No score tensor is
  allocated, it runs at 44–56 TFLOPS, and it's 14.7× faster than explicit attention at 9,728 rows. H3's qualification
  evidence stands. The step-0 transient in the admitted worst case and profile paths relative to `bundle.json` are
  follow-ups.
- **For Atelier** (user decision, 2026-10-29), for Atelier's v0.6, with the features that use them in its v0.7:
  - **H3 at a larger canvas:** first-frame encoder and tail exports for 1280×704 and 704×1280, measured, and added to
    measured admission.
  - **H3 last-frame conditioning:** `H3VideoRequest.lastFrame` is refused today.

  Atelier's team authors both as `v0.0.4/<topic>` PRs, which Kit reviews. They build on the current body graph.
  - **Shared declared residency** ([proposal](shared-residency-proposal.md)): a joint, announced calibration window on
    this Mac sets the capacity fraction. Kit's side follows: the lease rules, a records-backed `ResourceCoordinator`,
    and declarations by `aether-probe` and heavy CPU jobs.
- **Exporter safety,** from the 0.0.3 follow-up bug hunt on macOS 27.2:
  - An export-time check that runs the optimized program against the unoptimized one on random inputs. Core AI's
    pre-compilation rewrite gives silently wrong results for some reshape-into-matmul patterns.
  - A lint that refuses float masks of rank 2 or 3, and `is_causal=True`, into fp16 attention. On macOS 27.2's
    cpu_only path, those masks are misread. Kit's graphs use bool or 4-D masks today.
  - A guard for one-layer graphs, where removing identity copies deletes a state write.
- **Core AI retention on shape changes:** keep the prefill tail and decode widths within 2x of each other where
  possible, and measure whether turning `expectFrequentReshapes` off for language sessions avoids the retention at an
  acceptable first-call cost per new shape. Until then the memory budget stays.
- **Kokoro fp16 decoder:** keep the AdaIN style linears in float32 (the convolutions stay fp16). The float16 decoder's
  Neural Engine abort comes from the number of buffers one Neural Engine request needs, not from fp16 itself; update
  the [guide](kokoro-speech.md#reduced-precision-decoders).

## Spikes

- **Quantization,** go/no-go only, by user decision: Core AI's quantized paths for Qwen3-VL and the language decoders,
  measured for quality, speed and memory on the apps' reference models. Product work would be 0.0.5.
- **One length input for dense language contracts,** by user decision: Kit's bucketed Qwen3 and Qwen3-VL contracts pass
  the prompt length on two inputs (`input_ids` [1, q] and `cache_positions` [1, q]), the pattern behind the hybrid
  failures. Measure whether a single input (a `[1, 1]` start) changes anything for dense models (retention, prefill
  speed, the unexplained `Input has no defining op` compile message), then decide on a v2 contract. 0.0.2 and 0.0.3
  bundles keep working either way.

## Open questions

- The reference models the two apps choose, and therefore what quantization and iPhone qualification cover.
- iPhone scope for TripoSR and ACE-Step.
- The order within each track.
- Which Core AI defects to file with Apple; the follow-up bug hunt's drafts are with the user.
