# H3 FL2VA conditioner presentation: implementation plan

Status: plan, 2026-10-29, for review before any export or GPU work. It fixes the
[known deviation](h3-integration.md#known-deviation-fl2va-conditioner-presentation-found-2026-10-29): Kit's FL2VA
conditions on the prompt text alone, while upstream also presents each keyframe to the Qwen3-VL conditioner. Nothing
here has been exported or run on the GPU. Figures are quoted from the cited files. Anything marked **unverified** is
arithmetic or inference that a step below must measure.

Upstream references are diffusers `7ecb1e1b` (`src/diffusers/...`) and transformers 5.20.1, the version pinned in
`Forge/h3-oracle/uv.lock`, abbreviated `tf/` for `transformers/`. Kit references are repo-relative.

## 1. What upstream computes

### Presentation (`modular_pipelines/minimax_h3/encoders.py`, `MiniMaxH3FL2VATextEncoderStep`, 213–317)

- **Keyframes.** The step receives the keyframes already on the target canvas. `MiniMaxH3ResizeStep`
  (`before_encoder.py` 135–161) passes a keyframe through when it is already canvas-sized (139). Otherwise it stretches
  the first keyframe with LANCZOS (140–145) and cover-crops a follower, the last frame (146–160). Kit's `H3Keyframe`
  must already be at the clip size (`Sources/AetherH3/H3VideoSession.swift` 4–15, 29–31), so the canvas-sized RGB8 Kit
  already holds is the image upstream gives the processor.
- **Vision inputs.** `processor.image_processor(images=keyframes)` is called once, batched, at 278–280.
- **Token ids.** These are built per keyframe, in packed order (285–298):
  - `tokenizer("<Picture i>: ", add_special_tokens=False)`, then `<|vision_start|>`, then `<|image_pad|>` repeated
    `prod(grid_thw) // merge²` times, then `<|vision_end|>`;
  - after the keyframes, the prompt, tokenized on its own (299–301).

  Each piece is tokenized separately and the ids are concatenated; the joined string is never tokenized. With the
  FL2VA tokenizer, the label is 6 tokens: `[21604, 3826, 220, i, 26818, 220]`, where `i` is the digit's id (16 for 1,
  17 for 2). The special ids are 151652 (vision start), 151655 (image pad) and 151653 (vision end). These come from a
  tokenizer-only probe and match `FL2VA/text_encoder/config.json`.
- **Tags** (297–301, 313). The label rows and the prompt rows get the text tag (1). The *whole* vision block gets the
  video tag (0): start, pads and end. The constants are at `modular_pipeline.py` 24–26 (video 0, text 1, audio 2).
- **Conditioner call.** `get_qwen3vl_prompt_embeds` (30–99) builds `mm_token_type_ids` with
  `processor.create_mm_token_type_ids` (73): 1 for image-pad ids, 0 for everything else, so vision start and end count
  as Qwen *text*. It casts `pixel_values` to the conditioner dtype, bf16 (77). It calls `text_encoder.model(...)` with
  an all-ones `attention_mask`, `use_cache=False` and `output_hidden_states=True` (88–96), and returns
  `hidden_states[50]` (99). The block emits the conditioner dtype, bf16 (205, 311).

### Image processor (`FL2VA/processor/preprocessor_config.json`, `tf/models/qwen2_vl/image_processing_qwen2_vl.py`)

- **Settings.** Patch 16, merge 2, temporal patch 2, mean = std = 0.5, `shortest_edge` 65,536 and `longest_edge`
  16,777,216 pixels. `image_processor_type` is `Qwen2VLImageProcessorFast`, which is the torchvision backend in
  transformers 5.
- **`smart_resize`** (63–90) rounds each side to a multiple of 32. Kit canvases are already multiples of 32
  (`H3Geometry`), so the resize is the identity. A probe with the PIL backend, which needs no torchvision, confirmed
  this: `do_resize` on and off give equal `pixel_values` at 960×544, 1280×704 and 1344×768.
- **Normalization and patchify.** (x/255 − 0.5)/0.5, then `patchify` (178–210). Rows come in merge-window order, each
  `[C][T][16][16]`, and the frame is repeated across the two temporal slices. The processor outputs `pixel_values`
  [patches, 1536] f32.
- **Image tokens per keyframe.** From the probe:

| Canvas | `image_grid_thw` | Patches | Image tokens | Rows per keyframe (6 + 1 + N + 1) |
| --- | --- | ---: | ---: | ---: |
| 960×544 | (1, 34, 60) | 2,040 | 510 | 518 |
| 1280×704 | (1, 44, 80) | 3,520 | 880 | 888 |
| 1344×768 (upstream default canvas) | (1, 48, 84) | 4,032 | 1,008 | 1,016 |

544×960 is the transpose of 960×544, with the same counts. Two keyframes at 960×544 give a grid of
`[[1,34,60],[1,34,60]]` and 4,080 patch rows.

### Vision tower (`tf/models/qwen3_vl/modeling_qwen3_vl.py`, `Qwen3VLVisionModel` 628–736)

- **Config** (`FL2VA/text_encoder/config.json`): depth 27, hidden 1152, 16 heads, MLP 4304 with tanh GELU,
  `out_hidden_size` 5120, `deepstack_visual_indexes` [8, 16, 24], 2,304 learned positions (a 48×48 grid).
- **Embedding.** The patch embed is a Conv3d with kernel equal to stride, so it is a linear map 1536 → 1152 (85–102).
  The learned positions are resampled bilinearly with corners aligned (646–650, 690–697), in merge-block order.
- **Vision RoPE.** 2-D axial, with rows driving the first half of the frequencies and columns the second (105–163).
  `apply_rotary_pos_emb_vision` works in f32 and casts back (189–200).
- **Attention.** Attention is non-causal (`is_causal = False`, 252) and runs within each image: `cu_seqlens` is per
  image (699, 278–311). One tower call per keyframe is therefore equivalent to upstream's batched call.
- **DeepStack.** After blocks 8, 16 and 24, a DeepStack merger maps each 2×2 window to 5120, applying its LayerNorm
  after the concatenation (718–722). The final merger applies its LayerNorm per patch and then concatenates (724). The
  output is 510 rows × 5120 per 960×544 keyframe, plus three DeepStack levels of the same shape.
- **dtype.** The model is bf16 throughout. `get_image_features` casts the pixels to the tower dtype (1039).

### Into the decoder (`Qwen3VLModel.forward` 1143–1251, `Qwen3VLTextModel.forward` 762–845)

- **Scatter.** The merger rows replace the embedding rows of the image-pad tokens, in order (`masked_scatter`, 1190).
- **DeepStack injection.** After decoder layers 0, 1 and 2, level k is added at the image-pad rows only
  (833–838, `_deepstack_process` 847–856, built at 1219–1222). It is a bf16 add. The tap, `hidden_states[50]`, is the
  output of decoder layer 49, recorded from the layer's output (622–625). So DeepStack **does apply** to the layer-50
  read: its additions at layers 0–2 propagate through layers 3–49.
- **Mask and dtype.** The mask is causal (`create_causal_mask`, 807; text attention `is_causal = True`, 478) and the
  dtype is bf16.
- **M-RoPE** (`get_rope_index` 925–1016, `get_vision_position_ids` 873–923). Runs of `mm_token_type_ids` are walked in
  order:
  - a text run takes positions p, p+1, … on all three axes;
  - an image run at running position p takes (t, h, w) = (p, p + hi, p + wi) over the *merged* grid in raster order,
    and p then advances by `max(h, w) / merge` (1008), not by the token count.

  For one 960×544 keyframe, the label sits at 0–5, vision start at 6, image tokens at t = 7, h = 7…23, w = 7…36, vision
  end at 37, and the prompt continues from 38. Rotary positions are therefore far below row indices: a 567-row
  presentation tops out near 86 (**arithmetic**; the oracle in step 1 records the exact ids).
- **Rotary recomposition** (`Qwen3VLTextRotaryEmbedding.recomposition_frequencies` 406–416). Frequency j < 64 takes
  the height position when `j % 3 == 1 and j < 60`, the width position when `j % 3 == 2 and j < 60`, and otherwise the
  time position. The cos/sin are computed in f32 and cast to bf16 (394–404), and the 128 columns are `cat(f, f)`.

## 2. Weights already on disk

**Nothing needs downloading.** The profile's conditioner binds byte ranges of `FL2VA/text_encoder` in the pinned
snapshot `42ed227e…`: `bundle.json` → `conditioner.embedding_table` and 50 × 11 layer parameters, all in that
directory's shards. The same directory holds the complete vision tower. Its `model.safetensors.index.json` lists 351
`model.visual.*` tensors (of 1,058), all in shard 14:

- `patch_embed`;
- `pos_embed` [2304, 1152];
- 27 `blocks`;
- `merger`;
- `deepstack_merger_list.0–2`, whose `linear_fc1` is [4608, 4608] and whose norm is [4608].

They total 1,190,533,600 bytes, all bf16, read from the shard header only. All 14 shards resolve locally.
`FL2VA/processor/` holds the preprocessor config. The profile today references none of the vision tensors.

## 3. What Kit needs

### Reusable today

- **`AetherVision`** (`Package.swift` 43). It is pure and Core-AI-independent, and `AetherLanguageCoreAI` already
  depends on it.
  - `Qwen3VLImagePreprocessor.patchify` (`Sources/AetherVision/Qwen3VLImagePreprocessor.swift` 129–158) reproduces
    the processor's normalization and merge-window row layout. Its defaults match H3's processor config (21–36).
    Canvas sides are multiples of 32, so H3 calls `patchify` directly, with no resize and no `imageGrids` snapping.
  - `Qwen3VLVisionTables` (`Sources/AetherVision/Qwen3VLVisionTables.swift` 12–62) builds the position-table gather
    and the patch (row, column) inputs.
- **The Forge vision module.** `Forge/language/qwen3_vl_vision.py` is config-driven: depth, hidden, heads, DeepStack
  indexes and `out_hidden_size` all come from `vision_config` (175–209).
  - It outputs `image_embeds` and `deepstack [3, n/4, d]`.
  - It takes host tables as inputs: `pos_index`, `pos_weight` and `patch_rc` (84–90).
  - It uses one static-shape function per patch grid, because a dynamic export computed wrong outputs on the GPU
    (1–22).
  - `load_weights` (236–260) reads `model.visual.*` from any checkpoint directory.
  - What doesn't carry over: its recipe is fp16 storage with f32 compute and fp16 pixels (18–21, 215, 86 of
    `export_qwen3_vl_vision.py`), and it bakes its weights into the asset. Upstream H3 runs the tower in bf16 with bf16
    pixels.
- **M-RoPE.** The rule is the same as `LanguageVisionPrompt.positions`
  (`Sources/AetherLanguageCoreAI/LanguageVisionPrompt.swift` 107–131) and `mrope_axes`/`mrope_positions`
  (`Forge/language/qwen3_vl_text.py` 40–77). The Swift version is `internal` to `AetherLanguageCoreAI`, which
  `AetherH3CoreAI` must not depend on (`Package.swift` 52, 72).

### Changes

1. **Vision tower graph, new.** Add an H3 stage, `export-h3-profile.py --stage conditioner-vision`, built on
   `qwen3_vl_vision.py` with a precision parameter. It writes static functions for the admitted canvases: grid
   (34, 60) at 960×544 and (60, 34) at 544×960, plus (18, 18) for the 288×288 diagnostic canvas.
   - Weights are bound in place from shard 14, as the conditioner layers already are
     (`Scripts/export-h3-profile.py` 400–470), rather than baked in.
   - It writes an overlay bundle with `lineage`, the pattern of the tiled encoder overlay
     ([h3-integration](h3-integration.md#tiled-first-frame-encoder)).
   - The CPU self-check compares against upstream's own `Qwen3VLVisionModel`, not against Kit's module.
2. **Per-layer conditioner graph: no change.**
   - The graph already takes cos/sin per row (`[1,T,128]` bf16 inputs; `H3ProfileStages.swift` 10–13; export at
     `Scripts/export-h3-profile.py` 420–461).
   - M-RoPE tables come from the existing 1-D table: row r, column j is
     `table1D[position_axis(j mod 64)(r)][j]` from `conditioner/rope-{cos,sin}.raw` [4096, 128]
     (`export-h3-profile.py` 472–486). Upstream's recomposition only *selects* per-frequency values from per-axis
     tables computed by the same f32 multiply and bf16 cast, so the gather is bit-exact by construction. No new
     floating-point code lands in Swift, and positions stay below 4,096.
   - DeepStack is applied on the host. The conditioner already returns every layer's hidden state to the host
     (`H3ProfileStages.swift` 101–108). After layers 0, 1 and 2, Swift adds level k to the image rows with an f32 add
     and a round-to-nearest-even back to bf16, which is the expected result of torch's CPU bf16 add. The oracle checks
     it.
   - The image embeddings overwrite the embedding rows of the pad tokens before layer 0.
3. **Presentation in Swift.**
   - A pure `H3Presentation` in `AetherH3` builds ids, tags and spans for N keyframes from the tokenizer and grids, plus
     the 3 × T M-RoPE positions. It is roughly 30 lines, a small port of the rule above, with CPU fixtures produced by
     the oracle.
   - `AetherH3CoreAI` gains a vision kernel (protocol plus native implementation, faked in CPU tests) and a dependency
     on `AetherVision`.
   - `H3ProfileConditioner.encode` (`H3ProfileStages.swift` 89–110) builds the presentation, embeds, scatters, gathers
     cos/sin and streams the layers with DeepStack.
4. **Tags.** `H3Layout` already accepts tags 0/1 and up to 4,096 text rows (`Sources/AetherH3/H3Layout.swift` 26–27,
   73). Every caller hard-codes ones today:
   - `H3ProfileStages.swift` 221;
   - `Sources/AetherProbe/H3ProfileCommand.swift` 189 and 331;
   - `H3DenoisingPlan.swift` 43 and `H3PhaseProbePlan.swift` 66 (2026-10-18 probe paths).

   The tags must flow from the conditioner to the denoiser. That is a public API change:
   - `H3PromptConditioner.encode(_:geometry:onLayer:)` (`H3VideoSession.swift` 83–87) gains the keyframes and returns
     tags with the tap;
   - `H3Denoiser.denoise` takes the tags.

   The session already has the keyframe RGB8 before the conditioner runs (`H3VideoSession.swift` 218–233). Upstream,
   a tag changes the AdaLN row a token reads (`adaln_indices = timestep_indices * 3 + token_tags`,
   `models/transformers/transformer_minimax_h3.py` 647). Text length also shifts the whole media clock
   (`before_denoise.py` 320–323), so every media row's rotary time moves by the added rows.
5. **Text token limits.** `profile.textTokenLimits` (2…4,096) bounds the *presentation* length T. The conditioner,
   refine and pack graphs all declare `tokens` 2…4,096 (`export-h3-profile.py` 250, 386, 454). The prompt budget
   becomes 4,096 − Σ(N_i + 8): 3,578 tokens with one keyframe at 960×544, and 3,060 with two (**arithmetic**). The
   error should name both counts.
6. **Rows and admission.**
   - With the gate prompt (49 tokens), the FL2VA canvas at 960×544×124 goes from 19,843 rows
     (`docs/evidence/h3-profile-trajectory-fl2va-2026-10-26.json`, `canvas.rows`) to 20,361 (+518, +2.6 %;
     **arithmetic**).
   - The bundled measured entry (`Sources/AetherH3/Resources/h3-measured-resources.json`) holds a denoising peak of
     17,307,844,608 B and operation bytes of 201,326,592 B. With the 256 MiB margin, the worst stage needs
     17,777,606,656 B, leaving 1,549,746,176 B (1.44 GiB) under the 18 GiB bar (**arithmetic** from those numbers).
   - Whether +518 rows fits has to be re-measured.
   - The `encoding` stage, at 2,162,936,400 B, gains the vision tower's weights (1.19 GB) unless the tower is released
     before the layers start. Metal reclaim lags 1–6 s after release ([h3-integration](h3-integration.md#diagnosis)),
     so the two can overlap.
   - Admission is keyed by pixels and was measured with the 49-token prompt. That gap predates this work, but this
     change widens every FL2VA request by a fixed amount.
7. **First plus last.** The presentation code should be N-keyframe generic: labels are numbered 1… in packed order,
   first then last, and a two-keyframe batch is still one tower call per image. But Kit refuses last-frame requests
   (`H3VideoSession.swift` 169), and the denoiser lays out only a `.first` anchor (`H3ProfileStages.swift` 218–221).
   Enabling last-frame conditioning also needs:
   - the `.last` anchor time (`before_denoise.py` 325–342);
   - a last-frame condition encode.

   That is separate work. Note also that the deviation section says last-frame and first-plus-last conditioning
   "follow the same text-only presentation", but both are refused today. That sentence should be corrected.

## 4. Gates

### An oracle that cannot share Kit's input

Today's text gate runs upstream's model, but on Kit's ids (`Scripts/prepare-h3-profile-oracle.py` 112–178). It
recreates `create_mm_token_type_ids` (157–158) because the pinned environment has no torchvision, which the
Qwen3-VL processor's video processor needs. A probe in that environment confirmed `AutoProcessor.from_pretrained`
fails there with an `ImportError` for torchvision.

The new `--part presentation` takes **only** the prompt text file, the keyframe PNG (or PNGs) and the canvas. It runs
upstream's `MiniMaxH3FL2VATextEncoderStep.__call__` verbatim, so tokenization, presentation, tags, the processor,
M-RoPE and DeepStack are all upstream code:

- **Components.** A real `Qwen3VLProcessor` and the FL2VA tokenizer.
- **Conditioner.** A `Qwen3VLForConditionalGeneration` built on `meta`:
  - the vision tower loaded fully, 1.19 GB;
  - `embed_tokens` loaded;
  - decoder layers streamed in place with the existing pre/post hooks;
  - the forward stopped after layer 49.
- **Recorded by hooks:**
  - `token_ids` and `text_token_tags`;
  - `pixel_values` and `image_grid_thw`;
  - the position ids from `model.get_rope_index`;
  - merger and DeepStack outputs;
  - the tap.

Kit's dump is compared against these and is never fed in. The checks:

- ids, tags, the patchified pixels and the 3 × T positions must be exactly equal;
- the vision embeddings, DeepStack levels and tap are compared by relative L2 against the existing predefined limit of
  0.02, over all rows and over prompt rows separately;
- a second run with the keyframe through `MiniMaxH3ResizeStep` checks that passing a canvas-sized frame through is the
  identity.

### Cost

- **Existing evidence.** The text oracle at 49 tokens took 46.7 s at a sampled peak group RSS of 1,601,241,088 B
  (`docs/evidence/h3-profile-graphs-960-2026-10-26.json`, `controllers.oracle-text`). The tap there compared at cosine
  0.9999980 and relative L2 0.0085.
- **Memory.** Expect roughly 1.6 GB plus the 1.19 GB tower plus small activations, so about 3–4 GB (**unverified**).
  Streaming keeps it far below the 20 GiB controller limit (`rssLimitBytes` in the same record).
- **Time.** The work grows with T, 567 rows against 49. Upstream runs bf16 on the CPU, and the 46.7 s baseline is
  already slow for 49 tokens, so this could take several minutes (**unverified**). Measure a two-layer slice first.
- **FL2VA trajectory gate, for scale.**
  - The CPU oracle took 3,457.5 s at 10,819,092,480 B sampled.
  - The GPU capture took 408.7 s and the replay 1,025.2 s (`docs/evidence/h3-profile-trajectory-fl2va-2026-10-26.json`,
    `controllers`).

### Evidence the fix resets

**FL2VA** (the conditioning and the row count change):

- `h3-profile-trajectory-fl2va-2026-10-26.json`;
- the FL2VA renders in `h3-memory-gate-2026-10-28.json` and `h3-memory-gate-review-2026-10-28.json`, and so the
  bundled measured entry;
- the FL2VA renders in `h3-media-audit-2026-10-28.json`;
- the FL2VA timings in `h3-quiet-timing-2026-10-28.json` (342.8 s, conditioner 19.0 s);
- the FL2VA parts of `h3-session-lifecycle-2026-10-28.json`, `h3-turbo4-and-decode-tiling-2026-10-28.json` and
  `h3-second-specialization-2026-10-27.json`, where they render FL2VA.

**Still valid:**

- The keyframe encoder gates (`h3-encoder-tiled-export-2026-10-26.json`, and the FL2VA graph gate in
  `h3-profile-graphs-960-tiled-2026-10-26.json`, which checks only the encoder). The encoder graph and its inputs don't
  change.

**T2VA:**

- T2VA computation doesn't change, but the overlay changes the profile identity. The measured entry is keyed to the
  identity, so T2VA needs an identity replay of its 457 boundary hashes, as in `h3-profile-home-2026-10-27.json`.

## 5. Steps

All GPU work runs under the GPU lock with the existing controllers. Scratch files go to `/Volumes/Scratch` or
`~/.cache`.

| # | Step | CPU | GPU | Memory |
| --- | --- | --- | --- | --- |
| 0 | Add torchvision to `Forge/h3-oracle` (D4). Re-verify that the FL2VA keyframe hash still reproduces (`docs/evidence/forge-environment-2026-10-27.json`). | minutes | none | small |
| 1 | Oracle `--part presentation`. First a two-layer slice to time it and take peak RSS, then full runs at 960×544 and 288×288. Write fixtures (ids, tags, positions) for the Swift tests. | slice minutes; full run several minutes (**unverified**) | none | ~3–4 GB (**unverified**; measure on the slice) |
| 2 | `--stage conditioner-vision` overlay: static grids, in-place weights, CPU self-check against upstream's `Qwen3VLVisionModel` per grid. | minutes (**unverified**) | none | a few GB (**unverified**) |
| 3 | Swift: `H3Presentation`, profile schema fields, vision kernel, conditioner changes, tag plumbing, limits, probe dump of presentation inputs, CPU tests on oracle fixtures. | build and tests | none | small |
| 4 | Native vision qualification per grid against step 1 (specialized without `expectFrequentReshapes`), recording the Metal peak. | small | seconds per grid | ~1.2 GB weights plus scratch (**unverified**) |
| 5 | Native conditioner gate: exact equality of ids, tags and positions, then vision embeddings, DeepStack and tap against step 1. | small | ~20–40 s per run (19.0 s today at 49 tokens; **unverified** at 567) | encoding stage, re-measured |
| 6 | FL2VA trajectory gate re-run, with the oracle's own tap and tags. | ~1 h oracle (3,457.5 s before) | ~409 s capture + ~1,025 s replay (before) | ~10.8 GB sampled oracle (before) |
| 7 | T2VA identity replay under the overlay identity (457 hashes). | small | one T2VA render | as qualified |
| 8 | Memory gate: T2VA and FL2VA in both orientations. Re-bundle the measured entry and confirm the 18 GiB bar. | small | 4 renders, ~6 min each | must stay under the bar |
| 9 | Media audit, plus an A/B of old and new FL2VA renders for the user's review. | small | 2+ renders | as step 8 |
| 10 | Docs: resolve the deviation section (and fix the last-frame sentence), roadmap, CHANGELOG (API change), evidence. | none | none | none |

### Risks

- **Numerics.** Upstream feeds bf16 pixels to a bf16 tower. A recipe matching upstream (D1) keeps that rounding. The
  language recipe (fp16 pixels, f32 compute) is more precise but not upstream. Either way the parity bar is relative
  L2, not bytes.
- **Graph and runtime.**
  - Static grids mean an unlisted canvas is refused for FL2VA. Today only the admitted canvases plus the diagnostic
    one would have grids.
  - Metal reclaim lag after the tower's release may overlap the first conditioner layers.
  - A bf16 vision graph on the GPU is new for Kit. The conditioner layers are bf16 already; the language tower is not.
- **Memory.** +518 rows may not fit the 1.44 GiB headroom once the step-0 transient is carried. Step 8 decides.
- **Gate and release order.**
  - The torchvision addition changes the pinned oracle environment, so step 0 has to re-verify it.
  - Atelier's FL2VA canvas measurements depend on the new row count
    ([roadmap](roadmap-0.0.4.md#parallel-track-hardening-and-coverage)), so they should wait for step 8.

## 6. Decisions for the user

1. **D1, tower precision.** Recommended: an upstream-matching bf16 recipe (bf16 weights and pixels, f32 RoPE). The
   alternative is the language product's fp16-storage, f32-compute recipe.
2. **D2, weight binding.** Recommended: bind in place from checkpoint shard 14, with no copy. The alternative is to
   bake 1.19 GB into the asset, as the language product does.
3. **D3, canvases.** Recommended: static grids for 960×544, 544×960 and 288×288. Adding 1280×704 (880 tokens) means
   measuring admission for it.
4. **D4, oracle environment.** Recommended: add torchvision to the pinned H3 oracle environment, so upstream's
   processor runs verbatim. The alternative keeps the pin with a PIL-backend stand-in, which leaves a small
   independence gap.
5. **D5, gate bars.** Recommended: exact equality for ids, tags, pixels and positions; relative L2 ≤ 0.02 for the
   vision embeddings, DeepStack and tap.
6. **D6, profile identity.** Recommended: an overlay bundle with `lineage`, with T2VA carried by an identity replay.
   The alternative is a full T2VA re-gate.
7. **D7, first plus last.** Recommended: an N-keyframe presentation, with last-frame requests still refused in this
   change. The alternative is to implement last-frame conditioning in 0.0.4.
8. **D8, API.** Accept the public change to `H3PromptConditioner` and `H3Denoiser` (keyframes in, tags out) in
   0.0.4.
9. **D9, admission.** Re-measure at the gate prompt only, or also at a long prompt. The second closes the existing
   text-length gap.
10. **D10, the effect on output.** Whether the user wants an A/B review of old and new FL2VA renders before the
    measured entry is re-bundled.
