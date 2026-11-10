# VLM: the Qwen3-VL driver

Status: **slices V1–V4 implemented, 2026-11-09; G4 passed on the 4B export** (fp16 tied head, release build: 8 cases
on 8 synthetic pictures, 7 token-exact and 1 near-tie flip, 18.8–20.9 ms per decode step; see §6). V5 (G5 retention and memory) and the FoundationModels attachments follow. The
plan below was revised 2026-11-09 for Qwen3-VL **4B** and several images per turn. It supersedes the 2B-only draft
of 2026-11-06 (one image per context, one rope shift). The driver serves any Qwen3-VL export of `Forge/qwen3vl/`
(2B, 4B, 8B and their fine-tunes): sizes come from the graph descriptors, never from the model name. The export half of
M4g ([M4.md](M4.md)) merged as PR #135 for 2B; this plan changes its decoder contract (§3) and adds the 4B export.

It's built from:
- the exporter: `Forge/qwen3vl/*` and its 2B outputs (`forge-outputs/qwen3-vl-2b/`);
- `Sources/AetherRuntime/Decoder/*`, `LanguageEngine.swift`, `Vision/ImagePreprocessor.swift`,
  `Sources/AetherText/{Generator,ChatSession,PromptRenderer}.swift`, `Sources/AetherFoundationModels`,
  `Sources/AetherCore/Manifest/*`;
- [MANIFEST.md](MANIFEST.md) §1.2, §2.3, §2.5–§2.6; [DESIGN.md](DESIGN.md) §6.2, §7, §8, §10;
  [INTEGRATION.md](INTEGRATION.md) "Lessons";
- 0.0.7's Qwen3-VL path (`VisionLanguageSession`, `LanguagePromptPlan`, `Qwen3VLImagePreprocessor`, the
  `language-model.json` describer export), which is the reference the driver is gated against: port its behaviour,
  not its code.

---

## 0. The short answer

- **Model source.** The exporter takes any local Qwen3-VL checkpoint pinned by a source-pins file (repo, revision,
  SHA-256 of every file it reads). The 4B gate export is **Qwen3-VL-4B-Instruct, abliterated**
  (`huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` at `ce72a7c2`), the same weights the 0.0.7 describer export
  was made from, read from the local Hugging Face cache: a local Forge input, not a download.
- **Several images per turn.** Slots are packed: image k's tokens take extension ids `vocab + base_k + i`, its rows
  sit at `base_k …` in one static image buffer, and each image has its own entry in the rope-shift lists. A turn may
  carry up to `maxImages` images within `slots` image tokens (export parameters; 4B: 16 images, 4,096 slots).
- **The vision tower is grid-static:** one function, one capacity shape (2,048 patch rows), the grid an input, the
  padding masked in the graph. 0.0.x's dynamic-shape tower was wrong about 2.5 % of the time (INTEGRATION.md), and
  0.0.7's five fixed-grid functions forced every picture to one of five grids.
- **API:** the existing `ChatSession`/`Generator`; a user message carries images (`ChatMessage.user(text, images:)`),
  and a rendered prompt can carry them too (`Generator.generate(prompt:images:)`, 0.0.7's describer entry point).
- **`LanguageEngine` doesn't change.** `ImageConditionedEngine: LanguageEngine` adds image binding;
  `DecoderEngine` conforms when it loads `vlm-static-inject@1`.
- **Engine changes, generic:** `DecoderCore` owns `binding: static` inputs without a `source` (one buffer each, bound
  on every call) and accepts extension ids; `DecoderEngine` loads the tower and copies each encoded image's rows into
  the decoder's static buffers.
- **One GPU grant per reply,** covering every tower encode, the prefill and the decode. CPU preprocessing runs
  before the grant. 0.0.7 admitted each encode and the decode separately.
- **macOS first** (`variant.platform: macos`). 4B is about 9 GB of fp16.
- **The gate:** greedy output **token-exact** against the 0.0.7 describer on fixed synthetic pictures, single and
  multi-image, with INTEGRATION's near-tie rule as the only allowance.

## 1. What the export gives

Sizes are 4B's (2B in brackets where it differs). `N` = image slots, `M` = most images, `P` = tower patch rows.

| Graph | Contract | Notes |
|---|---|---|
| `vision` (`vision-tower@1`) | `patches` f32 [P=2048, 1536] + `grid` i32 [2] → `image_embeds` f16 [P/4, 2560 (2048)], `deepstack_embeds` f16 [3·P/4, 2560] (level-major) | One static shape. The real rows (h·w, block-major) come first, then zeros; padding rows are masked as keys; output rows past h·w/4 are zero. fp16 weights, fp32 compute, fp16 outputs (the decoder casts its embeddings to fp16 anyway, so nothing is lost). `expectFrequentReshapes: false` |
| `decoder` | `decoder-lm@1`'s running contract: `input_ids` [1, S≤512], `position_ids` [1, T], KV [36 (28), 1, 8, C, 128] fp16 in place, last-row logits [1, 1, 151936] f32 | Static inputs (fp16 / i32): `image_embeds` [N, 2560], `deepstack_embeds` [3N, 2560] (level k at rows k·N + slot), `image_positions` i32 [N, 3], `rope_shift_start`, `rope_shift_amount` i32 [M]. M-RoPE is computed in the graph (§3). `expectFrequentReshapes: true`, `maxContext` 32768, `prefillChunk` 512 |

What the host does for each image, in prompt order:
1. **Preprocess** (CPU). Smart resize (factor 32, 65,536 … P·256 = 524,288 pixels), torch uint8 bicubic,
   `(x/255 − 0.5)/0.5`, block-major `C,T,py,px` patches with T = 2, zero-padded to P rows; `grid` = (H/16, W/16).
2. **Plan the prompt.** The template renders one `<|vision_start|><|image_pad|><|vision_end|>` per image. Image k's
   single pad expands to `count_k = (H/32)(W/32)` pads, rewritten to `vocab + base_k + i` (`base_k` = the sum of the
   earlier counts). With `start_k` the position of its first pad and `shift_k = Σ_{j<k} amount_j`:
   - `s0_k = start_k − shift_k`, the text-rope position where the image begins;
   - `image_positions[base_k + i] = (s0_k, s0_k + i / gw_k, s0_k + i % gw_k)` (merged grid gh_k × gw_k);
   - `rope_shift_start[k] = start_k + count_k`, `rope_shift_amount[k] = count_k − max(gh_k, gw_k)`.
   That's upstream's `get_rope_index` and 0.0.7's `LanguagePromptPlan.positions`, which becomes the hermetic oracle.
   Unused entries: positions zero, start 2³⁰, amount 0. Text-only is all unused.
3. **Encode** each image whose (digest, slot base) isn't already bound, on the tower; copy its rows into the decoder's
   statics at `base_k` (and `k·N + base_k` per DeepStack level). Then write the positions and shift lists.
4. **Prefill and decode.** The statics are bound on every call, so prefill chunks need no per-chunk image rows.

## 2. The API

```swift
let session = try await ChatSession.load(bundle: bundle, manifest: manifest, device: .current)
for try await event in session.respond(to: "What changes between these frames?", images: [a, b, c]) { … }

// Generator level, as the FoundationModels executor and a describer use it:
generator.generate([.system(instructions), .user("Describe the picture.", images: [ChatImage(input)])], parameters: …)
generator.generate(prompt: rendered, images: [ChatImage(input)], parameters: …)  // one pad per image, in order
```

- **`ChatMessage.images: [ChatImage]`** (user turns). `ChatImage` is a `Sendable`, `Hashable` reference box over an
  `ImageInput` that caches its prepared form, so a history turn isn't decoded again on every reply. A message with
  images renders as content parts: its images, then `{type: "text", text}`. That's 0.0.7's order (and the order of
  every reference prompt); Qwen's template iterates over the parts. Text-only messages render as today.
- **`ChatSession.respond(to:images:)`**, with images kept in the history. `ChatSession.load` dispatches on the
  driver: `decoder-lm@1` or `vlm-static-inject@1`; `Registry.text` gains `VLMStaticInjectDriver`. Every turn renders
  the whole history, so the history's images count against one context's limits: once a conversation holds
  `maxImages` images or `slots` image tokens (16 and 4,096 for the 4B export), every later turn, text-only included,
  is refused (`generation(.invalidParameters)`) until `clearHistory()`. There is no eviction of old images.
- **The Generator stays the single loop.** A request with images on an engine that isn't an
  `ImageConditionedEngine` is refused (`unsupportedOperation`: "this model takes no images").
- **FoundationModels** (later slice, V4): `Attachment<ImageAttachmentContent>` maps to `ChatImage`, `.vision` is
  advertised for an `ImageConditionedEngine`; `.data` attachments stay refused.

**Rejected:** a separate `VisionLanguageSession` type (0.0.x's shape). It duplicates the history, prefix reuse,
constraints, tools and the bridge. The behaviour 0.0.7's session had is kept (§7).

## 3. Engine and graph changes

### 3.1 The decoder's M-RoPE, generalized to many images

Per fed token at running position p (`position_ids`), in the graph:
- an extension id `vocab + g` embeds as `image_embeds[g]`, gets DeepStack `deepstack_embeds[k·N + g]` after decoder
  layer k < 3, and takes the rotary positions `image_positions[g]`;
- a text id embeds as its vocabulary row and takes `p − Σ_j [p ≥ rope_shift_start[j]] · rope_shift_amount[j]` on
  all three axes.

One image is the 2B contract's special case. The positions of a text token before image k depend only on images
before k, so a KV prefix stays valid when a later image changes (the reuse rule, §3.4).

### 3.2 `ImageConditionedEngine`

```swift
public protocol ImageConditionedEngine: LanguageEngine {
    /// Token ids, slots, most images, the preprocess configuration.
    nonisolated var imageInput: ImageInputSpec { get }
    /// CPU only: decode, smart resize, patchify, grid, count, content digest.
    nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedPromptImage
    /// Binds a plan's images: encodes each one not already bound at its slots, then writes the positions and shift
    /// lists; an empty plan restores the text-only values. Refused during an operation.
    func bindImages(_ plan: ImagePromptPlan) async throws(AetherError)
    /// What the statics hold now, image by image.
    var boundImages: [BoundImage] { get }
}
```

`ImagePromptPlan` (pure, AetherRuntime) expands the pads, checks each pad sits between open and close and that pads
match images, and computes the statics. `LanguageEngineInvariants` runs unchanged on text-only prompts.

### 3.3 `DecoderCore` and `DecoderEngine`

- **Static inputs.** Roles with `binding: static` and no `source` get one arena buffer at the descriptor's shape
  (which must be fully static), bound whole on every call (prefill chunks, host steps, encoded greedy steps). Views
  get copied (INTEGRATION), so never a view. Writes only when idle. The text-only values are written at load and by
  `reset()`; a rewind keeps them (stale image rows are never read by the kept tokens).
- **Extension ids.** Token validation accepts `[base, base + count)` from `inputEncodings.tokens.extensionIds`.
  Greedy argmax runs over the head's rows, so it never yields one.
- **The tower** loads as a second `Graph` (`compute: gpu`, no reshapes), checked against `vision-tower@1`. The
  engine owns one patches buffer, one grid buffer and the tower's two output buffers; `bindImages` runs the tower per
  new image (about 0.2 s on 2B) and copies its rows into the statics with `memcpy` on shared memory (16–20 MiB per
  image: well under a millisecond). Binding the tower's outputs straight into the decoder's statics would need views
  at a row offset per image, and the DeepStack levels aren't contiguous per image.
- `KVSizing.modelBytes` counts both graphs; `fixedBytes` adds the statics and the tower's buffers.

### 3.4 The Generator: image-aware prefix reuse

1. **Before the grant:** collect the images, refuse more than `maxImages` or `slots`, `prepare` each (cached in its
   `ChatImage`).
2. **Inside the grant:** render, plan, then **cap the reused prefix at the first image whose (digest, slot base,
   start) differs from the bound one**. Two images with the same grid render identical ids (`vocab + slot`), so
   without the cap a new picture of the same size would reuse the old picture's KV. Rewind, `bindImages`, prefill.
3. **Sampler and constraints** see `padId` (151655) for every extension id, as Transformers' `input_ids` have them:
   the repetition penalty would otherwise index past the logits row.

Multi-turn: history turns re-render the same images at the same positions and slots, so a turn costs only its new
tokens and no encode. An image added in a later turn takes the next slots.

## 4. Manifest typing (slice V1)

| Field | Typed as |
|---|---|
| `preprocess.image.resize.mode` | `smartResize`, with `factor`, `minPixels`, `maxPixels` on `resize`. `maxPixels` must equal the tower's rows · patch², checked at load. The ties-to-even rule below |
| `graphs.<g>.kind` | `Manifest.Graph.kind: DriverID?`, which must be in `DriverCatalog.graphKinds`; `vlm-static-inject@1` requires `vision-tower@1` on `vision` |
| `image.tokens` | `{pad, padId, open?, openId?, close?, closeId?, idBase}`; at load `idBase == extensionIds.base` and the tokenizer maps `pad` to `padId` |
| `image.slots`, `image.maxImages` | the decoder's static rows N and shift entries M, checked against the descriptors |
| `image.rope` | normative for Qwen3-VL's rule (§3.1): `{rule: "qwen3vl-mrope", textOnly: {start: 1073741824, amount: 0}}` |
| `image.deepstack` | `{levels: 3, layout: levelMajor}` |
| `vlm-static-inject@1` contract | `vision` (kind-checked: `patches` + `grid` in, `imageEmbeds` out, `deepstackEmbeds` optional) and `decoder` = `decoder-lm@1`'s plus static `imageEmbeds` (required), `deepstackEmbeds`, `imagePositions`, `ropeShiftStart`, `ropeShiftAmount`; requires `image`, `preprocess.image` (`layout: patches`) and the tokenizer |

**Smart resize** (upstream's; 0.0.x's port is correct): sides round to multiples of the factor **ties to even**
(`.toNearestOrEven`: 80/32 = 2.5 rounds to 2); above `maxPixels` divide by β = √(HW/max) and floor (at least the
factor); below `minPixels` multiply by β = √(min/HW) and ceil; an aspect ratio over 200 is refused.

## 5. Memory and the GPU lock

| Item | 4B | 2B |
|---|---|---|
| Decoder weights (fp16) | ≈ 8.0 GB | 3.44 GB |
| Tower weights (fp16) | ≈ 0.8 GB | 0.81 GB |
| KV per token (fp16) | **147,456 B** (36·2·8·128·2): 144 MiB per 1k tokens, 576 MiB at 4,096 | 114,688 B |
| Decoder statics (fp16) | 80 MiB at N = 4,096 | 32 MiB at N = 2,048 |
| Tower buffers | 12 MiB patches + 10 MiB outputs | the same + 8 MiB |
| Bundle on disk | 8.89 GB (8.28 GiB) | |
| Metal (G4 run, release) | fp16 head (the default): 9.4 GiB after load, 10.3 GiB after the first picture, 12.9 GiB after a three-picture turn, 13.6 GiB at the end; fp32 head: 15.2, 17.3 and 20.5 GiB (process-wide `currentAllocatedSize`, capacity 4,096). 0.0.x's 4B VLM: 13.8 GiB at its first image request (0.0.7's docs/qwen3-vl.md). G5 breaks it down by phase | |

Both graphs stay resident for the engine's lifetime on macOS. A reply takes one `GPUAdmission` grant (DESIGN §10);
`bindImages` runs inside `Generator.prepare`, so the encodes are inside it by construction.

**Load-time accounting (V5).** The KV capacity is derived from the memory budget less the model's charge
(`KVSizing.modelBytes`). Loading allocates almost nothing (the memory arrives with the first run), so the Metal delta
across load is near zero and the charge used to be the graph files' size: 8.28 GiB for the 4B export, against
9.4–13.6 GiB of Metal measured. The engine now measures and accepts a declared figure, for `decoder-lm@1` and
`vlm-static-inject@1` alike:
- `DecoderEngine.memory` (`DecoderEngine.Memory`) reads the probe (`GPUMemoryProbe.system`, Metal included) at the
  call: the level before load, now and at its highest (read after load and the canary, and after each prefill, step,
  greedy decode and image bind), the engine's own buffers (KV, logits, statics, the tower's buffers), the charge, and
  `measuredModelBytes`: the most the level stood above the baseline less the engine's buffers at that read.
- `DecoderEngine.Options.modelBytes` declares the model's memory (an earlier session's `measuredModelBytes`, or a
  gate's). Load charges the larger of it and what load measured; a declared figure the budget can't hold beside the KV
  refuses the load with `capacity`.
- The probe reads the whole process, so memory something else in the process allocates while the engine is loaded is
  counted too: an overestimate, the direction admission prefers.
- **Qualified figures.** The kit ships measured model bytes for known exports, `DecoderEngine.qualifiedModelBytes`,
  keyed by the export's `Manifest.contentDigest` and the chip's Core AI architecture (`DecoderEngine.QualifiedExport`),
  the way MiniMax H3 scopes its `qualifiedFootprints` to `qualifiedExportDigest` and `qualifiedArchitecture`. A
  bundle load with `Options.modelBytes` nil declares the entry for its export and chip, so the first load already
  charges the measured memory and a host doesn't have to keep `measuredModelBytes` and pass it back; an explicit
  `modelBytes` wins (0 opts out). Another export (a requantization, a manifest edit, the other platform's variant) or
  another chip has no entry and charges as before. `DecoderEngine.declaredModelBytes` reports what load was given.
  The figures live in the kit rather than in the manifest because they are measured on an export that already exists:
  writing them into its manifest would change the digest they are keyed by. Entries come from the native gates, which
  print each one ready to add (`GATE qualified model bytes entry:`): G5 (`DescriberRetentionGateTests`) for the 4B
  Qwen3-VL export and the Qwen3 export gate (`DecoderLMExportGateTests`, teacher-forced test) for the `decoder-lm@1`
  one; once an entry exists, its gate checks that it still covers the measurement. Both entries are pending their
  gate runs.

**Retention.** 0.0.x measured retention with `expectFrequentReshapes` (257 MiB per text request on its 4B). G7
measures the new engine: Metal by phase over 20 mixed requests (`DescriberRetentionGateTests`, reading
`DecoderEngine.memory` after each); if it grows past the bar, slice V6 adds 0.0.x's reload budget at idle.

**Risk: statics bound every step.** 80 MiB of fp16 bound per decode step. If Core AI copies them per call, decode
slows; G7 records `GreedyTimings.bindings` and tok/s. The fix would be an export change (fewer slots, or statics as
states), not a runtime one.

## 6. Gates

| | Gate | Bar |
|---|---|---|
| G1 | **Prompt plan** (hermetic): 0.0.7's `positions` as the oracle over single, multi and text-only prompts; the four Qwen3-VL image cases of the 0.0.x chat oracle (`chat-oracle.json`) render id-exact through content parts | exact |
| G2 | **Engine on fake graphs** (hermetic): a fake tower and a fake decoder that reads its statics: chunked prefill (the image straddling chunks) equals one-shot; pipelined greedy equals host steps; a second turn on the same images reuses past them with no encode; a same-grid different image reuses nothing past its start; text-only after images equals a fresh text-only run | exact |
| G3 | **Export check** (Python, GPU): every graph against CPU float32 transformers on synthetic cases, single and multi-image | embeddings cosine ≥ 0.999; greedy 48/48 or the near-tie rule |
| **G4** | **The describer gate:** our 4B export through `Generator` vs the 0.0.7 describer (`VisionLanguageSession` on its 0.0.x export of the same weights), greedy, on fixed synthetic non-graphic pictures sized to 0.0.7's grids (704×704, 576×768, 768×576, 512×896, 896×512: the smart resize is the identity there, so both see the same pixels) and on multi-image turns | rendered ids exact; **every generated token equal**, except that at a divergence the 0.0.7 reference's top-1 − top-2 probability is < 0.1 and our token is in its top five (the reference's own teacher-forced logits); such a flip is reported, and the case ends there |
| G5 | **Retention and speed** | Metal by phase over 20 requests mixing grids, image counts and text-only: growth after warm-up < 1 GiB; decode ms/token and time to first token recorded against 0.0.7's |

0.0.7 fed its tower fp16 pixels and its decoder host-computed positions; ours feeds f32 patches and synthesizes the
positions in the graph. The pictures are chosen so the preprocessing is the identity in both, which leaves only the
pixel precision and the graphs' arithmetic to differ.

**G4 result (2026-11-09, 4B abliterated export, default 4,096 slots and 16 images, context capacity 4,096, release
build).** The reference is `Tools/DescriberReference007`, built from a local checkout of the 0.0.7 line, on the 0.0.x
export of the same weights. All 8 pictures' patch rows equal 0.0.7's at fp16. Every case renders 0.0.7's prompt ids.
Greedy replies: 7 of 8 cases token-exact (single pictures, a three-frame turn, text-only); the two-picture case flips at
token 59, where the reference's own top-1 − top-2 margin is 0.0000 and our token is in its top five. The same holds for
both head precisions:

| Per case | fp16 tied head (default) | fp32 tied head | 0.0.7 |
|---|---|---|---|
| decode, ms/token (cases past 20 tokens) | 18.8–20.9 | 25.5–28.8 | 20.3–22.7 |
| decode, text-only | 18.8 | 25.5 | 20.3 |
| prefill, one picture (s) | 0.24–0.34 | 0.24–0.41 | 0.26–1.30 (time to first token) |
| prefill, two / three pictures (s) | 0.50 / 0.78 | 0.52 / 0.84 | 0.52 / 0.82 |
| prefill, text-only (s) | 0.025 | 0.034 | 0.45 |
| Metal after load | 9.4 GiB | 15.2 GiB | |

Decode binds the 80 MiB of image statics on every step. With the fp16 head it's already at or below 0.0.7's per-step
cost, so the default slots stay. The fp32 head's extra cost is its float32 copy of the embedding (4·vocab·H bytes) and
the bandwidth to read it every step; the exporter's default is now fp16, as in 0.0.7's recipe (`--head-compute`).

## 7. From 0.0.7: port, and what changes

| 0.0.7 | Here |
|---|---|
| `VisionLanguageSession.generate(renderedTokenIDs:images:maximumTokens:)`: the caller renders one placeholder per image, images in order, greedy | `Generator.generate(prompt:images:parameters:)`, the same contract; and the messages API |
| Several images per request, 0.0.7's limit the image chunk and the context | `maxImages` and `slots` from the export, checked before the GPU grant |
| `LanguagePromptPlan.expand` (pad checks) and `positions` (M-RoPE) | `ImagePromptPlan`; `positions` becomes the hermetic oracle |
| Grid snapping to five fixed grids (aspect stretched up to 15 %) | **Dropped:** the grid-static tower takes any smart-resized size up to P patches |
| Aligned image rows per prefill chunk | **Dropped:** statics are indexed by slot |
| Separate GPU admissions for open, each encode and the request; `busy` on overlap | **Replaced:** one grant per reply; requests queue |
| `unload()` withdraws a waiting request | the Generator's cancellation and `SerialGate` |
| Reuse: none across requests | reuse with the image-identity cap (§3.4) |

## 8. Slices

| Slice | Contents | Gates |
|---|---|---|
| **V1** Manifest + export | §4 types; `DriverContract.vlmStaticInject`; the exporter's generalized decoder (§3.1), `--source-pins` for a local checkpoint, the 4B export and its check | G3 |
| **V2** Preprocess + plan | smart resize in `ImagePreprocessor` (padded to the descriptor's rows, `grid`, digest), `ImagePromptPlan` | G1 |
| **V3** Engine | `DecoderCore` statics and extension ids; the tower in `DecoderEngine`; `ImageConditionedEngine`; fake-graph tests | G2 |
| **V4** Text stack | `ChatMessage.images`, content-part rendering, Generator image reuse and the sampler fix, `ChatSession.respond(to:images:)`, `generate(prompt:images:)`, the driver in `Registry.text`; later the FoundationModels attachments | G1, G4 |
| **V5** Memory | G5, `KVSizing` with the tower; the measured and declared model charge, `DecoderEngine.memory` and the qualified figures (§5, built; the G5 run and the entry it prints are pending); docs | G5 |

V1–V4 are one PR for 0.2.0 (the describer is its blocker); V5 and the FoundationModels attachments follow.

## 9. Open questions for the owner

1. **Slots and images:** 4,096 slots and 16 images for the 4B export (80 MiB of fp16 statics bound per step), or
   fewer (2,048 slots: 4 full-size pictures or 8 frames at 448 tokens)? It's an export parameter. **Decided for now:**
   keep 4,096 and 16; G4 measured 18.8–20.9 ms per decode step at those defaults with the fp16 head (§6).
2. **Picture size:** 0.0.7 snapped every picture to one of five grids around 450–490 tokens. The new driver keeps the
   picture's aspect, up to 512 tokens a picture. A consumer that wants 0.0.7's sizes can resize first; should the kit
   offer 0.0.7's snapping as an opt-in preprocess mode? **Decided:** no opt-in.
3. **G4's bar:** strict token equality with the near-tie rule (as planned), or also every-row logits cosine ≥ 0.995
   against 0.0.7? **Decided:** token equality with the near-tie rule, as planned.
4. **Hosting:** the 4B abliterated export stays local (no catalog entry), as the 0.0.x describer did? **Decided:**
   it stays local.
