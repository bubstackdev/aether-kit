# VLM: the Qwen3-VL 2B driver (plan)

Status: **plan**, 2026-11-06. This is the first feature after 0.1.0, and the driver half of M4g
([M4.md](M4.md)). The export half merged as PR #135 (`Forge/qwen3vl/`). Nothing below is built yet. The plan covers
the API, the engine and manifest changes, memory, the gates, the PR slices and the questions for the owner.

It's built from:
- the export: `Forge/qwen3vl/*` and its outputs in `forge-outputs/qwen3-vl-2b/` (`bundle/.aether/aether.json`,
  `check-report.json`, `reference/`);
- `Sources/AetherRuntime/Decoder/{DecoderCore,DecoderEngine,DecoderPlanning}.swift`, `LanguageEngine.swift`,
  `Sources/AetherText/{Generator,ChatSession,PromptRenderer}.swift`, `Sources/AetherFoundationModels`,
  `Sources/AetherRuntime/Vision/ImagePreprocessor.swift`, `Sources/AetherHost/Pipeline/StatelessPipeline.swift`,
  `Sources/AetherCore/Manifest/*`;
- [MANIFEST.md](MANIFEST.md) §1.2, §2.3, §2.5–§2.6; [DESIGN.md](DESIGN.md) §6.2, §7, §8 and §10;
  [BACKENDS.md](BACKENDS.md) §1; [INTEGRATION.md](INTEGRATION.md) "Lessons";
- the FoundationModels interface in the macOS 27.2 SDK;
- 0.0.x's Qwen3-VL path in `_reference/aether-kit-remote` (`docs/qwen3-vl*.md`, `VisionLanguageSession`,
  `LanguageVisionPrompt`, `Qwen3VLImagePreprocessor`), which is trusted: port it, and review it.

---

## 0. The short answer

- **API:** use the existing `ChatSession` and `Generator`, and let a user message carry images
  (`ChatMessage.user(text, images:)`). There's no separate `VisionChat` type. FoundationModels 27 has image
  attachments and a `.vision` capability, so the bridge can take images too.
- **`LanguageEngine` doesn't change.** A refinement protocol, `ImageConditionedEngine: LanguageEngine`, adds image
  binding. `DecoderEngine` conforms to it when its manifest is `vlm-static-inject@1`. Text engines, MLX engines and the
  invariants suite don't see it.
- **Engine changes, generic and small:**
  - `DecoderCore` owns the static inputs (`binding: static`). It allocates one buffer per static input, binds it on
    every call, and resets them to the text-only values on a rewind to or before the image.
  - It accepts extension ids (`inputEncodings.tokens.extensionIds`).
  - `DecoderEngine` loads the tower, which writes straight into the decoder's static buffers: one arena, one actor,
    no copy.
- **Manifest:** four additions.
  - `x-smartResize` becomes a normative `smartResize` resize mode. Today the bundle's manifest doesn't decode at all,
    because `Mode` rejects `"x-smartResize"` and the `x-` pass drops its parameters.
  - `graphs.<g>.kind` gets a type.
  - The `vision-tower@1` contract gains a `grid` role.
  - The `image` block gets types, and its expression grammar gains `*`.
- **One GPU grant per reply,** covering the tower encode, the prefill and the decode. CPU preprocessing runs before
  the grant. 0.0.x admitted the encode and the decode separately.
- **macOS first.** The bundle is 4.27 GB of fp16. In 0.0.x, the VL-2B aborted while specializing on an iPhone, and
  it reached 8.7 GiB of Metal on the Mac. iOS needs a quantized decoder export and its own slice.
- **Gates:**
  - Swift parity on all five reference cases: 48/48 greedy tokens, and embedding cosine ≥ 0.999.
  - Engine invariants on an image prompt.
  - A Metal retention gate. The Python harness had to reload the decoder every 6 calls, and 0.0.x saw retention even
    with `expectFrequentReshapes`. The claim that the Swift engine "is not affected" is checked, not assumed.
- **Five PR slices, about 8–10 days.** Then M4h (OCR).

## 1. What the export gives

| Graph | File | Contract | Notes |
|---|---|---|---|
| `vision` (`vision-tower@1`) | 815 MB, fp16 weights, fp32 compute | `patches` f32 [2048, 1536] + `grid` i32 [2] → `image_embeds` f32 [512, 2048], `deepstack_embeds` f32 [1536, 2048] (level-major: level k at rows k·512 + slot) | One static shape for every image. The real rows (h·w) come first, then zeros. Padding rows are masked in the graph, and the output rows past h·w/4 are zero. `expectFrequentReshapes: false` |
| `decoder` | 3.44 GB fp16 | the `decoder-lm@1` running contract: `input_ids` [1,S≤512], `position_ids` [1,T], KV [28,1,8,C,128] fp16 in place, last-row logits [1,1,151936] f32 | Five static inputs: `image_embeds`, `deepstack_embeds`, `image_rc` i32 [512,2], `rope_shift_start` and `rope_shift_amount` i32 [1]. M-RoPE is synthesized in the graph from (id, position). `expectFrequentReshapes: true`, `maxContext` 32768, `prefillChunk` 512 |

What the host does for one image:
1. **Preprocess.** Smart resize (factor 32, 65,536–524,288 pixels), then torch uint8 bicubic, then
   `(x/255 − 0.5)/0.5`, then block-major `C,T,py,px` patches with T=2, zero-padded to 2,048 rows. `grid` = (H/16,
   W/16).
2. **Plan the prompt.** The template renders one `<|vision_start|><|image_pad|><|vision_end|>`.
   - Expand the pad to count = (H/32)·(W/32) pads (at most 512), and rewrite pad k to `151936 + k`.
   - `imageStart` is the first pad's position.
   - `rope_shift_start` = imageStart + count, and `rope_shift_amount` = count − max(gridH, gridW).
   - `image_rc[k]` = (k / gridW, k % gridW).
   - Text-only uses zeros, start 2³⁰ and amount 0.
3. **Wire.** The tower's outputs become the decoder's static inputs, bound on every call.
4. **Reset.** A rewind to or before `imageStart` makes the statics text-only again. That's exact: tokens before
   `imageStart` never read them.

`check-report.json` (Python harness, GPU):
- Embeddings: cosine ≥ 0.9999999984.
- Greedy: 48/48 tokens on all five cases (the Python bar was 32).
- First-token logits: cosine 0.9999975–0.9999995.
- Encode: 0.22 s. Prefill: 0.33 s for about 505 tokens.
- Decode: 600–950 ms per token, slowed by the harness's re-specialization and reloads (§5.3).

## 2. The API

**Recommendation: images are part of a user message, in the existing text stack.**

```swift
let session = try await ChatSession.load(bundle: bundle, manifest: manifest, device: .current)
for try await event in session.respond(to: "Describe this photo.", images: [ImageInput(contentsOf: url)]) { … }

// or directly, as the FoundationModels executor does:
generator.generate([.user("What is being sold?", images: [image])], parameters: …)
```

- **`ChatMessage` gains `images: [ChatImage]`** (user turns only). `ChatImage` is a small `Sendable`, `Hashable`
  reference box over an `ImageInput`. It caches the prepared form, so a history turn isn't decoded and resized again
  on every reply.
  - `templateMessage()` renders a message that has images as content parts: images first, then
    `{type: "text", text}`. That's the shape Qwen's template iterates over, and the order of every reference prompt.
  - Text-only messages render exactly as today.
- **`ChatSession`:**
  - It gains `respond(to:images:)` and keeps images in its history.
  - `load` dispatches on `manifest.driver`: `decoder-lm@1` or `vlm-static-inject@1`.
  - `Registry.text` adds `VLMStaticInjectDriver`.
- **The `Generator` stays the single loop** (DESIGN §7 already names "VLM" as a client). If a request carries images
  and the engine isn't an `ImageConditionedEngine`, it's refused with `unsupportedOperation` ("this model takes no
  images").
- **FoundationModels.** It *does* support images in the 27 SDKs:
  - `LanguageModelCapabilities.Capability.vision`;
  - `Attachment<ImageAttachmentContent>` (from a `CGImage`, `CIImage`, `CVPixelBuffer` or URL) as a
    `PromptRepresentable`;
  - `Transcript.Segment.attachment(.image(ImageAttachment))`, with `cgImage` and `orientation`.

  The bridge refuses every attachment today (`"This model takes no attachments"`). It will:
  - map `.image` to `ChatImage(ImageInput(cgImage, orientation:))`. `ImageInput(_ cgImage:)` gains an
    `orientation:` parameter, which uses `RGB8Image.oriented`, already there;
  - advertise `.vision` when `generator.engine is any ImageConditionedEngine`;
  - keep refusing `.data` attachments.

  This is the strongest argument against a separate `VisionChat`: the bridge, structured output and tools come free
  on the `Generator`.
- `generate(prompt: RenderedPrompt)` stays text-only in this slice (open question 8).

**Rejected: a `VisionChat`/`VisionLanguageSession` type.** That was 0.0.x's shape. It duplicates the history, prefix
reuse, constraints, tools and the bridge, or else wraps `ChatSession` and adds nothing.

## 3. Engine changes

### 3.1 `LanguageEngine` stays as it is

The protocol is the token-level seam (BACKENDS §1.2). An image isn't a token-level concept, and an MLX VLM would take
pixels, not embeddings. So a refinement protocol sits beside it in AetherRuntime:

```swift
public protocol ImageConditionedEngine: LanguageEngine {
    /// Pad/open/close ids, idBase, maxImages, the preprocess configuration (from `image` and `preprocess.image`).
    nonisolated var imageInput: ImageInputSpec { get }
    /// CPU only: decode, smart resize, patchify, grid, count, content digest. No GPU, no actor state.
    nonisolated func prepare(_ image: ImageInput) throws(AetherError) -> PreparedPromptImage
    /// Binds `image`'s conditioning for a prompt whose first pad is at `imageStart`: encodes it on the tower unless
    /// that digest is already bound there; nil restores the text-only values. Refused during an operation.
    func bindImage(_ image: PreparedPromptImage?, at imageStart: Int) async throws(AetherError)
    /// What the statics hold now: digest, imageStart, count (nil: text-only).
    var boundImage: BoundImage? { get }
}
```

`LanguageEngineInvariants` runs unchanged on the VLM engine for text-only prompts. A second invariant set covers image
prompts (§6, G5).

### 3.2 `DecoderCore`: static inputs and extension ids

These are generic: any `decoder-lm@1`-shaped graph with `binding: static` inputs and no `source` gets them.

- **Static inputs.**
  - `DecoderCore.init` currently refuses every input role besides `tokens` and `positions`. It will accept the
    driver contract's static roles instead.
  - Each one gets one arena buffer at the descriptor's shape, which must be fully static.
  - `bindings()` binds every one on every call (prefill chunks, host steps, encoded greedy steps), as whole,
    contiguous buffers. INTEGRATION's lesson: views get copied.
  - The defaults (`image.rope.textOnly`: zero buffers, start 2³⁰, amount 0) are written at load and by
    `resetStatics()`.
- **The binding record:** `(digest, imageStart, count)`. These rules hold it to the KV:
  - `rewind(to: t)` with t ≤ imageStart, and `reset()`, restore the defaults and clear the record.
  - A rewind past the image keeps it.
  - `snapshot()` stores the record, not the bytes (DESIGN §6.2). `restore` of a snapshot whose record differs from
    the bound one is refused (`unsupportedOperation`), so the caller binds first.
  - Qwen3-VL is attention-only, so the Generator never snapshots it. The rule is there for hybrids (lfm2.5-vl).
  - Writes happen only when idle (`requireIdle`), so a static never changes under a running operation.
- **Extension ids.** Token validation (`bad >= vocab`) accepts `[base, base + count)` from `inputEncodings`. Greedy
  argmax is over the head's rows, so it never yields one.

### 3.3 `DecoderEngine`: the tower

- `init(bundle:manifest:…)` accepts `vlm-static-inject@1` and validates `DriverContract.vlmStaticInject` (§4).
- It loads `graphs.vision` as a second `Graph`: `compute: gpu`, no reshapes, and the descriptor checked against the
  graph kind.
- The tower's `feed` outputs are **bound to the decoder's static buffers** (`vision.imageEmbeds` →
  `decoder.imageEmbeds`, the same arena), so an encode writes the conditioning in place. There's no 16 MiB host copy,
  and no second actor hop on the greedy path.
- `bindImage` writes the patches into one owned [2048, 1536] buffer, plus `grid`. It runs the tower (`run`, about
  0.22 s), then writes `image_rc` and the rope shift.
- Later optimization: encoding the tower on the greedy kernels' stream ahead of prefill chunk 1 (no host sync).
- `KVSizing.modelBytes` counts both graphs' measured bytes. `fixedBytes` adds the statics, the patches buffer and
  the grid.

**Rejected:** a `VisionLanguageEngine` actor wrapping `DecoderEngine` and the tower. Every greedy token would cost
two actor hops, and the design would need a second arena, a copy per image and two operation locks.

### 3.4 `Generator`: image-aware prefix reuse

1. **Before the GPU grant:** collect the request's images; more than `maxImages` (1) in the whole conversation is
   refused (`invalidParameters`). `prepare` each one on the CPU (cached in its `ChatImage`).
2. **Inside the grant:**
   - Render, then expand the pads (`ImagePromptPlan`, pure and in AetherRuntime). The plan checks that every pad sits
     between open and close and that pads match images.
   - **Reuse:** cap the common prefix at `boundImage.imageStart` when the prompt's image digest differs from the bound
     one. Two images with the same grid render *identical* token ids (`151936 + slot`). Without this cap, a new
     photo of the same size would silently reuse the old photo's KV. 0.0.x had no cross-request reuse, so it never
     met this.
   - Rewind, then `bindImage` (if the digest or start changed, or to clear), then prefill.
3. **Sampler history:** extension ids map back to `padId` (151655), as HF's `input_ids` have them. Today the
   repetition penalty would index logits out of bounds. Check `begin(prompt:)` constraints the same way.

**Multi-turn.** History turns re-render the same image at the same position. The digest matches, so a turn costs
only its new tokens, and the tower isn't run again. An image can be added in a later turn of a text conversation:
tokens before it don't read the statics. A *second* image while the first is still in the history is refused
(`maxImages: 1`). The graph has one rope shift (open question 2).

## 4. Manifest typing

| Field | Today | Typed as |
|---|---|---|
| `preprocess.image.resize.mode: "x-smartResize"` + `x-smartResize {factor, minPixels, maxPixels}` | `Mode` rejects the value, and the `x-` pass drops the parameters: **the bundle's manifest doesn't decode** | `mode: smartResize`, `factor`, `minPixels`, `maxPixels` on `resize` (MANIFEST §2.5). See the rule below. `maxPixels` must equal the tower's rows · patch² (2048 · 256), checked at load. The prepared shape is variable, so `patches` are zero-padded to the graph input's static row count, read from the descriptor (never guessed) |
| graph kind role `grid` (`vision.inputs.grid`) | not in §1.2 | optional `vision-tower@1` input: int, rank 1, [2] = (h, w) in patches. Bound, it marks a grid-static tower: one capacity shape, positions derived in the graph |
| `graphs.vision.kind: "vision-tower@1"` | an unknown field, ignored | `Manifest.Graph.kind: DriverID?`, which must be in `DriverCatalog.graphKinds`. The driver contract requires it and checks the kind's role contract (a reusable `GraphContract.visionTower`) |
| `image.tokens` | `[String: JSONValue]` | `{pad, padId, open?, openId?, close?, closeId?, count: Expr, idBase}`. At load: `idBase == inputEncodings.tokens.extensionIds.base`, `extensionIds.count ≥` the tower's row count / merge² (512), the tokenizer maps `pad` to `padId` |
| `image.rope` | `JSONValue` | `{start: Expr, amount: Expr, frameTail: Int [0], textOnly: {start, amount, zeroStatics}}`. The grammar (closed: `imageStart`, `count`, `gridH`, `gridW`, `frameTail`, ints, `+ − max`) gains **`*`**, for `count: "gridH*gridW"` |
| `image.grid` absent, with `x-grid` | the rule is only in `x-grid` | normative: with no `image.grid`, the grid is per image, from `preprocess` (patch grid ÷ merge) |
| `image.deepstack` + `x-layout`, `x-imageRC` | `JSONValue`, notes | `deepstack: {rowsPerToken: 3, layout: levelMajor}`. The `imageRC` role is defined in MANIFEST §1.2: int32 [slots, 2] = (k / gridW, k % gridW) for k < count, zero after |
| `vlm-static-inject@1` contract | not in `DriverCatalog.contracts` | `DriverContract.vlmStaticInject`: `vision` (kind-checked), and `decoder` = `decoderLM`'s decoder contract plus `imageEmbeds` (required, rank 2, float, static), with `deepstackEmbeds`, `imageRC`, `ropeShiftStart`, `ropeShiftAmount` optional (static). Requires `image`, `preprocess.image` (`layout: patches`) and the tokenizer. Each `image.feed` names a real output and a real static input |

**The smart-resize rule.** It's upstream's, and 0.0.x's port is correct:
- each side rounds to a multiple of the factor, **ties to even** (Python's `round`; Swift needs
  `.toNearestOrEven`, because `.rounded()` would turn 80/32 = 2.5 into 3);
- above `maxPixels`, divide by β = √(HW/max) and floor, at least `factor`;
- below `minPixels`, multiply by β = √(min/HW) and ceil;
- an aspect ratio over 200 is refused.

The exporter writes the typed fields, and the bundle's `aether.json` is regenerated before upload. The graphs don't
change, but `aetherJSONSHA256` does: re-run the check with `--graphs` against the new manifest (open question 4).

## 5. Memory, residency and the GPU lock

### 5.1 What's resident

| Item | Size | Note |
|---|---|---|
| Decoder files | 3.44 GB | Core AI allocates a multiple of the files on first run (G0: about 4× on qwen3-0.6b) |
| Tower files | 0.81 GB | one static function (0.0.x had five grid functions) |
| Metal at the first image request | **to measure.** 0.0.x one-function 2B: 8.7 GiB, warm level 5.9 GiB | expect less: one tower function instead of five |
| KV | **114,688 B/token** (28 × 2 × 8 × 128 × fp16): 109 MiB per 1k tokens, 448 MiB at 4,096, 3.5 GiB at 32,768 | one image prompt (about 505 tokens) ≈ 55 MiB |
| Statics + patches | 16 MiB (`image_embeds` 4, `deepstack_embeds` 12, `image_rc` 4 KiB) + 12 MiB patches | f32 as exported; the tower writes into them in place |

- **Both graphs stay resident for the engine's lifetime** on macOS, as in 0.0.x. Releasing the tower after each
  encode would save its share but pay a load and a specialization (0.7–1.2 s in the harness) per image (open
  question 3).
- Capacity comes from `KVSizing` as for text, with the tower in `modelBytes`. `ModelHost` measures the engine as one
  model: the larger of Metal and `phys_footprint` after a run, so the first measured request should be an image
  request.

### 5.2 The GPU lock

A reply takes **one** `GPUAdmission` grant at its `GenerationParameters.job` priority. The grant covers the tower
encode, the prefill and the decode, and is held until the decode's GPU work drains (DESIGN §10: "one grant is one
turn"). This holds by construction, because `bindImage` runs inside `Generator.prepare`, inside the grant.

Decode, resize, patchify and digest are CPU work, done in `prepare` before the grant, so they don't hold the GPU.

0.0.x took separate admissions for opening the model, for the encode and for the decoder request. Another owner's job
could run between the encode and the prefill.

### 5.3 Re-specialization

The Python runtime has no `expectFrequentReshapes`. It specialized the dynamic decoder anew for every decode length
and kept each specialization's GPU memory. After about a dozen lengths that ended in Metal command-buffer errors,
then a SIGKILL. That's why the harness reloads every 6 calls and runs one case per process, and why it decodes at
600–950 ms per token.

The exporter claims the Swift engine "is not affected". But 0.0.x measured retention **with** the flag too: about
60 MiB per image request on the 2B, 257 MiB per text request on the 4B, bounded by a reload budget (0.0.x
`language-generation.md` "Native memory retention"). The new engine's `metalRetentionOverLongDecode` gate only bounds
runaway growth (< 2 GiB over 2,400 steps). So G7 measures this; it isn't assumed.

The decoder's static inputs have static shapes, so they add no shape changes.

## 6. Gates

These run on the M5 Max (macOS 27), like the M4 gates, against `forge-outputs/qwen3-vl-2b/{bundle,reference}`. Each
gate is opt-in.

| | Gate | Bar |
|---|---|---|
| G1 | **Tower:** the reference `pixel_values.f32`, zero-padded, through `bindImage`'s tower path | cosine ≥ 0.999 for `imageEmbeds` and each DeepStack level (harness: ≥ 0.9999999984); rows past count exactly zero; a repeat bit-identical; 100 encodes across the four grids bit-stable (0.0.x's dynamic tower was wrong 2.5 % of the time; this one is static, and the gate proves it) |
| G2 | **Prompt plan** (CPU, hermetic): render + expand for each case's question | ids equal `input_ids.i32` exactly, extension ids included; `imageStart`, `ropeShiftStart`, `ropeShiftAmount` equal `case.json`; the host M-RoPE rule (0.0.x's `positions`, ported as a test oracle) equals `rope_positions.i32`; smart resize equals `case.json.resized` and the selftest's table |
| G3 | **Preprocess:** Swift from the source image | grid equals `gridTHW`. From Pillow's decode (add a `decoded.png` per case to the reference, as DA3 did): `pixel_values` **bit for bit** (`torchUint8Bicubic` is bit-exact since M4a). From the JPEG through ImageIO: pixel cosine ≥ 0.999, reported, not gated on tokens (0.0.x: ImageIO and PIL differ by up to 46 levels, and that can change the answer) |
| G4 | **Decoder parity:** the engine on each case, with the reference pixels, and end to end from `decoded.png` | **48/48 greedy tokens** equal `greedy.i32` on all five cases (the harness got 48/48 with no divergence); first-token logits cosine and the forced rows reported. A flip at a reference near-tie is reported to the owner, never absorbed by loosening the bar (open question 5) |
| G5 | **Engine invariants on an image prompt** | prefill in chunks of 512, 64 and 1 (the image straddling chunks) and pipelined greedy at depth 3 against host depth 1: the same tokens. A second turn on the same image equals a fresh run, with `reusedTokens` > imageStart + count. A same-grid *different* image reuses nothing past `imageStart`. A rewind before the image, then text, equals a fresh text-only run. `LanguageEngineInvariants` passes text-only |
| G6 | **Public API** | `ChatSession.respond(to:images:)` and a `LanguageModelSession` with an image `Attachment` give G4's tokens; a text model refuses images; the bridge advertises `.vision` only for VLMs |
| G7 | **Retention** (§5.3) | Metal and footprint by phase: load, first image request, then 20 requests mixing the four grids and text-only (64–256 new tokens each), then 2,400 decode steps. Bars: no GPU error; growth after warm-up < 1 GiB over the 20 requests and < 2 GiB over the long decode (as `metalRetentionOverLongDecode`); the tower flat across 100 encodes. Recorded: decode ms per token (0.0.x 2B: 12.6–14.4), `GreedyTimings.bindings` (five more inputs per step), time to first token. If it fails, slice V6 adds 0.0.x's reload budget (the KV is engine-owned, so it survives a reload) |
| G8 | **iPhone feasibility** | not a gate in M4g: see below |

**iPhone: macOS first, and the bundle says so** (`variant.platform: macos`).
- 4.27 GB of fp16 meets 6.0 GiB available on the iPhone Duo with the increased-memory entitlement (DEVICE.md).
- Core AI allocates a multiple of the files on first run.
- In 0.0.x the VL-2B aborted on an iPhone18,2 inside MPSGraph's specialization (`std::bad_alloc`, attributed to
  virtual address space) at the default limit, and the entitled run was never completed.
- An iOS variant needs an int8 (or 4-bit) decoder export (about 1.7 GB), the extended-virtual-addressing entitlement,
  and probably a KV cap (`kvCapacityCap.ios`). It's its own slice after the Mac gates: first a load-and-one-reply
  probe in `Tools/DeviceGate`, then G1/G4 on the device.

## 7. From 0.0.x: port, and what the review found

| 0.0.x | Verdict |
|---|---|
| `Qwen3VLImagePreprocessor.targetSize` (smart resize) | **Port.** It's correct: ties to even, floor with `max(factor)`, ceil up, refusal over 200 |
| Its antialiased bicubic ("within one level of torchvision") | **Superseded** by M4a's bit-exact `torchUint8Bicubic` |
| `LanguagePromptPlan.expand` (the pad checks) and `positions` (M-RoPE) | **Port** `expand`'s checks into `ImagePromptPlan`; `positions` becomes G2's oracle. The graph now synthesizes M-RoPE itself |
| Grid snapping (five fixed grids, aspect stretched by up to 15 %) | **Dropped.** The grid-static tower takes any smart-resized size up to 2,048 patches, as upstream does |
| Aligned image rows per prefill chunk | **Dropped.** The statics are indexed by slot, so chunks need no per-chunk image rows |
| Separate GPU admissions for open, encode and request; `busy` on overlap | **Replaced:** one grant per reply; requests queue (the Generator's gate) |
| Reuse without an image identity | 0.0.x didn't reuse across requests, so it never had the bug. Here, the digest cap (§3.4) is required |
| JPEG decode mismatch, measured and recorded | **Kept as a rule:** gate on identical decodes, and report the independent decode |

## 8. Then OCR (M4h)

On the M4g plumbing, each with its own Forge export:
1. **glm-ocr** (`vlm-static-inject@1`). This is the generality test: a different template, a static `image.grid`, a
   `pixels` tower input, a white letterbox pad (MANIFEST §2.5), and its own rope expressions.
2. **minicpm-v-4.7-1b** and **mineru2.5-pro** (`frameTail: 2`, `from` renames).
3. **Unlimited-OCR on `vlm-embeds@1`.** This is a new decoder shape: `prefill` + `decode` functions fed `embeds`
   [1,L,H] / [1,1,H] with a scalar `position`, `tables.embedTokens`, `image.arrange` (rows, newline and separator
   tables), and the `generation` guards. It needs its own engine mode (embeds instead of tokens) and is the bigger
   half of M4h.

OCR is a prompt on the same `ChatSession` API. A one-call `Aether.readText(image)` belongs to M6. Gates as in G1–G6,
plus character error rate against upstream on a document set.

## 9. Slices

Each slice is one PR with its docs, the API baseline and CI, like M4a–M4f.

| Slice | Contents | Gates | Effort |
|---|---|---|---|
| **V1** Manifest | §4: the `smartResize` mode, `Graph.kind`, the `vision-tower@1` contract with `grid`, typed `ImageBlock` and the `*` grammar, `DriverContract.vlmStaticInject`, MANIFEST §1.2/§2.5/§2.6. The Forge exporter writes the typed fields; regenerate the bundle's `aether.json` and re-check | the bundle's manifest decodes and validates; hermetic refusals | 1 day |
| **V2** Preprocess + plan | smart resize in `ImagePreprocessor` (padded to the descriptor's rows, plus `grid`, digest), `ImagePromptPlan`, `ImageInput(_:orientation:)`; the reference gains `decoded.png` (a CPU-only re-run) | G2, G3 (hermetic parts in CI) | 1.5 days |
| **V3** Engine | §3.2–§3.3: `DecoderCore` statics, extension ids, the rewind/snapshot rules; the tower in `DecoderEngine`; `ImageConditionedEngine`; tests on fake graphs (a fake tower and a fake decoder that reads its statics) | G1, G4 (reference pixels), G5 | 2.5–3 days |
| **V4** Text stack + bridge | `ChatMessage.images`/`ChatImage`, content-part rendering, Generator reuse and the sampler fix, `ChatSession.respond(to:images:)` and load dispatch, `Registry.text`, FoundationModels `.vision` and image attachments | G4 end to end, G6 | 2 days |
| **V5** Memory + catalog | G7; `KVSizing` with the tower; docs (DESIGN §6.2/§7, M4.md M4g done, CHANGELOG); catalog `qwen3-vl-2b` (macOS) once the owner uploads the bundle | G7 | 1 day + upload |
| V6 (only if G7 fails) | a decoder reload budget at idle (0.0.x's mitigation) | G7 again | 1–2 days |

Then the iOS variant (§6), and M4h (§8).

## 10. Risks

- **Static inputs bound every step.** If Core AI copies 16 MiB of f32 statics on each decode call (it copies
  oversized or non-contiguous views, at 3–8×), the decode slows by milliseconds per token. G7 records
  `GreedyTimings.bindings` and tok/s. The fix would be an export change (fp16 statics, or statics as states), not a
  runtime one.
- **Retention** (§5.3): G7 measures it, and V6 is the mitigation.
- **swift-jinja and content parts.** Qwen's template tests `content.type == 'image' or 'image' in content` on dicts.
  G2 must show the rendered ids equal the reference's before expansion. If swift-jinja's `in` on a dict misbehaves,
  render the parts in a family renderer.
- **Extension ids in other paths:** the sampler's penalty history, `JSONSchemaConstraint.begin(prompt:)`,
  `TokenText`. V4 tests each with ids ≥ vocab.
- **One image per context:** apps that send a second photo mid-conversation get a refusal (open question 2).
- **Decode speed is unmeasured in Swift.** The harness's 600–950 ms per token is the reload workaround, not the
  model. If Swift isn't near 0.0.x's 13 ms, investigate before V4.

## 11. Open questions for the owner

1. **API shape:** `ChatMessage.images` rendered before the text (recommended: one image per context, the reference
   order), or ordered content parts (`[.text, .image, .text]`) now?
2. **A second image** while one is in the history: refuse (recommended), drop the earlier image from the rendering,
   or plan a re-export with one rope shift per image (`maxImages` > 1)?
3. **Tower residency:** resident for the engine's lifetime (recommended on macOS), or released after each encode?
4. **Manifest:** promote `x-smartResize` to a normative `smartResize` and regenerate the bundle's `aether.json`
   (graphs unchanged) before the upload? The upload to `aether-models` stays your step.
5. **G4's bar:** strict 48/48, or the near-tie rule (INTEGRATION) if one case flips at a reference margin < 0.1?
6. **iOS:** take on the int8 decoder export and the device run inside M4g, or after M4h?
7. **FoundationModels:** advertise `.vision` and take image attachments in V4 (recommended), or keep the bridge
   text-only for 0.2?
8. **`generate(prompt: RenderedPrompt)` with images:** does a client need pre-rendered prompts with images (0.0.x's
   Atelier describer path), or is the messages API enough?
