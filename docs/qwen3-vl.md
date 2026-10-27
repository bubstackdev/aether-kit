# Qwen3-VL image input (experimental)

Image + text prompts to the language runtime with `Qwen/Qwen3-VL-2B-Instruct`
(Apache-2.0, not gated; pinned revision `89644892e4d85e24eaac8bacfd4f463576704203`). The
design follows the verified aether-engine prototype (`docs/spec/11-vision.md` Part A);
every piece is re-implemented here and qualified against upstream Transformers.

## Using it

```swift
import AetherLanguageCoreAI
import AetherVision

let session = try await VisionLanguageSession.open(bundle: bundleURL, platform: .macos)
let photo = try VisionRGBImage(contentsOf: photoURL)          // or VisionRGBImage(cgImage:)
// The consumer renders the chat template, with one <|vision_start|><|image_pad|><|vision_end|>
// placeholder per image, and tokenizes it (tokenization stays the consumer's).
let reply = try await session.generate(renderedTokenIDs: promptIDs, images: [photo], maximumTokens: 256)
await session.unload()
```

The bundle's `language-model.json` carries a `vision` registration
(`LanguageVisionRegistration`): the encoder asset and its hash, the image tokens, the
preprocessing parameters, an image-token budget (`maximumImageTokens`) and the encoder's fixed
patch grids (`imageGrids`), plus `exportBounds` with a `maximumImageChunk` covering the budget.
`Forge/language/register_qwen3_vl_bundle.py` writes one for an assembled bundle, listing every tokenizer file (`tokenizerFiles`) so `verifyTokenizer` checks the exact set. Each image
snaps to the grid nearest its aspect ratio, is resized to that grid exactly, encoded once by
that grid's static-shape function, and spliced into the decoder. Decoding is greedy. One request at a time; overlap fails with `busy`. With a `gpu:` owner, the
encode phase and the decoder's request are each admitted. `unload()` first withdraws a request
still waiting for either admission (it fails with `closed`), so close never waits behind another
owner's GPU work; an admitted request finishes first, then both models are released. A
registration without grids is refused (no dynamic-shape encoder is opened), as is a grid that is
repeated, not a multiple of the merge size, or over the token budget; the encoder refuses a grid
without a matching all-static function at open, and the session refuses an unregistered grid
while planning. The registration's `maximumImageTokens · (patch · merge)²` must cover
`minimumPixels` (an overflowing product is refused, not trapped), and the decoder's export bounds must declare a
`maximumImageChunk` covering the budget (also on the raw `open(textModel:…)` path); an image
larger than the exported chunk is refused before any native work. A bare
`LanguageGenerationSession` over an image-capable asset without a declared chunk still opens for
text; its image prompts are refused while planning. Image resampling runs off the session actor.

**The encoder is the guarded boundary.** The grid restriction holds on the `VisionLanguageSession`
path, where Kit preprocesses, encodes and checks every image. The lower-level entry points are not
grid-restricted: `LanguageGenerationSession.generate(prompt: LanguageVisionPrompt…)` and the public
`LanguageImageFeatures` initializer accept features for any grid, so a direct caller can submit an
image span of any size up to the decoder's `maximumImageChunk` (2,048 here). The decoder's
`withImage` then sees shapes outside the qualified set; those are the caller's responsibility.

## Pieces

| Piece | Where |
| --- | --- |
| Vision encoder asset (`encode_<h>x<w>`, one per grid) | `Forge/language/export_qwen3_vl_vision.py`; run by `LanguageVisionEncoder` |
| Per-patch host tables | `AetherVision.Qwen3VLVisionTables` (bit-exact with the Forge mirror) |
| Image decoding and preprocessing | `AetherVision.VisionRGBImage`, `Qwen3VLImagePreprocessor` |
| Text decoder (`main` + `withImage`, bucketed contract) | `Forge/language/export_qwen3_vl_text.py`; run by `LanguageGenerationSession` |
| M-RoPE positions and prompt expansion | `LanguagePromptPlan` (pure, CPU-tested against upstream's rope index) |
| Public API | `VisionLanguageSession` |

**Vision encoder.** One static-shape function per fixed grid, `encode_<h>x<w>(pixels [n, 1536]
fp16, pos_index [n, 4] i32, pos_weight [n, 4] f32, patch_rc [n, 2] i32) -> image_embeds [n/4,
2048], deepstack [3, n/4, 2048]` with n = h·w, rows in merge-window order. No dimension is
dynamic. The five functions share one copy of the weights (778 MB). Weights are stored fp16 and
every matmul, activation and the residual stream compute in float32: the residual stream
outgrows fp16, and a pure-fp16 tower loses its embeddings on Core AI's cpu_only path (engine
F-P3-21).

| Grid (patches) | Pixels | Aspect | Image tokens |
| --- | --- | --- | --- |
| 44×44 | 704×704 | 1:1 | 484 |
| 36×48 | 768×576 | 4:3 | 432 |
| 48×36 | 576×768 | 3:4 | 432 |
| 32×56 | 896×512 | 7:4 (≈ 16:9) | 448 |
| 56×32 | 512×896 | 4:7 (≈ 9:16) | 448 |

**Text decoder.** One fp16 asset (3.2 GB) with two bucketed functions over shared weights:
`main(input_ids, cache_positions, rope_positions [3, q], keyCache, valueCache)` and
`withImage(..., image_embeds [n, 2048], deepstack [3, n, 2048], ...)`, each returning last-row
logits and the chunk's cache rows. `withImage` splices image rows at `<|image_pad|>` tokens with
an embedding gather and adds the DeepStack features after the first three decoder layers. With
the fixed grids, `withImage` only ever sees three image submission sizes (484, 448, 432). Rotary
positions are 3-axis M-RoPE (`rope_positions`), separate from the cache positions that address
rows and the mask (rope positions inside an image are not monotonic, engine F-P3-1). Text runs are
prefilled in power-of-two chunks through `main`; each image's tokens go in one `withImage` call
(at most `maximumImageChunk`, 2,048 here).

**Preprocessing.** The image snaps to the grid whose aspect ratio is nearest its own in log
space (the first listed wins a tie; `Qwen3VLImagePreprocessor.nearestGrid`, mirrored by Forge's
`nearest_grid` and CPU-tested against it) and is resized to that grid's pixel size exactly: no
padding, so the aspect ratio is stretched by up to about 15 % between neighbouring grids, more
beyond 16:9. Without grids, the preprocessor keeps the reference's smart resize (multiples of 32
within the pixel bounds, identical to the reference for 46 sizes). Then antialiased bicubic resampling of the 8-bit image (Keys a = −0.5,
PIL-style coefficients; within one level of torchvision's uint8 path, ≥ 95 % identical bytes),
(x/255 − 0.5)/0.5, the frame repeated twice, merge-window patchify. Decoding uses ImageIO,
applies EXIF orientation, and draws in the image's own RGB colour space (no colour management,
like the reference's PIL decoding). EXIF orientation is applied **before** grid snapping, so a
rotated portrait snaps by its displayed aspect ratio. Parity is not exercised for rotated photos:
every reference photo carries orientation 1 (or none), and the reference decodes without an EXIF
transpose.

## Dynamic shapes are a correctness hazard for the encoder

The first encoder export had one dynamic-shape function. Specialized **without**
`expectFrequentReshapes`, it still computed wrong outputs on the GPU intermittently: encoding the
same pixels (the portrait photo, grid 1×48×40) 360 times across three processes gave 9 wrong
outputs (2.5 %; 6, 3 and 0 per process, scattered, once twice in a row). The bad outputs were
always the same bits, with cosine 0.526 (image embeddings) and 0.651 (DeepStack) against the
good output, which was bit-stable across processes. They decode to a different, still
image-related answer ([control](evidence/qwen3-vl-vision-encoder-2026-10-26-grids-dynamic-control-1.json)).
The decoder was deterministic on fixed features (80 of 80). Those cosines are in the range of
F-P3-26 below, so the reshape-optimized wrong path appears to be taken occasionally even with
the flag off. This is proposed as a new engine finding next to F-P3-26.

The encoder is therefore exported with static shapes only, one function per fixed grid, and Kit
refuses anything else. Static functions were bit-stable in every encode measured (see
Qualification).

### The reshape flag

Specialized with `expectFrequentReshapes`, the dynamic-shape vision encoder computes **wrong**
outputs on the GPU: cosine 0.16–0.67 against the reference here, reproducing engine F-P3-26
([evidence](evidence/qwen3-vl-vision-encoder-2026-10-26.json)). It is therefore not a tuning
choice. The flag comes from the registration (`LanguageVisionRegistration.expectFrequentReshapes`),
like FLUX's placement; verification refuses a registration that requests it while
`LanguageVisionRegistration.reshapesQualified` is false, and `LanguageVisionEncoder` resolves
every request through the same rule, so the encoder is never opened with it. CPU tests cover
both. Lifting the refusal needs a qualification showing the flag correct. The text decoder keeps
the language runtime's own specialization (reshapes on), which its qualification covers.

## Qualification

Evidence: [encoder](evidence/qwen3-vl-vision-encoder-2026-10-26-grids.json) and
[image + text](evidence/qwen3-vl-image-text-2026-10-26-grids.json); M5 Max (`h17c`), GPU-preferred,
release builds, shared host. They supersede the dynamic-shape records
([encoder](evidence/qwen3-vl-vision-encoder-2026-10-26.json),
[image + text](evidence/qwen3-vl-image-text-2026-10-26.json)), which stay for history.

**Inputs.** Five deterministic synthetic scenes, one at each grid's exact pixel size (encoder
parity); two more scenes (448×448, 320×256) and three real photographs from Wikimedia Commons at
different aspect ratios (image + text), with source and licence recorded in the evidence. Every
image is snapped to its grid by Kit and by the reference alike:

| Image | Original size | Grid | Licence |
| --- | --- | --- | --- |
| `File:Bully-dog-2314909-1920.jpg` (portrait) | 1200×1412 | 48×36 | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) |
| `File:Bangkok fruit stall.jpg` (16:9) | 4032×2268 | 32×56 | Public domain, [`{{PD-self}}`](https://commons.wikimedia.org/wiki/Template:PD-self) |
| `File:'Byculla Fruit Market' near Byculla Bridge(east)..JPG` (4:3) | 810×608 | 36×48 | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) |
| Synthetic scene | 448×448 | 44×44 | generated (seeded) |
| Synthetic scene | 320×256 | 36×48 | generated (seeded) |

**Reference.** Upstream `Qwen3VLForConditionalGeneration` with its own processor and chat
template, float32 on the CPU (`Forge/language/reference_qwen3_vl.py --grids`): the processor's
`smart_resize` is replaced by the snapped grid's pixel size, so upstream resizes the image to the
same grid Kit encodes. The Forge eager port (vision + text towers) matches it at cosine 1.0 with
every argmax equal, and the host M-RoPE positions equal the model's rope index, on all five
cases. Encoder parity uses upstream's `Qwen3VLVisionModel` on the scenes' own pixels
(`reference_qwen3_vl_vision.py`).

**Encoder parity** (per grid, same pixels as upstream): worst cosine 0.9999999 for the image
embeddings and every DeepStack level, on all five grids. Each grid's function is specialized at
load, so the first encode is not slower than a repeat (0.23–0.29 s at 432–484 tokens).

**Encoder stability** (the portrait resized to every grid, encoded round-robin across the five
grids, each output compared bit for bit with that grid's first encode): **600 encodes per grid
across three processes, zero differences**, and the same first output in every process. The
dynamic-shape control on the same photo: 9 wrong outputs in 360 encodes.

**Memory** (process footprint, encoder only, all five grids warmed): about 0.95–1.0 GB after
open, about 1.6 GB once every grid has encoded (1.60–1.65 GB), flat after 1,000 more encodes
(1.66–1.67 GB). The grids are a fixed set, so this does not grow with the number of photos.

**Model parity** (reference pixel tensors through Kit's encoder and decoder; gate: forced-row
cosine ≥ 0.995, argmax equal unless a near-tie, greedy identical up to the reference's end):
all five cases pass: forced-row cosine ≥ 0.99992, no argmax differences, and every greedy
continuation identical to upstream, photos included.

**Kit end to end** (Kit's ImageIO decoding and grid snapping of the files; gate: equal grids,
pixel cosine ≥ 0.999 against the reference processor, forced-row cosine ≥ 0.98, tokens compared
and reported): grids equal on all five; pixel cosine 0.99997, 0.99999 and 0.99990 for the photos
and 1.0 for the scenes; forced-row cosine ≥ 0.99201. Greedy output is identical to upstream for
the portrait and both scenes. The fruit stall agrees for 46 of 48 tokens ("both whole and **cut
open**" vs upstream "both whole and **in various**"), diverging at a reference near-tie: token 46
is past the 32 forced rows, so its margin was measured separately with upstream on the same
prompt and prefix ("in" 0.457, "cut" 0.403, gap 0.054; Kit's token is upstream's second choice;
[record](evidence/qwen3-vl-image-text-2026-10-26-grids-stall-divergence.json)). The 4:3 market photo agrees for 16 tokens,
then diverges at a reference near-tie (top-1/top-2 probability gap 0.066): Kit answers "likely
taken in **India**, specifically in a bustling **local market**", upstream "likely taken in
**Mumbai, India**". The pixels differ slightly because a different JPEG decoder and resampler
produce them, which is expected and recorded rather than hidden.

**Public API.** `VisionLanguageSession.open(bundle:)` on the registered bundle reproduces the
component-level Kit run token for token on all five cases, and closes on `unload()`.

## Limits and follow-ups

- **Aspect ratio is stretched to the nearest grid.** No padding: an image between two grids is
  stretched by up to about 15 %, and panoramas or tall strips beyond 16:9 more. More grids (for
  example 2:1) can be added to the export and registration without code changes.
- **No resource admission** for the encoder and the decoder yet (GPU admission only).
- **Coverage still missing:** a multi-image M-RoPE fixture (several images use the same path,
  but only single-image prompts are qualified), iPhone qualification (vision asset release after
  use, image-token caps, memory), and a quantized decoder (fp16 weights only today).
- The encoder stays loaded for the session's lifetime.
