# Qwen3-VL image input (experimental)

Image + text prompts to the language runtime with `Qwen/Qwen3-VL-2B-Instruct`
(Apache-2.0, not gated; pinned revision `89644892e4d85e24eaac8bacfd4f463576704203`), and with the
4B describer `huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` ([below](#qwen3-vl-4b-describer)). The
design follows the verified aether-engine prototype (`docs/spec/11-vision.md` Part A);
every piece is re-implemented here and qualified against upstream Transformers.

## Using it

```swift
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision

let session = try await VisionLanguageSession.open(bundle: bundleURL, platform: .macos)
let photo = try VisionRGBImage(contentsOf: photoURL)          // or VisionRGBImage(cgImage:)
// The chat template renders one <|vision_start|><|image_pad|><|vision_end|> placeholder per image;
// AetherLanguageTokenizers renders and tokenizes it from the bundle's own tokenizer files.
let tokenizer = try await LanguageTokenizer.open(bundle: bundleURL)
let prompt = try tokenizer.render([.user("Describe her look in one line.", images: 1)])
let reply = try await session.generate(renderedTokenIDs: prompt.tokenIDs, images: [photo], maximumTokens: 256)
let text = tokenizer.decode(reply.tokens)
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
like the reference's PIL decoding); grayscale expands to equal channels. Alpha is composited over
black, where the reference's `convert("RGB")` drops alpha and keeps the stored colour, so
transparent regions differ (opaque images are unaffected). `VisionRGBImageTests` checks channel
order, colour, grayscale, alpha and each EXIF orientation on the CPU with generated fixtures.
EXIF orientation is applied **before** grid snapping, so a
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
produce them, which is expected and recorded rather than hidden. (For the 4B the decoder difference
is larger than this band, so its end-to-end gate compares against upstream fed Kit's own decode, and
an independent-decode run like this one gates only decoder sanity (equal grids, pixel cosine ≥ 0.999)
and reports its logits and tokens; see [below](#qwen3-vl-4b-describer). This 2B record predates that
change.)

**Public API.** `VisionLanguageSession.open(bundle:)` on the registered bundle reproduces the
component-level Kit run token for token on all five cases, and closes on `unload()`.

## Qwen3-VL-4B describer

`huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated` (Apache-2.0, not gated; pinned revision
`ce72a7c22aacb493fb94478de3bfbe834c61844a`; base model `Qwen/Qwen3-VL-4B-Instruct`, recorded as
`baseModel`), Atelier's character describer. It is the 2B's architecture at a larger width (text:
2,560 hidden, 36 layers, 32/8 heads; vision: 24 blocks projecting to 2,560, DeepStack at blocks 5,
11, 17), so the same exporters, preprocessing, five grids and session serve it with no Kit code
changes. Evidence: [encoder](evidence/qwen3-vl-4b-vision-encoder-2026-10-26.json) and
[image + text](evidence/qwen3-vl-4b-image-text-2026-10-26.json); M5 Max (`h17c`), release, shared host.

**Bundle.** Vision asset 0.83 GB, five static functions emitting `[n/4, 2560]` (CPU self-check
cosine 1.0); text decoder 8.05 GB fp16 (self-check: `withImage` 0.99891, `main` ≥ 0.99993, argmax
equal). The fine-tune keeps its chat template in `chat_template.jinja`, which the text exporter now
copies with the token maps and licence; the registration lists all of them in `tokenizerFiles`.

**Encoder.** Parity per grid against upstream's vision model: worst cosine 0.99999995 for the image
embeddings and every DeepStack level. Stability: the portrait resized to every grid, 200 encodes per
grid in each of three processes (600 per grid), every encode bit-identical to that grid's first and
the same first output in every process; 0.22–0.27 s per encode. Footprint about 1.1 GB after open,
1.7–1.8 GB with all grids warmed.

**Why the end-to-end gate uses Kit's decode.** The first end-to-end run, on the JPEG files, failed
the 0.98 band on the 4:3 market photo (forced-row cosine 0.9706, greedy diverging at token 6 at a
confident reference row, margin 0.625): upstream answers "likely taken in **Chandni Chowk**", Kit
"likely taken in a **local market or bazaar in a South Asian country**". The cause is the JPEG
decoder, not Kit: the upstream processor applied to an ImageIO decode of that photo gives pixel
cosine 0.9999015 against the reference, Kit's whole path 0.9999019, and a PIL decode 1.0000001.
ImageIO and PIL's libjpeg-turbo agree on 74–80 % of bytes and differ by up to 46 levels, and the
4B moves further on that difference than the 2B did: it can name a different place for the same photo. So the gated run gives upstream the decode Kit sees: each photo's
`VisionRGBImage` decode saved as a lossless PNG without a colour profile
(`Scripts/write-kit-decoded-photos.swift`, round trip byte-identical; hashes and per-photo byte
comparison with PIL's decode recorded under `decodedPhotos`), referenced with
`reference_qwen3_vl.py --grids` as before, and `qwen3VLImageTextQualification` with
`AETHER_QWEN3VL_PREPROCESS=swift AETHER_QWEN3VL_INPUTS=same-decode`. Only Kit's resampler then
differs from upstream, and the run must meet the model-parity bar. Upstream fed Kit's decode gives
exactly Kit's 48 market-photo tokens. The JPEG-file run stays in the evidence
(`kit-independent-decode`), gated on decoder sanity: equal grids and pixel cosine ≥ 0.999 against
the reference processor, which a swapped channel or a lost orientation fails. Its logits and tokens
are reported (`logitsWithinDecoderBand`), not gated.

**Results** (gates as in the 2B qualification above, with the end-to-end gate now: equal grids,
pixel cosine ≥ 0.9999, then the model-parity rule):

| Run | Gated | Result |
| --- | --- | --- |
| Model parity, Kit-decoded reference | yes | forced-row cosine ≥ 0.99993, no argmax differences, greedy identical on all five |
| Model parity, PIL-decoded reference | yes | forced-row cosine ≥ 0.99995, no argmax differences, greedy identical on all five |
| Kit end to end, same decode | yes | pixel cosine ≥ 0.9999996 (photos), forced-row ≥ 0.99989, greedy identical on all five |
| Public session (`VisionLanguageSession.open(bundle:)`) | yes | token for token with the Kit run, closes on unload; `tokenizerFiles` verified; `LanguageTokenizer.render([.user(question, images: 1)])` equals each reference prompt |
| Kit end to end, independent decode (JPEG files) | decoder sanity only | grids equal, pixel cosine ≥ 0.99990; reported: portrait, stall and scenes greedy identical, market photo diverges as above |

**Memory** (public session, five cases in one process; footprint / Metal allocated):

| Phase | Footprint | Metal |
| --- | --- | --- |
| Before open | 57 MiB | 0 |
| After open | 1.2 GiB | 1.0 GiB |
| After the first request (portrait) | 2.0 GiB | 19.0 GiB |
| After all five requests | 3.0–3.3 GiB | 22.3 GiB |
| After unload | 656 MiB | 495 MiB |

Metal grows with the first image request to about 19 GiB, then by about 1 GiB per newly used grid
or prompt size, against 8.4 GiB for the dense 4B writer with the same decoder width. See the limits.
Warm requests also keep adding memory (Core AI retains allocations on every shape change; text
requests about 257 MiB each on the 4B) until the decoder's memory budget reloads it between
requests: see [Native memory retention](language-generation.md#native-memory-retention).

## Limits and follow-ups

- **4B decoder memory.** The VL-4B session holds about 19–22 GiB of Metal allocations after
  image requests, more than twice the dense 4B writer (8.4 GiB). Not yet attributed; the likely
  cause is the runtime keeping weights per function (`main` and `withImage`) as conversion does
  (see the text export note below). Measure per phase (encoder only, text-only `main`, first
  `withImage`) before sharing weights across the two functions in the export.
- **JPEG decoding differs from upstream's.** Kit decodes with ImageIO, upstream with PIL
  (libjpeg-turbo). The difference is small in pixels (cosine ≥ 0.9999) and can still change what a
  model says about a photo; the 4B named a different place for the market photo. The model is
  qualified on identical decodes; the decoder difference is measured, not removed. Images with
  transparency also differ: Kit composites alpha over black, PIL drops it.

- **Aspect ratio is stretched to the nearest grid.** No padding: an image between two grids is
  stretched by up to about 15 %, and panoramas or tall strips beyond 16:9 more. More grids (for
  example 2:1) can be added to the export and registration without code changes.
- **No resource admission** for the encoder and the decoder yet (GPU admission only).
- **Coverage still missing:** a multi-image M-RoPE fixture (several images use the same path,
  but only single-image prompts are qualified), iPhone qualification (vision asset release after
  use, image-token caps, memory), and a quantized decoder (fp16 weights only today).
- The encoder stays loaded for the session's lifetime.
- **Text export memory (0.0.3).** `export_qwen3_vl_text.py` converts `main` and `withImage` as two programs over the
  same fp16 weights, and conversion holds weights per program on top of the eager module: the 4B text tower passes
  35 GB of RSS during conversion (it was exported on a 64 GB host with a 45 GB guard). Share the weights across the
  two functions in conversion, or free the eager module before converting, before exporting larger towers or on
  smaller hosts.
