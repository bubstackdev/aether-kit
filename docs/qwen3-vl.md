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
preprocessing parameters, an image-token budget per image (`maximumImageTokens`) and the encoder's fixed
patch grids (`imageGrids`), plus `exportBounds` with a `maximumImageChunk` covering the budget.
`Forge/language/register_qwen3_vl_bundle.py` writes one for an assembled bundle, listing every tokenizer file (`tokenizerFiles`) so `verifyTokenizer` checks the exact set. Each image
snaps to the grid nearest its aspect ratio, is resized to that grid exactly, encoded once by
that grid's static-shape function, and spliced into the decoder. Decoding is greedy by default; the sampling
entry point takes a `LanguageSampling` and a constraint as a text request does ([sampling](language-generation.md)). One request at a time; overlap fails with `busy`. With a `gpu:` owner, opening each model (the
encoder, then the decoder: load and specialization), the encode phase and the decoder's request are each admitted.
With `resources:` (`LanguageResourceAdmission`), the encoder's and decoder's residency is reserved while they are
open and each encode's and request's scratch while it runs ([resource admission](qwen3-vl-resource-admission.md)). `unload()` first withdraws a request
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
image span of any size up to the decoder's `maximumImageChunk` (2,048 here). The decoder then sees
image shapes outside the qualified set; those are the caller's responsibility.

## Pieces

| Piece | Where |
| --- | --- |
| Vision encoder asset (`encode_<h>x<w>`, one per grid) | `Forge/language/export_qwen3_vl_vision.py`; run by `LanguageVisionEncoder` |
| Per-patch host tables | `AetherVision.Qwen3VLVisionTables` (bit-exact with the Forge mirror) |
| Image decoding and preprocessing | `AetherVision.VisionRGBImage`, `Qwen3VLImagePreprocessor` |
| Text decoder (one `main`, bucketed contract) | `Forge/language/export_qwen3_vl_text.py`; run by `LanguageGenerationSession` |
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

**Text decoder.** One fp16 asset (3.2 GB) with one bucketed function (0.0.3):
`main(input_ids, cache_positions, rope_positions [3, q], image_embeds [q, 2048], deepstack [3, q, 2048],
keyCache, valueCache)`, returning last-row logits and the chunk's cache rows. Image rows are aligned
with the tokens and used only where the token is `<|image_pad|>`: there they replace the embedding,
and the DeepStack features are added after the first three decoder layers. Text submissions and
decode steps pass zero rows. One function serves every step because Core AI wires a separate copy
of the weights for each function it runs ([design](qwen3-vl-single-decoder.md)); the 0.0.2 bundles,
with `main` for text and `withImage` (`image_embeds` [n, 2048]) for image spans, still open and
keep their two copies. Rotary positions are 3-axis M-RoPE (`rope_positions`), separate from the
cache positions that address rows and the mask (rope positions inside an image are not monotonic,
engine F-P3-1). The whole prompt, image tokens included, is prefilled in the contract's fixed width
(64), the last chunk stepping back over rows already written
([step contracts](language-generation.md#step-contracts)); each chunk carries the rows of its own
image tokens, so an image may straddle chunks. Sending each image as one submission instead made the
function change to the image's width on every image request, and Core AI retains memory on each
shape change: about 360 MiB per image request on the 4B, against about 100 MiB with uniform chunks
([below](#one-decoder-function-003)). A 0.0.2 bundle sends each image's tokens in one `withImage`
call (at most `maximumImageChunk`, 2,048 here), with three sizes on the fixed grids (484, 448, 432).

**Preprocessing.** The image snaps to the grid whose aspect ratio is nearest its own in log
space (the first listed wins a tie; `Qwen3VLImagePreprocessor.nearestGrid`, mirrored by Forge's
`nearest_grid` and CPU-tested against it) and is resized to that grid's pixel size exactly: no
padding, so the aspect ratio is stretched by up to about 15 % between neighbouring grids, more
beyond 16:9. Without grids, the preprocessor keeps the reference's smart resize (multiples of 32
within the pixel bounds, identical to the reference for 46 sizes). Then antialiased bicubic resampling of the 8-bit image (Keys a = −0.5,
PIL-style coefficients; within one level of torchvision's uint8 path, ≥ 95 % identical bytes),
(x/255 − 0.5)/0.5, the frame repeated twice, merge-window patchify. Decoding uses ImageIO,
applies EXIF orientation, and draws in the image's own RGB colour space (no colour management,
like the reference's PIL decoding); grayscale expands to equal channels. Alpha is dropped and each pixel's stored
colour kept, as the reference's `convert("RGB")` does (`VisionAlphaPolicy.dropAlpha`, the default since 0.0.3): exact
for straight-alpha 8- and 16-bit images, which is how ImageIO returns PNG; a source that arrives premultiplied has lost
its transparent pixels' stored colour, so those become black. `.composite(red:green:blue:)` draws over a colour
instead; over black it reproduces 0.0.2's decoding. Kit's bytes equal upstream's on PNG fixtures in four layouts, and a
transparent scene passes the same-decode run ([evidence](evidence/qwen3-vl-alpha-2026-10-27.json)). Krea's reference
images still drop alpha onto black, following the prototype's oracle. `VisionRGBImageTests` checks channel order,
colour, grayscale, alpha and each EXIF orientation on the CPU with generated fixtures.
EXIF orientation is applied **before** grid snapping, so a
rotated portrait snaps by its displayed aspect ratio. The transform is exact: tagged lossless images decode to their
stored pixels moved where each orientation puts them, byte for byte, for all eight orientations (CPU). Rotated photos
are qualified against upstream fed as Transformers' `load_image` feeds it (`exif_transpose`, then `convert("RGB")`):
four lossless variants of the Kit-decoded photos, stored rotated and tagged with orientation 6, 8 or 3, give the
unrotated images' grids, pixels, forced rows and greedy tokens exactly, on the reference and through Kit's same-decode
run ([evidence](evidence/qwen3-vl-rotated-photos-2026-10-27.json)). That includes the 4:3 market photo's divergence
from upstream at token 16, a reference near-tie (margin 0.012) that its unrotated run shows too.

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

**Several images** ([evidence](evidence/qwen3-vl-multi-image-2026-10-27.json), VL-2B). Three prompts against
upstream with several images each, one case per reference process (14.3 GB peak each): the portrait and the fruit
stall (48×36 and 32×56, 880 image tokens), three images (the portrait, the 4:3 market photo and the 448×448 scene, 1,348 tokens), and the
two photos with a sentence between them. On the CPU, the prompt plan's expansion and M-RoPE positions equal the
processor's ids and the model's rope index for every token (`LanguageVisionPromptTests.severalImagesMatchUpstream`):
each image starts its span at the position after the previous text and the following text resumes at max(t, h, w) of the
merged grid past the span's start, with no temporal offset between images. Natively, model parity passes on all three (forced-row
cosine ≥ 0.99978, no argmax differences, greedy identical), and so does Kit preprocessing on the same decode (pixel
cosine ≥ 0.9999998 and equal grids per image, forced-row cosine ≥ 0.99994, greedy identical). The overlapped-prefill
check adds a two-image prompt whose 64-token chunks cross every image edge: default chunks, 512-token chunks and
1-token steps give identical greedy output. The 4B is not yet qualified with several images.

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

**Memory** (the 0.0.2 two-function bundle, public session, five cases in one process; footprint /
Metal allocated; the one-function bundle's figures are [below](#one-decoder-function-003)):

| Phase | Footprint | Metal |
| --- | --- | --- |
| Before open | 57 MiB | 0 |
| After open | 1.2 GiB | 1.0 GiB |
| After the first request (portrait) | 2.0 GiB | 19.0 GiB |
| After all five requests | 3.0–3.3 GiB | 22.3 GiB |
| After unload | 656 MiB | 495 MiB |

Metal grows with the first image request to about 19 GiB, then by about 0.8 GiB per newly used grid,
against 8.4 GiB for the dense 4B writer with the same decoder width. The decoder part is real
duplication: Core AI wires a separate copy of the text weights for each of its two functions (one
function since 0.0.3). The
per-grid growth is not; those are views of one copy (see the limits).
Warm requests also keep adding memory (Core AI retains allocations on every shape change; text
requests about 257 MiB each on the 4B) until the decoder's memory budget reloads it between
requests: see [Native memory retention](language-generation.md#native-memory-retention).

## One decoder function (0.0.3)

Both bundles were re-exported with one decoder function
(`qwen3-vl-{2b,4b-abliterated}-one-function-bundle-macos`; the vision assets are unchanged) and
qualified again: [VL-2B image + text](evidence/qwen3-vl-2b-one-function-image-text-2026-10-27.json),
[VL-4B image + text](evidence/qwen3-vl-4b-one-function-image-text-2026-10-27.json), and
[memory, retention, timing and export](evidence/qwen3-vl-one-function-2026-10-27.json); M5 Max
(`h17c`), release, shared host. The design and its toy measurements are in
[qwen3-vl-single-decoder.md](qwen3-vl-single-decoder.md).

**Parity.** Every gate of the 0.0.2 qualification passes again on the same references, with the
whole prompt prefilled in 64-token chunks. VL-2B: model parity on the PIL-decoded reference (forced-row
cosine ≥ 0.99990, greedy identical on all five) and on a new Kit-decoded reference of the same cases
(`qwen3-vl-2b-reference-grids-kit-decoded`, made like the 4B's: ≥ 0.99985, greedy identical on all
five), Kit end to end on the same decode (≥ 0.99992; the market photo flips at a reference near-tie,
margin 0.012, at token 16), the independent-decode run (the same two near-tie divergences as before)
and the public session, which 0.0.2 could only run on the long question. VL-4B: model parity on the
Kit-decoded and the PIL-decoded references (≥ 0.99992 and ≥ 0.99995, greedy identical on all five), Kit
end to end on the same decode (≥ 0.99989, greedy identical), the independent-decode run (decoder
sanity; the market photo diverges at token 6 as before) and the public session. The long-question
runs pass on both models (the 4B's greedy flips at the same reference near-tie as in 0.0.2, margin
0.0046). The overlapped-prefill check passes with two image prompts whose 432-token image straddles
chunk boundaries at both edges: after a long question the step-back tail lands in the text, after a
short one it steps back 54 rows into the image; greedy output is identical across 64-token chunks,
512-token chunks and 1-token steps. The decoder timing test's 64-token replies are identical, token for
token, between the two-function and one-function bundles, for a text and an image prompt on both models.

**Memory** (`visionLanguageMemoryByPhase`, no budget; `languageMemoryBudgetBoundsRepeatedRequests`,
60 alternating text and image requests, 2 GiB budget):

| | VL-4B, two functions | VL-4B, one function | VL-2B, two functions | VL-2B, one function |
| --- | --- | --- | --- | --- |
| Wired mapped-file pages after the first image request | 16,086 MiB | 8,414 MiB | not measured | 3,875 MiB |
| Metal at the first image request (by-phase test) | 22.6 GiB | 13.8 GiB | 12.7 GiB | 8.7 GiB |
| Compiled cache, read in at open | 22.5 GiB | 15.0 GiB | about 9 GiB | 6.4 GiB |
| Warm level in the budget test | 19,566 MiB | 10,847 MiB | 10,113 MiB | 6,065 MiB |
| Budget reloads in 60 requests (first after request) | 1 (34) | 2 (20) | 1 (58) | 1 (34) |

An image request now adds about 100 MiB of retained memory on the 4B (60 MiB on the 2B) where the
two-function decoder added about 20 MiB, so the budget reloads the 4B about twice as often; each reload
re-wires one copy of the weights, at idle.

**Timing** (decoder only, 16 requests per kind, 64 tokens; shared host, so medians and minima are
both given). Chunking makes image requests slower to the first token: a 505-token image prompt is eight
64-token submissions instead of one image submission and text chunks. On the 4B the image time to first
token rises from 0.350 to 0.448 s at the median (+28 %) and from 0.293 to 0.374 s at the minimum; on the
2B the minimum rises from 0.193 to 0.239 s while the median moved within noise (0.297 to 0.274 s). Text
requests are unchanged within noise (4B: 0.233 to 0.229 s to the first token, 21.9 to 22.9 ms per
token; 2B: 0.148 to 0.135 s, 14.4 to 12.6 ms).

**Export.** The bounded pipeline exported the 4B text tower in 109 s with a peak of 16.3 GB RSS
(0.0.2: 31.7 GB) and the 2B in 46 s at 7.1 GB, self-check included.

## Limits and follow-ups

- **0.0.2 bundles keep two copies of the decoder's weights.** A two-function bundle (`main` and
  `withImage`) still opens and wires a copy for each function; the one-function export above wires one
  ([One decoder function](#one-decoder-function-003)). The measurement that found the duplication, by
  phase on the 0.0.2 bundles (`visionLanguageMemoryByPhase`, opt-in; one M5 Max;
  [evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)):

  | | VL-2B, Metal allocated | VL-4B, Metal allocated | VL-4B, wired on the host |
  | --- | --- | --- | --- |
  | Encoder open | 0.9 GiB | 1.0 GiB | +1.1 GiB |
  | First grid function | +1.1 GiB | +1.1 GiB | +1.4 GiB |
  | Each further grid function (four) | +0.8 GiB | +0.8 GiB | about 0: views of the same weights |
  | Decoder open | −0.3 GiB | −0.3 GiB | none wired; the compiled file is read in (host free −23.3 GiB) |
  | Decoder `main`, first use | +3.8 GiB | +8.2 GiB | +8.4 GiB |
  | Decoder `withImage`, first use | +4.1 GiB | +8.7 GiB | +7.9 GiB, distinct pages from `main`'s |
  | A warm image request | about 10 MiB | about 20 MiB | about 0 |
  | Decoder unload | −8.1 GiB | −17.4 GiB | −16.6 GiB; host free +21.7 GiB |

  The pages are file-backed, so the process footprint stays under 3 GiB, but they are RAM while the
  functions are in use; under memory pressure an idle model's pages were unwired (the encoder's, when the
  decoder opened). Opening a decoder still reads its whole compiled file into the page cache (15 GiB for
  the one-function 4B): clean, evictable pages that push out other processes' cache. Fewer encoder grids
  would save nothing resident. Cross-function weight sharing in Core AI would make the duplication moot;
  it is not available in this release.
- **JPEG decoding differs from upstream's.** Kit decodes with ImageIO, upstream with PIL
  (libjpeg-turbo). The difference is small in pixels (cosine ≥ 0.9999) and can still change what a
  model says about a photo; the 4B named a different place for the market photo. The model is
  qualified on identical decodes; the decoder difference is measured, not removed.

- **Aspect ratio is stretched to the nearest grid.** No padding: an image between two grids is
  stretched by up to about 15 %, and panoramas or tall strips beyond 16:9 more. More grids (for
  example 2:1) can be added to the export and registration without code changes.
- **Resource admission is per process.** `LanguageResourceAdmission` reserves the encoder's and decoder's memory
  against a host's `ResourceCoordinator`, but a describer in its own process (as Atelier runs it) is invisible to a
  Krea process's ledger. The cross-process case waits for the [shared-protocol proposal](shared-residency-proposal.md).
- **Coverage still missing:** iPhone qualification (vision asset release after
  use, image-token caps, memory), and a quantized decoder (fp16 weights only today).
- The encoder stays loaded for the session's lifetime.
- **Text export memory.** The 0.0.2 export of the 4B text tower peaked at 31.7 GB of RSS (it was run on
  a 64 GB host with a 45 GB guard). The cause was not one copy of the weights per program (the converter
  keeps one copy however many programs share them) but about five copies at once during the cpu_only
  self-check, plus a float32 build of the tower. The exporter now builds in fp16, computes the eager
  reference first and runs the self-check in a fresh process: 16.3 GB for the 4B, 7.1 GB for the 2B
  ([design](qwen3-vl-single-decoder.md#export-memory)).
