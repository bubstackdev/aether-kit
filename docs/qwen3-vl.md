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
preprocessing parameters and an image-token budget (`maximumImageTokens`), plus
`exportBounds` with a `maximumImageChunk` covering it. `Forge/language/register_qwen3_vl_bundle.py`
writes one for an assembled bundle. Each image is preprocessed like the reference processor,
with its pixels capped to the token budget, encoded once, and spliced into the decoder.
Decoding is greedy. One request at a time; overlap fails with `busy`. With a `gpu:` owner, the
encode phase and the decoder's request are each admitted. `unload()` first withdraws a request
still waiting for either admission (it fails with `closed`), so close never waits behind another
owner's GPU work; an admitted request finishes first, then both models are released. The
registration's `maximumImageTokens · (patch · merge)²` must cover `minimumPixels` (an
overflowing product is refused, not trapped), and the decoder's export bounds must declare a
`maximumImageChunk` covering the budget (also on the raw `open(textModel:…)` path); an image
larger than the exported chunk is refused before any native work. A bare
`LanguageGenerationSession` over an image-capable asset without a declared chunk still opens for
text; its image prompts are refused while planning. Image resampling runs off the session actor.

## Pieces

| Piece | Where |
| --- | --- |
| Vision encoder asset (`encode`) | `Forge/language/export_qwen3_vl_vision.py`; run by `LanguageVisionEncoder` |
| Per-patch host tables | `AetherVision.Qwen3VLVisionTables` (bit-exact with the Forge mirror) |
| Image decoding and preprocessing | `AetherVision.VisionRGBImage`, `Qwen3VLImagePreprocessor` |
| Text decoder (`main` + `withImage`, bucketed contract) | `Forge/language/export_qwen3_vl_text.py`; run by `LanguageGenerationSession` |
| M-RoPE positions and prompt expansion | `LanguagePromptPlan` (pure, CPU-tested against upstream's rope index) |
| Public API | `VisionLanguageSession` |

**Vision encoder.** `encode(pixels [n, 1536] fp16, pos_index [n, 4] i32, pos_weight [n, 4] f32,
patch_rc [n, 2] i32) -> image_embeds [n/4, 2048], deepstack [3, n/4, 2048]`, rows in merge-window
order. Weights are stored fp16 and every matmul, activation and the residual stream compute in
float32: the residual stream outgrows fp16, and a pure-fp16 tower loses its embeddings on Core
AI's cpu_only path (engine F-P3-21). 777 MB.

**Text decoder.** One fp16 asset (3.2 GB) with two bucketed functions over shared weights:
`main(input_ids, cache_positions, rope_positions [3, q], keyCache, valueCache)` and
`withImage(..., image_embeds [n, 2048], deepstack [3, n, 2048], ...)`, each returning last-row
logits and the chunk's cache rows. `withImage` splices image rows at `<|image_pad|>` tokens with
an embedding gather and adds the DeepStack features after the first three decoder layers. Rotary
positions are 3-axis M-RoPE (`rope_positions`), separate from the cache positions that address
rows and the mask (rope positions inside an image are not monotonic, engine F-P3-1). Text runs are
prefilled in power-of-two chunks through `main`; each image's tokens go in one `withImage` call
(at most `maximumImageChunk`, 2,048 here).

**Preprocessing.** Smart resize to multiples of 32 within the pixel bounds (identical to the
reference for 46 sizes), antialiased bicubic resampling of the 8-bit image (Keys a = −0.5,
PIL-style coefficients; within one level of torchvision's uint8 path, ≥ 95 % identical bytes),
(x/255 − 0.5)/0.5, the frame repeated twice, merge-window patchify. Decoding uses ImageIO,
applies EXIF orientation, and draws in the image's own RGB colour space (no colour management,
like the reference's PIL decoding).

## The encoder's reshape flag is a correctness hazard

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

Evidence: [encoder](evidence/qwen3-vl-vision-encoder-2026-10-26.json) and
[image + text](evidence/qwen3-vl-image-text-2026-10-26.json); M5 Max (`h17c`), GPU-preferred,
release builds, shared host.

**Inputs.** Two deterministic synthetic scenes (448×448, 320×256) and three real photographs
from Wikimedia Commons at different aspect ratios, capped at 524,288 pixels (512 image tokens),
with source and licence recorded in the evidence:

| Photo | Original size | Licence |
| --- | --- | --- |
| `File:Bully-dog-2314909-1920.jpg` (portrait) | 1200×1412 | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) |
| `File:Bangkok fruit stall.jpg` (16:9) | 4032×2268 | Public domain, [`{{PD-self}}`](https://commons.wikimedia.org/wiki/Template:PD-self) |
| `File:'Byculla Fruit Market' near Byculla Bridge(east)..JPG` (4:3) | 810×608 | [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en) |

**Reference.** Upstream `Qwen3VLForConditionalGeneration` with its own processor and chat
template, float32 on the CPU (`Forge/language/reference_qwen3_vl.py`). The Forge eager port
(vision + text towers) matches it at cosine 1.0 with every argmax equal, and the host M-RoPE
positions equal the model's rope index, on all five cases.

**Encoder** (synthetic scenes, reference pixels): worst cosine 0.9999999 for the image embeddings
and every DeepStack level.

**Model parity** (reference pixel tensors through Kit's encoder and decoder; gate: forced-row
cosine ≥ 0.995, argmax equal unless a near-tie, greedy identical up to the reference's end):
all five cases pass — forced-row cosine ≥ 0.99996, no argmax differences, and every greedy
continuation identical to upstream, photos included.

**Kit end to end** (Kit's ImageIO decoding and preprocessing of the photo files; gate: equal
grids, pixel cosine ≥ 0.999 against the reference processor, forced-row cosine ≥ 0.98, tokens
compared and reported): pixel cosine 0.99997, 0.99999 and 0.99988; forced-row cosine ≥ 0.98977.
Greedy output is identical to upstream for the portrait and the fruit stall. For the 4:3 market
photo it agrees for 16 tokens, then diverges at a reference near-tie (top-1/top-2 probability gap
0.095): Kit answers "likely taken in **India**, specifically in a bustling **local market**",
upstream "likely taken in **Mumbai, India**". The pixels differ slightly because a different JPEG
decoder and resampler produce them, which is expected and recorded rather than hidden.

**Public API.** `VisionLanguageSession.open(bundle:)` on the registered bundle reproduces the
component-level Kit run token for token on the three photos, and closes on `unload()`.

## Limits and follow-ups

- **Native shapes grow without bound across image sizes — a memory risk.** Every distinct image
  token count adds a decoder `withImage` shape (its q is the image's token count) and every
  distinct patch count adds an encoder specialization (about 1.5–2.5 s the first time here; the
  engine measured roughly 1.5 GB resident per encoder shape, F-P3-29). Neither session recycles
  its native model, so an app showing many differently sized photos accumulates native shape
  caches for the session's lifetime. Until this is fixed, reopen the session (or keep image sizes
  to a few) in long-lived use. Plan for 0.0.3: snap smart resize to a few token buckets so shapes
  form a small fixed set, or recycle after K new image shapes.
- **No resource admission** for the encoder and the decoder yet (GPU admission only).
- **Coverage still missing:** a multi-image M-RoPE fixture (several images use the same path,
  but only single-image prompts are qualified), iPhone qualification (vision asset release after
  use, image-token caps, memory), and a quantized decoder (fp16 weights only today).
- The encoder stays loaded for the session's lifetime.
