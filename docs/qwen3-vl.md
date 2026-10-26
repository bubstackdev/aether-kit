# Qwen3-VL image input (in progress)

Goal for 0.0.2: image + text prompts to the language runtime with
`Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0, not gated; pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203`). The design follows the verified
aether-engine prototype (`docs/spec/11-vision.md` Part A); every piece is re-implemented
and qualified here against upstream Transformers.

## Status

| Piece | State |
| --- | --- |
| Vision encoder asset (`encode`) | Exported and natively qualified (below) |
| Host-side patch tables | `AetherVision.Qwen3VLVisionTables`, bit-exact with the Forge mirror |
| Image preprocessing (resize, normalize, patchify) | Not started in Swift; Forge `patchify` covers exact-multiple-of-32 images |
| Text decoder with M-RoPE and image splice (`withImage`) | Not started; will use the bucketed step contract |
| Public image + text API | Not started |

## Vision encoder

`Forge/language/export_qwen3_vl_vision.py` exports the tower as its own asset with one
function:

```
encode(pixels [n, 1536] fp16, pos_index [n, 4] i32, pos_weight [n, 4] f32, patch_rc [n, 2] i32)
  -> image_embeds [n/4, 2048], deepstack [3, n/4, 2048]
```

`n` is one image's patch count (dynamic, a multiple of 4), rows in merge-window order.
Weights are stored fp16 and every matmul, activation and the residual stream compute in
float32: the residual stream outgrows fp16, and a pure-fp16 tower loses its embeddings on
Core AI's cpu_only path (engine F-P3-21). The 777 MB asset must be specialized **without**
`expectFrequentReshapes`: with it, the GPU computes wrong outputs for this dynamic-shape
encoder (cosine 0.16–0.67 here, reproducing engine F-P3-26).

Qualification ([evidence](evidence/qwen3-vl-vision-encoder-2026-10-26.json), opt-in test
`qwen3VLVisionEncoderQualification`), three deterministic synthetic scenes (448×448,
320×256, 256×416; 80–196 image tokens), M5 Max:

- The Forge eager tower matches upstream `Qwen3VLVisionModel` (float32, CPU) at cosine
  1.0000 for the image embeddings and all three DeepStack levels, in float32 and with fp16
  storage.
- The asset's cpu_only self-check against the eager tower: worst cosine 1.0000.
- Native, GPU-preferred, specialized without reshapes, with Swift-computed patch tables:
  worst cosine 0.9999999 against the upstream reference over all outputs; first encode of
  each size about 2.5 s (specialization), repeats 35–109 ms on a shared host.

## Next steps

1. Export the text decoder with the bucketed contract plus Qwen3-VL's 3-axis rotary
   positions (`rope_positions` [3, q], separate from `cache_positions`, engine F-P3-1) and a
   `withImage` entry point that splices image rows at `<|image_pad|>` tokens and adds the
   DeepStack features after the first decoder layers (engine F-P3-2/3).
2. Swift preprocessing in `AetherVision`: smart resize (multiples of 32, pixel bounds from
   `preprocessor_config.json`), antialiased bicubic resampling, normalization, patchify, and
   M-RoPE positions; qualify against the reference processor.
3. A typed image + text session over the language runtime that encodes images once, releases
   the vision asset after use, and qualifies greedy image answers against upstream.
