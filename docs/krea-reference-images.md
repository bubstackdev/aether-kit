# Krea reference images (`visionTokens`)

Status: experimental, 2026-10-19. The host-side pieces, the scripted session path and the bundle contract are
implemented and covered by CPU tests. One native qualification of the encoders passed (below); denoising with
references has not been run with this package.

The Krea 2 Turbo realism `@3` export already contains the reference path: a `vision` asset (Qwen3-VL-4B's tower,
0.83 GB) and a second text-encoder function, `encodeWithImages`. Images reach the denoiser only as extra text rows,
and the denoiser and decoder graphs are unchanged. The design follows the Aether prototype's B15 specification.

## Contract

- The manifest must declare exactly `referenceImages {kind: visionTokens, encoderAsset: vision, maxImages 4, maxSide
  384, maxTotalTokens 576, label "Picture {n}: "}` plus the matching `vision` block, function roles and tensor
  bindings. Any other declaration is refused at `KreaAssets.open`. A bundle without `referenceImages` stays text-only.
- `KreaAssets.fingerprint` is unchanged for text-only use. The local bundle keeps the identity that Atelier's
  qualification recorded (`cf93b04e…`). `referenceFingerprint` additionally covers the vision asset's declarations and
  the reference contract. `verify()` still streams six assets; `verifyIncludingReferences()` streams seven.
- Denoiser text rows: up to 512 text-only, and up to 1,116 with references (the export's segment bound).

## Host preparation (`AetherKrea`, CPU)

- `KreaReferenceImage(encoded:)` decodes PNG/JPEG/HEIC from **memory only** (32 MiB, 8192-px bounds), applies EXIF
  orientation, and keeps exact RGB bytes for untagged or sRGB sources. It never takes a path and never writes temp files.
- `KreaReferencePreprocessing.prepare` fits the long side to 384, smart-resizes to multiples of 32, resizes with
  antialiased bicubic, then normalizes and patchifies into `[patches, 1536]` rows in merge-window order.
  Normalization is rounded once, `(v − 127.5)/127.5`, which reproduces the oracle's float32 tensor **bit for bit**.
  The prototype's affine `v·a + b` differs at some byte values.
- `KreaPatchTables` holds the bilinear position-table neighbours, weights and patch coordinates for the tower.
- `KreaReferencePrompt` handles labelled image parts, image-pad expansion, text-only truncation (images are never
  cut), M-RoPE positions `[3, L]`, and the image-token and row limits, all checked before any model loads.
  `KreaTokenizer.prepare(_:references:)` renders and tokenizes the prompt.

## Session (`AetherKreaCoreAI`)

`KreaGenerationSession.generate(configuration:referencePrompt:references:…)` and `prepare(referencePrompt:references:)`
validate the references against the prompt (digest, grid, count) and the bundle's support before any load. Then:
1. in balanced residency, release the denoisers;
2. load the vision asset (no reshapes) and encode each image once, with cancellation checked between images;
3. release the vision encoder (unless resident);
4. run `encodeWithImages` with the concatenated `image_embeds [n, 2560]` and level-major `deepstack [3, n, 2560]`;
5. release the text encoder (unless resident).

Signatures are validated before each call. Failures and cancellation always discard the encoder involved, and the
operation slot is held until in-flight native work returns. There is no new public event case, so existing exhaustive
switches over `KreaGenerationEvent` are unaffected: `conditioningReady` follows the reference phase.

## Evidence

CPU (`swift test --triple arm64-apple-macosx27.0`; native suites stay env-gated):
- smart resize against 100+ reference-processor sizes (`grids.json`);
- patch tables against forge for every recorded grid;
- the oracle fixture `krea2-reference` `c4f6229db52d11b9`, 6 cases with 1–4 images: pixel tensors **bit-exact** (SHA-256),
  token expansion and M-RoPE positions **exact**;
- scripted session ordering, residency, refusal-before-load, failure, and cancellation before and during vision
  encoding.

Opt-in with local files and no model load (`AETHER_KREA_BUNDLE`, `AETHER_KREA_REFERENCE_FIXTURES`):
- the tokenizer's reference prompts equal the oracle ids and positions for all 6 cases;
- the local bundle declares references, keeps its text-only fingerprint and has a separate reference fingerprint.

Native (M5 Max, 2026-10-19, one approved run under the shared GPU lock):
- `AETHER_NATIVE_REFERENCE_QUALIFICATION=1` compares the conditioning rows with the oracle's `text.f32`. **Passed 6/6**:
  mean row cosine 0.99998945–0.9999945, worst row 0.9987659–0.9997663 (bars ≥ 0.999 and ≥ 0.99).
- The whole run, with compile-cold encoders, took 192 s wall. The test process's sampled peak footprint was 4.5 GB.
  The compiled cache entries were 21 GB (text encoder) and 0.8 GB (vision).
- Generation with references (denoise and decode), its peak and timing, in-call cancellation and image parity with the
  lab's frozen `labrender` remain unmeasured.
