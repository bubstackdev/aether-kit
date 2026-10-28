# Qwen3-VL coverage: multi-image prompts, rotated photos and alpha (0.0.3 plan)

Status: plan, 2026-10-27. Rotated photos ([evidence](evidence/qwen3-vl-rotated-photos-2026-10-27.json)) and alpha
([evidence](evidence/qwen3-vl-alpha-2026-10-27.json); the lead chose to match upstream) are done. Multi-image is done
on the VL-2B ([evidence](evidence/qwen3-vl-multi-image-2026-10-27.json)); the 4B runs and the session's summed
image-token check (step 3, which needs `LanguageResourceAdmission`) remain. These are the three Qwen3-VL coverage items in the
[0.0.3 roadmap](roadmap-0.0.3.md#coverage-and-iphone). Each is CPU work first, then one native run under the GPU lock
against a new upstream reference.

What upstream does is the bar for all three. Transformers' `load_image`, the path an application using the reference
processor takes, applies `PIL.ImageOps.exif_transpose` and then `convert("RGB")`. Kit's references so far decoded with
`Image.open(...).convert("RGB")` and no transpose, which was correct only because every reference photo carries
orientation 1 or none ([qwen3-vl.md](qwen3-vl.md#qualification)).

## 1. Multi-image M-RoPE

**Today.** `LanguagePromptPlan` already expands several placeholders, gives each image its own span, and advances the
rotary position by max(t, h, w) after each image ([qwen3-vl.md](qwen3-vl.md#pieces)). The one-function decoder prefills
in uniform chunks, so an image boundary may fall anywhere, including between two images in one chunk. None of this
is checked against upstream with more than one image.

**Plan:**
1. **Reference.** Extend `Forge/language/reference_qwen3_vl.py` so a case can name several images
   (`NAME=IMAGE1|IMAGE2::QUESTION`), with the template rendering one placeholder per image. It records the processor's
   ids, each image's grid, and the model's own rope index for the whole prompt. Three cases on the VL-2B:
   - two photos at different grids (portrait 48×36 and the 16:9 stall 32×56);
   - three images, one of them a synthetic scene;
   - two images with text between them, so a text run sits between two spans.

   The 2B's float32 reference stayed near 11 GB for one case and peaked at 17.4 GB for five. Three cases need the
   lead's clearance as a borderline heavy job, or a run split into one case per process.
2. **CPU test.** `LanguageVisionPromptTests` gains a fixture with those cases' ids, grids and rope index. The plan's
   expansion and positions must equal upstream's for every token.
3. **Session.** The VL session's per-request image-token check (`LanguageResourceAdmission`) sums over images; the
   registration's `maximumImageTokens` stays per image. The guide gets one line saying so.
4. **Native.** `qwen3VLImageTextQualification` on the three cases, on the 2B and then the 4B, under the existing gates
   (forced-row cosine ≥ 0.995, argmax equal unless a near-tie, greedy identical to the reference end). Plus
   `languageOverlappedPrefillMatchesOtherPlans` on one two-image prompt, so default chunks, 512-token chunks and 1-token
   steps agree with two spans in play.

**Risk.** Upstream's rope index for several images is the part most likely to differ: in particular how the position
advances between images, and whether video-style temporal offsets appear. The CPU fixture finds that before any GPU
time.

## 2. Rotated photos

**Today.** `VisionRGBImage(contentsOf:)` applies the EXIF orientation through ImageIO's thumbnail path, and grid snapping
uses the displayed aspect ratio. `VisionRGBImageTests` checks every orientation on generated fixtures. No model parity
run uses a rotated file.

**Plan:**
1. **Fixtures.** From the three committed-provenance photos, write rotated variants whose stored pixels are rotated
   and whose EXIF orientation (6, 8 and 3) undoes it, so the displayed image is the original. They are written with
   ImageIO (`kCGImagePropertyOrientation`) by a script like `Scripts/write-kit-decoded-photos.swift`, as PNG (lossless)
   so the decoder difference stays out of it. Hashes and provenance are recorded.
2. **CPU checks:**
   - Kit's decode of each rotated variant equals its decode of the original, byte for byte. Since the stored pixels
     are an exact rotation, that compares the orientation transform alone.
   - The thumbnail path must not resample: a full-size thumbnail must equal a lossless rotation of the full decode.
     The test proves it rather than assuming it.
   - The snapped grid of a rotated portrait is the portrait's grid, not the landscape one.
3. **Reference.** `reference_qwen3_vl.py` gains `--exif-transpose`, decoding as `load_image` does (transpose, then
   `convert("RGB")`), and runs on the rotated variants.
4. **Native.** The same-decode Kit run on the rotated variants, gated at the model-parity bar. Its tokens must equal the
   unrotated originals' run, since the displayed images are identical.

**Risk.** If ImageIO's thumbnail path resamples, the decode differs slightly from a plain rotation. The fix would be to
decode at full size and rotate the bitmap directly (with vImage), which is the CPU test's other branch.

## 3. Alpha

**Today.** Kit draws every image into an opaque RGB context, so transparent pixels are composited over black. Upstream
`convert("RGB")` drops the alpha channel and keeps each pixel's stored colour. Opaque images are unaffected; transparent
regions differ.

**Decision needed (lead).** Match upstream by default, so parity holds for every input, or keep compositing and
document it. Proposed: match upstream by default and offer compositing as an explicit option, since an app that wants
transparent regions shown as a colour can say which colour:

```swift
public enum VisionAlphaPolicy: Sendable { case dropAlpha, composite(over: (UInt8, UInt8, UInt8)) }   // default .dropAlpha
```

**Plan:**
1. **Decoding.** For `.dropAlpha`, read the decoded image's own bytes instead of drawing them.
   - When the `CGImage` is 8-bit RGBA or ARGB with non-premultiplied alpha (`.last` or `.first`), the RGB bytes are
     taken directly from its data provider, in its colour space as today.
   - Premultiplied sources (`.premultipliedLast/First`) have already lost the stored colour of transparent pixels.
     They are unpremultiplied where alpha > 0, and alpha-0 pixels become black. That matches PIL only for formats PIL
     also premultiplies, so the guide states the limit.
   - Other layouts (16-bit, indexed, CMYK) keep today's drawing path, composited over black, and are documented.
2. **CPU tests:**
   - Generated PNGs with known stored colours under alpha 0, 128 and 255, in RGBA, ARGB and premultiplied layouts.
     Kit's bytes must equal PIL's `convert("RGB")` bytes, recorded as a fixture by a Forge script so no Python runs in
     the Swift suite.
   - The composite policy's output equals the reference compositing formula.
3. **Native.** One synthetic transparent scene through the same-decode Kit run against an upstream reference of the
   same file, gated at the model-parity bar. Today's composite path is reported beside it, showing the difference the
   change removes.
4. **Krea.** `KreaReferenceImage` also drops alpha onto black, but it follows the Aether prototype's oracle, not
   Transformers. It stays as it is unless Krea's own reference says otherwise; the guide notes that the two differ.

**Risk.** ImageIO may hand back premultiplied images even for straight-alpha PNGs. The CPU test establishes which
layouts arrive non-premultiplied, and the policy's documented limit follows from it.

## Order and cost

1. **Rotated photos.** CPU only, apart from one 2B reference and one native run. It needs no API change.
2. **Alpha.** Needs the policy decision; then CPU work and one native run.
3. **Multi-image.** The largest: a reference extension, a fixture, and native runs on both models. The reference needs
   the lead's clearance for its CPU peak.

All native runs use the one-function bundles under the GPU lock. All references use the pinned checkpoints already on
disk; nothing is downloaded.
