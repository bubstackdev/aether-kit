# FLUX.2 [klein] 4B: text-to-image and multi-reference generation

Status: experimental, 2026-10-26 (0.0.2). The products, local bundle, independent references and opt-in native
parity and lifecycle gates are in place. Results and their scope are in [Verification](#verification). There is no iOS execution,
benchmark protocol or published bundle yet.

## Model and license

- Checkpoint: `black-forest-labs/FLUX.2-klein-4B` at `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Not gated; the model
  card declares `license: apache-2.0` and the repository's `LICENSE.md` is the Apache License 2.0 text
  (sha256 `ca02bc51…a4af1fe`). The finalized bundle keeps that file beside the assets.
- Components (from the checkpoint configs): a Qwen3-4B text encoder cut after layer 27 (taps 9/18/27 concatenated,
  7,680 wide), a 5 double + 20 single block transformer (hidden 3,072, 4-axis RoPE θ 2000), and the 32-channel VAE with
  2×2 packing (128 channels per token, one token per 16×16 pixels). Step-distilled: four steps, no guidance.
- The design follows the Aether prototype's spec 11 Part B (B2–B4, B12, B14) and its recipe
  `flux2-klein-4b-reference.toml@1`. AetherKit owns its own implementation, references and qualification.

## Products

| Product | Contents |
| --- | --- |
| `AetherFlux2` | `Flux2Assets` (bundle validation, provenance pins, fingerprints, portable installation), `Flux2Configuration`, `Flux2Schedule` (empirical μ), `Flux2Rotary`, `Flux2Prompt`/`Flux2Conditioning`, `Flux2ReferenceImage` and `Flux2ReferencePolicy`. No Core AI dependency. |
| `AetherFlux2Tokenizers` | `Flux2Tokenizer`: the bundle's Qwen3 tokenizer through swift-transformers; never downloads. |
| `AetherFlux2CoreAI` | `Flux2GenerationSession`, `Flux2ResourceAdmission`. |

Shared pieces added for this work and usable by other families: `AetherKit.SeededGaussianNoise` (Krea's noise now
delegates to it, unchanged values), `AetherAssets.DeclaredModelAsset` and `BundleFiles` (declared Core AI asset
identity, confined paths, streaming hashes, portable entries), and the package-internal `StagedModel` protocol with a
budgeted decorator in `AetherCoreAI`. Krea's own types are unchanged.

## Bundle

One bundle holds both modes: `textEncoder` (`encode`), `denoiser0…2` (`segment`, blocks 0–8, 8–16, 16–25),
`vaeDecoder` (`decode`) and the optional `vaeEncoder` (`encode`) that enables references.

| Asset | Inputs → outputs | Weights |
| --- | --- | --- |
| textEncoder | `input_ids`, `position_ids`, `attention_mask` int32 `[1,512]` → `text_embeds` fp16 `[512,7680]` | int8 per-channel, 3.12 GB |
| denoiser0 | `latents` fp16 `[N,128]`, `text_embeds`, `sigma` f32 `[1]`, `rope_text` f32 `[512,2,128]`, `rope_image` f32 `[N,2,128]` → `image_1`, `text_1`, `cond` f32 | int8 (fp16 `proj_out`), 1.90 GB |
| denoiser1 | `image_1`, `text_1`, `cond`, ropes → `image_2`, `text_2` | 1.28 GB |
| denoiser2 | `image_2`, `text_2`, `cond`, ropes → `velocity` fp16 `[N,128]` | 1.44 GB |
| vaeDecoder | `latents` fp16 `[h,w,128]` (still batch-normalized) → `image` f32 `[3,16h,16w]` | fp16 weights, f32 compute, 102 MB |
| vaeEncoder | `reference_pixels` f32 `[3,H,W]` → `reference_latents` f32 `[H/16,W/16,128]` (posterior mode, packed, normalized) | fp16 weights, f32 compute, 71 MB |

`Flux2Assets.open` refuses any other recipe: schema/driver/target, the exact empirical shift constants, rotary axes and
θ, prompt template/length/padding/pad token, segment list, function roles, tensor bindings, the reference contract
(`packedLatents`, stride 10, bounded counts) and the source repository and revision. Provenance pins every file by
SHA-256; opening checks model declarations and metadata against them, and `verify()` streams every model (about 8 GB).
`fingerprint` identifies the text-to-image contract; `referenceFingerprint` adds the encoder and reference contract.
`installationManifest()`/`open(in:)` and `Flux2GenerationSession(installed:)` follow the Krea installation contract.

### Producing the bundle

The prototype's Forge exports the components; a kit-side script adds the target and provenance fields:

```sh
cd <aether-engine>/forge
~/.cache/aether-kit/with-gpu-lock.sh "flux2 export" uv run --frozen --project . python -m aether_forge export \
  --recipe ../recipes/flux2-klein-4b-reference.toml --out ~/.cache/huggingface/aether-kit/flux2-klein-4b/export-2026-10-26 \
  --cache ~/Library/Caches/aether/forge-checkpoints --repo-root <aether-engine>
python3 Scripts/finalize-flux2-bundle.py ~/.cache/huggingface/aether-kit/flux2-klein-4b/export-2026-10-26 \
  <checkpoint snapshot> ~/.cache/huggingface/aether-kit/flux2-klein-4b/macos-any-gpu
```

The finalizer streams every `main.mlirb`, refuses one that differs from its `main.hash`, and writes `aether.json` (Forge
fragment + target + per-asset placement + provenance file hashes + license) and `metadata.json`. Core AI exports are not
byte-reproducible, so a re-export changes model hashes and fingerprints.

## Host math (`AetherFlux2`)

- **Prompt:** Qwen3 chat template with thinking off, right-padded with `<|endoftext|>` (151643) to exactly 512 tokens
  plus mask; truncation keeps the prefix. All 512 encoder rows condition the denoiser.
- **Schedule:** μ from the pipeline's empirical rule of the *generated* token count and step count (two lines at 10
  and 200 steps, high line only above 4,300 tokens), sigmas `linspace(1, 1/steps, steps)` in Double → Float, shifted by
  `e/(e + 1/σ − 1)` in Float, final 0; denoiser `sigma` = `Float(σ·1000)/1000`. Euler rounds the product before the sum.
- **Rotary:** ids `(t,row,col,index)`; text `(0,0,0,i)`, generated grid `(0,r,c,0)`, reference k `(10(k+1),r,c,0)`; 32
  dims per axis, split-halves `[n,2,128]` tables computed in Double.
- **Noise:** `SeededGaussianNoise` read as `[tokens,128]`. Seeds do not reproduce PyTorch's generator; parity runs feed
  the reference noise.
- **References:** planar `[3,H,W]` on [−1,1], sides multiples of 16. `Flux2ReferenceImage(image:maxSide:)` fits a
  `CGImage` inside `maxSide` without upscaling, floors to multiples of 16 and resamples with Core Graphics' high-quality
  filter (not PIL Lanczos; exact replay uses `init(pixels:)`). `Flux2ReferencePolicy` refuses more than `maxImages`
  (2), a side above `maxSide` (512) or more than `maxTotalTokens` (4,096) generated + reference tokens before any load.

## Session (`AetherFlux2CoreAI`)

```swift
import AetherFlux2
import AetherFlux2CoreAI
import AetherFlux2Tokenizers

let assets = try Flux2Assets.open(at: bundleURL)
let tokenizer = try await Flux2Tokenizer.open(assets: assets)
let session = Flux2GenerationSession(assets: assets)   // preferGPU, balanced
do {
    // Text to image.
    let image = try await session.generate(configuration: try Flux2Configuration(width: 512, height: 512, seed: 7),
                                           prompt: try tokenizer.prepare("A lighthouse at dusk, watercolor"))
    try image.savePNG(to: outputURL)

    // Multi-reference conditioning: encode once, reuse across seeds and prompts.
    let references = try await session.prepare(
        references: [try Flux2ReferenceImage(image: fox, maxSide: 512), try Flux2ReferenceImage(image: lighthouse, maxSide: 512)],
        generatedTokens: 768)
    let prompt = try await session.prepare(prompt: try tokenizer.prepare("The fox from picture one in front of the lighthouse"))
    for seed: UInt64 in [1, 2] {
        let edited = try await session.generate(configuration: try Flux2Configuration(width: 512, height: 384, seed: seed),
                                                preparedPrompt: prompt, references: references)
        try edited.savePNG(to: outputDirectory.appendingPathComponent("edit-\(seed).png"))
    }
    await session.unload()
} catch {
    await session.unload()
    throw error
}
```

- **Pipeline:** text encoding → optional reference encoding (one call per image) → three segments per step with
  carries passed by name → CPU Euler on generated tokens → VAE decode → `GeneratedImage` (float32 planar RGB; PNG uses
  the same `x/2 + 0.5` mapping as the reference post-processing). Reference tokens and rotary rows are appended to
  every denoiser call, unchanged; their velocities are discarded; μ uses generated tokens only.
- **Residency:** `balanced` releases denoisers before any encoder loads and the text encoder after encoding;
  `resident` keeps the text encoder and denoisers. Both release the reference encoder after each preparation and
  recycle denoisers when the total token count changes and the decoder when the output size changes, so shape
  specializations do not accumulate (prototype F-P3-75). Placement per asset: text encoder and denoisers specialize with
  frequent reshapes, decoder and reference encoder without (prototype F-P3-48/68).
- **Lifecycle:** one operation at a time (`sessionBusy` otherwise); cancellation is cooperative between native calls
  and waits for the submitted call; failed or cancelled encoders are discarded; partial denoiser loads are closed;
  `unload()` rejects new work, waits for the active operation and closes permanently.
- **Prepared values:** `Flux2PreparedPrompt` (7.5 MiB fp16 rows + bundle fingerprint) and `Flux2PreparedReferences`
  (float32 packed latents + reference fingerprint) hold no native models; another bundle refuses them before loading.
- **Admission:** optional `Flux2ResourceAdmission` reserves caller-qualified estimates (operation storage, per-model
  residency keyed by stream + metadata hash, per-call scratch) on a shared `ResourceCoordinator`; unknown models or
  larger token counts refuse before any stage is released or loaded. These are estimates, not measured limits.

All tensors copy through host memory between calls; there is no device-resident loop and no performance claim.

## Verification

### CPU (default suite)

`Tests/AetherFlux2Tests` (20 tests, weight-free): schedule against the reference pipeline's observed sigmas and the μ
rule, separately rounded Euler, rotary values and reference offsets, prompt padding/truncation, configuration bounds,
reference policy and CGImage preparation, bundle acceptance/refusal for 11 recipe deviations, provenance and mutation,
installation round trip, segment-contract refusal, and scripted-session ordering: balanced/resident residency, reference
appending and slicing, prepared reuse and shape recycling, refusal before load, busy/cancellation/recovery, partial-load
failure, unload during work, and resource admission accounting.

Opt-in with local files only (`AETHER_FLUX2_FIXTURES`, `AETHER_FLUX2_REFERENCE_FIXTURES`, `AETHER_FLUX2_BUNDLE`):
`samplerArithmeticMatchesTheIndependentReference` and `bundleTokenizerMatchesReferenceTokenIDs`.

### Independent references

| Fixture | Producer | Cases |
| --- | --- | --- |
| `6a8a53d8f3b30817` (text to image) | Forge `fixtures` stage: unmodified Diffusers 0.37.1 `Flux2KleinPipeline`, float32 on MPS, float32 text encoder on CPU | 10 prompts, 512² to 1344×768, incl. French/Japanese text |
| `079a8fba9d596a3b` (reference images) | `Scripts/prepare-flux2-reference-fixtures.py`: the same pipeline with `image=[…]` plus hooks; references are Lanczos-resized oracle outputs | 2 cases, one and two references |

Both record token ids/mask, float32 text rows, packed initial noise, per-step latents/timestep/velocity, final latent
and float32 image; the reference fixture also records encoder input pixels, the appended reference tokens and the
transformer position ids. Neither uses the Core AI bundle or AetherKit code.

### CPU checks against the references (2026-10-26)

With the fixture paths set, the weight-free tests pass. The bundle tokenizer reproduces all 10 reference token-id and
mask rows exactly, including the French/Japanese prompt. Timesteps are exact for all 12 cases, μ included (the
reference cases confirm μ uses generated tokens only). CPU Euler replay of the reference's own velocities reproduces
every following step input and the final latent bit for bit. Reference position ids match the transformer's recorded
`img_ids` exactly.

### Native parity (M5 Max, macOS 27.0.1, `preferGPU`, 2026-10-26)

Both runs passed every check. They used the finalized int8 bundle (fingerprint `f28ef0e8…`, reference fingerprint
`8fb2df90…`) after a full stream verification.

| Check (bar) | 512² class + references (7 cases, 68 checks) | 768²–1344×768 (5 cases, 45 checks) |
| --- | --- | --- |
| Token ids and mask (exact) | 7/7 exact | 5/5 exact |
| Text real-row cosine (≥ 0.999) | min 0.999972 | min 0.999972 |
| Teacher-forced velocity cosine per step (≥ 0.999) | min 0.99945 (two-reference case), text-only min 0.99962 | min 0.99924 (1024×768, step 1) |
| VAE decoder PSNR on the reference final latent (≥ 35 dB) | min 87.97 dB | min 87.69 dB |
| Reference-encoder latent cosine (≥ 0.999) | 3/3 images ≥ 0.9999999 | — |
| Image path SSIM through the public session (≥ 0.94) | min 0.9602 (`cafe-512`); references 0.9984 and 0.9881 | min 0.9618 (1024×576); 1024² 0.9642 |
| Whole pipeline SSIM (recorded) | 0.812–0.999 | 0.839–0.927 |

The reference fixture's own bf16 pipeline SSIM against its float32 image is 0.865–0.999 across the ten
text-to-image cases. The whole-pipeline numbers sit in that range, as expected: the padding rows move the image more
than conversion error does (prototype F-P3-60). The PNGs for every case, provided-conditioning and from-prompt, are
under `~/.cache/huggingface/aether-kit/flux2-klein-4b/native-parity-*-2026-10-26/`. Both editing cases were visually
inspected: the fox becomes a watercolor, and the two-reference prompt places that fox in front of the lighthouse.

Timing and memory, with no benchmark protocol, on a shared host, with the Core AI specialization cache warm from the export:

- **Component steps:** about 0.21 s per denoiser step at 512², 0.25 s with 256–448 reference tokens, 0.55–0.62 s at
  768² and 1024×576, about 0.94 s at 1024² and 1344×768. Text encoding takes 0.05 s. Decoding takes 0.13 s at 512²
  and 0.6–1.1 s at the larger sizes.
- **First use:** a new shape's first denoiser step took 1.2 s at 512² and 7.3 s at 1024².
- **End to end** (`generate`, including model loads and recycling): 1.0–2.9 s at 512² and 576×448 with provided
  conditioning, 2.5–3.8 s from a prompt (balanced mode reloads denoisers after encoding), 3.4–4.6 s with references,
  and 6.0–6.6 s at 1024² and 1344×768.
- **Memory:** the test process's lifetime peak footprint was 4.04 GB for the 512² run. For the large run it was 14.6
  GB, reached in the component VAE phase, which decoded five sizes in one decoder session. That is the accumulation
  the session avoids by recycling the decoder per size (prototype F-P3-75), so it is not a session peak. Per-stage
  session peaks were not isolated.
- **GPU-lock time** for all FLUX.2 work: about 16.5 minutes (export 1.5, text-to-image fixtures 11, reference
  fixtures 0.5, native runs 1 and 1.5).

Records: [qualification summary](evidence/flux2-klein-qualification-2026-10-26.json) (checkpoint file hashes, bundle
identity, fixtures, minima, phases, output PNG hashes and lock windows), the
[512² report](evidence/flux2-klein-native-parity-2026-10-26.json) and
[large-size report](evidence/flux2-klein-native-parity-large-2026-10-26.json) with their logs, the
[bundle file pins](evidence/flux2-klein-bundle-finalize-2026-10-26.json), the
[Forge manifest fragment](evidence/flux2-klein-forge-manifest-fragment-2026-10-26.json) and
[export log](evidence/flux2-klein-forge-export-log-2026-10-26.jsonl), and both fixture documents with their case lists.

Run the native gate (one suite at a time, under the shared lock):

```sh
AETHER_NATIVE_FLUX2_QUALIFICATION=1 \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/macos-any-gpu \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-2026-10-26/fixtures/6a8a53d8f3b30817 \
AETHER_FLUX2_REFERENCE_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-reference-2026-10-26/fixtures/079a8fba9d596a3b \
AETHER_FLUX2_OUTPUT=/new/output/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 native" swift test -c release --filter flux2NativeParity
```

`AETHER_FLUX2_CASES` selects text-to-image cases (default: the five 512²-class cases; `all` for all ten).

### Post-rebase parity (2026-10-26)

After rebasing onto the 0.0.1 release (origin/main `128573d`), the parity suite ran again with all 12 cases in one
process (`AETHER_FLUX2_CASES=all`). It passed **113/113 checks**, and every check value for the seven 512²-class
cases equals the pre-rebase report bit for bit. The minimums are unchanged: velocity 0.99924, image-path SSIM 0.9602,
decoder 87.7 dB. The process's lifetime peak was 17.3 GB, reached while the component VAE phase decoded twelve sizes
in one decoder session (see the memory note above). See the
[report](evidence/flux2-klein-native-parity-rebased-2026-10-26.json) and
[log](evidence/flux2-klein-native-parity-rebased-2026-10-26.log).

### Native lifecycle qualification (2026-10-26)

`flux2NativeLifecycleQualification` (`AETHER_NATIVE_FLUX2_LIFECYCLE=1`, `AETHER_FLUX2_LIFECYCLE_OUTPUT` set to a new
directory, plus the bundle and fixture paths) is modeled on Krea's native session suite. A tracked factory opens real
Core AI sessions. Backend observers bracket the awaited native `function.run`, and an event journal is written as each
event happens. Nothing gates or delays native work. It uses the `fox-512` reference conditioning and noise as a
baseline, and passed **44/44 checks** in 27.6 s: 176 native calls, 34 model instances (each closed exactly once), 32
function loads and 144 reuses.

- **Repeat:** an exact pixel repeat with no model opens or closes and no native function loads (13 calls).
- **Cancellation during a native denoiser call and during the native decoder call:**
  - an overlapping `prepare` is rejected with `sessionBusy`;
  - the observed call returns before cancellation does;
  - no native call starts after the request;
  - the state returns to ready;
  - recovery reproduces the baseline pixels exactly while reusing every model.
- **Balanced prompt:** the denoisers close before the text encoder opens, and the encoder closes before the denoisers
  reload.
- **References:** the VAE encoder opens once and is released; the denoisers are recycled for the new token count; SSIM
  against the reference is 0.998.
- **Unload during an active denoiser call:**
  - new work is rejected while closing;
  - the active generation completes with the baseline pixels;
  - all four releases follow the final native return;
  - work is rejected after close, and the owned output stays valid.
- **Resident reuse across prompts and sizes:**
  - a second prompt reuses the text encoder and denoisers;
  - a new size keeps the encoder and recycles exactly the denoisers and decoder;
  - returning to the baseline reproduces its pixels exactly.
- **Admission:** these use logical estimates, not measured memory.
  - During an in-flight call the coordinator holds operation storage, three denoiser residencies and one call's
    scratch; after completion it holds only model residency.
  - An oversized request, and a reference encoder with no estimate, are refused before any stage is released or loaded.
  - Unload returns the ledger to zero.
  - With capacity for only two segments, generation fails with `insufficientCapacity`, the opened segments are
    released, and the ledger returns to zero.

The first attempt passed 43/44 checks. Its unload-ordering check also counted the shape-recycle closes at the start of
that generation, while its own journal shows every unload release after the final native return. The check now counts
only releases after an explicit `unloadRequested` marker, and no library code changed. That attempt is preserved as
[attempt 1](evidence/flux2-klein-native-lifecycle-attempt1-2026-10-26.json). See the
[lifecycle record](evidence/flux2-klein-lifecycle-qualification-2026-10-26.json),
[report](evidence/flux2-klein-native-lifecycle-2026-10-26.json),
[event journal](evidence/flux2-klein-native-lifecycle-events-2026-10-26.jsonl) and
[log](evidence/flux2-klein-native-lifecycle-2026-10-26.log).

These additional GPU-lock jobs took about 3.5 minutes: parity 2.3, lifecycle 0.5 + 0.5.

### Not yet qualified

- Native cancellation of the text encoder and the reference encoder. Their discard-on-failure paths are covered by CPU
  tests only.
- iOS: no build or run has been qualified. The prototype ran this recipe on iPhone with staged residency.
- Prompt adherence (T3) and a subjective review across more prompts and seeds.
- References prepared from a `CGImage`: parity used the reference's exact encoder pixels.
- Benchmarking under a quiet, controlled protocol, and per-stage memory peaks.
- Device-resident carries, or a GPU sampling loop.
