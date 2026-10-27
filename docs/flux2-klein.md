# FLUX.2 [klein] 4B: text-to-image and multi-reference generation

Status: experimental, 2026-10-27 (0.0.2). The products, local bundle, independent float32 Diffusers references on CPU and opt-in native
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
SHA-256; opening checks model declarations and metadata against them. `verify()` streams every model (about 8 GB) on
concurrent threads, with each file confined to its asset directory. It reports the first failing asset in serial order
and stops when the calling task is cancelled.
Per-asset placement is read from the manifest: `diffusion.assetReshapes` overrides the bundle-wide
`expectFrequentReshapes`. Undeclared assets are refused.
`fingerprint` identifies the text-to-image contract; `referenceFingerprint` adds the encoder and reference contract.
`installationManifest()`/`open(in:)` and `Flux2GenerationSession(installed:)` follow the Krea installation contract.

### Producing the bundle

`Forge/flux2/` exports the bundle from the pinned snapshot in Forge's locked environment. It is written from upstream
Diffusers 0.37.1 and Transformers 4.57.3 against this contract and reproduces the 2026-10-26 bundle's recipe; the
superseded prototype exporter that produced that bundle is no longer used. Export is CPU-only (Python loads assets
`cpu_only` for its self-check), takes about 1.5 minutes on an M5 Max and peaks at an 18.7 GB footprint:

```sh
uv sync --frozen --project Forge
C=~/Library/Caches/aether/forge-checkpoints/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
O=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27
Forge/.venv/bin/python -I Forge/flux2/check_flux2_graphs.py --workdir $O/graph-check-real --checkpoint $C   # optional, ~28 GB RSS
Forge/.venv/bin/python -I Forge/flux2/export_flux2.py --checkpoint $C --output $O/export
python3 Scripts/finalize-flux2-bundle.py $O/export $C $O/bundle
Forge/.venv/bin/python -I Forge/flux2/check_flux2_contract.py ~/.cache/huggingface/aether-kit/flux2-klein-4b/macos-any-gpu $O/bundle
```

- `flux2_graphs.py` holds the six graphs. The text encoder is Qwen3 layers 0–26 with the pipeline's taps, fp16, q/k/v
  in one projection. The denoiser segments keep the residual streams, LayerNorms, modulation and rotary products in
  float32 and run matmuls and attention in fp16; every query/key projection row and q/k norm gain is permuted within its
  head so Kit's split-halves rotary tables give Diffusers' interleaved rotation. `cond` holds the 17 modulation rows
  (image 6, text 6, single 3, output scale and shift). Both VAE graphs store fp16 weights and compute in float32.
- `export_flux2.py` verifies every checkpoint file it reads against the pins recorded in the qualification record,
  applies weight-only int8 per-channel PTQ with coreai-opt (205 modules; the 20 single-block `proj_out` stay fp16, as
  `quantization.exceptions` declares), converts, and runs each asset `cpu_only` against its eager fp16 module. Core AI's
  `cpu_only` runtime segfaults on the VAE decoder (the 2026-10-26 asset too), so that one is qualified natively only.
  It writes `bundle/`, `manifest-fragment.json`, `apple-metadata.json`, `export-report.json` and `forge-log.jsonl`.
- `check_flux2_graphs.py` compares the graphs in float32 with the unmodified upstream modules, on a random-weight
  snapshot it writes or on the real one. On the real weights the text encoder and both VAE graphs are exact and the
  chained denoiser segments reach cosine 1.0 (max error 7·10⁻⁵ at output scale 3) against `Flux2Transformer2DModel`,
  with and without a reference grid.
- `check_flux2_contract.py` diffs two finalized bundles: `metadata.json`, every `aether.json` field except the ones a
  re-export changes (file pins, export date, tool, asset bytes), each function's graph type and argument/result
  attributes, the OS compatibility version, storage and compute scalar types, and the tokenizer files.
- `record_flux2_evidence.py` writes the evidence record with parity side by side against a shipped bundle's report.

The finalizer streams every `main.mlirb`, refuses one that differs from its `main.hash`, and writes `aether.json` (Forge
fragment + target + per-asset placement + provenance file hashes + license) and `metadata.json`. Core AI exports are not
byte-reproducible, so a re-export changes model hashes and fingerprints.

**The 2026-10-27 Forge export** (`forge-export-2026-10-27/bundle`, fingerprint `a48aaf07…`) has the shipped bundle's
contract exactly: every function signature, the segment split, `metadata.json`, all 111 compared `aether.json` fields
(the quantization notes included) and the tokenizer files. The internal graphs differ: LayerNorm and GroupNorm lower to
Core AI composites, the VAE decoder upsamples by nearest-neighbour interpolation where the prototype used conv + pixel
shuffle, and the q/k norm feeds the rotary product in float32 without an fp16 round trip. `Flux2Assets.open` and
`verify()` accept it, the `AetherFlux2Tests` suite passes 28/28 with the bundle and CPU fixture paths set, and against
the CPU float32 references it passes native parity (113/113) and lifecycle (44/44, the same 176 native calls, 34
model instances, 32 loads and 144 reuses):

| Check (bar), CPU references | Shipped bundle (`f28ef0e8…`) | Forge export (`a48aaf07…`) |
| --- | --- | --- |
| Token ids and mask (exact) | 12/12 | 12/12 |
| Text real-row cosine (≥ 0.999) | min 0.999972 | min 0.999972 |
| Teacher-forced velocity cosine per step (≥ 0.999) | min 0.999239 (1024×768) | min 0.999247 (1024×768) |
| VAE decoder PSNR (≥ 35 dB) | min 87.684 dB | min 87.684 dB |
| Reference-encoder latent cosine (≥ 0.999) | 3/3 ≥ 0.9999999 | 3/3 ≥ 0.9999999 |
| Image path SSIM (≥ 0.94) | min 0.96016 (`cafe-512`) | min 0.95953 (`cafe-512`) |
| Whole pipeline SSIM (recorded) | 0.812–0.999 | 0.832–0.999 |

Per check, image-path SSIM is within 0.0008 of the shipped bundle's, velocity cosine within 3·10⁻⁵ and decoder PSNR
within 0.001 dB. The ungated whole-pipeline SSIM moves by up to 0.06 (`fox-lighthouse-ref2` 0.812 → 0.871): that path
conditions on the native text encoder's padding rows, which move the image more than conversion error does (see the
MPS-reference section below). The parity
process peaked at a 17.3 GB lifetime footprint (component VAE phase, as before). The shipped bundle stays the default
(`macos-any-gpu`). See the [record](evidence/flux2-forge-export-2026-10-27.json) (commands, versions, file hashes,
every check side by side), the [contract diff](evidence/flux2-forge-export-contract-diff-2026-10-27.txt), the
[graph check](evidence/flux2-forge-export-graph-check-2026-10-27.json), the
[parity report](evidence/flux2-forge-export-native-parity-2026-10-27.json) and
[lifecycle report](evidence/flux2-forge-export-native-lifecycle-2026-10-27.json) with their logs, the
[manifest fragment](evidence/flux2-forge-export-manifest-fragment-2026-10-27.json) and the
[export log](evidence/flux2-forge-export-log-2026-10-27.jsonl). GPU-lock time: 4.4 minutes (parity 3.6, lifecycle 0.8).

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
  filter. `Flux2ReferencePolicy` refuses, before any load, more than `maxImages` (2), a side above `maxSide` (512), or
  more than `maxTotalTokens` (4,096) generated + reference tokens.
- **Difference from Diffusers' reference preparation:** this package bounds the *longest side* by `maxSide` (512 for
  the export, so at most 32×32 tokens per reference). Diffusers' `Flux2KleinPipeline` instead bounds the *area*: it
  resizes only images above 1024² pixels (Lanczos, to that area) and then floors each side to a multiple of 16.
  - A 1024×768 picture is 512×384 here but passes through Diffusers unchanged.
  - A tall 400×1200 picture is 160×512 here but 400×1200 in Diffusers.
  - Diffusers would accept references the export's 512-pixel side bound cannot encode.

  The filters also differ (Core Graphics versus PIL Lanczos). For upstream-identical tensors, prepare the pixels
  yourself and use `init(pixels:)`, as the parity fixtures do.

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
  specializations do not accumulate (prototype F-P3-75). Placement per asset comes from the manifest. The finalized
  bundle declares text encoder and denoisers with frequent reshapes, and decoder and reference encoder without
  (prototype F-P3-48/68).
- **Lifecycle:** one operation at a time (`sessionBusy` otherwise); cancellation is cooperative between native calls
  and waits for the submitted call; failed or cancelled encoders are discarded; partial denoiser loads are closed;
  `unload()` rejects new work, waits for the active operation and closes permanently.
- **Prepared values:** `Flux2PreparedPrompt` (7.5 MiB fp16 rows + bundle fingerprint) and `Flux2PreparedReferences`
  (float32 packed latents + reference fingerprint) hold no native models; another bundle refuses them before loading.
- **Admission:** optional `Flux2ResourceAdmission` reserves caller-qualified estimates (operation storage, per-model
  residency keyed by stream + metadata hash, per-call scratch) on a shared `ResourceCoordinator`. Unknown models, or
  token counts above the profile, are refused before any stage is released or loaded. For `generate` that is the
  output's tokens plus any references. For `prepare(references:generatedTokens:)` it is the references' tokens plus
  `generatedTokens`, which must be at least 1. These are estimates, not measured limits.

All tensors copy through host memory between calls; there is no device-resident loop and no performance claim.

## Verification

### CPU (default suite)

`Tests/AetherFlux2Tests` (25 tests, weight-free). They cover:

- **Host math:** schedule against the reference pipeline's observed sigmas and the μ rule, separately rounded Euler,
  rotary values and reference offsets, prompt padding/truncation, configuration bounds, reference policy and CGImage
  preparation.
- **Bundle:** acceptance and refusal for 13 recipe deviations (placement included), manifest-driven placement,
  provenance and mutation, concurrent verification reporting the first failure in serial order, confinement against a
  symlinked stream, cancellation of verification, installation round trip, and segment-contract refusal.
- **Scripted sessions:**
  - balanced/resident residency, reference appending and slicing, prepared reuse and shape recycling;
  - refusal before load, busy/cancellation/recovery during a denoiser call, partial denoiser-load failure, unload
    during work, and resource admission accounting, including the reference-preparation profile check;
  - encoder discard paths: a failed text-encoder run and a failed VAE-encoder run, and a gated cancellation of each.
    The encoder closes, and a retry opens a new one (in resident mode, a retained text encoder is discarded and
    replaced).

Opt-in with local files only (`AETHER_FLUX2_FIXTURES`, `AETHER_FLUX2_REFERENCE_FIXTURES`, `AETHER_FLUX2_BUNDLE`):
`samplerArithmeticMatchesTheIndependentReference`, `bundleTokenizerMatchesReferenceTokenIDs` and
`paddedPromptMatchesTheLanguageTokenizersThinkingOffRender`. The last test checks that the padded 512-token prompt equals
the bundle's Qwen3 template rendered through `AetherLanguageTokenizers` with thinking off. `Flux2Tokenizer` keeps its own
fixed render, because the bundle's `chat_template.jinja` is not among the hashed tokenizer files.

### Independent references

Independent float32 Diffusers references on CPU, like every other model's references. Both fixtures come from
`Scripts/prepare-flux2-reference-fixtures.py`: the unmodified Diffusers 0.37.1 `Flux2KleinPipeline` call plus module
hooks, float32 on the CPU (the default; `--device mps` exists only for comparison), with the text encoder in float32 on
the CPU.

| Fixture | Mode | Cases |
| --- | --- | --- |
| `ffc4b1365159e40c` (text to image) | `--text-to-image` | 10 prompts, 512² to 1344×768, incl. French/Japanese text |
| `caf92bb7d83d685c` (reference images) | default; the pipeline with `image=[…]`, references are the Lanczos-resized `fox-512` and `lighthouse-512` outputs of the 2026-10-26 text-to-image fixture, unchanged | 2 cases, one and two references |

Both record token ids/mask, float32 text rows, packed initial noise, per-step latents/timestep/velocity, final latent
and float32 image; the reference fixture also records encoder input pixels, the appended reference tokens and the
transformer position ids. Neither uses the Core AI bundle or AetherKit code.

They replace the 2026-10-26 fixtures `6a8a53d8f3b30817` and `079a8fba9d596a3b`, which ran the pipeline on MPS (text
encoder on CPU). The CPU-computed inputs are byte-identical to those fixtures for all 12 cases: token ids, mask, text
rows, noise and timesteps, plus reference pixels, PNGs and position ids. The MPS-computed outputs move by at most
1.1·10⁻⁴ relative L2 per tensor (velocity cosine ≥ 0.999999994). A second CPU run reproduces every file byte for byte.
Generation took 14 minutes on 6 threads at a 30.1 GiB peak RSS for the ten text-to-image cases, and 2 minutes at 22 GiB
for the reference cases ([record](evidence/flux2-klein-cpu-references-2026-10-27.json)). The weight-free fixture tests
(`samplerArithmeticMatchesTheIndependentReference`, `bundleTokenizerMatchesReferenceTokenIDs`,
`paddedPromptMatchesTheLanguageTokenizersThinkingOffRender`) pass against the new fixtures.

### Native parity against the CPU references (M5 Max, macOS 27.0.1, `preferGPU`, 2026-10-27)

All 12 cases in one process (`AETHER_FLUX2_CASES=all`) on the same int8 bundle (fingerprint `f28ef0e8…`) pass
**113/113 checks** at the unchanged bars. Against the run on the MPS references, no minimum moves by more than
3·10⁻⁶ in cosine or SSIM, or 0.006 dB in PSNR:

| Check (bar) | MPS references (2026-10-26, rebased run) | CPU references (2026-10-27) |
| --- | --- | --- |
| Token ids and mask (exact) | 12/12 | 12/12 |
| Text real-row cosine (≥ 0.999) | min 0.999972 | min 0.999972 |
| Timesteps (exact) | 12/12 | 12/12 |
| Teacher-forced velocity cosine per step (≥ 0.999) | min 0.999242 (1024×768) | min 0.999239 (1024×768) |
| VAE decoder PSNR on the reference final latent (≥ 35 dB) | min 87.690 dB | min 87.684 dB |
| Reference-encoder latent cosine (≥ 0.999) | 3/3 ≥ 0.9999999 | 3/3 ≥ 0.9999999 |
| Image path SSIM through the public session (≥ 0.94) | min 0.960158 (`cafe-512`) | min 0.960160 (`cafe-512`) |
| Whole pipeline SSIM (recorded) | 0.812–0.999 | 0.812–0.999 |

The test process's lifetime peak footprint was 16.1 GiB (the component VAE phase, as before) and the GPU-lock hold
3.7 minutes. See the [report](evidence/flux2-klein-native-parity-cpu-references-2026-10-27.json) and
[log](evidence/flux2-klein-native-parity-cpu-references-2026-10-27.log). The 2026-10-26 sections below compared with
the MPS fixtures and are kept as history. The lifecycle suite was re-run against the CPU fixtures too (see
[Native lifecycle qualification](#native-lifecycle-qualification-2026-10-26)).

### CPU checks against the MPS references (2026-10-26)

With the fixture paths set, the weight-free tests pass. The bundle tokenizer reproduces all 10 reference token-id and
mask rows exactly, including the French/Japanese prompt. Timesteps are exact for all 12 cases, μ included (the
reference cases confirm μ uses generated tokens only). CPU Euler replay of the reference's own velocities reproduces
every following step input and the final latent bit for bit. Reference position ids match the transformer's recorded
`img_ids` exactly.

### Native parity against the MPS references (M5 Max, macOS 27.0.1, `preferGPU`, 2026-10-26)

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

Regenerate the references (CPU only, no GPU lock; about 30 GiB peak RSS):

```sh
CKPT=~/Library/Caches/aether/forge-checkpoints/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
uv run --frozen --project Forge python -I Scripts/prepare-flux2-reference-fixtures.py --text-to-image $CKPT NEW_T2I_DIR
uv run --frozen --project Forge python -I Scripts/prepare-flux2-reference-fixtures.py $CKPT \
  .../oracle-t2i-2026-10-26/fixtures/6a8a53d8f3b30817 NEW_REFERENCE_DIR
```

Reference mode reads the fox-512 and lighthouse-512 source pictures from the 2026-10-26 text-to-image fixture directory, so keep that directory: it is an input of regeneration, not only a superseded record. Its two PNGs are hashed in [its fixture document](evidence/flux2-klein-t2i-fixture-6a8a53d8f3b30817.json). Moving them to the CPU text-to-image fixture would change the reference fixture's inputs and ID, so it waits for the next intentional regeneration.

Run the native gate (one suite at a time, under the shared lock):

```sh
AETHER_NATIVE_FLUX2_QUALIFICATION=1 \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/macos-any-gpu \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c \
AETHER_FLUX2_REFERENCE_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-reference-cpu-2026-10-27/fixtures/caf92bb7d83d685c \
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
directory, plus the bundle and fixture paths) is modeled on Krea's native session suite.

```sh
AETHER_NATIVE_FLUX2_LIFECYCLE=1 \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/macos-any-gpu \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c \
AETHER_FLUX2_REFERENCE_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-reference-cpu-2026-10-27/fixtures/caf92bb7d83d685c \
AETHER_FLUX2_LIFECYCLE_OUTPUT=/new/output/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 lifecycle" swift test -c release --filter flux2NativeLifecycle
```

**Re-run against the CPU references (2026-10-27).** The suite was re-run with no code change against the CPU fixtures
(`fox-512` from `ffc4b136…`, the first reference case from `caf92bb7…`). Its inputs from them (size, seed, text rows,
noise, prompt and reference pixels) are byte-identical to the MPS fixtures'; only the images it compares with
changed. It passed **44/44 checks** with the same 176 native calls, 34 model instances, 32 function loads and 144
reuses. The gated SSIM checks moved in the sixth decimal: baseline 0.990223 → 0.990221 and reference 0.998442 →
0.998442 (bar 0.94 for both). The recorded balanced-prompt SSIM moved from 0.901650 to 0.901649. It took 51.3 s, the
rebased build's first run. See the [report](evidence/flux2-klein-native-lifecycle-cpu-references-2026-10-27.json),
[event journal](evidence/flux2-klein-native-lifecycle-events-cpu-references-2026-10-27.jsonl) and
[log](evidence/flux2-klein-native-lifecycle-cpu-references-2026-10-27.log). The 2026-10-26 results below compared
with the MPS fixtures. A tracked factory opens real
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

### Review fixes smoke (2026-10-26)

After the review fixes (concurrent confined verification, manifest-driven placement, and the reference-preparation
admission check), the default parity suite ran again under the shared lock. All 68/68 checks pass, with values
identical to the post-rebase report. The bundle's concurrent `verify()` took 1.2 s with a warm page cache, against
2.8 s for the earlier serial pass; this is not a benchmark. Every asset loaded with the placement read from the
finalized `aether.json`. See the [report](evidence/flux2-klein-native-smoke-review-2026-10-26.json) and
[log](evidence/flux2-klein-native-smoke-review-2026-10-26.log).

### Not yet qualified

- Native cancellation of the text encoder and the reference encoder. Their run-failure and cancellation discard paths
  are covered by scripted CPU tests only.
- iOS: no build or run has been qualified. The prototype ran this recipe on iPhone with staged residency.
- Prompt adherence (T3) and a subjective review across more prompts and seeds.
- References prepared from a `CGImage`: parity used the reference's exact encoder pixels.
- Benchmarking under a quiet, controlled protocol, and per-stage memory peaks.
- Device-resident carries, or a GPU sampling loop.
