# FLUX.2 [klein] 4B: text-to-image and multi-reference generation

Status: experimental, 2026-10-27 (0.0.2). The products, the qualified local bundle, independent float32 Diffusers
references on CPU and opt-in native parity and lifecycle gates are in place. Results and their scope are in
[Verification](#verification). There is no iOS execution or published bundle yet. A benchmark harness is in place; its
quiet-window run is pending ([Benchmark harness](#benchmark-harness-003)).

**Qualified bundle:** 0.0.2 qualifies the bundle with fingerprint `a48aaf07…` (reference fingerprint `a69abb8e…`),
produced from the pinned checkpoint by `Forge/flux2` in this repository ([Producing the bundle](#producing-the-bundle)).
It replaces `f28ef0e8…` (reference fingerprint `8fb2df90…`), which the superseded prototype exported and which the
earlier evidence refers to; the two have an identical contract. Core AI exports are not byte-reproducible, so every new
export gets a new fingerprint and must pass the contract diff, native parity and lifecycle before it becomes the
default.

**Fingerprint since 0.0.3:** the fingerprint now also hashes the bundle's `tokenizer/chat_template.jinja`, which
renders the prompt. The same bundle bytes therefore have fingerprint `d27cefc4…` (reference fingerprint `d8a98cb0…`).
The 0.0.2 records, the qualification below included, name it `a48aaf07…`; see
[Chat template in the fingerprint](#chat-template-in-the-fingerprint-003).

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
| `AetherFlux2Tokenizers` | `Flux2Tokenizer`: the bundle's Qwen3 tokenizer and chat template through `AetherLanguageTokenizers`; never downloads. |
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
| textEncoder | `input_ids`, `position_ids`, `attention_mask` int32 `[1,512]` → `text_embeds` fp16 `[512,7680]` | int8 per-channel, 2.90 GiB |
| denoiser0 | `latents` fp16 `[N,128]`, `text_embeds`, `sigma` f32 `[1]`, `rope_text` f32 `[512,2,128]`, `rope_image` f32 `[N,2,128]` → `image_1`, `text_1`, `cond` f32 | int8 (fp16 `proj_out`), 1.77 GiB |
| denoiser1 | `image_1`, `text_1`, `cond`, ropes → `image_2`, `text_2` | 1.20 GiB |
| denoiser2 | `image_2`, `text_2`, `cond`, ropes → `velocity` fp16 `[N,128]` | 1.35 GiB |
| vaeDecoder | `latents` fp16 `[h,w,128]` (still batch-normalized) → `image` f32 `[3,16h,16w]` | fp16 weights, f32 compute, 97 MiB |
| vaeEncoder | `reference_pixels` f32 `[3,H,W]` → `reference_latents` f32 `[H/16,W/16,128]` (posterior mode, packed, normalized) | fp16 weights, f32 compute, 67 MiB |

`Flux2Assets.open` refuses any other recipe: schema/driver/target, the exact empirical shift constants, rotary axes and
θ, prompt template/length/padding/pad token, segment list, function roles, tensor bindings, the reference contract
(`packedLatents`, stride 10, bounded counts) and the source repository and revision. Provenance pins every file by
SHA-256; opening checks model declarations and metadata against them. `verify()` streams every model (about 7.4 GiB) on
concurrent threads, with each file confined to its asset directory. It reports the first failing asset in serial order
and stops when the calling task is cancelled.
Per-asset placement is read from the manifest: `diffusion.assetReshapes` overrides the bundle-wide
`expectFrequentReshapes`. Undeclared assets are refused.
`fingerprint` identifies the text-to-image contract: metadata, manifest, the tokenizer files (`tokenizer.json`,
`tokenizer_config.json` and `chat_template.jinja`, which must be present) and the model declarations.
`referenceFingerprint` adds the encoder and reference contract.
`installationManifest()`/`open(in:)` and `Flux2GenerationSession(installed:)` follow the Krea installation contract.

### Producing the bundle

`Forge/flux2/` exports the bundle from the pinned snapshot in Forge's locked environment. It is written from upstream
Diffusers 0.37.1 and Transformers 4.57.3 against this contract and reproduces the 2026-10-26 bundle's recipe; the
superseded prototype exporter that produced `f28ef0e8…` is no longer used. Export is CPU-only (Python loads assets
`cpu_only` for its self-check), takes about 1.5 minutes on an M5 Max and peaks at a 17.4 GiB footprint:

```sh
uv sync --frozen --project Forge
C=~/Library/Caches/aether/forge-checkpoints/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294
O=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27
Forge/.venv/bin/python -I Forge/flux2/check_flux2_graphs.py --workdir $O/graph-check-real --checkpoint $C   # optional, about 26 GiB RSS
Forge/.venv/bin/python -I Forge/flux2/export_flux2.py --checkpoint $C --output $O/export
python3 Scripts/finalize-flux2-bundle.py $O/export $C $O/bundle
# against the f28ef0e8 export, kept at macos-any-gpu as the historical reference
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

**The 2026-10-27 export** (`forge-export-2026-10-27/bundle`, `a48aaf07…`; `d27cefc4…` since 0.0.3) is the qualified bundle. Its contract
equals `f28ef0e8…`'s exactly: every function signature, the segment split, `metadata.json`, all 111 compared
`aether.json` fields (the quantization notes included) and the tokenizer files. The internal graphs differ: LayerNorm
and GroupNorm lower to Core AI composites, the VAE decoder upsamples by nearest-neighbour interpolation where the
prototype used conv + pixel shuffle, and the q/k norm feeds the rotary product in float32 without an fp16 round trip.
Its qualification is under [Verification](#qualified-bundle-a48aaf07-against-the-cpu-references-2026-10-27). See the
[record](evidence/flux2-forge-export-2026-10-27.json) (commands, versions, file hashes, every parity check side by
side), the [contract diff](evidence/flux2-forge-export-contract-diff-2026-10-27.txt), the
[graph check](evidence/flux2-forge-export-graph-check-2026-10-27.json), the
[manifest fragment](evidence/flux2-forge-export-manifest-fragment-2026-10-27.json) and the
[export log](evidence/flux2-forge-export-log-2026-10-27.jsonl).

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
  `unload()` rejects new work, waits for the active operation and closes permanently. Cancelling while the prompt is
  being encoded returns in about 0.17 s on an M5 Max: the rest of the encoder's one native call, then its discard
  ([Text encoder cancellation](#text-encoder-cancellation-003)).
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
`templateRenderReproducesTheFixedQwen3Render`. `Flux2Tokenizer` renders one user message through the bundle's own
template with `AetherLanguageTokenizers`, thinking off. The last test checks that its padded 512-token prompt equals the
fixed render it replaced in 0.0.3, encoded by the same tokenizer. It covers every fixture prompt and edge cases: empty,
surrounding whitespace, think tags and a prompt longer than 512 tokens. Until 0.0.2 the test was
`paddedPromptMatchesTheLanguageTokenizersThinkingOffRender`, which made the same comparison the other way round.

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

### Prompt adherence (0.0.3)

`flux2PromptAdherenceGenerate` and `flux2PromptAdherenceJudge` (opt-in, in `AetherNativeTests`) score whether images
show what their prompt asks for, against the plan [`Benchmarks/flux2-prompt-adherence.json`](../Benchmarks/flux2-prompt-adherence.json).
The plan breaks each of the ten text-to-image fixture prompts into two to five yes/no questions (38 in all), plus one
control question per prompt whose answer should be no. A local Qwen3-VL-4B judge (the one-function bundle,
Apache-2.0) answers every question about each image, greedily, from a one-image chat.

- **Paired comparison:** Kit's image uses the fixture's own prompt, size, steps and initial noise, so it pairs with the
  float32 Diffusers reference image. Kit passes if its score is no lower than the reference's by more than 3
  percentage points, a margin fixed in the plan before the run. Three extra seeds per prompt (Kit's own noise, Kit
  only) show the spread across seeds.
- **Judge check:** the control questions must be answered no at least 90% of the time, so a judge that agrees with
  everything fails the run.

M5 Max, macOS 27.0.1, bundle `d27cefc4…`, fixture `ffc4b136…`. Generation took 212 s for 40 images (one GPU hold of
3.7 min) and judging 311 s for 50 images and 240 questions (a hold of 5.7 min):

| Images | Positive questions answered yes |
| --- | ---: |
| Kit, paired with the reference | 38/38 (100%) |
| Float32 Diffusers reference | 38/38 (100%) |
| Kit, three extra seeds | 114/114 (100%) |
| Control questions answered no | 50/50 (100%) |

Kit passes, with no unclear answers. The result is at the ceiling, though: these prompts and questions don't
separate Kit from the reference, so the test shows no adherence loss on this set but can't measure a small one. A
harder set (counts, spatial relations, rendered text, attribute binding) would be needed for that. The run writes a
`contact-sheet.html` beside the images, with every answer under its image, for a subjective review. See the
[report](evidence/flux2-prompt-adherence-2026-10-27.json).

```sh
C=~/.cache/huggingface/aether-kit
AETHER_NATIVE_FLUX2_ADHERENCE=generate AETHER_FLUX2_BUNDLE=$C/flux2-klein-4b/forge-export-2026-10-27/bundle \
AETHER_FLUX2_FIXTURES=$C/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c AETHER_FLUX2_ADHERENCE_OUTPUT=/new/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 adherence generate" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2PromptAdherenceGenerate
AETHER_NATIVE_FLUX2_ADHERENCE=judge AETHER_FLUX2_ADHERENCE_JUDGE=$C/qwen3-vl-4b-abliterated-one-function-bundle-macos \
AETHER_FLUX2_FIXTURES=$C/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c AETHER_FLUX2_ADHERENCE_OUTPUT=/new/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 adherence judge" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2PromptAdherenceJudge
```

### Benchmark harness (0.0.3)

`aether-probe benchmark-flux2` (wrapper: `Scripts/benchmark-flux2.sh prepare|run NEW_DIR [PLAN]`) follows the Krea
harness ([Krea benchmarks](krea-benchmarks.md)). The plans [`flux2-baseline.json`](../Benchmarks/flux2-baseline.json)
(encoded) and [`flux2-provided.json`](../Benchmarks/flux2-provided.json) pin the bundle fingerprint and both fixtures.
They run `fox-512`, `lighthouse-512`, `cat-1024` and the one-reference `fox-watercolor-ref1`, each with the fixture's
prompt, size, steps and noise. Each case gets 2 warm-ups and 5 measured iterations on one balanced `preferGPU` session.

- **Profiles:** `encoded` encodes the prompt and references on every iteration. `provided` uses the fixture's text rows,
  and prepares references once per case outside iteration timing.
- **What it records:** per-stage timings (conditioning, references, denoiser setup, each step, decode) with observer
  time excluded, sampled memory peaks, and the usual executable, source and plan identities with postflight rechecks.
- **Correctness:** every iteration must reproduce its case's first-iteration pixels exactly. A provided-profile case
  must reach SSIM 0.94 against the fixture image on its first measured iteration; the encoded profile records it only,
  as parity does for the whole pipeline.
- **Prepare:** reads the plan, the fixtures (every file hash-checked) and the tokenizer, and checks the bundle's token
  IDs against each fixture's, all without the GPU. `prepare` passes for both plans.

A short harness smoke ran on a busy host (one warm-up, three measured iterations, `fox-512` and `fox-watercolor-ref1`;
1.6 min of GPU-lock time). It completed with identical pixels across iterations and SSIM 0.990/0.999 (provided) and
0.901/0.966 (encoded) against the fixtures. Its timings are not benchmark results. The measured run waits for a quiet
window: with 4 cases × 7 iterations × 2 profiles at about 1.5–7 s each, it holds the GPU for about 6–8 minutes.

### Text encoder cancellation (0.0.3)

`flux2NativeEncoderCancellation` (opt-in, in `AetherNativeTests`) cancels prompt requests while the text encoder runs.
It uses the `fox-512` prompt and noise with balanced residency, so each request opens the encoder, makes one native call
over the 512-token prompt and discards the encoder. A tracked factory brackets the awaited native call and the model
open; nothing gates or delays native work. M5 Max, macOS 27.0.1, `preferGPU`, bundle `d27cefc4…`, 57/57 checks:

| Measurement | Result |
| --- | --- |
| Encoder open (`AIModel(contentsOf:)`) | 1.6–3.7 ms |
| Encoder native call, balanced (opened per request) | 124–166 ms |
| Encoder native call, resident (retained) | 119–128 ms |
| Cancel during the native call, until `generate` throws | 172–179 ms (3 trials), 175 ms with admission; 138 ms resident |
| Cancel during the open, until `generate` throws | 0.4–1.3 ms (3 trials) |

Core AI does not interrupt a submitted call when the Swift task is cancelled. In every trial the cancelled call ran to
completion and returned its outputs, and Kit's own check after the joined call threw `CancellationError`. The latency
is therefore bounded by two things: the rest of that one encoder call (up to its whole duration, about 120–170 ms here),
then the encoder's discard (47–59 ms). The process's first call to the encoder takes longer while Core AI
specializes it: 0.59 s in the parity run, and longer still on a bundle that has never been specialized on the Mac.
Cancelling during the open costs almost nothing: the open is short, and `CoreAIBackend.load` checks cancellation after it
and drops the model, so it never reaches the native call.

Checks in each trial: the request throws `CancellationError`; the encoder's native call returns before cancellation
does; the encoder is discarded before the request returns; no native call starts after the cancel; the session returns
to ready; and the next prompt request reproduces the uncancelled baseline pixels exactly. With
`Flux2ResourceAdmission` and an in-process `GPUAdmission`, the session afterwards holds the same reservations as after
its baseline, no GPU grant is running or queued, and a fail-fast admission succeeds at once. Every model instance is
closed exactly once, and every native call returns.

Splitting the encoder into chunked calls is the only way to make cancellation take effect sooner, and it would need a
re-export and a new qualification. At about 0.17 s that is not worth it, so the encoder stays one call. See the
[report](evidence/flux2-encoder-cancellation-2026-10-27.json), [log](evidence/flux2-encoder-cancellation-2026-10-27.log)
and [event journal](evidence/flux2-encoder-cancellation-events-2026-10-27.jsonl).

```sh
AETHER_NATIVE_FLUX2_ENCODER_CANCEL=1 \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27/bundle \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c \
AETHER_FLUX2_ENCODER_CANCEL_OUTPUT=/new/output/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 encoder cancellation" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2NativeEncoderCancellation
```

`AETHER_FLUX2_ENCODER_CANCEL_TRIALS` sets the number of trials of each kind (default 3).

### Memory after unload (0.0.3)

`flux2UnloadCycling` (opt-in, in `AetherNativeTests`) runs five cycles in one process. Each cycle opens a session,
generates one 256² image in four steps from a prompt, calls `unload()` and drops the session. After each unload it
records the footprint, Metal allocation, live malloc bytes and system wired memory. Readings are taken at once, after
2 s and after `malloc_zone_pressure_relief`. Footprint 2 s after each unload, against a 48 MiB baseline:

| Cycle | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Footprint (MiB) | 368 | 531 | 683 | 679 | 677 |
| Metal allocation (MiB) | 242 | 262 | 262 | 262 | 262 |

The residue grows for three cycles and then stops. Live malloc memory stays within 12 MiB of the baseline. A memory
graph after the fifth cycle holds 54 MB of live heap (41 MB at the baseline) against a 751 MB footprint. The rest is
freed large allocations that the macOS allocator keeps dirty in its large-block cache ("Malloc Large (empty)" in
`vmmap`), and `malloc_zone_pressure_relief` does not return them. This is macOS allocator behaviour, bounded and
reused by the next load, not a Kit or Core AI leak. As a diagnostic only, the same run with that cache disabled
(`MallocLargeCache=0` in the test process's environment) settles 27 MiB above the baseline from the first cycle.

Allocation histories of those regions (`MallocStackLogging=full`, `malloc_history` on each one) attribute most of the
identified bytes to the ANE compiler. It validates graph regions while MPSGraph specializes each model inside Core AI,
on every load, even with `preferGPU`. Kit's own allocations (output tensor copies and array growth) account for about
13 of the 365 MiB after two cycles. Reusing those buffers would not change the picture, so Kit makes no change. See the
[summary](evidence/unload-cycling-2026-10-27.json).

```sh
AETHER_NATIVE_UNLOAD_CYCLING=flux2 AETHER_NATIVE_UNLOAD_OUTPUT=/new/output/dir \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27/bundle \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 unload cycling" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2UnloadCycling
```

`AETHER_NATIVE_UNLOAD_CYCLES` changes the cycle count (default 5). `AETHER_NATIVE_UNLOAD_MEMGRAPH=1` adds memory graphs
of the baseline and the last cycle, and `full` adds the whole allocation history for a run with `MallocStackLogging=full`.

### Chat template in the fingerprint (0.0.3)

Until 0.0.2 the fingerprint hashed `tokenizer.json` and `tokenizer_config.json` but not `chat_template.jinja`, and
`Flux2Tokenizer` wrote the Qwen3 thinking-off render out by hand. Since 0.0.3 the template is the third hashed tokenizer
file, so a changed template changes both fingerprints and is refused after opening, a bundle without one does not
open, and `Flux2Tokenizer` renders through it with `AetherLanguageTokenizers`. `Flux2Prompt.render` was removed. The
bundle's provenance already pinned the template (`a55ee1b1…`), so the qualified bundle opens unchanged under new
identities:

| | 0.0.2 | Since 0.0.3 |
| --- | --- | --- |
| `fingerprint` | `a48aaf07eab1fdee106c52b3e16c903c629846bf64dc92b5b35860b8a295fa39` | `d27cefc4b63c0f25e9f4bf581f182745c874ca59583f1d311322de1b55337e66` |
| `referenceFingerprint` | `a69abb8e68cadcc56f2b6f5d9822f1600c318712914fae0146237ee24152c988` | `d8a98cb04e2a28167a6354443677fa19d9a5ad6b8019cc8bbb3bd184f34d7664` |

The token IDs do not change. On CPU, with the bundle and both CPU fixture paths set, `AetherFlux2Tests` passes 28/28:
`bundleTokenizerMatchesReferenceTokenIDs` reproduces the ten text-to-image fixtures' IDs and masks, and
`templateRenderReproducesTheFixedQwen3Render` gives the same padded prompt as the fixed render for all twelve fixture
prompts and six edge cases. Rendering with thinking on instead makes both tests fail, so they do detect a render change.
The evidence records named above keep the 0.0.2 values.

The new identities were recorded natively once (M5 Max, macOS 27.0.1, `preferGPU`, the parity and lifecycle
commands below with `AETHER_FLUX2_CASES=all`, 2.7 minutes of GPU-lock time). Parity passes **113/113** and lifecycle **44/44**, and every check value equals the 0.0.2
qualification's exactly, so the qualification carries over to `d27cefc4…`. Both reports name both fixtures
(`ffc4b136…` and `caf92bb7…`). See the [parity report](evidence/flux2-template-fingerprint-native-parity-2026-10-27.json)
and [log](evidence/flux2-template-fingerprint-native-parity-2026-10-27.log), and the
[lifecycle report](evidence/flux2-template-fingerprint-native-lifecycle-2026-10-27.json) and
[log](evidence/flux2-template-fingerprint-native-lifecycle-2026-10-27.log).

### Qualified bundle a48aaf07 against the CPU references (2026-10-27)

M5 Max, macOS 27.0.1, `preferGPU`. All 12 cases in one process (`AETHER_FLUX2_CASES=all`) on `a48aaf07…` pass
**113/113 checks** at the unchanged bars, and the lifecycle suite passes **44/44**. `Flux2Assets.open` and `verify()`
accept the bundle, and `AetherFlux2Tests` passes 28/28 with the bundle and CPU fixture paths set. The `f28ef0e8…`
column is the baseline from the same fixtures ([below](#baseline-f28ef0e8-against-the-cpu-references-2026-10-27)):

| Check (bar) | `f28ef0e8…` baseline | `a48aaf07…` qualified |
| --- | --- | --- |
| Token ids and mask (exact) | 12/12 | 12/12 |
| Text real-row cosine (≥ 0.999) | min 0.999972 | min 0.999972 |
| Timesteps (exact) | 12/12 | 12/12 |
| Teacher-forced velocity cosine per step (≥ 0.999) | min 0.999239 (1024×768) | min 0.999247 (1024×768) |
| VAE decoder PSNR on the reference final latent (≥ 35 dB) | min 87.684 dB | min 87.684 dB |
| Reference-encoder latent cosine (≥ 0.999) | 3/3 ≥ 0.9999999 | 3/3 ≥ 0.9999999 |
| Image path SSIM through the public session (≥ 0.94) | min 0.96016 (`cafe-512`) | min 0.95953 (`cafe-512`) |
| Whole pipeline SSIM (recorded) | 0.812–0.999 | 0.832–0.999 |

Per check, image-path SSIM is within 0.0008 of the baseline, velocity cosine within 3·10⁻⁵ and decoder PSNR within
0.001 dB. The ungated whole-pipeline SSIM moves both ways across cases, from −0.012 (`desk-1024x768`) to +0.059
(`fox-lighthouse-ref2`, 0.812 → 0.871): that free-running path starts from the native text rows, padding rows
included, and its trajectory is sensitive to any change in them (see the MPS-reference section below). It is not an
improvement. The lifecycle suite made the same 176 native calls with 34 model instances, 32 function loads and 144
reuses as the baseline; its gated SSIMs were 0.989678 (baseline pixels) and 0.998520 (reference), against 0.990221 and
0.998442 for `f28ef0e8…` (bar 0.94), and the recorded balanced-prompt SSIM 0.901143 (0.901649). The parity process
peaked at a 16.1 GiB lifetime footprint in the component VAE phase, the same as the baseline. GPU-lock time was 4.4
minutes (parity 3.6, lifecycle 0.8). See the [parity report](evidence/flux2-forge-export-native-parity-2026-10-27.json)
and [log](evidence/flux2-forge-export-native-parity-2026-10-27.log), the
[lifecycle report](evidence/flux2-forge-export-native-lifecycle-2026-10-27.json) and
[log](evidence/flux2-forge-export-native-lifecycle-2026-10-27.log), and the
[record](evidence/flux2-forge-export-2026-10-27.json).

### Baseline: f28ef0e8 against the CPU references (2026-10-27)

Measured on `f28ef0e8…`; identical contract. M5 Max, macOS 27.0.1, `preferGPU`. These reports are the baseline the
qualified bundle is compared with, not superseded by it. All 12 cases in one process (`AETHER_FLUX2_CASES=all`) pass
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
[log](evidence/flux2-klein-native-parity-cpu-references-2026-10-27.log). The 2026-10-26 sections below were measured on
`f28ef0e8…` against the MPS fixtures and are kept as history. The lifecycle suite was re-run against the CPU fixtures too (see
[Native lifecycle qualification](#native-lifecycle-qualification-2026-10-26)).

### CPU checks against the MPS references (2026-10-26)

With the fixture paths set, the weight-free tests pass. The bundle tokenizer reproduces all 10 reference token-id and
mask rows exactly, including the French/Japanese prompt. Timesteps are exact for all 12 cases, μ included (the
reference cases confirm μ uses generated tokens only). CPU Euler replay of the reference's own velocities reproduces
every following step input and the final latent bit for bit. Reference position ids match the transformer's recorded
`img_ids` exactly.

### f28ef0e8 against the MPS references (M5 Max, macOS 27.0.1, `preferGPU`, 2026-10-26)

Measured on `f28ef0e8…`; identical contract. Both runs passed every check. They used the finalized int8 bundle (fingerprint `f28ef0e8…`, reference fingerprint
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

Timing and memory, measured on `f28ef0e8…` (identical contract), with no benchmark protocol, on a shared host, with the
Core AI specialization cache warm from the export:

- **Component steps:** about 0.21 s per denoiser step at 512², 0.25 s with 256–448 reference tokens, 0.55–0.62 s at
  768² and 1024×576, about 0.94 s at 1024² and 1344×768. Text encoding takes 0.05 s. Decoding takes 0.13 s at 512²
  and 0.6–1.1 s at the larger sizes.
- **First use:** a new shape's first denoiser step took 1.2 s at 512² and 7.3 s at 1024².
- **End to end** (`generate`, including model loads and recycling): 1.0–2.9 s at 512² and 576×448 with provided
  conditioning, 2.5–3.8 s from a prompt (balanced mode reloads denoisers after encoding), 3.4–4.6 s with references,
  and 6.0–6.6 s at 1024² and 1344×768.
- **Memory:** the test process's lifetime peak footprint was 3.76 GiB for the 512² run. For the large run it was
  13.6 GiB, reached in the component VAE phase, which decoded five sizes in one decoder session. That is the accumulation
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
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27/bundle \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c \
AETHER_FLUX2_REFERENCE_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-reference-cpu-2026-10-27/fixtures/caf92bb7d83d685c \
AETHER_FLUX2_OUTPUT=/new/output/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 native" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2NativeParity
```

`AETHER_FLUX2_CASES` selects text-to-image cases (default: the five 512²-class cases; `all` for all ten).

Since 0.0.3, the parity and lifecycle reports both record their fixtures under `fixtures`: `t2i` for the text-to-image
fixture ID and, when `AETHER_FLUX2_REFERENCE_FIXTURES` is set, `reference` for the reference fixture ID. Earlier parity
reports name only `t2i`, with the reference cases listed in `cases`, and earlier lifecycle reports name neither.

### Post-rebase parity (2026-10-26)

Measured on `f28ef0e8…`; identical contract. After rebasing onto the 0.0.1 release (origin/main `128573d`), the parity suite ran again with all 12 cases in one
process (`AETHER_FLUX2_CASES=all`). It passed **113/113 checks**, and every check value for the seven 512²-class
cases equals the pre-rebase report bit for bit. The minimums are unchanged: velocity 0.99924, image-path SSIM 0.9602,
decoder 87.7 dB. The process's lifetime peak was 16.1 GiB, reached while the component VAE phase decoded twelve sizes
in one decoder session (see the memory note above). See the
[report](evidence/flux2-klein-native-parity-rebased-2026-10-26.json) and
[log](evidence/flux2-klein-native-parity-rebased-2026-10-26.log).

### Native lifecycle qualification (2026-10-26)

`flux2NativeLifecycleQualification` (`AETHER_NATIVE_FLUX2_LIFECYCLE=1`, `AETHER_FLUX2_LIFECYCLE_OUTPUT` set to a new
directory, plus the bundle and fixture paths) is modeled on Krea's native session suite.

```sh
AETHER_NATIVE_FLUX2_LIFECYCLE=1 \
AETHER_FLUX2_BUNDLE=~/.cache/huggingface/aether-kit/flux2-klein-4b/forge-export-2026-10-27/bundle \
AETHER_FLUX2_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-t2i-cpu-2026-10-27/fixtures/ffc4b1365159e40c \
AETHER_FLUX2_REFERENCE_FIXTURES=~/.cache/huggingface/aether-kit/flux2-klein-4b/oracle-reference-cpu-2026-10-27/fixtures/caf92bb7d83d685c \
AETHER_FLUX2_LIFECYCLE_OUTPUT=/new/output/dir \
~/.cache/aether-kit/with-gpu-lock.sh "flux2 lifecycle" swift test --triple arm64-apple-macosx27.0 -c release --filter flux2NativeLifecycle
```

**Qualified bundle (`a48aaf07…`, 2026-10-27):** 44/44 against the CPU fixtures; see
[above](#qualified-bundle-a48aaf07-against-the-cpu-references-2026-10-27).

**Baseline: `f28ef0e8…` re-run against the CPU references (2026-10-27).** Measured on `f28ef0e8…`; identical contract.
The suite was re-run with no code change against the CPU fixtures
(`fox-512` from `ffc4b136…`, the first reference case from `caf92bb7…`). Its inputs from them (size, seed, text rows,
noise, prompt and reference pixels) are byte-identical to the MPS fixtures'; only the images it compares with
changed. It passed **44/44 checks** with the same 176 native calls, 34 model instances, 32 function loads and 144
reuses. The gated SSIM checks moved in the sixth decimal: baseline 0.990223 → 0.990221 and reference 0.998442 →
0.998442 (bar 0.94 for both). The recorded balanced-prompt SSIM moved from 0.901650 to 0.901649. It took 51.3 s, the
rebased build's first run. See the [report](evidence/flux2-klein-native-lifecycle-cpu-references-2026-10-27.json),
[event journal](evidence/flux2-klein-native-lifecycle-events-cpu-references-2026-10-27.jsonl) and
[log](evidence/flux2-klein-native-lifecycle-cpu-references-2026-10-27.log). The 2026-10-26 results below were
measured on `f28ef0e8…` against the MPS fixtures. A tracked factory opens real
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

Measured on `f28ef0e8…`; identical contract. After the review fixes (concurrent confined verification, manifest-driven placement, and the reference-preparation
admission check), the default parity suite ran again under the shared lock. All 68/68 checks pass, with values
identical to the post-rebase report. The bundle's concurrent `verify()` took 1.2 s with a warm page cache, against
2.8 s for the earlier serial pass; this is not a benchmark. Every asset loaded with the placement read from the
finalized `aether.json`. See the [report](evidence/flux2-klein-native-smoke-review-2026-10-26.json) and
[log](evidence/flux2-klein-native-smoke-review-2026-10-26.log).

### Not yet qualified

- Native cancellation of the reference encoder. Its run-failure and cancellation discard paths are covered by scripted
  CPU tests only. The text encoder's is qualified natively ([above](#text-encoder-cancellation-003)).
- iOS: no build or run has been qualified. The prototype ran this recipe on iPhone with staged residency.
- A subjective review across more prompts and seeds, and a harder adherence set: the fixture prompts score at the
  ceiling ([Prompt adherence](#prompt-adherence-003)).
- References prepared from a `CGImage`: parity used the reference's exact encoder pixels.
- The benchmark run in a quiet window ([Benchmark harness](#benchmark-harness-003)), and per-stage memory peaks.
- Device-resident carries, or a GPU sampling loop.
