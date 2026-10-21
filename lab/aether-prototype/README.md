# Aether prototype

A Rust crate for local generative pipelines using [Corsairs](../corsairs). The first pipeline is Krea 2 Turbo realism: prompt → Qwen fp16 text encoder → four quantized denoiser segments → eight float32 Euler steps → VAE → PNG. Inference runs through Corsairs' safe public API. Python and Swift are absent from the runtime.

Requires Apple Silicon, macOS 27, Rust 1.99, Corsairs 0.3.0, and external Core AI model assets. The runtime dependency is pinned to the published crates.io release; Aether itself is unpublished. Models and tokenizer assets are supplied by the application; nothing downloads weights or copies them into the crate.

[`H3Pipeline`](docs/h3.md) implements text-to-audio/video with a 50-layer Qwen conditioner, real packed rows, all 50 transformer blocks at every one of 50 schedule pairs, and both source-precision decoders. A short diagnostic clip has completed native execution; see [H3 validation](docs/h3-validation.md) for the scope of the current evidence. Corsairs owns checked tensors and native runtime qualification; Aether owns the export recipe, generation contracts, residency and media delivery.

The [published-v0.3 H3 benchmark](docs/h3-benchmarks.md) measured preparation and three production-profile steps before termination. The original 50-step path projects beyond the requested **20-minute start-to-finished-clip limit**.

The separate [eight-step Turbo recipe](docs/h3-turbo.md) now produces a usable visual pilot: [960×544/124-frame MP4 with audio](docs/evidence/h3-qkv-render/clip.mp4) in **11m 48s including its numerical gate**, or 11m 26s for generation alone. Sampled frames show the crafted bird raising its wings on a wooden desk. Raw/media/timing and audio signal checks pass; subjective audio listening and native cache-hit timing remain unassessed. A missing raw-checkpoint QKV normalization in Aether's exports caused the earlier gray-texture failure and is covered by a regression. Older H3 bundles must be re-exported. GPU work is stopped; no Corsairs defect was isolated.

## Generate

```rust,no_run
use aether::{ImagePipeline, ImageRequest, ImageSize};
use aether::krea::{KreaAssets, KreaOptions, KreaPipeline};

async fn render() -> aether::Result<()> {
    let assets = KreaAssets::open("/path/to/macOS-dynamic-bundle")?;
    let mut pipeline = KreaPipeline::new(assets, KreaOptions::default())?;
    let request = ImageRequest::new(
        "a tall clock tower in a snowy old town at night, lanterns glowing",
        ImageSize::new(768, 1344),
        71344,
    );
    let image = pipeline.generate(&request).await?;
    image.save_png("tower.png")?;
    // Keep this pipeline alive for subsequent images.
    pipeline.unload().await
}
```

Use any async executor. The included example uses `futures-executor` (a development dependency):

```sh
cargo run --offline --release --example render -- \
  /Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/variants/macos-any-gpu \
  tower.png \
  'a tall clock tower in a snowy old town at night, lanterns glowing' \
  768 1344 71344
```

Dimensions must be positive multiples of 16 with at most 4,096 image tokens (`width × height / 256`), matching the dynamic export's limit. The default is eight steps; explicit requests allow 1–50. Seeds follow Aether's Swift SplitMix64/Box–Muller generator in packed token order. PyTorch fixture noise uses a different generator; replay it with the optional noise argument when checking those references.

`GeneratedImage` retains finite float32 CHW pixels before clipping. PNG output uses the lab's float32 scaling and ties-to-even quantization. `generate_with` exposes text, step-latent and decode progress; `generate_prepared` accepts exact rendered token IDs; `generate_conditioned` accepts supplied text embeddings. These paths share the same denoising and decoder implementation.

## Lab optimizations

The default `KreaOptions` keeps the fp16 text encoder, one four-segment denoiser set, and decoder resident. It also keeps a two-entry LRU text cache keyed by rendered token IDs. The four denoiser assets load in a single Corsairs batch, after text encoding. `Residency::Balanced` drops the text encoder after each uncached encode; `LoadPolicy::Serial` provides a serial loading option. Resident mode follows the lab's speed preference and its 64 GiB test host; it is not a measured minimum-memory profile.

The lab's fixed export is useful only at **4,032 image tokens and 20 conditioning rows**. Generate a small registration from the retained provenance:

```sh
python3 scripts/lab-shape.py /Users/zakkeown/Code/atelier/lab/krea-lab .local/tower-20.json
```

Then register it before creating the pipeline:

```rust,no_run
# use aether::krea::{KreaAssets, ShapeSpecialization};
# fn setup() -> aether::Result<()> {
let mut assets = KreaAssets::open("/path/to/original-dynamic-bundle")?;
assets.add_specialization(ShapeSpecialization::from_file(".local/tower-20.json")?)?;
# Ok(()) }
```

Registration verifies the original four source hashes and all four specialized source streams using SHA-256. This one-time setup reads about 25.7 GB for the lab export. The provenance mapping between those streams comes from the trusted offline forge record. Selection requires both shape dimensions and the pinned original source identities; other shapes use the original dynamic assets. Registration is capped at eight exports. Only one denoiser set stays loaded; switching to another export releases the previous set. Keep registered files immutable while a pipeline is alive. `KreaAssets::verify()` additionally hashes the original encoder and decoder; ordinary dynamic-bundle opening checks paths, declarations and supported recipe fields, without streaming every weight byte.

To render using the registration, append its filename to the example command after the seed. Append a repeat count after that to reuse the pipeline; output filenames get a numbered suffix. Use `-` for a dynamic-only repeat run.

Native carries stay in Corsairs between segments. Only the half-precision velocity returns to Rust each step. Authoritative latents remain float32, with a separate rounded multiply followed by an add, then narrowing to fp16 at denoiser and decoder boundaries. This preserves the correctness path proven by Corsairs. The lab's retained-encoder, parallel-load and shape-export results motivate these choices; this crate has no independent performance claim yet.

## Architecture and ownership

`ImagePipeline`, `ImageRequest`, and `GeneratedImage` form the image modality boundary. `krea` owns the Krea-specific template, tokenizer, shapes, asset selection and schedule. A new image family can implement `ImagePipeline` with its own configuration and conditioning. `AudioVideoPipeline`, `AudioVideoRequest`, and `GeneratedAudioVideo` provide a separate joint media boundary with incremental frame output and float PCM. No universal request enum or diffusion-specific base trait constrains other modalities.

Each pipeline owns or accepts an injected `corsairs::Runtime`. There are no global model or GPU handles. `&mut self` serializes generation on that pipeline. Krea's `CancellationToken` is cooperative between model calls; an already submitted native job completes under Corsairs ownership, including if the generation future is dropped. Krea's resident models remain available after cancellation. `unload()` clears models and prompt cache, then awaits `Runtime::drain()` for transitive release; the pipeline can be used again. In Krea, the same drain precedes denoiser switching and follows balanced encoding. H3 acquires and releases models by phase, with drain between phases; it has no cooperative cancellation-token API yet. Drain preserves externally held handles and does not promise OS cache/page eviction. Applications sharing a Runtime across multiple pipelines still own admission and total residency policy.

Krea segments are checked for compatible shapes, dtypes and upstream named carries before execution. Conditioning is preserved through all consumers, including nonadjacent segments. Request dimensions, steps, supplied noise, text rows and finite values are checked before large native execution. Unsupported recipes fail explicitly. Krea supports text-to-image; its reference-image conditioning, guidance, streaming residency, export authoring and tiled decoding remain future work. H3's export recipe and source tiled decoding are implemented separately in the audio/video modality.

## Verify

```sh
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo fmt --package aether -- --check
```

Optional input checks read the external lab without running Core AI:

```sh
AETHER_KREA_BUNDLE=/path/to/original-dynamic-bundle \
AETHER_KREA_FIXTURES=/path/to/fixtures/4c12ae019488ff1c \
AETHER_KREA_LAB=/path/to/krea-lab \
cargo test --offline --test lab_inputs -- --ignored
```

The native acceptance runner uses the lab's tower noise and compares all eight latent-step hashes and float32 pixel hashes. `encoded` also checks tokenization against saved IDs and the selected encoder rows against the separate full-encoding Swift baseline. `provided` uses saved text embeddings and their distinct baseline:

```sh
cargo run --offline --release --example parity -- \
  /path/to/original-dynamic-bundle /path/to/krea-lab \
  /path/to/fixtures/4c12ae019488ff1c .local/tower-20.json \
  .local/parity-provided provided
```

Run `encoded` separately with its own output directory. An optional final `serial` argument switches off batch loading for diagnosis. The runner verifies all model source streams, writes a structured report and PNG, and returns a failing status on any mismatch. Use a permitted native execution environment and run one GPU acceptance job at a time. A sandbox trap or unavailable native environment is not a passing comparison.

## Corsairs boundary

No missing v0.2 capability blocks this first pipeline: it uses `load_model_with_options`, `load_models`, dynamic descriptors, `upload`, `run_native`, `download`, and owned `run`. It enables no `abi-research` feature and contains no unsafe Rust.

To reproduce the lab's device-resident Euler loop, Aether still needs an application Metal adapter and kernel using Corsairs' queue/buffer interoperability. The current safe Corsairs API fences individual calls and offers no built-in Euler kernel; Aether currently updates latents on the CPU. This is an application integration/performance gap, not a demonstrated missing model-execution primitive in Corsairs. In-flight GPU pipelining and native interruption are outside its current guarantees.

See [VALIDATION.md](VALIDATION.md) for checks actually completed in this checkout.
