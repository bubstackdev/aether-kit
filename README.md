# AetherKit

Production generative AI on Apple Core AI, with model-specific optimization for Apple silicon.

Status: experimental implementation, 2026-10-17. The Krea reference image pipeline has passed native provided-conditioning and full encoded parity for the retained 768×1344 tower fixture, including all eight latent hashes and decoded pixels against each mode's independent reference. See [generation validation](docs/krea-generation.md) for scope and remaining gates. Publication is pending.

Requires Apple silicon, macOS 27 and Xcode 27.1 / Swift 6.4 for the currently tested build. The package also declares iOS 27; iOS builds and execution are not qualified yet. The optional tokenizer adapter uses Hugging Face's `Tokenizers` product; the core, execution and Krea contract targets do not link it.

| Product | Current implementation |
| --- | --- |
| `AetherAssets` | Immutable local file installation, SHA-256 verification and reader leases; no native execution |
| `AetherKit` | Owned tensors, typed errors and decoded RGB/PNG artifacts; no Core AI dependency |
| `AetherCoreAI` | Local model loading/introspection and a stateless typed reference executor |
| `AetherH3CoreAI` | Experimental BF16 runtime; eight-step provided-conditioning GPU gate passes |
| `AetherH3` | Experimental CPU H3 geometry, packing, Turbo8 schedule and export-header identity checks |
| `AetherKrea` | Bundle validation, prompt/conditioning contracts, rotary inputs, schedule and noise |
| `AetherKreaTokenizers` | Optional local tokenizer adapter; no model/tokenizer downloads |
| `AetherKreaCoreAI` | Reference Qwen → four denoisers → Euler sampling → VAE generation session |
| `aether-probe` | Device information, loaded-model inspection and affine native smoke check |

## Build and check

```sh
Scripts/check.sh
.build/arm64-apple-macosx/release/aether-probe doctor
```

The check script keeps caches inside `.build` and uses SwiftPM's deprecated native build engine to avoid a sandboxed `dsymutil` failure in this environment. On unrestricted machines, use `AETHER_BUILD_SYSTEM=swiftbuild Scripts/check.sh`, or normal `swift build` / `swift test -c release`.

Real-weight session qualification is opt-in. With the local bundle and CPU fixture paths configured, set `AETHER_NATIVE_QUALIFICATION=1`, `AETHER_KREA_SHAPE` to the tower registration JSON, and `AETHER_NATIVE_OUTPUT` to a new output directory before running `Scripts/check.sh`. It checks native model reuse, cancellation/recovery, unload during work and dynamic fallback at two additional shapes. Ordinary runs skip this test and do not load model weights. See [generation validation](docs/krea-generation.md) for the complete command and evidence.

Full eight-step dynamic parity and encoder/decoder cancellation use a separate opt-in suite: `AETHER_NATIVE_DYNAMIC_QUALIFICATION=1`, `AETHER_NATIVE_DYNAMIC_OUTPUT` set to a new directory, and `AETHER_KREA_REFERENCE_PROBE` pointing to the existing independent KreaLab executable. Targeted decoder checks use `AETHER_NATIVE_DECODER_QUALIFICATION` (isolated native decoder) or `AETHER_NATIVE_DECODER_PIPELINE_QUALIFICATION` (one-step native pipeline by default). Run one native suite at a time; ordinary runs skip all four. See [current qualification and limits](docs/krea-generation.md): lazy per-session native function reuse now passes full eight-step independent parity at both dynamic shapes, encoder cancellation with re-encoding recovery, and decoder cancellation with bit-exact recovery. The previous GPU-residency failures remain preserved as historical evidence.

This checkout's local Core AI exports live under `~/.cache/huggingface/aether-kit/`. Source `.local/native-assets.env` to set the cached Krea bundle, tower registration and affine fixture paths, plus the existing reference/fixture paths. Original export locations remain compatibility symlinks. These are local Core AI exports stored under the Hugging Face cache root; their model assets and registration are separate from Apple's derived `coreai-cache`.


A repeatable benchmark harness is available through `Scripts/benchmark-krea.sh`. Use `prepare NEW_OUTPUT_DIRECTORY` to validate the pinned two-workload plan without inference, and `run NEW_OUTPUT_DIRECTORY` during the quiet window. Warm-ups are separate from measurements; reports include stage timings, sampled memory, numerical hashes and source/model/environment identities. See the [benchmark workflow](docs/krea-benchmarks.md). Preparation does not establish a performance result. Matched encoded/provided comparison plans add three warm-ups and five measurements per workload, with one-time encoding excluded and recorded separately for the provided profile.

## Native integration probe

Use a Python environment with PyTorch and Apple's `coreai-torch` installed. Python is only needed to export this weight-free fixture, not to run the SDK.

```sh
python3 Forge/export_smoke.py ~/.cache/huggingface/aether-kit/new-smoke
.build/arm64-apple-macosx/release/aether-probe inspect ~/.cache/huggingface/aether-kit/new-smoke/affine.aimodel
.build/arm64-apple-macosx/release/aether-probe smoke ~/.cache/huggingface/aether-kit/new-smoke/affine.aimodel cpuOnly
```

The exporter requires a new output directory and records dependency versions and source/model hashes. To use the already exported fixture, source `.local/native-assets.env` and pass `"$AETHER_SMOKE_MODEL"` to `smoke`. `smoke` checks two executions and retained output after unload. `inspect` loads/specializes the model and can create a Core AI cache entry. Neither command downloads weights. Native execution needs normal access to Apple's model cache. On 2026-10-17, smoke checks passed with `cpuOnly`, `automatic` and `preferGPU` when run outside the agent sandbox. The sandbox itself currently fails to start because its configured cache writable root is a symlink; use approved execution outside the sandbox for these checks.

## Local Swift integration

Add `.package(path: "/path/to/aether-kit")` and the desired products to a consumer's `Package.swift`. For the reference executor:

```swift
import AetherKit
import AetherCoreAI
import Foundation

let session = try await ModelSession.open(at: modelURL)
do {
    let x = try FloatTensor(shape: [1, 4], values: [-2, 0, 1, 3])
    let result = try await session.run(inputs: ["x": x])
    await session.unload()
    print(result["y"]?.values ?? [])
} catch {
    await session.unload()
    throw error
}
```

This path copies inputs and outputs and rejects concurrent calls on one session. `run` is the float32 convenience API; `runTensors` supports float32, fp16 and int32. It is a correctness reference for runtime evaluation, not the eventual optimized generation API. The code example assumes the affine fixture's names and shape.

See [current validation](docs/implementation-status.md) and the [engineering documentation](docs/README.md). The Rust prototype remains in `../lab/aether-prototype` as the migration reference and is unchanged. Source uses Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). External model assets retain their own terms.

A [separate consumer package](Examples/LocalPackage) exercises the Krea product without loading Core AI or weights.

See [Krea asset and conditioning preparation](docs/krea-conditioning.md) for local bundle usage and fixture checks. The tokenizer dependency is pinned to swift-transformers 1.1.0; `Package.resolved` records this checkout's full dependency resolution.

The CLI exposes the same reuse through `aether-probe render-krea-batch BUNDLE NEW_OUTPUT_DIRECTORY PROMPT WIDTH HEIGHT SEED[,SEED...] [SHAPE.json] [--steps=N]`. It renders seeds serially after one preparation, saving indexed PNGs and a report with hashes and timing scope. See the [batch command example](docs/krea-generation.md#batch-rendering-from-the-cli).

For repeated images of one prompt, use `session.prepare(prompt:)` once, then `generate(configuration:preparedPrompt:)` on the same session. This releases the encoder after preparation and reuses conditioning, denoisers and decoder. Prepared values are caller-owned, report their conditioning byte count and reject incompatible bundles. See the [prepared-prompt example](docs/krea-generation.md#prepare-once-render-multiple-seeds).

See [Krea reference generation](docs/krea-generation.md) for shape registration, PNG rendering, and strict native parity commands. Segment tensors currently copy through host memory; this implementation makes no optimized performance claim.

MiniMax H3 has CPU preparation contracts in `AetherH3` and an experimental BF16 runtime in `AetherH3CoreAI`. The [eight-step provided-conditioning GPU gate](docs/h3-integration.md#denoising-qualification-2026-10-18) passes 464 scoped CPU comparisons and reproduces all 864 captured hashes with one resident block-weight set. [Earlier native lifecycle probes](docs/h3-integration.md#outer-phase-qualification-2026-10-18) pass per-module cancellation/recovery. Whole-trajectory lifecycle qualification, the conditioner, decoders and media acceptance remain pending; ordinary tests perform no H3 GPU work.

Optional SDK foundations include shared resource reservations and installed-asset model loading. See [contracts and validation](docs/sdk-foundations.md) and the [remaining implementation sequence](docs/sdk-roadmap.md).
