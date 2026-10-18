# AetherKit

Production generative AI on Apple Core AI, with model-specific optimization for Apple silicon.

Status: initial implementation, 2026-10-17. APIs are experimental; no published release or complete generation pipeline exists yet.

Requires Apple silicon, macOS 27 and Xcode 27.1 / Swift 6.4 for the currently tested build. The package also declares iOS 27; iOS builds and execution are not qualified yet. The optional tokenizer adapter uses Hugging Face's `Tokenizers` product; the core, execution and Krea contract targets do not link it.

| Product | Current implementation |
| --- | --- |
| `AetherKit` | Owned float32/fp16/int32 tensors and typed errors; no Core AI dependency |
| `AetherCoreAI` | Local model loading/introspection and a stateless typed reference executor |
| `AetherKrea` | Bundle validation, prompt/conditioning contracts, rotary inputs, schedule and noise |
| `AetherKreaTokenizers` | Optional local tokenizer adapter; no model/tokenizer downloads |
| `AetherKreaCoreAI` | Experimental resident Qwen text encoder and conditioning row selection |
| `aether-probe` | Device information, loaded-model inspection and affine native smoke check |

## Build and check

```sh
Scripts/check.sh
.build/arm64-apple-macosx/release/aether-probe doctor
```

The check script keeps caches inside `.build` and uses SwiftPM's deprecated native build engine to avoid a sandboxed `dsymutil` failure in this environment. On unrestricted machines, use `AETHER_BUILD_SYSTEM=swiftbuild Scripts/check.sh`, or normal `swift build` / `swift test -c release`.

## Native integration probe

Use a Python environment with PyTorch and Apple's `coreai-torch` installed. Python is only needed to export this weight-free fixture, not to run the SDK.

```sh
python3 Forge/export_smoke.py .local/smoke
.build/arm64-apple-macosx/release/aether-probe inspect .local/smoke/affine.aimodel
.build/arm64-apple-macosx/release/aether-probe smoke .local/smoke/affine.aimodel cpuOnly
```

The exporter requires a new output directory and records dependency versions and source/model hashes. `smoke` checks two executions and retained output after unload. `inspect` loads/specializes the model and can create a Core AI cache entry. Neither command downloads weights. Native execution needs normal access to Apple's model cache; the agent shell's restricted cache access currently prevents this check from completing.

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
