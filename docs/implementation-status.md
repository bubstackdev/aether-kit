# Initial implementation — 2026-10-17

This records the initial scaffold. The subsequent [Krea conditioning slice](krea-conditioning.md) adds assets, typed tensors, tokenizer integration and further validation.

## Implemented

- SwiftPM package with separately selectable AetherKit, AetherCoreAI and AetherKrea libraries, plus `aether-probe`. No external Swift dependencies or model weights.
- Owned, finite float32 tensors with shape/count/overflow validation.
- Direct public Core AI model loading, function descriptors and stateless float32 execution. Input names, declared shape/type and output type are checked before dispatch. Unsupported states/images/noncontiguous outputs fail explicitly.
- Session lifecycle rejects overlap and new work during close; cancellation retains the active slot until the backend completes; unload awaits active inference and releases session model ownership. Native framework behavior still needs qualification.
- Krea geometry limits, original flow schedule, separately rounded CPU Euler arithmetic and SplitMix64/Box–Muller noise.
- Tiny weight-free affine exporter recording recipe/dependency/content identities, and a CLI for doctor, inspect and two-run smoke checks.
- Apache-2.0 source license and inherited algorithm/fixture notices. This does not license any external model weights.

## Verification performed

Host: 64 GiB Apple silicon, Core AI architecture `h17c`, macOS 27.0.1 `26A434`; Xcode 27.1 `27A9275`, Swift 6.4.

`Scripts/check.sh` built Release and passed nine Swift Testing tests. Coverage includes independent retained schedule/noise bit fixtures, rounding, geometry/tensor validation, overlapping runs, failure recovery, cancellation waiting for controlled backend completion, and closing during suspended inference. These lifecycle tests use an injected controlled backend and establish no native cancellation guarantee.

The weight-free fixture exported successfully using torch 2.11.0 and coreai-torch 0.4.2. `aether-probe doctor` ran and reported CPU/GPU/Neural Engine availability. CPU-only native smoke exited 1 before inference because Core AI could not open its cache lock under `~/Library/Caches/coreai-cache`: `Operation not permitted`. This is an incomplete native gate, not a successful model execution.

The separate `Examples/LocalPackage` consumer built and ran in Release, resolving the local AetherKrea product and reproducing the 4,032-token geometry, eight-step schedule and seed-71344 noise bit patterns. This verifies a separate package boundary without Core AI execution.

The default Swift Build engine reached a `dsymutil` operation denied by the agent sandbox. The local check script uses SwiftPM's native engine, which passes but is deprecated. Recheck the default engine on an unrestricted machine before release. This is an environment workaround, not a permanent build-system commitment.

See [machine-readable initial evidence](evidence/scaffold-2026-10-17.json) for source/fixture identities and observed status. Smoke timing is not a performance benchmark.

## Remaining work

1. Complete the native affine gate in an environment with normal Core AI cache access, including GPU/automatic policies and malformed-input cases.
2. Inspect current Corsairs binding compatibility and compare matched Krea stages before resolving D02. The direct Swift executor is an experimental candidate.
3. Port Krea tokenization/conditioning, rotary inputs, asset manifests, residency and decoding, then reproduce the existing step/pixel fixtures.
4. Introduce device-resident tensors and sampling only after the reference path is qualified. No optimized throughput claim exists yet.
5. Build and qualify the sandboxed consumer app, physical iOS devices, model distribution, CI infrastructure and public release.

Earlier Git initialization attempts were denied by the environment. The user subsequently initialized the repository; see Git history for commits. No Git remote, public release, asset hosting or external publication has been configured.
