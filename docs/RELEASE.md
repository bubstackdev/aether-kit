# Releasing

How a release of Aether-Kit is cut. Tagging is the owner's action.

## Pre-tag checklist (0.1.0)

**Scope.** Every row of INTEGRATION.md's port table is ported and gated, including **MiniMax H3** (its exporter, its
oracle gate and the `minimax-h3@1` driver). The CHANGELOG's "In progress" items are either done or moved to a later
release.

**Gates.** Run them on a Mac with the models and Forge outputs present, serially, with nothing else heavy running.

```bash
AETHER_GATES=1 scripts/ci.sh
```

| Suite | File | What it checks | Needs |
| --- | --- | --- | --- |
| `Gates` (runtime) | `Tests/AetherRuntimeTests/GateTests.swift`, `EngineGateTests.swift` | the decoder engine: parity with the lab spike, chunked prefill, pipelined decode and its canary, speed | the lab models (`lab/models` or `AETHER_LAB_MODELS`) |
|  | `GenerativeGateTests.swift` | measured phase footprints see Core AI memory | qwen3-0.6b |
|  | `GraphChainGateTests.swift` | device chain and weight banks match the unchained path byte for byte | the committed fixture |
| `TextGateTests` | `Tests/AetherTextTests/TextGateTests.swift` | the chat flow, templates and stops on real models | the lab models |
| `FoundationModelsGateTests` | `Tests/AetherFoundationModelsTests/FoundationModelsGateTests.swift` | `LanguageModelSession`, `@Generable` and tools | the lab models |
| `Latent diffusion gates` | `Tests/AetherImageTests/LatentDiffusionGateTests.swift` | SD-Turbo from the seed vs CPU Diffusers | `forge-outputs/sd-turbo` + references |
| `FLUX.2 [klein] gates` | `Tests/AetherImageTests/Flux2KleinGateTests.swift` | FLUX.2 vs CPU Diffusers | `forge-outputs/flux2-klein-4b` + references |
| `Krea 2 gates` | `Tests/AetherImageTests/Krea2GateTests.swift` | Krea 2 vanilla and realism from the seed vs CPU Diffusers | `forge-outputs/krea-2-turbo*` + references (about 34 GiB a generation) |
| `Kokoro gates` | `Tests/AetherSpeechTests/KokoroGateTests.swift` | Kokoro vs unmodified upstream on CPU | `forge-outputs/kokoro-82m` + references |
| `MiniMax H3 gates` | `Tests/AetherVideoTests/MiniMaxH3GateTests.swift` | the driver's first steps vs the graph reference (0.02 rel. L2), a whole clip and its repeat, each phase's peak within the 18 GiB bar | a local `forge-outputs/minimax-h3` export (or `AETHER_H3_EXPORT`); the per-graph oracle `Forge/h3/check_h3.py` runs with the export |
|  | `Tests/AetherRuntimeTests/H3ChainGateTests.swift` | the export's block graph through `GraphChain` and `WeightBanks` vs `Forge/h3/reference_chain_h3.py`, chained bytes equal to unchained | `forge-outputs/minimax-h3-chain` (`export_h3.py --random`) or a real export (`AETHER_H3_CHAIN`) |
| `DepthEstimatorGateTests` | `Tests/AetherVisionTests/DepthEstimatorGateTests.swift` | Depth Anything 3 small vs CPU float32 upstream, preprocessing bit for bit, end to end | `forge-outputs/depth-anything-3-small` + references |
| `ObjectDetectorDenseGateTests` | `Tests/AetherVisionTests/ObjectDetectorDenseGateTests.swift` | YOLOX-S detections vs CPU float32 upstream | `forge-outputs/yolox-s` + references |
| `ObjectDetectorDetrGateTests` | `Tests/AetherVisionTests/ObjectDetectorDetrGateTests.swift` | RF-DETR nano and seg-nano detections and masks vs CPU float32 upstream | `forge-outputs/rf-detr-nano`, `rf-detr-seg-nano` + references |
| `ImageTextEncoderGateTests` | `Tests/AetherVisionTests/ImageTextEncoderGateTests.swift` | CLIP ViT-B/32 embeddings and zero-shot vs transformers on CPU | `forge-outputs/clip-vit-b32` + references |
| `TextEmbedderGateTests` | `Tests/AetherEmbeddingsTests/TextEmbedderGateTests.swift` | EmbeddingGemma token ids and embeddings vs CPU float32 upstream | `forge-outputs/embeddinggemma-300m` + references |

(`Tests/AetherCoreTests/ZooAdapterTests.swift`'s `signatures.json` fixtures are re-recorded only when a zoo bundle
changes: `AETHER_GATES=1 AETHER_RECORD_SIGNATURES=1 swift test --filter GateTests/recordSignatures`.)

**The GPU lease.** Kit processes share one cross-process lease, `/tmp/atelier-gpu.lock` (`SharedGPULease`), with
Atelier and the lab scripts. GPU work outside the kit (Forge checks and exporters, reference renders, one-off scripts)
takes it with `scripts/gpu-lease`, which waits for the lease, runs the command, and releases when it exits:

```bash
scripts/gpu-lease -- uv run --project Forge python Forge/yolox/check_yolox.py --bundle <bundle> --reference <reference>
AETHER_GATES=1 scripts/gpu-lease --timeout 600 -- swift test --filter KokoroGateTests
```

The command runs with `AETHER_GPU_LEASE_HOLDER` set to the wrapper's pid, so kit processes inside it inherit the lease
instead of waiting on it, and a nested `scripts/gpu-lease` just runs its command. The wrapper reports a busy holder
on stderr (`--quiet` hides it), passes the command's exit status through, forwards SIGINT, SIGTERM and SIGHUP to it,
and exits 75 on `--timeout`. `AETHER_GATES=1 scripts/ci.sh` already runs its gate step under it. Don't hand-roll the
`mkdir` protocol: the wrapper also does the stale-holder reclaim and the guarded release, interop-tested against the
Swift implementation in `SharedGPULeaseTests`.

**Device gate.** SD-Turbo (iOS bundle), Kokoro and the six M4 models on a physical iPhone, recorded in docs/DEVICE.md:

```bash
ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… Tools/DeviceGate/run-on-device.sh <udid>
```

**Catalog and hosting.**

- Every catalog entry is pinned to a 40-character revision with per-file SHA-256: `scripts/gen-catalog refresh`,
  then `scripts/gen-catalog check`. The CI step runs the check.
- Each hosted bundle carries its license and NOTICE.
- The Krea repos begin with "Krea", and the realism repo is gated and marked mature.
- H3 is not publicly hosted and has no catalog entry: its license excludes the EU, UK, South Korea and USA. The README
  and CHANGELOG say it needs a locally exported bundle.

**Licenses.** NOTICE lists only third-party work re-expressed in this repository. The per-model terms are in the
README's model table and in each bundle.

**Docs.** CHANGELOG.md's 0.1.0 section is final, with no "in progress" items. The README's status line and the
`branch: "main"` dependency line read as released.

## Release steps

1. On `main`, after the last port has merged, build and refresh the public-API baseline. Review the diff, which should
   hold only intended changes, and commit it:
   ```bash
   swift build --scratch-path .build-ci --build-tests
   scripts/check-api.sh --update
   ```
2. Freeze the API: set `"enforce": true` in `scripts/api-freeze.json`. From then on, CI fails a breaking change, and
   also a compatible change not yet recorded in `api/`.
3. Finalize CHANGELOG.md (the date, no "in progress" section) and the README:
   - the status line;
   - the quick-start dependency `.package(url: …, from: "0.1.0")`.
4. Open the release PR with steps 1–3. `scripts/ci.sh` must be green locally and in CI ("CI passed"). Run the full
   checklist above on that commit.
5. Merge the PR.
6. **Owner:** tag the merge commit `0.1.0` and push the tag, then publish the GitHub release with the CHANGELOG
   section as its notes.

After the tag, an intended API change updates `api/` in the same PR (`scripts/check-api.sh --update`). A breaking
change waits for the next minor version.
