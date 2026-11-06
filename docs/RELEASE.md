# Releasing

How a release of Aether-Kit is cut. Tagging is the owner's action.

## Pre-tag checklist (0.1.0)

**Scope.** Every row of INTEGRATION.md's port table is ported and gated. That includes **MiniMax H3**, its exporter, its
oracle gate and the `minimax-h3@1` driver; until H3 lands, 0.1.0 is not tagged.

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
| H3 gates | added with H3 | the per-graph oracle (0.02 rel. L2) and the memory gate at the qualified canvas | `forge-outputs/minimax-h3` |

(`Tests/AetherCoreTests/ZooAdapterTests.swift`'s `signatures.json` fixtures are re-recorded only when a zoo bundle
changes: `AETHER_GATES=1 AETHER_RECORD_SIGNATURES=1 swift test --filter GateTests/recordSignatures`.)

**Device gate.** SD-Turbo (iOS bundle) and Kokoro on a physical iPhone, recorded in docs/DEVICE.md:

```bash
ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… Tools/DeviceGate/run-on-device.sh <udid>
```

**Catalog and hosting.**

- Every catalog entry is pinned to a 40-character revision with per-file SHA-256: `scripts/gen-catalog refresh`,
  then `scripts/gen-catalog check`. The CI step runs the check.
- Each hosted bundle carries its license and NOTICE.
- The Krea repos begin with "Krea", and the realism repo is gated and marked mature.
- H3 is not publicly hosted: its license excludes the EU, UK, South Korea and USA.

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
