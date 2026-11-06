# Changelog

## 0.1.0 — unreleased

A from-scratch re-architecture of AetherKit. 0.0.x continues at the `0.0.7` tag and on `legacy/0.0.x`.

### Breaking

- **Platform floor:** macOS 27 / iOS 27 (0.0.x linked at macOS 14 / iOS 17 and reported `unsupportedOS` below 27).
- **All products renamed or replaced:** AetherCore, AetherRuntime, AetherText, AetherFoundationModels, AetherHost
  (and the planned AetherVision, AetherEmbeddings, AetherDecide, Aether). The 0.0.x frozen APIs (the `AetherLanguage`
  facade, `AetherAssets` preparation and readiness, `@AetherGenerable`) are not carried over. See the README's
  migration notes.
- **Model bundles** are described by the `aether.json` manifest in docs/MANIFEST.md. The 0.0.x prototype manifests are
  rejected.
- **`from: "0.0.7"` dependencies will resolve to 0.1.0.** Pin `exact: "0.0.7"` or `.upToNextMinor(from: "0.0.7")` to
  stay.

### New

- **Pinned catalog and manifest:** a catalog with per-file SHA-256 and no runtime catalog fetch. Manifest schema v1,
  with a zoo-bundle adapter driven by a reviewed override table.
- **Owned decode engine** on Core AI (`DecoderEngine`):
  - memory-sized KV;
  - manifest-driven prefill (prefill phases, ≤ 512-token chunks);
  - O(1) rewind, and snapshots for hybrid models;
  - token-exact depth-3 pipelined greedy decode (about 500 tok/s on qwen3-0.6b, M5 Max) guarded by a load-time canary.
- **Generator and ChatSession:** prefix reuse across turns, stop and tag parsing, a seeded sampler,
  JSON-schema structured output, Hermes tool calls.
- **`AetherLanguageModel`** for FoundationModels' `LanguageModelSession`, with guided generation and tools.

### Carried over from 0.0.x (re-implemented, validated against 0.0.x's tests and fixtures)

- **Store:** staged SHA-256 installs and resume, plus reader leases, stage recovery, cross-process install locks, a
  free-space check, token and redirect safety, and iOS background downloads.
- **AetherHost:** the GPU lease (Atelier-compatible) and admission, plus lifecycle (idle unload, memory-pressure
  release, iOS easing).
- **Text:** the sampler (matching HF Transformers' recorded processors), the structured-output constraint, and the
  text-stream behaviour, plus the tokenizer oracle cases.
- **Generative foundation (G0):**
  - the seeded noise stream (bit-identical to 0.0.x);
  - the Krea and FLUX.2 flow-matching schedule arithmetic;
  - `GeneratedImage` and `GeneratedAudio`;
  - `ResourceCoordinator` reservations;
  - new on top: typed `schedule`, `latent` and `residency` manifest blocks (MANIFEST.md §5), and `GraphResidency`
    with Metal-measured phase footprints.
- **Public-API freeze check** (`scripts/check-api.sh`, run by `scripts/ci.sh`): `swift-api-digester` dumps of
  AetherCore, AetherRuntime, AetherHost, AetherText and AetherFoundationModels compared with `api/`. Report-only until
  the 0.1.0 freeze sets `"enforce": true`.

### In progress before 0.1.0 is tagged

Krea 2 Turbo (vanilla and realism), MiniMax H3, FLUX.2 [klein], SD-Turbo and Kokoro-82M on the new runtime.
