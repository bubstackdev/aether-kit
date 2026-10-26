# v0.0.1 release checklist

Status: released 2026-10-26 (`0.0.1` → `d3da985`). Unchecked items are open; checked items cite their evidence. This release is cut from `main`'s line (branch `release/0.0.1`, based on `c62ac37`). A parallel local line developed from the same scaffold is preserved as `local-line-2026-10-26`; its language runtime, concurrent stream verification and benchmark residency option were ported here.

## Scope and decisions

- **Products:** every current product ships.
  - Preview: `AetherKit`, `AetherAssets`, `AetherCoreAI`, `AetherKrea`, `AetherKreaTokenizers`, `AetherKreaCoreAI`.
  - Experimental: `AetherLanguageCoreAI` and the three H3 products.
  - `aether-probe` stays a product, documented as a dev tool.
- **What 0.0.1 promises:** an experimental developer preview with no API compatibility promise in 0.0.x. No model weights are distributed.
- **Support bar:** qualified on one Apple M5 Max running macOS 27.0.1. Other Macs and iOS are listed as unqualified.
- **Toolchain:** Xcode 27.1 / Swift 6.4, with the experimental `Lifetimes` feature accepted.
- **Krea residency:** `balanced` stays the default; `resident` is a documented opt-in.
- **H3:** ships as experimental. The 2026-10-18 prototype-bundle gates are historical, because their assets were lost. Schema-3 `H3VideoSession` qualification is 0.0.2 work.
- **`supportsReasoning`:** kept as consumer metadata that Kit doesn't act on.
- **Release build system:** SwiftPM's default engine; `native` stays an opt-in for restricted sandboxes.
- **Tag:** `0.0.1`.
- **Evidence:** committed as-is.

## Repository and package

- [x] **Ports onto the release branch:**
  - language runtime (`4e26f33`)
  - concurrent stream verification (`83820e0`)
  - benchmark residency option (`ded78b5`)
- [x] **Local line preserved:** remote branch `local-line-2026-10-26`.
- [x] **Third-party notices:** `NOTICE` lists swift-transformers, swift-jinja and swift-collections, used by both tokenizer products.
- [x] **Scripts:** `Scripts/check.sh` and `Scripts/benchmark-krea.sh` default to the default build engine.
- [x] **CPU suite:** 246 Release tests pass on `b091f58` (`sh Scripts/check-cpu.sh`), with the opt-in native and external-fixture tests skipped.
- [x] **Clean clone:** a fresh clone of `b091f58` resolves, passes the CPU suite, builds `Examples/LocalPackage` (every product, with the README snippets type-checked) and runs `Examples/Foundations`.
- [x] **Warnings:** the only warnings are the 24 known Core AI `ComputeStream` "sending risks data race" diagnostics.

## Qualification on this line

- [x] **Krea native suites:** all passed on this line, using the byte-identical regenerated tower fixture ([record](evidence/krea-release-native-2026-10-26.json)): dynamic parity (65 checks), session lifecycle (23) and decoder pipeline (10).
- [x] **Krea performance:** measured on this line under heavy concurrent load ([record](evidence/krea-release-benchmarks-2026-10-26.json)): resident 5.0 s, matching the quiet-machine figure; balanced fresh prompts reload-bound. A quiet-window balanced re-measure is a follow-up.
- [x] **H3:** not rerun for 0.0.1 (decision above).
- [x] **Language:** CPU-qualified in this repository. On-device qualification is tracked by the consuming app.

## Documentation

- [x] **README:** maturity table, support matrix, toolchain floor, quick starts, known limitations.
- [x] **CHANGELOG.**
- [x] **DocC:** experimental catalogs for the language and H3 modules, and an updated `AetherKit` overview.
- [x] **Release notes:** [0.0.1](release-notes/0.0.1.md).

## Deferred to 0.1.0

- **Product gaps:** a Krea bundle specification, hosted assets and download transport, cold-start reduction, asynchronous bounded progress delivery, and the language export contract (fixed-shape decode removes the 128-submission reload).
- **Hardening:** a soak test, slow-consumer behaviour, cancellation at every stage, concurrent-session admission with qualified estimates, a sandboxed consumer app, an equipped CI runner and an iOS compile check.

## Tag and verify

- [x] Annotated tag `0.0.1` on `d3da985`.
- [x] Pushed `release/0.0.1`, fast-forwarded `main` from `c62ac37` to `d3da985`, and pushed the tag.
- [x] Install check: a fresh consumer resolved `.package(url: "https://github.com/atelier-private-dev/aether-kit", exact: "0.0.1")` to `d3da985`, then built and ran against Kit, Assets, Krea, language and H3. Consumers resolve their own transitive tokenizer dependency versions (swift-collections 1.7.4 and swift-jinja 2.5.2 in that check); `NOTICE` cites this repository's lockfile versions.
