# v0.0.1 release checklist

Status: in progress, 2026-10-26. Unchecked items are open; checked items cite their evidence. This release is cut from `main`'s line (branch `release/0.0.1`, based on `c62ac37`). A parallel local line developed from the same scaffold is preserved as `local-line-2026-10-26`; its language runtime, concurrent stream verification and benchmark residency option were ported here.

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
- [ ] **CPU suite** on the release commit: `sh Scripts/check-cpu.sh`.
- [ ] **Clean clone:** resolve, test, and build both examples (`Examples/LocalPackage` imports every product and type-checks the README snippets; `Examples/Foundations` is CI's independent consumer).
- [ ] **Warnings:** only the known Core AI `ComputeStream` sending diagnostics remain.

## Qualification on this line

- [ ] **Krea native suites:** dynamic parity, session lifecycle and decoder pipeline, using the byte-identical regenerated tower fixture at `~/.cache/huggingface/aether-kit/krea-2-turbo-realism/fixtures/4c12ae019488ff1c`.
- [ ] **Krea performance:** re-measure provided, balanced and resident fresh-prompt timings on this line's session.
- [x] **H3:** not rerun for 0.0.1 (decision above).
- [x] **Language:** CPU-qualified in this repository. On-device qualification is tracked by the consuming app.

## Documentation

- [x] **README:** maturity table, support matrix, toolchain floor, quick starts, known limitations.
- [x] **CHANGELOG.**
- [x] **DocC:** experimental catalogs for the language and H3 modules, and an updated `AetherKit` overview.
- [ ] **Release notes** with the qualification results.

## Deferred to 0.1.0

- **Product gaps:** a Krea bundle specification, hosted assets and download transport, cold-start reduction, asynchronous bounded progress delivery, and the language export contract (fixed-shape decode removes the 128-submission reload).
- **Hardening:** a soak test, slow-consumer behaviour, cancellation at every stage, concurrent-session admission with qualified estimates, a sandboxed consumer app, an equipped CI runner and an iOS compile check.

## Tag and verify

- [ ] Annotated tag `0.0.1` on the release commit.
- [ ] Push `release/0.0.1` and the tag; land the branch on `main` by pull request or fast-forward.
- [ ] Install check: a fresh consumer resolves `.package(url:…, exact: "0.0.1")` and builds.
