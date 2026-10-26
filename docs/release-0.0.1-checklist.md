# v0.0.1 release checklist

Status: proposed, 2026-10-26. Unchecked items are open; checked items cite their evidence.

## Scope

- **What ships:** every current product. The Krea stack (`AetherKit`, `AetherCoreAI`, `AetherKrea`, `AetherKreaTokenizers`, `AetherKreaCoreAI`) is the reference pipeline. `AetherLanguageCoreAI`, `AetherH3` and `AetherH3CoreAI` ship labelled **experimental**.
- **Where:** a tag in this repository. Hosting, publication and package-index submission belong to the stricter `0.1.0` plan in [qualification and release](qualification-and-release.md).
- **What 0.0.1 promises:** an experimental developer preview. There is no API compatibility promise within 0.0.x, and no model weights are distributed. Every performance or correctness claim names its device, OS and toolchain; anything untested is stated as unqualified.

## Decisions (2026-10-26)

- **Evidence:** commit all of `docs/evidence` as-is so every doc link resolves.
- **Branch:** commit directly on `main`.
- **Krea residency:** keep `balanced` as the default; document `resident` as the opt-in for machines with headroom.
- **H3 modulation tables:** keep recomputing per run; retention is deferred past 0.0.1.
- **`supportsReasoning`:** keep it as consumer-facing metadata and document that Kit doesn't act on it.
- **H3 public API:** stays public, presented with experimental/qualification-only banners in the README and DocC (no `@_spi`).
- **`aether-probe`:** stays a product, documented as a qualification and benchmark tool, not SDK surface.
- **Tag:** `0.0.1`, no `v` prefix.
- **Support bar:** qualified on one Apple M5 Max running macOS 27.0.1; other Macs and iOS are listed as unqualified.
- **`Lifetimes` feature:** accepted; the Xcode 27.1 / Swift 6.4 toolchain floor is documented.
- **Language close/unload:** added before 0.0.1.
- **Release build system:** SwiftPM's default engine; `native` stays an opt-in for the agent sandbox.

## 1. Repository state

- [x] **Commit the working tree.** Organized into reviewable commits on `main` (2026-10-26). An external edit to `docs/language-generation.md` (consumer iPhone qualification) is left to its author.
- [x] **Evidence policy.** Commit all 248 files (11 MB) as-is. Pruning raw logs into an archive remains optional follow-up.
- [x] **Ignore generated files.** Add `Scripts/__pycache__/` to `.gitignore`. `.ai-radio/` already ignores itself.
- [x] **Remove experiment-only code.** The temporary `AETHER_EXPERIMENT_PREPARE` toggle was removed with the concurrent-preparation change after the A/B ([record](evidence/krea-residency-load-path-2026-10-26.json)).
- [ ] **Audit findings.** Fix [CODEBASE-REVIEW](../CODEBASE-REVIEW.md) M1 (the qualification wrappers leave a detached probe on SIGTERM/SIGHUP), or list it as a known issue in the scripts. Then move the review into `docs/` or retire it.

## 2. Package and build

- [x] **Clean clone.** A fresh clone of `93a769c` resolves from `Package.resolved`, passes all 136 Release tests on the default build engine, and builds the consumer example. `Scripts/check.sh` defaults to that engine, with `AETHER_BUILD_SYSTEM=native` as the sandbox opt-in.
- [x] **Consumer example.** `Examples/LocalPackage` depends on every library product and type-checks the README examples; it builds and runs.
- [x] **Toolchain floor.** The README states Xcode 27.1 / Swift 6.4 and the `Lifetimes` requirement; the separate example package builds with it.
- [ ] **Warnings.** The release build is warning-free apart from the eight known Core AI `ComputeStream` "sending risks data race" diagnostics. List those as SDK interop warnings or remove them.
- [x] **Third-party notices.** `NOTICE` lists swift-transformers, swift-jinja and swift-collections (all Apache-2.0).
- [ ] **iOS compile check.** The package declares iOS 27. Build every library product for a generic iOS destination. This is a compile claim only; on-device execution stays unqualified.
- [ ] **`aether-probe`.** Keep it as a product and document it as a qualification and benchmark tool, not SDK surface.

## 3. Requalification after the 2026-10-26 changes

- [x] **CPU suite.** 133 Release tests pass; the opt-in native and external-fixture suites are skipped. Rerun on the release commit.
- [ ] **Krea native suites.** The dynamic parity suite passed on the release candidate: 65 checks, including real-bundle concurrent verification ([record](evidence/krea-release-dynamic-2026-10-26.json)). The `AETHER_NATIVE_QUALIFICATION` and decoder pipeline suites need the tower fixture directory, which was deleted and is being regenerated.
- [ ] **Krea benchmarks.** Rerun the matched encoded/provided comparison, plus a resident plan, on the release commit and replace the October 18 figures. Those showed 2.08 s per step; [2026-10-26 runs](krea-benchmarks.md#residency-and-load-path-2026-10-26) show about 0.535 s at 512×512 with identical outputs.
- [x] **H3 trajectory.** Resolved by reverting the H3 verification change (`998f72f`) for 0.0.1. The shipped H3 library sources are byte-identical to those pinned by the October 18 GPU gates. The qualification inputs (exported models, fixtures, render bundle) were deleted by macOS temporary-file cleanup, and Core AI exports aren't byte-reproducible. Regeneration into `/Volumes/Scratch` plus a full capture/CPU-reference/qualification redo is a post-0.0.1 project, after which the change can be re-landed.
- [ ] **H3 whole-run lifecycle.** Complete `--lifecycle`. The last attempt stopped at user request on 2026-10-19 ([record](evidence/h3-denoising-lifecycle-interruption-2026-10-18.json), [handoff](handoffs/h3-denoising-lifecycle-2026-10-18.md)). Otherwise state it as unqualified in the release notes.
- [ ] **Language native check.** The `onToken` overlap is tested on CPU only. Run a registered model with streaming against the consumer's reference fixtures, and record time to first token and tokens/s. An opt-in native test inside aether-kit would remove the dependency on external fixtures.

## 4. Product readiness

### Krea: reference pipeline

- [ ] **Bundle format.** `Forge/` contains only the smoke exporter, so consumers have no way to produce a bundle. Either write a bring-your-own-export spec (the `aether.json` recipe, `metadata.json` assets, `main.hash`, provenance pins and the shape-registration JSON), or add the export recipe.
- [x] **Residency guidance.** `balanced` stays the default; the README and [benchmark notes](krea-benchmarks.md#residency-and-load-path-2026-10-26) document `resident` as the opt-in.
- [ ] **Cold start.** The first image in a process took 55–79 s, mostly first model opens. Document this, and evaluate ahead-of-time specialization through `AIModel.specialize`.
- [ ] **Progress callbacks.** These are synchronous and awaited within generation, so a slow observer stalls sampling. Document it, or move to bounded asynchronous delivery.
- [ ] **Requirements table.** Memory and timing at each tested shape (512×512, 384×512, 768×1344 tower), with residency mode.

### Language: experimental

- [x] **Streaming overlap.** `generate(onToken:)` delivers tokens while the next step runs, with ordered delivery and joined errors (`LanguageDecodeLoop`, four CPU tests).
- [x] **Verification memory.** The manifest hash loop drains each chunk, keeping memory bounded on multi-GB models.
- [x] **Explicit close/unload.** `unload()` rejects new requests with `closed`, waits for an active request, then releases the model, function, stream and KV caches (`LanguageSessionLifecycle`, three CPU tests).
- [ ] **Export contract.** Document the full running `position_ids`, the five-dimensional fp16 KV cache and the model reload every 128 submissions, including its cost. Record the planned fixed-shape decode export as follow-up work.
- [x] **`supportsReasoning`.** Documented in the [language guide](language-generation.md) as consumer metadata that Kit does not act on.
- [ ] **Open cost.** `open(bundle:)` re-hashes the whole model on every open. Document the cost, or cache verification by file identity.
- [ ] **Registration example.** A complete `language-model.json` example with field reference.

### H3: experimental

- [x] **Label the surface.** The README and the new `AetherH3` / `AetherH3CoreAI` DocC catalogs mark them experimental and qualification-only.
- [ ] **Assets.** The bundles are prototype exports, and manifests can carry absolute paths; relocation goes through the plan initializers. Document how to point a plan at local exports.
- [x] **Modulation tables.** Deferred past 0.0.1. Retention would remove 50 model opens per run but changes the qualified lifecycle trace, which expects 408 time/modulation submissions on every run.
- [ ] **Known limits.** Document the 10.6 GiB peak footprint and the 20/18 GiB limits, and that weight streaming isn't double-buffered yet.

## 5. Documentation

- [x] **README.** Purpose, requirements, tested matrix, product table with maturity labels (language is missing today), and a minimal Krea example. The current status line is from October 17.
- [ ] **DocC.** Getting-started articles for each library product. Only `AetherKit` has a catalog, and it still describes the scaffold. The docs build without warnings.
- [ ] **Support matrix.** Device, OS build, Xcode and per-product qualification. The accepted 0.0.1 bar is one Apple M5 Max on macOS 27.0.1 (26A434); other Macs and iOS are listed as unqualified.
- [ ] **Performance page.** Refresh [the benchmark doc](krea-benchmarks.md) with current numbers, methodology and the cold/warm distinction.
- [ ] **Consolidation.** Turn [implementation status](implementation-status.md) and the [docs index](README.md) into a qualification log linked from user docs, rather than the entry point.
- [ ] **Known limitations.** Collect them in one place per product.
- [x] **CHANGELOG.** Add `CHANGELOG.md` with a 0.0.1 entry.
- [ ] **License review.** Confirm Apache-2.0 for the SDK, keep the derivation notices, and state that model weights carry their own licenses and Krea distribution terms are unresolved (decision D03).

## 6. Hardening

These are recommended for 0.0.1 and required by the 0.1.0 gates.

- [ ] **Soak test.** 100 or more Krea generations in one session with flat settled memory; the same for a language session across reloads.
- [ ] **Slow consumers.** Measure generation with a slow progress observer and a slow `onToken` consumer, and document the effect.
- [ ] **Cancellation everywhere.** Cancel during preparation, model loading, function preparation, generation and decode, and verify recovery.
- [ ] **Concurrent sessions.** Global memory admission is unimplemented; document the risk of opening several sessions, or add an admission check.
- [ ] **Sandboxed app.** Run a sandboxed consumer app that uses only documented storage locations. This is untested.
- [ ] **CI or release script.** At minimum, a `Scripts/release-check.sh` that runs the clean build, CPU tests, example build, iOS compile and docs build. A self-hosted Apple-silicon runner can follow.

## 7. Tag and verify

- [ ] **Freeze and record.** Run sections 2 and 3 on the release commit and record the results in `docs/evidence/release-0.0.1-<date>.json`, with source, executable and model identities.
- [ ] **Tag.** Tag `0.0.1`.
- [ ] **Install check.** A fresh consumer package resolves `.package(url:…, exact: "0.0.1")` and builds the extended example.
- [ ] **Release notes.** Scope, maturity labels, the tested matrix, asset identities used for qualification, known limitations, and links to evidence.
