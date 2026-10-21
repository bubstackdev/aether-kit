# Validation — 2026-10-15

The crate and all eight examples build with Rust 1.99.0 and published Corsairs `0.3.0`, resolved from crates.io with its archive checksum verified. No Corsairs or Atelier files were modified by this task. Aether's runtime contains no unsafe code and enables no Corsairs research feature. Checked tensors and their native runtime qualification belong to Corsairs. Earlier captures below used development sources; they are not retroactively assigned to the published release.

## Passed

- Release test suite: **27 passed**, including nine optional external-input/media tests. H3 checks compare actual packed rows, every T2VA timestep assignment, all 50 base and eight Turbo source scheduler updates, patch order and source temporal decoder stitching. Preparation-cache tests cover model/adapter/schedule/timestep identity and reject corruption and incompatible shapes.
- All six saved fixture prompts produce exactly the saved token IDs and conditioning-row counts. Long-prompt truncation preserves the five-token chat suffix.
- The 768 × 1344 rotary table matches every byte of the independent saved Swift input.
- All eight model-timestep and Euler-delta bit patterns match the independent Swift schedule oracle.
- Noise for seeds 0, 42 and 71344 matches the lab's unchanged Swift `SeededNoise` implementation, eight float32 values per seed.
- The release Euler test distinguishes separately rounded multiplication/addition from fused multiply-add.
- Image scaling/rounding, PNG roundtrip, invalid dimensions, nonfinite data, shape selection, source-identity mismatch and tampered bytecode checks pass.
- Both native acceptance attempts completed full source-stream SHA-256 verification for the dynamic models and registered specialized denoisers before attempting execution.
- `cargo clippy --offline --all-targets -- -D warnings`, formatting, and documentation checks pass.

The optional input tests used the original dynamic bundle and fixtures under `/Users/zakkeown/Code/aether-engine/forge/out/krea-2-turbo-realism/`, and the lab at `/Users/zakkeown/Code/atelier/lab/krea-lab`. Weights remain external. The shape registration was generated from the lab's retained `records/round4/fixed-assets.json`.

## Native Krea provided-text and prompt-encoding acceptance pass

A subsequent controlled native run through REA completed all eight Krea latent steps and decoded float32 pixels, each matching the independent Swift reference exactly. It used parallel denoiser loads, the verified 4,032-token/20-row specialization and the saved text/noise fixtures.

- Final latent: `3869b757030e018ed74b48644dcb013179c6059633540e160ace531ea83f0d74`.
- Decoded pixels: `4fa6be3acbb37f7e0b52f6b7b0cbe0376757af308d333d436641fa8a0152f8c4`.
- [Report](docs/evidence/krea-provided-native.json), [PNG](.local/parity-provided-native/image.png), and [controlled native capture](.local/krea-native-regression-observation.json), Evidence `ev_1517151fb377c77ef01933063153449eb7e206b0ee4caa0fc356a15e868ebf47`.

The separate encoded gate also passes. The runner tokenizes the actual prompt string, verifies its saved IDs, executes the original fp16 encoder, selects the 20 conditioning rows, runs all eight latent steps and decodes. Every text/step/pixel hash matches the independent full-encoding Swift baseline:

- Selected text: `10c6d354ae88f4f33d950ef9f9d45e06d8a0a2308dbcc6576e7968a4480f7cab`.
- Final latent: `898f4ff656cedfca1348b0b139fc07145252bdf0e9fff86d49278d27b2bde788`.
- Decoded pixels: `9b40e6017aabb0fddb81c29e11f928b9ceb4eda4d8e9639f49a9508e88937d25`.
- [Report](docs/evidence/krea-encoded-native.json), [PNG](.local/parity-encoded-native/image.png), and [native capture](.local/krea-encoded-retry-observation.json), Evidence `ev_1d311629e61438b686ce3a3e97c843cb938db3af332c89c911e997f0244aac23`.

An initial encoded attempt was terminated by the capture tool's idle timeout during its silent first encoder stage (Evidence `ev_1a95f1b92e6d1a700691921dc6ff76dcf58ba1e37fee0052f676654478a848f3`). It is not a pass. The completed retry explicitly set both total and idle allowances to 300 seconds. The recorded generation interval was about 200 seconds, including first encoder preparation; it is a correctness observation and does not measure warm throughput.

The native command exited zero. This single instrumented correctness run establishes no speedup or warm-performance result. The native provider records a prelaunch executable sample, not an atomic OS image binding, and does not capture the complete inherited environment.

## Earlier ordinary-shell failure retained

The provided-text tower acceptance run failed at the first model-loading phase in both loading modes:

| Loading | Error |
| --- | --- |
| Parallel batch | `Core AI load_models_batch failed: CoreAIDelegates.AIModelCacheError` |
| Serial control | `Core AI load_model failed: CoreAIDelegates.AIModelCacheError` |

Neither ordinary-shell run reached a denoiser call or produced an image. Both retain `passed: false`. The error does not identify the cache path or underlying I/O cause. It is consistent with the native model-cache restrictions described in [Corsairs compatibility](../corsairs/docs/compatibility.md). The successful controlled native run above exercises a different execution environment; it does not retroactively turn the failed shell checks into passes.

Local reports are retained at [.local/parity-provided/report.json](.local/parity-provided/report.json) and [.local/parity-serial/report.json](.local/parity-serial/report.json). The [README](README.md) includes reproduction commands for both provided-text and full-encoding acceptance. These are distinct reference baselines.

## Remaining integration

Corsairs exposes every execution primitive used by the implemented Krea baseline. Both the provided-text gate and the separate full prompt-encoding gate now pass. Residency reuse, cancellation/drop recovery, and performance still need their own native coverage.

Matching the lab's device-resident loop additionally needs application Metal kernel integration for Euler/float16 preparation. This version uses retained native carries with an exact CPU Euler step. Resident-encoder, parallel-loading and exact-shape policies follow the lab's validated choices; their performance has not been measured in this Rust application. Cancellation/drop recovery and residency reuse have been implemented but have not received end-to-end native validation here.

## MiniMax H3

**QKV correction:** earlier H3 transformer exports omitted raw-shard per-head QKV normalization. Their byte/numerical gates validate those bytes and APIs, but are superseded as semantic checkpoint qualification. Corrected exports pass the 27 Rust tests, a new independent QKV sentinel regression, the canonical 1,100-binding audit and CPU phase checks. Rust rejects manifests without the canonical-layout marker. The corrected native gate and complete render now pass; see [the correction and native result](docs/h3-turbo.md).

The bounded block gates remain available. Aether additionally exports and runs the complete T2VA path. The 128×128/22-frame provided-conditioning native diagnostic completes all 50 joint steps, all 50 blocks per step, both decoders, PNG frames and stereo PCM. Native Qwen encoding and input preparation pass separate original CPU oracle comparisons; raw video/audio decoding also passes. This diagnostic is outside the production duration/geometry matrix and makes no visual-quality, speed or full-trajectory byte-parity claim. The 960×544/124-frame production attempt was stopped at the user’s request after 22/50 steps because it was too slow; neither decoder ran. See [H3 validation](docs/h3-validation.md) for the retained reports and source bounds.

The subsequent [published-v0.3 base benchmark](docs/h3-benchmarks.md) measured three complete production-profile steps after 275.6 seconds of preparation: 35.7, 45.7 and 54.2 seconds. All three video/audio latent hashes match the earlier development run. Step times were rising; that attempt established no stable throughput or decoder time. The arithmetic 50-step projection is roughly 35–50 minutes before decoding. The user set a **20-minute maximum from start to finished clip**, including preparation and decoding. The base benchmark was stopped and its child joined. Its partial report retains `passed: false`.

The [Turbo8 integration](docs/h3-turbo.md) preserves the pinned adapter as dynamic PEFT branches, maps all 50 blocks, and checks original input preparation plus a 128-row block byte for byte on CPU. All eight schedule updates match independent upstream hashes. The corrected native gate passes 31 output comparisons across 25 supported calls in 22.25 seconds, with worst relative L2 0.0110056 against the predeclared 0.02 limit and identical hashes across tested execution routes.

The corrected [complete render](docs/evidence/h3-qkv-render-summary.json) finishes with a joined exit 0 in **686.02 seconds**, or **708.33 seconds including the gate**, inside the same 1,200-second window. The [audit](docs/evidence/h3-qkv-render-media-audit.json) passes raw hashes, all eight steps, all seven decoder windows/105 spatial tiles, exactly 124 decoded frames at 960×544/24 fps, aligned stereo 32 kHz audio and frozen artifact identity. [Retained clip](docs/evidence/h3-qkv-render/clip.mp4). Sampled frames show a recognizable crafted bird raising its wings on a wooden desk: a usable visual pilot. Audio signal checks pass; subjective listening, exhaustive real-time motion review and native cache-hit timing remain unassessed. No full-trajectory CPU/Swift numerical parity is claimed. GPU work is stopped. The earlier 701.01-second gray-texture render remains rejected historical evidence.

A CPU dummy-process check verifies the wall controller's successful-exit path and timeout cleanup, including a spawned descendant; this executes no model or GPU inference. Eight additional CPU checks exercise the shared-window audit, including overbudget combined timing, failed gates, changed identities and unjoined children.
