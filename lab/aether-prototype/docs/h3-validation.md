# H3 validation — 2026-10-15

**Semantic qualification correction:** the earlier transformer exports below omitted raw-shard per-head QKV normalization. Their numerical agreement remains valid for their recorded tensors, but does not qualify the intended trained checkpoint. Aether has corrected the export contract and rejects old manifests. [Current correction and native result](h3-turbo.md).

The corrected Turbo8 path now produces a [usable visual pilot with audio](evidence/h3-qkv-render/clip.mp4) at 960×544/124 frames, finishing in **708.33 seconds including its numerical gate** under one twenty-minute cap. The gate passes 31 comparisons; the [complete-window/media audit](evidence/h3-qkv-render-media-audit.json) verifies both joined children, source identity, all steps/windows, raw hashes and media streams. Sampled frames show coherent subject/wing progression; subjective audio listening remains unassessed. GPU work is stopped. The diagnostic and base50 records below remain historical observations with their original scope.

Aether owns the offline exports and Rust T2VA generator. The current dependency is published Corsairs `0.3.0`; Corsairs owns its checked tensors and native runtime qualification. The earlier diagnostic captures below used the development checkout and do not qualify the later published artifact. No Corsairs or Atelier files were modified. Checkpoint: FL2VA `42ed227ee7df40d41602854ae760620d6eb651fe`; original Diffusers commit: `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941`.

## Complete diagnostic generation

The original-weight **128×128, 22-frame diagnostic** completed 50 joint denoising steps with all 50 blocks at each step, both decoders, 22 PNG frames and stereo float PCM at 32 kHz. It supplied the original CPU conditioning and initial noise, exercising Rust packing, actual timestep cardinalities, cached modulation, body execution, output heads, source scheduler, decoding and media output. The native process exited zero.

- [Generation report](evidence/h3-provided-generation.json), [native observation](../.local/h3-provided-observation.json), Evidence `ev_c92866fa8a7419910ee22c39d4bfa4f5ece4d6bd866b782463523bfeef0ac478`.
- [Raw output and frames](../.local/h3-provided-clip/) and [CPU-muxed preview](../.local/h3-provided-clip/clip.mp4).
- Final video latent SHA-256: `b5590a2fc8fb7c2ccfce4f827e4fade83fb6f3f6ee3cf6be2cd1d814dbf27f57`.
- Final audio latent SHA-256: `e4e814f900ae4d9f0088c85856e65e9ae5e5b57fbfe102b6fd0ac2c365eb14e6`.

This deliberately small diagnostic is outside the production duration/geometry matrix. A sampled frame is visually blocky; completion and finite output do not establish production visual quality. There is no independent complete-trajectory byte oracle for this generation. MP4 is lossy and is not used for numerical acceptance.

## Independent original-source boundaries

Each comparison below used an original CPU source-module result and a **2% relative-L2 limit declared before the observation**. These cross-backend passes are not byte-exact claims.

| Boundary | Relative L2 | Scope |
| --- | ---: | --- |
| [Input preparation](evidence/h3-prepare.json) | 0.0028912163 | Original projections and text refinement before block 0 |
| [Native conditioner](evidence/h3-conditioner.json) | 0.0013552201 | Original 50-layer Qwen prefix, hidden_states[50], no final normalization |
| [Video decoder](evidence/h3-video-decoder.json) | 0.0000009733 | All raw original-weight decoded samples, 22×128×128 RGB |
| [Audio decoder](evidence/h3-audio-decoder.json) | 0.0000172216 | All raw original-weight decoded samples, stereo, 29,600 samples/channel |

Preparation: Evidence `ev_6826b234a830e5ff3cc944868dc96b8f095f7f4db1cea9254cef29e8f8cc5e02`, [observation](../.local/h3-prepare-observation.json). Conditioner: Evidence `ev_f91d923a6dc0750f0ce48ec3e64c0dd38f9221efd7783f63c1a8b948997197df`, [observation](../.local/h3-encode-observation.json). Both native processes exited zero. The decoder comparisons ran on the CPU against final latent rows from the completed native diagnostic; neither uses a decoder stub. Video maximum absolute error was `6.1094761e-6`; audio was `1.1321343e-4`. All compared samples were finite.

The native diagnostic's retained video is the earlier NCTHW format `[1,3,22,128,128]`. The current streaming implementation retains FCHW and records that layout in `decoded.json`.

## Production-profile check

The attempted base50 production render used **960×544, 124 frames, 24 fps**, native prompt encoding and Aether-seeded initial latents. It was stopped at the user’s request because generation was too slow, and GPU work was paused for Corsairs. The run reached 22/50 denoising steps and never reached either decoder. This attempt produced no clip or completed production pass. The user subsequently authorized bounded performance measurements after v0.3 publication; those and the corrected Turbo8 completion above are separate from this stopped generation.

The executable, bundle and source snapshots were frozen with [the snapshot helper](../scripts/freeze-native.py). [Retained manifest](evidence/h3-production-build.json): executable SHA-256 `ec135d7a62662f0b7cbdb10e0bd8ab9e855b2a449551b14cc9cb36f869122e50`; bundle SHA-256 `31b7011343599a4f5c4fded70fd8fe6d65faf7a961c27364d7399aa4c87871fc`. Model streams and uploaded parameter bytes are independently hash-checked by the application.

The native capture RPC timed out after 300 seconds while its child continued running. The [retained progress journal](evidence/h3-production-events.jsonl), [stop record](evidence/h3-production-stop.json) and [initial application report](../.local/h3-production-report.json) remain available. The application report stays `passed: false` and was never finalized by the application. SIGTERM was delivered through the native process controller (Evidence `ev_23fac2f37c4aa754a14bb148d3a6df624a1e6235d2793282eea8c24da2f4d636`), then an independent process query confirmed PID 15586 absent. The CPU monitor was also stopped. The [capture transport error](../.local/h3-production-observation.json) and [termination observation](../.local/h3-production-stop-observation.json) are retained. Neither is a completed controlled observation of the H3 render’s exit status.

A two-second process sample during denoising showed the Corsairs worker waiting in a Metal fence and MPSGraph execution, with the Rust executor parked; it did not show blocked stdout during that sample. This bounds one suspected cause but does not establish the cause of the overall speed. [Sample](../.local/h3-production-sample.txt). No new inference is needed to preserve or inspect these records.

## Published-v0.3 performance observation

The separate [base50 benchmark](h3-benchmarks.md) records an interrupted production-profile run with published Corsairs 0.3.0. Preparation took 275.6 seconds; three complete steps took 35.7, 45.7 and 54.2 seconds. Each video/audio latent hash matches the earlier development run at the same step. Neither decoder ran in that attempt. Its partial result exceeded the user's twenty-minute target by projection and retains `passed: false`; its child was joined. It establishes no completed base50 render or stable throughput. The corrected Turbo8 completion above is a separate recipe and run.

## CPU application checks

The complete release suite passes **21 tests**, including six optional external-input/media tests. It compares all four saved row layouts, exact index narrowing, actual T2VA timestep values/indices at all 50 calls, source float32 scheduler hashes at all 50 updates, patch order, BF16/fp16 separation, geometry, provenance and raw PCM handling. The 124-frame temporal-stitching oracle calls the original VAE control flow with a deterministic decoder stub to isolate window/crossfade behavior; original-weight decoder inference is checked separately above.

FFprobe confirms the synthetic mux has three video frames and an audio stream. This media test performs no H3 inference. All six Krea fixture prompts and saved rotary bytes remain exact. Clippy, formatting and rustdoc are checked separately from native execution. See [root validation](../VALIDATION.md) for earlier exact Krea native gates and their source bounds.

```sh
AETHER_H3_LAB=/Users/zakkeown/Code/atelier/lab/minimax-h3-lab \
AETHER_FFMPEG=/opt/homebrew/bin/ffmpeg \
AETHER_FFPROBE=/opt/homebrew/bin/ffprobe \
cargo test --offline --release --test h3 -- --include-ignored
```

## Earlier bounded native gates

Earlier Aether runs against the v0.2 development source pass the original-weight 128-row block 0 and all 50 base-schedule pairs exactly against Swift, including native BF16 chaining. They are separate from complete clip generation and predate the drain migration.

- [Block report](evidence/h3-block.json), [observation](../.local/h3-native-observation.json), Evidence `ev_f9e9b03bbf64d87cf7f98ffea5a781204261448ce64151d03b8a39660439a4ed`.
- [Schedule report](evidence/h3-schedule.json), [observation](../.local/h3-schedule-observation.json), Evidence `ev_baa5ee7c346de93dc552378be0cc00f185d0442bbcc91bde9cb3f85338ceb4c1`.

Single-block SHA-256: `001045fc523793248c74b3894507a72fb1e17c1aac3e2499c47fdf2b3c836c70`. Modulation SHA-256: `28942f089a76cb06977401e74dfc01aaf9b03398779850b03c0d38673ed48817`.

Corsairs' [synthetic-backbone acceptance](../../corsairs/docs/v0.2-h3.md) covers all 50 blocks, including staged loading in one runtime. Its final `[1,19350,5376]` BF16 hash is `e4aabaed96045e8b86cd292f7c6234f4ad5d86dda2e33edc41330d3e9772e75b`. This remains upstream evidence. Aether's [staged input gate](evidence/h3-backbone-inputs.json) verifies all input/table/reference bytes and 50 segment identities; its separate native staged-backbone command has not been run here.

## Bounds

First/last-frame conditioning and Ref2VA are not implemented in the generator. Profiles fix canvas, aligned frame count and prompt token count; supporting another profile currently requires export. Experimental quantization, approximation, caching optimizations and kernels have not been promoted. Cancellation/recovery and repeated production residency need separate native coverage. These application tasks remain in Aether.

Native process captures sample an executable digest before launch rather than atomically binding the OS image. Process sampling may miss short-lived descendants; filesystem observations are snapshots rather than syscall traces, and the inherited environment is not fully recorded. These observations establish no sustained throughput, minimum-memory profile or performance improvement claim. Corsairs' frozen-source package acceptance is separate from Aether's application evidence.
