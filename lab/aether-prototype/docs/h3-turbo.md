# H3 Turbo8 — 2026-10-15

## Goal and current state

The corrected export produces a usable visual pilot at **960×544, 124 frames, 24 fps with stereo audio**. The numerical gate plus finished MP4 completed in **708.33 seconds (11m 48s)** under one shared 1,200-second deadline. Generation itself took **686.02 seconds (11m 26s)**, including native prompt encoding, a preparation-cache miss, all eight denoising steps, both decoders and mux. GPU work is stopped and the native child is joined. [Clip](evidence/h3-qkv-render/clip.mp4), [first frame](evidence/h3-qkv-render/frame-000.png), [last frame](evidence/h3-qkv-render/frame-123.png), [measured result](evidence/h3-qkv-render-summary.json).

Sampled source frames 0, 30, 60, 90 and 123 show a recognizable crafted bird raising its wings on a wooden desk, with consistent subject and background. This resolves the previous gray-texture failure. It is an initial visual pilot, not perfect paper appearance/takeoff adherence or an exhaustive real-time motion review. The [audio signal audit](evidence/h3-qkv-render-audio-signal.json) verifies finite, non-silent stereo samples with no clipping; subjective listening remains unassessed. [Review scope](evidence/h3-qkv-render-quality.json). No complete-trajectory CPU/Swift numerical parity is claimed.

## Corrected native result

The [canonical gate](evidence/h3-qkv-phase-summary.json) passed in **22.25 seconds**: 25 supported calls and 31 output comparisons, worst relative L2 **0.0110056** against the predeclared 0.02 limit. All tested execution routes produced identical hashes. Two caller-bound dynamic time outputs are skipped under the public API's fixed-descriptor requirement. [Gate report](evidence/h3-qkv-phase-report.json), [joined controller](evidence/h3-qkv-phase-controller.json).

The [shared-window coordinator](evidence/h3-qkv-qualified-window.json) records the gate followed by the complete render, both with joined exit 0. The [completion audit](evidence/h3-qkv-render-media-audit.json) matches both frozen binary/input identities, all eight complete evaluations, seven decoder windows/105 tiles, retained raw hashes, exactly 124 decoded MP4 frames at 24 fps and aligned stereo 32 kHz audio. The video duration is 5.166667 seconds after the model's frame-grid snapping. [Render report](evidence/h3-qkv-render-report.json), [controller](evidence/h3-qkv-render-controller.json), [journal](evidence/h3-qkv-render-events.jsonl).

| Phase | Measured seconds |
| --- | ---: |
| Numerical gate, separate joined child | 22.25 |
| Generation integrity checks | 26.79 |
| Native prompt encoding | 141.34 |
| Time embeddings and modulation preparation | 70.90 |
| Cache publication | 0.08 |
| Body setup | 25.08 |
| Eight complete steps | 308.98 |
| Audio decode | 3.43 |
| Video decode, stitching and frame output | 108.48 |
| MP4 mux | 0.35 |
| Generation start to joined process | 686.02 |
| Shared window, gate through joined render | 708.33 |

The first step took 33.53 seconds; steps 2–8 took 38.57–40.20 seconds. This is one run with backend/OS caches retained, not a controlled speedup comparison. The [corrected native preparation cache](evidence/h3-qkv-preparation-cache-integrity.json) contains 408 checked tensors, 145,313,280 bytes. Their hashes equal the prior run's unchanged time/modulation functions. A native cache-hit generation and its timing remain unmeasured.

The [capture RPC](evidence/h3-qkv-qualified-capture.json) timed out after 300 seconds while the same native child continued under its controller. The coordinator/controller records establish successful completion and cleanup; there was no second launch. The runtime remains published Corsairs 0.3.0, with no runtime defect isolated.

## QKV export correction

`convert_transformer_key` consumes a loaded state dictionary: its fused QKV rows must already be contiguous Q, K and V. Raw checkpoint shards store QKV interleaved per head. The upstream shard streamer calls `reorder_interleaved_qkv` first; Aether's direct calls omitted that step. Its original CPU comparisons and weight audit repeated the same omission, so their numerical agreement did not validate the checkpoint's intended attention weights. [Upstream conversion contract and streamer](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py).

`forge/h3_weights.py` now calls both original operations in the required order. It replaces 150 body bindings and fixes both text-refiner attention projections. Other weights retain their source hashes. [Correction record](evidence/h3-qkv-correction.json), [canonical 1,100-binding audit](evidence/h3-qkv-weight-audit.json). The head/projection sentinel regression independently distinguishes the correct per-head ordering from the omitted-normalization result. Rust requires the `raw_per_head_qkv_to_contiguous_qkv_v1` layout marker and rejects old manifests before inference.

The corrected [Turbo CPU export checks](evidence/h3-qkv-turbo-cpu.json), [phase references](evidence/h3-qkv-phase-cpu.json), [binding identity check](evidence/h3-qkv-binding-cpu.json), 27 Rust tests, Clippy, formatting and rustdoc pass. Old numerical gates remain evidence for their specific bytes/API routes; they are superseded as semantic validation of the trained transformer. The video converter already normalizes its own fused QKV internally, so the independently checked spatial VAE is reused.

[Executed frozen commands](evidence/h3-qkv-ready.json) use a numerical gate followed by the corrected full render under one 1,200-second coordinator deadline. The render does not start after a failed gate. Its [CPU controller tests](evidence/h3-qualified-render-controller-cpu.json) cover success, gate failure and interruption forwarding; they perform no model/GPU inference. Corsairs 0.3.0 remains pinned; [0.3.1 coordination](corsairs-v0.3.1-h3.md) records that this finding belongs to Aether's export layer.

The prepared completion audit passes `--coordinator` to `verify-h3-clip.py`. It requires the gate and render to finish within the same 1,200-second window, matches both stages to their frozen binaries/inputs and controller records, and rejects pending cleanup. [Eight CPU acceptance checks](evidence/h3-qualified-window-audit-cpu.json) include a green render inside an overbudget combined window, a failed gate, changed identities and an unjoined child. These tests perform no inference or quality assessment. [Frozen artifact identity audit](evidence/h3-qkv-frozen-identity-audit.json).

## Earlier incorrect-QKV render: timing pass, quality failure

This earlier run used the incorrect QKV conversion. The [controller](evidence/h3-turbo-render-controller.json) records exit 0 and a joined child after 701.0053 seconds. The [application report](evidence/h3-turbo-render-report.json) and [journal](evidence/h3-turbo-render-events.jsonl) contain all eight complete joint evaluations, both decoders, all seven temporal windows/105 spatial tiles and MP4 mux. The [media audit](evidence/h3-turbo-render-media-audit.json) verifies retained raw hashes, exactly 124 decoded frames at 960×544/24 fps, stereo 32 kHz audio and aligned durations. The snapped video duration is 5.166667 seconds. Its time does not qualify the corrected export's speed or quality.

| Phase | Measured seconds |
| --- | ---: |
| Integrity checks | 26.81 |
| Native prompt encoding | 144.01 |
| Time embeddings and modulation preparation | 69.02 |
| Cache publication | 0.08 |
| Body setup | 25.35 |
| Eight complete steps | 318.46 |
| Audio decode | 3.46 |
| Video decode, stitching and frame output | 112.70 |
| MP4 mux | 0.33 |
| Controller start to joined process | 701.01 |

The first step took 34.21 seconds; steps 2–8 took 39.90–41.38 seconds. This run built its preparation tables on a cache miss; [all 408 saved tensors pass a CPU integrity audit](evidence/h3-turbo-preparation-cache-integrity.json). A native cache-hit generation has not been timed. No matched speedup against the unfinished base50 trajectory is claimed.

[Quality review](evidence/h3-turbo-render-quality.json) rejects source frames 0, 16 and 123: none shows the requested subject. [Rejected MP4](evidence/h3-turbo-render-rejected/clip.mp4), [first frame](evidence/h3-turbo-render-rejected/frame-000.png), [last frame](evidence/h3-turbo-render-rejected/frame-123.png). Audio listening acceptance is also absent. A successful finite trajectory and valid media do not establish usable output. No Corsairs defect has yet been isolated. The [capture RPC](evidence/h3-turbo-render-capture.json) timed out at 300 seconds; the same native child continued under its wall controller, whose report is authoritative for completion and cleanup. There was no second launch.

## Native progress and decoder startup fix

The subsequent CPU investigation compares all eight time embeddings and 400 modulation tables with the first render's saved native values; all pass the predefined 0.02 relative-L2 limit, with worst error 0.000239603. The corrected three-route execution probe passed in 20.80 seconds with a joined child: 25 supported calls and 31 output comparisons. All tested routes have identical output hashes; worst CPU relative L2 is 0.00384147. Two dynamic time outputs were skipped for explicit binding, as required by the public API. These attention comparisons still used the old QKV conversion and do not qualify the corrected model. [Report](evidence/h3-turbo-phase-corrected-report.json), [controller](evidence/h3-turbo-phase-corrected-controller.json), [capture](evidence/h3-turbo-phase-corrected-capture.json). No Corsairs defect has been isolated.

The [adapter gate](evidence/h3-turbo-adapter-report.json) passes with published Corsairs 0.3.0. Its 128-row result has relative L2 **0.00463157**, within the predefined 0.02 bound. Both full-row timestep cardinalities return finite results with identical repeat hashes. The [controller](evidence/h3-turbo-adapter-controller.json) records a successful joined child after **6.17 seconds**. [Capture](evidence/h3-turbo-adapter-capture.json), Evidence `ev_05849a194ac4efd969b4e2df90845d2df6095995fc334b48a6d6d91a2e1c0b75`. This is a block gate, not a complete Turbo generation.

Two decoder attempts reached their 290-second cleanup boundary during video model loading, before a production window completed. Both children were joined with status -15. The original embedded-weight export's audio decode took 4.16 seconds; the later explicit-weight attempt took 2.01 seconds and first passed its small original CPU VAE comparison in 9.67 seconds. [Original decoder controller](evidence/h3-turbo-decoder-original-controller.json), [explicit full-graph controller](evidence/h3-video-functional-fullgraph-controller.json), [journals](evidence/h3-video-functional-fullgraph-report.events.jsonl). An [original-load sample](evidence/h3-turbo-decoder-original-load-sample.txt) observed MPSGraph compilation/constant optimization; it does not establish a complete causal performance profile.

`forge/h3_video_functional.py --spatial-tiles` now exports a single original 256px F32 decoder tile with 703 explicit weight bindings. Rust runs the original 3×5 spatial grid for each temporal window, preserving 256px tiles, source overlaps, vertical-then-horizontal blending against original neighbors, and cropping. This removes the fifteen repeated tile decodes from the native graph. It changes orchestration, not neural weights, precision or decoder context. The [CPU VAE check](evidence/h3-video-spatial-cpu.json) matches the unchanged 256px module byte for byte. The [independent source spatial oracle](evidence/h3-video-spatial-oracle.json) verifies all packed-patch extraction and complete stitching bytes, including starts on odd latent rows. The release suite now passes **27 tests**, including nine optional input/media checks.

The [single-tile native decoder check](evidence/h3-video-spatial-decoder-report.json) passed in **135.43 seconds** end to end, with a [successful joined child](evidence/h3-video-spatial-decoder-controller.json). The 256px original CPU comparison has relative L2 **1.56954e-6** against the predefined 0.02 limit. All seven production temporal windows and 105 spatial tiles completed; audio decoding took 3.79 seconds and video decoding 97.95 seconds including parameter upload and stitching. Individual windows took 12.19–14.21 seconds. These are seeded synthetic decoder latents, not a denoised or visually accepted clip. [Capture](evidence/h3-video-spatial-decoder-capture.json), Evidence `ev_58fd102ed113fcd2cb03cd9dc801f78c378a8c4f119a86f8d5b8f53d4de8fcee`.

The [published recipe](https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs) supports FL2VA/T2VA at 544p with eight evaluations and video/audio shifts 12/3. Aether uses the lab's existing adapter from `lightx2v/Minimax-h3-Turbo`, revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`, file `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors`, SHA-256 `e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f`. Rank is 128, alpha 8, effective scale 8/128.

## Implementation

`forge/h3_turbo.py` calls the pinned original Diffusers modules and upstream PEFT loader on CPU. It preserves low-rank matrix products, rank scaling, casts and residual additions as dynamic branches. It never fuses the updates into base BF16 tensors. Original base weights remain referenced by their existing hashes. Each transformer block has 22 explicit bindings: ten base tensors and twelve adapter tensors. The text-refiner adapter is included in the new input-preparation export. The conditioner, time/AdaLN functions, final heads and both VAEs are unchanged and reused from the base bundle.

`H3Recipe::Base50` remains the default for older manifests. `Turbo8` requires the exact pinned adapter identity, dynamic execution contract and captured eight-step 12/3 schedule. Generation evaluates the complete bundle recipe. A benchmark prefix cannot shorten ordinary generation. Manifest validation rejects a Turbo adapter assigned to the base recipe, merged execution, wrong adapter hash/alpha or incorrect binding names.

`H3Options::preparation_cache` stores completed native time embeddings and modulation tables. Identity covers both model hashes and entry points, adapter, recipe, full schedule and exact timestep bits. Entries publish through a temporary directory and rename; reads validate hashes, shapes and dtypes. The directory is application-owned trusted data. It contains no text conditioning or initial noise. Fresh-start and reused preparation must be timed separately; the native equality check is still pending.

## CPU evidence

- [Export checks](evidence/h3-turbo-cpu-checks.json): all 312 adapter target shapes checked; all 50 binding maps prepared; original adapter-enabled outer preparation and a 128-row original adapter block match the export wrapper/parameter-lifted graph byte for byte. Block zero's exported base/adapter bindings also match the actual CPU graph parameters. This is a boundary check, not complete eight-step generation.
- [Captured schedule](evidence/h3-turbo-schedule.json), [scheduler oracle](evidence/h3-turbo-schedule-oracle.json): all eight video/audio update hashes match Rust using the unchanged source scheduler. Canonical schedule bits are shipped with the Rust contract; unsupported grids fail before inference.
- Release suite: 27 tests passed, including nine optional input/media tests. Cache identity/corruption checks and adapter/recipe rejection are CPU-only. Clippy, package formatting and rustdoc checks pass.
- [Controller CPU check](evidence/h3-turbo-controller-cpu.json): a dummy command's successful application/exit path passed. A deliberately slow dummy and its descendant were terminated, the child joined, and a process query confirmed the descendant absent. No model/GPU work occurred. [Native capture](evidence/h3-turbo-controller-capture.json), Evidence `ev_50f20aab8471f566150713df856c386d23c14ecb29c1bf3e57fb90c77513b616`. Native process sampling was partial; the application assertion and independent descendant query are retained separately.

## Earlier GPU qualification sequence

These historical commands used the earlier transformer conversion. The canonical gate and render command at the top of this page supersedes them for trained-model validation; the unchanged VAE's independent qualification remains applicable.

1. `h3_turbo_probe`: compare the 128-row Metal result with the original CPU module using a **2% relative-L2 limit declared before execution**, then run a single full-geometry block for each actual timestep cardinality. Every case repeats twice, checks finite values and requires matching repeat hashes. These full-row cases establish binding/feasibility only; they have no full-row CPU numerical oracle. Maximum wall budget: five minutes.
2. `h3_decode_probe`: decode seeded synthetic latents using both original VAEs at the complete production profile, including every temporal window and stitch. It skips conditioning, denoising and media output. It establishes decoder feasibility/time only. Maximum wall budget: five minutes.
3. After those pass, `h3_generate` with the Turbo8 bundle, optional preparation-cache directory and `--ffmpeg=/opt/homebrew/bin/ffmpeg`. It runs all eight steps, both VAEs, PNG/WAV output and MP4 mux under a twenty-minute wall budget. Check streams/frame count and inspect/listen to the clip before accepting quality. A first native cache miss builds its tables inside this measured window; a later hit needs separate evidence.

`scripts/verify-h3-clip.py` performs the CPU completion audit after rendering: controller/build identity and joined exit, the 20-minute wall limit, all eight joint steps and seven decoder windows, retained raw hashes, exactly 124 decoded MP4 frames at 24 fps, 960×544 canvas, stereo 32 kHz audio and aligned stream durations. It keeps `quality_accepted: false` until visual/audio review. Its [CPU checks](evidence/h3-turbo-media-audit-cpu.json) use a synthetic three-frame MP4 and synthetic journal/controller inputs, and establish no H3 generation result.

`scripts/run-h3-check.py` owns one process group, including any FFmpeg descendant. Probe budgets are capped at 300 seconds and renders at 1,200, with ten seconds reserved for stop/join. Known competing render drivers abort only Aether's owned job. An observation RPC timeout is separate from the controller's application/child-exit report. GPU availability must be coordinated before invoking these prepared commands.

[Earlier commands and readiness](evidence/h3-turbo-ready.json) reference frozen [adapter](evidence/h3-turbo-adapter-build.json), [single-tile decoder](evidence/h3-video-spatial-decoder-build.json) and [render](evidence/h3-video-spatial-render-build.json) builds. Each preserves the compiled canonical schedule, application/runtime source snapshots and published Corsairs 0.3.0 identity. These commands completed; that earlier complete render failed quality review. The corrected result and exact command are linked at the top of this page.

CPU export command (no GPU inference):

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/zakkeown/Code/aether-engine/forge/.venv/bin/python -B \
  forge/h3_turbo.py --lab /Users/zakkeown/Code/atelier/lab/minimax-h3-lab \
  --base .local/h3-production/bundle.json --output /private/tmp/aether-h3-turbo8
```

Generated weights/assets stay outside the package. No checkpoint, dependency or lab source was modified. The completed corrected render exposes no missing Corsairs capability or confirmed wrapper defect.
