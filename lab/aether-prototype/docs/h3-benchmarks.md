# H3 rendering pace — 2026-10-15

## Result and budget

The user requires **20 minutes maximum from start to finished clip**, including preparation, denoising, decoding and media delivery. The corrected [Turbo8 render](h3-turbo.md) now produces a usable visual pilot in **708.33 seconds (11m 48s), including its numerical gate**. Generation itself takes 686.02 seconds, with native prompt encoding, a preparation-cache miss, all eight complete steps, both decoders and MP4 mux. [Clip](evidence/h3-qkv-render/clip.mp4), [measured phases and review scope](evidence/h3-qkv-render-summary.json), [shared-window/media audit](evidence/h3-qkv-render-media-audit.json). Subjective audio listening and native cache-hit throughput remain unassessed. GPU work is stopped.

The first Turbo8 render took 701.01 seconds but failed visual quality because Aether omitted raw-checkpoint QKV normalization. It is retained as rejected evidence. The original-weight 50-step path did not meet the limit; its final benchmark child was joined after 453.9 seconds without a production clip. The measurements below describe those earlier attempts with the incorrect QKV conversion; they are not corrected-model quality or speed qualification.

The quiet attempt measured **275.6 seconds of preparation**, followed by three complete denoising steps taking **35.7, 45.7 and 54.2 seconds**. Applying those observed step costs to 50 steps gives **34.3–49.8 minutes before decoding**. This is arithmetic on a partial run: step times were rising, there is only one complete step after both initial graph cardinalities, and neither decoder ran. No stable throughput or full-render timing is claimed. [Machine-readable summary](evidence/h3-v03-benchmark-summary.json).

## Quiet attempt: partial production-profile measurement

| Observed phase | Seconds |
| --- | ---: |
| Full source integrity checks | 30.500 |
| 50-layer prompt encoding | 145.734 |
| Time embeddings | 0.311 |
| All 2,500 modulation calls and stage loads | 73.679 |
| Body parameter verification/upload and setup | 25.323 |
| Preparation through first-step start | 275.587 |
| Step 1, including 50 body blocks | 35.669 |
| Step 2, including 50 body blocks | 45.733 |
| Step 3, including 50 body blocks | 54.197 |

The body accounted for 34.8, 45.3 and 54.1 seconds respectively. All three video/audio latent hashes match the earlier development run exactly. This establishes consistency at these checkpoints. The run stopped during step 4, before either decoder. Both the controller and unfinalized application report retain `passed: false`; the controller records its child joined with status `-15`.

No known competing renderer was detected before or during this attempt. Host samples recorded no thermal warning. Cumulative swap traffic over the run was about 51 MiB in and 35 MiB out, unlike the heavily contended first attempt. Compression increased substantially during phase transitions. These observations do not identify why successive steps slowed, and clocks/ambient temperature were not controlled.

[Controller and system samples](evidence/h3-v03-quiet-controller.json), [application journal](evidence/h3-v03-quiet-report.events.jsonl), [unfinalized report](evidence/h3-v03-quiet-report.json), [stop observation](evidence/h3-v03-quiet-stop.json). The [capture transport error](evidence/h3-v03-benchmark-quiet-capture.json) is separate from the controller's retained child-exit record.

The later Turbo8 implementation adds reusable preparation and the pinned step-distilled adapter with its intended sampling schedule. After correcting the QKV export, its complete trajectory produces a usable visual pilot within budget. Merely truncating the existing 50-step schedule does not establish a finished generation or quality-equivalent recipe.

## Protocol

The benchmark uses the published crates.io **Corsairs 0.3.0** archive, SHA-256 `e7f23287a680009a43aa5a46e8646a66918f5c4c3e2bdb627a18091b21fe3e3a`, verified against the registry index. [Release identity](evidence/corsairs-0.3.0-release.json). Host: Apple M5 Max, 64 GiB, macOS 27.0.1. The [frozen build](evidence/h3-v03-benchmark-build.json) pins application/runtime source snapshots, executable and profile bundle.

Workload: T2VA, original BF16/F32 weights, 960×544, 124 frames at 24 fps, 49 text rows, 19,333 joint rows, seed 1043. Each denoising step evaluates all 50 transformer blocks. The benchmark driver plans all 50 source-schedule modulation pairs, a five-step prefix, then both decoders at the complete profile geometry. The observed runs below stopped before this plan completed. A prefix is neither a five-step sampling recipe nor a completed render. Decoder inputs would come from the partial trajectory; no quality claim is made.

Timers use monotonic wall time around completed public calls. They include fences/readback, finite checks, parameter upload and host schedule updates as named. Each body-block timing includes its modulation upload and completed encode. Decoder measurements include full raw readback, finite checks, original temporal crossfades and hashing. PNG/WAV encoding and MP4 mux are omitted. A fifty-step estimate adds 45 times the median of prefix steps 2–4 (zero-based) to observed benchmark time; its sample range is not a confidence interval. This assumes later steps cost the sampled warm graph calls. OS/backend caches are retained, and clocks/ambient temperature are not controlled.

`scripts/run-h3-benchmark.py` owns a native child with a 15-minute deadline, records system-memory/thermal samples and joins the child on completion or termination. This preserves its exit independently of the capture RPC's 300-second transport timeout. No inference is launched concurrently by Aether.

A second attempt was interrupted by a native-server restart during conditioning. A third attempt was stopped by the controller when another prompt-matrix renderer started; that renderer was left alone. Neither reached a complete denoising step. The quiet fourth attempt above followed the user's confirmation that the other queue was paused.

## First attempt: concurrent render, excluded

The first attempt began with about 43 GiB of free system pages. A separate prompt-matrix `labrender` started during modulation preparation. Both were sampled inside Core AI inference. H3's sampled footprint was 43.1 GiB; the other renderer's was 10.7 GiB. The host was on AC power and was swapping heavily. These are overlapping process footprint observations, not an additive memory accounting model.

| Observed phase | Seconds |
| --- | ---: |
| Full source integrity checks | 31.113 |
| 50-layer prompt encoding | 151.131 |
| Time embeddings | 0.294 |
| All 2,500 modulation calls and stage loads | 86.021 |
| Body parameter verification/upload and setup | 31.840 |
| First denoising step under contention | 179.419 |
| Of that, all 50 body blocks | 177.620 |

The native child's first video/audio latent hashes match the earlier development run exactly. This is one prefix checkpoint, not full-generation parity. H3 was terminated through its controller after about 487 seconds; the controller joined its child with status `-15`. Neither decoder ran. The other renderer was left alone. **This is not an isolated v0.3 throughput result or evidence of a v0.3 speed regression.**

[Controller](evidence/h3-v03-contended-controller.json), [journal](evidence/h3-v03-contended-report.events.jsonl), [status](evidence/h3-v03-contended-status.json), [H3 sample](evidence/h3-v03-contended-first-pass-sample.txt), [concurrent renderer sample](evidence/h3-v03-contended-other-render-sample.txt). The [capture error](evidence/h3-v03-benchmark-capture.json) is retained separately and is not a completed render observation.

## Published context

The nearest hardware report is an author's M5 Max 64 GB MLX experiment: **31.3 minutes for a 10-second 544×960 clip at 15 steps**, using a 6-bit DiT and image conditioning. Its four-step Turbo configuration takes **10.5 minutes**. These are different precision, duration, conditioning and schedule, so neither is a matched Aether baseline. [Original measurement](https://github.com/skymars13/minimax-h3-mlx-64gb/blob/b788e667c62f63c358c705b6490295b9b62597a3/README.md).

For faster discrete-GPU context, SGLang reports **112.1–112.2 seconds** for 864×480, 124 frames, 20 steps on a physical RTX 5090 with 60 GB host RAM: **5.14–5.17 seconds/step**, 4.2 seconds encoding and 6.4 seconds decoding. This uses a different GPU, smaller canvas, different step count and serving/residency stack. [SGLang's measured desktop recipe](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#rtx-5090-single-card-physical-desktop).

MiniMax describes H3 as a 33B dense transformer, with roughly 13B parameters in cacheable AdaLN branches. The initial open release uses full attention. [Model card](https://huggingface.co/MiniMaxAI/MiniMax-H3). The measurements above place Aether's original 50-step baseline in the tens-of-minutes range by projection; they are not completed render measurements.
