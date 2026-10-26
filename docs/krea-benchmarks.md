# Krea benchmark harness

`aether-probe benchmark-krea` runs an opt-in, instrumented end-to-end baseline using the existing generation API. The checked-in [plan](../Benchmarks/krea-baseline.json) pins the two independently qualified workloads: the 512×512 snow fox and 384×512 watercolor forest prompts, eight steps and seed 17017. Each case has one warm-up and three measured iterations. Every iteration, including warm-up, must match independent conditioning, all eight latent hashes and float32 pixels.

The user completed the first planned quiet-window run; see the [baseline analysis](evidence/krea-benchmark-baseline-2026-10-18.json). Preparation and CPU tests remain separate evidence ([readiness](evidence/krea-benchmark-readiness-2026-10-18.json)); prepare-only validation establishes no performance result.

## Prepare without inference

From the repository root, configure the existing cached bundle and choose a new output directory:

```sh
source .local/native-assets.env
Scripts/benchmark-krea.sh prepare "$PWD/.local/benchmark-preparation-01"
```

The wrapper builds the Release executable and records its actual binary hash plus the source tree's Swift files, scripts, package manifest and lockfile. Preparation checks the bundle fingerprint, pinned reference bytes and workload metadata, tokenizer output, deterministic noise and dynamic selection identity. It reads asset declarations and tokenizer files; it does not load native model sessions, execute inference or read all model streams. `report.json` ends with `stage: ready`, `passed: true`, `benchmarkExecuted: false` and empty statistics. Source and installed-toolchain identities describe the files/tools observed by the probe; the wrapper builds that checkout immediately before invocation.

## Run during the quiet window

After arranging the quiet window, use a different new directory:

```sh
source .local/native-assets.env
Scripts/benchmark-krea.sh run "$PWD/.local/benchmark-quiet-02"
```

`run` is the explicit opt-in. There is no background benchmark or automatic measurement after preparation. The command does not stop other workloads, change power settings or clear OS caches. It runs cases serially, checks all six model streams before warm-up, and creates one generation session per case. Each case reuses its session across warm-up and measured iterations, then unloads before the next case. Output directories are created exclusively; an existing directory is rejected.

The baseline uses `preferGPU` and balanced residency with native prompt encoding on every iteration. Balanced mode closes preceding denoisers before acquiring the encoder, then opens denoisers for that request; the decoder stays in the session. Thus warm-up does not mean every model remains resident. Verification reads all streams before iteration timing, so this protocol does not measure a cold filesystem/cache start. Models and reference files must stay immutable during execution.

To increase sample counts, copy the plan and change `warmupIterations` (1–10) and `measuredIterations` (3–100). Reference paths resolve relative to the plan file, so preserve or adjust them when moving it. The entire plan is hashed into the report. Workload changes require a matching independent reference, its pinned SHA-256 and the correct bundle fingerprint; changing only a dimension, prompt or seed fails preparation. The standalone reference records supply correctness hashes, not performance baselines.

## Compare encoded and provided conditioning

The [encoded comparison plan](../Benchmarks/krea-encoded-comparison.json) and [provided comparison plan](../Benchmarks/krea-provided-comparison.json) use identical models, prompts, seeds, shapes and independent references. Each has three warm-ups and five measured iterations per workload: 16 image generations per plan, 32 for the pair. The original one-warm-up/three-measurement baseline plan is unchanged. Additional warm-ups do not guarantee stationary timing; preserve every measured row and inspect the trend.

| Profile | What happens before iteration timing | What timed iterations do |
| --- | --- | --- |
| `encoded` (default for older plans) | Tokenization/noise preparation and model-stream verification | Encode each request with balanced cleanup, reacquire denoisers, sample and decode |
| `provided` | Encode once per workload, verify the independent conditioning hash, then unload the encoder | Reuse the provided conditioning, denoisers, functions and decoder across warm-ups and measurements |

The provided profile records one-time setup in `conditioningSetups`, including load/encode/hash/unload duration. The encoder closes before the generation session is acquired. Every timed iteration still verifies conditioning, all eight latents and pixels against the same independent reference. A setup error or hash mismatch prevents timing that case. Setup is excluded from iteration totals, so this profile measures repeated-image rendering with existing conditioning, not fresh-prompt end-to-end latency. Kernel process-lifetime memory peaks still include setup; per-iteration sampled peaks begin after setup.

Run both plans serially during an available quiet window, using new output directories:

```sh
source .local/native-assets.env
Scripts/benchmark-krea.sh run "$PWD/.local/compare-encoded-02" \
  Benchmarks/krea-encoded-comparison.json
Scripts/benchmark-krea.sh run "$PWD/.local/compare-provided-02" \
  Benchmarks/krea-provided-comparison.json
```

Replace `run` with `prepare` to validate without native model loading or inference. Both plans passed [comparison readiness checks](evidence/krea-benchmark-comparison-readiness-2026-10-18.json), followed by the successful [native comparison](evidence/krea-benchmark-comparison-2026-10-18.json) recorded below. Each report and iteration journal row names its profile, and statistics reject mixed-profile rows. Missing profile fields in older plans retain the encoded protocol; unknown profiles fail decoding.

Compare sampling and individual step durations as well as the separately labelled total. In the provided profile, `conditioning` is the provided-tensor handoff and measured `denoiserSetup` reuses the existing selection. Its first warm-up's `step0` acquires functions, while later iterations retain them. A total-time difference includes removed encoding work as well as changed residency; it cannot be attributed entirely to loading or host copies. Do not combine profile samples or treat the historical three-sample run as a matched new control. Save both reports with their source/executable/plan identities and verify matching workloads/counts before comparison. Case/process order can affect cache state; a later window can reverse profile order if a difference warrants further testing.

## Results and timing boundaries

The output contains an atomic `report.json` and an append-only `iterations.jsonl`. A complete run has `stage: complete`, `passed: true`, `benchmarkExecuted: true`, model-stream verification and stable postflight provenance. Each iteration records its role, ordinal, hashes, selection identity, timings, sample count and memory counters. Statistics contain measured iterations only, grouped by case, with count/minimum/median/mean/maximum. Warm-ups and failed rows never enter summaries. Failure retains completed/failed iterations and an error with no aggregate statistics. An interrupted process leaves an incomplete report, not a passing benchmark.

The following are the encoded profile’s application-stage wall-clock boundaries, not GPU kernel timings. The provided profile differences are described above:

| Metric | Included work |
| --- | --- |
| `conditioning` | Balanced cleanup of preceding denoisers, encoder model/function loading, native encoding and conditioning extraction |
| `denoiserSetup` | Dynamic selection, model acquisition and signature validation |
| `step0` | Rotary/schedule preparation, lazy denoiser function loading, four native segments, host tensor copies and Euler update |
| `step1` … `step7` | Four native segments, host copies and Euler update |
| `decode` | Decoder model/function loading when needed, packing, native decode, output copies and image validation |
| `completion` | Return from the decoded callback through generation completion |
| `sampling` | Sum of all eight step intervals |
| `pipelineExcludingObservers` | Sum of the stage intervals |
| `totalIncludingObservers` | Full generation duration, including validation observers and instrumentation |
| `observer` | Progress hashing, event bookkeeping and boundary memory samples |

Stage clocks stop on progress-callback entry and resume after reference validation, preventing hashing from being attributed to the next stage. Final pixel hashing, tokenization, noise preparation, report IO and unload are outside generation timing. The observer-inclusive total remains available so instrumentation overhead is visible. Source, executable, plan, references and asset declarations are rechecked before statistics are published; model stream bytes are verified before warm-up.

Memory is sampled every 50 ms and at stage boundaries. Each iteration records separate maximum-observed physical-footprint, resident and default Metal device allocation counters. These counters overlap and must not be summed. Sampling can miss brief peaks and includes validation callbacks. Kernel process-lifetime footprint/resident peaks are also saved and explicitly cannot be reset per iteration. Before-iteration and after-unload snapshots, device/OS/RAM, installed Swift toolchain, thermal state and low-power mode provide context; they do not prove the machine was idle. Thermal state values are the operating system's raw enum values.

For direct invocation without the build wrapper:

```sh
.build/arm64-apple-macosx/release/aether-probe benchmark-krea \
  "$AETHER_KREA_BUNDLE" Benchmarks/krea-baseline.json "$PWD" \
  "$PWD/.local/benchmark-direct-01" --prepare-only
```

Use `--run` only for the intended measurement window. Native boundary diagnostics and cancellation qualification remain separate from this harness; they are described in [generation qualification](krea-generation.md).


## First baseline: 2026-10-18

The user ran `benchmark-quiet-01` successfully on the 64 GiB Apple M5 Max. All eight generations (two warm-ups and six measurements) matched independent conditioning, every latent step and pixels. Model streams were verified, provenance remained stable and the iteration journal matches the report. The [raw report](evidence/krea-benchmark-quiet-01-2026-10-18.json), [journal](evidence/krea-benchmark-quiet-01-2026-10-18.jsonl) and [analysis with artifact hashes](evidence/krea-benchmark-baseline-2026-10-18.json) are preserved.

| Workload | Measured totals (s) | Median (s) | Mean (s) |
| --- | --- | ---: | ---: |
| 512×512, eight steps | 43.36, 30.86, 28.72 | 30.86 | 34.31 |
| 384×512, eight steps | 23.71, 23.86, 24.26 | 23.86 | 23.94 |

The 512×512 case remained visibly nonstationary: conditioning declined from 19.55 to 6.77 to 5.31 seconds across measured iterations. Retain the slow sample; one warm-up did not establish stable timing. The second case was more consistent but ran later in the same process, with only three measured samples. Thermal state was nominal at the recorded endpoints and low-power mode was off; neither establishes system idleness throughout the run.

First sampling steps cost about 8–9 seconds, versus 1.6–2.2 seconds for later steps. Decode took roughly 0.09–0.13 seconds. Balanced encoded generation reloads denoisers/functions after prompt encoding, so the next comparison should separate provided conditioning with retained models from repeated encoded generation. That follow-up comparison is now recorded below, using increased warm-up and sample counts. The historical baseline alone does not establish a speedup.

Measured maximum observed Metal allocations were 15.34 GiB (512×512) and 14.58 GiB (384×512); physical-footprint counters were 4.03 and 3.30 GiB, while resident counters were 34.39 and 34.09 GiB. These are overlapping views, not additive memory requirements. Full kernel-lifetime peaks also include preflight and warm-up and are labelled separately in the raw report.

## Matched comparison: 2026-10-18

The user completed both plans on the same 64 GiB Apple M5 Max, encoded first and provided second in separate processes. All 32 images (12 warm-ups and 20 measurements) exactly matched independent conditioning, all eight latent hashes and float32 pixels. The source, executable, models, workloads, references and iteration counts matched between profiles. Both reports completed with verified model streams and stable postflight provenance. Journal contents and every saved statistic were independently rechecked during archival.

| Workload | Encoded median (s) | Provided median (s) | Provided one-time conditioning setup (s) |
| --- | ---: | ---: | ---: |
| 512×512, eight steps | 38.09 | 16.72 | 28.49 |
| 384×512, eight steps | 25.70 | 13.26 | 14.93 |

Each median uses five measurements after three warm-ups. Provided totals exclude the separately recorded setup and warm-ups: they describe repeated rendering with existing conditioning and retained models. The observed median ratios are 2.28 and 1.94, respectively; these are profile comparisons, not fresh-prompt end-to-end or isolated GPU-kernel speedups.

The first sampling step's median fell from 13.91 to 2.08 seconds at 512×512, and from 9.16 to 1.66 seconds at 384×512. Later steps averaged 2.10 versus 2.12 seconds and 1.67 versus 1.65 seconds, respectively. This supports prioritizing conditioning and generation-session reuse for repeated prompts. Stage timings include model/function acquisition, materialization, host operations and native execution; they do not identify an individual framework mechanism. Production residency policy has not changed as part of this comparison.

All measured samples remain included. Encoded totals ranged from 35.47–40.45 seconds and 23.41–31.85 seconds; provided totals ranged from 16.62–18.46 seconds and 13.22–13.33 seconds. Cache state and host activity were uncontrolled, and profile order was not counterbalanced. Encoded thermal endpoints were nominal (0); provided started nominal and ended fair (1). Low-power mode was off at both endpoints. These observations do not establish host idleness or thermal state throughout either run.

The [analysis and artifact hashes](evidence/krea-benchmark-comparison-2026-10-18.json), [encoded report](evidence/krea-benchmark-comparison-encoded-2026-10-18.json), [encoded journal](evidence/krea-benchmark-comparison-encoded-2026-10-18.jsonl), [provided report](evidence/krea-benchmark-comparison-provided-2026-10-18.json) and [provided journal](evidence/krea-benchmark-comparison-provided-2026-10-18.jsonl) preserve the full results. The earlier readiness record remains a historical preparation checkpoint.

The library now exposes this repeated-generation pattern through the [prepared-prompt workflow](krea-generation.md#prepare-once-render-multiple-seeds): prepare once, then vary image configuration while retaining the generation session. Its CPU and native qualification status is recorded separately from these historical benchmark reports.

## Residency and load path: 2026-10-26

Plans may now set `"residency": "resident"`. When it's omitted, earlier plans keep their balanced sessions, and reports record the residency used. The [summary record](evidence/krea-residency-load-path-2026-10-26.json) covers small 512×512 runs (one warm-up, three to five measurements) on the same M5 Max and OS build. Every iteration matched the independent references. These are exploratory measurements, not release benchmarks.

- **Steady sampling is about 0.535 s per step**, against 2.08 s in the October 18 comparison, with identical latent and pixel hashes. A standalone Core AI harness matches it: four segments take 0.52 s with or without fresh host copies, so boundary copies are not the bottleneck. The October 18 figures should be treated as superseded.
- **Fresh prompts (encoded profile):** the median is 14.1 s balanced and 5.0 s resident. Resident mode keeps the encoder (0.17 s per encode) and denoisers loaded, peaking at 22.9 GiB Metal allocation against 15.3 GiB. Balanced mode spends about 4–6 s reacquiring the encoder and 5–7 s reloading denoiser functions per request.
- **The first image in a process stays slow:** 55–79 s encoded and about 23 s provided, mostly first model opens and function loading.
- **Concurrent function loading was tried and reverted.** Loading the four denoiser functions concurrently cut the first image to 10–12 s. However, three of five concurrent runs settled into slower steady steps (0.8–1.0 s), while all five lazy runs stayed at 0.52–0.64 s. Repeated generation is the common case, so lazy loading remains.
