# On-device qualification (iOS)

What 0.1.0 qualifies on a physical iPhone, against the same CPU references and bars as the Mac gates. The harness is
`Tools/DeviceGate` (an xcodegen app outside the package, so CI never builds it):

```bash
ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… Tools/DeviceGate/run-on-device.sh <udid>   # build Release, sign, install, copy bundles + references, run, pull results
GATES=m4 ASC_KEY_ID=… … Tools/DeviceGate/run-on-device.sh <udid> [result-name]           # one family only (sd-turbo, kokoro, m4)
```

The device must be unlocked and reachable (cable, or the same network with Developer Mode on): `devicectl` refuses to
launch on a locked phone.

It signs with the App Store Connect API key in `~/.appstoreconnect/private_keys/`, has the increased-memory-limit
entitlement, copies bundles and references into the app's Documents with `devicectl` (bundles aren't embedded), launches
with a run ID, and pulls `Documents/gate.json` plus each output as raw float32. The Mac gates write the same outputs
with `AETHER_GATE_DUMP=<dir>`, for the device-versus-Mac comparison.

## 2026-11-06: iPhone Duo (`iPhone19,4`), Core AI `h19p`, iOS 27.2

Release build; 6.0 GiB available to the app at launch. Raw result: `Tools/DeviceGate/results/gate-iPhone19,4.json`.
All 27 checks pass.

### SD-Turbo: the iOS bundle is mixed precision

The float32 bundle plans about 4.8 GiB per phase (prompt held while the steps run), too close to the phone's 5.5 GiB
budget. The Forge exporter gained per-graph precision (`--precision`, `--text-precision`, `--vae-precision`;
float16 keeps the checkpoint's own fp16 weights, and graph inputs and outputs stay float32, so the driver is unchanged):

| Export (on the Mac, vs CPU float32 Diffusers) | Per-graph check | Whole image, 1 step | Whole image, 4 steps (bar RMSE 0.03) |
|---|---|---|---|
| float32 (the macOS bundle) | all exact | RMSE 1.0e-5 | RMSE 3.3e-5 |
| everything fp16 | all pass (text cosine 0.9999986) | RMSE 0.0030 | **RMSE 0.0329: fails** |
| **text encoder fp32, U-Net and VAE fp16** (the iOS bundle) | all pass (text 1.0, ε ≥ 0.9999998, VAE RMSE ≤ 0.0004) | RMSE 0.0030 | RMSE 0.0226 |

The bars are unchanged. With fp16 everywhere, the text rows' small error conditions every step and compounds over four;
keeping the text encoder in float32 costs nothing at the peak (it's the prompt phase, not the steps). The iOS bundle is
3.2 GB (float32: 4.7 GB) with `variant.platform: ios`.

On the iPhone (the iOS bundle):

| Case | RMSE vs reference (bar 0.03) | PSNR | Repeat |
|---|---|---|---|
| teapot, 1 step | 0.0029 | 56.7 dB | bit-identical |
| fox, 4 steps | 0.0124 | 44.2 dB | bit-identical |

- **Not bit-identical to the Mac's output** for the same bundle and seed (fox, 4 steps: RMS difference 0.017 on
  [-1, 1]); both are within the bar against the reference. fp16 GPU arithmetic differs across devices, as the decoder
  engine's cross-device finding already showed, so the device gate compares against the reference, not the Mac.
- **Memory:** phase peaks 1.25 GiB (prompt), 1.71 GiB (steps), 0.63 GiB (image), measured by the injected probe. The
  plan fits the image-generation budget on iOS (`os_proc_available_memory` less 512 MB: 5.5 GiB at load). Available
  memory after generating: 5.3 GiB.
- **Time:** the first generation takes 23 s (on-device specialization of the graphs, cached afterwards); after that,
  1.2 s for 1 step and 0.8 s for 4 steps.

### Kokoro-82M: frame cap 512 on iOS

The macOS bundle runs unchanged (durations on the GPU with the reshape hint, the decoder's buckets).

| Case | Frames → bucket | rel. L2 (bound) | log-mel mean (bound 0.03) | Repeat |
|---|---|---|---|---|
| short-af_heart | 98 → 128 | 0.0083 (0.029) | 0.0033 | bit-identical |
| sentence-am_michael | 247 → 256 | 0.0258 (0.05) | 0.0107 | bit-identical |
| speed-bf_emma ×1.2 | 180 → 256 | 0.0277 (0.05) | 0.0068 | bit-identical |
| long-af_bella ×0.9 | 913 | refused: above the 512-frame cap (`invalidParameters`, before decoding) | | |
| medium-af_bella | 442 → 512 | 0.0222 (0.05) | 0.0145 | bit-identical |

- Token ids and frames per token equal upstream's in every case; the waveforms differ from the Mac's in low bits (rel.
  0.6–2.2%), within the bounds against the reference.
- **Memory:** 0.85 GiB peak, 0.77 GiB held with buckets up to 512; peak physical footprint 2.3 GiB.
- **Time:** the first utterance 7.5 s (specialization), then 0.7–1.6 s per utterance.

### FLUX.2 and Krea

macOS-only in 0.1.0: both pipelines refuse an iOS device (`unsupportedDevice`) before loading anything, covered by
`Flux2KleinTests` and `Krea2Tests`.

## M4 models (`GATES=m4`)

```bash
GATES=m4 ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… Tools/DeviceGate/run-on-device.sh <udid>   # → results/gate-<productType>-m4.json
```

The harness is `Tools/DeviceGate/Sources/VisionGates.swift`. It covers `depth-anything-3-small`,
`embeddinggemma-300m`, `clip-vit-b32`, `yolox-s`, `rf-detr-nano` and `rf-detr-seg-nano`.

- **Bundles:** the Forge exports in `forge-outputs/<id>/bundle`, copied unchanged. They're JIT `.aimodel` packages with
  nothing per architecture, so the macOS files serve iOS. Each loads with an **iOS variant manifest**: the script
  writes a copy of the export's `.aether/aether.json` with `variant.platform: ios` to `Tools/DeviceGate/.staging/manifests/`
  (gitignored) and copies it to `Documents/manifests/<id>.json`. The app checks the platform before it loads anything.
  The result records the SHA-256 of each bundle's `SHA256SUMS` as copied (`bundles`), so it names the export that was
  measured.
- **Parity (the qualification, `checks`):** each case's reference input tensor goes straight into the driver, on the
  GPU. It's compared against the same CPU float32 reference and bars as the Mac gates, and a repeat must be
  bit-identical. The RF-DETR reference detections are the reference's raw outputs through the driver's own head. The
  end-to-end parts of the Mac gates need the upstream source images, so they aren't run on the device.
- **Placement (a measurement, `placement`; it never fails the gate):** the same graphs with every `compute` set to
  `gpu`, then to `neuralEngine`. Each setting runs the parity cases, then a sustained run of at least 30 inferences and
  10 s on the first case. The run records:
  - latency: median, p90 and minimum;
  - the CPU energy (`task_energy`), GPU time (`task_gpu_utilisation`) and CPU time (`getrusage`) billed to the app;
  - the ledger peak, the footprint delta, and the thermal state before and after.
- **Build:** the M4 gates `@testable`-import AetherRuntime and AetherVision (to build `PreparedImage` from a tensor and
  to reach the detector heads), so the script builds every target with `ENABLE_TESTABILITY=YES`.

### 2026-11-06: iPhone Duo (`iPhone19,4`), Core AI `h19p`, iOS 27.2

Release build; 6.0 GiB available at launch. `ComputeUnitKind.availableKinds` is `cpu`, `gpu` and `neuralEngine`. Raw
result: `Tools/DeviceGate/results/gate-iPhone19,4-m4.json` (run `run-20261106-162859`). All 92 checks pass.

The exports tested (the first 12 hex digits of the SHA-256 of each bundle's `SHA256SUMS`):

| Model | Export | `SHA256SUMS` |
|---|---|---|
| `depth-anything-3-small` | parity-resize re-export, 15:21 | `fe9ccedb5cf5` |
| `embeddinggemma-300m` | 13:05 | `c4953b6f2d98` |
| `clip-vit-b32` | 12:55 | `233956bd6632` |
| `yolox-s` | parity-resize re-export, 14:08 | `48bd6b75091d` |
| `rf-detr-nano` | parity-resize re-export, 14:08 | `10db698f7d05` |
| `rf-detr-seg-nano` | parity-resize re-export, 14:08 | `ac53fac5b4a8` |

#### Parity on the GPU (Mac gate bars)

| Model | Cases | Result (bar) |
|---|---|---|
| DA3-small | 3 | depth rel L2 0.00017–0.00065, confidence 0.00027–0.0015 (0.02) |
| EmbeddingGemma 300M | 7 | ids and masks exact; cosine ≥ 0.9999999 (0.999); retrieval 3/3, cosine table within 5.1e-5 (0.005) |
| CLIP ViT-B/32 | 17 texts, 7 images | tokenizer 17/17; text cosine ≥ 0.9999992, image 0.9999983–0.9999996 (0.999); zero-shot argmax 7/7 |
| YOLOX-S | 6 | 34/34 detections matched, IoU ≥ 0.9999978 (0.95), \|Δscore\| ≤ 2.4e-6 (0.01) |
| RF-DETR nano | 6 | 54/54 matched, IoU ≥ 0.9999964 (0.95), \|Δscore\| ≤ 3.8e-6 (0.01), none unmatched |
| RF-DETR seg-nano | 6 | 47/47 matched, IoU ≥ 0.9999968 (0.95), \|Δscore\| ≤ 1.6e-6 (0.01), mask IoU ≥ 0.9973 (0.95) |

Every case's repeat is bit-identical. The numbers match the Mac gates' to the same order of magnitude.

#### GPU versus `neuralEngine`

The sustained run is on the first case, and the thermal state was nominal before and after every run. "First request"
covers loading, specializing (or reading the cache) and all the parity cases. CPU energy is what's billed to the app
per inference.

| Model | Compute | Median ms | p90 ms | Min ms | CPU ms/run | CPU mJ/run | Ledger peak MB | First request s |
|---|---|---|---|---|---|---|---|---|
| DA3-small (504²) | gpu | 37.3 | 37.7 | 36.7 | 9.0 | 39.4 | 208 | 0.37 |
| | neuralEngine | 36.6 | 36.9 | 36.1 | 6.0 | 29.9 | 126 | 0.29 |
| EmbeddingGemma (256 tokens) | gpu | 32.3 | 32.9 | 31.4 | 7.9 | 29.2 | 503 | 1.53 |
| | neuralEngine | 32.4 | 33.2 | 31.3 | 5.6 | 20.3 | 500 | 1.52 |
| CLIP image tower | gpu | 4.9 | 5.2 | 3.6 | 1.5 | 6.2 | 308 | 0.55 |
| | neuralEngine | 5.6 | 5.9 | 3.8 | 1.3 | 5.4 | 377 | 0.51 |
| YOLOX-S (640²) | gpu | 20.6 | 20.9 | 17.7 | 4.9 | 22.3 | 100 | 2.10 |
| | neuralEngine | 20.2 | 20.9 | 19.6 | 4.3 | 21.5 | 74 | 0.43 |
| RF-DETR nano | gpu | 23.9 | 24.2 | 17.6 | 3.8 | 15.8 | 79 | 3.91 |
| | neuralEngine | 24.3 | 24.8 | 23.0 | 3.0 | 13.4 | 58 | 0.62 |
| RF-DETR seg-nano | gpu | 38.6 | 39.3 | 33.9 | 7.2 | 31.1 | 88 | 1.96 |
| | neuralEngine | 38.4 | 41.4 | 37.8 | 5.7 | 27.0 | 98 | 0.90 |

- **Equal within noise, and identical outputs.** The medians agree within about 1 ms for every model; CLIP's image
  tower is 0.7 ms slower on `neuralEngine`. Every parity metric of the `neuralEngine` run equals the GPU run's to full
  printed precision, on every case of every model. That is the Mac's finding again (GPU-identical output on 9 of 11
  graphs): `.neuralEngine` is only a preference (`allowedComputeUnitKinds` is read-only), and nothing here shows these
  graphs left the GPU.
- **Energy: what is and isn't measurable.**
  - `task_gpu_utilisation` reads 0 for **both** settings, including the plain GPU runs, so Core AI's GPU work isn't
    billed to the app through it. It can't show placement or GPU energy.
  - Neural Engine work runs in `aned` and isn't billed to the app either.
  - `task_energy` is the app's CPU energy only: 10–30% lower with `neuralEngine` (with matching CPU time). That's host
    overhead, not the accelerator's energy.
  - `powermetrics` measures only the Mac. No whole-device power figure was taken.

  So no measured energy win exists for either unit.
- **Memory:** the ledger peaks are 58–503 MB (EmbeddingGemma's 0.5 GB is its weights); the app's peak physical
  footprint over the whole run was 383 MB. The footprint deltas are dominated by earlier models' memory being released, so
  the ledger is the figure to use. The first-request times after the GPU run are lower partly because the input and
  reference files are already in the page cache.
- **Decision (SUMMARY #2):** no M4 entry earns a `neuralEngine` override. There's no latency win, the outputs are
  identical, and there's no measurable accelerator energy. GPU stays the placement for every class.

**Proposed, not applied:** all six pass, so each could add an `ios` catalog variant built from the same files with
`variant.platform: ios`. Each needs a re-pinned manifest (a new revision with its own SHA-256s) under
`aether-models`; that's the owner's call. The candidates: `depth-anything-3-small` (its exporter already has
`--platform ios`), `embeddinggemma-300m`, `clip-vit-b32`, `yolox-s`, `rf-detr-nano` and `rf-detr-seg-nano`.
