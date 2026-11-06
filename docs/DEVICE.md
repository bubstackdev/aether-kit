# On-device qualification (iOS)

What 0.1.0 qualifies on a physical iPhone, against the same CPU references and bars as the Mac gates. The harness is
`Tools/DeviceGate` (an xcodegen app outside the package, so CI never builds it):

```bash
ASC_KEY_ID=… ASC_ISSUER_ID=… DEVELOPMENT_TEAM=… Tools/DeviceGate/run-on-device.sh <udid>   # build Release, sign, install, copy bundles + references, run, pull results
```

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
