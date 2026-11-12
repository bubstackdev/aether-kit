# Krea 2 speed experiments, 2026-11-12

On the M5 Max (64 GiB, macOS 27.2), fp16 VAE computation was the repeatable export-side speed improvement.
At 1024² with 512-pixel tiles and 256-pixel overlap, warm isolated decodes took about 0.71 s versus 1.00–1.02 s
for the current fp32 decoder. Retained Metal allocation stayed at about 2.54 GiB (slightly lower with fp16).
The comparison includes all nine sequential decoder calls and Python stitching, with no denoisers loaded.
This is a decoder result, not a claim of 30% faster whole renders.

The fp16 tiled output reached SSIM 0.999974 / PSNR 67.93 dB against the current tiled decoder on the harbor
reference. Full fp16 decodes passed the saved Diffusers references at 512² (SSIM 0.999978 / PSNR 69.71 dB)
and 1024² (SSIM 0.999959 / PSNR 65.46 dB). Repeats were identical and all values finite. These are two images,
so fp16 remains opt-in and fp32 remains the default. The structural self-test also checks fp16 compute,
float32 output, and the unchanged float32 encoder path.

To make a new bundle with just this decoder changed:

```sh
Forge/.venv/bin/python -I Forge/krea2/export_krea2.py \
  --variant realism --replace --only vaeDecoder --vae-compute float16 \
  --base-bundle <existing bundle> --output <new bundle>
```

Weights remain fp16. Decoder convolutions and learned normalization parameters use fp16; latent
denormalization and returned RGB remain fp32. The manifest and export report record decoder precision,
including when a later partial export keeps the base bundle's decoder. Existing installed bundles are untouched.
Use `bench_krea2_vae_tiles.py` under `scripts/gpu-lease` to compare an export with saved reference latents,
both full and tiled, and `check_krea2.py` for the full graph/oracle gate. Its default image bars still apply.

Other candidates were tried without changing their production behavior:

- **256 fixed text rows:** one 1024² step comparison improved from 5.09 to 3.86 s (later-step means), but a repeat
  measured 4.54 versus 4.76 s. The repeat retained about 0.48 GiB less Metal with 256 rows. This is inconclusive
  speed evidence on a machine also doing CPU development/test work, so no default or routing change was made.
  The existing `--mask-text-rows 256` option remains available, with its existing prompt-capacity restriction.
- **Hoisted text fusion:** the split prototype produced bit-identical GPU segment outputs, saving about 56 ms
  per warm segment-zero call at 256 rows (0.838 to 0.782 s). Fusion itself took about 41 ms warm / 372 ms first-call,
  plus a separate graph load. The segment's first call took 12.37 versus 10.19 s in these processes; compiler/cache
  effects make this a report, not a controlled cold-speed comparison. There is no measured whole-render benefit
  sufficient to introduce another conditioning representation and graph lifecycle into the runtime. The reproducible
  `bench_krea2_fusion.py` probe is kept for further measurement; its assets are not kit bundles.
- **Zero overlap:** four decoder calls instead of nine took about 0.46 s at the same allocation, but SSIM fell to
  0.98638 against full decode, below the 0.99 tiled comparison target. Mean seam-band error rose to 0.0563 on the
  decoder's [-1, 1] scale. Keep overlap for quality.

Raw timing arrays, memory boundary samples, quality metrics, and source hashes are in `speed-report.json`.
`bench_krea2_steps.py` now records Metal allocations and warm-step medians as well as the existing timings/swap
figures. Each GPU measurement used a fresh process under the shared lease. These are development measurements;
CPU load and compiler-cache warmth were not held constant. Metal boundary samples are not continuous peak measurements.
