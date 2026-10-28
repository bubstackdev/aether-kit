# Spike: TripoSR image-to-3D on Core AI (2026-10)

**Status: 0.0.3 go/no-go result (runs on 2026-10-28).** This is the TripoSR decision for the [0.0.3 roadmap](../roadmap-0.0.3.md#spikes). The [0.0.3 decision](#003-decision-2026-10-28) section answers what the 0.0.2 feasibility result left open; the [0.0.2 feasibility result](#002-feasibility-result-runs-on-2026-10-26) follows it, kept as it was.

**Verdict: go.** On six real images through Kit-style preprocessing, Core AI fp16 matches eager fp32 on every gate criterion: occupancy IoU 0.996–0.998 at 256³ (bar 0.98), Chamfer distance 0.14–0.16% of the bounding-box diagonal (bar 1%, and at the sampling noise floor), 0.37–0.38 s from image to coloured mesh after load (bar about 3 s; the sum of separately timed stages), and a self-authored USDZ that passes `usdchecker --arkit`. Every figure is from a single run. The gather-free grid function cut the density grid from 1.20 s to 0.16–0.17 s, but only after replacing an `einsum` that Core AI miscompiles silently. iPhone is outside this decision.

## 0.0.3 decision (2026-10-28)

### Setup

- Same device, exporter and runtime as the 0.0.2 run below: M5 Max, 64 GB, macOS 27.0.1 (26A434); coreai-torch 0.4.2 and torch 2.11 from Forge's locked environment, plus the spike venv in `Spikes/triposr/requirements-spike.txt`; OS Core AI, GPU-preferred specialization.
- Model `stabilityai/TripoSR@5b52193` (`model.ckpt` SHA-256 `429e2c6b…`), the `facebook/dino-vitb16@f205d5d` config, and the upstream source at `VAST-AI-Research/TripoSR@107cefdc244c` (MIT), imported read-only.
- **Images:** six of the 14 examples in the upstream repository (`examples/`): a phone photo of a plush toy (`captured.jpeg`), `chair`, `hamburger`, `poly_fox`, `robot` and `teapot`. They're used in scratch only and not committed, because their individual provenance isn't stated; `results/0.0.3/inputs.json` records their hashes.
- **Preprocessing, as Kit would do it:**
  - Background: the three images with transparency keep it, as upstream's `remove_background` does. The other three get Vision's foreground-instance mask (`foreground_mask.swift`) instead of upstream's `rembg`.
  - Then upstream's `resize_foreground(0.85)`, the 0.5 grey composite and the model's 512×512 resize (`preprocess.py`).
  - Eager and Core AI read the same `.npy` inputs, so the comparison is of the model only.
- **Exports (fp16, static shapes):** `scene` (800 MB), `field` at 65,536 and at 262,144 points, and `grid-256x16`, the gather-free density over 16 z-planes of the 256³ grid (each about 0.1 MB). Each export took 0.3–2.9 s, at up to 3.85 GB peak RSS.
- **References:** eager fp32 on the CPU, through upstream's own `renderer.query_triplane` (`grid_sample`): scene codes, the 256³ density grid, and the marching-cubes mesh at threshold 25 (`reference_images.py`). Per image, 2.3–2.6 s for the scene codes and 4.7–6.2 s for the grid on a shared host.
- **Core AI run:** one GPU hold of 27 s under `with-gpu-lock.sh` (`gpu_run_images.sh`, 08:09:10–08:09:37), on a lightly loaded shared host: one-minute load 5.1 before the lock wait and 7.8 right after the hold (`host-load.log`). Peak RSS 3.72 GB, footprint 3.86 GB.

### The gate

| Image | IoU, field 65,536 | IoU, field 262,144 | IoU, slab grid | Chamfer / diagonal | Image to coloured mesh |
| --- | --- | --- | --- | --- | --- |
| captured (Vision mask) | 0.9968 | 0.9968 | 0.9968 | 0.143% | 0.384 s |
| chair | 0.9964 | 0.9964 | 0.9965 | 0.143% | 0.369 s |
| hamburger | 0.9964 | 0.9964 | 0.9967 | 0.161% | 0.374 s |
| poly_fox (Vision mask) | 0.9965 | 0.9965 | 0.9968 | 0.150% | 0.369 s |
| robot | 0.9983 | 0.9983 | 0.9982 | 0.162% | 0.379 s |
| teapot (Vision mask) | 0.9977 | 0.9977 | 0.9977 | 0.142% | 0.374 s |
| **Bar** | ≥ 0.98 | ≥ 0.98 | ≥ 0.98 | < 1% | < about 3 s |

All six pass, and so does the USDZ check below ([gate record](../../Spikes/triposr/results/0.0.3/gate.json)).

- **IoU** is occupancy at threshold 25 over the 256³ grid, against the eager fp32 grid.
- **Chamfer** is the mean of both directions' nearest-neighbour distances over 100,000 surface samples per mesh. It sits at the metric's own floor: two different samplings of the *same* eager mesh give 0.140–0.162% (`mesh_metrics.py`). So the bar can't tell these meshes from the reference; the vertex distances below are the sharper measure.
- **Vertices** land within 0.15% of the diagonal of an eager vertex at the 99th percentile, and within 0.57% at worst (hamburger). Vertex counts are within 0.25% of the eager meshes' (40,814–80,682).
- **Scene codes:** cosine ≥ 0.9999992 against eager fp32, max relative error ≤ 2.4e-3.
- **Faces** wind outward: every mesh is watertight with a positive signed volume.
- **Time** is image to coloured mesh after load, summed from separately timed stages of one run: warm scene codes (59–60 ms), the slab grid (159–168 ms), marching cubes on the CPU (135–146 ms, PyMCubes) and vertex colours through `field` (7–20 ms). It excludes background removal and the resize, which weren't timed, and the first scene call (75–92 ms). Loading the four assets took 0.95 s with their specializations already cached.

### The grid: about 7× faster, after working around a Core AI miscompile

- **Times for the 256³ grid:** `field` in chunks of 65,536 takes 1.20–1.21 s (256 calls), and in chunks of 262,144 it is slower, at 2.31–2.43 s (64 calls). The gather-free slab function takes 0.16–0.17 s (16 calls). That's the 0.0.2 report's open item 3: the grid is no longer most of the time.
- **The miscompile.** The slab function first sampled each plane with one `einsum` (`"jv,cvu,iu->ijc"`). It ran correctly in eager: on one slab of the chair's grid, max relative error 5.2e-6 against upstream's `grid_sample` at fp32 and IoU 0.996 at fp16 (`diag-grid-export.json`). But the Core AI export returned unrelated values, with no error: slab-grid occupancy IoU 0.001–0.003 on all six images, with 426–6,322 non-finite densities each, which gave marching cubes NaN vertices (`einsum-grid-iou.json`, `run-images-256-fp16-einsum-grid.json`).
  - `diag_grid.py` narrows it down: the einsum alone, exported as its own function, has cosine −0.0003 against eager.
  - The same interpolation as two `F.linear` calls (the hat-function weights as the weight matrix) exports correctly: cosine 0.99998 and IoU 0.996 on the same slab.
  - Two other spellings, a broadcast `matmul` and a `bmm` over an expanded batch, fail coreai-torch's optimizer instead: `coreai.broadcast_in_dims` from `tensor<1x40x256x256xf16>` to `tensor<40x256x256xf16>`, whose operand and result ranks differ (`diag-grid-export.json`).
  - `tsr_graphs.GridField` now uses the linear form, and `check_triplane_sampling.py` still matches `grid_sample` to 2.7e-16 (`check-triplane-sampling.log`).
  - This is an Apple Feedback candidate (coreai-torch 0.4.2 on macOS 27.0.1). `diag_grid.py --export` then `--run` reproduces it with any scene codes. Kit's port must check its grid function against eager on real inputs; a silent miscompile looks like a modelling bug.
- **fp16 overflow.** The density is `exp(…)`, and deep inside dense objects the eager fp32 grid reaches 108,673 on the hamburger, above fp16's 65,504; 45 of its points exceed that limit, and no other image's do (`mesh-metrics.json`). Core AI's grid overflowed to infinity at 46 points there (`run-images-256-fp16.json`), and occupancy is unaffected. Marching cubes would turn the infinities into NaN vertices, so `run_images.py` clamps them to 65,504; Kit's extraction must do the same, or compute the `exp` in fp32.

### USDZ

`write_usdz.py` authors the stage directly from the mesh, with no OBJ round trip:
- the mesh's points and faces;
- per-vertex colours as a `displayColor` primvar;
- a UsdPreviewSurface material reading that primvar;
- `metersPerUnit`, `upAxis` and an extent.

It converts the stage to usdc with `usdcat`, then packages it as a stored zip whose entry data starts on a 64-byte boundary. It does the packaging itself because `usdzip` doesn't exist on iOS.

The teapot (1,258,071 bytes) and the plush toy (1,461,497 bytes) both pass `usdchecker --arkit` ("Success!"). Writing each took 0.14 s and 0.17 s, with a heavy CPU job running alongside (`write-usdz.log`, the two `usdchecker` logs). That closes the 0.0.2 report's open item 4. A Kit port still needs a Swift writer (the stage is a few hundred lines of text plus usdc conversion; whether Kit can write usdc without `usdcat` on iOS is the port's question).

### What the 0.0.2 open items came to

1. **Real images:** six real images, three through Vision's mask, all pass. The Vision mask stands in for `rembg` for Kit; the references use the same masked inputs, so the gate measures the model, not the matting.
2. **iPhone:** outside this go/no-go, by decision. The fp16 weights are 800 MB, and the slab grid still runs the decoder on every one of the 16.8M grid points.
3. **Grid cost:** solved by the slab function, 0.16 s.
4. **USDZ:** solved, `usdchecker --arkit` passes. Face winding: outward, on every mesh.
5. **fp16 error:** within the gate on every image; the overflow above needs the clamp.

### Effort, revised

The 0.0.2 estimate (about 1.5–2 engineer-weeks) stands, with two items now cheaper and two added:
- Cheaper: the grid formulation (done, two small matmuls per plane) and USDZ (the stage and packaging are known to pass).
- Added: the overflow clamp, and a native check of the grid function against eager in Kit's qualification, given the silent miscompile.

### Evidence (0.0.3)

- **Code** (`Spikes/triposr/`):
  - preprocessing: `foreground_mask.swift`, `preprocess.py`;
  - exports and the graphs: `export_triposr.py`, `tsr_graphs.py`;
  - references: `reference_images.py`;
  - the GPU run: `run_images.py`, `gpu_run_images.sh`;
  - judging: `gate_triposr.py`, `mesh_metrics.py`;
  - output: `write_usdz.py`;
  - checks: `check_triplane_sampling.py` and `diag_grid.py`.
- **Results** (`Spikes/triposr/results/0.0.3/`):
  - `gate.json` and `mesh-metrics.json`;
  - `run-images-256-fp16.json`, the GPU run, and `run-images-256-fp16-einsum-grid.json`, the first run, with the miscompiled grid;
  - `references.json`;
  - `diag-grid-run.json` (the GPU comparison) and `diag-grid-export.json` (the eager checks and the two export failures);
  - `einsum-grid-iou.json`: the first run's IoUs and non-finite counts, from `gate_triposr.py --iou-only`;
  - `write-usdz.log` and the two `usdchecker` logs;
  - `check-triplane-sampling.log`, `gpu-lock.log` (this spike's lock lines), `host-load.log` and `data-footprint.log`;
  - `inputs.json`: image and checkpoint hashes;
  - `/usr/bin/time` excerpts for the exports, references and GPU run.
- **GPU lock:**
  - `spikes: triposr images`, 07:49:17–07:49:58 (the einsum grid);
  - `spikes: triposr grid diagnosis`, 07:53:18–07:53:20;
  - `spikes: triposr images (linear grid)`, 08:09:10–08:09:37.
- **Data:** weights, inputs, exports, grids and meshes are under the spike data directory on the Scratch volume (4.5 GB without the venv, `data-footprint.log`), not committed.

### Reproducing (0.0.3)

With `HF_HUB_CACHE` pointing at a cache holding the two pinned snapshots, and the spike venv:
1. `swift foreground_mask.swift <image> <masked.png>` per image. It runs Vision, so put it under the GPU lock.
2. `preprocess.py <inputs> <masked…>`.
3. `export_triposr.py <src> <exports> --only scene,field`, again with `--only field --points 262144`, and with `--only none --grid-slab 16 --res 256` (into separate directories, since each writes its own reference points).
4. `reference_images.py <src> <inputs> <references>`.
5. `gpu_run_images.sh` under the lock, with `SPIKE_ASSETS` set to the data directory that holds `triposr/{exports,inputs}` and the venv.
6. `gate_triposr.py <references> <results> gate.json --usdz-log <log>` after `write_usdz.py` and `usdchecker --arkit`.
7. `mesh_metrics.py`.
8. For the miscompile: `diag_grid.py <src> <codes.npy> <dir> --export --ref-grid <stem>-ref-grid.npy`, then `--run` under the GPU lock.

## 0.0.2 feasibility result (runs on 2026-10-26)

The rest of this report is the 0.0.2 result as it was imported for 0.0.3. Its verdict was **go for 0.0.3**: both neural stages exported on the first try and matched eager fp32, single-image-to-mesh took about 2.5 s on the M5 Max, and the one real gap was Core AI's missing `grid_sample`, replaced by an explicit gather that matches upstream to 3.4e-6. The code under `Spikes/triposr/` is throwaway and is not wired into `Package.swift`.

## Setup

- Device: Apple M5 Max, 64 GB, macOS 27.0.1 (26A434).
- Exporter: coreai-torch 0.4.2 with torch 2.11 from the prototype's Forge venv, used read-only. A separate spike venv adds omegaconf, trimesh, PyMCubes and scikit-image, and points at Forge's site-packages for torch and Core AI.
- Runtime: `coreai.runtime` (OS Core AI, `USE_OS_COREAI=1`), GPU-preferred specialization.
- Model: `stabilityai/TripoSR@5b52193`, 419.3M parameters:
  - DINO ViT-B/16 image encoder, 86.4M;
  - triplane tokens, 3.1M;
  - 16-layer triplane transformer, 329.5M;
  - ConvTranspose upsampler;
  - 10-layer NeRF MLP.
- Code: the upstream MIT source tarball (VAST-AI-Research/TripoSR, `main`), imported read-only.
- Input: a fixed synthetic 512×512 image (a coloured disc on grey). It exercises the numerics and timing, not quality. Every GPU job ran under `with-gpu-lock.sh`.

## What was exported

| Asset | Function | Inputs → outputs | Size (fp16) |
|---|---|---|---|
| `scene-fp16.aimodel` | `main` | image [1,3,512,512] → scene codes [3,40,64,64] | 800 MB |
| `field-fp16.aimodel` | `main` | scene codes [3,40,64,64], points [65536,3] → density [N,1], colour [N,3] | 108 KB |

Both exports are static-shape. Conversion took 3.6 s (scene) and 0.4 s (field) on CPU.

Two changes from upstream were needed:

1. **Position embedding.** DINO's position embedding is interpolated bicubically from 14×14 to 32×32. The input size is fixed at 512, so it is precomputed in eager and baked in as a constant, keeping bicubic interpolation out of the graph.
2. **`grid_sample` has no lowering.** coreai-torch rejects `aten.grid_sampler_2d` ("unsupported ATen ops … use register_torch_lowering()"; `Spikes/triposr/probe_grid_sample.py`). The field function samples bilinearly with an explicit `index_select` gather instead. It zero-pads each plane by one texel, which reproduces `align_corners=False` with zero padding.
   - Against upstream `renderer.query_triplane` (grid_sample) on 65,536 random points: density relative error **3.4e-6**, colour max error 1.8e-5.

## Measurements (M5 Max GPU, fp16)

| Stage | Result |
|---|---|
| Model load (both assets, cached specialization) | 2.37 s |
| Scene codes (image → triplanes), first / warm | 0.64 s / **61 ms** |
| Scene codes vs eager fp32 | cosine 0.9999997, relative max error 1.2e-3 |
| Density field on random points vs eager fp32 | density relative error 1.1%, colour max error 0.077 |
| Density grid 256³ (16.8M points, 256 calls of 65,536) | **2.07 s** |
| Grid occupancy vs eager fp32 grid (threshold 25) | IoU **0.9990** |
| Marching cubes 256³ (PyMCubes, CPU) | 0.17 s, 99,424 vertices / 198,844 faces (eager-grid mesh: 99,454 / 198,904) |
| Vertex colours (Core AI field at the vertices) | 0.02 s |
| OBJ write (trimesh, vertex colours) | 0.21 s (10.8 MB) |
| **Total after load, excluding background removal** | **≈ 2.5 s** |
| Process peak RSS (whole run) | 3.46 GB |

For comparison, the eager fp32 CPU reference takes 3.7 s for the scene codes and 5.9 s for the 256³ grid.

The grid dominates the total: 256 sequential calls at about 8 ms each. A bigger chunk, or a separable grid formulation, would cut that substantially. On a regular grid, sampling each plane reduces to two small interpolation matmuls, which also removes the gather.

## Mesh extraction and output formats

- **Marching cubes.** Upstream uses `torchmcubes`, which is CUDA or C++ with no Apple build. PyMCubes on CPU does 256³ in 0.17 s, so a Swift port of classic marching cubes on CPU (a few hundred lines, or Accelerate-assisted) is enough. A Metal version is optional; it would only matter at 512³ or above. Mesh orientation (face winding) still needs checking against upstream in a real port.
- **OBJ.** Trivial to write, with per-vertex colours as `v x y z r g b`.
- **USDZ.** Model I/O on macOS 27.0.1 reports `canExportFileExtension("usdz") == false`, though `usdc`/`usda` are true. Two routes were tried:
  - Model I/O → `.usdc` → `/usr/bin/usdzip` gives a valid archive (2.8 MB, 0.47 s export).
  - SceneKit `SCNScene.write(to: …usdz)` works directly (0.36 s) and also exists on iOS.
  - **Both fail `usdchecker`:** a missing `metersPerUnit`, an invalid material binding from the OBJ's MDL material, and a constant `occlusion = 0`.
  - The kit should author the USD stage itself rather than round-trip through OBJ: a mesh with a `displayColor` primvar or a baked texture, a UsdPreviewSurface material, and `metersPerUnit`. It then packages that as USDZ, a zip with 64-byte-aligned stored entries. `usdzip` is not available on iOS.

## Preprocessing and licensing

- **Model.** `stabilityai/TripoSR` is **MIT**: confirmed from the model card (`license: mit`) and the source repository's LICENSE (© 2024 Tripo AI & Stability AI). Ungated.
- **Image encoder.** The DINO ViT-B/16 config comes from `facebook/dino-vitb16`, **Apache-2.0**. Only its config is fetched; the weights are inside TripoSR's checkpoint.
- **Training-data note** from the card: curated Objaverse renders (CC-BY).
- **Background removal.** Upstream uses `rembg` (u2net). The kit should use Apple's Vision foreground-instance mask instead, avoiding a third-party matting model, and then replicate upstream's `resize_foreground(0.85)` and grey composite. This changes inputs relative to upstream, so qualification needs its own references.

## TRELLIS.2: could its sparse structured latents map onto Core AI?

Not as a near-term item. TRELLIS.2-4B (`microsoft/TRELLIS.2-4B`) is MIT and ungated, but:

- **Licence blockers in the default pipeline.**
  - The image conditioner is DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`): custom licence (`license:other`), manually gated.
  - Background removal is `briaai/RMBG-2.0`: custom licence, gated.
  - Under the ground rules (no gated acceptance) neither can be downloaded, and the pipeline would need a substitute conditioner, which means retraining.
- **What maps:**
  - The flow transformers: the dense 16³ sparse-structure DiT (1.3B, 30 blocks, 1536-d) and the 1.3B SLat DiTs over about 9.6K active voxel tokens at 1024³.
  - These are ordinary transformers with RoPE and cross-attention. With token-count buckets plus masks they would export like Krea or FLUX.
- **What does not map:**
  - The sparse 3D VAE decoders (`FlexiDualGridVaeDecoder`: SparseConvNeXt blocks plus sparse conv-to-subdivision upsampling) need submanifold sparse convolution over hashed neighbour maps (spconv, flex_gemm). Core AI has no sparse-conv primitive.
  - An emulation would build the neighbour index maps on the host or in a Metal kernel, then express each sparse conv as a 27-way gather plus matmul with bucketed voxel counts.
  - O-Voxel → mesh conversion is a CUDA extension (`o_voxel`) that would need a full port.
- **Scale.** 16.2 GB of checkpoints (4B parameters). Authors' targets are 24 GB or larger NVIDIA GPUs: about 3 s at 512³ and 17 s at 1024³ on an H100.
- **Assessment.** Several weeks of custom-kernel work plus a licence-clean conditioner. Not a 0.0.3 candidate.

## Effort estimate (TripoSR into AetherKit 0.0.3)

About 1.5–2 engineer-weeks:

- Two-asset export recipe with fixed shapes and fp16: about 1 day, done in prototype form here.
- Swift pipeline: preprocessing (Vision mask, crop, resize, composite), scene and field sessions with the kit's admission, cancellation and unload contract, grid chunking, CPU marching cubes, and vertex colours. About 4–5 days.
- Output: an OBJ writer plus a self-authored USDZ writer that passes `usdchecker`. About 2 days.
- Qualification: eager fp32 references on a fixed image set (scene-code tensors, density grids, mesh IoU or Chamfer distance), an iPhone run (800 MB of fp16 weights; grid about 1.4 TFLOP at 256³), and a subjective review. About 3 days.

## Recommendation

**Go for 0.0.3**, scoped as image → mesh (OBJ + USDZ) at 256³ with vertex colours, on Mac first with iPhone qualification. Keep TRELLIS.2 out of scope. Revisit it only if a licence-clean conditioner appears and Core AI gains, or the kit takes on, a sparse-convolution story.

## Disk footprint

- Peak: about 3.3 GB.
  - TripoSR checkpoint in the HF hub cache: 1.6 GB.
  - Exports: 0.8 GB.
  - Density grids, mesh and USD files: about 0.2 GB.
  - Share of the spike venv (also used by ACE-Step): 0.7 GB.
- After cleanup: **0 B**. TripoSR is a go, but 0.0.3 work is not scheduled yet. The exports regenerate in about 5 s, and the checkpoint is a 1.7 GB download, so nothing was kept.
- TRELLIS.2 and DINO: only configs and model cards were downloaded (under 100 KB), and those are deleted too.

## Evidence

- Code: `Spikes/triposr/`
  - `export_triposr.py`: export plus eager references;
  - `eager_grid.py`;
  - `run_triposr.py`: GPU run, mesh and comparisons, and `gpu_run.sh`, which runs it under `/usr/bin/time -l`;
  - `probe_grid_sample.py`;
  - `obj2usdz*.swift`.
- Results: `Spikes/triposr/results/`
  - `run-256-fp16.json`;
  - `run-256-fp16-time.log`: the run's `/usr/bin/time -l` output, the source of the 3.46 GB peak RSS (excerpted from the shared lock hold's log).
  - Large tensors, meshes and assets were written under `~/.cache/huggingface/aether-kit/spikes/triposr/` and deleted; they are not committed.
- GPU lock: one shared hold of 28 s with the ACE-Step stages (`spikes: triposr+ace-step stages`, 14:44:57–14:45:25).

## Reproducing

- **Weights and source.** The scripts read the newest snapshot of `stabilityai/TripoSR` from `~/.cache/huggingface/hub/` (the spike used revision `5b52193`). The upstream code is the MIT source tarball of VAST-AI-Research/TripoSR at `main`, passed as `<tsr-source-dir>`; the spike didn't record its commit. `torchmcubes` and `rembg` are stubbed, so neither is needed.
- **Python.** Forge's locked environment has the torch 2.11 and coreai-torch 0.4.2 the spike used, so `probe_grid_sample.py` runs in it. The upstream package and `run_triposr.py` need more: the spike's venv added omegaconf, trimesh, PyMCubes and scikit-image over Forge's packages, and upstream also imports einops. None of these is in Forge's lock, so install them in a separate venv and point `SPIKE_PYTHON` at it for `gpu_run.sh`.
- **USDZ checks.** `obj2usdz.swift` and `obj2usdz_scenekit.swift` run with `swift <script> <in.obj> <out.usdz>`; `usdchecker` and `usdzip` come from Apple's USD tools.

## Open for 0.0.3

What these findings leave unanswered before a go/no-go:

1. **Real images.** Only one synthetic disc on grey ran, which tests numerics and timing, not mesh quality. Neither the Vision foreground-mask preprocessing nor its effect relative to upstream's `rembg` path was tried, and a mesh metric (IoU or Chamfer) on a fixed image set has no threshold yet.
2. **iPhone.** Not run: 800 MB of fp16 weights and about 1.4 TFLOP for a 256³ grid.
3. **The grid cost.** The density grid is 2.07 s of the 2.5 s total. Bigger chunks and the separable, gather-free formulation were proposed but not tried.
4. **Output that passes `usdchecker`.** Both USDZ routes failed it. The self-authored USD stage and USDZ writer, a Swift marching-cubes port, and face winding against upstream are still to be done.
5. **fp16 error budget.** The field on random points differs from eager fp32 by 1.1% in density and 0.077 in colour, though grid occupancy IoU is 0.999. Whether that holds on real scenes, and what tolerance the qualification uses, is open.
