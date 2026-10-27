# Spike: TripoSR image-to-3D on Core AI (2026-10)

**Status: 0.0.2 feasibility result (runs on 2026-10-26), imported for 0.0.3.** It is the evidence for the TripoSR go/no-go in the [0.0.3 roadmap](../roadmap-0.0.3.md#spikes). The verdict below is the 0.0.2 recommendation, not the 0.0.3 decision; [Open for 0.0.3](#open-for-003) lists what still has to be answered first.

**Verdict: go for 0.0.3.** Both neural stages export to Core AI on the first try and match eager fp32. Single-image-to-mesh takes about 2.5 s on an M5 Max. The remaining work is ordinary engineering: preprocessing, marching cubes, USD packaging and qualification. The one real gap is that Core AI has no `grid_sample`. An explicit gather replaces it and matches upstream to 3.4e-6.

This is a feasibility spike, not product work. The code under `Spikes/triposr/` is throwaway and is not wired into `Package.swift`.

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
