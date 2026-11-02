# v0.0.6 spike: SD-Turbo and spatial latent diffusion

Worktree: `.worktrees/v006-sd-turbo`; branch: `spike/v0.0.6-sd-turbo`.
This is an experimental port for v0.0.6, outside the published qualified-model list.

## Selection

The checkpoint is [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo/tree/b261bac6fd2cf515557d5d0707481eafa0485ec2),
revision `b261bac6fd2cf515557d5d0707481eafa0485ec2`. Unlike Kit's Krea and FLUX.2 image
paths, it brings a convolutional U-Net with multi-resolution skip connections,
a single CLIP text encoder and an epsilon-prediction Euler scheduler. Its one-to-four
step distillation makes complete real-weight qualification practical. The upstream
[model card](https://huggingface.co/stabilityai/sd-turbo/blob/b261bac6fd2cf515557d5d0707481eafa0485ec2/README.md)
recommends 512×512 and disables classifier-free guidance and negative prompts.

## Architecture added

- `AetherDiffusion` has no Core AI dependency. `EpsilonDiffusionBackend` separates text
  encoding, spatial denoising and decoding. `DiffusionSession` owns the host iteration,
  checks concurrent operations, propagates cancellation and joins native work before
  cleanup. Optional `GPUAdmissionOwner` integrates with Kit's existing GPU coordination.
- `EpsilonSchedule` consumes explicit timesteps, sigmas and initial-noise scaling,
  validates their monotonicity and implements Euler input scaling and epsilon updates.
  This avoids confusing flow-matching velocity with diffusion epsilon, or applying
  FLUX packing to NCHW latents. Schedules are captured from the pinned Diffusers runtime.
- `AetherDiffusionCoreAI` implements strict stateless float32 stage contracts. It
  releases CLIP before opening the U-Net, keeps the U-Net through the step loop,
  and releases it before decoding. Every operation releases its models. New families
  can reuse these bindings when they share these contracts, or supply a backend with
  different conditioning while retaining the host scheduler loop.
- `AetherSDTurbo` supplies checkpoint identity, checksummed bundle validation, geometry,
  supported step counts, seeded noise, local CLIP byte BPE and a convenient session factory.
  CLIP's `</w>` BPE suffix is missing from the existing generic tokenizer implementation;
  the adapter handles it explicitly, including added tokens and zero padding.
- `Forge/diffusion/graphs.py` wraps upstream modules behind the shared stateless contracts.
  Model architecture stays in pinned Diffusers/Transformers implementations; the spike
  does not duplicate the U-Net in hand-written tensor operations.

The seam is deliberately bounded. SDXL's pooled/size conditioning, classifier-free
guidance, v-prediction, ancestral sampling and flow matching require explicit extensions.
Existing Krea and FLUX.2 pipelines are unchanged. This is not a promise that their
orchestration can be replaced by this seam unchanged.

## Use

With the qualified local bundle in this worktree:

```swift
import AetherSDTurbo
import Foundation

let bundle = try SDTurboBundle(at: URL(filePath: "/absolute/worktree/.local/sd-turbo/ratification/qualification/bundle"))
let tokenizer = try SDTurboTokenizer.open(bundle: bundle)
let session = try bundle.makeSession()
let image = try await session.generate(
    tokenIDs: tokenizer.prepare("A blue ceramic teapot on a wooden table."),
    noise: bundle.noise(seed: 42), schedule: bundle.schedule(steps: 1))
try image.savePNG(to: URL(filePath: "/absolute/output.png"))
await session.unload()
```

Native execution requires macOS 27 / iOS 27; only the macOS host is qualified here.
The portable scheduling and tokenizer APIs keep the package's existing deployment floor.
Bundle verification streams SHA-256 over the model files; open the bundle once and reuse it.
The seeded generator is Kit's SplitMix64/Box–Muller generator and does not reproduce
PyTorch's seed stream. Qualification uses the exact reference noise instead.

## Reproduction

```sh
uv sync --frozen --project Forge
Forge/.venv/bin/python -I Spikes/sd-turbo/fetch.py /absolute/checkpoint
sh Spikes/sd-turbo/qualify.sh /absolute/checkpoint /absolute/new-output
sh Scripts/check-cpu.sh
```

The fetcher downloads only the pinned source files; the exporter verifies every
source against `checkpoint-sha256.json` before loading. Conversion and independent
Diffusers references run on CPU. The native test takes Kit's shared GPU lease.
Weights, compiled assets, binary reference tensors and PNGs stay in ignored `.local/`
(or an explicit output directory), never in Git. The default geometry is fixed 512²;
`prepare.py --size` can export a different fixed square from 64 through 512 in
multiples of 64, but only measured shapes should be called qualified.

## Qualification and limits

**Result: ratified locally for merging as an experimental v0.0.6 port.**
The clean `qualify.sh` workflow passes from a fresh export directory, including
exact repeat generation after stage release/reopen. The retained bundle is at
`.local/sd-turbo/ratification/qualification/bundle`; reference and native PNGs are alongside it.
The [ratification record](../evidence/sd-turbo-v006-ratification-2026-11-02.json) pins source,
exported model and reference hashes. Host: Apple M5 Max, 64 GB, macOS 27.2,
Swift 6.4, Torch 2.11.0, Core AI Torch 0.4.2, Diffusers 0.37.1 and Transformers 4.57.3.
Native execution uses `preferGPU`; per-operator placement was not instrumented.

| Check | Result |
| --- | --- |
| Local tokenizer vs upstream | All eight edge cases and all four generation prompts exact |
| CLIP native vs independent reference | Minimum cosine > 0.99999999999 |
| U-Net native vs reference | All ten teacher-forced steps; minimum cosine > 0.99999999999 |
| VAE native vs reference | Maximum pixel RMSE < 0.000001 |
| Whole native pipeline, 512² | Steps 1/2/3/4: RMSE 0.00001087 / 0.00001103 / 0.00001063 / 0.00002519 |
| Repeat after native release/reopen | Exact float32 pixels |
| Full weight-free Release suite | 665 executed, 55 intentionally skipped, 720 registered; all pass |
| Package deployment floor | All products build at macOS 14 / iOS 17; both consumers weak-link CoreAI |
| Native lifecycle | Cancellation after step 0, recovery, exact repeat and terminal close pass |
| CPU lifecycle and rejection tests | Failure/recovery, overlap, in-flight cancellation, terminal close, corrupt bundle and geometry rejection pass |

The native PNGs were visually inspected. Timings came from a shared host and
are not benchmark results. Native cancellation between submitted denoising calls is qualified. Cancellation inside a
submitted call and memory-pressure stress remain unqualified; controlled backend tests verify
joined release for encoder, denoiser and decoder calls.

The qualification records are generated by `prepare.py` and `sdTurboNativeParity`.
They distinguish teacher-forced component comparisons from the complete native trajectory.
Tokenizer comparisons are exact, including empty strings, punctuation, Unicode,
combining marks, whitespace, truncation and embedded special tokens. Numerical bars
are text cosine ≥ 0.99999, epsilon cosine ≥ 0.9999 at each reference step,
decoder RMSE ≤ 0.01 and full-image RMSE ≤ 0.03 on the model's [-1,1] scale.
These bars were declared before the native run and are not speed claims.

Float32 weights and activations are an intentional reference baseline (~4.7 GiB of
exported assets). Quantization, device-resident latents, multiple native geometries,
managed installation/catalog integration, memory pressure handling and iOS qualification
remain follow-ups. The adapter matches the pinned fast tokenizer; the slow Python
CLIP tokenizer's optional `ftfy` text repair is outside this contract.

The checkpoint uses the Stability AI Community License, separate from Aether Kit's
Apache-2.0 code license. The exporter copies its pinned license into the local bundle.
No model assets are published as part of this spike.

To regenerate the compact record after the full CPU suite and native workflow:

```sh
Forge/.venv/bin/python -I Spikes/sd-turbo/record.py \
  --qualification /absolute/qualification \
  --cpu-log /absolute/full-cpu.log --native-log /absolute/qualify.log \
  --floor-log /absolute/floor-macos.log --ios-floor-log /absolute/floor-ios.log \
  --api-log /absolute/api.log --output /absolute/evidence.json
```

## Merge ratification (2026-11-02)

The review found and resolved the following before opening the merge candidate:

- Native backend unload now joins every active encoder, denoiser and decoder stage,
  including loading, and serializes concurrent release. Stage calls arriving during
  release fail with `sessionBusy`. Terminal session close also joins concurrent callers.
- Bundle verification rejects undeclared files and symlinks, normalizes Foundation's
  `/var` and `/private/var` aliases, and uses AetherAssets' bounded, cancellation-aware hashing.
- Qualification now covers all four supported step counts and real native cancellation/recovery.
  Weight-free tests cover GPU admission lifetime, queued withdrawal and backend release ordering.
- The independent SDK consumer type-checks the documented API. The floor consumer links
  the image adapter at macOS 14 and iOS 17, with CoreAI weak-linked on both platforms.
- All four frozen API modules match their committed baselines. No frozen API changed.

This ratifies merging the experimental port, not promoting it to a production support tier.
The repository's required `CI passed` check on an up-to-date PR remains the merge gate;
this document does not substitute local checks for that gate. Model assets remain local
and separately licensed. The earlier [initial qualification](../evidence/sd-turbo-v006-2026-11-02.json)
is retained as historical evidence; the ratification record supersedes its two-case coverage.
