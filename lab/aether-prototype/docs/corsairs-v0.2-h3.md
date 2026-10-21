# Corsairs v0.2 boundary for MiniMax H3

Historical boundary record. The current checkout is `0.3.0-dev`: **Corsairs owns larger checked tensors and native runtime qualification. Aether owns all remaining H3 application/export work.** The implementation has advanced beyond the remaining-work list below; see [H3 integration](h3.md) and [validation](h3-validation.md). This record retains the original 1 GiB arithmetic without asserting the current development cap.

This record originally described the v0.2 development checkout and `/Users/zakkeown/Code/atelier/lab/minimax-h3-lab` on 2026-10-15. BF16 and transitive `Runtime::drain()` were included in **v0.2**, along with dynamic shapes, per-model reshape settings, native chaining, Metal interoperability and concurrent model/function loading. Those bounded H3 workloads needed no additional runtime capability. Publication and final frozen-source/package checks were pending at that point.

At that point Aether depended on the unpublished sibling `corsairs` 0.2.0 source. It did not qualify a published package.

## Satisfied execution requirements

Corsairs provides distinct two-byte `DType::BFloat16` storage and an owned `BFloat16` scalar. Aether preserves BF16 hidden states and modulation tables, float32 timestep/rotary inputs, and exact checked int64→int32 tag lowering. It never substitutes float16 for BF16.

Aether has independently reproduced the original 128-row live/cached block and all 50 block-0 schedule pairs, including retained native BF16 modulation chaining. Exact Rust-versus-Swift bytes/hashes govern these gates; the separate 2% CPU cross-backend tolerance cannot relax them. See [Aether's H3 validation](h3-validation.md).

Corsairs additionally retains exact acceptance of the synthetic 50-block backbone, both across a verified process checkpoint and through two serial 25-block graphs in **one Runtime**. Its [H3 acceptance record](../../corsairs/docs/v0.2-h3.md) pins the source/build and directly verifies the final BF16 bytes. Aether's own staged runner follows that proven owned-host-carry boundary. None of these checks is a full generation trajectory or repeated-residency benchmark.

Aether now uses `Runtime::drain()` after releasing stages and in Krea's balanced-encoder, denoiser-switch and unload paths. One drain waits for preceding work and transitive cleanup, including cleanup from abandoned or failed requests. Stage handles and unwanted tasks must be dropped first; retained handles/results stay alive. Later independent submissions do not extend the captured boundary. This replaces the previous repeated-flush workaround. It does not promise immediate eviction of Core AI caches, compiler pages or GPU allocations. See the [runtime contract](../../corsairs/docs/runtime-drain.md).

## Conditional capacity requirement

The public tensor cap at the time was **1 GiB**. The synthetic backbone tensor, BF16 `[1,19350,5376]`, is 208,051,200 bytes and fits that historical cap. No cap increase was needed for this gate.

A float32 RGB output for 124 frames at 1344×768 would be 1,535,901,696 bytes and exceed the cap. At 960×544 it would be 777,093,120 bytes and fit the byte limit; this does not establish peak-memory fit. A qualifying larger decoder export should determine whether Corsairs needs a checked larger limit or Forge should provide bounded decode chunks. Arbitrarily splitting temporal context can change the output. Aether's frame writer can consume bounded delivery, but that does not itself qualify a chunked decoder.

The longer supported profile also exceeds the cap before decode: 345 pixel frames map to 102 latent frames. At 1344×768 that is 102,816 target video rows, so BF16 `hidden [1,102816,5376]` occupies **1,105,477,632 bytes**, before text/audio/keyframe rows. This profile needs a larger checked tensor boundary or a qualified alternative export. Independent attention row batches are not equivalent. The initial 124-frame profile remains below this hidden-tensor limit. Full float32 RGB for 345 frames would occupy 4,273,274,880 bytes.

Keep the existing cap until an export and native acceptance establish the required boundary. Any increase needs shape/stride/overflow, owned/native download and Metal extent coverage.

## Remaining Aether/Forge work

At the time the lab had no complete H3 generation bundle. Its retained segments covered one synthetic 19,350-row pass through the original 50 blocks at one timestep pair. They did not define arbitrary packed generation inputs.

The application work identified then was:

1. A pinned FL2VA conditioner/vision export, processor, unnormalized layer-50 tap and text-refinement contract.
2. Actual packed text/keyframe/audio/video rows, tags and positions, input/output projections and global joint attention. Residual width is 5376; attention width is 56×128.
3. Source-precision joint latent updates and trajectory oracles. The captured base schedule has 50 paired evaluations, video/audio shifts 12/3, and no extra unconditional CFG evaluation.
4. Video/audio VAE normalization and decode exports preserving temporal context, plus separate encoded-keyframe qualification.
5. Complete latent and decoded sample acceptance before RGB8 conversion or lossy mux.

The full driver must release conditioning models before acquiring the denoiser, then release denoising models before decode. The demonstrated staged pass does not qualify conditioner co-residency or repeated generation on a 64 GiB host. Admission, application Metal kernels and resource policy belong to Aether; experimental low-bit/grouping/cache choices remain outside the generation baseline until qualified.

The initial partition is FL2VA snapshot `42ed227ee7df40d41602854ae760620d6eb651fe`, with Diffusers commit `7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941`. Ref2VA weights are absent. Modulation caches must include weights/adapters, exact schedules, conditioning noise, mode and dtype in their identity.
