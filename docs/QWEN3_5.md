# Qwen3.5 / Qwen3.6: the hybrid VLM export

Status (2026-11-10): `Forge/qwen3_5/` exports any Qwen3_5ForConditionalGeneration checkpoint as a
`vlm-static-inject@1` bundle; the Generator takes images on its recurrent decoder; the Qwen3.6-27B abliterated export
passes `Qwen35GateTests`. The first user is Atelier's outfit describer (owner's choice, 2026-11-09).

## The model

A ViT tower (Qwen3-VL's: patch 16, temporal patch 2, 2x2 merge, bilinear-resampled position table, axial 2-D rotary;
no DeepStack) and a hybrid decoder: three Gated DeltaNet (linear attention, a causal conv and a delta-rule state) to
each gated full-attention layer, untied head, interleaved 3-axis M-RoPE over a quarter of each head. 27B: 64 layers
(48 DeltaNet, 16 attention), hidden 5,120, vocabulary 248,320, tower 27 blocks.

## The bundle

| Graph | Contract |
|---|---|
| `vision` (`vision-tower@1`) | Forge/qwen3vl's: `patches` f32 [2048, 1536] + `grid` i32 [2] → `image_embeds` f16 [512, H]; fp16 weights, fp32 compute |
| `decoder` | `decoder-lm@1`'s running contract, one function: `input_ids` [1, S] (S 1..32), `position_ids` [1, T]; static `image_embeds` [N, H], `image_positions` [N, 3], `rope_shift_start`/`amount` [M] (`qwen3vl-mrope`); states `keyCache`/`valueCache` (sequence) and `convState`/`recState` (recurrent, fp32 delta-rule state); logits f32, last row; `prefillChunk` 32 |

Decoder Linears int4 (blocks of 32, symmetric, coreai-opt), the head int8 per channel; embedding fp16. 27B: 18.5 GB.

`--embedding-bits 8` is an opt-in memory variant: symmetric int8 per vocabulary row with float32 scales. The graph
gathers packed rows and their scales before dequantizing, so the full table stays packed. Quantization runs in
1,024-row pieces to avoid a full float32 copy during export. `--embedding-bits 16` remains the default.

Why one function of up to 32 tokens rather than a one-token function and a wider prefill function sharing the states:
Core AI keeps a copy of the weights for every function that runs (27B: 16.6 GB of Metal after the first one-token
call, 33.0 GB once a 64-token prefill function had also run), and calls wider than 32 tokens on a reshaping graph
retain Metal (docs/VLM.md, G5).

The DeltaNet scan runs in float32, in 16-token pieces: the chunked form's triangular inverse (a fixed product of
doublings) is exact in arithmetic but wrong in float32 at 64 tokens (27B layer 0: a state of 2.1e6 against 32). A graph
can't loop a dynamic number of times, so a call's tokens are written into 33-row buffers whose padding rows are made
exactly inert (decay and beta 0) by a mask, not by the buffer's initial zeros: the converted graph fills the buffer in
place, and an unmasked export read leftover memory there (27B: NaN logits on the third one-token call). 33 rather
than 32 because torch.export refuses a call that fills the buffer exactly.

Two more Core AI rules the graphs follow (both from the 2026-11-10 spike): state writes are slice assignments at a
slice layer index (an integer-index `copy_` into the conv state was dropped), and nothing writes through a matmul
with a one-row operand (the value cache landed in the wrong rows).

## The runtime

The engine needed nothing new. The Generator refused images on a model with recurrent state; it now reuses only the
live state there (a prompt that extends what was fed, with every bound image still the plan's at the same place),
otherwise it starts from an empty state, and keeps no snapshots while images are bound (`Generator.prepareRecurrent`).
`VisionGeneratorTests` covers it on `FakeVLM`'s new recurrent state.

KV growth now allocates and copies only sequence states, keeping recurrent buffers in place. At 27B this avoids
146.8 MiB of temporary allocation and copying on each growth (144 MiB delta-rule state, 2.8 MiB conv state).
`DecoderEngineTests.kvGrowthKeepsRecurrentStorageAndValues` checks storage reuse and output after growth.

## Measured (27B abliterated, M5 Max 64 GB, macOS 26B123)

| | |
|---|---|
| Export | 3.6 minutes, 55 GB peak RSS (swap to 20 GB): layers are read from the shards, quantized and released one at a time |
| Load | 18 s; Metal 17.1 GB |
| Two-picture describer prompt (1,071 tokens) | bind 1.1 s, prompt 2.6 s, 55 ms a generated token |
| Metal over 7 mixed requests | 19.8 GB (int4 head; 20.0 GB with the int8 head), 2 MiB of growth after the first, no reloads |
| Against mlx-vlm 8-bit, teacher-forced 48 steps | int8 head: argmax 40/48, the reference's token in our top five 48/48, KL 0.081; int4 head: 38/48, KL 0.103 (MLX's own 4-bit: 42/48, KL 0.077) |

### Int8 embedding experiment (2026-11-11)

Compared against the bundle Atelier currently resolves through `~/Atelier/Models/qwen3.6-27b-abliterated`, keeping
the decoder int4 and head int8. Every GPU run held `scripts/gpu-lease`. Same M5 Max 64 GB, macOS 26B123;
`Qwen35GateTests`, capacity 2048, two pictures, 1,071 prompt tokens and 48 teacher-forced steps.

| Metric | Atelier baseline (fp16 embedding) | Int8 embedding |
|---|---:|---:|
| Bundle bytes (export report) | 18,466,061,600 | 17,195,657,178 |
| Metal after load | 17,603 MiB | 16,392 MiB |
| Metal after teacher-forced prompt/steps (first / repeat) | 19,970 / 19,970 MiB | 18,757 / 18,753 MiB |
| Reference argmax agreement | 40/48 | 41/48 |
| Reference token in top five | 48/48 | 48/48 |
| KL mean | 0.0813 | 0.0818 |
| Prefill seconds (first / repeat) | 2.57 / 4.87 | 3.37 / 4.08 |
| Median decode milliseconds (first / repeat) | 49.3 / 50.7 | 49.7 / 50.7 |
| Metal across seven mixed requests | 20,334–20,346 MiB | 18,849–19,117 MiB |
| Reloads across those requests | 0 | 0 |

The table saves 1.27 GB on disk and about 1.2 GiB of Metal. Neither variant accumulated memory over seven requests.
Timing varied enough that these two runs do not establish a speed improvement or regression. The accuracy gate is
one fixture, not a broad quality evaluation; greedy wording changes. The fp16 embedding remains the default.
`MTLDevice.currentAllocatedSize` measures Metal allocation, not process RSS.

Export a candidate with the usual exporter command plus `--embedding-bits 8`; run the existing gate with
`AETHER_QWEN35_BUNDLE=<candidate>` and `AETHER_QWEN35_REFERENCE=<same reference>`, adding
`AETHER_QWEN35_RETENTION=1` for mixed requests and call-size NaN checks, under `scripts/gpu-lease`.
The tested candidate is `/private/tmp/qwen35-memory-int8-bundle` (a temporary experiment, not installed into Atelier).
The CPU graph/quantization checks, GPU float16 selftest with `--embedding-bits 8`, all three 27B gates, and
`scripts/ci.sh` passed. Machine-readable measurements: [qwen35-memory-2026-11-11.json](evidence/qwen35-memory-2026-11-11.json).

## Checks

- `Forge/qwen3_5/test_qwen3_5_graphs.py` (CPU, a tiny random model): positions, tower and decoder against transformers
  to 2e-7, fed as the runtime feeds them.
- `Forge/qwen3_5/selftest_qwen3_5.py` (Core AI GPU, the tiny model): converted against eager, every call and state.
- `Qwen35GateTests` (Swift, opt-in): a bundle against a reference from the same checkpoint (`AETHER_QWEN35_*`), with
  Metal across requests and NaN checks by call size behind `AETHER_QWEN35_RETENTION=1`.

## Open

- The int4 recipe: symmetric without zero points; the int8 head closed most of the gap to MLX's affine 4-bit, affine
  int4 for the decoder Linears is the next knob.
- The int8 embedding variant needs broader picture/prompt quality checks before becoming the default.
- A float reference gate on a size transformers runs on the CPU (Qwen3.5-9B).
- Image-aware snapshots for a recurrent model (prefix reuse across different picture sets).
