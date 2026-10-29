# Spike: H3 fused attention (go/no-go)

Status: done 2026-10-29. **No re-export.** H3's qualified body graph already uses Core AI's fused attention, and Core AI
runs it fused on the GPU. The roadmap item closes, and H3's profile identity and qualification evidence stay as they are.
[Record](../../Spikes/h3-attention/results/probe-2026-10-29.json).

## Question

"H3 fused attention" was deferred from 0.0.2 and again from 0.0.3. Re-exporting the body graph resets H3's qualification
evidence, and nobody had checked whether Core AI fuses attention on the GPU
([0.0.3 roadmap](../roadmap-0.0.3.md)). Atelier's larger canvas and last-frame conditioning build on whichever body graph
results, so the answer decides their order of work.

## What the body already does

The exported block is Diffusers' `MiniMaxH3TransformerBlock` (`Scripts/export-h3-profile.py`, `Scripts/h3_export_lib.py`).
Its attention processor calls `F.scaled_dot_product_attention` with no mask, `is_causal=False` and the default scale, on
`[1, 56, R, 128]` bf16 queries, keys and values after the q/k RMSNorm and rotary. R is dynamic in 64…40960 (19,333 at
960×544×124) and there is no chunking. coreai-torch exports that call as a private composite: the qualified
`body.aimodel` carries `scaled_dot_product_attention` / `sdpa_maskless_…`, with a `batch_matmul` + `softmax` body as its
decomposition. So the graph was never unfused. The open question was what the GPU does with the composite.

## Probe

`Spikes/h3-attention/h3_attention_probe.py` exports attention alone at the body's shape, weight-free, with R dynamic like the
body, three ways:
- **`sdpa_bf16`**, the body's form. It exports to the same `sdpa_maskless` composite.
- **`explicit_bf16`**: matmul, softmax and matmul, which is what an unfused graph would be. No SDPA composite in its
  bytecode.
- **`sdpa_fp16`**, to test whether the fused path prefers fp16.

`h3-attention-probe.swift` runs each asset on the GPU with `expectFrequentReshapes`, as the body runs. At each R it
records the first call, the warm median of 5, Metal's allocated size, and 64 query rows of 5 heads. Those rows are scored
against a float64 reference built from the same hashed inputs, which both sides compute exactly. One GPU-lock run took
38 s, ending with a repeat of `sdpa_bf16` to check for drift. A watchdog guarded memory; the lowest
`memorystatus_level` was 31, during `explicit_bf16`.

## Results (M5 Max, macOS 27.2)

| R | sdpa_bf16 | sdpa_fp16 | explicit_bf16 | Metal allocated: sdpa / explicit |
| --- | --- | --- | --- | --- |
| 4,096 | 8.57 ms, 56.1 TFLOPS | 9.37 ms | 23.66 ms | 224.5 MiB / 5,826.7 MiB |
| 9,728 | 51.93 ms, 52.2 TFLOPS | 52.99 ms | 762.00 ms | 532.5 MiB / 36,990.7 MiB |
| 19,333 | 224.18 ms, 47.8 TFLOPS | 250.48 ms | not run | 1,058.0 MiB |
| 33,023 | 714.39 ms, 43.8 TFLOPS | 803.98 ms | not run | 1,806.7 MiB |

TFLOPS counts the two matmuls, 4 · 56 · R² · 128 FLOPs, over the median.

- **It's fused.** With SDPA, Metal holds the probe's own q, k, v and output buffers and less than 1 MiB more, at every R.
  The 56 · R² bf16 score tensor is never materialized: it would be 39.0 GiB at 19,333 rows and 113.7 GiB at 33,023.
  The explicit graph's Metal allocation reached 5.7 GiB at 4,096 rows and 36.1 GiB at 9,728. SDPA was 2.8× faster at 4,096 and 14.7× faster at 9,728.
- **fp16 isn't faster.** It was 12–13% slower at 19,333 and 33,023 rows, and 2–9% slower at the smaller row counts.
- **Accurate and repeatable.** SDPA bf16 is within 0.0015 relative L2 of float64 at every R, against H3's 0.02 bound.
  The repeat pass matched its accuracy exactly, and its times within 6%.
- **Explicit bf16 attention is wrong on the GPU at 9,728 rows:** 0.676 relative L2, against 0.003 at 4,096. It wasn't
  investigated further: bf16 softmax over 9,728 keys, or a Core AI defect. The lesson for exporters is not to hand-write
  long attention in bf16, and to keep `F.scaled_dot_product_attention`.

## What it means

- **No re-export.** No variant beat the body's form by the 15% the spike required. H3's profile identity, its
  trajectory, memory and lifecycle gates, the media audit and the quiet timing all stand.
- **Attention's share of a block.** At 960×544×124 (19,333 rows), attention is 224 ms of the block's roughly 680 ms of
  GPU work. The 680 ms comes from the busy-host [speed profile](../evidence/h3-speed-profile-2026-10-26.json), so the
  share is about a third. The rest is the projections and the feed-forward network. A faster step would have to come from
  there, or from fewer rows.
- **For Atelier's 1280×704 canvas (33,023 rows):** attention alone is about 0.71 s per block, roughly 36 s per 50-block
  step. Throughput falls only from 47.8 to 43.8 TFLOPS as R grows, so the scaling is close to R².

## Reproduce

```sh
uv sync --frozen --project Forge/h3-oracle
for v in sdpa_bf16 explicit_bf16 sdpa_fp16; do
  Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py export $v --output <dir>/$v.aimodel
done
xcrun swiftc -O -swift-version 5 -o .build/h3-attention-probe Spikes/h3-attention/h3-attention-probe.swift -framework CoreAI -framework Metal
~/.cache/aether-kit/with-gpu-lock.sh h3-attention .build/h3-attention-probe <dir>/sdpa_bf16.aimodel bf16 sdpa_bf16 <results> 4096,9728,19333,33023
Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py compare <results>
```

Each export takes about 450 MB of memory on the CPU. `explicit_bf16` at 9,728 rows holds about 36 GiB of Metal memory, so
run it only with a memory watchdog.
