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
- **`explicit_fp16`**, added in the follow-up below.

`h3-attention-probe.swift` runs each asset on the GPU with `expectFrequentReshapes`, as the body runs. At each R it
records the first call, the warm median of 5, Metal's allocated size, and 64 query rows of 5 heads. Those rows are scored
against a float64 reference built from the same hashed inputs, which both sides compute exactly. One GPU-lock run took
38 s, ending with a repeat of `sdpa_bf16` to check for drift. A watchdog guarded memory; the lowest
`memorystatus_level` was 31, during `explicit_bf16`. A follow-up run of 80 s sampled every head of the explicit graphs,
each case in a fresh process.

## Results (M5 Max, macOS 27.2)

| R | sdpa_bf16 | sdpa_fp16 | explicit_bf16 | Metal allocated: sdpa / explicit |
| --- | --- | --- | --- | --- |
| 4,096 | 8.57 ms, 56.1 TFLOPS | 9.37 ms | 23.66 ms | 224.5 MiB / 5,826.7 MiB |
| 9,728 | 51.93 ms, 52.2 TFLOPS | 52.99 ms | 762.00 ms (wrong output) | 532.5 MiB / 36,990.7 MiB |
| 19,333 | 224.18 ms, 47.8 TFLOPS | 250.48 ms | not run | 1,058.0 MiB |
| 33,023 | 714.39 ms, 43.8 TFLOPS | 803.98 ms | not run | 1,806.7 MiB |

TFLOPS counts the two matmuls, 4 · 56 · R² · 128 FLOPs, over the median. The explicit graph's allocation at 9,728
includes about 5.5 GiB still held from its 4,096 run in the same process.

- **It's fused.** After its calls, SDPA leaves Metal holding the probe's own q, k, v and output buffers and less than
  1 MiB more, at every R. The explicit graph keeps its intermediates allocated: in a fresh process, its 9,728 call grew
  Metal by 30.1 GiB, about three times its 9.9 GiB bf16 score tensor. The counter is read between calls, so it can't rule
  out scratch used and freed inside a call, for example per head. But the full 56 · R² bf16 score tensor would be
  39.0 GiB at 19,333 rows and 113.7 GiB at 33,023, which can't fit on this 64 GiB Mac. Timing agrees: at 4,096 rows,
  the clean comparison, SDPA was 2.8× faster. At 9,728 the explicit graph was 14.7× slower, but that run held 36 GiB
  under memory pressure, varied 24% between rounds, and gave wrong output (below), so it isn't a fair speed figure.
- **fp16 isn't faster.** It was 11.7% and 12.5% slower at 19,333 and 33,023 rows, and 2–9% slower at the smaller row
  counts.
- **Accurate and repeatable.** SDPA bf16 is within 0.0015 relative L2 of float64 at every R, against H3's 0.02 bound.
  The repeat pass matched its accuracy exactly, and its times within 6%.
- **Explicit attention is wrong past 2^32 score elements: a probable Core AI defect.** At 9,728 rows the score tensor has
  5,299,503,104 elements.
  - In the follow-up, with every head sampled, exactly heads 46–55 are wrong (relative L2 about 0.64 overall), in bf16
    and fp16 alike. Those are the heads whose first score element lies past 2^32 (2^32 / 9,728² = 45.4).
  - Every other head is within 0.0034. Head 45 straddles the boundary and is correct, which suggests a 32-bit per-head
    offset rather than a per-element one.
  - At 8,704 rows, where the whole tensor (4,242,538,496 elements) is under 2^32, all 56 heads are correct.
  - PyTorch's own bf16 explicit attention is 0.0045 from float64 at 9,728 rows, so precision isn't the cause.
  - This belongs with the roadmap's exporter-safety checks, and is a Feedback candidate. SDPA doesn't build the tensor
    and isn't affected: keep `F.scaled_dot_product_attention`.

## What it means

- **No re-export.** No variant beat the body's form by the 15% the spike required. H3's profile identity, its
  trajectory, memory and lifecycle gates, the media audit and the quiet timing all stand.
- **Attention's share of a block.** At 960×544×124 (19,333 rows), attention alone is 224 ms. The busy-host
  [speed profile](../evidence/h3-speed-profile-2026-10-26.json) gives about 680 ms of GPU work per block, and 682–787 ms
  per pass, so attention is 28–33% of a block. The probe leaves out the body's transposes around SDPA. The rest is the
  projections and the feed-forward network. A faster step would have to come from there, or from fewer rows.
- **For Atelier's 1280×704 canvas (33,023 rows):** attention alone is about 0.71 s per block, roughly 36 s per 50-block
  step. Throughput falls only from 47.8 to 43.8 TFLOPS as R grows, so the scaling is close to R².

## Reproduce

```sh
uv sync --frozen --project Forge/h3-oracle
for v in sdpa_bf16 explicit_bf16 sdpa_fp16 explicit_fp16; do
  Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py export $v --output <dir>/$v.aimodel
done
xcrun swiftc -O -swift-version 5 -o .build/h3-attention-probe Spikes/h3-attention/h3-attention-probe.swift -framework CoreAI -framework Metal
~/.cache/aether-kit/with-gpu-lock.sh h3-attention .build/h3-attention-probe <dir>/sdpa_bf16.aimodel bf16 sdpa_bf16 <results> 4096,9728,19333,33023
# the follow-up: every head, one case per process
~/.cache/aether-kit/with-gpu-lock.sh h3-attention .build/h3-attention-probe <dir>/explicit_bf16.aimodel bf16 explicit_bf16 <results> 9728 5 all
Forge/h3-oracle/.venv/bin/python -I -B Spikes/h3-attention/h3_attention_probe.py compare <results>
```

Each export peaks at 448–456 MB resident on the CPU. The explicit graph at 9,728 rows holds about 30 GiB of Metal memory,
so run it only with a memory watchdog.
