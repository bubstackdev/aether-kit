# Qwen3.6-27B hybrid prefill spike: findings (2026-11-10)

Spike 1 of the 27B describer plan: can one Core AI bundle carry a one-token `main` and a 64-token `prefill` that
share Qwen3.6's four hybrid states (KV cache, conv state, recurrent state), agree with an exact reference, and run
fast? **Yes**, once four problems below are worked around.

Setup: the first 4 real layers of Qwen3.6-27B (3 GatedDeltaNet + 1 gated full attention, real dimensions), fp16
weights and KV, fp32 recurrent state, a 192-token describer prompt, macOS 26B123, coreai-torch 0.4.2, the GPU through
the Python runtime with fixed input shapes. Reference: the zoo's one-token loop in fp32 torch on the CPU.

| | Result |
|---|---|
| `main` (S=1), 192 chained tokens | worst row cosine 0.99999992, last 0.99999999 |
| `prefill` (S=64) chunk ends 63/127/191 | 1.0 / 1.0 / 1.0; final recurrent state 0.9999999 |
| `main` speed | 7.7 ms/token (4 layers, fp16 weights) |
| `prefill` speed | 12.2 ms per 64-token chunk, about 5,250 tok/s (4 layers) |
| Bundle | 3.04 GB for 4 layers; both functions share the weights; export peak RSS 18.5 GB |

Scaled to 64 layers (linear, fp16): about 330 tok/s prefill (a 3.7k-token, several-picture prompt in about 11 s) and
about 120 ms/token decode before int4.

## What broke, and the workaround in `spike.py`

1. **The in-graph chunk scan diverges at 64-token blocks, even in fp32.** The zoo's `_gated_delta_chunk` inverts
   (I+M) by repeated squaring; on layer 0 (|v| up to 35, near-identical keys) the recurrent state reached 2.1e6
   against 32 exact after 64 tokens. 8 and 16 are exact (6e-7 relative), 32 is 3e-4. In fp16 the overflow showed as
   NaN at the third chunk (GPU, and eager torch alike). Workaround: the `prefill` function still takes 64 tokens,
   so the weight-heavy projections are shared over the whole block, but the scan runs as four 16-token pieces in
   sequence inside the graph (`--subchunk 16`). No measurable speed cost (12-14 ms per chunk before and after).
2. **`write @ v` with a one-row query lowers in the wrong layout.** The value written for cache row 5 of 4 KV
   heads landed in rows 20-23 (row x 4 + head); keys, which pass a norm first, were right. Workaround at S=1: a
   broadcast multiply (`write` as [1,1,C,1] times `v`) instead of the K=1 matmul.
3. **A `copy_` into the conv state is never applied.** The optimized IR writes `convState` correctly (checked:
   `write_handle` from a `where` chain with the right per-layer masks and shift matrix), yet the runtime returned it
   unchanged from both functions, in either layout ([L,1,conv_dim,3] or taps-major [L,1,3,conv_dim]). The KV and
   recurrent states, written the same way, were fine. Workaround: write it through the zoo's `SSMState.update_states`
   (`mutable_slice_update` with explicit begin/end), the path the zoo's own stateful exports use. Not reduced
   further; worth a minimal repro for Apple.
4. **The Python runtime re-specializes on every new input shape** and MPSGraph writes temp files for each (2026-11-09:
   369 GB in 20 minutes). Every input here has a fixed shape (`positions` [1,S], a one-hot `write` [C,S], an additive
   `mask` [1,1,S,C]), and `run` stops itself past 20 GB of MPSGraph temp or a 30 GB disk drop.

Not problems: the GPU's numerics (once 1-3 were fixed, GPU matches the fp32 reference to 1e-7), fp32 recurrent state
inside an fp16 graph, and two functions sharing states in one bundle on this macOS.

## How the bugs were isolated (for the next hybrid model)

- Eager `StaticCore` in fp32 and fp16 on the CPU against the reference: separates the graph's maths from Core AI.
- One `main` step from the exact eager states vs chained steps: separates the step's compute from its state writes
  (the step itself was 0.996-1.0 throughout; chaining fell to 0.21).
- From zero states, one call per function and a nonzero count per state: found the conv write missing.
- `str(program)` before and after `optimize()` (17 GB of text at 4 layers, weights inline: grep, then delete).

## Next

Spike 2: an 8-layer int4 slice through Forge's quantization (exporter peak memory, int4 speed, size), then the 9B
staging model with the vision tower and M-RoPE.
