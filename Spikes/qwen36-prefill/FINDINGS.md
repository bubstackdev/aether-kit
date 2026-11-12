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

# Spike 2: int4, up to the whole decoder (2026-11-10)

Weights now come from the pinned abliterated checkpoint (`huihui-ai/Huihui-Qwen3.6-27B-abliterated@27502c87`, bf16)
through `extract_hf_layers.py` (8-layer fp16 shards, 28 s and 11 GB peak for all 64 layers). Quantization is Forge's
shieldstral recipe: coreai-opt eager weight-only symmetric PTQ, every decoder Linear int4 in blocks of 32 along the
input axis (`--weight-bits 4`); norms, conv1d, A_log and dt_bias untouched.

| Layers | Bundle | Export (wall) | Export peak RSS | GPU `main` | GPU `prefill` (64 tokens) |
|---|---|---|---|---|---|
| 8 | 1.71 GB | 47 s | 15.8 GB | 6.3 ms/token | 26 ms (2,450 tok/s) |
| 16 | 3.43 GB | 2 min | 28.4 GB | | |
| 32 | 6.85 GB | 2 min | 50.4 GB (no new swap) | | |
| **64 (the whole decoder)** | **13.7 GB** | **6 min** | 50.5 GB RSS + ~35 GB in the compressor; swap 12 to 40 GB | **77 ms/token (13 tok/s)** | **170 ms (377 tok/s)** |

Accuracy: at 8 layers, int4 against the unquantized fp32 reference is 0.9964 worst row / 0.9990 last for the step
loop, and prefill equals the step loop (1.0). At 64 layers there is no fp32 reference (it would need ~100 GB); prefill
against the step loop is 0.999998 at all five chunk ends and the recurrent state 0.9999998, no NaN. A real-output
check (tokens, against MLX or HF) needs the head and is the next slice.

Runtime memory of the 64-layer bundle, after one call of each function (`footprint`): 13 GB mapped file (the weights,
clean) and 3.0 GB dirty (4.0 GB peak). The first load took 59 s and ~32 GB while MPSGraph compiled. So the decoder
needs ~17 GB at rest; with the vision tower and an int4 head about 19-20 GB, beside Krea's 31 GB on this 64 GB M5 Max.

For a describe (~3.7k prompt tokens with pictures, ~130 out): ~10 s to read the prompt and ~10 s to write, plus the
vision tower.

## Exporter memory, and what fixed most of it

- The zoo model was built in fp32 with its 248k-row embedding table, then cast: 24.5 GB to export 8 layers, 1.9 GB a
  layer. Building on the meta device and adopting the loaded tensors (`load_state_dict(assign=True)`, the table a
  1-row stand-in, never used here) brought it to 15.8 GB, 1.6 GB a layer, most of it the memory-mapped fp16 pages
  quantization reads (file-backed, droppable).
- After quantization a Linear keeps its int4 values in an int8 tensor (half the fp16 size) plus fp16 scales and int8
  zero points per block of 32; the fp16 original is emptied.
- The whole decoder in one process works on 64 GB but only by paging (swap peaked at 40 GB, the guard's limit). The
  production exporter should quantize layer by layer straight from the mmapped shards, or convert the decoder in
  segments, before the vision tower and head are added to the same export.

## Next

The vision tower (27 blocks, hidden 1152) and M-RoPE, the head, and a token-level check against HF on real pictures,
staged on Qwen3.5-9B (same tower) per the plan.

# Stage 3: pictures in, words out, on the 27B (2026-11-10)

The vision tower, M-RoPE and the head, end to end on the abliterated 27B, checked against an independent
implementation. transformers' float32 27B doesn't fit (108 GB), so the reference is mlx-vlm's own Qwen3.5 code on an
8-bit MLX conversion of the same checkpoint (`ref_mlx.py`), from identical inputs: `ref_vlm.py --prep-only` runs
transformers' processor and `get_rope_index` (a meta-device model, no weights) on a describer prompt with two Samara
reference pictures: 1,071 tokens, 768 of them image tokens (grids 48x24 and 48x40), M-RoPE positions up to 350, the
text after the images shifted by -720. MLX's own positions are identical to transformers'.

- **Vision tower** (`vlm.py export-vision`): Forge's Qwen3-VL `VisionTower` without DeepStack (Qwen3.5's tower has the
  same blocks, merger, bilinear position table and axial 2-D rotary). fp16, 2,048 patch rows, one static function.
  Against transformers' float32 tower on three real pictures: cosine 0.99997-0.99999, worst row 0.9968-0.9997;
  0.08 s a picture on the GPU once warm. (Against MLX's bf16 tower: 0.9987.)
- **Decoder** (`vlm.py export-decoder`): the spike's static hybrid trunk with Qwen3.5's interleaved M-RoPE (positions
  [3, S] in, the [11, 11, 10] interleave over the 32 rotary frequencies), the final norm and the untied head, every
  Linear int4 block 32 including the head: 14.4 GB, exported in 6.5 minutes (swap peaked at 42 GB). Host side for now:
  the embedding lookup with the image rows spliced in at `<|image_pad|>`, positions from `get_rope_index`.
- **Against MLX 8-bit, teacher-forced on its 48 greedy tokens**: argmax agreement 39/48, mean KL 0.109 (max 0.65), top-5
  overlap 0.85. Control, MLX's own 4-bit (affine, group 32) on the same tokens: 42/48, KL 0.077, 0.88. So the Core AI
  pipeline sits at 4-bit quantization noise, a little behind MLX's affine 4-bit; a wiring fault (positions, splice,
  head) would show as near-zero agreement. Symmetric int4 without a zero point is the likely gap; affine int4 or an
  int8 head are the knobs.
- **Free greedy** reads right: "Red textured bodysuit, knee-length, form-fitting, with black side panels and gold trim,
  deep V-neckline, long black sleeves, gold collar, black gloves, black boots with red accents, segmented armor on thighs
  and calves" (MLX 8-bit: "Red and black armored bodysuit, knee-length, textured fabric with gold trim, featuring
  segmented panels and rivets; deep V-neckline with gold accents, long sleeves with black gloves, ...").
- **Speed on the GPU**: prompt of 1,071 tokens in 7.0 s (150 ms per 64-token chunk; the remainder one token at a time),
  81 ms a generated token; decoder load 58 s (compile), vision load under 1 s.

Open: an exact (float) decoder check against transformers on Qwen3.5-9B (same tower and M-RoPE; downloading), the
int4 recipe (affine or an int8 head), and the bundle contract (embedding and splice in the graph, `vlm-static-inject@1`
with the hybrid's two functions).
