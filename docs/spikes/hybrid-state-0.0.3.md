# Spike: hybrid recurrent-state language model, 0.0.3 go/no-go (Granite 4.0-H 350M)

**Status: 0.0.3 go/no-go result (runs on 2026-10-27 and 2026-10-28).** Its input is the 0.0.2 feasibility report, [hybrid-state-2026-10.md](hybrid-state-2026-10.md), which stays as written. This spike answers that report's open items for the [0.0.3 roadmap](../roadmap-0.0.3.md#spikes):
- fit with the bucketed contract;
- an exporter inside AetherKit with an independent reference;
- chunked prefill and long-context state;
- qualification depth;
- quiet-host timing;
- the `staticq` compile message.

It is not product work: Forge gains the exporter, but Kit's runtime is unchanged.

**Verdict: go, on a static-width step contract (S1 or S2 below).** As exported in this spike, the dynamic-q contracts that mirror today's bucketed language contract failed on the GPU:
- with one function (C1), the GPU aborts at q = 1;
- with a separate decode function (C2), the dynamic-q `main` doesn't specialize on the GPU at full depth.

**The cause is our export, not Core AI** (follow-up bug hunt, 2026-10-28). `main` carries the same dynamic q on two inputs, `input_ids` [1, q] and `cache_positions` [1, q]. torch knows the two are equal, but coreai-torch emits two independent dynamic dimensions, so every q-dependent shape becomes a broadcast between two unrelated symbols, nesting one level per layer.
- **Fix:** pass the chunk's start position as a [1, 1] scalar and build the positions in the graph.
- **Result:** both failures go away (see [Contracts tried](#contracts-tried)).
- **Why the verdict stands:** that start-scalar dynamic contract is measured but not parity-qualified, so the 0.0.3 verdict stays on the static-width contract, which is. Re-qualifying the dynamic one is planned for 0.0.4.

Both static-width layouts open in seconds and match the float32 reference. On a quiet M5 Max:

| | S1 (`main` at width 64 + `decode` at q = 1) | S2 (`main` at width 64 only) |
|---|---|---|
| Wired weights | 1,446 MiB (two copies) | 797 MiB (one copy) |
| Decode, KV capacity 8,448 | 163–166 tok/s | 126–129 tok/s |
| Decode, KV capacity 1,280 | 182 tok/s | 134 tok/s |
| Time to first token at 2,048 tokens | 248–255 ms | 247–252 ms |
| Teacher-forced rows (3 chat cases) | min cosine 0.99998, 0 argmax mismatches | the same |
| 128-token greedy | identical on all 3 cases | identical on all 3 cases |
| Drift to 4,096 and 8,192 tokens | cosine ≥ 0.999992, 16/16 argmax at each length | the same |

S2 is the one-weight-copy choice. S1 costs another 0.63 GiB wired (649 MiB) for 26–32% faster decode at capacity 8,448 and 36% at 1,280.

The GPU failures of the dynamic contracts, and the cpu_only defects found on the way, are listed in [the defects section](#core-ai-defects-and-apple-feedback-candidates) with what remains for Apple. Every workaround in the code is labelled a workaround, not a fix.

## Setup

- **Device:** Apple M5 Max, 64 GB, macOS 27.0.1 (26A434). torch 2.11.0, coreai-torch 0.4.2, coreai-core 1.0.0b2 (Forge's locked environment). Every GPU job ran under `with-gpu-lock.sh`.
- **Host load:** the qualifying runs (S1, S2 and the baseline, 2026-10-28 10:59–11:03) were on a quiet host, 1-minute load average 1.3–2.2 (`results/hold-windows-and-load.txt`). The earlier probes ran while the host was CPU-saturated (load 60–125, a transcode). Their only timing claims are whether a graph opened within a cap; the C2 timeout was repeated at load 5.
- **Checkpoint:** `ibm-granite/granite-4.0-h-350m` at revision `3b17b717b8f2f5d305b0a92c1491e239aeda19c8`, Apache-2.0, ungated. It was read in place, read-only, from an existing local cache; nothing was downloaded. SHA-256:

  | File | SHA-256 |
  |---|---|
  | `model.safetensors` | `5e8e1b9957216467f080a186b13ef8886f23a6da273f55b8287f41de54faafc7` |
  | `config.json` | `c11be93f2e5f83096042b19c3df3b0d982819b63c5378e0b6a82c4fdbe017fac` |
  | `tokenizer.json` | `e2bad66439538cb4d5a7580680932432ed9ece9d3b8577e675512bdf11599253` |
  | `tokenizer_config.json` | `a5ec5daab12ba090a90f3dd169c8f9c275557013a87b9c1258dc7cb497a35c86` |
  | `chat_template.jinja` | `9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce` |
  | `generation_config.json` | `7c04cb9d2ba771f7528fba5a7104999cdaf7566d02b5fbd58472829f62716177` |
  | `special_tokens_map.json` | `c08676c49fd7969a3130f72be6d4bf34da66aa484a6e21dffe359893a1bd5f2e` |
  | `vocab.json` | `8af71076de8b0b626eed0f4c984faf0a7c062479164b2a31308a948524d4f69c` |
  | `merges.txt` | `b6fe424e334903f7fb84d3a106d9730455f4744b9fe3c21ee136d97a00e72502` |

- **Model:**
  - 32 layers: 28 Mamba-2 and 4 NoPE attention, at layers 10, 13, 17 and 27;
  - hidden 768, vocab 100,352, tied head;
  - μP multipliers: embedding × 12, residual × 0.246, attention 1/64, logits ÷ 3;
  - recurrent state per request: `convState` [28, 1, 1792, 3] fp16 and `ssmState` [28, 1, 48, 32, 128] fp32, 22 MB.

## The Forge port and its reference

- **`Forge/language/granite_hybrid.py`:** a dense Granite 4.0-H decoder written for AetherKit from the config and the published architecture. It doesn't import or copy the prototype engine. Each Mamba layer runs a submission as one SSD chunk, with an optional padded tail, or as the one-token recurrence. The SSM state is fp32. It includes an F-P2-20 lint: every read of a state's old contents must feed that state's write.
- **`Forge/language/export_granite_hybrid.py`:** exports `main`, optionally at a static width, and optionally `decode`. It records versions, hashes and the lint result. The cpu_only self-check covers the dynamic contracts; static-width exports qualify natively (see the cpu_only defects).
- **`Forge/language/reference_granite_hybrid.py`:** the fp32 CPU reference from upstream Transformers (`GraniteMoeHybridForCausalLM`, 4.57.3), independent of the port.
  - **Chat cases:** three chat-rendered cases, each with a 128-token greedy continuation and 32 teacher-forced rows.
  - **Long context:** a 4,096-token document (this repository's docs) fed one token at a time through upstream's recurrent decode path, with rows every 256 tokens. Upstream builds its hybrid cache only inside `generate`, so the script creates it explicitly; without it, each call would see no context.
  - **Port check:** the eager float32 port matches upstream to maxAbs 1.2e-4 on the chat rows and 4.9e-5 on the long rows, with argmax equal on every row (`results/reference-port-check.json`).
- **`Spikes/hybrid-state-003/port_long.py`:** extends the drift reference to 8,192 tokens through the eager port, because upstream's one-token path takes about 0.07–0.15 s per token on the CPU.
- **Prefill plans are exact.**
  - On a random 4-layer model against upstream, max |Δ| is ≤ 4.8e-7 for whole-prompt, uneven-chunk, chunk-then-recurrent, all-recurrent and padded fixed-width plans (`tiny_vs_upstream.py`, `results/tiny-vs-upstream.txt`).
  - The same script forces the padded-source overflow, with large decay rates (A_log = log 60, dt_bias = 4). Before the fix, the padded submission's logits are non-finite. After it, they are finite and within 1.2e-7 of the unpadded plan.
  - On the real weights in float32 (`padded_exactness.py`, `results/padded-exactness-cpu.json`), the three chat prompts and a 1,000-token document prefix were each fed as one whole submission and compared with three other plans: the dynamic plan, padded 64-wide chunks and padded 16-wide chunks. Every plan matches to rounding: last-row logits within 9e-5, conv state within 2.1e-4, SSM state within 5e-6 relative. The padded plans are as close as the dynamic one.
  - Padded 64-wide prefill matches the fp32 reference's teacher-forced rows (cosine ≥ 0.99999999992, maxAbs ≤ 1.3e-4, 32/32 argmax on each case) with either decode: one token at a time through the recurrence (S1's `decode`), or as a padded 64-wide submission with one valid token (S2).

## Contracts tried

Attention keeps the 0.0.2 bucketed KV contract in every layout: read-only `keyCache` and `valueCache` inputs, `keyUpdate` and `valueUpdate` outputs, and masks built from position data. The Mamba layers update `convState` and `ssmState` in place. **Recurrent state can't step back**, so the bucketed contract's step-back tail is replaced:
- the dynamic layouts split the remainder into powers of two;
- the static layouts pad it.

| Layout | Functions | GPU outcome (full 32 layers) |
|---|---|---|
| C1 | `main`, dynamic q, positions [1, q] | **Failed as exported.** At q = 1, `main` aborts once the stack has 4 or more Mamba layers. Separately, the full graph didn't open in 6 min on a busy host. The cause is the two q-carrying inputs (defect 1). |
| C2 | `main` dynamic q + `decode` (q = 1, recurrence), positions [1, q] | **Failed as exported.** The separate `decode` avoids the q = 1 abort (8 layers: decode 4.9 ms warm), but `main` doesn't specialize at full depth: 16 layers open in 16 s (loadSeconds 16.22), 24 don't open in 3 min, and 32 don't open in 15 min at load 5. Also caused by the two q-carrying inputs (defect 2). |
| C2, start scalar | as C2, with `cache_positions` a [1, 1] start and the positions built in the graph | **Opens and runs** (bug-j's measurements; real dimensions, random weights). It isn't parity-qualified in 0.0.3; that re-qualification is planned for 0.0.4. |
| `decode` only | static q = 1 | Opens and runs in 2 s at 32 layers, so the blow-up is in the dynamic-q `main`. |
| **S1** | `main` at width 64 (padded tail) + `decode` | **Go:** opens in 2–3 s and qualifies (table above). |
| **S2** | `main` at width 64 only; decode is a padded 64-wide call | **Go:** opens in 1–2 s and qualifies. |
| Baseline | 0.0.2 `static-lean` rebuilt on the port: KV states fixed at 1,024 rows and blended in-graph, q = 1 | 209–210 tok/s, 796 MiB wired. Not Kit-compatible (fixed KV, one token per call); it anchors the 0.0.2 number on today's quiet host. |

Static width works for every layer, so KV capacity stays the only dynamic dimension, and that one only appears in attention. That remaining dynamic capacity was fine at full depth: S1 and S2 specialize in seconds with it. For a hybrid the KV cache is small (4 attention layers: about 32 MiB at 8,192 rows), so a fixed capacity per registration is the natural choice, with no growth ladder. All the S1 and S2 timings use one capacity per run.

## Results

From `Spikes/hybrid-state-003/results/`, warm medians, two interleaved rounds.

**Numerics.** These are the same for S1 and S2, on the GPU, against the fp32 reference:
- teacher-forced rows: cosine ≥ 0.99998, 0 argmax mismatches;
- greedy continuations: identical through 128 tokens on all three cases.

**Long-context drift, fp32 SSM state:**
- against upstream to 4,096 tokens, rows every 256: cosine ≥ 0.999992, argmax equal on 16/16;
- against the eager port to 8,192 tokens, rows every 512: the same.

No drift is visible.

**Time to first token.** KV capacity 8,448; prompts in 64-token chunks with the tail padded:

| Prompt tokens | Submissions | S1 | S2 |
|---|---|---|---|
| 64 | 1 | 9.3–9.4 ms (cold 17 ms) | 9.1–9.5 ms (cold 17 ms) |
| 256 | 4 | 32–36 ms | 32–33 ms |
| 1,024 | 16 | 124–128 ms | 124–128 ms |
| 2,048 | 32 | 248–255 ms | 247–252 ms |

For scale, at the same 64-token width the bucketed Qwen3-0.6B took 496 ms at 2,048 tokens ([0.0.3 record](../evidence/language-time-to-first-token-2026-10-27.json)), against the hybrid's 248–255 ms at W = 64. Since #64, main prefills the bucketed 0.6B in 256-token chunks with the cache sized up front, at 147–155 ms ([language guide](../language-generation.md#time-to-first-token)). A wider hybrid width is the next step (see [Open](#open)), not measured here.

**Decode:**
- **S1:** 163–166 tok/s after 256 and after 2,048 tokens at capacity 8,448; 182 tok/s at capacity 1,280.
- **S2:** 126–129 and 134 tok/s.
- **Baseline:** 209–210 tok/s.

S1's remaining gap to the baseline is probably the dynamic-capacity attention and the host's update-row copies, against the baseline's in-graph one-hot writes.

**Memory:**
- **S1** wires 799 MiB after its first `main` call and 1,446 MiB once `decode` has run: a second copy of the weights ([Qwen3-VL measurement](../qwen3-vl-single-decoder.md#what-core-ai-does-with-two-entry-points));
- **S2** wires 797 MiB.

**Cold specialization:**
- the static layouts open and run their first calls in 2–4 s on a quiet host;
- each new KV capacity is a new specialization of the attention shapes, which is cheap here but should be measured per registration.

The dynamic-q exports were the problem, because of how their positions input was declared (next section), not dynamic shapes as such.

## Core AI defects and Apple Feedback candidates

Each has a repro in `Spikes/hybrid-state-003/`. A follow-up bug hunt on 2026-10-28 traced defects 1 and 2 to this spike's export, and found that defects 3 and 4 don't reproduce on the coreai-core wheel's newer bundled runtime. Each item says what remains for Apple.

1. **GPU: a dynamic-q graph aborts at q = 1. The trigger is our export; not an Apple candidate as filed.**
   - **Cause:** bug-h's minimal pair isolates it. With an extra, unused `positions` [1, q] input that becomes the shape source of q, a stack with 5 or more layers aborts at q = 1. Without that input, the same stack runs. An MPSGraph robustness bug may still sit underneath, but Kit avoids it by giving q a single carrier (the start-scalar export in defect 2).
   - **As first observed in this spike:**
     - **Error:** `MPSNDArrayMatrixMultiplication.mm:212: failed assertion 'K dimension is inconsistently defined...'`.
     - **Graph:** a Mamba-2 stack in SSD chunk form with 4 or more layers, each updating two states in place, plus one attention layer.
     - **Shapes:** q = 4 and q = 2 run; with 3 Mamba layers, q = 1 runs too (`results/scale-L4.json`); cpu_only runs every q.
     - **Repro:** `q1_repro.py --layers mmmam` (torch and coreai-torch only, random weights), run with `q1-repro.swift`.
     - **A second abort:** the 3-layer standalone variant (`mmam`) instead aborts at q = 4 with `GPUMemrefOps.mm:164: failed assertion 'Failed to resolve dynamic dimensions for memref.alloc'`.
2. **GPU: a deep dynamic-q graph doesn't specialize. The cause is our export; a narrower MPSGraph slowdown remains for Apple.**
   - **Cause (bug-j):**
     - `main` takes the same dynamic q on `input_ids` [1, q] and `cache_positions` [1, q]. torch ties them (one shared Dim, and it takes q's size from the positions), but coreai-torch emits two independent dynamic dimensions.
     - Every q-dependent shape expression then becomes a broadcast between two unrelated symbols, nesting one level per layer. That nesting is the `generateIte` / `generateCondition` recursion in the samples below.
   - **Fix tested (bug-j):** pass `cache_positions` as a [1, 1] start scalar and build the positions in the graph (start + arange(q)), with the model otherwise unchanged.
     - **Real-dimension 32-layer C2** (dynamic-q `main` + `decode`, `expectFrequentReshapes`): it opens in **7.52 s**, against more than 15 min. Widths 4, 2, 8, 64, 256 and 3 all run with finite logits, and warm calls take 32–39 ms.
     - **Tiny-dimension 24-layer version:** it opens in 2.43 s, matches eager fp32, and runs q = 1 twice with no K-dimension abort.
   - **Remaining Apple candidate (bug-j):** MPSGraph open time grows exponentially for a single-input stack of cat, then a dynamic-end slice, then a residual add: 12 layers 0.77 s, 14 layers 2.73 s, 16 layers 9.80 s, 18 layers over 30 s. That is a separate, narrower defect than the one this spike first reported.
   - **As first observed in this spike:**
     - **Symptom:** `AIModel(contentsOf:options:)` with GPU preference runs for over 15 min at load 5. Every sample is in MPSGraph's `ShapeFunctionCanonicalizationPass` → `mpsx::shape_equation::performShapeEquationGeneration` → `z3_solver::simplifyShapeEquationsWithZ3` → `Z3ToMLIRGenerator::generateShapeFunctionFromZ3` → `generateIte` → `generateCondition`, recursing at least 20 deep.
     - **Depth:** 16 layers open in 16 s (loadSeconds 16.22, `results/depth-L16d.json`); 24 layers don't open in 180 s (`results/gpu-probe-outcomes.json`). With static width at all 32 layers, the dynamic KV capacity alone specializes in seconds.
     - **Repro:** `export_scaled.py --checkpoint <snapshot> --layers real:4:28 --decode`, run with the harness's `widths` phase. It needs the checkpoint for its `config.json` only; the weights are random.
     - **First guess at the time** (superseded by the cause above): compile also prints `FAILED: Input has no defining op (block argument)` twice, the 0.0.2 report's unexplained `staticq` message. The q-dependent shape expressions (Min/Max stride terms that torch can't prove either) probably compound per layer.
3. **cpu_only: a 2-D float additive SDPA mask computes wrong attention, silently** (the most serious of these, because it gives wrong output with no error). It's an Apple defect in the OS cpu_only path, present in macOS 27.0.1 (26A434). bug-d found that the coreai-core wheel's newer bundled runtime doesn't reproduce it, so it may already be fixed upstream. Run `mask_repro.py --save <dir>` (and `--mha`) for the attention-only assets; `mask-repro-gpu.swift <dir>` runs them on the GPU, or with `--cpu` on cpu_only. Output: `results/mask-repro-cpu.txt` and `results/mask-repro-gpu.txt`.
   - **cpu_only:** WRONG for the 2-D float mask over explicitly repeated KV heads with GQA (12/4, cosine 0.376), and for any 2-D float mask in plain multi-head attention (12/12, cosine 0.239, with or without `enable_gqa`).
   - **GPU:** all 24 variants are correct: {GQA, MHA} × {bool, float} × {2-D, 4-D} × {`enable_gqa`, explicit repeat} × {with a state, without}.
   - **No shipped graph is affected.** Forge's exporters use bool masks (Qwen3, Qwen3-VL text, FLUX.2's text encoder), no mask (Qwen3-VL vision, H3's conditioner), or 4-D masks (Kokoro's ALBERT).
   - **Matrix** (OK means cosine ≥ 0.999 against eager float32):

     | Variant (heads; mask; layout; GQA) | cpu_only | GPU |
     |---|---|---|
     | 12/4; bool; 2-D; repeat | OK | OK |
     | 12/4; bool; 2-D; `enable_gqa` | OK | OK |
     | 12/4; bool; 4-D; `enable_gqa` | OK | OK |
     | 12/4; float; 2-D; repeat | **WRONG 0.3764** | OK |
     | 12/4; float; 2-D; `enable_gqa` | OK | OK |
     | 12/4; float; 4-D; repeat | OK | OK |
     | 12/12; bool; 2-D; repeat | OK | OK |
     | 12/12; bool; 2-D; `enable_gqa` | OK | OK |
     | 12/12; bool; 4-D; `enable_gqa` | OK | OK |
     | 12/12; float; 2-D; repeat | **WRONG 0.2388** | OK |
     | 12/12; float; 2-D; `enable_gqa` | **WRONG 0.2388** | OK |
     | 12/12; float; 4-D; repeat | OK | OK |

     Each row was run with and without an in-place state update, which makes 24 variants on each compute unit. The verdicts and cosines are the same either way, so the state plays no part in defect 3 (it does in defect 4).
4. **cpu_only: `inferenceFailed` with state mutation plus a bool mask or `enable_gqa`.** It's an Apple defect in the OS cpu_only path, present in macOS 27.0.1. bug-e found that the bundled runtime doesn't reproduce it; it may be fixed upstream.
   - **Graph:** the 4-layer random Granite hybrid, whose Mamba layers mutate the states.
   - **Fails:** SDPA with `enable_gqa` (any mask form) or with a bool mask (2-D or 4-D) raises `(loc: unknown, error: inferenceFailed(-1))`.
   - **Runs:** the same graph with the states passed as plain inputs. Float masks without `enable_gqa` also run, but only the 4-D one is correct (defect 3).
   - **Workaround in `granite_hybrid.py`:** a 4-D float mask over KV heads repeated in the graph.
   - **Repro:** `mask_repro.py <dir> granite` (no checkpoint). Output: `results/mask-repro-granite-cpu.txt`.
5. **cpu_only: `inferenceFailed` with a static q > 1 and dynamic KV.**
   - **Not about states:** first observed here on a static-q SSD-chunk `main` with mutated states, padded or not, while the dynamic-q graph runs. The states don't matter. bug-k found that a single fp16 SDPA with a static q > 1 over dynamic-capacity KV fails on the OS cpu_only path regardless of states, and runs on the bundled runtime and on the GPU.
   - **Repro and consequence:** `export_granite_hybrid.py --tiny --static-width 8 --output <dir>`, without `--no-self-check`, so its cpu_only self-check raises. No checkpoint is needed. Static-width exports therefore skip the Forge self-check and qualify natively.
6. **cpu_only: a `torch.cond` abort.** The `torch.cond` abort from the [split-attention spike](split-attention-2026-10.md) is a separate defect, already recorded there.

## Caveats

- **Native-only qualification.** The static-width graphs can't run on the OS cpu_only path in macOS 27.0.1 (defect 5). Their accuracy rests on the GPU runs against the fp32 reference, plus the CPU proof of the padded maths in eager.
- **Quiet-host, few runs.**
  - The timings are two interleaved rounds on a quiet host.
  - The chat numerics are single runs per layout. The drift ran in both timing rounds, with identical cosines.
  - The baseline and capacity comparisons are single runs of 2–6 rounds of 256 steps each.
- **Random weights in the scaling variants.** The depth probes (4–32 layers) and the standalone repros use random weights at the real dimensions. They show compile and abort behaviour only, not accuracy.
- **What the product still needs:**
  - the Kit runtime contract below;
  - a registration format declaring the width, the capacity and the S1/S2 choice;
  - iPhone runs;
  - the T2 and T3 qualification fixtures;
  - specialization-time checks per registered capacity.

## Proposed Kit contract

The 0.0.2 report's state-role design stands, with these changes:
- **One `main` at a fixed width W (64).**
  - Prompts go in W-token submissions, with the last one padded: cache position −1 and token 0.
  - Logits are the last valid row's.
  - The host writes only valid KV rows.
  - There is no step-back tail and no remainder split.
- **An optional `decode` at q = 1 (S1).**
  - The registration chooses S1 or S2, which is a memory-for-speed choice the manifest should state.
  - Detection works as for the language contracts today: from function names and signatures, with the states `convState` and `ssmState` declared by role.
- **Fixed KV capacity per registration,** allocated once.
- **Recurrent states:**
  - they're Kit-owned buffers bound as states and zeroed per request;
  - rewinding means reset and replay.
  - The cache-statistics and admission terms from the 0.0.2 design apply.
- **Native GPU qualification:** teacher-forced rows with the near-tie rule, greedy, and a drift check. The cpu_only path can't run these graphs (defect 5).

**Estimate.** About 1.5–2 engineer-weeks of runtime work, plus about 3 days of qualification and the registration and manifest work:
- the new contract type;
- the padded prefill planner;
- state binding;
- the fixed-capacity cache;
- S1/S2 function selection;
- reset and replay.

## Open

- **Next step: wider widths.** Measure W = 128 and 256. Each submission costs about the same fixed amount here too, so a wider W should cut time to first token on long prompts, as #64's 256-wide chunks did for Qwen. It pads more, and S2's padded decode grows with W. That's the S1/S2 trade-off to re-measure at each width.

- The iPhone hasn't been run.
- Quantized weights haven't been tried. F-36 keeps `in_proj` at ≥ 8 bits.
- The T3 task gates and the prototype's 19-case T2 fixtures haven't been run. This spike's numerics cover 3 chat cases, plus drift to 8,192 tokens.
- **0.0.4: re-qualify a start-scalar dynamic contract.** Positions go in as a [1, 1] start (defect 2), and the dynamic-q `main` is measured and parity-qualified like S1 and S2. That would let hybrids use the bucketed planner without padding.

## Reproduce

```
uv sync --frozen --project Forge
Forge/.venv/bin/python -I Forge/language/reference_granite_hybrid.py --checkpoint <snapshot> \
    --source-repository ibm-granite/granite-4.0-h-350m --source-revision 3b17b717b8f2f5d305b0a92c1491e239aeda19c8 \
    --license apache-2.0 --output <new dir> --long-tokens 4096 --long-every 256
Forge/.venv/bin/python -I Spikes/hybrid-state-003/port_long.py --checkpoint <snapshot> --output <new dir>
Forge/.venv/bin/python -I Forge/language/export_granite_hybrid.py --checkpoint <snapshot> --static-width 64 [--decode] \
    --no-self-check --output <new dir>                                     # S1 with --decode, S2 without
Forge/.venv/bin/python -I Spikes/hybrid-state-003/export_baseline.py --checkpoint <snapshot> --output <new dir>
swift build -c release --package-path Spikes/hybrid-state-003/harness
~/.cache/aether-kit/with-gpu-lock.sh hybrid Spikes/hybrid-state-003/harness/.build/release/hybrid-step \
    <asset> <reference dir> <out.json> [--decode] --static-width 64 --capacity 8448 \
    --phases memory,numerics,long,ttft,decode
~/.cache/aether-kit/with-gpu-lock.sh hybrid-rounds Spikes/hybrid-state-003/run_rounds.sh \
    <dir with s1/ s2/ baseline/> <reference dir> <port-long dir> <out dir>    # the two interleaved rounds + 8k drift
```

CPU proof of the padded prefill: `Forge/.venv/bin/python -I Spikes/hybrid-state-003/padded_exactness.py --checkpoint <snapshot> --reference <dir> --output <json>`.

Defect repros:
- `q1_repro.py` and `q1-repro.swift` (defect 1 as first observed: both carry q on two inputs);
- `export_scaled.py` and the harness's `widths` phase (defects 1 and 2 as first observed);
- `mask_repro.py` and `mask-repro-gpu.swift` (defects 3 and 4).

The results are in `Spikes/hybrid-state-003/results/`. `gpu-probe-outcomes.json` records the runs that ended in an abort or a timeout, and `c1-c2-self-check.json` holds the C1 and C2 cpu_only self-checks against eager fp16 (logits cosine ≥ 0.99992, min 0.999929; conv state ≥ 0.99988, min 0.999885; SSM state ≥ 0.9999, min 0.999902). `tiny-c1-c2-self-check.json` holds the same checks on the random tiny model (all ≥ 0.999997). Every harness JSON in `results/` predates the harness's run-identity fields (`argv`, `assetIdentity`, `fixedCapacity`, `staticWidth`, load averages). For those runs, the capacity, width and options are in `run_rounds.sh` and the hold scripts' commands, and the load averages are in `hold-windows-and-load.txt`.
