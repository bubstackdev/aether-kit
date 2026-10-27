# Spike: hybrid recurrent-state language model (Granite 4.0-H) (2026-10)

**Status: 0.0.2 feasibility result (runs on 2026-10-26), imported for 0.0.3.** It is the evidence for the hybrid-state go/no-go in the [0.0.3 roadmap](../roadmap-0.0.3.md#spikes). The verdict below is the 0.0.2 recommendation, not the 0.0.3 decision; [Open for 0.0.3](#open-for-003) lists what still has to be answered first. The report was written against the 0.0.1-era runtime. Since then 0.0.2 shipped its fixed-shape decode work as the `bucketedPositions` contract ([language guide](../language-generation.md#step-contracts)), which differs from the fixed-capacity variants measured here.

**Verdict: go for 0.0.3, as a language-runtime contract extension, provided the fixed-shape decode work from 0.0.2 lands first.**

- **Recurrent state works.** Mamba-2 state alongside a KV cache works through Core AI with kit-owned buffers and the kit's sequential encode→join contract. Every correct variant reproduces the fp32 reference: cosine ≥ 0.999995, 16/16 argmax, and identical 257-token greedy runs.
- **Speed.** The prototype's graph decodes at about 120 tok/s on the M5 Max. A fixed-shape decode function with a low-op recurrence reaches **about 190 tok/s**; a fused custom-Metal-kernel recurrence reaches **about 200 tok/s**.
- **The "~28 tok/s" premise is stale.** The engine itself superseded F-P2-24 with F-P3-14: about 112–117 tok/s, with contention the likely cause of the 28. This spike reproduces about 120 tok/s from a fresh export of the same graph.

This is a feasibility spike, not product work. The code under `Spikes/hybrid-state/` is throwaway and is not wired into `Package.swift`.

## Setup

- Device: Apple M5 Max, 64 GB, macOS 27.0.1 (26A434). Exporter: coreai-torch 0.4.2 and torch 2.11 from the prototype Forge venv (read-only).
- The model definition and weight loading reuse the engine's `aether_forge.arch.granite_hybrid` / `prim.ssm` through `PYTHONPATH`, unmodified.
- Model: `ibm-granite/granite-4.0-h-350m@3b17b71`, fp16.
  - 28 Mamba-2 layers (48 heads × 32, d_state 128, conv kernel 4, one group) and 4 attention layers (NoPE, 12 query heads, 4 KV heads).
  - Recurrent state: `convState` [28,1,1792,3] fp16 and `ssmState` [28,1,48,32,128] fp32, 22 MB total.
- Timing harness: `Spikes/hybrid-state/harness`, a standalone Swift package.
  - It mirrors the kit's `LanguageGenerationSession`: `AIModel` with GPU preference and `expectFrequentReshapes`, kit-owned shared `MTLBuffer` states bound as `AsyncMutableValue`s, and an encode → `currentWorkCompleted()` → host argmax per token.
  - It allocates every state the function declares, from its descriptor. `AsyncMutableValue` is `~Copyable`, so a variable state count is bound by recursion.
  - Each run uses a 46-token prompt (fed one token per call), then 128–256 greedy decode steps. The metric is the warm median per-token latency. Cells are interleaved forward/reverse across 3–4 rounds.
- Numerics: the first 16 logits rows of every run are compared with the authors' Transformers model, run in fp32 on CPU in full-recompute mode (`compare_logits.py`).
- Profiling uses Core AI's runtime profiler events (`coreai.runtime.Profiler`) and operation counts from the asset summary.
- **Contention caveat.** Four other agents were building and running on this Mac; the 1-minute load average was 4–16 during runs. Absolute timings move by up to ±25% between rounds. Comparisons are made within a round. Every GPU job ran under `with-gpu-lock.sh`.

## Variants and results

All variants are q = 1 decode functions over the same fp16 weights.

| Variant | What changes | Correct? | ms/token (tok/s), clean rounds 4–6 |
|---|---|---|---|
| `ref:decode` | Prototype recipe @5 companion: direct recurrence, dynamic q/total/capacity metadata | yes | 8.1–8.3 (**120–123**) |
| `ref:main` | Prototype dynamic SSD step, run at q = 1 | yes | rounds 1–2: about 10–15% slower than `ref:decode` |
| `nomamba` | Mamba mixers removed: the attention + MLP + head floor (timing only) | no (by design) | 2.4–2.9 (350–410) |
| `split` | One state tensor per layer instead of stacked [28,…] | yes | ≈ `ref:decode` (no gain) |
| `nobarrier` | No F-P2-20 read-before-write barrier (timing only) | yes here | ≈ `ref:decode` (no gain) |
| `fp16state` | SSM state stored and updated in fp16 | yes (16 steps) | **slower**: 10.3 vs 9.6 ms (casts) |
| `staticq` | Static q = 1, total/capacity still dynamic | yes | **slower** (10.7 ms); compile prints "FAILED: Input has no defining op" twice |
| `static-full` | Fully fixed shapes (below), forge recurrence | yes | 6.7 (149); 8.5 at load 14 |
| `static-lean` | Fixed shapes + recurrence rewritten for fewer ops | yes, 257/257 tokens identical | 5.1–5.3 (**188–198**) |
| `static-kernel` | Fixed shapes + whole recurrence as one custom Metal kernel per layer | yes, 257/257 tokens identical | 5.0 (**200**) |
| `static-nomamba` | Fixed-shape floor | no (by design) | 2.4–2.9 (350–410) |

`static-*` means: `input_ids` [1,1], `position_ids` [1,1] holding the *current* position, and KV capacity fixed at 1024. The new KV row is blended in at the runtime position (a one-hot over capacity, data-derived per F-P2-6) and the attention mask is `arange(C) <= pos`. That is cheap here because only 4 layers have attention. The first `static-kernel` build (round 5) produced NaNs: Metal tensor subscripts run innermost-first, so the indices were reversed. The corrected build is round 6.

## Why the hybrid is slow: host-side encode, not the SSD maths or bandwidth

- **The recurrence formulation hardly matters.** The recurrent single-token companion is only about 10–15% faster than running the chunked/quadratic SSD graph at q = 1. At q = 1 the SSD "scan" is O(1), so the formulation was never the bottleneck.
- **The time goes to per-layer graph overhead.** Removing all 28 Mamba mixers drops decode from 8.3 to about 2.5 ms. Each Mamba layer costs about 210 µs, while its weights (in_proj 5.2 MB, out_proj 2.4 MB) and state (0.8 MB read + write) need about 25 µs at the measured ~400 GB/s. The prototype's own F-P3-14 already showed the hybrid reads only 76 GB/s.
- **The GPU waits on encoding.** Core AI's profiler shows each step dominated by `Function Encode`, i.e. CPU-side MPSGraph encoding:
  - `static-full`: 9.25 ms encode, 0.12 ms GPU wait;
  - `static-lean`: 5.5 ms encode, 1.1 ms wait;
  - `static-kernel`: 4.8 ms encode, 0.8 ms wait;
  - `static-nomamba`: 2.1 ms encode, 2.5 ms wait (only here does the GPU become the limit).
  - Encode cost scales with the number of dispatched ops, about 3–4 µs each. That also explains the timing noise under CPU contention.
- **The dynamic-shape path adds per-call shape inference.** The Swift path with `expectFrequentReshapes` hides most of it. The Python runtime, which has no such option, re-infers types on every call: about 2 s of type inference plus 2 s of encode per token for `ref:decode`.
- **The conversion inflates op counts.** The Mamba layer converts to about 350 IR ops per layer (`opcount-static.json`). Real compute ops number about 100 (constant, reshape and broadcast plumbing dominate), against about 4,400 IR ops for the whole fixed-shape attention/MLP floor.
- **Fusion helps, with diminishing returns.** Collapsing the whole recurrence into one kernel (conv shift, conv + SiLU, softplus, decay, state update, y reduction) saves only about 0.2 ms more than the lean rewrite. What remains per layer is the ~20 surrounding dispatches: RMSNorm, in_proj, two state writes, gate, norm, out_proj and residual.
- **Ceiling estimate.** About 190–200 tok/s is in hand with fixed shapes plus a lean or fused recurrence. Fusing RMSNorm and the gate (coreai-torch ships an `RMSNormImpl` composite) or folding the in_proj split might reach roughly 230–260 tok/s. The bandwidth floor is about 590 tok/s (0.68 GB at about 400 GB/s), and the fixed-shape attention/MLP floor alone is about 400 tok/s.
- **Iteration-level pipelining would not help much.** The cost is CPU encode, not GPU idle: overlapping the next encode with GPU execution saves at most the 0.8–1.1 ms wait.

## State dtype (F-36)

F-36 governs quantization: projections that feed recurrent state updates (`in_proj`, tagged `StateUpdateProjection`) must stay ≥ 8-bit. For Granite 350M, even per-channel int8 failed the two-level T2 gate and per-block int8 failed on iPhone (F-P2-25), so fp16 weights are the shipping choice.

For the state itself:

- **Keep the SSM state in fp32.** It is 22 MB, and the fp16 variant was *slower* because of the extra casts. Its 16-step logits matched, but the prototype measured 1.1–1.4% state error in fp16 compute (F-P2-21), and long-context drift is untested.
- **The conv state can stay fp16.** It holds raw in_proj outputs.

## Proposed state contract for the language runtime

On origin/main, stateful language generation is "Not implemented". The local-line runtime (`origin/local-line-2026-10-26`, `Sources/AetherLanguageCoreAI/LanguageGenerationSession.swift`) hard-requires:

- `main` with inputs exactly `{input_ids, position_ids}`;
- states exactly `{keyCache, valueCache}`: fp16, 5-D, dynamic axis 3;
- fp16 logits.

The proposed extension:

1. **Declared state roles in `language-model.json`**, matched against the function descriptor, instead of hard-coded names.
   - `kvGrowing(axis:)` (today's caches) or `kvFixed(capacity:)` (the fixed-shape decode work);
   - `recurrent(scalarType:, shape:)`: fixed-size, written whole every call, never truncated.
   - The runtime allocates one zeroed kit-owned buffer per declared state from the descriptor, as the spike harness does. It enforces scalar types (fp32 for the SSM state) and exposes byte counts in `CacheStatistics`.
2. **Named functions with roles.** `step`/`prefill` (dynamic or chunked q, used for prompt chunks) and `decode` (q = 1, fixed shapes, current-position input).
   - The runtime picks `decode` only for one-token calls, as the prototype does for its companion (F-HYBRID-1). It verifies at open that both functions share input, output and state names.
3. **Rewind and reset semantics.**
   - Recurrent state cannot be truncated, so any rewind below the current end means reset plus replay. Prefix reuse becomes "replay from a stored snapshot".
   - The kit's sequential engine joins every submission before reading or mutating state, so it needs no snapshot ring for stop tokens. That ring exists in the prototype only because its pipelined engine runs ahead (F-P2-22).
   - A future pipelined path must snapshot or restore recurrent state: 22 MB per step for Granite 350M.
4. **Prefill.** Chunked prefill through the SSD `step` is exact across chunks (F-P2-21). The `[q, q, heads]` fp32 decay tensor bounds the chunk: about 50 MB per layer at q = 512, so the prototype used `prefill_chunk = 256`.
5. **Ordering.** Exports must make every read of a state's old contents feed its write (F-P2-20, `after()`). The spike's barrier-free variant happened to be correct, but that is luck, not a contract. The kit should keep a lint at export time.
6. **Memory admission.** Recurrent bytes are constant in context length, which is the point of hybrids. Admission adds them as a fixed term next to KV growth.

The same contract covers other recurrent families: Zamba2 (Mamba-2 + shared attention); Gated DeltaNet hybrids such as Qwen3-Next and Qwen3.5, for which coreai-torch already ships a `GatedDeltaUpdate` composite; and RWKV-style state.

## Effort estimate (0.0.3)

About 2–3 engineer-weeks, assuming the 0.0.2 fixed-shape decode export has landed:

- Runtime: declared-state allocation and binding, multi-function selection, reset/replay rewind, admission and statistics. About 1 week.
- Export recipe for Granite 4.0-H 350M (fixed-shape decode with the lean recurrence or custom-kernel variant, plus chunked SSD prefill), with self-checks. About 3–4 days.
- Qualification: strict T2 with the prototype's 19-case fixtures or an independent reference, T3 task gates, iPhone JIT run, and long-context state-drift check. About 4–5 days.
- Optional: RMSNorm/gate fusion, and custom-kernel productization (Metal source in the export, compile cost and iPhone validation). About 3 days.

## Recommendation

**Go for 0.0.3**, ordered after the 0.0.2 fixed-shape decode work:

1. Ship the state-role contract with Granite 4.0-H 350M (Apache-2.0) as the first hybrid.
2. Use the fixed-shape lean recurrence as the baseline: about 190 tok/s on the M5 Max, bit-for-bit greedy-equal to the prototype graph.
3. Treat the custom Metal kernel as an optional optimization, since it adds only about 5%.

Don't put recurrent snapshots into the sequential runtime.

## Licensing

`ibm-granite/granite-4.0-h-350m` is **Apache-2.0**: model card `license: apache-2.0` and HF metadata, ungated, pinned revision `3b17b717…`. The checkpoint was already in the prototype's cache (`~/Library/Caches/aether/checkpoints`); nothing new was downloaded for this spike.

## Disk footprint

- Peak: about 5.5 GB of Granite exports under `~/.cache/huggingface/aether-kit/spikes/granite/` (deleted progressively).
- After cleanup: **0 B**. All exports were deleted after the measurements were committed; regenerating one takes 5–15 s (`export_variants.py`).
- The 658 MB Granite checkpoint belongs to the prototype cache and was left untouched.
- The harness's Core AI specialization cache (`~/Library/Caches/coreai-cache/26A434/hybrid-decode`, 4.8 GB) and this spike's Python-runtime entries (1.6 GB) were deleted.

## Evidence

- Code: `Spikes/hybrid-state/`
  - `export_variants.py`: every variant, by name (`ref`, `nomamba`, `split`, `nobarrier`, `fp16state`, `staticq`, `static-full`, `static-lean`, `static-kernel`, `static-nomamba`);
  - `harness/`: the Swift timing harness;
  - `run_rounds.sh` and `profile.sh`: the lock-held timing rounds and profiler captures;
  - `compare_logits.py`, `summarize_profile.py`, `opcount.py`, `make_prompt.py`.
- Results: `Spikes/hybrid-state/results/`
  - `timing-summary.json`: every cell and round;
  - `round*-numerics.json`: the teacher-forced comparison of each run's first 16 logits rows (no round 2 file);
  - `greedy-tokens.json`: the reference greedy run, and for every run its length and whether its tokens match the round's `ref:decode`;
  - `profile-summary.txt`: per-step profiler intervals for each captured variant;
  - `round*.log`: harness output per round, with the load average before each run;
  - `opcount*.json`, `prompt.txt`.
- Not committed: the per-run harness reports (`round*/*.json`, with per-step milliseconds), the logits rows, the profiler event files and a one-run smoke report. `greedy-tokens.json` and `profile-summary.txt` were derived from them before they were dropped; every number above is in a committed file.
- GPU lock holds, 439 s (7.3 minutes) in total. The command column gives today's equivalent invocation; each hold ran its profiler captures, then its rounds.

| Job | Window | Duration | Command (`profile.sh`; `run_rounds.sh`) |
|---|---|---|---|
| smoke | 14:19:30–14:20:04 | 34 s | one harness run of `ref:decode` |
| variants | 14:23:12–14:24:24 | 72 s | `round1 256 2 ref:decode ref:main nomamba:main split:main nobarrier:main fp16state:main staticq:main` |
| profile + timing | 14:27:10–14:29:28 | 138 s | `6 ref:decode nomamba:main ref:main`; `round2 128 4` with round 1's cells |
| static | 14:43:28–14:44:44 | 76 s | `12 ref:decode static-full:main static-nomamba:main`; `round3 128 3 ref:decode nomamba:main static-full:main static-nomamba:main` |
| lean | 14:53:29–14:54:01 | 32 s | `12 static-lean:main`; `round4 256 3 ref:decode static-full:main static-lean:main static-nomamba:main` |
| kernel | 15:00:33–15:01:31 | 58 s | `12 static-kernel:main`; `round5 256 3 ref:decode static-kernel:main static-lean:main static-nomamba:main` |
| kernel v2 | 15:07:15–15:07:44 | 29 s | as kernel, into `round6`; its capture is kept as `static-kernel2-main` |

## Reproducing

The environment has changed since these runs:

- **Export needs the prototype engine.** `export_variants.py` imports the prototype engine's `aether_forge` package (`arch.granite_hybrid`, `prim.ssm`, `export`, `arch.qwen3`) through `PYTHONPATH`. That package is not part of AetherKit, and Forge has no Granite hybrid module, so the exports can't be regenerated from this repository alone. Because it relies on `PYTHONPATH`, the exporter runs without `-I`: `PYTHONPATH=<engine>/forge Forge/.venv/bin/python Spikes/hybrid-state/export_variants.py <variant>…`. Exports go to `~/.cache/huggingface/aether-kit/spikes/granite/`.
- **The checkpoint path is the prototype's cache.** `export_variants.py`, `compare_logits.py` and `make_prompt.py` read revision `3b17b71` from `~/Library/Caches/aether/checkpoints/` rather than the Hugging Face hub cache.
- **Everything else runs in Forge's locked environment** (`uv sync --frozen --project Forge`): it has the same torch 2.11 and coreai-torch 0.4.2 as the prototype venv, and its transformers 4.57.3 includes the Granite hybrid model that `compare_logits.py` and `make_prompt.py` load. `profile.sh` uses it by default (`SPIKE_PYTHON` overrides it).
- **The harness still builds** with `swift build -c release --package-path Spikes/hybrid-state/harness` on the current toolchain (checked 2026-10-27, with Swift 6 concurrency warnings only). Timing runs go through `run_rounds.sh` under `~/.cache/aether-kit/with-gpu-lock.sh`.

## Open for 0.0.3

What these findings leave unanswered before a go/no-go:

1. **Fit with the bucketed contract.** The proposal assumed in-place KV states. The `bucketedPositions` contract that shipped instead passes the KV cache as read-only inputs, copies `keyUpdate`/`valueUpdate` rows after each join, and ends prefill with a chunk that steps back over rows already written. Recurrent state can't be stepped back: recomputing those rows would apply the tokens twice. The contract for a hybrid (in-place recurrent states next to bucketed KV, or a separate one) has to be chosen, and the ~190 tok/s re-measured under it, since every `static-*` number here used a fixed 1024-row KV capacity blended in the graph.
2. **An exporter inside AetherKit.** The Granite hybrid module lives only in the prototype engine. It must be ported into Forge, with a reference that is independent of the exporter.
3. **Prefill and long-context state.** Every run fed its 46-token prompt one token per call. Neither chunked SSD prefill through Core AI nor fp32 SSM-state drift over long contexts was measured.
4. **Qualification depth.** The comparison covered 16 teacher-forced rows and greedy runs of up to 257 tokens on one prompt, not the prototype's 19-case T2 fixtures or T3 task gates. No iPhone run was made, including for the custom kernel.
5. **Quiet-host timing.** Every timing was taken with a 1-minute load average of 4–16, so the absolute numbers are only good to about ±25%.
6. **The `staticq` compile failures.** "FAILED: Input has no defining op" printed twice and wasn't explained. It doesn't block the fixed-shape path, but its cause is unknown.
