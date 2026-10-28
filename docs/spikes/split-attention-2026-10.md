# Spike: split-attention prefill for the bucketed language contract (2026-10)

**Status: 0.0.3 go/no-go result (runs on 2026-10-27).** This spike follows up the bucketed time-to-first-token gap in the [0.0.3 roadmap](../roadmap-0.0.3.md#speed) and the [language guide](../language-generation.md#time-to-first-token). The idea was to re-export with attention that reads the read-only cache and the new chunk separately and merges the softmax partials, instead of concatenating `[cache ‖ chunk]` for every 64-token chunk. The code under `Spikes/split-attention/` is a probe and is not wired into Forge's exporters.

**Verdict: no-go under the current Core AI** (macOS 27.0.1, 26A434; coreai-torch 0.4.2, coreai-core 1.0.0b2, torch 2.11.0). Every placement fails:

1. **A second function wires a second copy of the weights.**
2. **Split attention in the one function halves decode.**
3. **A `torch.cond` branch in the one function converts, but runs on neither compute unit.**

Even if a placement worked, the gain at the 64-token width would be capped at about 25% on the 0.6B and 18% on the 4B, because most of each chunk is fixed per-submission cost.

## Where prefill time goes

The table shows warm time to first token for the bucketed contract (C = 64), with the host fixes applied, divided by the number of submissions ([evidence](../evidence/language-time-to-first-token-2026-10-27.json)). The device is an M5 Max on a shared host.

| Prompt tokens | Submissions | 0.6B: TTFT, ms per chunk | 4B writer: TTFT, ms per chunk | 0.6B running contract: TTFT, submissions |
| --- | --- | --- | --- | --- |
| 64 | 1 | 13.7, 13.7 | 31.2, 31.2 | 14.2, 1 |
| 256 | 4 | 46.3, 11.6 | 121.0, 30.3 | 16.8, 1 |
| 1,024 | 16 | 215.9, 13.5 | 530.3, 33.1 | 39.8, 2 |
| 2,048 | 32 | 495.7, 15.5 | 1,178.0, 36.8 | 83.0, 4 |

Per-chunk time hardly depends on the cache.
- **The fixed cost** is about 11.6 ms for the 0.6B and 30.3 ms for the 4B per submission. For the 4B that is mostly reading about 8 GB of weights.
- **The part that grows with the cache** is everything split attention could remove: the concatenation, SDPA over every capacity row, and growth on the host. At 2,048 tokens it is about (15.5 − 11.6) × 32 ≈ 125 ms of 496 (0.6B) and (36.8 − 30.3) × 32 ≈ 208 ms of 1,178 (4B).

**Best-case ceiling.** At C = 64, split attention would reach about 370 ms (−25%) on the 0.6B and 970 ms (−18%) on the 4B, and only if its extra kernels cost nothing. The targets are much lower:

| Model | Core AI running contract (C = 512) | MLX, one-pass prefill ([spike](mlx-backend-2026-10.md)) |
| --- | --- | --- |
| 0.6B | 83 ms | 50 ms |
| 4B | not measured | 501 ms |

The remaining lever on Core AI is fewer submissions per prompt.

## The three placements

### 1. A separate `prefill` function: a second wired copy of the weights

`main` would stay the fused decode graph, and a `prefill` function with the same signature would carry split attention. Core AI wires a copy of the weights for each entry point that runs, even for two identical graphs, and `SpecializationOptions` has no option to share them ([measurement](../qwen3-vl-single-decoder.md#what-core-ai-does-with-two-entry-points)). On the VL-4B that meant 16,086 MiB wired with two functions, against 8,414 MiB with one. A `prefill` function would add about 7.5 GiB on the writer.

### 2. Split attention in `main` for every step: decode halves

The decode steps also run through `main`. A hand-written split softmax in `main` was exact and CPU-clean, but decoded the 0.6B at **70 tok/s against 169 tok/s** for the fused graph (Forge spikes, 2026-10-26, recorded in `Forge/language/qwen3_decode.py`). It adds about a dozen kernels per layer, and they cost the same at q = 1.

The probe's split graph has one shared max and denominator and no cache copy. It confirms the numerics hold:
- eager: cosine ≥ 0.999997 against fused;
- cpu_only: ≥ 0.99981 over chunked prefill plus four decode steps, argmax equal;
- GPU: the same argmax on all three probe calls (the results file records each call's argmax and logit sum).
- It wires the weights once (92 MiB, the same as fused). Its fp32 scores raise Metal's allocation (242 against 143 MiB on the probe).

### 3. `torch.cond` in `main`: converts, then hangs the GPU compile and crashes cpu_only

One function would pick fused attention at q == 1 and split attention at q > 1, so the weights stay shared and decode is unchanged. The probe has two layers and random weights, with the 0.6B's attention shape (16 query heads, 8 KV heads, head dim 128, hidden 1,024); the asset is 76 MB.

**Conversion needs three workarounds:**
- **Where the cond goes.** It has to wrap the whole layer, which puts the projections inside both branches. Around the attention alone, `torch.export` fails on the branches' symbolic output strides (`KeyError: Min(128, 256*q)`).
- **The predicate.** It must be a tensor (`cache_positions[0, -1] == cache_positions[0, 0]`, true exactly when q == 1 under the contract). A size predicate lowers to a Python `eq` that coreai-torch cannot convert (`Unsupported ATen op: eq`).
- **Shape asserts.** The deferred shape asserts inside the branch subgraphs have to be removed before conversion (`KeyError: 'eq'` in `convert_branch_subgraph`).

coreai-torch then lowers the cond to Core AI's `if_`, and the asset saves at the same size as fused.

**Running it fails on both compute units:**
- **GPU** (`preferredComputeUnitKind: .gpu`, `expectFrequentReshapes`). Opening the asset did not finish in 7.5 minutes. It sat at 100% CPU in Core AI's compile, inside MPSGraph's AICode-to-MPS conversion, and was killed. The fused and split assets each opened and ran in seconds. Sampled stack, innermost last:

  ```
  mlir::ODIE::Compiler::Transforms::CompileForDelegates::generateCompilationTask()  (libODIECompiler)
  Compiler_coreAI.compile(moduleBytecode:to:with:)  (MetalPerformanceShadersGraph)
  -[MPSGraphAICodeCompilerDelegate getGraphExecutableFromBytecodeReleasingSource:aiCodeModule:sourceCtx:execDescriptor:]
  -[MPSGraphExecutable runMLIRModulePassesAndCommonInit:executableDescriptor:]
  (anonymous namespace)::AICodeToMPS::runOnOperation()
  mlir::applyPartialConversion(...)
  mlir::OperationConverter::convertOperations(...)
  mlir::reconcileUnrealizedCasts(...)  (MetalPerformanceShadersGraph)   <- every sample, 3 s
  ```

- **cpu_only.** The first call aborts the process: `CoreAIRuntime/NDArrayKernels.swift:105: Fatal error: Non tensor operand was passed to NDArray.alloc`. That also rules out Forge's cpu_only export self-check and Kit's CPU tests.

Because the cond asset never opened on the GPU, its wired memory and q = 1 cost could not be measured.

## Apple Feedback candidates

Both start from the asset `Spikes/split-attention/split_attention_probe.py export cond` writes (seconds of CPU, no checkpoint). Item 1 reproduces by running that asset with `split-attention-probe.swift` (GPU preferred); item 2 by `split_attention_probe.py check <fused .aimodel> <cond .aimodel>` (cpu_only).

1. **The GPU specialization of a function containing `if_` never finishes.**
   - The function has two `if_` regions, one per layer. The condition is a 0-d boolean computed from an int32 input. Each branch is a transformer layer (matmuls, RMSNorm, RoPE, SDPA, or an explicit softmax) returning three fp16 tensors with a dynamic dimension.
   - `AIModel(contentsOf:options:)` with GPU preference spins in `mlir::reconcileUnrealizedCasts` under `AICodeToMPS::runOnOperation` for more than 7 minutes.
   - The same graph without the `if_` (either branch alone) compiles in seconds.
2. **cpu_only aborts on the first call of a function with `if_`:** `Fatal error: Non tensor operand was passed to NDArray.alloc` (`CoreAIRuntime/NDArrayKernels.swift:105`). The asset and the inputs are the same as in item 1.

## Reproduce

```
Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py eager
Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py export <fused|split|cond> --output <new .aimodel>
Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py check <fused .aimodel> <split .aimodel>
Forge/.venv/bin/python -I Spikes/split-attention/split_attention_probe.py check <fused .aimodel> <cond .aimodel>   # Feedback item 2: aborts
xcrun swiftc -O -swift-version 5 -o .build/split-attention-probe Spikes/split-attention/split-attention-probe.swift \
    -framework CoreAI -framework Metal
~/.cache/aether-kit/with-gpu-lock.sh split-attention-probe .build/split-attention-probe <.aimodel>   # on the cond asset: Feedback item 1, never opens
```

The compact results are in `Spikes/split-attention/results/probe-2026-10-27.json`. With two layers, the probe's GPU timings are dominated by per-call overhead and are not comparable to real models.

## Open

- Revisit when Core AI either shares weights between entry points or compiles `if_` on the GPU. Option 1 or option 3 would then work without changing Kit's contract.
- Fewer submissions per prompt (a wider chunk) is the remaining Core AI lever. On the fused contract it was set aside because of memory retention per long request and greedy near-tie flips ([guide](../language-generation.md#time-to-first-token)).
