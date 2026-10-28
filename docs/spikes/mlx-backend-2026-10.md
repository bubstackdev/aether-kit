# Spike: MLX older-OS language backend (2026-10)

**Status: 0.0.3 go/no-go result (runs on 2026-10-27).** It answers the older-OS backend question in the [0.0.3 roadmap](../roadmap-0.0.3.md). It is a feasibility spike, not product work: the code under `Spikes/mlx-backend/` is its own package, pinned to mlx-swift 0.33.0, and is not wired into `Package.swift`.

**Verdict: go for an MLX language backend at a macOS 14 / iOS 17 floor, as a separate optional product, on three conditions:**

1. The floor PR lands first. Kit's products are still macOS 27 / iOS 27, so a macOS 14 package cannot conform to `LanguageBackend` yet. The spike runs standalone for that reason.
2. Depend on mlx-swift core only, not mlx-swift-lm. mlx-swift-lm 3.32.3 pins swift-syntax `602..<604`, which cannot resolve alongside Kit's exact 604.0.0. The decoder here is about 170 lines adapted from mlx-swift-lm (MIT; its license is in `Spikes/mlx-backend/MLXQwen3/LICENSE-mlx-swift-lm`).
3. CI can build Metal shaders (see [Dependency cost](#dependency-cost)).

Results:

- **Numerics.** Both Qwen3 models pass Kit's own qualification protocol against the same fp32 references as the Core AI bundles: cosine at least 0.999998, and argmax and greedy agree under the near-tie rule.
- **Speed.** At these lengths MLX is faster than Kit's Core AI bucketed contract, for both time to first token and decode.

## Setup

- Device: Apple M5 Max (`applegpu_g17s`), 64 GB, macOS 27. Every GPU job ran under `with-gpu-lock.sh`. The host was shared with other agents, so timings are observations, not benchmarks.
- Package: `Spikes/mlx-backend/MLXQwen3`, with platforms macOS 14 / iOS 17 and products `MLX` and `MLXNN` from mlx-swift 0.33.0.
  - `Qwen3.swift` loads the Hugging Face snapshot's safetensors directly and casts them to the run dtype; the tied head is used when the embeddings are tied.
  - `MLXQwen3Runner` provides teacher-forced logits and greedy generation: one whole-prompt prefill, then one token per step, with a concatenating KV cache.
- Models, both fp16:
  - `Qwen/Qwen3-0.6B`;
  - the writer, `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated`.
- `mlx-qwen3-check` mirrors Kit's language qualification:
  - forced rows: cosine ≥ 0.995, and argmax equal unless at a near-tie (reference top-1 − top-2 < 0.1 and the reference token ranks < 5 natively);
  - greedy divergence: judged with the same rule;
  - warm time to first token: the median of the timed rounds, at 64, 256, 1,024 and 2,048 prompt tokens;
  - decode rate: over 256 tokens;
  - MLX peak and active memory.
- Raw reports: `Spikes/mlx-backend/results/`.

## Results

| Model | Min cosine | Max \|Δ\| | Argmax mismatches (near ties) | Greedy | Decode | Peak MLX memory |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.999998 | 0.053 | 1 (1) | diverges at token 58, at an accepted near tie | 163 tok/s | 2,012 MiB |
| Writer 4B | 0.999998 | 0.064 | 0 | identical | 54 tok/s | 8,692 MiB |

The table below gives warm time to first token in milliseconds. The Core AI columns are Kit's warm figures on the same Mac, from the 0.0.3 TTFT test with the language-speed host fixes applied (`docs/evidence/language-time-to-first-token-2026-10-27.json` on that branch).

| Model | Prompt tokens | MLX | Core AI, bucketed | Core AI, running |
|---|---|---|---|---|
| Qwen3-0.6B | 64 | 7.7 | 14 | 14 |
| | 256 | 9.4 | 46 | 17 |
| | 1,024 | 24.6 | 216 | 40 |
| | 2,048 | 50.0 | 496 | 83 |
| Writer 4B | 64 | 19.9 | 31 | — |
| | 256 | 46.3 | 121 | — |
| | 1,024 | 170.9 | 530 | — |
| | 2,048 | 501.1 | 1,178 | — |

Most of the bucketed gap comes from its fixed 64-token prefill chunk, which attends over the whole cache once per chunk. That is being addressed separately in 0.0.3 (split-attention re-export). MLX prefills the whole prompt in one pass.

The spike did not measure Core AI decode in the same harness. A same-harness decode comparison is listed under open items.

## Dependency cost

- **Build.** mlx-swift's Metal shaders have to be compiled into `default.metallib`. mlx-swift documents that SwiftPM's native build system cannot do this. `swift build -c release --build-system swiftbuild` does, but it needs the Metal toolchain component, which is installed separately from Xcode on this OS.
  - The cold release build takes 2 min 45 s.
  - Every CI runner that builds the product needs the Metal toolchain.
- **Size.** The release `mlx-qwen3-check` executable is 30 MB, plus a 4.7 MB `default.metallib` that has to ship with the app.
- **Dependencies.** mlx-swift 0.33.0 brings in swift-numerics 1.1.1. Pin it exactly, as Kit pins its other dependencies.
- **Model format.** The loader reads Hugging Face safetensors as they are, with no export step, so the asset pipeline differs from Kit's `.aimodel` bundles. A product would need its own manifest kind and download plan.

## Open items

- Conformance to `LanguageBackend` (session, cancellation and stop handling) once the floor PR lands. A product also needs a chat template and tokenizer path shared with the Core AI backend.
- Metal toolchain availability on CI runners.
- Same-harness decode comparison against Core AI, and the cost of quantized MLX weights (4-bit), which the spike did not try.
- Memory: the 4B peaks at 8.7 GiB in fp16, more than an iOS 17 device can hold. iOS needs quantized weights or the 0.6B.
- VL models are out of scope for this spike.
