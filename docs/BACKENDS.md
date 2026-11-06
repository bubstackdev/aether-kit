# Alternate compute backends: MLX first (scope)

Status: **scope, not a plan of record**, 2026-11-06. No code depends on it. It answers what adding MLX (through
mlx-swift) as a second backend would take, and how to leave room for others without building a framework for them.

It's built from:
- [DESIGN.md](DESIGN.md) §3 (layering), §6 (runtime), §7 (text), §8 (vision), §10 (host);
- [MANIFEST.md](MANIFEST.md) §1–§2 and §5.3, and [M4.md](M4.md)'s owner decisions;
- the code at `origin/main` 3a7e6e9: `GraphBackend`, `Graph`, `TensorArena`, `DecoderEngine`, `Generator`,
  `ChatSession`, `Registry`, `StatelessPipeline`, `GPUAdmission`, `ResourceCoordinator`, `ModelRef`, the catalog;
- mlx-swift 0.32.3 and mlx-swift-lm 3.32.3, read from source after a resolve (below), and their READMEs.

**Experiment (cheap, deleted after):** a scratch package under this worktree depending on `mlx-swift-lm`
`upToNextMinor(from: "3.32.3")` *and* our pinned `swift-transformers` `exact: "1.4.0"` resolves cleanly (mlx-swift
0.32.3, swift-syntax 603.0.2 from a prebuilt, swift-crypto, swift-huggingface 1.0.0, swift-jinja 3.0.0, EventSource,
yyjson). Nothing was built or run on the GPU.

---

## 0. The short answer

- **The seam is not `GraphBackend`.** It's the decoder engine's surface as the `Generator` uses it: tokens in,
  logits (or greedy tokens) out, plus position, rewind, snapshot and restore. Extract that as one protocol; an MLX
  engine is a second implementation of it. Everything above it (Generator, ChatSession, structured output, tools,
  the FoundationModels bridge, AetherDecide's readout) carries over unchanged.
- **Start with one driver, `mlx-lm@1`,** in a separate SwiftPM package in this repo, so no Core AI user fetches or
  links MLX.
- **Don't** put diffusion, H3, Kokoro, or the M4 stateless drivers on MLX. They are done, gated and faster to keep.
- **When:** the seam extraction is about a day of work and changes public API, so it fits before the 0.1.0 freeze
  (`enforce` is still `false`). The MLX slices come after M4's exit criterion, behind a lab spike.

## 1. The seam

### 1.1 Why `GraphBackend` is the wrong boundary

`GraphBackend` (`Sources/AetherRuntime/GraphBackend.swift`) is "one precompiled function with a fixed signature":
`signature`, `makeBuffer(for:role:dimensions:in: TensorArena)` and `run(_: inout Bindings)`. Under it:

- `TensorArena` owns Core AI `NDArray`s (host) or Metal buffers wrapped for `InferenceFunction.encode`.
  `TensorBuffer`, `Bindings` and `StateSet` are views over that arena. The protocol is backend-neutral only for the
  CPU fakes in tests; it's how the fakes get in, not a backend boundary.
- An MLX model is Swift code (an `MLXNN.Module`) plus safetensors weights, evaluated lazily. Its KV cache is
  `[KVCache]` objects that grow in 256-slot steps (`KVCacheSimple.step`), trim in O(1), and quantize
  (`QuantizedKVCache`, `toQuantized(groupSize:bits:)`). Its state isn't named tensors in a signature.
- Wrapping MLX as a `GraphBackend` would mean exporting KV into arena buffers and copying them back every step, and
  inventing a `GraphSignature` for a model that has none. It would throw away MLX's lazy graph, its own cache, and
  `asyncEval` pipelining, all to fit a contract designed around `InferenceFunctionDescriptor`.

So `GraphBackend`, `Graph`, `TensorArena`, `Bindings`, `StateSet`, `GraphChain`, `WeightBanks`, `ShapeProfile` and
`GreedyKernels` stay what they are: **the Core AI backend's internals**. They don't get generalized.

### 1.2 The right seam: the language engine

`Generator` (`Sources/AetherText/Generator.swift`) holds `public nonisolated let engine: DecoderEngine` and uses
exactly this from it:

| member | use |
|---|---|
| `description.vocab`, `.maxCapacity`, `.hasRecurrentState` | sizing, JSON vocabulary, rewind vs snapshot |
| `position`, `stateRanAhead`, `decodeMode` | the token mirror and diagnostics |
| `prefill(_:logits:)`, `step(_:)` | prompt and depth-1 sampled decode |
| `rewind(to:)`, `reset()`, `snapshot()`, `restore(_:)` | prefix reuse |
| `beginGreedy(first:limit:)`, `nextGreedy()`, `endGreedy()` | the pipelined greedy path |

That's the boundary. A `LanguageEngine` protocol (an `Actor`-constrained protocol in AetherRuntime) with these
members, with `Snapshot` made opaque (`EngineSnapshot`: position plus a `Sendable` payload the engine owns), lets
`Generator`, `ChatSession.generator(…)`, `AetherLanguageModel` and `LMReadoutDecider` take `any LanguageEngine`.
`DecoderEngine` conforms with no behaviour change; the fake decoder keeps entering through `GraphBackend`.
`hidden(at:)` (DESIGN.md §6.2, for decision heads) joins the protocol when M5 needs it, as an optional capability.

`Logits` (host `[Float]`), `LogitsSelection`, the sampler, `SeededRandom`, `MemoryBudget`, `KVSizing`,
`GPUMemoryProbe`, `Registry`, `LoadOptions` and `ImagePreprocessor` already have no Core AI types in their API.
They live in AetherRuntime, which links Core AI, but every supported OS (macOS and iOS 27) has Core AI, so an MLX
module can depend on AetherRuntime. **Splitting AetherRuntime into a neutral module and a Core AI module isn't
needed** and should not be done for MLX.

### 1.3 What is backend-neutral and what is Core AI-specific

| Layer | Neutral | Core AI-specific |
|---|---|---|
| AetherCore | manifest decode and validation, catalog, `ModelRef` resolution, `ModelStore` (download, sha256, install), `DriverID`, `DriverCatalog` | `graphs.<g>` semantics (`function(s)`, `functionFamily`, `compute`, `expectFrequentReshapes`, `warmup`, `states` named by tensor), the zoo adapter |
| AetherRuntime | `Logits`, sampler and RNG, `MemoryBudget`, `KVSizing` arithmetic, `Registry`/`ModelDriver`, `LoadOptions`, `GPUMemoryProbe`, `FootprintLedger`, schedules and noise, `ImageInput`/`ImagePreprocessor` | `Graph`, `GraphOptions`, `TensorArena`, `TensorBuffer`, `Bindings`, `StateSet`, `DecoderCore`, `GreedyKernels` and the canary, `GraphChain`, `WeightBanks`, `GraphResidency`'s loader, `Device.current` (from `AIModel.deviceArchitectureName`) |
| AetherHost | `GPUAdmission`, `SharedGPULease`, `PriorityInferenceQueue`, `ResourceCoordinator`, `ModelHost` lifecycle, memory-pressure handling | `StatelessPipeline` and `ResidentAdmission` (built on `GraphResidency` roles and graph files) |
| AetherText | tokenizer adapter, renderers, stop and tag parsing, structured output, tools, `ChatSession`, `Generator` (once it takes the protocol) | none, apart from the `DecoderEngine` type in two signatures |
| AetherFoundationModels | everything (it sits on the Generator) | none |
| Task drivers | — | every driver today (image, speech, video, vision) is graph code |

## 2. Bundles and manifests

### 2.1 What an MLX model is on disk

An mlx-community repo (the `mlx_lm.convert` output) is `config.json` (with `model_type` and, when quantized,
`quantization: {group_size, bits}`), `model.safetensors` or shards plus `model.safetensors.index.json`, the
tokenizer files, `generation_config.json` and usually `chat_template.jinja`. mlx-swift-lm picks the model class from
`model_type` (`LLMTypeRegistry`) and loads from a local directory (`ModelConfiguration.directory(URL)`).

This fits our store as it is. A catalog entry is a repo, a revision and a list of files with sha256, and `ModelStore`
installs exactly those, so an MLX variant needs **no store or catalog-format change.**

### 2.2 The manifest: a driver, not a backend axis

Three options were considered:

1. **A `backend` field or variant axis** (`variant: {platform, backend: mlx}`, catalog key `macos-mlx`): this means
   `ModelRef.resolve` needs a preference order, and a policy for "both exist", which is new API on the hot path of
   every load.
2. **A backend-suffixed graph `compute`** (`compute: mlx`): wrong layer. `compute` picks a Core AI compute unit for
   one graph, and MLX has no graphs.
3. **A driver family:** `mlx-lm@1` (later `mlx-vlm@1`, `mlx-embedder@1`). **Recommended.** The driver already decides
   which code runs and which blocks it owns; that's what a backend is here. A model that exists on both backends
   becomes **two catalog ids** (`qwen3-0.6b`, `qwen3-0.6b-mlx`), so resolution doesn't change at all. If the owner
   later wants one id with a backend preference, that's a `ModelRef.resolve` option added on top, without a schema
   change.

The `mlx-lm@1` manifest:

```jsonc
{ "schema": 1, "id": "qwen3-0.6b-mlx", "task": "text-generation", "driver": "mlx-lm@1",
  "variant": { "platform": "macos" },
  "assets": { "main": { "subtree": "" } },
  "graphs": {},                                   // weights-only driver: see below
  "mlx": {                                        // owned by mlx-lm@1, like `krea2` or `kokoro`
    "config": "main:config.json",                 // model_type picks the mlx-swift-lm model class
    "weights": "main:model.safetensors.index.json",
    "modelType": "qwen3",                         // an assertion the driver checks against config.json
    "quantization": { "bits": 4, "groupSize": 64 },  // assertion, as for shapes elsewhere
    "kvCache": { "kind": "standard" },            // standard | quantized {bits, groupSize} | rotating {maxSize}
    "limits": { "maxContext": 40960, "prefillChunk": 512 },
    "device": "gpu"                               // gpu | cpu (cpu is the reference path for gates)
  },
  "tokenizer": { /* unchanged §2.4: source, chatTemplate, stop, thinking, toolDialect */ },
  "provenance": { "source": "Qwen/Qwen3-0.6B", "sourceRevision": "…", "producer": "mlx-lm 0.x convert", … } }
```

What changes in AetherCore: `mlx-lm@1` joins `DriverCatalog.drivers`; `Manifest.validate` stops requiring a
non-empty `graphs` when the driver's contract declares itself weights-only (today it's `require(!graphs.isEmpty)`);
a typed `mlx` block (closed vocabularies, like §5) with its own validation. The `tokenizer` block, `calibration` and
`decision` are reused as they are, which is the point: the text layers read the same blocks whichever engine runs.

### 2.3 Where MLX bundles come from

The owner's rule for M4 is "our own Forge exports, not zoo bundles". For MLX, mlx-community repos are the zoo
equivalent: third-party conversions, usually at an unpinned converter version. Recommended:

- **Forge `mlx/` exporter** (`Forge/mlx/export_mlx.py`): pins the upstream HF revision and an `mlx-lm` version in
  `uv.lock`, runs `mlx_lm.convert` (with `-q --q-bits 4 --q-group-size 64`, or 8-bit, or bf16), writes `aether.json`,
  records the greedy reference (§5) and uploads to `aether-models/aether-kit-<model>-mlx`. This is much less work than
  a Core AI export, with no graph tracing and no shape buckets, because conversion is a weight remap plus
  quantization.
- **mlx-community pinned directly** only for the lab spike, or for a model whose conversion we don't want to own,
  with the revision and hashes pinned like any zoo entry. Owner decision (§7, Q2).

### 2.4 `compute`, `residency` and footprint under MLX

- **`compute`:** no per-graph equivalent. MLX runs on its default GPU stream; `mlx.device: cpu` exists for
  references. The Neural Engine is not reachable from MLX.
- **`residency`:** an MLX LLM is resident for its lifetime (weights and cache), like `decoder-lm@1`. The phased
  `perPrompt`/`perStep`/`perImage` plan doesn't apply to one-model drivers.
- **Footprint is more predictable than on Core AI.** Core AI allocates about 4× the files on first run (G0 gate:
  +1.33 GiB for 320 MiB of qwen3-0.6b files), and retains memory for every input shape (lab/h3-spike). MLX holds the
  weights (about their quantized file size once evaluated), the KV cache (`layers × 2 × kvHeads × headDim × capacity
  × dtype`, rounded up to 256-slot steps), and transient activations, which are freed into **MLX's buffer cache**
  rather than to the OS. Without `compile` there's no per-shape retention. That makes the existing `KVSizing`
  arithmetic (bytes per token from the config instead of a signature, plus fixed bytes) directly applicable, and the
  model's charge can be computed and then checked against a measurement, not just guessed from file size.
- **Measurement:** MLX allocates `MTLBuffer`s on the same system `MTLDevice`, so our `GPUMemoryProbe`
  (`currentAllocatedSize` vs `phys_footprint`) sees it, **cache included**. `MLX.Memory.snapshot()` gives
  `activeMemory`, `cacheMemory` and `peakMemory` separately; the gate should record both and check that they agree.

## 3. Which drivers on MLX, and in what order

| Order | Driver | Why | Compared with Core AI |
|---|---|---|---|
| 1 | `mlx-lm@1` (causal LMs) | Biggest reach for least code: mlx-swift-lm has about 55 LLM architectures, including the ones our catalog lists as *unverified* on Core AI (MoE: Qwen3.5/3.6 MoE, GLM-4 MoE, LFM2 MoE, GPT-OSS; hybrids: Qwen3-Next, NemotronH, GraniteMoeHybrid, Jamba, Mamba2). New open models usually land there within days, with quantized KV and rotating caches, and with no export, tracing or bucket work. | qwen3-0.6b: Core AI depth-3 ≈ 500 tok/s (M5 Max). MLX 4-bit at this size is overhead-bound (per-layer kernel launches and the lazy-graph build), and published figures for 0.6B are sparse: an M4 Max community report implies roughly 550 tok/s. Expect **comparable, ±30 %, to be measured.** At 4B and up, decode is bandwidth-bound and MLX's 4-bit weights read fewer bytes than our int8 or fp16 Core AI bundles, so **MLX is likely faster** there, and its memory is about weight size instead of about 4× the files. Prefill: our chunked Core AI is about 3.8k tok/s at 3072 tokens; MLX prefill on a Max chip is in the same range for small models (measure it). |
| 2 | `mlx-vlm@1` | MLXVLM has Qwen2/2.5/3-VL, Qwen3.5 VL, LFM2-VL, GLM-OCR, Gemma 3/4, SmolVLM2, FastVLM, Pixtral, Idefics3, PaliGemma. Our Core AI VLMs (M4g–h) need grid-static tower exports (dynamic towers gave wrong GPU outputs about 2.5 % of the time in 0.0.x) and the `vlm-static-inject@1` plumbing. On MLX the image path is model code. | Unmeasured. The open question is preprocessing parity: MLXVLM ships its own processors, while ours is `ImagePreprocessor` with bit-exact Pillow and torch filters. We'd feed ours, as the one image primitive. |
| 3 | `mlx-embedder@1` | MLXEmbedders has BERT, NomicBERT, Gemma 3 (EmbeddingGemma), Qwen3 and LFM2-bidirectional; MLXRerankers exists. | Low value. EmbeddingGemma is already gated on Core AI (M4c), and stateless graphs are Core AI's strength. Only worth it for a model we can't export. |
| — | diffusion (`latent-diffusion`, `flux2-klein`, `krea2`), `minimax-h3`, `kokoro`, detectors, depth, CLIP | **Not worth it.** Each is ported, exported, gated against CPU references and tuned (chains, weight banks, residency). An MLX version would be a second implementation of the same model, contrary to "one of each primitive", for no capability gain. | — |

Things MLX can't do that Core AI can: the Neural Engine (no measured win yet anyway), per-device AOT `.aimodelc`,
and Apple's own system-model pipeline. Things MLX does that our engine doesn't yet: quantized and rotating KV, MoE,
speculative decoding (MTP drafters), and a wider set of architectures.

## 4. Host integration

### 4.1 GPU admission and the lease

Admission is enforced in the `Generator`, not the engine (`gpu.admit(job:)` before the turn, released after the
decode's GPU work joins). An MLX engine behind the same `Generator` **takes one grant per reply automatically**, at
the request's `GPUJob` priority, and holds the cross-process `SharedGPULease` on unsandboxed macOS like any other
reply. The engine's obligation is the same as `DecoderEngine`'s: **don't return from `endGreedy()` (or from a
`prefill`/`step`) while GPU work is in flight.** MLX's `asyncEval` (which `TokenIterator` uses to keep one step
ahead) leaves work queued after a call returns, so the engine must `eval` or synchronize its stream before
`endGreedy()` returns, and on cancellation. A gate checks it (§5).

Running work is never preempted, which is right for MLX too: there's no safe point to stop a command buffer.

### 4.2 Memory

MLX's controls (`MLX.Memory`, all process-global, static):

| Control | Semantics (mlx-swift 0.32.3 source) | Mapping |
|---|---|---|
| `memoryLimit` | **Soft:** past it, allocations *wait on scheduled work*; they aren't refused. Default 1.5× the device's recommended working set. | A backstop only. Our `ResourceCoordinator`/`MemoryBudget` stays the refusal authority (`capacity` before anything loads). Set it to the process budget once, when the first MLX engine loads. |
| `cacheLimit` | Caps freed buffers MLX keeps for reuse. **Defaults to the memory limit**, so after one long generation the cache can hold gigabytes the OS doesn't get back. | **Must be set low** (for example, the engine's transient activation estimate, or a fixed 256 MiB) or the probe charges the cache to the model, and a co-resident Core AI pipeline gets refused or pressured. |
| `clearCache()` | Frees the cache now. | Call it on unload, on memory-pressure release (`ModelHost`), and after a reply when another pipeline waits. |
| `activeMemory`, `cacheMemory`, `peakMemory`, `snapshot()` | Live arrays, cache, peak since the last reset. | Recorded in the `FootprintLedger` entry, next to the probe's figure. |
| wired memory (`WiredMemoryManager`, tickets) | mlx-swift-lm's own admission for wiring (`withWiredLimit` is deprecated). | Don't use it; it's a second admission policy. Leave wiring off unless a measurement shows a win. |

KV capacity uses the same rule as `decoder-lm@1`: `min(maxContext, platform cap, what fits the budget after
weights)`, with bytes per token from `config.json`. The engine pre-sizes its cache steps up to that capacity and
throws `capacity` beyond it. Because the MLX knobs are global, two MLX engines in one process share them; the MLX
module owns one small `MLXMemory` configuration point, not one per engine.

### 4.3 Core AI and MLX in one process

Both submit to the same `MTLDevice` through their own command queues. Nothing in either framework prevents
coexistence, and our layers already account for it:

- the shared `GPUAdmission` serializes their turns (no two grants run at once by default);
- `currentAllocatedSize` counts both, so measured footprints stay honest, *provided MLX's cache is clamped*;
- `ModelHost` treats an MLX engine like any loaded model (lifecycle, idle unload, pressure).

Risks, all to measure in the spike: MLX's metallib and binary-size cost in an app that also links Core AI; whether
iOS jetsam counts the MLX cache (it's dirty memory, so it likely does); and how MLX's process-wide default stream
and eval lock behave when two MLX engines run. The engine actor should keep all of its MLX work on its own executor,
as `DecoderEngine` does with its graphs.

## 5. Validation

The gate style carries over almost unchanged:

| Gate | Core AI today | MLX |
|---|---|---|
| Greedy parity | token-exact against a reference recorded from the same bundle | **token-exact against `mlx_lm` Python's greedy output on the same converted weights**, recorded by the Forge exporter. Same C++ core and kernels, so exact is the right bar; any drift is a bug in our loop. |
| Against upstream | near-tie rule against HF fp32 (row cosine ≥ 0.995; argmax may differ only when top-1 − top-2 < 0.1 and ours is in the top five) | the same rule, against the same HF fp32 reference, at 4-bit, 8-bit and bf16. The quantization level is a gate input. |
| Cross-backend | — | Core AI and MLX variants of one model go through the same near-tie rule against the same HF reference. Never token-exact between backends. |
| Engine invariants | chunked prefill = token-by-token at about 3k tokens; two-turn rewind; hybrid snapshot and restore | identical tests on the `LanguageEngine` protocol, so both engines run one suite. Hybrids (Qwen3.5, LFM2.5, Granite-H) use `ArraysCache`/`MambaCache`, which don't trim, so they go through snapshot and restore, exactly the `hasRecurrentState` path. |
| CPU reference | CPU references in Forge (rel L2 bars for graphs) | `mlx.device: cpu` runs the same model code on MLX's CPU backend: a rel L2 bar on logits rows (for example ≤ 1e-3 bf16), cheap and in-process. |
| Text and FM | `TextGateTests`, `FoundationModelsGateTests` | the same suites, parameterized over the engine. |
| Host | admission, lease, coordinator tests | one new gate: a Core AI pipeline and an MLX reply queued on `GPUAdmission.shared` never overlap, the grant outlives the MLX work (an `asyncEval` canary, like the delayed-argmax one), and `cacheMemory` returns under `cacheLimit` after a reply. |

**Build caveat (important):** mlx-swift's README says SwiftPM on the command line **cannot build MLX's Metal
shaders**; `xcodebuild` can. Our `scripts/ci.sh` is `swift build`/`swift test` only, so MLX tests and gates need an
`xcodebuild test` lane (Xcode is at `/Applications/Xcode.app`; the active developer directory is the Command Line
Tools today). Apps build with Xcode, so consumers aren't affected. mlx-swift also declares `swift-tools-version: 6.3`
against our 6.2; the local toolchain is Swift 6.4.2, so it resolves, but it raises the floor for anyone building the
MLX product.

## 6. Phased plan

Each slice is one PR. Effort is focused days.

| # | Slice | Contents | Effort |
|---|---|---|---|
| **B0** | **Lab spike** (`lab/mlx-spike`, not the kit) | mlx-swift-lm on qwen3-0.6b (4-bit, 8-bit, bf16) and the hybrid qwen3.5-0.8b: decode and prefill tok/s on the M5 Max and on the test iPhone; `Memory.snapshot` against our probe; greedy token-exact against `mlx_lm` Python; a Core AI and an MLX model in one process under `GPUAdmission.shared`; metallib and binary size; the `xcodebuild` lane. **Go/no-go for B3.** | 2–3 |
| **B1** | **`LanguageEngine` seam** (kit, no MLX) | The protocol and opaque `EngineSnapshot` in AetherRuntime; `DecoderEngine` conforms; `Generator`, `ChatSession.generator`, `AetherLanguageModel` and the engine test suite take `any LanguageEngine`; API baseline updated. No behaviour change, and all gates rerun. | 1–2 |
| **B2** | **Manifest** (AetherCore only) | `mlx-lm@1` in `DriverCatalog`; weights-only contracts (`graphs` may be empty); the typed `mlx` block and validation; fixtures and decode tests. | 1–2 |
| **B3** | **`AetherMLX` package** | `Packages/AetherMLX/Package.swift` in this repo, depending on the root package by path plus `MLXLLM`/`MLXLMCommon` only (not `MLXHuggingFace`, `MLXFoundationModels` or the downloader: we have the store, the tokenizer and the bridge). An `MLXLanguageEngine` actor, `MLXLMDriver`, KV sizing from the config, greedy on the GPU (`argMax` + `asyncEval`, depth 2) with synchronize-before-return, host logits for sampling and constraints, `MLXMemory` (limit, clamped cache, `clearCache` hooks). | 4–6 |
| **B4** | **Gates** | §5 on qwen3-0.6b-mlx and one hybrid, in the `xcodebuild` lane: parity, chunked prefill, two-turn rewind and restore, the coexistence gate, tok/s recorded in this doc. | 2–3 |
| **B5** | **Forge `mlx/` + catalog** | The exporter (pinned `mlx-lm`, convert and quantize, `aether.json`, reference tokens), uploads to `aether-models`, catalog entries such as `qwen3-0.6b-mlx` and one MoE model Core AI can't run. | 2 |
| B6 | `mlx-vlm@1` (optional) | `ImagePreprocessor` output into MLXVLM's models; the `image` block reused; parity against `mlx_vlm` Python. | 5–10 |

About 12–18 days through B5. B1 is the only slice that touches the Core AI path.

### 6.1 Risks

- **mlx-swift-lm churn.** It went 2.x → 3.x in 2026 with breaking changes (the tokenizer and downloader decoupling).
  Pin `upToNextMinor` and gate every bump. We depend on the model classes and `KVCache`, not on its generate loop,
  `ChatSession` or bridges, which keeps the surface small.
- **Duplicate machinery.** mlx-swift-lm ships its own `ChatSession`, `MLXGuidedGeneration`, tool parsing, stop
  filter, reasoning parser, and a FoundationModels `LanguageModel` bridge (`MLXFoundationModels`). Using any of them
  would fork our text stack. The rule: MLX supplies **logits and cache**, nothing above.
- **Logits readback cost.** Copying a 150k-float row to the host every step (sampling, constraints) is the same cost
  as Core AI's depth-1 path. The greedy path stays on the GPU.
- **Build lane:** SwiftPM CLI can't build the shaders (§5).
- **iOS memory.** MLX's cache is dirty memory under jetsam, and the large-model entitlement question is the same as
  for Core AI.
- **Licensing:** mlx-swift and mlx-swift-lm are MIT (compatible, NOTICE entry). Model licenses are unchanged by the
  backend. The converted weights are a derivative and carry the upstream license, as our Core AI exports do.

### 6.2 Keeping the door open for other backends

What's minimal and enough:

1. **The driver id is the backend switch.** Registry, catalog, store and manifest already support it: a new
   backend is a new driver family plus its own top-level block. No `Backend` protocol, no backend enum in the
   manifest, no capability negotiation.
2. **`LanguageEngine` is the one shared protocol,** at the token level, because that's where everything worth sharing
   (Generator, structured output, tools, FoundationModels, decision readout) sits. **llama.cpp** fits it directly
   (it exposes logits, `llama_memory_seq_rm` for rewind, and state save and restore), and so would any future
   on-device runtime.
3. **A remote or server backend does not fit and shouldn't be forced in.** Server APIs don't return full logits,
   so constrained decoding, readout and prefix reuse don't apply. It belongs above `ChatSession`, and Apple's
   FoundationModels now covers it (WWDC 2026 added server-side third-party models to `LanguageModelSession`): an app
   that wants a cloud model uses that, side by side with our `AetherLanguageModel`.
4. **No cross-task backend abstraction.** Stateless vision on another backend would get its own pipeline beside
   `StatelessPipeline` (sharing `GPUAdmission`, `ResourceCoordinator` and `ImagePreprocessor`), and only when a real
   model needs it.

## 7. Recommendation

**Do:**
1. Land **B1 (the `LanguageEngine` seam) before the 0.1.0 tag**, while `api-freeze.json` is still report-only. After
   the freeze, the same change means deprecating `Generator.engine: DecoderEngine` and adding a second property.
2. Run **B0 (lab spike)** after M4's exit-criterion slices (M4b–M4f), before M4g–h (VLMs). Its numbers decide B3,
   and whether VLMs should go MLX-first (Q4).
3. If B0 is a go, do **B2–B5** as a separate track, with `mlx-lm@1` only and the package split, so no Core AI user
   fetches or links MLX.
4. Position MLX as **reach** (MoE, new architectures, large quantized models, long context), not as a replacement:
   Core AI stays the default and the only backend for generation, vision and speech.

**Don't:**
- generalize `GraphBackend`, `TensorArena` or `Bindings`, or split AetherRuntime;
- add a `backend` axis to `variant` or `ModelRef.resolve` yet;
- use mlx-swift-lm's ChatSession, guided generation, tool parsing, Hugging Face downloader or FoundationModels bridge;
- port diffusion, H3, Kokoro or the stateless M4 drivers to MLX;
- build a remote backend.

### 7.1 Open questions for the owner

1. **Seam timing:** may B1 (a public-API change to `Generator`, `ChatSession.generator` and the FM bridge
   initializer) land before 0.1.0?
2. **Bundles:** Forge-converted MLX bundles hosted under `aether-models` (consistent with M4), or pinned
   mlx-community repos (zero export work, third-party converter)?
3. **Product shape:** an `AetherMLX` product from a sub-package in this repo (recommended), a trait on the main
   package (SwiftPM still *resolves* mlx-swift and swift-syntax for every consumer), or a separate repo?
4. **VLMs:** should M4g–h stay Core AI exports, or would an MLX-first VLM path (B6) be acceptable for some models
   (Qwen3-VL, GLM-OCR) to skip the static-tower export work?
5. **Catalog ids:** are `<id>-mlx` sibling ids acceptable, or is one id with a backend preference wanted (a later
   `ModelRef` option)?
6. **iOS:** is MLX on the iPhone in scope for the first track, or macOS-only like FLUX.2, Krea and H3?
7. **Which first MLX-only model** justifies the track: an MoE (Qwen3.6-MoE, LFM2.5-8B-A1B), a bigger dense model
   (Qwen3-8B 4-bit), or long context with quantized KV?

## Sources

- mlx-swift (MIT; SwiftPM CLI can't build the Metal shaders, `xcodebuild` can):
  <https://github.com/ml-explore/mlx-swift>, `Package.swift` and `Source/MLX/Memory.swift` at 0.32.3.
- mlx-swift-lm 3.32.3 (MIT; MLXLLM, MLXVLM, MLXLMCommon, MLXEmbedders, MLXRerankers, MLXFoundationModels,
  MLXGuidedGeneration; tokenizer and downloader decoupled in 3.x): <https://github.com/ml-explore/mlx-swift-lm>,
  read from source after a resolve (`KVCache.swift`, `LanguageModel.swift`, `ModelFactory.swift`, `Evaluate.swift`,
  `ModelConfiguration.swift`, `WiredMemoryPolicies.swift`).
- MLX memory semantics: <https://ml-explore.github.io/mlx/build/html/python/memory_management.html>.
- WWDC 2026 Platforms State of the Union (Core AI as the new custom-model framework; MLX on Metal 4; server-side
  models in FoundationModels): <https://macrumors.com/2026/06/09/apple-outlines-major-ai-and-developer-tool-updates>.
- Small-model MLX decode figures are sparse; the about 550 tok/s for Qwen3-0.6B 4-bit on an M4 Max is inferred from
  a community post comparing a custom Metal backend with MLX. Treat it as a placeholder until B0 measures it.
