# Aether-Kit — Design Spec

Status: **draft v0.1**, 2026-11-05
Inputs: the coreai-kit architecture walkthrough (`../_reference/coreai-kit-architecture.md`), the Core AI SDK
interfaces in Xcode 27.2 (`MacOSX27.2.sdk`), and the `coreai-models` fork's engines as a reference for graph contracts.

Aether-Kit is a Swift package for running open-weight models on-device with Apple's Core AI framework. It is a
from-scratch re-architecture of `john-rocky/coreai-kit`. It shares no code with that repo or with `lab/`.

---

## 1. Principles

1. **Models describe themselves; code never guesses.** Every loadable bundle has a manifest that states its graphs,
   tensor roles, preprocessing, tokenizer and engine requirements. Code binds to *roles* from the manifest, not to tensor
   names, ranks, directory names or catalog ids.
2. **One of each primitive.** One graph runtime, one decode loop, one image pipeline, one tokenizer adapter, one JSON
   formatter, one RNG. A model family *parameterizes* these primitives; it doesn't copy them.
3. **Open registration, closed contracts.** Model families plug in through a registry keyed by `driver`. Adding a family
   never means editing a central `switch`. An unknown driver is reported as *unsupported*, not loaded wrongly.
4. **Ownership is explicit.** Every loaded model is owned by exactly one isolation domain. There are no
   `@unchecked Sendable` types whose safety rests on a comment, and no process-global configuration such as `setenv`.
5. **What ships is what was tested.** Catalog pins and per-file hashes are compiled into the package. There is no
   runtime catalog fetch.
6. **Pay for what you link.** Products split by task, so linking chat never pulls in vision kernels.

## 2. Settled decisions

| Topic | Decision |
|---|---|
| Bundles | The native `aether.json` manifest is the contract. Existing zoo bundles load through an **adapter** that synthesizes a manifest. |
| LLM engine | **We own it**, written directly on Core AI (`CoreAIRuntime`). No dependency on `coreai-models`. |
| v1 scope | Chat, the FoundationModels bridge, vision, embeddings, and LM-readout typed decisions. Speech is deferred. |
| Catalog | Our own catalog, compiled in. No runtime fetch. Weights may come from the zoo's HF repos. |
| OS floor | `platforms: [.macOS(.v27), .iOS(.v27)]`, with no `@available` noise. |
| System One | The Swift API comes first. The wire codec, HTTP server and MCP server come later, on the same `Decider` protocol. |
| Bespoke deciders | Re-ported in `lab/`, then promoted once they fit the runtime and manifest contracts. |
| Multi-asset bundles | One catalog entry per model, listing every asset subtree with per-file hashes; installed together, manifest in `.aether/`. |
| State roles | Dynamic sequence axis → sequence state, fully static → recurrent; exceptions in a reviewed override table. |
| Context | Capacity derived from the memory budget at load; the manifest carries only the declared maximum. |
| Per-arch AOT | `ios-<arch>/` if present, else the generic JIT subtree; `unsupportedDevice` only when neither exists. |
| Compute | GPU by default for every graph class (measured); Neural Engine only per entry with a measured win. |

## 3. Module layout

```
Aether (umbrella: one-line task ops; re-exports the task modules below)
 ├── AetherText         tokenizer adapter · prompt rendering · Generator · ChatSession · constrained decoding
 │    └── AetherFoundationModels   LanguageModel/Executor bridge (separate product; FoundationModels link)
 ├── AetherVision       image I/O + preprocessing · detection · depth · CLIP · super-resolution
 ├── AetherImage        image generation: latent-diffusion@1 (SD-Turbo), flux2-klein@1 (FLUX.2), then Krea 2 (§6.5)
 ├── AetherEmbeddings   text embedder · reranker · classifier
 ├── AetherDecide       Decision types · Decider protocol · LMReadoutDecider
 └── AetherHost         ModelHost: load-through-registry · residency · memory policy · capability()
          │
     AetherRuntime      Graph · TensorBuffer · StateSet · Bindings · DecoderEngine · Registry · AetherError
          │
     AetherCore         ModelRef · Catalog (generated) · ModelStore (download/install/verify) · Manifest · Device
```

Rules:
- `AetherCore` has no Core AI import. It covers identity, manifests and bytes on disk. It can be tested anywhere.
- `AetherRuntime` is the **only** module that touches `CoreAIRuntime` types. Nothing above it sees `NDArray` or `AIModel`.
- Task modules depend on `AetherRuntime` + `AetherCore`, never on each other. The exception is
  `AetherDecide → AetherText`, because LM readout uses the Generator's prefix machinery.
- Third-party dependencies are limited to `swift-transformers` (tokenizers, Jinja chat templates), in `AetherText` only.
  Downloads use `URLSession` directly; we don't need swift-huggingface.

Products: `AetherText`, `AetherFoundationModels`, `AetherVision`, `AetherImage`, `AetherEmbeddings`, `AetherDecide`,
`Aether`.
`AetherHost`, `AetherRuntime` and `AetherCore` are reachable through any of them.

## 4. The manifest (`aether.json`)

**The normative schema and the driver catalog are in [`MANIFEST.md`](MANIFEST.md).** This section is the summary.

- **One manifest per catalog variant.** A variant may span several asset subtrees of one repo (VLM towers, Gemma 4
  per-layer tables, retriever query/doc encoders) and may use a different driver from other variants of the same id
  (qwen3-0.6b on iOS is `static-chunked-lm@1`). It is generated at install into `.aether/aether.json` at the
  revision root, or shipped by our own exports. Versioned with `"schema": 1`. Unknown fields and `x-` fields are
  ignored. A catalogued driver with no registered implementation fails as `unsupported`, an uncatalogued driver
  id as `unknownDriver`, and a missing required block names its field. None of these ever falls back.
- **The manifest declares only what the graph descriptor cannot show.** Shapes, dtypes and static vs dynamic
  dimensions come from `InferenceFunctionDescriptor` at load. The manifest names *roles* and *meanings*: which input
  is tokens, which state is recurrent, which axis is sequence, which logits space, what preprocessing applies. Any
  shape written in a manifest is an assertion the driver checks, never an input.
- **Every role binding has one shape:** `name | {name, binding: perCall|static, source: table|from|fill}`. Static
  tables (Gemma 4 PLE), graph-to-graph chaining (vision tower → decoder) and generated fills are the same mechanism.
- **Graphs** declare `file` (`"<asset>:<path>"`), one of `function` / `functions` (phases such as `step` and
  `prefill` sharing states) / `functionFamily` (capacity × chunk buckets), `inputs`, `outputs` (with `rows:
  all|last|none` and `space`), `states` (`sequence[]` with a required `seqAxis`, and optional `recurrent[]`),
  `limits`, a required `compute`, and an optional `expectFrequentReshapes` and `warmup`.
- **Driver-owned blocks** at top level: `tokenizer`, `preprocess`, `image`, `decision`, `postprocess`, `labels`,
  `generation`, `calibration`. Each block belongs to the drivers that declare it in MANIFEST.md §1.
- **About 17 drivers** cover the 38 surveyed bundles, including `decoder-lm@1`, `static-chunked-lm@1`,
  `vlm-static-inject@1`, `detector-dense@1`/`detector-detr@1`, `depth-estimator@1`, `clip-joint@1`,
  `text-embedder@1`, `late-interaction-retriever@1`, `gliner-classifier@1` and `decision-lm@1`. v1 implements
  only the drivers its milestones need. The rest are reserved names with fixed contracts.

```jsonc
{ "schema": 1, "id": "qwen3.5-0.8b", "task": "text-generation", "driver": "decoder-lm@1",
  "variant": { "platform": "macos" },
  "assets": { "main": { "subtree": "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym" } },
  "graphs": { "decoder": {
      "file": "main:qwen3_5_0_8b_decode_int8hu_block32_sym.aimodel", "function": "main",
      "inputs":  { "tokens": "input_ids", "positions": "position_ids" },
      "outputs": { "logits": { "name": "logits", "rows": "last" } },
      "states":  { "sequence":  [ { "tensors": ["keyCache", "valueCache"], "seqAxis": 3 } ],
                   "recurrent": [ { "name": "convState" }, { "name": "recState" } ] },
      "limits":  { "maxContext": 4096, "staticSequence": 1 },
      "compute": "gpu", "expectFrequentReshapes": true } },
  "tokenizer": { "source": { "dir": "main:tokenizer" }, "chatTemplate": "main:tokenizer/chat_template.jinja",
                 "addBos": "template", "stop": ["<|im_end|>", "<|endoftext|>"],
                 "thinking": { "open": "<think>", "close": "</think>", "toggle": { "templateVar": "enable_thinking" } } },
  "provenance": { "source": "Qwen/Qwen3.5-0.8B", "zooRepo": "mlboydaisuke/qwen3.5-0.8B-CoreAI", "zooRevision": "1b8c020…" } }
```

**Zoo adapter.** `ZooBundleAdapter` builds a manifest from the zoo `metadata.json`, the graph descriptors, tokenizer
files and sidecars, plus a checked-in **override table** (`catalog/zoo-overrides.json`, keyed by catalog id and
variant at the pinned revision). The override table is the only place family knowledge lives. MANIFEST.md §2.9 lists
exactly which fields need an override and why. The adapter's heuristics live only there, run once per install, and
are pinned by a golden manifest per catalog entry.

**Later:** embed the manifest in our own exports through `AIModelAsset.Metadata.creatorDefinedMetadata`
(`["aether": .dictionary(…)]`). The `.aether/` sidecar stays the override.

## 5. AetherCore

- **`ModelRef`**: `{ catalogID, repo, revision, variant, assets: [(name, repo, revision, subtree, files)] }`. It is
  resolved from the catalog for the current device **before any download**: the device's `ios-<arch>/` variant
  first, else the generic JIT variant (`ios/` or `macos/`), else `unsupportedDevice`. A per-arch `.aimodelc` for
  another chip fails to load, so there is no load-time fallback across architectures. Device support is claimed only
  after a hardware gate.
- **`Catalog`**: Swift source generated from `catalog/catalog.json` by `scripts/gen-catalog`. CI fails if the two differ.
  Each entry holds `id, task, repo, revision, variants{platform[-arch]: {driver, assets{name: {repo?, revision?,
  subtree, files[{path, size, sha256}]}}, architectures?}}, license`. Assets include foreign tokenizer sources
  (Mistral-7B's tokenizer from `mistralai/…` at a pinned revision), so nothing is ever fetched unpinned at runtime. **Per-file SHA-256** comes from the HF LFS `oid`, which the gen script records. It is
  verified after download, replacing coreai-kit's size-only check.
- **`ModelStore`** (actor):
  - Root directory: `Application Support/Aether/Models/<repo>/<revision>/`. Every asset subtree of the variant
    installs there as one unit, with the generated manifest in `<revision>/.aether/`. One model is one install.
  - Installs atomically: files go into a hidden staging directory, and one rename moves the bundle into place.
  - Each file's hash is verified before it is accepted.
  - Downloads support resume through HTTP `Range` into the staging directory.
  - Concurrent requests for the same model join one download, and *every* caller gets progress (via `AsyncStream` fan-out).
  - Bundles are marked excluded from backup.
  - Offline fallback: a complete cached sibling revision is returned, but only when the caller opts in with
    `.allowStaleRevision`, and the result reports that it is stale.
- **Store safety** (ported from 0.0.x; its tests are the spec):
  - **Reader leases:** `remove` throws `inUse(resource:)` while a lease or a running install holds the location.
  - **A per-installation cross-process lock** in `.locks/`: a second process waits and then finds the bundle installed.
  - **Recovery of abandoned stages:** once, before the first install or remove, and public as
    `recoverAbandonedStages()`. Removal and replacement first rename into `.trash/`.
  - **A free-space check** for the missing bytes, which throws `capacity`.
  - **Headers:** the Hugging Face token goes to `huggingface.co` only, and app headers go only to the same host
    across redirects. An HTTPS→HTTP redirect is refused, and a 416 restarts the file once.
  - **Transports:** `HTTPTransport` gains a `download(_:progress:)` requirement, with a default built on `get`.
    `BackgroundTransport` (iOS background `URLSession`, resume data, joining after relaunch) is the second
    conformer, with one instance per session identifier and no global registry.
- **`Device`**: architecture name (`AIModel.deviceArchitectureName`, injected so tests can fake it), available memory,
  and free disk space.
- **`Manifest`**: the Codable schema, its validation, and the zoo adapter.

## 6. AetherRuntime

### 6.1 Graph

`Graph` wraps a loaded `AIModel` + `InferenceFunction`. Its declared I/O comes from `InferenceFunctionDescriptor`:
`inputNames`, `stateNames`, `outputNames`, and `*Descriptor(of:)` for shapes and scalar types.

```swift
public struct GraphOptions: Sendable {            // per-instance; replaces every setenv in coreai-kit
    public var specialization: SpecializationOptions = .default   // compute units etc. (CoreAIDelegates)
    public var cachePolicy: AIModelCache.Policy = .default
}

final class Graph {                               // non-Sendable; owned by one engine/pipeline actor
    init(url: URL, function: String, options: GraphOptions) async throws   // AIModel.specialize(contentsOf:…) + loadFunction
    let signature: GraphSignature                 // names, shapes (−1 = dynamic), scalar types
    func run(_ bindings: inout Bindings) async throws                       // inputs + states + outputViews, no allocation
}
```

- **Specialization comes from the manifest and is never left at `.default`.** In M0, a dynamic-shape decoder under
  `SpecializationOptions.default` decoded at 4 tok/s. With `.gpu` and `expectFrequentReshapes = true` it decoded at
  274 tok/s (qwen3-0.6b, M5 Max). The compute unit is the manifest's required `graphs.<g>.compute`.
  `expectFrequentReshapes` defaults to true when the unit is GPU and any input or state has a dynamic dimension.
  The adapter writes `gpu` for every class. This was measured on 11 graphs in 5 classes: a `.neuralEngine`
  preference gave GPU-identical output on 9 (most likely not honoured), the one graph that clearly moved was 6× slower,
  and CPU failed or was wrong on 6 of 11.
- **Functions and warmup:** a graph with `functions` or `functionFamily` loads one `InferenceFunction` per phase or
  bucket, lazily, and they all share one `StateSet`. Warmup only ever uses the declared static shapes, or nothing when
  `warmup: false`. It never assumes a fixed S (the Gemma 4 S=1 graph rejects S=256).
- **`TensorBuffer`** owns an `NDArray`, which is allocated once from a descriptor resolved with
  `resolvingDynamicDimensions`. It exposes typed read/write through `mutableView(as:)`. Callers reuse buffers across
  steps; `Graph.run` never allocates.
- **`Bindings`** maps role-resolved names to `TensorBuffer`s for inputs, states and outputs. It is built once per
  shape and rebuilt only when a dynamic dimension changes. An input or output can bind the leading extent of a larger
  buffer, so one capacity-sized positions buffer serves every step (verified against real Core AI in the M1 gates).
  Inputs with `binding: static` are bound once per engine. A `table` source is owned or copy-on-write mapped per its
  `storage` (`auto` = owned when the residency budget allows). A `from` source aliases the producing graph's output
  without a copy. A `fill` source is generated once.
- **`StateSet`** is a named group of state buffers that knows how to `zero()`, `snapshot()` and `restore()`. Snapshot
  is a host or Metal copy.
- Lifetimes: Core AI's `MutableViews` are `~Escapable`. That lifetime escape stays in **one** audited helper inside
  `AetherRuntime`, the same `_lifetime(immortal)` pattern the fork uses, and nowhere else.
- **Pipelined execution** (`InferenceFunction.encode(…, to: ComputeStream)`) is an internal optimization of
  `DecoderEngine`; the public `Graph` API doesn't change. Design, verified in the lab spike (token-exact over 1200
  greedy tokens at KV capacity 2048 and 4096, plus 3k→4k long context, on qwen3-0.6b and hybrid qwen3.5-0.8b):
  - **Depth 3, greedy, GPU argmax.** Encode step N, then commit a small argmax kernel on the *same*
    `MTLCommandQueue` that backs the `ComputeStream`. Its output is step N+1's `input_ids`.
  - **Token slots.** Each in-flight step owns a small (64 B) shared `MTLBuffer` bound at offset 0. **Never** a view
    into a large or page-aligned buffer. Core AI orders its GPU read after the argmax only for suballocated buffers;
    a page-aligned token-history buffer (as at capacity ≥ 4096) raced and produced wrong tokens. Slots are allocated
    at load, and the engine asserts none is page-aligned.
  - **No per-step sync.** Tokens reach the host only in the argmax completion handler, which also releases a
    depth-sized gate.
  - **The ordering is observed behaviour, not a documented contract.** A delayed-argmax canary (about 64 steps)
    runs at engine load and falls back to depth 1 if it fails. CI runs it at capacity ≥ 4096 for every bundle class.
    We also want Core AI to document the guarantee or add an explicit fence on `ComputeStream`.
  - **Depth 1** (encode + GPU argmax, one step in flight) covers CPU or non-greedy sampling, per-token inputs, and
    anything that needs the token on the host before the next `encode`.
  - **Measured on an M5 Max:** qwen3-0.6b `run()` ≈ 270, depth 1 ≈ 370, depth 3 ≈ 500–505 tok/s against the
    fork's ≈ 535 (0.90–0.94×). The hybrid's depth 3 runs at ≈ 234 against the fork's ≈ 238 (parity; host `encode`
    cost bounds it). Prefill goes through `encode` and reads only the logits rows it needs: converting every row to
    `[Float]` made `run()` prefill 8× slower.

### 6.2 DecoderEngine (the owned LLM engine)

There is one engine type for every causal decoder. Chat, VLM decoders, ASR-LLMs and the decision readout all use it.
The sketch below is the shape; the implemented API is `Sources/AetherRuntime/Decoder/DecoderEngine.swift` (M2).
Every async entry point is non-reentrant: a call that arrives during another operation is refused, not interleaved.

```swift
actor DecoderEngine {
    init(manifest: Manifest.Graph, url: URL, options: EngineOptions) async throws
    var position: Int { get }                                  // tokens resident in state
    func prefill(_ tokens: [Int32], logitsAt: LogitsSelection = .last) async throws -> Logits
    func step(_ token: Int32) async throws -> Logits           // one decode step
    func hidden(at positions: [Int]) async throws -> Tensor     // requires a `hidden` output role
    func rewind(to position: Int) throws                        // KV: drop positions beyond; recurrent: requires snapshot
    func snapshot() throws -> EngineSnapshot                    // KV up to position + recurrent-state copy
    func restore(_ s: EngineSnapshot) throws                    // validates before touching state
}
```

- **Graph contract** (`decoder-lm@1`):
  - Inputs are `tokens [1,S]` and `positions [1,P]`. `S` is dynamic, or static per phase (`staticSequence`, such
    as step 1 and prefill 64). `tokens` may carry ids ≥ vocab when `inputEncodings.tokens.extensionIds` is declared.
  - States are `states.sequence` entries (KV pairs or MLA latents) with a declared `seqAxis`, plus optional
    `recurrent` states. Dtype comes from the descriptor (fp16 or bf16).
  - The output is `logits [1,S,V]` (`rows: all`) or `[1,1,V]` (`rows: last`). Asking a `last` graph for another
    position is a `manifestMismatch`, not a silent misread. `V` is the vocab unless `space` says slots, options or scalar.
  - Static inputs belong to the engine. `snapshot()` records which image or table is bound, not its bytes.
  - This matches today's zoo exports. The M0 spike confirmed it on qwen3-0.6b: `input_ids`/`position_ids` int32
    `[1×?]`, `keyCache`/`valueCache` fp16 `[28×1×8×?×128]` (sequence on axis 3), and `logits` fp16 `[1×?×V]`.
    Positions carry the whole range `0..<position+S`.
  - On the hybrid qwen3.5-0.8b bundle (decode-only), `input_ids` is static `[1×1]`. KV covers only the 6 attention
    layers, and two **static recurrent states** (`convState`, `recState`) cover the 18 linear-attention layers. The
    `states.kv` / `states.recurrent` roles must come from the manifest, because shape heuristics misread `recState`
    as a KV cache.
- **KV capacity is derived from memory at load.** The default capacity is `min(maxContext, kvCapacityCap[platform],
  what fits the device memory budget after weights)`, and a dynamic axis grows geometrically up to it. An app may
  request a capacity; asking beyond the cap throws `capacity`. A static axis is fixed. `functionFamily` graphs move
  between capacity buckets by copying KV into the next bucket's state.
  Rewinding attention-only models is O(1), because the position counter just moves back and stale slots are
  overwritten. Hybrid and SSM models can't rewind partially, so they rewind by `restore(snapshot)`. The Generator takes
  a snapshot at the end of each prompt, which replaces the fork's `checkpointAfter`.
- **Prefill is always chunked, with at most `prefillChunk` tokens per call (default 512).** This is a correctness limit.
  In M0, a single 896+-token batch on qwen3-0.6b produced logits off by 7–16 from both chunked and token-by-token
  feeding, and at 2048 tokens it changed the greedy output. Batches of ≤ 768 tokens matched. Chunking is also faster
  (3.8k against 2.7k tok/s at 3072 tokens). The schedule, all from the manifest, never from a bundle name or env var:
  1. If a `prefill` phase exists with static S=C, feed full chunks on `prefill`, then the remainder on `step`.
     With `padTail`, a padded final chunk replaces the remainder; a 64-token tail fed one token at a time costs as
     much as 15 chunks. In M0, LFM3's pf64 prefill ran at 2.1–5.5k tok/s against 162 one at a time, within tolerance.
  2. Else, if `step` has static S=1, feed one token per call.
  3. Else, feed dynamic chunks of ≤ `prefillChunk`.

  The parity gate checks chunked prefill against token-by-token feeding at about 3k tokens for every model.
- **Sampling** runs on the CPU over `Logits` in v1: greedy, temperature, top-k, top-p and repetition penalty, all with a
  seeded RNG. A GPU sampler is an optimization behind the same interface.

### 6.3 Registry

```swift
public protocol ModelDriver: Sendable {
    static var id: DriverID { get }                  // "decoder-lm@1", "image-detector@1", …
    associatedtype Loaded: Sendable
    static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Loaded
}
```

Each task module registers its drivers when its product is linked; `Registry.default` is assembled from the linked
modules. `ModelHost.load(catalogID)` resolves the entry, checks that a driver for `entry.driver` is registered, then
installs and loads the model. A missing driver produces `AetherError.unsupported(driver:)` **before** anything downloads.

### 6.4 The generative foundation (G0)

What the image, video and speech drivers share, in `AetherRuntime/Generative/` (no driver code). It lives one layer
below every task module that needs it, beside the sampler and `MemoryBudget`, rather than in a new module.

- **Manifest blocks** (AetherCore, MANIFEST.md §5.1–§5.3): typed `schedule`, `latent` and per-graph `residency`,
  with closed vocabularies.
- **`FlowMatchSchedule`**: sigmas, timesteps and the Float Euler update, in the reference's arithmetic order, so they
  are bit-identical to 0.0.x's Krea and FLUX.2 schedules.
- **`SeededGaussianNoise`**: SplitMix64 + double Box–Muller. It is prefix-stable and the same stream 0.0.x's Krea,
  FLUX.2, H3 and Kokoro drew, so H3 reads consecutive ranges of one call.
- **`GeneratedImage`** (planar float32 `[-1, 1]`, ties-to-even RGB8, PNG) and **`GeneratedAudio`** (planar PCM, WAV).
- **Residency and measured footprints:**
  - `GraphResidency` loads graphs on demand, measures each phase and releases graphs per a `ResidencyPlan`.
  - The plan keeps everything resident if the measured peak fits, else releases per phase, else refuses before
    loading.
  - Each phase's footprint is its sampled peak and the bytes it holds at its end, read as the larger of
    `MTLDevice.currentAllocatedSize` and `phys_footprint` (0.0.x saw the footprint miss Metal memory; the H3 spike saw
    it include it, so neither alone is trusted). They go in a `FootprintLedger` keyed by model and profile, which can
    be seeded with qualified values.
  - A measurement holds only for the shapes it ran at: each distinct input shape keeps its own memory until the model
    is released (lab/h3-spike), so a profile pins its shapes.
  - Graph file sizes stand in until a phase is measured, and the plan reports `isMeasured: false`.
  - The pipeline reserves the plan's peak from AetherHost's shared `ResourceCoordinator` (ported from 0.0.7).
- **Measured, G0 gate (qwen3-0.6b, M5 Max):**
  - A graph load allocates almost nothing (+0.4 MiB for 320 MiB of files).
  - Memory arrives with the first run: +1.33 GiB peak, held while loaded, about 4× the files. So load deltas measure
    nothing, and file sizes badly underestimate.
  - A released graph's memory comes back about 5 s later (0.0.x: 1–6 s). So the per-phase peak counts each phase on
    top of what the phase before it held.
- **`EpsilonEulerSchedule`** (G1): Diffusers' Euler epsilon step from the manifest's captured tables, in Float in
  Diffusers' order.

### 6.5 AetherImage (G1, G2)

Image generation drivers, on the generative foundation. The first is **`latent-diffusion@1`** (SD-Turbo):
`LatentDiffusionPipeline` is an actor that encodes the prompt (its own `CLIPTokenizer`, since swift-transformers' BPE
lacks CLIP's `</w>`), runs the U-Net per step with the latents on the host (seeded NCHW noise × `initNoiseSigma`,
scaled input, the table's timestep, the Euler epsilon update), then the VAE decoder.

- Before each generation it plans residency from the `FootprintLedger` (keyed by model and `<width>x<height>`), fits
  the plan's peak to a shared `ResourceCoordinator` (reserving it) or a `MemoryBudget`, and refuses before loading if
  nothing fits. Graphs that stay resident are reused across generations.
- Each graph is checked against its manifest role when it loads (`SignatureValidator`); a request for a size the export
  doesn't have (`latent.sizes`) is refused before any work.
- Generations run one at a time; cancellation stops between graph calls, and any failure releases every graph.
- **Measured (gate, M5 Max, float32 512²):** phase peaks 1.25 GiB (prompt), 3.5 GiB (steps), 1.25 GiB (image); 4
  steps take under a second warm. Per phase that is about 4.8 GiB, the figure an iPhone has to fit.

The second is **`flux2-klein@1`** (FLUX.2 [klein] 4B, G2). `Flux2KleinPipeline` has the same shape (the options,
request, admission prologue and load-time contract check are shared: `ImagePipelineOptions`, `ImageRequest`), with
FLUX.2's host recipe:

- The prompt is the bundle's Qwen3 chat template (thinking off) through AetherText's `TokenizerAdapter`, right-padded
  to 512 ids with a mask; every row, padding included, conditions the denoiser.
- The transformer is three segments chained by their carries. The driver wires them from the manifest's
  `source.from` bindings, checked once at load, and binds the same arena buffers across calls: the text rows, the
  rotary tables (built on the host per `flux2.rotary`), the carries and the shared condition.
- Packed latents start from the seeded stream in packed-token order, go in as fp16, and take the float32 Euler update on
  the host with the empirical shift's μ for the generated token count. The packed tokens, in raster order, are the
  decoder's `[rows, columns, 128]`.
- macOS only in 0.1.0: on iOS the pipeline refuses (`unsupportedDevice`) before anything loads. Reference images are
  deferred.
- **Measured (gate, M5 Max, int8 512², 4 steps):** phase peaks 3.2 GiB (prompt), 5.6 GiB (steps), 1.2 GiB (image);
  about 15 s cold (loading included) and 1.4 s warm.

### 6.6 Device chains, weight banks and shape profiles (G4 step 1)

The runtime pieces H3 needs, measured in `lab/h3-spike` and built on the 0.0.x prototype's working patterns
(`Sources/AetherRuntime/Chain/`).

- **`GraphChain`:** call *i*'s output is call *i + 1*'s input with no host copy. The chain owns two slots of the
  linked tensor and alternates them (never an output over its own input), binds the head and the other slot itself,
  and refuses a call that rebinds them. On a Metal arena (`TensorArena.metal()`) every call is *encoded* on one
  `ComputeStream`, which orders consecutive calls, so the chain waits once at the end or at a `synchronize()` the
  driver places; on a host arena every call is `run()` (the unchained reference path, and the one fakes take).
  Every exit, a cancellation or an error included, waits for the encoded work before the arena is lent back.
- **`WeightBanks`:** one weights-as-inputs graph serves every block. Two banks of its weight inputs are refilled with
  `pread` from the installed bundle's files (a `WeightTable`, MANIFEST.md §5.8): no Hugging Face cache, no per-open
  re-hashing (the store verified the files at install; the driver holds the reader lease). The sync rule is
  enforced: a bank whose buffers a call since the chain's last `synchronize()` bound can't be refilled. The order is
  encode block *i*, load block *i + 1* into the other bank while the GPU runs, synchronize.
- **`ShapeProfile`:** each distinct input shape keeps its own specialization's memory until the model is released,
  so a measured footprint holds only for the shapes it was measured at. A profile names a graph role's allowed input
  shapes; a chain with a profile refuses a call outside it before anything runs, and the profile's `identity` keys its
  `FootprintLedger` entries. `Graph.distinctShapeCount` counts the shapes a graph has specialized.
- **`SerialGate`:** the runtime's one busy policy is to queue, not fail; a chain's runs take turns through a gate that
  chains sharing graphs share. No per-call byte cap: admission owns the limit, from the measured entry.
- **Measured (gate, M5 Max, the 64-wide fixture block, 6 blocks × 256 rows, float16):** chained and banked calls
  are byte-identical to the unchained path, and so are calls encoded back to back with one wait; 9.2 ms against
  49.7 ms unchained.

## 7. AetherText

- **`TokenizerAdapter`** wraps swift-transformers' `Tokenizer`. It reads special ids (pad, eos, stop markers) from the
  bundle's tokenizer config and the manifest. Nothing is hardcoded. Three rules come from measurement:
  - **BOS:** the chat template owns BOS and the runtime adds none. swift-transformers `applyChatTemplate` is
    token-identical to HF `apply_chat_template` on 11 chat bundles × 3 message shapes, and that parity script
    becomes a per-entry adapter gate.
  - **Stops:** the source model's `generation_config.json` `eos_token_id` list, unioned with the tokenizer eos and
    pinned per entry. Gemma 4 needs `<|tool_response>`, its tool-call hand-back, besides `<eos>` and `<turn|>`.
  - **Source:** a tokenizer is loaded only from a hashed, installed asset. There is never a hub fetch at runtime.
- **`PromptRenderer`** is a protocol. `JinjaChatTemplate` is the default, and it handles `enable_thinking` along with the
  "template ignores the flag" fallback. Family-specific renderers, such as Gemma's, register by `driver` or `template`
  id rather than through a catalog-id check.
- **`Generator`** is the single decode loop shared by ChatSession, the FoundationModels executor, VLM and future audio-LLMs:
  - It keeps a token mirror of the engine state and reuses the common prefix through `rewind`/`restore`.
  - It snapshots after each prompt on hybrid models.
  - It checks stop sequences, detokenizes incrementally and parses tags (`thinking`, `tool_call`), driven by
    `OutputProfile` from the manifest.
  - Constrained decoding (JSON Schema, `@Generable`) is a logits mask applied inside the same loop. It is available
    whenever the engine exposes logits, and with the owned engine it always does.
  - It emits `AsyncThrowingStream<GenerationEvent>`. Batching events for UI cadence is **not** its job; that is a
    helper in the UI layer.
- **Ported from 0.0.x and validated against its fixtures (port-language):**
  - **Sampling:** repetition penalty → constraint → temperature / top-k / top-p (no full-vocab sort) → one seeded
    draw. `SeededRandom` is xoshiro256** seeded by SplitMix64; it matches 0.0.x's draws and the 18 rows recorded from
    HF Transformers. A plain `SplitMix64` remains separate for the generative noise stream (G0).
  - **Structured output** is a `JSONSchemaConstraint` behind the `LogitsProcessor` hook
    (`begin(prompt:)` / `accept(_:)` return a `TokenConstraint`). It walks a byte trie of the vocabulary, so it needs
    a byte-level tokenizer; FoundationModels' guided generation is advertised only then.
    - **Thinking delimiters are matched by token id.** Inside a structured value, tag text is plain value text.
    - Whether the reply starts inside thinking is decided from the prompt.
    - Unenforceable schemas (`enum` or `const` on non-strings, empty `anyOf`, cycles that consume no byte) are
      refused at compile time.
    - The schema instruction goes at the end of the current user turn, so prefix reuse survives when structured
      and plain turns alternate.
  - **Text stream:** stop strings apply to response text only (not thinking or tool calls), with grapheme-safe
    hold-back. Reasoning framing drops the newlines around the tags. Special tokens contribute no text, except the
    manifest's tags.
  - **Tools:** a `ToolDialect` per manifest `toolDialect`; only `hermes-json` is implemented. A body may hold
    several calls. Arguments are kept byte for byte, so the next turn renders exactly what the model wrote. A body
    that doesn't parse is surfaced as text.
- **`ChatSession`** (actor) holds the history, configuration and a Generator, and nothing more.
- **AetherFoundationModels:**
  - `AetherLanguageModel: LanguageModel` and its executor sit on the same Generator.
  - Tool-call dialects (Hermes/ChatML, LFM3, …) are `ToolDialect` types chosen by manifest `tokenizer.toolDialect`.
    The adapter may infer this field; the runtime never probes for it.
  - Capabilities are derived from the manifest: tools if a dialect is declared, reasoning if thinking tags are, and
    guided generation always.

## 8. AetherVision and AetherEmbeddings

- **One image pipeline** (`ImagePipeline`): decode with ImageIO, apply **EXIF orientation**, convert colour space,
  resize, normalize, then lay out.
  - Resize modes are `stretch`, `letterbox` and `centerCrop`. The filter is either `vImage` (fast) or
    `pillowBicubic` / `pillowBilinear` (bit-exact parity mode, needed by the zoo-ported decision towers).
  - Every parameter comes from manifest `preprocess`.
  - Inputs are `CGImage`, `CVPixelBuffer` or a URL. The real-time path is a `PreparedInput` type split into
    `prepare` (CPU) and `run` (GPU) stages, as a type rather than a naming convention.
- **`Pipeline<Input, Output>`** has three stages: `preprocess`, `Graph.run` and `postprocess`. The graph binds by role
  (e.g. `image`, `boxes`, `scores`, `depth`, `embedding`).
  - Detector, depth, CLIP, super-resolution, embedder and reranker are each a driver plus a postprocess function.
  - Pad ids and pooling (mean or CLS) come from the manifest and tokenizer config.
- Live camera: `LiveVision` wraps any `Pipeline` with frame dropping and back-pressure. Overlay geometry, mapping
  model space to view space, is part of the result type.

## 9. AetherDecide

- **Value types:** `Decision.Question` is `.choice`, `.score` or `.noul` (yes/no). The answer is a distribution over
  options. `State` is a typed enum: `.text`, `.json(JSONValue)`, `.image`, `.audio`.
- **`Decider`** is a protocol with a `capabilities` OptionSet (`readsImages`, `readsAudio`, `maxOptions`) and
  `decide(state:questions:) async throws -> [Answer]`.
- **`LMReadoutDecider`** (v1) runs on `DecoderEngine`:
  1. Prefill the state once and snapshot.
  2. For each question, restore, prefill the question and read the answer-slot logits.
  3. Apply calibration temperature (from the catalog).

  Prompt formats are `PromptFormat` types (slot, letter-list, shared-state, …) selected by manifest, not by
  directory name.
- **The bespoke families** (Kev, Clef, D1, D1Omni, DeciderVision) are re-ported in `lab/`. They are promoted once they
  are expressed as a driver plus a head reducer over `hidden(at:)`. Python-compatible JSON/float formatting
  (`PyCompat`) and Pillow resampling are shared primitives in `AetherCore` and `AetherVision`; they are never vendored
  per family.
- **Later:** a `SystemOneWire` codec, `aether serve` (HTTP) and `aether mcp`, all built on `Decider`.

## 10. AetherHost — residency and capability

- **`ModelHost`** (actor, injectable; `.shared` by default) is the single place models are loaded. Every entry point,
  including direct `ChatSession(catalog:)`, goes through it.
- **Residency:**
  - Each loaded model is recorded with *measured* bytes, not catalog guesses. The measure is the larger of
    Metal's `currentAllocatedSize` and `phys_footprint` after the model has run, since a load alone allocates almost
    nothing and neither counter alone is reliable (§6.4).
  - Eviction is LRU with pins (a model in use is pinned).
  - The budget is `os_proc_available_memory()` on iOS and a configurable fraction of RAM on macOS.
  - Idle models are dropped under memory pressure.
- **Lifecycle** (ported from 0.0.x, validated on an iPhone under real pressure). The entry point is
  `withModel(_:policy:load:_:)`, keyed by `ModelRef`; it goes through the Registry once M6 wires it in.
  - **Idle unload** after 60 s on iOS and 5 min on the Mac, with one release per pressure notice.
  - **A load gate under critical pressure** that throws `memoryPressure`.
  - **Easing** once 2× the recovered memory is free, or after 60 s, because iOS never sends "normal" after a
    critical notice.
  - **Unloads requested during a request** wait until it ends.
  - Pressure arrives through a `MemoryPressureMonitor` actor (`.system` or `.manual`).
- **GPU lease** (ported; wire-compatible with Atelier's `SharedGPULock`): `SharedGPULease`, `GPUAdmission` and
  `FairInferenceQueue`. These are `Sendable` classes over `Mutex`, not actors, because a grant must release
  synchronously and safely in `deinit`. The signal handling uses one static `Mutex`, which is forced because signal
  dispositions are process-wide. Real-model gates and the generative pipelines hold a `GPUAdmission`.
- **Capability and prefetch:**
  - `capability(for:)` returns `.ready`, `.needsDownload(bytes)`, `.insufficientStorage`, `.unsupportedDevice` or
    `.unsupportedDriver`.
  - `prepare(_:)` prefetches.
  - An `aether-doctor` CLI totals an app's downloads. These carry over from coreai-kit, the ideas worth keeping.

## 11. Concurrency and errors

- **Ownership:** each loaded model lives inside one actor (an engine or pipeline actor). `Graph`, `TensorBuffer` and
  `StateSet` are non-`Sendable` and never leave it. Public results are `Sendable` value types.
- **Concurrent use:** one model handles one request at a time, enforced by the owning actor plus an internal FIFO
  (the `ModelHost` request queue). There are no per-type mutex variants.
- **Streaming and work:** every stream is an `AsyncThrowingStream`, with cancellation that propagates through
  `onTermination`. Prewarm is `async`, so no semaphores block on tasks.
- **Errors:** one error domain, `AetherError`, with structured cases (`unsupported(driver:)`, `inUse(resource:)`,
  `admission(AdmissionFailure)`, `memoryPressure`,
  `manifestMismatch(field:expected:found:)`, `download(…)`, `integrity(file:)`, `capacity(…)`, `runtime(underlying:)`).
  Family-specific detail goes in an associated payload, not in a separate public enum per family.
- **Debug data:** hashes, per-stage timings and gate artifacts go behind `@_spi(Diagnostics)`.

## 12. Testing

| Layer | How |
|---|---|
| Core | Pure unit tests: manifest decode/validate, catalog generation, ModelRef resolution with a faked `Device`, store install, resume and hash failure against a local HTTP fixture. |
| Runtime | A `GraphBackend` protocol lets the engine run against a **fake decoder**, a deterministic toy LM in Swift. Rewind, snapshot, chunking and KV growth are tested hermetically. |
| Text | Golden prompt renders per chat template, a stop/tag-parser corpus, and constrained-decoding grammar tests on the fake engine. |
| Parity gates | Opt-in (`AETHER_GATES=1`) on a real Mac: per-catalog-entry token-exact greedy output against a recorded reference, `cos ≥ 0.999` for embeddings and vision, plus a two-turn rewind check on one hybrid model. |
| Adapter | Golden `aether.json` for every catalog entry, generated by the adapter from the pinned zoo bundle metadata. |
| Public API | `scripts/check-api.sh` (in `ci.sh`) dumps each module `scripts/api-freeze.json` names with `swift-api-digester` and compares it with the committed `api/` baseline. Additions pass, and so does a case added to a `@nonexhaustive` enum; anything else breaks, and a compatible change still has to be recorded (`--update`). Report-only until 0.1.0; `"enforce": true` makes it fail. The placeholder modules join when they have API. |

## 13. Relationship with `lab/`

`lab/` is a separate repo with no code links. Experiments there answer questions the spec depends on. Promotion means
**re-implementing** the result in Aether-Kit against its contracts, never copying the code across. An experiment is
ready to promote when:

1. its graph contract is expressible as a manifest plus a driver;
2. it uses only Aether primitives (Graph, DecoderEngine, ImagePipeline, PyCompat), with nothing vendored;
3. it has a recorded parity reference that becomes an Aether gate.

## 14. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| **M0** | `lab/engine-spike`: the owned decode loop on raw Core AI | qwen3-0.6b greedy output token-exact against a reference, two-turn rewind correct, logits readable at a chosen position, tok/s recorded (Mac) |
| M1 | AetherCore + AetherRuntime (Graph, TensorBuffer, StateSet, Registry) | Hermetic tests green; schema v1 per MANIFEST.md with the override table; adapter goldens for the chat entries; real-graph gates (signature + greedy parity) on one attention and one hybrid model |
| M2 | DecoderEngine + Generator + ChatSession | Parity gate on 3 chat models including one hybrid; ≥1k-token chunked-prefill parity; depth-3 pipelined decode ≥ 480 tok/s on qwen3-0.6b (Mac), token-exact, with the load-time canary |
| M3 | AetherFoundationModels | Two-turn `LanguageModelSession` recall, a tool call (Hermes), and `@Generable` guided output |
| M4 | AetherVision + AetherEmbeddings | Detector, depth, CLIP and embedder parity gates |
| M5 | AetherDecide (LM readout) | Matches recorded distributions on a decision fixture |
| M6 | AetherHost + `Aether` ops + doctor | Capability/prepare/residency tests; ops cookbook |

## 15. Risks and unknowns (each should become a lab question)

- **Deep pipelining relies on undocumented ordering.** It's solved in lab (§6.1), but correctness depends on
  Core AI ordering its read of a suballocated buffer after a same-queue kernel. Mitigation: the load-time canary
  with fallback to depth 1. There's also a remaining 6–10 % qwen3 gap to the fork, which is GPU-bound (2.65 against
  2.40 ms per token); the next step there is a Metal System Trace of both engines. The hybrid is bounded by about
  4.1 ms of host-side `encode` per step, which is a Core AI question.
- **Hybrid prefill.** Decode-only hybrids take one token per call (164 tok/s). LFM3 shows that a `prefill` function
  fixes this (2.1–5.5k tok/s). Qwen3.5 has no prefill bundle that loads on macOS 27.2, so it needs a pf64 export
  sharing the decoder's four states.
- **Rewind on non-standard sequence states.** MLA latents (Youtu) and shared-KV layers (Gemma 4) are assumed to rewind
  like KV until a per-model two-turn gate confirms it.
- **The per-architecture AOT `.aimodelc`** loading and `AIModelCache` specialization on iPhone are untested. Only an
  h19p graph failing on an h17c Mac was observed.
- **iOS memory and power:** whether memory-derived capacity and the residency policy hold for models of 2 GB and up,
  and whether the Neural Engine wins on power on iPhone, even though it showed no speed win on the Mac.
