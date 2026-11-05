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

## 3. Module layout

```
Aether (umbrella: one-line task ops; re-exports the task modules below)
 ├── AetherText         tokenizer adapter · prompt rendering · Generator · ChatSession · constrained decoding
 │    └── AetherFoundationModels   LanguageModel/Executor bridge (separate product; FoundationModels link)
 ├── AetherVision       image I/O + preprocessing · detection · depth · CLIP · super-resolution
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

Products: `AetherText`, `AetherFoundationModels`, `AetherVision`, `AetherEmbeddings`, `AetherDecide`, `Aether`.
`AetherHost`, `AetherRuntime` and `AetherCore` are reachable through any of them.

## 4. The manifest (`aether.json`)

The manifest sits at the bundle root, next to the `.aimodel` / `.aimodelc` graphs. It is versioned (`"schema": 1`).
Unknown fields are ignored. An unknown `driver` is an error ("unsupported"), never a fallback.

```jsonc
{
  "schema": 1,
  "id": "qwen3-0.6b",                       // catalog id this bundle belongs to
  "task": "text-generation",                // what it does (task protocol it can serve)
  "driver": "decoder-lm@1",                 // who loads it (registry key, versioned)
  "graphs": {
    "decoder": {
      "file": "qwen3_0_6b_4bit_dynamic.aimodel",   // or per-arch: {"h18p": "x.h18p.aimodelc", "*": "x.aimodel"}
      "function": "main",
      "inputs":  { "tokens": "input_ids", "positions": "position_ids" },   // role -> graph name
      "outputs": { "logits": "logits" },
      "states":  { "kv": ["k_cache", "v_cache"], "recurrent": [] },
      "limits":  { "maxContext": 8192, "prefillChunk": null, "staticSequence": null }
    }
  },
  "tokenizer": { "dir": "tokenizer", "chatTemplate": "tokenizer/chat_template.jinja",
                 "stop": ["<|im_end|>"], "thinking": { "open": "<think>", "close": "</think>" } },
  "preprocess": null,                        // vision: { resize, mode, mean, std, layout, orientation }
  "precision": { "weights": "int4", "activations": "fp16" },
  "provenance": { "source": "Qwen/Qwen3-0.6B", "gate": "token-exact vs fp16 ref", "zooRevision": "943eb6a…" }
}
```

Roles are a **closed set per driver**. For example, `decoder-lm@1` requires `tokens`, `positions`, `logits` and at
least one `kv` pair, and accepts optional `recurrent` states and a `hidden` output. The driver validates the manifest
against the graph's `InferenceFunctionDescriptor` at load time: names exist, ranks match, and the scalar types are
supported. A mismatch fails the load and names the offending field.

**Zoo adapter.** `ZooBundleAdapter` reads the zoo's `metadata.json` (`kind: "llm"`, `assets.main`,
`language.function_map`, `vocab_size`, `max_context_length`) and the graph's descriptor, then emits an `aether.json`.
Its heuristics, such as "the first input is tokens", live **only** in the adapter and are covered by golden tests for
each catalog entry. The generated manifest is written next to the installed bundle, so the heuristic runs once per
install and never at inference time.

**Later:** embed the manifest in the `.aimodel` itself through `AIModelAsset.Metadata.creatorDefinedMetadata`
(`["aether": .dictionary(…)]`) for bundles we export ourselves. The sidecar stays the override.

## 5. AetherCore

- **`ModelRef`**: `{ catalogID, repo, revision, subtree }`. It is resolved from the catalog for the current device:
  `ios-<arch>/` first when the entry lists the device's architecture, then `ios/` or `macos/`.
- **`Catalog`**: Swift source generated from `catalog/catalog.json` by `scripts/gen-catalog`. CI fails if the two differ.
  Each entry holds `id, task, driver, repo, revision, variants{platform: {subtree, bytes, files[{path, size, sha256}],
  architectures?}}, license`. **Per-file SHA-256** comes from the HF LFS `oid`, which the gen script records. It is
  verified after download, replacing coreai-kit's size-only check.
- **`ModelStore`** (actor):
  - Root directory: `Application Support/Aether/Models/<repo>/<revision>/<subtree>`.
  - Installs atomically: files go into a hidden staging directory, and one rename moves the bundle into place.
  - Each file's hash is verified before it is accepted.
  - Downloads support resume through HTTP `Range` into the staging directory.
  - Concurrent requests for the same model join one download, and *every* caller gets progress (via `AsyncStream` fan-out).
  - Bundles are marked excluded from backup.
  - Offline fallback: a complete cached sibling revision is returned, but only when the caller opts in with
    `.allowStaleRevision`, and the result reports that it is stale.
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
  274 tok/s (qwen3-0.6b, M5 Max). The driver derives the options from the graph's shape kind: dynamic → GPU plus the
  reshape hint; static chunks → Neural Engine.
- **`TensorBuffer`** owns an `NDArray`, which is allocated once from a descriptor resolved with
  `resolvingDynamicDimensions`. It exposes typed read/write through `mutableView(as:)`. Callers reuse buffers across
  steps; `Graph.run` never allocates.
- **`Bindings`** maps role-resolved names to `TensorBuffer`s for inputs, states and outputs. It is built once per
  shape and rebuilt only when a dynamic dimension changes.
- **`StateSet`** is a named group of state buffers that knows how to `zero()`, `snapshot()` and `restore()`. Snapshot
  is a host or Metal copy.
- Lifetimes: Core AI's `MutableViews` are `~Escapable`. That lifetime escape stays in **one** audited helper inside
  `AetherRuntime`, the same `_lifetime(immortal)` pattern the fork uses, and nowhere else.
- **Pipelined execution** (`InferenceFunction.encode(…, to: ComputeStream)`) is an internal optimization of
  `DecoderEngine`, adopted only after the `lab/` spike measures the gain. The public `Graph` API doesn't change.

### 6.2 DecoderEngine (the owned LLM engine)

There is one engine type for every causal decoder. Chat, VLM decoders, ASR-LLMs and the decision readout all use it.

```swift
actor DecoderEngine {
    init(manifest: Manifest.Graph, url: URL, options: EngineOptions) async throws
    var position: Int { get }                                  // tokens resident in state
    func prefill(_ tokens: [Int32], logitsAt: LogitsSelection = .last) async throws -> Logits
    func step(_ token: Int32) async throws -> Logits           // one decode step
    func hidden(at positions: [Int]) async throws -> Tensor     // requires a `hidden` output role
    func rewind(to position: Int) throws                        // KV: drop positions beyond; recurrent: requires snapshot
    func snapshot() throws -> EngineSnapshot                    // KV position + recurrent-state copy
    func restore(_ s: EngineSnapshot) throws
}
```

- **Graph contract** (`decoder-lm@1`):
  - Inputs are `tokens [1,S]` and `positions [1,P]`.
  - States are the KV pairs, with the sequence on a declared axis, plus optional recurrent states.
  - The output is `logits [1,S,V]`.
  - This matches today's zoo exports. The M0 spike confirmed it on qwen3-0.6b: `input_ids`/`position_ids` int32
    `[1×?]`, `keyCache`/`valueCache` fp16 `[28×1×8×?×128]` (sequence on axis 3), and `logits` fp16 `[1×?×V]`.
    Positions carry the whole range `0..<position+S`.
  - On the hybrid qwen3.5-0.8b bundle (decode-only), `input_ids` is static `[1×1]`. KV covers only the 6 attention
    layers, and two **static recurrent states** (`convState`, `recState`) cover the 18 linear-attention layers. The
    `states.kv` / `states.recurrent` roles must come from the manifest, because shape heuristics misread `recState`
    as a KV cache.
- **KV capacity:** a dynamic sequence axis grows geometrically up to `maxContext`, and a static axis is fixed.
  Rewinding attention-only models is O(1), because the position counter just moves back and stale slots are
  overwritten. Hybrid and SSM models can't rewind partially, so they rewind by `restore(snapshot)`. The Generator takes
  a snapshot at the end of each prompt, which replaces the fork's `checkpointAfter`.
- **Prefill is always chunked, with at most `prefillChunk` tokens per call (default 512).** This is a correctness limit.
  In M0, a single 896+-token batch on qwen3-0.6b produced logits off by 7–16 from both chunked and token-by-token
  feeding, and at 2048 tokens it changed the greedy output. Batches of ≤ 768 tokens matched. Chunking is also faster
  (3.8k against 2.7k tok/s at 3072 tokens). Graphs with `staticSequence == 1` feed one token at a time. All of these
  values come from the manifest, never from the bundle name or an env var. The parity gate checks chunked prefill
  against token-by-token feeding at about 3k tokens for every model.
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

## 7. AetherText

- **`TokenizerAdapter`** wraps swift-transformers' `Tokenizer`. It reads special ids (pad, eos, stop markers) from the
  bundle's tokenizer config and the manifest. Nothing is hardcoded.
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
  - Each loaded model is recorded with *measured* bytes: the phys-footprint delta across the load, not catalog guesses.
  - Eviction is LRU with pins (a model in use is pinned).
  - The budget is `os_proc_available_memory()` on iOS and a configurable fraction of RAM on macOS.
  - Idle models are dropped under memory pressure.
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
- **Errors:** one error domain, `AetherError`, with structured cases (`unsupported(driver:)`,
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
| M1 | AetherCore + AetherRuntime (Graph, TensorBuffer, StateSet, Registry) | Hermetic tests green; zoo adapter goldens for the chat entries |
| M2 | DecoderEngine + Generator + ChatSession | Parity gate on 3 chat models including one hybrid; tok/s within 10% of the M0 spike |
| M3 | AetherFoundationModels | Two-turn `LanguageModelSession` recall, a tool call (Hermes), and `@Generable` guided output |
| M4 | AetherVision + AetherEmbeddings | Detector, depth, CLIP and embedder parity gates |
| M5 | AetherDecide (LM readout) | Matches recorded distributions on a decision fixture |
| M6 | AetherHost + `Aether` ops + doctor | Capability/prepare/residency tests; ops cookbook |

## 15. Risks and unknowns (each should become a lab question)

- **Performance without the fork's pipelined engine.** The fork's `CoreAIPipelinedEngine` (2.6k lines) uses
  `ComputeStream` encode and GPU sampling. Our sequential loop may be noticeably slower. M0 measures the gap, and a
  follow-up spike measures `encode(to:)`.
- **The hybrid/SSM recurrent-state contract.** Confirmed on Qwen3.5: snapshot/restore is bit-exact, and all 35 MB of
  state copies in 4 ms. Still to check: LFM2.5 and Granite 4. The open cost is **prefill on decode-only hybrid
  bundles**: 164 tok/s, because they take one token per call. Supporting the zoo's chunked "prefill" function, or
  exporting a dynamic-S prefill graph, is the M2 performance item.
- **No independent token-exact reference yet.** M0 criteria 2 and 6 (parity and speed against the fork's
  `llm-runner`) are still open.
- **The per-architecture AOT `.aimodelc`** selection and `AIModelCache` specialization behaviour on iPhone.
- **iOS memory:** whether the measured-footprint residency policy holds for models of 2 GB and up.
- **Nonescapable `MutableViews` lifetimes** across `await`: the single audited escape helper must be proven sound
  under Swift 6.4 strict concurrency.
