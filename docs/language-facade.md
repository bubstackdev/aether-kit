# AetherLanguage text facade

Status: design approved, 2026-10-27, with the decisions below ([Decided](#decided)); implementation in progress. This is the third piece of the 0.0.3 app-facing language
track ([roadmap](roadmap-0.0.3.md#parallel-track-an-app-facing-language-product)). It builds on the
[runtime seam](language-runtime-seam.md) (`AetherLanguageRuntime`) and on `AetherLanguageTokenizers`. Type names are
illustrative until the facade has run in a real app (see [architecture](architecture.md)).

## The finding that shapes this design

RenaML and Face Hugger use FoundationModels today: `LanguageModelSession`, `respond(to:)`, `streamResponse(to:)` and
`@Generable`. On macOS 27 and iOS 27, FoundationModels has a public `LanguageModel` protocol and
`LanguageModelSession(model: some LanguageModel, …)`, so an app can run its existing session code on a model that isn't
Apple's. A model supplies a `LanguageModelExecutor`, which receives a `LanguageModelExecutorGenerationRequest`:
- the whole `Transcript`;
- the `GenerationSchema` when the app asked for a `@Generable` type;
- the `GenerationOptions` (sampling, temperature, maximum tokens);
- `ContextOptions` (whether to put the schema in the prompt, and a reasoning level).

It streams response and reasoning text back through a `LanguageModelExecutorGenerationChannel`.

A scratch probe confirmed the path end to end on this machine ([record](evidence/foundation-models-custom-model-probe-2026-10-27.txt)). It used a fake executor that streams canned text, so no
model and no GPU were involved:
- `respond(to:options:)` returned the streamed text.
- The executor saw the instructions and prompt entries, and the sampling options (`randomTopK(40, seed: 7)`, 32
  maximum tokens).
- For `respond(to:generating: Look.self)`, the executor received the `@Generable` type's schema as JSON Schema: `title`,
  `properties` with `@Guide` descriptions, `required`, `additionalProperties: false`, and an `x-order` key order.
- When the executor streamed `{"look": "silver braid", "count": 3}`, FoundationModels decoded it into `Look`.
- `streamResponse(to:generating:)` produced six partially generated snapshots as the JSON arrived.

So on OS 27, FoundationModels already owns what the apps use most: `@Generable`, its schema, partial snapshots, the
transcript and the session API. What it needs from a model is the part Kit is good at: rendering the transcript with the
model's own template, decoding with the requested sampling, and constraining the output to the schema. Kit's Core AI
backend only runs on OS 27 anyway.

## Decisions

1. **Two app-facing surfaces over one engine.**
   - **`AetherLanguage`** is Kit's own facade: a session with `respond` and `streamResponse`, a transcript, options and
     errors. It links on macOS 14 and iOS 17, and an older-OS backend (the MLX spike) can sit behind it.
   - **`AetherLanguageFoundationModels`** is a small adapter target. It makes `AetherLanguageModel` conform to
     FoundationModels' `LanguageModel` on OS 27, so an app switches by changing one line:
     `LanguageModelSession(model: aetherModel, instructions: …)`.
   - Both drive the same engine, which owns the tokenizer, template, stop handling, reasoning, sampling and streaming.
2. **The facade mirrors FoundationModels' shape, not its types.**
   - It uses the same verbs (`respond(to:options:)`, `streamResponse(to:options:)`), the same snapshot streaming, and
     instructions plus a transcript. Its options use the same names as `GenerationOptions` where the meaning is the same.
   - It doesn't import FoundationModels, which is absent on macOS 14 and 15.
   - It doesn't copy what doesn't fit a local model with a token-level runtime: tools, guardrails, prompt builders and
     availability reasons about Apple Intelligence. These are covered below.
3. **Greedy by default.** Apps opt into sampling per request, as the user decided. Through the adapter,
   FoundationModels' `.greedy`, `.random(top:seed:)` and `.random(probabilityThreshold:seed:)` map onto
   `LanguageSampling`. A request that gives no sampling mode is greedy.
4. **Structured output is one constraint shape for both surfaces:** a JSON-schema `LanguageLogitsProcessor` built from a
   JSON Schema and the tokenizer's per-token bytes.
   - Through the adapter, FoundationModels' `@Generable` supplies the schema and does the decoding, so Kit needs no macro
     for OS 27 apps.
   - The native facade's `respond(to:generating:)` takes a Kit `Decodable` type with a schema. Whether Kit ships its own
     `@Generable`-style macro (and the swift-syntax dependency) in 0.0.3 is a question for the user (see
     [Questions](#questions-for-the-user-and-the-lead)).
5. **One per-token byte table drives both UTF-8-safe streaming and structured output.** `AetherLanguageTokenizers` gains
   the raw bytes of every token, and of byte-level BPE tokens in particular. The streaming decoder emits only complete
   UTF-8 scalars, and the JSON constraint walks bytes.
6. **Each request renders the whole transcript.** The adapter has no choice, because FoundationModels hands the executor
   the full transcript each turn. The native session renders it the same way. The bucketed backend resets its cache per
   request, so every turn prefills the whole conversation. Prefix reuse is a later backend option (see the seam's
   [next needs](language-runtime-seam.md#what-the-facade-will-need-next)) and doesn't change this API.
7. **The Core AI backend is chosen at runtime behind `#available(macOS 27, iOS 27, *)`.** Below that, the model reports
   `.unavailable(.unsupportedOS)` until another backend exists. This needs the deployment-floor work described in
   [The floor](#the-floor).

## Where the facade mirrors FoundationModels, and where it doesn't

| FoundationModels | `AetherLanguage` | Why |
| --- | --- | --- |
| `SystemLanguageModel.default`, `.availability` (`.available`, `.unavailable(reason)`) | `AetherLanguageModel(bundle:)`, `.availability` (`.available`, `.unavailable(reason)`) | Same pattern; Kit's reasons are about the OS, the device and the bundle |
| `LanguageModelSession(model:instructions:)`, `init(model:transcript:)` | `AetherLanguageSession(model:instructions:)`, `init(model:transcript:)` | Same |
| `respond(to:options:) -> Response<String>` | Same, `Response` with `content`, `reasoning`, `usage`, `stopReason`, `seed` | Adds what a local, seeded model can report |
| `streamResponse(to:options:) -> ResponseStream`, snapshots with cumulative `content` | Same, plus each snapshot's `delta` since the snapshot the consumer last saw | Lossless deltas even when snapshots coalesce |
| `respond(to:generating:)` with `@Generable` | `respond(to:generating:)` with a Kit generable `Decodable` type | Same verb; the type's schema comes from Kit, not FoundationModels |
| `GenerationOptions(samplingMode:temperature:maximumResponseTokens:)` | `AetherGenerationOptions` with the same three names, plus `stopSequences`, `repetitionPenalty`, `seed` and `reasoning` | Same names where the meaning is the same |
| `prewarm(promptPrefix:)`, `isResponding`, `transcript` | `prewarm()`, `isResponding`, `transcript` | Same |
| `Transcript` with instructions, prompt, response and reasoning entries | `AetherTranscript` with the same four kinds of entry | No tool entries |
| Tools, guardrails, `@PromptBuilder`, Private Cloud Compute | None | Not local token-level work. Tools could come later, as a constrained tool-call format |
| `LanguageModelSession` is an `Observable` class | `AetherLanguageSession` is an actor | Kit's sessions are actors; an `@Observable` main-actor wrapper is a small later convenience |

## What the engine owns

The engine is internal to `AetherLanguage`, and both surfaces call it. Per request it renders, generates through a
`LanguageBackend`, and decodes.

**Tokenizer and chat template.** `LanguageTokenizer.open(bundle:)` loads the registered tokenizer and the model's own
template.
- The transcript renders as `LanguageChatMessage`s: instructions as the system message, prompts as user messages,
  responses as assistant messages.
- Earlier turns' reasoning is not rendered back. Qwen3's template drops earlier `<think>` blocks itself; the engine never
  sends them.
- Template exactness is verified for Qwen3-family templates (as today). Other templates render unverified, and the model
  reports that in its description.

**Stop tokens and stop strings.**
- Stop tokens are the registration's `stopTokenIDs`, which the backend already honours.
- Stop strings (`stopSequences`) are the engine's. It matches them against decoded text as it streams. When a match
  completes, the delivery returns `.stop`, and the reply ends before the stop string, which is not included.
- Text that might be the start of a stop string is held back from deltas until it either completes or can't match, so a
  stop string never flashes into the UI.
- The cost is the one decode step already running when the stop is seen (see the seam).

**Reasoning.**
- The registration's `supportsReasoning` says whether the model has a thinking mode. Qwen3 hybrids do; the 2507 instruct
  writer doesn't.
- `reasoning: .off` (the default) renders the template with thinking off. A hybrid Qwen3 then gets its empty think block,
  and the model answers directly. `.on` renders with thinking on.
- The engine splits the token stream at the reasoning delimiters, `<think>` and `</think>` (Qwen3's IDs 151667 and
  151668, which are added tokens, not special ones). Tokens in between become `Response.reasoning`, and the rest is
  `content`. The delimiters themselves appear in neither.
- In a stream, snapshots carry both fields, so an app can show or hide the reasoning.
- Asking for reasoning from a model without it is an error at request time.
- The delimiters come from a small per-template table keyed by the registration, not from string matching on output.

**Sampling.** `AetherGenerationOptions` maps onto `LanguageSampling` and `LanguageGenerationRequest`:

| Option | `LanguageSampling` | Note |
| --- | --- | --- |
| no `samplingMode` | `.greedy` | The default: token-identical to the runtime's greedy |
| `.greedy` | `.greedy` | |
| `.random(top: k, seed:)` | `.random(temperature: t, topK: k, seed:)` | `t` is `temperature`, or 1 when not given |
| `.random(probabilityThreshold: p, seed:)` | `.random(temperature: t, topP: p, seed:)` | |
| `temperature` without a random mode | ignored | As FoundationModels ignores temperature under greedy |
| `repetitionPenalty` | `repetitionPenalty` | Transformers' semantics, off by default |
| `maximumResponseTokens` | `maximumTokens` | Default: what's left of the registered context, capped at 4,096 |

`Response.seed` reports the seed used, so any sampled reply can be replayed.

**UTF-8-safe streaming.** One token can be part of a multi-byte character. Decoding a growing prefix also changes text it
already returned, because swift-transformers emits U+FFFD for an incomplete byte sequence. So the engine doesn't
re-decode:
- **Bytes, not text.** It appends each token's bytes, from the byte table, to a pending buffer, and moves out only the
  complete UTF-8 scalars at the front. A delta therefore never contains a partial scalar or a replacement character
  that later turns into text.
- **No grapheme surprises.** A delta never ends inside a grapheme cluster that the next scalar could extend. Combining
  marks, emoji ZWJ sequences, flags and Hangul jamo are held back until the next scalar shows the cluster has ended, or
  the reply ends. So the cumulative `content` only ever grows by whole characters, and its existing characters never
  change.
- **Special tokens** (for example `<|im_end|>`) contribute no text, as `decode(skipSpecialTokens:)` does today.
- **Reconciled at the end.** When the reply ends, the engine compares the cumulative text with the tokenizer's own
  `decode` of the whole reply. CPU and native tests require them to be equal. If they ever differ in the field (a
  tokenizer whose decoder the byte table doesn't model), `Response.content` is the tokenizer's decode, and the
  difference is reported in diagnostics.
- **Other tokenizers.** Tokenizers that aren't byte-level BPE (SentencePiece with byte fallback) need their own byte
  table. Until one is added, the engine falls back to Text Generation Inference's prefix-offset decoding, which is
  slower and still holds back incomplete characters.

**Cancellation.**
- Cancelling the task that called `respond` stops at the next decode step. The backend joins its native work before the
  error is thrown (the runtime's contract), and the error is `CancellationError`.
- A `ResponseStream` that the consumer abandons (stops iterating, or drops) cancels the task producing it, with the same
  joining. A stream is single-pass and consumed once, as FoundationModels' is.
- A cancelled or failed turn is not added to the transcript, the equivalent of FoundationModels' `.revertTranscript`.

**Multi-turn transcript.**
- A session holds an `AetherTranscript`. Each `respond` renders it plus the new prompt, and when the reply succeeds,
  appends the prompt and the response (and the reasoning, kept for the app but not rendered back).
- One request at a time; overlap throws `busy`, as FoundationModels' `concurrentRequests` does.
- Before generating, the engine checks the rendered prompt plus `maximumResponseTokens` against the registered context.
  If it doesn't fit, it throws `contextWindowExceeded(promptTokens:, limit:)` and doesn't trim the transcript itself.
  Trimming policy belongs to the app, as with FoundationModels' context errors.
- An app can build a session from a saved transcript. `AetherTranscript` is `Codable`.

## API sketch

Illustrative, not frozen:

```swift
import AetherLanguage

let model = AetherLanguageModel(bundle: installedBundleURL)   // cheap: reads language-model.json, checks the OS
switch model.availability {
case .available: break
case .unavailable(let reason): showUnavailable(reason)        // .unsupportedOS, .bundleMissing, .bundleInvalid(…), …
}

let session = AetherLanguageSession(model: model, instructions: "You describe fictional adult characters.")
let reply = try await session.respond(to: "Describe her look in one line.")
print(reply.content)

for try await snapshot in session.streamResponse(to: "And her coat?",
                                                 options: .init(samplingMode: .random(top: 40, seed: 7))) {
    label.text = snapshot.content          // cumulative, whole characters only
    log(snapshot.delta)                    // what's new since the snapshot this loop last saw
}
```

```swift
public struct AetherLanguageModel: Sendable {
    public init(bundle: URL, platform: LanguageModelManifest.Platform = .current)
    public var availability: Availability { get }          // no hashing, no loading
    public var supportsReasoning: Bool { get }
    public var contextLength: Int { get }
    public func prewarm() async throws                      // verify and load now, instead of on the first request
    public func unload() async
}

public actor AetherLanguageSession {
    public init(model: AetherLanguageModel, instructions: String? = nil)
    public init(model: AetherLanguageModel, transcript: AetherTranscript)
    public var transcript: AetherTranscript { get }
    public var isResponding: Bool { get }
    public func respond(to prompt: String, options: AetherGenerationOptions = .init()) async throws -> Response<String>
    public nonisolated func streamResponse(to prompt: String, options: AetherGenerationOptions = .init())
        -> ResponseStream<String>
    // The structured-output piece adds respond(to:generating:) and streamResponse(to:generating:).
}
```

One model value is shared by its sessions. It loads the backend and tokenizer once, lazily on the first request or on
`prewarm()`, and keeps them until `unload()`.
- **Requests queue.** The model's backend admits one request at a time, as the runtime does. Two sessions on one model
  queue their requests in the engine; each session itself still refuses overlap with `busy`.
- **Idle unload and memory pressure are 0.0.4's managed lifecycle.**

## The floor

The facade must link and launch on macOS 14 and iOS 17, and report "unsupported OS" there. Its dependencies, as they
stand:

| Target | Floor today | What it needs |
| --- | --- | --- |
| `AetherLanguageRuntime` | macOS 14 / iOS 17 | Nothing; typechecked there in CI |
| `AetherKit` | macOS 15 / iOS 18 | `Mutex` from `Synchronization` in `GPUAdmission`, `SharedGPULease` and `ResourceCoordinator` (eight uses). A macOS 14 typecheck of the target fails on `Mutex` and `withLock` and on nothing else |
| `AetherLanguageTokenizers` | `AetherKit`'s, and swift-transformers 1.1.0 (macOS 13 / iOS 16) | Only `AetherKit` |
| `AetherLanguageCoreAI`, `AetherCoreAI` | macOS 27 / iOS 27 | Availability annotations, and `CoreAI` weak-linked |
| FoundationModels (adapter only) | macOS 26 / iOS 26; `LanguageModel` is 27 | A separate target that apps on older OSes don't link, or a weak link |

**What has to move or be gated:**
1. **`AetherKit`'s locks.** Replace `Mutex` with `OSAllocatedUnfairLock` (macOS 13, iOS 16). It has the same
   `withLock` shape and needs no availability annotations, which would otherwise spread through every caller.
   `LanguageModelManifest` (CryptoKit) and `AetherError` are already fine. Nothing needs to move: the tokenizer target
   keeps depending on `AetherKit`.
2. **Package `platforms`.** These drop to `.macOS(.v14), .iOS(.v17)`. Every target that touches Core AI or OS 27 APIs then
   needs `@available(macOS 27, iOS 27, *)` on its public declarations. That covers `AetherCoreAI` and the `*CoreAI`
   targets, plus the probe and the native test targets, which can stay 27-only by being annotated as a whole.
3. **`CoreAI` weak-linked.** It's `linkedFramework("CoreAI")` today. The floor work decides how: `-weak_framework` via
   `linkerSettings: [.unsafeFlags]` is not allowed for a remote dependency, so this needs SwiftPM's own weak linking or a
   build-setting route. Of everything here, it's the one item the facade can't work around.

**The facade asks this of the floor work** (it doesn't do it itself):
- package platforms at macOS 14 and iOS 17;
- availability annotations on the Core AI targets;
- a weak `CoreAI` link;
- a launch check: an app linking `AetherLanguage` that runs on a macOS 14 machine or VM and reports `.unsupportedOS`.

The `AetherKit` lock replacement is small and blocks the facade's own macOS 14 typecheck. It could be the facade's first
commit or part of the floor work (a question for the lead).

**Choosing the backend at runtime.** `AetherLanguage` depends on `AetherLanguageCoreAI` in SwiftPM, but calls it only
inside `if #available(macOS 27, iOS 27, *)`. Otherwise `availability` is `.unavailable(.unsupportedOS)`, and requests
throw the same reason. If the MLX spike goes ahead, its backend slots in at that branch. No `AetherLanguage` declaration
carries an availability annotation.

## Model selection and readiness in 0.0.3

Preparation (import, installation, download, verification caching) is 0.0.4. In 0.0.3 an app hands the facade a bundle
it already installed: a directory with `language-model.json`, registered by Kit's Forge tools, as `open(bundle:)` takes
today.

**`availability` is cheap and synchronous.** It reads the manifest and checks the OS, the platform field and that the
model and tokenizer paths exist. It never hashes. Today `open(bundle:)` hashes the whole model file, 7.5 GB for the
writer, which takes seconds. The reasons:

| Reason | When |
| --- | --- |
| `.unsupportedOS` | Below macOS 27 or iOS 27, with no other backend |
| `.unsupportedPlatform` | The registration's `platform` isn't this one (a macOS export on iPhone) |
| `.bundleMissing` | No `language-model.json`, or a listed path is absent |
| `.bundleInvalid(String)` | The manifest doesn't parse or fails its schema checks (unknown schema version, paths escaping the bundle) |
| `.unsupportedTemplate` | The tokenizer has no chat template |

**Loading (`prewarm()` or the first request) runs the full verification.** That includes the model and tokenizer hashes,
the export bounds and the contract. Failures throw `AetherLanguageError.unavailable(.bundleInvalid(…))`, with the
`LanguageGenerationError` or `AetherError` underneath. Other errors:
- `contextWindowExceeded`;
- `busy`;
- `closed` (after `unload()`);
- `CancellationError`;
- the runtime's `memoryBudget` (iOS cache growth), surfaced as `insufficientMemory`;
- from the structured-output piece, `decodingFailure` and `unsupportedSchema`.

**Not in 0.0.3.** There's no model catalog, recommended model or download. Each app picks its own reference model, and
Kit qualifies it (roadmap).

## The FoundationModels adapter

`AetherLanguageFoundationModels` is a target for OS 27 that depends on `AetherLanguage` and FoundationModels:

```swift
@available(macOS 27, iOS 27, *)
extension AetherLanguageModel: FoundationModels.LanguageModel {
    public typealias Executor = AetherLanguageExecutor
    public var capabilities: LanguageModelCapabilities     // [.reasoning] when supportsReasoning; + .guidedGeneration later
    public var executorConfiguration: AetherLanguageModel.Configuration
}
```

The executor:
- **Renders the request's transcript.** Instructions become the system message, prompts user messages, responses
  assistant messages, and reasoning entries are skipped. When `contextOptions.includeSchemaInPrompt` is true, the
  schema's JSON goes into the system message.
- **Maps the options:** `generationOptions` through the table above, and `contextOptions.reasoningLevel` to
  `reasoning: .on` when it's set and the model supports it; no level means the bundle's default. The three levels
  aren't distinguished, because the model has one switch.
- **Streams** response and reasoning text through the channel as `.response(.appendText(delta, tokenCount:))` and
  `.reasoning(.appendText(…))`, using the same UTF-8-safe deltas. When the reply ends, it sends `updateUsage` (prompt and
  output token counts).
- **Maps errors** into FoundationModels' errors where they exist (`LanguageModelError.contextSizeExceeded`), and
  otherwise passes them through.
- **Doesn't declare `.toolCalling` or `.vision`.** FoundationModels then refuses tool use and attachments for this model
  itself.
- **Doesn't declare `.guidedGeneration` until the JSON-schema constraint exists.** Until then a `@Generable` request
  would only be prompted, not constrained, and that's not good enough to ship. Without the capability,
  FoundationModels refuses a `@Generable` request itself, before it reaches the executor ("The selected model does not
  support guided generation"; second probe run), while text requests still work. So the adapter needs no refusal code
  of its own.

## Structured output: the next piece

**The seam is a runtime `LanguageLogitsProcessor`, as the runtime seam designed it.** For each request,
`LanguageJSONSchemaConstraint`:
- is built from a JSON Schema and the tokenizer's byte table;
- returns from `begin(prompt:)` the bytes that can start the value;
- advances, in `accept(_:)`, a byte-level JSON automaton by the accepted token's bytes, and returns the allowed set for
  the next token;
- allows only the stop IDs once the value is complete, so the existing end-of-sequence stop ends the reply.

The allowed set comes from walking a trie of token byte strings with the automaton's state. That work runs during the
GPU step, which hides up to 21 ms a token on the 4B and about 6 ms on the 0.6B; its cost is measured when it's built.

**The schema subset is what FoundationModels' `@Generable` emits,** as the probe recorded:
- objects with `properties`, `required`, `additionalProperties: false` and `x-order` (key order is fixed, which narrows
  the automaton);
- `string`, `integer`, `number`, `boolean`;
- `array` with `minItems` and `maxItems`;
- `enum` for `@Guide(.anyOf)`;
- `minimum` and `maximum`;
- optional properties.

Regex guides (`pattern`) and anything outside the subset are refused at request time with `unsupportedSchema` (through
the adapter, FoundationModels' `unsupportedGuide`). The output is never left unconstrained.

**Facade API.** `respond(to:generating:)` and `streamResponse(to:generating:)`, where `T: AetherGenerable`:

```swift
public protocol AetherGenerable: Decodable, Sendable {
    static var generationSchema: AetherGenerationSchema { get }   // a JSON Schema in the supported subset
}
```

Partial snapshots for the native stream come from an incremental JSON parser over the constrained text. That's simpler
than FoundationModels' partially generated types: `ResponseStream<T>.Snapshot` carries the partial JSON value and the
decoded `T` once the value is complete. A Kit `@Generable` macro would generate `generationSchema` from the type's
properties and doc comments, the counterpart of FoundationModels' macro. This design doesn't specify it further.

## Implementation plan

Each commit builds and passes `Scripts/check-cpu.sh`. GPU checks run under `with-gpu-lock.sh`.

1. **`AetherKit` locks:** `Mutex` → `OSAllocatedUnfairLock`, unless the floor work takes this.
   - Tests: the existing `GPUAdmission`, lease and resource suites are unchanged.
   - Check: extend the floor script to typecheck `AetherKit` at macOS 14 and iOS 17.
2. **The token byte table in `AetherLanguageTokenizers`:** each token ID's raw bytes for byte-level BPE (inverting GPT-2's
   byte-to-unicode map), with special and added tokens flagged.
   - CPU tests: a small synthetic byte-level tokenizer fixture.
   - Opt-in test with the Qwen3 hub snapshots (`AETHER_LANGUAGE_TOKENIZER_HUB`): decoding the concatenated bytes of
     every token in the chat oracle's replies equals the tokenizer's `decode`.
3. **The `AetherLanguage` target:**
   - `AetherLanguageModel` (bundle, cheap `availability`, lazy verified load, `prewarm`, `unload`) and the backend
     choice behind `#available`.
   - An internal engine protocol, so CPU tests drive the engine with a fake `LanguageBackend` that replays scripted
     tokens, and a fake tokenizer.
   - Tests: readiness reasons for missing, invalid, wrong-platform and template-less bundles, without loading.
4. **The streaming decoder:** UTF-8 and grapheme-safe deltas, stop strings with hold-back, and the reasoning splitter. All
   pure. CPU tests on scripted byte streams:
   - every split of multi-byte scalars, combining marks, ZWJ emoji, flags and Hangul;
   - stop strings across token boundaries, and prefixes that never complete;
   - reasoning delimiters in and out;
   - a fuzz test that concatenated deltas equal the whole decode.
5. **`AetherLanguageSession`:**
   - `respond`, `streamResponse` (snapshots with deltas, abandonment cancels), options mapping, the transcript (append
     on success, revert on failure or cancellation), context checks, `busy`, `closed`.
   - CPU tests with the fake backend, including cancellation mid-stream joining before the error, and a 3-turn
     transcript rendering the expected messages.
6. **The Core AI backend behind the facade.** An opt-in native test, `languageFacadeQualification`, on the 4B writer
   bundle and on a 0.6B bundle (a `language-model.json` registered for the existing 0.6B export by Forge's
   `register_qwen3_bundle.py`). It checks:
   - a greedy `respond` equals `LanguageGenerationSession`'s tokens on the same rendered prompt, and its content equals
     the tokenizer's decode;
   - a stream's deltas concatenate to the final content;
   - stop strings;
   - cancellation mid-stream, then a follow-up request;
   - a 3-turn conversation;
   - a seeded request repeats;
   - timing against the runtime alone (the engine's overhead per token).
7. **`AetherLanguageFoundationModels`:** the adapter, text and reasoning only.
   - CPU tests drive a real `LanguageModelSession` over the fake backend, as the probe did: transcript rendering,
     options mapping, streamed text, and FoundationModels' refusal of `@Generable` requests while the capability is
     off.
   - Native: `LanguageModelSession(model:)` on the 4B writer equals the facade's reply for the same transcript.

Then the structured-output piece: the JSON automaton and the constraint, `.guidedGeneration` in the adapter, the
facade's `generating:` API, and the macro if the user wants it.

## Decided

The user answered the first two on 2026-10-27; the rest are the lead's.

1. **Both surfaces ship in 0.0.3:** the native `AetherLanguage` facade, which links on macOS 14 and iOS 17, and the
   `AetherLanguageFoundationModels` adapter, the recommended path on OS 27, over one engine.
2. **Kit's own `@Generable`-style macro ships in 0.0.3** with the swift-syntax dependency, in the structured-output piece
   after the seven commits above. `AetherGenerable` stays macro-friendly: the macro generates the static schema, and its
   JSON Schema output matches what FoundationModels' `@Generable` emits (the probe's record shows the shape), so one
   constraint serves both surfaces.
3. **Reasoning levels:** any FoundationModels `reasoningLevel` means thinking on; no level means the bundle's default
   (off for every registration today).
4. **Default output limit:** the remaining context, capped at 4,096 tokens, and always overridable with
   `maximumResponseTokens`.
5. **`AetherKit`'s `Mutex`** is replaced in commit 1 with `OSAllocatedUnfairLock`, as a mechanical swap with the existing
   tests unchanged (`GPUAdmission` and `SharedGPULease` are lease-critical). The rest of the floor work (package
   platforms, availability on the Core AI targets, the weak `CoreAI` link, the macOS 14 launch check) comes after the
   decoder rework lands, because it touches every target.
