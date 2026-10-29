# Language runtime seam

Status: design approved and all seven commits implemented, 2026-10-27 (see [Implementation plan](#implementation-plan));
commit 6 followed the decoder rework. This is the first piece of the 0.0.3 app-facing language
track ([roadmap](roadmap-0.0.3.md#parallel-track-an-app-facing-language-product)): a sampler and logits hook, and a
`LanguageBackend` protocol that the `AetherLanguage` facade can drive without knowing about Core AI. Type names are
illustrative until the facade has run in a real app (see [architecture](architecture.md)).

## Where things stand

`LanguageGenerationSession` decodes greedily. Each `step` encodes one submission, joins it, copies the bucketed
contract's cache update rows into the Kit-owned caches, and returns the last logits row as a new `[Float]`.
`LanguageDecodeLoop.run` turns a `step(token, position) -> next token` closure into a reply, delivering each emitted
token while the following step runs. The session's `generate` methods fold `LanguageGenerationSession.greedy` into
that closure. Greedy picks the first index of the largest finite logit, skips infinities, and throws on NaN or when no
logit is finite. Sampling state never touches the KV caches, the step shapes, the memory budget or GPU admission. The
seam keeps it that way.

### Where logits live, and what a row costs on the CPU

Core AI returns the logits as an output `NDArray`: [1, 1, vocab] fp16 for the bucketed contract, and [1, q, vocab]
for the running one, of which Kit reads only the last row. After the join, the row is readable from the host (Apple
silicon's unified memory). Kit never sees a device-only copy, so "bringing the row to the CPU" is a read, not a
transfer. For Qwen3's 151,936-row head, that row is 297 KiB of fp16.

Measured per one-token step, as medians on one M5 Max, a shared host, under the GPU lock
([evidence](evidence/language-logits-readback-2026-10-27.json)). The method mirrors `step` against Core AI alone:

| | Qwen3-0.6B, cache 256 | Qwen3-0.6B, cache 2,048 | Qwen3-4B writer, cache 256 | Qwen3-4B writer, cache 2,048 |
| --- | --- | --- | --- | --- |
| Step, from encode through greedy | 6.50 ms | 8.40 ms | 23.25 ms | 24.84 ms |
| `await logits.ndArray` after the join | 15 µs | 14 µs | 23 µs | 21 µs |
| Today's conversion into a new `[Float]` | 256 µs | 260 µs | 276 µs | 283 µs |
| Greedy argmax over that array | 144 µs | 145 µs | 151 µs | 156 µs |
| Host share of the step (the three rows above) | 6.4% | 5.0% | 1.9% | 1.9% |

The row costs about 0.4 ms per step today: almost all of it is the per-step allocation and conversion and the scalar
argmax, not reaching the memory. Two cheaper paths were checked:
- A reused Float buffer filled by `vImageConvert_Planar16FtoPlanarF` takes 6 µs instead of 81–276 µs.
- Passing a Kit-owned shared `MTLBuffer` as the logits output view (`encode(…, outputViews:)`) works on the bucketed
  asset. On every measured step it gave the same argmax as the `NDArray` path, with nothing allocated per step.

Neither is needed for correctness. Commit 7 adopted the first: `step` now converts the row with vImage (see the plan
below).

Host sampling over one full row (single thread; the 4B's first-step row):

| Operation | Median |
| --- | --- |
| Repetition penalty over 1,024 previous tokens | 8 µs |
| Mask by a vocabulary bitset | 26 µs |
| Argmax over a 2,000-token allowed list | 2 µs |
| Top-k 50 by a selection scan, then top-p within it | 39–41 µs |
| Temperature and softmax over the whole vocabulary: scalar / vDSP and vForce | 297 / 88 µs |
| Seeded categorical draw over the whole vocabulary | 75 µs |
| Top-p by sorting the whole vocabulary | 10.4 ms |

Everything except the full sort fits well inside a step: under 0.5 ms against 6.5 ms (0.6B) and 23 ms (4B). The full
sort is ruled out. Top-p never sorts the vocabulary (see [Sampling](#sampling)).

One more observation shapes the overlap design. On both models `encode` returned only after most of the step's time,
and the join that followed took 40–120 µs. So while a step runs, the session actor's thread is busy inside Core AI. Work
meant to overlap the GPU, such as token delivery today and constraint computation below, has to run in a child task,
not on the session actor.

## Decisions

1. **A new target, `AetherLanguageRuntime`, holds the protocol, the sampler, the constraint hook and the shared decode
   loop.** It depends only on the Swift standard library and Foundation: no Core AI, Metal, `Synchronization` or
   AetherKit. `AetherLanguageCoreAI` depends on it and re-exports it. A future MLX backend and the `AetherLanguage`
   facade depend on it too. It's a library product, so an out-of-package backend can conform; the shared decode loop
   and sampler are `package` for now, and become public only if a backend outside the package needs them.
2. **`LanguageBackend` is token-level.** Token IDs go in, and token IDs come out through a backpressured callback. The
   tokenizer, chat template, reasoning and text belong to the facade.
3. **Sampling happens on the host, after the join, in backend-independent code.** Every backend gets the same
   semantics, and the same seed gives the same tokens on the same logits.
4. **The constraint hook returns a token constraint and never mutates logits.** It runs in a child task, concurrently
   with the GPU step whose logits it will constrain. The loop applies the constraint after the join.
5. **The default stays greedy, byte for byte:** same argmax function, same `[Float]` row, same `step`. The existing
   public methods keep their signatures and become wrappers.
6. **No step, prefill, shape, cache, reload or admission code changes in the sampling commits.** Those belong to the
   decoder wiring, which the `v0.0.3/vl-decoder` branch is reworking (see [Boundary](#boundary-with-the-decoder-rework)).

## Where the protocol lives, and the deployment floor

`AetherLanguageRuntime` builds at the package floor, macOS 14 and iOS 17, with no availability annotations: it has no
Core AI or Metal types in its surface or its imports. The facade holds an `any LanguageBackend` and reports "unsupported
OS" when no conforming backend is available. The [deployment floor](deployment-floor.md) gates the Core AI targets at OS
27 and checks that a macOS 14 facade binary links `CoreAI` weakly and launches on macOS 15, which has no CoreAI.

## `LanguageBackend`

As implemented (abridged; not frozen):

```swift
// AetherLanguageRuntime
public protocol LanguageBackend: Sendable {
    /// Rows of the logits head. Prompt and sampled token IDs are in 0..<vocabularySize.
    var vocabularySize: Int { get }
    /// The registered context: prompt plus output.
    var maximumContextLength: Int { get }
    /// The registration's end-of-sequence IDs (`stopTokenIDs` in the manifest).
    var stopTokenIDs: Set<Int> { get }

    /// One request. Each emitted token goes to `onToken` while the following step runs; at most one delivery is in
    /// flight, and every delivery finishes before this returns or throws. Returning `.stop` ends the request after
    /// the overlapping step joins (its result is discarded). Cancelling the calling task stops at the next step;
    /// submitted native work joins before the error is thrown. One request at a time: overlap throws `busy`.
    func generate(_ request: LanguageGenerationRequest,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult

    /// Terminal close, as `LanguageGenerationSession.unload()`: later requests throw `closed`.
    func unload() async
}

public struct LanguageGenerationRequest: Sendable {
    public var prompt: [Int]
    public var maximumTokens: Int
    public var sampling: LanguageSampling             // .greedy by default
    public var constraint: (any LanguageLogitsProcessor)?
}

/// A struct, so per-token data such as log-probabilities can be added later without changing the callback's type.
public struct LanguageGeneratedToken: Sendable, Hashable {
    public let id: Int
    public let index: Int                              // its index in the reply
}

public enum LanguageTokenDisposition: Sendable { case `continue`, stop }

public struct LanguageGenerationResult: Sendable {
    public let tokens: [Int]
    public let stopReason: LanguageStopReason          // .endOfSequence, .outputLimit, .stopped
    public let seed: UInt64?                           // the seed used, when sampling was random
    public let timeToFirstTokenSeconds: Double
    public let totalSeconds: Double
}
```

**Actors conform directly.** The protocol has `async` requirements and get-only properties, so an actor conforms
with `nonisolated let` properties. `LanguageGenerationSession` already keeps its vocabulary size, context and stop IDs
as `let`s.

**Mapping onto today's API.** `LanguageGenerationSession` conforms in an extension, as it already conforms to
`VisionLanguageDecoding`. Its body is today's text `generate`, with the greedy closure replaced by the shared loop and
the request's sampler. The public `generate(prompt:maximumTokens:prefillChunkSize:onToken:observer:)` keeps its
signature and its `Reply`. It calls the same private body with `.greedy` and no constraint, and wraps `onToken` to return
`.continue`. Three things stay Core-AI-specific and outside the protocol, in the Core AI backend's API:
- `prefillChunkSize`;
- the `Observer` of native submissions;
- the qualification entry points (`prefillLogits`, `teacherForcedLogits`). Teacher-forced logits are also on
  `LanguageBackendQualification`, an SPI protocol (`@_spi(Qualification)`) a backend can adopt so a parity harness runs
  against any backend without them being app-facing.

**Why the delivery callback can stop the request.** The facade needs it for stop strings, which it detects in decoded
text, and for an app that has seen enough. Without it, the facade would have to throw from the callback and lose the
result. The cost is one step: the one already running when delivery says stop. It's joined and discarded, as a failed
delivery's is today.

**Why a callback is the primitive, not an `AsyncSequence`.** The [architecture](architecture.md) requires token
delivery to be lossless and bounded, or to apply backpressure, and abandoning a stream must not strand native work. The
callback has both properties today, by construction: there's one delivery in flight, the request is structured, and it
joins before returning. The facade builds its streams on top. A FoundationModels-style stream of cumulative snapshots can
buffer only the newest element losslessly, because each snapshot supersedes the last. An app that abandons the stream
cancels the task that runs `generate`.

**Errors.** `LanguageGenerationError` (`invalid`, `busy`, `closed`, `contextLimit`, `memoryBudget`) moves to
`AetherLanguageRuntime`, so every backend speaks it. `AetherLanguageCoreAI` re-exports the runtime module
(`@_exported import`), so Shilo's source doesn't change; a typealias would have made the name ambiguous in files that
import both modules. This is the only existing public type that moves.

**Images are out of the protocol for now.** `VisionLanguageSession` keeps its own API, and since commit 6 its sampling
entry point, and the image `generate`'s, take a `LanguageSampling` and a constraint through the same decode loop. An image-capable
protocol is a later refinement, if the apps need images through the facade.

## Sampling

```swift
public struct LanguageSampling: Sendable, Hashable {
    public enum Mode: Sendable, Hashable {
        case greedy
        case random(temperature: Double, topK: Int?, topP: Double?, seed: UInt64?)
    }
    public var mode: Mode                              // .greedy by default
    /// Transformers' rule: positive logits divided, negative multiplied. 1 (the default): off.
    public var repetitionPenalty: Double
    /// Most recent tokens (prompt included) the penalty sees. nil (the default): the whole request.
    public var repetitionWindow: Int?
    public static let greedy: LanguageSampling
    public static func random(temperature: Double = 1, topK: Int? = nil, topP: Double? = nil,
                              seed: UInt64? = nil) -> LanguageSampling
}
```

**Order per step,** following Transformers' `generate` so that the warpers can be checked against it:
1. repetition penalty;
2. the constraint;
3. then, when random: temperature, top-k, top-p, normalization and one draw.

Greedy with no penalty and no constraint calls today's `greedy` on today's row and does nothing else. That is the
invariant path.

**Semantics:**
- **Temperature** must be greater than 0 in random mode. Greedy is its own mode, not temperature 0.
- **Top-k keeps every token tied at the k-th logit,** as Transformers does.
- **Top-k finds its threshold with one bounded selection scan** (it keeps at most 2k values), on the logits divided by
  the temperature, as Transformers truncates them.
- **Top-p is exact and keeps at least one token:** in descending probability (ties in ascending token ID), every
  candidate whose preceding mass is below p. Candidates at or below (1 − p)/n are pruned first: together they hold at
  most 1 − p and rank below the rest, so they can never be kept. The rest are selected by in-place three-way
  partitioning on value. Nothing sorts the vocabulary.
- **Probabilities** are float32 weights, as Transformers' softmax, normalized in double precision.
- **The repetition penalty counts each distinct token once,** over the prompt and the output so far. Transformers
  applies it to `input_ids`, prompt included, and in greedy mode too. A penalty in greedy mode is therefore
  allowed, and it is not the invariant path.
- **Disallowed tokens become −∞** before temperature and truncation, so top-k and top-p only ever see allowed tokens.
  If the constraint leaves no finite logit, the request throws `invalid("The constraint allows no token")`.

**Seeded and deterministic:**
- **The generator** is Kit's own: SplitMix64 seeding xoshiro256\*\*, specified in the source, never
  `SystemRandomNumberGenerator`.
- **Draws:** random mode consumes exactly one 64-bit draw per sampled token, even when one candidate remains, so a replay
  never drifts. Greedy consumes none.
- **When no seed is given,** one is drawn from the system generator and reported in the result, so any reply can be
  replayed.
- **The draw is an inverse CDF** over the candidates in ascending token-ID order, so the kept set never needs sorting.
  Its reductions are scalar, in a fixed order, with double-precision accumulation, and the target doesn't use
  Accelerate, so the token picked doesn't depend on which kernel a library selects.

**Checked against Transformers.** `Forge/language/reference_sampling.py` (run with Forge's environment) records what
Transformers 4.57.3's own `RepetitionPenaltyLogitsProcessor`, `PrefixConstrainedLogitsProcessor`,
`TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper` make of 18 seeded synthetic rows, including ties at
the top-k boundary and at the greedy maximum. The CPU suite requires the same greedy token, the same kept set and
probabilities within a relative 1e-4 on every case. The script also records SplitMix64 and xoshiro256\*\* known answers
from its own implementation of both algorithms.

**What a seed promises.** The same seed gives the same tokens for the same asset, device class, OS and prefill plan. That
holds across memory-budget reloads and running-contract recycling, because both leave the logits unchanged: replies are
already identical across reloads ([evidence](evidence/language-memory-budget-2026-10-27.json)). Kit doesn't promise the
same tokens across devices or OS updates, because the GPU kernels' rounding may change.

**Head rows past the tokenizer.** Qwen3's head has 151,936 rows; its tokenizer has fewer. Greedy never picks the padding
rows in practice, but random sampling at a high temperature could. The facade masks them with a base constraint (below).
The runtime doesn't need to know the tokenizer.

## The constraint hook

```swift
public protocol LanguageLogitsProcessor: Sendable {
    /// The constraint on the first sampled token. Runs concurrently with the prompt's prefill.
    func begin(prompt: [Int]) async throws -> LanguageTokenConstraint
    /// Records `token`, just sampled and now submitted, and returns the constraint on the token after it.
    /// Runs concurrently with `token`'s decode step.
    func accept(_ token: Int) async throws -> LanguageTokenConstraint
}

public enum LanguageTokenConstraint: Sendable {
    case unconstrained
    case allowed(LanguageTokenSet)          // a vocabulary bitset or a range; intersectable
}
```

The constraint for token t+1 depends only on tokens up to t, and those are all known once t is submitted. So a grammar
or JSON-schema matcher advances its state and builds the next allowed set while step t runs on the GPU. When
the step joins, the loop only has to apply the result: 26 µs for a bitset, or about 2 µs to choose among a short
allowed list. The matcher's own cost is hidden up to the step time (6.5 ms on the 0.6B, 23 ms on the 4B). Only the
excess adds to each token. While it runs, the request holds its GPU grant, and the GPU idles for that excess.

**Why it doesn't mutate the logits in place:**
- **No overlap.** A hook that takes the logits row can only start after the join, so all of the matcher's work would
  be serial with the GPU.
- **Shared state across the loop.** It would also need its state on the session actor, or mutable state shared across
  the loop's child tasks.

A constraint that is a value keeps the processor's state in the processor: an actor, or a class with its own lock.
`Sendable` is required because it runs in a child task, beside token delivery.

**Per request, not per session.** A processor holds one request's state. `begin` resets it.

**When it isn't called.** `accept` is not called for a stop token, or for the last token at the output limit, since
nothing is sampled after them.

**Failure follows the delivery rules.** A processor that throws has its overlapping step joined first. That step's result
is discarded, and the request throws. A step that fails cancels and awaits both the delivery and the processor.

**Finishing structured output.** When the grammar has accepted a complete value, the processor allows only the stop
IDs, and the existing EOS stop ends the request. This needs no special path.

**Not covered.** Logit biases, and processors that need the logits' values (min-p, classifier-free guidance) aren't
covered. They can be added later as another case of the constraint, or as a synchronous transform, without changing
backends.

## Invariants

**Default greedy output is token-identical to today's.** The default request is `.greedy`, with no penalty and no
constraint. On that path the loop calls `LanguageGenerationSession.greedy` on the same `[Float]` row that `step` returns
today, with the same stop checks, positions and delivery overlap. `greedy` moves to the runtime target unchanged,
and the session's static forwards to it. The proof:
- **Existing CPU tests unchanged:** `LanguageDecodeTests`, `LanguageLifecycleTests`, `LanguageRequestAdmissionTests` and
  `VisionLanguageSessionCloseTests` pass without edits. The old `LanguageDecodeLoop.run(first:step:)` entry stays as a
  wrapper over the new loop.
- **New CPU equivalence tests:**
  - The sampler's greedy path equals `greedy` on fuzzed rows that include ties, −∞, +∞, all-non-finite rows and NaN
    (the same error).
  - The new loop, driven by a fake step that returns seeded random rows, emits exactly what the old greedy loop does.
- **The real-bundle check:**
  - On `cd6894b` (before), record the 4B writer's chat round trip (`languageChatBundleRendersGeneratesAndDecodes`) and the
    0.6B bucketed long run (`languageBucketedDecodeQualification`: 64-token greedy continuation and 1,024 steps).
  - After each commit that touches the session, record them again and require identical token lists, through both the
    old API and the protocol.
  - Each run takes a few minutes under the GPU lock on assets already on disk.

**Bucketed decode.** Sampling and constraints run on the host after the join. They add no input, no output view, no
new (q, capacity) pair and no submission. The shape set, the warm-shape accounting and the 1-token decode step are as
today. A different sampled continuation has a different length, so its capacity grows differently, as any reply does.
- **Jump-forward is out.** Grammar-forced runs could be submitted as multi-token chunks ("jump-forward" decoding). But
  every new q is a new shape: it warms once, and Core AI retains memory each time a request changes to it. So
  jump-forward is not in 0.0.3. If it comes later, it may only use the prefill width C, never a new q.

**Memory budget and recycling.** Sampler state is per request: the generator, the penalty history and the processor live
in the loop, never in the session. The memory budget's reloads happen between requests, in admission, and never meet it.
Running-contract recycling happens inside a request. It replaces the model, function and stream but keeps the Kit-owned
caches, so the loop is unaffected. Constraint work done while a request holds its grant lengthens the request's memory
window but allocates no Metal memory, so the budget's per-token rate is unaffected.

**The overlap gate.** The delivery guarantees stay as they are: in order, at most one in flight, all delivered before
return, a failed delivery's overlapping step joined and discarded, a failed step cancelling and awaiting the delivery.
The processor's `accept` joins the same overlap under the same rules, and `begin` overlaps the prefill. New CPU tests
cover the following:
- `accept(t)` starts before step t ends;
- step t+1 never starts before `accept(t)` returns;
- a processor that throws, and one that is cancelled;
- `.stop` from delivery;
- an empty allowed set.

**Cancellation and unload.** These are unchanged. The loop checks cancellation before every step, and child tasks
(delivery, processor) are cancelled and awaited before the request throws. `unload()` still waits for an active
request's native join.

## Boundary with the decoder rework

`v0.0.3/vl-decoder` owns how the decoder is wired: `step`'s inputs and function selection, the image prefill, and the
model reload's function loading. This seam owns the generation loop and sampling: `LanguageDecodeLoop`, the
`generate` bodies after the prefill, `greedy`, and the new runtime target. In practice:
- **Commits 1–5 never edit `step`, either `prefill`, `imagePlan`, `reloadModel` or the image `generate`.** The image path
  keeps calling the old greedy `LanguageDecodeLoop.run` entry, which stays source-compatible, so the two branches merge
  without conflicts in shared lines.
- **The one shared seam is `step`'s return value** (the logits row). Changing it, to read into a reused buffer or an
  output view, waits until the decoder rework has landed (commit 7).
- **If the rework changes `step`'s signature,** the only adaptation here is the closure in the text `generate`.

## What the facade will need next

These are not in the seam's commits. They're listed so the seam doesn't block them:
- **A base constraint:** the tokenizer's vocabulary range, masking the head's padding rows, intersected with any
  structured-output constraint (`LanguageTokenSet` supports intersection).
- **A token-to-bytes table** from `AetherLanguageTokenizers`, so a JSON-schema matcher can map tokens to the grammar's
  terminals. Byte-level BPE tokens must be decoded to bytes, not to display strings.
- **Reasoning:** with thinking on, the processor returns `.unconstrained` until it sees the end-of-thinking token.
  No seam change is needed.
- **UTF-8-safe streaming and stop strings** are the facade's. It uses the delivery's `.stop` disposition.
- **Default sampling, mapped from FoundationModels.** FoundationModels' options (`.greedy`, `.random(top:seed:)`,
  `.random(probabilityThreshold:seed:)`, `temperature`, `maximumResponseTokens`) map onto `LanguageSampling` and
  `maximumTokens`. The facade defaults to greedy; apps opt into sampling per request.
- **Prefix reuse across turns.** A bucketed request resets the cache today, so every chat turn prefills the whole
  transcript again. Prefix reuse is a later backend option on the request and doesn't change the protocol's shape.
- **Backend selection:** `#available(macOS 27, iOS 27, *)` picks the Core AI backend; otherwise it's an MLX backend if
  that spike goes ahead, or "unsupported OS".

## Implementation plan

Each commit builds and passes `Scripts/check.sh` on its own. The GPU checks run under `with-gpu-lock.sh`. Commits 1–5
landed on `v0.0.3/runtime-seam` as planned, with two changes: `LanguageGenerationError` moved in commit 1 (the sampler
throws it from commit 2), and a commit between 4 and 5 sped up random sampling after the native run measured it (below).

1. **Add the `AetherLanguageRuntime` target and its test target.** It holds the protocol, request and result types,
   `LanguageSampling`, `LanguageTokenSet`, `LanguageTokenConstraint`, `LanguageLogitsProcessor` and the generator. No
   behaviour yet.
   - Tests: the generator's known-answer vectors; token-set operations.
   - Check: macOS 14 typecheck of the target's sources.
2. **Add the sampler: penalty, constraint, temperature, top-k, top-p and the seeded draw.** `greedy` moves here, and
   `LanguageGenerationSession.greedy` forwards to it.
   - Tests:
     - greedy equivalence on fuzzed rows, including NaN and infinities;
     - every warper against fixtures that `Forge/language` records from Transformers' own processors and warpers
       (small synthetic rows, kept sets and probabilities to a tolerance; development tooling only);
     - draw replay and the draw distribution over many seeds;
     - empty-constraint errors.
3. **Generalize the decode loop into the runtime target, with sampler, processor and disposition.** The old greedy
   entry becomes a wrapper.
   - Tests: the existing `LanguageDecodeTests` unchanged; the new overlap, failure, stop and equivalence tests above.
4. **Make `LanguageGenerationSession` conform to `LanguageBackend`.** The text `generate` routes through the shared
   body, and `LanguageGenerationError` moves with a typealias. The image path is untouched.
   - Tests: CPU lifecycle and admission suites unchanged.
   - Native: the before/after token identity check for the 4B chat round trip and the 0.6B long run, through both APIs.
5. **Add an opt-in native test, `languageSamplingQualification`** (the 0.6B raw asset, the 4B writer bundle). It records
   evidence and documents sampling in [language-generation.md](language-generation.md). It checks:
   - that the protocol's default equals the old API;
   - that the same seed repeats exactly, after other requests and in a freshly loaded session (a model load, as a reload
     is), and that different seeds differ;
   - that an allowed-set constraint (for example, digits and EOS only) holds on every token;
   - the per-token cost of each sampling mode, and of a deliberately slow processor, against greedy.
6. **After the decoder rework lands, give the image path the same options.** The image `generate` and
   `VisionLanguageSession` take a `LanguageSampling` and a constraint through the shared body.
   - Native: a VL token-identity check against the committed qualification tokens.

   Done: each gained a sampling entry point (`sampling:` and an optional `constraint:`) returning a
   `LanguageGenerationResult`, and the image `generate` runs through `LanguageDecodeLoop.generate`; the greedy entry
   points wrap it with `.greedy`. The sampler and the processor see the expanded prompt. Before (`147c4b1`) and after,
   under the GPU lock, the greedy replies are identical: the decoder-level text and image replies of the VL-2B and VL-4B,
   one-function and two-function bundles; the public session's five image cases on the one-function 2B and 4B and the
   two-function 4B, each also matching its qualified Kit run; and the writer's chat round trip. An opt-in
   `languageSamplingQualificationWithImages` checks a VL bundle's public session: the sampling entry point's greedy equals
   the greedy one, a seeded image request repeats (after other requests and in a freshly opened session) and differs by
   seed, and the digits-then-stop constraint holds on every token of an image prompt
   ([evidence](evidence/qwen3-vl-sampling-2026-10-27.json)).
7. **Optional, measured: drop the per-step row allocation.** Fill a reused buffer by vImage, or use a Kit-owned logits
   output view.
   - Expected gain: about 0.25 ms a step, 4% of a 0.6B step and 1% of a 4B step.
   - Tests:
     - CPU: vImage's fp16-to-fp32 conversion equals `Float(Float16)` on all 65,536 bit patterns;
     - native: the token-identity check again.
   - It's dropped if the identity check or the gain doesn't hold.

   Done with vImage (`LanguageLogitsRow`), not the output view, which would have changed the step's encoding for no
   further gain: the conversion takes 7.3 µs including the new array, against 256–283 µs before, and is exact on all
   65,536 bit patterns. After it, the 0.6B and 4B token-identity runs are identical again, and so are the 0.6B's
   teacher-forced row metrics, bit for bit. Greedy per-token time went from 21.1 to 20.7 ms on the 4B and from 7.3 to
   6.2 ms on the 0.6B in the same native test (separate runs on a shared host: observations). The argmax itself (about
   0.15 ms) is unchanged, so the default path's arithmetic is too.

### Results of commits 1–5

All on one M5 Max, release build, shared host, under the GPU lock
([evidence](evidence/language-runtime-seam-2026-10-27.json)).

- **Token identity.** Through the existing API, before (`cd6894b`) and after (the session's conformance, and the branch
  head): the Qwen3-0.6B bucketed 64-token greedy continuation and its 1,024-step long run (no reloads, six step shapes
  both times), and the Qwen3-4B writer's three chat replies, are identical. Through the protocol, the default request
  equals the existing API on two prompts per model (35 and 96 tokens on the 4B, 23 and 96 on the 0.6B).
- **Seeds.** A seeded top-k 40 / top-p 0.95 request repeats its 64 tokens after other requests and in a freshly loaded
  session, and another seed differs, on both models.
- **Constraint.** Digits only for 12 tokens, then only the stop IDs: every token was a digit and the reply ended at end
  of sequence (`100000000000` on the 4B, `123456789012` on the 0.6B).
- **Cost per token** (warm, up to 96 tokens; medians of two rounds; the 0.6B's are within the host's noise):

| Mode | Qwen3-4B writer | Qwen3-0.6B |
| --- | --- | --- |
| Greedy (the default) | 21.1 ms | 7.3 ms |
| Greedy, repetition penalty 1.1 | 21.4 ms | 7.3 ms |
| Random, temperature 0.8 | 22.5 ms | 7.5 ms |
| Random, top-k 40, top-p 0.95 | 21.5 ms | 6.8 ms |
| Random, top-p 0.9 | 22.5 ms | 7.6 ms |
| Greedy, a processor allowing everything | 22.4 ms | 6.8 ms |
| Greedy, a processor taking 5 ms a token | 21.7 ms | 6.8 ms |
| Greedy, a processor taking 40 ms a token | 43.6 ms | 44.3 ms |

  A 5 ms processor is hidden behind the step on both models; a 40 ms one sets the pace, as designed.
- **The sampler alone,** on the 4B's 151,936-row first-step logits (single thread, medians): greedy 0.14 ms; top-k 40
  with top-p 0.95, 0.15 ms; temperature only, 0.52 ms; top-p 0.9 without top-k, 0.75 ms. The first native run measured
  random modes 0.7–1.5 ms a token slower than greedy on the 4B (0.97–1.67 ms in the sampler alone, from quickselect over
  the vocabulary and allocations); the speed-up commit brought them to these numbers with the same results.

## Rejected alternatives

- **Sampling on the GPU inside the exported graph.** This needs a re-export of every asset and a per-step mask input,
  and the host row already costs 15–23 µs to reach.
- **An `AsyncThrowingStream` as the backend primitive.** It has no backpressure into the decode loop, and abandoning it
  leaves cancellation to a termination handler. The facade layers streams on the callback instead.
- **A processor that edits the logits on the session actor.** Its work would serialize after the join instead of
  overlapping the step.
- **The protocol in AetherKit.** It would tie the macOS 14 launch to the `Mutex` gating described above.
- **Top-p by sorting the vocabulary.** This costs 10.4 ms a step.

## Decided

The user confirmed the first four on 2026-10-27; the rest are the lead's answers to the design's questions.

- **Facade default:** greedy. Apps opt into sampling per request. The seam's default is greedy too, and the token
  identity above holds for it.
- **Floor:** macOS 14 and iOS 17, the package floor ([deployment floor](deployment-floor.md)).
- **Images through the facade:** none in 0.0.3, so commit 6 stays after the decoder rework.
- **Log-probabilities:** none for now. The per-token callback takes a struct (`LanguageGeneratedToken`), so they can be
  added without breaking it.
- **Target:** `AetherLanguageRuntime`, a new library product.
- **Repetition penalty:** Transformers' semantics, over the whole request including the prompt, off (1) by default,
  with an optional window.
- **Teacher-forced logits:** on the SPI `LanguageBackendQualification` protocol, not on `LanguageBackend`.
