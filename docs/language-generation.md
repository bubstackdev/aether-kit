# Experimental language generation

`AetherLanguageCoreAI` is Aether Kit’s local, sequential dense-language reference
runtime. Shilo consumes this product rather than carrying a second inference
implementation. It accepts two step contracts over FP16 logits and a contiguous
five-dimensional KV cache (see [Step contracts](#step-contracts)). This is an
experimental consumer-driven API, with no compatibility promise yet.

## Local registration and execution

A caller supplies `language-model.json` in the model bundle. The manifest names
relative model/tokenizer directories, immutable upstream source identity, SHA-256
hashes, platform, quantization, stop IDs and a maximum context. `LanguageModelManifest`
lives in the Core-AI-independent `AetherKit` target. Verification rejects mismatched
platforms, integrity failures and paths escaping the local bundle. The optional
`supportsReasoning` field is consumer-facing metadata, for example whether a chat
app offers a reasoning mode; Kit decodes it but does not act on it. The optional
`exportBounds` (`maximumPrefillChunk`, `maximumCacheCapacity`) declare the shape
maxima the asset was exported with. They are required to open a bucketed-contract
asset, the registered context may not exceed the exported cache capacity, and
prefill submissions stay within the exported chunk bound (`Forge/language` records
both in its export report).

```swift
import AetherKit
import AetherLanguageCoreAI

let session = try await LanguageGenerationSession.open(bundle: localBundleURL, platform: .ios)
let reply = try await session.generate(
    prompt: renderedTokenIDs,
    maximumTokens: 8192,
    prefillChunkSize: 64
)
```

The example requires a caller-qualified registration large enough for the prompt
plus response. The API never downloads assets, changes the requested budget, or
falls back to another device. GPU preference is a Core AI specialization preference;
actual processor placement remains controlled by Core AI. Tokenization and template
rendering remain explicit consumer responsibilities in this first product.

## Tokenizer and chat template

`AetherLanguageTokenizers` renders a chat with the model's own template, so callers no longer hand-build prompts:

```swift
import AetherLanguageTokenizers

let tokenizer = try await LanguageTokenizer.open(bundle: localBundleURL)   // verifies tokenizer.json
let prompt = try tokenizer.render([
    .system("You describe fictional adult characters for an image generator."),
    .user("Describe her look in one line.", images: 1),                      // one placeholder, before the text
], thinking: false)
// Text only: session.generate(prompt: prompt.tokenIDs, ...)
// With images: visionSession.generate(renderedTokenIDs: prompt.tokenIDs, images: photos, ...)
//              (prompt.imageCount images, in message order)
let text = tokenizer.decode(reply.tokens)                                     // special tokens dropped
```

- **The model's template, not a copy.** Exactness against upstream is verified for Qwen3-family templates (below);
  others render through swift-jinja unverified. A hybrid Qwen3 renders its empty think block when `thinking` is false, an
  instruct-only Qwen3 (2507) renders none, and Qwen3-VL renders `<|vision_start|><|image_pad|><|vision_end|>` per
  image. Placeholders stay unexpanded: the vision session expands them to the image's token count, so the image
  grid has one owner.
- **Sources.** The template is read like the swift-transformers loader (`chat_template.jinja`, then
  `chat_template.json`, then `tokenizer_config.json`). Upstream renders with Jinja's `keep_trailing_newline=False`,
  dropping one trailing newline from the template source; swift-jinja keeps it (a stray `\n` after Qwen3-VL's
  generation prompt), so the kit drops it.
- **Local only, verified.** `open(directory:)` reads a snapshot or export folder; `open(bundle:)` runs
  `LanguageModelManifest.verifyTokenizer(in:)` before and after loading. It always checks `tokenizer.json`; when the
  registration lists `tokenizerFiles` (every tokenizer file's SHA-256, as the exporters record), the directory must
  hold exactly those files, so a swapped chat template or special-token map is refused. The model asset isn't
  hashed. Nothing is downloaded. The Qwen3 exporter now copies the template files with the tokenizer.
- **One text part per text-only message.** It renders as the template's plain string content. Messages with images
  render as typed parts, each text part separately.
- **Decode whole replies.** One token from `generate(onToken:)` can be part of a multi-byte character; decode the
  whole reply or a growing prefix.
- **Qualification.** `Forge/language/reference_chat_tokens.py` records upstream Transformers' `apply_chat_template`
  for pinned snapshots of Qwen/Qwen3-0.6B, huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated,
  Qwen/Qwen3-VL-2B-Instruct and huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated (Apache-2.0): text-only, system, multi-turn, Unicode, thinking on and
  image placeholders before and after text. With `AETHER_LANGUAGE_TOKENIZER_HUB` set to a hub cache holding
  them, all 24 cases match token for token, as does the decode check. Qwen3's thinking-off render equals the fixed
  render FLUX.2 used until 0.0.2. Since 0.0.3, `Flux2Tokenizer` renders through this product with the same token IDs
  ([FLUX.2 guide](flux2-klein.md#chat-template-in-the-fingerprint-003)).

## State and ownership

Each actor admits one request; overlap fails. Submitted work is joined before
reading or changing buffers, including cancellation. Starting a new request resets
its own cache. Registered EOS IDs stop decoding. A cancelled generation throws after
the native join.

`unload()` closes the session permanently. New requests fail with `closed`. An
active request finishes and joins its native work first (cancel its task to end a
long generation sooner), and then the model, function, compute stream and KV caches
are released. Concurrent calls return once the release completes; open a new
session to load the model again.

`generate(onToken:)` streams each emitted token while the next decode step runs on
the GPU, so consumer work (for example a main-actor UI update) no longer idles the
device between tokens. Deliveries stay in order with at most one in flight, and
every delivery finishes before `generate` returns or throws. If a delivery throws,
the step already submitted alongside it joins first and its result is discarded; if
that step fails, the in-flight delivery is cancelled and awaited.

## Sampling and the backend protocol

`AetherLanguageRuntime` holds what doesn't depend on Core AI: the `LanguageBackend` protocol, `LanguageSampling`, the
constraint hook and the decode loop every backend shares ([design](language-runtime-seam.md)). It imports only the
standard library and Foundation, so it builds at the package floor (macOS 14 and iOS 17) without availability
annotations, and `AetherLanguageCoreAI` re-exports it. `LanguageGenerationSession` is its Core
AI backend:

```swift
let request = LanguageGenerationRequest(prompt: prompt.tokenIDs, maximumTokens: 512,
                                        sampling: .random(temperature: 0.7, topK: 40, topP: 0.95, seed: 7))
let result = try await session.generate(request) { token in
    // token.id streams while the next step runs; return .stop to end the request (for example at a stop string).
    .continue
}
// result.seed replays the reply; result.stopReason is .endOfSequence, .outputLimit or .stopped.
```

- **Greedy stays the default, token for token.** A request with no sampling options, like `generate(prompt:...)`, runs
  the same argmax on the same logits as before: the Qwen3-0.6B and 4B writer replies are identical before and after the
  change ([evidence](evidence/language-runtime-seam-2026-10-27.json)).
- **Sampling follows Transformers' order:** repetition penalty (over the whole request, prompt included; off by
  default), the request's constraint, then temperature, top-k and top-p, and one seeded draw. The warpers match
  Transformers 4.57.3's own on recorded fixtures (`Forge/language/reference_sampling.py`).
- **A seed replays exactly** for the same asset, device class, OS and prefill plan, across memory-budget reloads and
  in a newly opened session; without one, the session draws a seed and reports it.
- **Constraints** (`LanguageLogitsProcessor`) return the allowed tokens for the next step and run while the current step
  runs on the GPU: a 5 ms-per-token processor added nothing measurable on either model, while one slower than the step
  sets the pace. The loop applies the allowed set after the join; disallowed tokens become −∞ before temperature and
  truncation.
- **What it costs:** sampling runs on the host after each join and adds no input, shape or submission, so step shapes,
  the memory budget and GPU admission are those of greedy decoding. On the 4B writer, random modes added 0.4–1.4 ms to a
  21 ms token.
- **Image prompts** take the same options: `VisionLanguageSession.generate(renderedTokenIDs:images:maximumTokens:sampling:constraint:...)`
  and the image `generate(prompt:imageTokens:...:sampling:...)` run through the same decode loop and return a
  `LanguageGenerationResult`. The repetition penalty and the processor's `begin(prompt:)` see the expanded prompt (each
  image's pad token repeated to its token count), as Transformers' `input_ids` do. The greedy entry points are
  unchanged, token for token, on one-function and two-function Qwen3-VL bundles
  ([evidence](evidence/qwen3-vl-sampling-2026-10-27.json)).
- **Next:** the facade's tokenizer, templates and structured output.

## Resource admission

`LanguageGenerationSession.open(bundle:…, resources:)` and `VisionLanguageSession.open(bundle:…, resources:)` take an
opt-in `LanguageResourceAdmission`: caller estimates (`LanguageModelResourceEstimate`, keyed by the registration's
model hash) reserved against a host-selected `ResourceCoordinator`, as Krea and FLUX.2 do.
- **Residency** is reserved before the model loads (after the open grant, with `gpu:`), held while it is loaded,
  across memory-budget reloads, and released on unload or a failed load. For a decoder it covers the weights, the KV
  caches at the registered context and the memory budget's growth allowance.
- **Scratch** is reserved after each request's grant (and each image encode's) and released when its native work
  joins. A request's is reserved before a memory-budget reload, so a refusal never costs a model load.
- **Refusals before native work:** model bytes without an estimate and a registered context above the profile's (a
  bundle's before its files are hashed), and a request with more image tokens than the profile allows, summed over its
  images (every image is preprocessed and the total checked before the first encode).
- **Opening is admitted.** With `gpu:`, the model's load and its functions' specialization run under a grant, so a
  session never specializes beside another owner's GPU work.
- **Per process.** The ledger cannot see other processes, and it leaves out the page cache a model's compiled file is
  read into at open.

Measured estimates for the Qwen3-VL bundles, and why the feature exists, are in
[Qwen3-VL resource admission](qwen3-vl-resource-admission.md).

## Step contracts

The session detects the contract from the `main` function signature
(`LanguageGenerationSession.contract`); running-contract bundles need no manifest
change, and bucketed ones declare their `exportBounds`. A Qwen3-VL bucketed `main` also takes
`rope_positions` and, since 0.0.3, image rows aligned with the tokens (`image_embeds`,
`deepstack`), so one function serves text, image and decode steps
(`imagesThroughMain`; [Qwen3-VL](qwen3-vl.md#pieces)).

| | `runningPositions` (0.0.1, Apple LLM exports) | `bucketedPositions` (0.0.2) |
| --- | --- | --- |
| Positions | `position_ids` [1, total] = 0..<total | `cache_positions` [1, q] = start..<start+q |
| KV cache | states `keyCache`/`valueCache`, written in place | read-only inputs; outputs `keyUpdate`/`valueUpdate` [layers, 1, heads, q, dim], copied into rows start..<start+q after the join |
| Logits | [1, q, vocab] | [1, 1, vocab] |
| Prefill submissions | uniform `prefillChunkSize` chunks (within a declared chunk bound) | one width C (`prefillChunkSize`, default 64, within the chunk bound); a run that is not a multiple of C ends with a C-token chunk stepping back over rows already written; a run shorter than C is one chunk |
| Attention reads | the `total` materialized rows | every cache row (the real length rounded up to 256), concatenated with the chunk's |
| Input shapes | a new `total` every step | (q, cache capacity): q is C, 1 or a run shorter than C, times one capacity per 256 tokens of context |
| Shape recycling | every 128 submissions by default | none by default |
| Memory-budget reloads | between requests, past the budget | between requests, past the budget |

Because the running contract submits a new input shape on every decode step, Kit
recycles its native model/function/stream ownership after a bounded number of
joined submissions (128 by default), preserving the Kit-owned KV state; this bounds
the lifetime of native shape caches. The bucketed contract's shapes form a bounded
set (decoding adds one shape per 256 tokens it crosses), so
`maximumSubmissionsPerModel` defaults to nil (never) for it. A bounded shape set does
not bound memory, though: both contracts are also reloaded between requests once
Core AI's retention on shape changes passes the memory budget (see
[Native memory retention](#native-memory-retention)). Either policy is
explicit and can be overridden on raw-asset opening; reload counts and the number of
distinct submitted (q, capacity) pairs are exposed in cache statistics. A bucketed
asset's position input has a different name, so a 0.0.1 loader rejects it instead of
misreading it. Asset files must remain immutable for the lifetime of the session.

The bucketed graph never writes the cache: each layer attends with one fused SDPA
over [cache ‖ chunk] keys under a boolean mask built from position data, and
returns the chunk's keys and values. The session passes the caches as zero-copy
contiguous inputs and grows them in 256-row steps with the real length (a strided
view into a larger buffer measured 3–8× slower: Core AI copies non-contiguous inputs
every step). The [cache ‖ chunk] concatenation copies every cache row each step, so a
bucketed step costs more than a running one and the gap grows with the context; what
the contract buys is a bounded shape set — no shape recycling, no new shapes once warm —
not faster steps. It does not avoid the memory-budget reloads: those come from Core AI
retaining memory on every shape change, and a request always changes shape. Kit-side
export tooling for Qwen3 lives in `Forge/language/` (development only, not a
shipping dependency); its module documentation records the rejected alternatives and
their measurements.

The prefill keeps every chunk at one width C so that a warm request changes shape as
little as possible (see [Native memory retention](#native-memory-retention)). A text run
that is not a multiple of C ends with a C-token chunk over its last C tokens, starting
before rows the previous chunk already wrote. This is the same computation, not an
approximation: the graph's cache mask admits only rows before the chunk's first position
(`key_positions < cache_positions[0]`), so the stepped-over rows are invisible to the chunk
and are recomputed from the same prefix, then overwritten in the cache with the chunk's
own keys and values. They can differ from the first computation only by floating-point
rounding (a different grouping of the same operations), as any change of chunking does.
The cost is up to C − 1 recomputed tokens per run. A run shorter than C (a short prompt,
or a short text run beside a Qwen3-VL image) is one chunk of its own length,
and a chunk never steps back into an image span. `languageOverlappedPrefillMatchesOtherPlans`
(opt-in) checks this on every model against a whole-prompt chunk and 1-token steps.

### Time to first token

The bounded shape set costs time to first token on long prompts
([evidence](evidence/language-time-to-first-token-2026-10-27.json), `languageTimeToFirstToken`, opt-in). These are warm
medians on one M5 Max, a shared host, for one-token requests, in milliseconds. The running export of the 0.6B was made for
the comparison; there is no running export of the 4B:

| Prompt tokens | 0.6B bucketed (C = 64) | 0.6B running | 0.6B bucketed, C = 512 | 4B bucketed (C = 64) | 4B bucketed, C = 512 |
| --- | --- | --- | --- | --- | --- |
| 64 | 14 | 14 | 25 | 31 | 39 |
| 256 | 46 | 17 | 25 | 121 | 66 |
| 1,024 | 216 | 40 | 86 | 530 | 280 |
| 2,048 | 496 | 83 | 187 | 1,178 | 604 |

**Where the time goes.** Almost all of it is in the native steps: a 2,048-token prompt is 32 submissions at C = 64, and
each one reads every weight and concatenates the whole cache, where the running contract's four 512-token submissions
do far less.
- **Cache growth,** the host's largest share, took 83 ms (0.6B) and 133 ms (4B) at 2,048 tokens before these changes.
  That's bound by first touches of each newly allocated buffer, since a bucketed request starts again at 256 rows.
- **Update-row copies and input building** took under 20 ms before these changes.

Growth now zeroes only each row's unwritten tail, copies rows in parallel, and copies a step's update rows one row at a
time instead of one token at a time. All three write the same bytes, and replies are identical. The default-width
figures above are with these changes (before: 554 and 1,383 ms at 2,048 tokens, partly host noise).

**Rejected.** Reusing large buffers across growths and requests cut growth by 57 ms (0.6B) and 68 ms (4B) at 2,048
tokens, but Core AI's encode rose by 51 and 58 ms, so time to first token didn't improve
([probe](evidence/language-ttft-buffer-reuse-probe-2026-10-27.json)). Core AI pays for an input buffer larger than its
view, as for the strided views above.

**What would close the gap** isn't exact, so it isn't done:
- **A wider chunk for long prompts.** C = 512 halves the 4B's time at 2,048 tokens. But it retains more memory per request
  (see [Native memory retention](#native-memory-retention)), and a different prefill plan can flip a greedy near-tie.
- **An export whose attention reads the cache and the chunk separately,** instead of concatenating them every chunk.
  It is a no-go under the current Core AI: a second function wires a second copy of the weights, split attention in
  `main` halves decode, and a `torch.cond` between the two hangs the GPU compile
  ([spike report](spikes/split-attention-2026-10.md)).

## Cache allocation

The registered context is an admission maximum. Allocation starts at 256 tokens
and grows with actual materialization. KV prefixes are copied row by row across
layer/head strides. Growth doubles small caches; above 4K it uses 1K increments to
bound transient allocation size. Bucketed caches instead grow in 256-row steps,
because bucketed attention reads every allocated row. Each step copies the whole
cache into the larger one, so growth copies add up over a long generation (estimated, not measured: about
1 GB by 2K tokens and tens of GB over a full 16K generation for Qwen3-0.6B);
larger steps would trade per-step attention cost for fewer copies. This reduces eager allocation
but does not promise that every registered context fits every device’s current memory
budget. A bucketed session starts every request again at the initial capacity rather
than keeping a previous long request's caches; running-contract sessions keep their
grown caches.

Raw-asset opening and teacher-forced logits are available for explicit qualification
before registration. Registration itself is not evidence of correctness.

## Native memory retention

This section describes a workaround for a Core AI bug, not a design choice. Core AI keeps
Metal allocations every time a function specialized with `expectFrequentReshapes` (as every
language session's is) changes input shape, and returns them only when the model is
released. A request always changes shape (its prefill chunks, then 1-token decode steps), so
a warm session keeps growing on every request even though every shape is already warm.
Repeating one shape retains nothing ([reproduction](#minimal-reproduction-for-apple)). The
memory is not Kit's: new KV buffers, reused inputs, Kit-owned output views, autorelease pools
and a fresh compute stream per request all left it unchanged. Specializing without
`expectFrequentReshapes` retains nothing per request, but allocated 34 GiB at the writer's
first request. The real fix is Apple's.

**What you see.** Memory stays bounded at the post-warm level plus a budget (default 2 GiB).
When requests have retained that much, the session reloads the model, usually at idle right
after the request that crossed the budget, so the next request does not wait for it. A
request that arrives during an idle reload waits for the rest of it. A reload costs the
model load plus the first use of each function again: about 2.6 s for the writer. With the
default prefill width and budget, the writer reloads about once every 35 back-to-back
449-token requests, and less often with shorter or mixed prompts. Replies are identical,
token for token, across reloads (the KV state is Kit-owned and unaffected).

**Prefill width.** The retained amount follows the shape changed to: about 0.85 MiB per
token of prefill width on the writer. The bucketed contract therefore prefills in one fixed
width C (64), so a warm request switches only between q = C and the 1-token step; see
[Step contracts](#step-contracts) for the step-back tail that keeps every chunk at C. C was
chosen on the writer (`languagePrefillChunkSweep`, opt-in; one M5 Max; 8 tokens per
request; retention without a budget; reloads over 60 long requests at the 2 GiB default):

| C | Retained per 449-token request | Per request, alternating 12/449 | Median time to first token, 449 / 12 tokens | Reloads per 60 long requests |
| --- | --- | --- | --- | --- |
| 32 | 32.5 MiB | 20.0 MiB | 0.349 / 0.022 s | 0 |
| **64** | **58.6 MiB** | **33.1 MiB** | **0.206 / 0.022 s** | **1** |
| 128 | 110.8 MiB | 59.2 MiB | 0.126 / 0.022 s | 3 |
| 256 | 215.3 MiB | 111.5 MiB | 0.124 / 0.023 s | 5 |
| power-of-two chunks (0.0.2 before this change) | 137 MiB | not measured | about 0.28 s / not measured (a more loaded host) | about 4 (estimated from retention) |

64 retains 2.3 times less than the earlier power-of-two plan and reaches the first token
sooner. 128 saves 80 ms on a 449-token prompt but triples the reload rate. Replies were
identical at every width.

**The budget.** `memoryGrowthBudgetBytes` is on both `open` entry points and
`VisionLanguageSession.open`: default `LanguageGenerationSession.defaultMemoryGrowthBudgetBytes`
(2 GiB), nil disables it, and budgets below `minimumMemoryGrowthBudgetBytes` (256 MiB) are
refused. Each request is measured from its GPU admission to its end: Metal's allocated size,
net of the session's own KV cache changes. When a request ends with the growth counted since
the model loaded past the budget, the session starts the reload at once, in the background,
under its own GPU grant (`LanguageRequestAdmission`):
- a request arriving while it reloads waits for it;
- a request admitted before the idle reload holds its grant cancels it and reloads in front,
  under its own grant (waiting would deadlock on that grant); so does any request finding the
  budget still spent because an idle reload was cancelled or failed;
- `unload()` withdraws an idle reload or request still waiting for GPU admission, cancels an
  idle reload (a load in progress finishes and is dropped) and ends a request reloading in
  front with `closed` when its reload returns: close never waits behind the GPU queue, at
  most for one model load.

Reloads never run inside a request and never load while another owner holds the GPU.
`CacheStatistics` reports `modelReloadCount` (all reloads), `idleMemoryReloads` and
`requestMemoryReloads`, `retainedMemoryBytes` and the budget;
`VisionLanguageSession.decoderStatistics()` reports its decoder's. A session made with
`init(model:)` cannot reload and has no budget.

Metal's allocated size is process-wide, so the accounting sets aside what is not this
model's retention (`LanguageMemoryBudget`):
- a request that submits a function or (q, capacity) shape for the first time since the
  load is not counted (the weight copy or specialization);
- a net decrease is another allocator's (the session frees nothing of its own inside a
  window) and never offsets the count;
- growth above half the budget and above four times the request's expected growth (its
  submitted tokens times the median per-token rate of recent counted requests, once
  three are known) is another allocator's, until such outliers since the load reach the
  budget: then every further outlier counts until the next reload. An outlier is above
  half the budget, so at most two are set aside per load, and retention that always
  looks like an outlier counts from the third.
Set-aside requests are counted in `foreignMemoryRequests` and `foreignMemoryBytes`.
Sharing a `GPUAdmission` between a host's sessions keeps their admitted GPU work out of
each other's windows; opening is admitted too, but releasing is not, so another session's
release inside a window is caught only by the net-decrease rule. Without `gpu:`, the accounting
assumes no concurrent Metal work beyond what those rules catch. Before a model's rate is
known, a foreign load inside a window counts and can cause one unnecessary reload; this
errs towards reloading, because uncounted retention would grow without bound.

**Bound qualification** (`languageMemoryBudgetBoundsRepeatedRequests`, opt-in; one M5 Max, release build,
shared host; [evidence](evidence/language-memory-budget-2026-10-27.json)). Each model is opened as an application
would and warmed, then serves 60 back-to-back requests of 8 tokens: a 449-token text prompt (453 for the 0.6B), and
for the Qwen3-VL bundles alternating with a one-line question about a 512×384 synthetic image. Gate: Metal after every request
at most the post-warm level plus the budget (2 GiB) plus the largest warm request's growth, at least one reload, and
every reply identical to the first of its kind.

| Model | Warm | Highest after a request | Bound | Reloads (at idle / in front) | First reload after request | Replies identical |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | 1,557 MiB | 3,631 MiB | 3,642 MiB | 1 (1 / 0) | 58 | 60 of 60 |
| Qwen3-4B writer | 8,581 MiB | 10,631 MiB | 10,688 MiB | 1 (1 / 0) | 36 | 60 of 60 |
| Qwen3-VL-2B | 10,113 MiB | 12,174 MiB | 12,222 MiB | 1 (1 / 0) | 58 | 60 of 60 |
| Qwen3-VL-4B describer | 19,566 MiB | 21,705 MiB | 21,918 MiB | 1 (1 / 0) | 34 | 60 of 60 |
| Qwen3-VL-2B, one function (0.0.3) | 6,065 MiB | 8,066 MiB | 8,180 MiB | 1 (1 / 0) | 34 | 60 of 60 |
| Qwen3-VL-4B, one function (0.0.3) | 10,847 MiB | 12,822 MiB | 13,311 MiB | 2 (2 / 0) | 20 | 60 of 60 |

The one-function rows are the re-exported Qwen3-VL bundles
([evidence](evidence/qwen3-vl-one-function-2026-10-27.json), [Qwen3-VL](qwen3-vl.md#one-decoder-function-003)).
Every reload ran at idle; none delayed a request. Without the budget, the same requests grow without limit (the
writer by 58.6 MiB per request at this width, 3.4 GiB over 60). Upstream parity was re-run at the fixed width for all
four models ([evidence](evidence/language-prefill-width-2026-10-27.json)); see
[Evidence and limits](#evidence-and-limits) for the two greedy near-tie flips it found.

### Minimal reproduction for Apple

`coreAIRetainsMemoryPerReshapeUntilRelease` (opt-in;
[record](evidence/core-ai-reshape-retention-2026-10-27.json)) uses Core AI alone. It loads
the bucketed Qwen3-0.6B asset with `SpecializationOptions(preferredComputeUnitKind: .gpu)`
and `expectFrequentReshapes = true`, loads its `main` function, and encodes submissions
one at a time on one `ComputeStream`, joining each with `currentWorkCompleted()` and
reading its logits. Every submission is a fresh set of `NDArray` inputs: token IDs and
cache positions [1, q] and zeroed [28, 1, 8, 256, 128] key and value caches. Metal's
`currentAllocatedSize` after each submission (one M5 Max, 40 submissions per phase):

| Specialization | 40 × q = 256 | then 40 alternating q = 256, 1 | after releasing function, stream and model |
| --- | --- | --- | --- |
| `expectFrequentReshapes = true` | 1,554 MiB, flat | 1,554 → 2,751 MiB: +61 MiB per change to 256, +2 MiB per change to 1 | 28 MiB |
| `expectFrequentReshapes = false` | 1,669 MiB, flat | 3,108 MiB after the first q = 1 (its specialization), then flat | 90 MiB |

Expected: once both shapes have run, alternating between them allocates nothing more, as
without `expectFrequentReshapes`. Observed: every shape change retains memory until the
model is released. The Feedback is the user's to file.

## On-device host (prepared, 0.0.3)

`Examples/LanguageDevice` is an XcodeGen app, LanguageDeviceProbe. It opens a staged bundle with `platform: .ios` and
checks it against the float32 Transformers reference, printing one JSON report with timings, footprint and the app's
available memory.

- **Text** (a dense Qwen3): every teacher-forced row and a greedy continuation, judged by the native qualification's
  rule.
- **Vision** (Qwen3-VL): Kit decodes and preprocesses the photos the reference was made from (the `kit-decoded`
  reference grids). The greedy answer must match, or first diverge at a near-tie of the reference's own row.

`Forge/language/stage_device_assets.py text|vision BUNDLE REFERENCE` copies the model asset(s) and tokenizer, and
registers them again for iOS with the same registration script and source. The asset is not re-exported, and the iOS
registration differs from the macOS one only in `platform`. Staged, the Qwen3-0.6B is 1.25 GB and the Qwen3-VL-2B
4.29 GB. The app builds for a generic iOS device; no device has run it yet. The VL-2B's weights exceed the app memory
an iPhone grants by default, so the app requests the increased memory limit
(`com.apple.developer.kernel.increased-memory-limit`, in `LanguageDeviceProbe.entitlements`). That capability needs an
explicit App ID with it enabled; a wildcard App ID can't carry it. Explicit App IDs are unique across teams, so outside
the owner's team, first change `PRODUCT_BUNDLE_IDENTIFIER` in `project.yml` to an ID in your own team. Then either
register that ID with the capability in your developer account, or let Xcode register it: select your team for the
target in the generated project (the spec sets none) and build once with automatic signing. The team selection lives in
the generated, git-ignored `.xcodeproj`, so the next `xcodegen generate` resets it. The report's
`availableMemoryMiBAtStart` shows whether the larger limit applies. Without it, the VL-2B run may end in a refusal or a
jetsam kill.

## Evidence and limits

Different prefill plans (a different width, or 1-token steps) round differently, which can flip a greedy token
where upstream's top two are nearly tied (observed margins 0.0014 and 0.0115). This is kernel rounding, not a
step-back effect: the default width and a single whole-prompt chunk give identical tokens on every qualified model
([evidence](evidence/language-prefill-width-2026-10-27.json)).

### Qwen3-4B writer (bucketed, fp16)

Atelier's writer, `huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated` (Apache-2.0, revision `c9bd464`, an
abliterated fine-tune of base model `Qwen/Qwen3-4B-Instruct-2507`: same architecture, no thinking mode), qualified on the bucketed
contract alone ([evidence](evidence/language-qwen3-4b-writer-2026-10-26.json)); M5 Max (`h17c`), GPU-preferred,
release build, shared host:

- **Export:** `export_qwen3.py` (bucketed, prefill chunk 512, capacity 16,384), 7.5 GB; its cpu_only self-check
  matches the eager module (minimum cosine 0.99998, argmax equal, no lost cache rows).
  `register_qwen3_bundle.py` registers the bundle with its base model and every tokenizer file (`tokenizerFiles`;
  dotfiles skipped, symbolic links refused, as `verifyTokenizer` sees them).
- **Reference:** upstream Transformers float32 on CPU (`reference_qwen3.py`, peak RSS 18 GB, now freeing the upstream
  model before loading the eager port). The eager port matches it on all 32 forced rows (cosine 1.0).
- **Parity:** 32 teacher-forced rows: minimum cosine 0.9999979, maximum absolute logit difference 0.036, no argmax
  differences and no near-ties. The 64-token greedy continuation is identical to the reference.
- **Long run:** 1,024 decode steps in one request: no model reloads, eight distinct (q, capacity) submissions,
  capacity 1,280.
- **Chat round trip** (`languageChatBundleRendersGeneratesAndDecodes`): `LanguageTokenizer` renders two short
  instruction prompts (72 and 73 tokens) with the model's template, `open(bundle:)` decodes them greedily, and both
  replies stop at `<|im_end|>`; a repeated prompt reproduces its tokens. Memory, sampled after each request (not a
  continuous peak): Metal allocations reach 8,570 MiB and return to 16 MiB after `unload()`; the process footprint
  (which does not count Core AI's model memory) reaches 400 MiB. Warm requests reach the first token in 0.05–0.07 s
  and decode about 50 tokens/s. In the recorded run the open took 24 s and the first request 22 s to its first token
  on the shared host; expect the first request after opening to be slow.

Not covered for the 4B: phone, other quantizations, long contexts beyond 1,280 tokens of cache, and cancellation or
unload under load (the lifecycle code paths are those the 0.6B qualification exercises).

### Qwen3-0.6B decode contracts

CPU tests cover cache-growth admission, per-row prefix preservation, platform checks,
path boundaries, asset integrity, contract detection, export-bound admission and
registration, position inputs, fixed-width prefill planning with the step-back tail within
declared chunk bounds over every prompt length to 2,048, bucketed capacity steps and the shape count
over a 16K context, request capacity reset, placement of returned cache rows for
nested and per-row update layouts, and that a failed cache write does not advance the
materialized length.

The bucketed contract's native qualification
([evidence](evidence/language-bucketed-decode-2026-10-26.json), opt-in test
`languageBucketedDecodeQualification`) used Qwen3-0.6B (Apache-2.0, revision
`c1899de`) exported at fp16 in both contracts from the same checkpoint by
`Forge/language`, on one M5 Max (`h17c`), GPU-preferred, release build:

- Teacher-forced logits for 64 rows against upstream Transformers float32 on CPU:
  minimum cosine 0.9999985; one argmax difference, at a reference near-tie (top-2
  probability gap 0.00015). The running export behaves the same.
- Greedy continuation: identical to the reference until that near-tie (token 58).
  Over 1,024 decoded tokens the bucketed and running exports agree token for token.
- 1,024 decode steps in one request: no model reloads (running: 8); six distinct
  (q, capacity) submissions; footprint grows only with the cache (bucketed 366→554
  MiB at capacity 1,280; running 367→564 MiB at 2,048).
- Paired, interleaved warm timing against the running export both with its default
  recycling (what 0.0.1 consumers get) and with recycling disabled (its pure step
  cost); six rounds, 256 greedy tokens. The host was heavily loaded (load average
  ≈50), so absolute rates are low and only the same-round ratios are meaningful:
  - per step, bucketed decode is 0.88× the running contract without recycling after
    both a 33-token and a 1,000-token prompt (median of per-round ratios 0.879 and
    0.876);
  - against default recycling it is 1.43× and 1.37×, which is the avoided reload
    cost, not faster steps;
  - time to first token is 2.1× and 3.0–3.2× longer than either running variant
    (more, power-of-two prefill submissions, each concatenating the cache; measured
    before the fixed prefill width, which reaches the first token sooner).
- 8K context (7,168-token prompt, 512 decoded tokens): bucketed decodes at 0.84× the
  unrecycled running contract and 1.22× the recycled one, with a 2.8× longer time to
  first token. Its process footprint is the lowest (3.3 GiB, against 4.1 GiB recycled
  and 5.9 GiB unrecycled: without recycling the running contract's native shape cache
  grows by about 1.8 GiB), with no reloads and 16 distinct shapes.
- Both exports pass the Forge cpu_only self-check (cosine ≥ 0.99, argmax equal, no
  lost cache rows) and declare their export bounds (chunk 512, capacity 16,384).
- GPU admission (`languageSessionGPUAdmissionAdmitsAndWithdraws`, opt-in): with a
  `gpu:` owner, the bucketed session holds one grant for exactly the request's
  native work, and `unload()` withdraws a request queued behind another holder (it
  throws `closed`) without waiting for the GPU.

Not yet covered: phone, other model sizes or quantizations, a lower prefill
submission count (padding prompts to a bucket), and cancellation or unload
qualification specific to the bucketed contract (the lifecycle code paths are shared
with the running contract). Shilo holds device reports and original-checkpoint
reference fixtures under its `.local/` directory. Its independent oracle uses upstream
Transformers on CPU; Python is development tooling and is absent from shipping targets.

As of 2026-10-26, the Mac consumer has exercised 16K context with the 1.7B and 4B
candidates and completed the larger-budget reasoning evaluation. Phone context and
memory qualification is tracked separately. Earlier bounded Shilo fixtures cover
cancellation/recovery and numerical chunking through 2K; larger contexts and larger
budgets do not automatically inherit all lifecycle guarantees. General model quality,
optimized generation performance, streaming and prefix reuse remain
follow-up work driven by the chat consumer.
