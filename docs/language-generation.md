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
  them, all 24 cases match token for token, as does the decode check. Qwen3's thinking-off render equals FLUX.2's
  `Flux2Prompt.render`, so FLUX can delegate to this product without changing a token.

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

## Step contracts

The session detects the contract from the `main` function signature
(`LanguageGenerationSession.contract`); running-contract bundles need no manifest
change, and bucketed ones declare their `exportBounds`.

| | `runningPositions` (0.0.1, Apple LLM exports) | `bucketedPositions` (0.0.2) |
| --- | --- | --- |
| Positions | `position_ids` [1, total] = 0..<total | `cache_positions` [1, q] = start..<start+q |
| KV cache | states `keyCache`/`valueCache`, written in place | read-only inputs; outputs `keyUpdate`/`valueUpdate` [layers, 1, heads, q, dim], copied into rows start..<start+q after the join |
| Logits | [1, q, vocab] | [1, 1, vocab] |
| Prefill submissions | uniform `prefillChunkSize` chunks (within a declared chunk bound) | largest power of two that fits the remainder, `prefillChunkSize` and the chunk bound |
| Attention reads | the `total` materialized rows | every cache row (the real length rounded up to 256), concatenated with the chunk's |
| Input shapes | a new `total` every step | (q, cache capacity): at most log2(chunk bound) + 1 q sizes times one capacity per 256 tokens of context |
| Native recycling | every 128 submissions by default | none by default |

Because the running contract submits a new input shape on every decode step, Kit
recycles its native model/function/stream ownership after a bounded number of
joined submissions (128 by default), preserving the Kit-owned KV state; this bounds
the lifetime of native shape caches. The bucketed contract's shapes form a bounded
set (decoding adds one shape per 256 tokens it crosses), so
`maximumSubmissionsPerModel` defaults to nil (never) for it. Either policy is
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
the contract buys is a bounded shape set — no reloads, no shape-cache growth — not
faster steps. Kit-side export tooling for Qwen3 lives in `Forge/language/` (development only, not a
shipping dependency); its module documentation records the rejected alternatives and
their measurements.

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

## Evidence and limits

CPU tests cover cache-growth admission, per-row prefix preservation, platform checks,
path boundaries, asset integrity, contract detection, export-bound admission and
registration, position inputs, power-of-two prefill planning within declared chunk
bounds over every prompt length to 2,048, bucketed capacity steps and the shape count
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
    (more, power-of-two prefill submissions, each concatenating the cache).
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
