# Experimental language generation

`AetherLanguageCoreAI` is Aether Kit’s local, sequential dense-language reference
runtime. Shilo consumes this product rather than carrying a second inference
implementation. It currently accepts the demonstrated dynamic token/position,
FP16 logits and contiguous five-dimensional KV-cache contract. This is an
experimental consumer-driven API, with no compatibility promise yet.

## Local registration and execution

A caller supplies `language-model.json` in the model bundle. The manifest names
relative model/tokenizer directories, immutable upstream source identity, SHA-256
hashes, platform, quantization, stop IDs and a maximum context. `LanguageModelManifest`
lives in the Core-AI-independent `AetherKit` target. Verification rejects mismatched
platforms, integrity failures and paths escaping the local bundle.

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

## State and ownership

Each actor admits one request; overlap fails. Submitted work is joined before
reading or changing buffers, including cancellation. Starting a new request resets
its own cache. Prefill uses the full running position range, and registered EOS IDs
stop decoding. A cancelled generation throws after the native join; partial token
streaming and explicit close/unload are not exposed by this initial API.

For raw-asset opening, Kit can recycle native model/function/stream ownership
after a bounded number of joined submissions (128 by default), preserving the
Kit-owned KV state. This bounds the lifetime of native shape caches. The policy
is explicit and reload counts are exposed in cache statistics. Asset files must
remain immutable for the lifetime of the session.

The registered context is an admission maximum. Allocation starts at 256 tokens
and grows with actual materialization. KV prefixes are copied row by row across
layer/head strides. Growth doubles small caches; above 4K it uses 1K increments to
bound transient allocation size. This reduces eager allocation but does not promise
that every registered context fits every device’s current memory budget.

Raw-asset opening and teacher-forced logits are available for explicit qualification
before registration. Registration itself is not evidence of correctness.

## Evidence and limits

CPU tests cover cache-growth admission, per-row prefix preservation, platform checks,
path boundaries and asset integrity. Shilo holds device reports and original-checkpoint
reference fixtures under its `.local/` directory. Its independent oracle uses upstream
Transformers on CPU; Python is development tooling and is absent from shipping targets.

As of 2026-10-26, the Mac consumer has exercised 16K context with the 1.7B and 4B
candidates and completed the larger-budget reasoning evaluation. Phone context and
memory qualification is tracked separately. Earlier bounded Shilo fixtures cover
cancellation/recovery and numerical chunking through 2K; larger contexts and larger
budgets do not automatically inherit all lifecycle guarantees. General model quality,
optimized generation performance, streaming, prefix reuse and explicit unload remain
follow-up work driven by the chat consumer.
