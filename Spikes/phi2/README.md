# Phi-2 v0.0.6 spike

Port of `microsoft/phi-2` at revision
`9a5117d15e407c48fa40f8adbac1244abae52d6f` (MIT), through Aether's existing
language bundle format and `LanguageGenerationSession` completion API.

Phi-2 adds a parallel attention/MLP residual block, LayerNorm, biased Q/K/V and
output projections, and partial rotary embeddings (32 of each head's 80 channels).
The shared dense decoder now accepts a rotary dimension and an optional output
head bias. Family blocks continue to own their normalization, projections and
residual topology. The existing running and bucketed caches, causal masks,
exporter, reference recorder, registration and Swift decode loop are reused.
Future families can compose these options without adding model dispatch to Swift.

Phi-2 also exposed a native runtime assumption: Core AI pads its 80-channel heads,
so its preferred cache strides are not contiguous. Host updates and cache growth
now honor noninterleaved strides, including padding and channel strides. The
existing contiguous fast path is retained. Growth copies only materialized rows
and clears padding and the unused tail. A synthetic layout regression and a real
224-token prompt plus 64 decode tokens exercise this path across 256 → 512 rows.

The `phi` registry entry exports the Phi architecture; this spike qualifies the
original Phi-2 checkpoint. Scaled RoPE, fp16 attention and other activations are refused. Optional
per-head Q/K LayerNorm, GQA and tied embeddings are checked with synthetic models;
that does not qualify arbitrary Phi fine-tunes. Context is capped at the trained
2,048 tokens. No chat template is invented; the chat facade reports
`unsupportedTemplate` for this base completion model.

## Reproduce

Use Forge's pinned environment (`uv sync --frozen --project Forge`). Download the
pinned snapshot's safetensors, configuration, tokenizer and LICENSE files into a
local checkpoint directory. No remote model code is executed.

```sh
OMP_NUM_THREADS=4 sh Spikes/phi2/qualify.sh /absolute/checkpoint /absolute/new-output
```

`AETHER_FORGE_PYTHON` may point to an existing environment with the Forge pins.
The script refuses existing outputs, exports fp16 weights/cache/logits with float32 arithmetic,
registers a macOS bundle, and compares short, long and cache-growth prompts with the upstream
Transformers float32 reference. Native tests take the shared GPU lease.

```swift
import AetherLanguageCoreAI
import AetherLanguageTokenizers

let tokenizer = try await LanguageTokenizer.open(bundle: bundleURL)
let session = try await LanguageGenerationSession.open(bundle: bundleURL, platform: .macos)
let result = try await session.generate(
    prompt: tokenizer.encode("def fibonacci(n):"), maximumTokens: 64)
print(tokenizer.decode(result.tokens))
await session.unload()
```

The synthetic parity tests exercise both contracts, chunked prefill, decode,
cache growth and poisoned future rows, the checkpoint's 0.4 rotary factor, full
rotation, GQA, tied/untied heads, nonuniform norm gains and nonzero biases. Tiny
Core AI exports gate exact argmax and no lost cache rows for all dense families.
Real-weight float32 references require every row's argmax to agree and cosine
>= 0.999999 with `atol=0.01, rtol=0` for Phi only. Transformers itself differs
by up to 0.00815 between full-sequence and cached float32 execution on this
checkpoint; `upstream_roundoff.py` records this independently of the port. The
other families retain their existing tolerances. Native fp16 follows the existing
language qualification policy (cosine >= 0.995; differing argmax only when the
reference probability margin is < 0.1 and native choice is in the reference top
five). Free generation is compared at its first divergence and must repeat across
prefill widths 7 and 16. No iOS qualification is claimed.

## Precision findings

Direct fp16 residual/projection arithmetic failed the full checkpoint's Core AI
CPU gate despite passing tiny tests. Promoting normalization alone did not fix
it. Phi uses explicit float32 projections, normalization, residuals and attention,
with fp16 weight storage, KV cache rows and public logits. Export reports record
this arithmetic policy. Rotary frequencies match upstream's float32 construction.
This is specific to Phi; the qualified Qwen3, OLMo2 and GPT-2 recipes are unchanged.

To reproduce the upstream-only roundoff baseline against any reference directory:

```sh
OMP_NUM_THREADS=4 Forge/.venv/bin/python -I Spikes/phi2/upstream_roundoff.py \
  /absolute/checkpoint /absolute/reference /absolute/new-baseline.json
```

## Sources

- [Pinned checkpoint and config](https://huggingface.co/microsoft/phi-2/tree/9a5117d15e407c48fa40f8adbac1244abae52d6f)
- [Pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/phi/modeling_phi.py)

## Recorded result (2026-11-02)

[Qualification summary](evidence/qualification.json) and the adjacent source
reports record a qualified macOS completion spike. All 714 reported Swift
Release tests across 18 test executables pass. Ten Python test methods cover Phi,
Qwen3, OLMo2 and GPT-2, including tiny Core AI exports; Phi's running-state export
also passes.

The full checkpoint's Core AI CPU self-check passes exact argmax and no lost
cache rows, with minimum cosine 0.99298. Across prompts of 7, 280 and 224 tokens,
both float32 port contracts match all 64 reference argmax rows per prompt.
Maximum float32 logit error is 0.008518; upstream's own chunked execution reaches
0.008142 on the same long prompt. These measurements set Phi's 0.01 absolute
bar; the existing 0.999999 cosine and exact-argmax requirements are retained.

Native GPU qualification matches all 192 forced-token choices and all three
64-token continuations exactly, repeatably across prefill widths 7 and 16.
Minimum native cosine is 0.99847. The 224-token case records three cache growths
(one per request) from 256 to 512 rows. The existing real-weight GPT-2 native
qualification also passes, checking the preserved contiguous cache path.

The usable local registered bundle is `.local/phi2-qualified/bundle` in this
worktree. No model weights or tokenizer assets are committed. The local output
also retains upstream logits for rerunning native qualification. This is a
macOS spike; iOS and interleaved cache storage remain outside its qualification.

## Published bundle

The bundle is public at [aether-models/aether-kit-phi2](https://huggingface.co/aether-models/aether-kit-phi2).
Pin downloads to revision `986b7a8373e30a9542218a8bbc15c2acdd3af4d3` and use `mac/`.
[`catalog-macos.json`](catalog-macos.json) is a ready `ModelCatalogEntry` for
Aether's verified preparation and installation flow. Open the installed bundle
through `LanguageGenerationSession`; Phi-2 remains completion-only. Use an Aether
Kit checkout containing commit `1881cf4` or a descendant, which adds the required
stride-aware cache handling; published 0.0.4 does not include it.

[Publication verification](evidence/huggingface.json) records all staged files'
sizes and SHA-256 values. Small files were downloaded anonymously and checked;
the hosted compiled model's LFS SHA-256 and size match the qualified local bytes.
The HF repository includes the original MIT license, conversion attribution, model
card, export provenance and compact qualification evidence. No iOS bundle is published.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "aether-models/aether-kit-phi2",
    revision="986b7a8373e30a9542218a8bbc15c2acdd3af4d3",
    allow_patterns=["mac/**", "LICENSE", "NOTICE"],
    local_dir="phi2-kit",
)
```

Open `phi2-kit/mac` with the completion API above.

Before merging the publication update, the combined tree including SD-Turbo passed
722 reported Swift Release tests across 19 test executables, and the frozen API
matched its committed baseline.
