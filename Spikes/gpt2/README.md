# v0.0.6 GPT-2 completion port spike

Port `openai-community/gpt2` (124M) at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` to the existing Aether language runtime.
This is a development spike with a published, pinned macOS bundle and catalog entry.
The checkpoint has a 1,024-token learned position table and MIT license.

## Why this model

Qwen3 and OLMo2 already exercise rotary positions, RMSNorm and gated MLPs.
GPT-2 adds learned absolute position embeddings, LayerNorm with bias, biased
attention/MLP projections, GELU, and transposed Conv1D checkpoint matrices. Its
small checkpoint permits complete numerical qualification on the development Mac.
These differences test whether the shared dense executor can support families
beyond its original rotary decoder assumptions.

The changes provide reusable extension points:

- `dense_families.py` registers a decoder, loader, tiny configuration, context
  ceiling, reference prompt style and float32 reference tolerance. Export and
  reference CLIs both consult it; neither contains family-specific dispatch.
- `DenseDecoder` accepts a final normalization factory, optional rotary positions,
  and an `embed_inputs` hook for token/position embeddings. Cache allocation,
  causal masks, running/bucketed contracts and host updates stay shared.
- `TensorMapping` supports transposed checkpoint weights alongside existing fused
  Q/K/V row mappings. GPT-2 accepts original unprefixed Hub keys and Transformers'
  `transformer.` keys; known legacy causal-mask buffers are ignored. Unknown or
  missing learned tensors refuse loading.

## Use in Aether

Export with `Forge/language/export_gpt2.py`, then register with the existing
`register_dense_bundle.py`, using maximum context 1024. The asset uses fp16 weights
and caches with float32 SDPA. The bucketed contract is the default.

```swift
let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
let session = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
let result = try await session.generate(
    prompt: tokenizer.encode("The future of artificial intelligence is"),
    maximumTokens: 64)
let text = tokenizer.decode(result.tokens)
await session.unload()
```

Import `AetherLanguageTokenizers` and `AetherLanguageCoreAI`; run on supported
macOS/iOS 27 devices. Only macOS is qualified here. GPT-2 is a base completion
model with no chat template: use plain token prompts through
`LanguageGenerationSession`. The chat facade reports `unsupportedTemplate`.
The spike does not invent a chat template or claim instruction-following support.

## Reproduce

Use the pinned Forge environment (`uv sync --frozen --project Forge`) and a local
snapshot containing `model.safetensors`, `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` from the
revision above. The script does not download or publish anything.

```sh
OMP_NUM_THREADS=4 sh Spikes/gpt2/qualify.sh /absolute/checkpoint /absolute/new-output
```

Override `AETHER_FORGE_PYTHON` to reuse an existing pinned Forge environment.
Native tests take the repository's shared GPU lease themselves.

The weight-free tests compare tied and untied heads, chunked prefill, single-token
decode, cache growth, poisoned future cache rows, and the final position-table
entry against Transformers 4.57.3. They also cover both checkpoint layouts and
refusal of unsupported activation, cross-attention, pruning and attention-scaling
variants. Existing Qwen3/OLMo2 decoder and CPU export regressions are included.

Full-checkpoint reference gates compare 64 rows in float32 through both contracts.
GPT-2 uses `atol=0.0005, rtol=0.00001`: its logits have large common offsets, and
chunked versus full-sequence BLAS produces about 0.0007 absolute roundoff in this
checkpoint. Every element must meet that tolerance, cosine must reach 0.999999,
and every row's argmax must match. Qwen3/OLMo2 retain their existing absolute-only
tolerance. The float32 eager-port gate requires exact argmax equality.

Core AI CPU self-checks compare logits and cache updates. Native Swift qualification
verifies bundle hashes, tokenizer IDs/text, 64 teacher-forced rows and a 64-token
greedy continuation against upstream, and repeatability with prefill widths 7 and
16. Native fp16 comparisons use the existing language qualification policy:
cosine >= 0.995 per row; a differing argmax is accepted only when the reference's
top-two probability margin is below 0.1 and the native token is in its top five.
A free-running divergence is judged at its first differing row, before the
prefixes differ; its index and acceptance are recorded separately from exact
continuation equality. The short prompt's fp16 run first diverges at token index
40 despite the first 32 forced rows matching exactly; this spike does not claim
bit-exact float32 generation. The 600-token long prompt additionally exercises the larger 768-row cache bucket. Evidence and
observed results are recorded under `evidence/` after qualification.

## Sources

- [Checkpoint/configuration](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
- [Pinned Transformers implementation](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/gpt2/modeling_gpt2.py)
- [Original GPT-2 implementation and license](https://github.com/openai/gpt-2)

## Recorded result (2026-11-02)

Both full-checkpoint float32 port contracts match all 64 reference argmax rows on
both prompts. All six weight-free test methods and the Swift release build pass;
tiny Core AI exports pass for all three dense families, plus GPT-2's running
contract. Both registered-bundle GPU cases pass the native policy above.
See [qualification.json](evidence/qualification.json) for hashes and the numerical
results, and the adjacent reference/native/export reports for the source records.
The exported local bundle remains in `.local/bundle-exact/` in this spike worktree.

## Published bundle

The bundle is public at [aether-models/aether-kit-gpt2](https://huggingface.co/aether-models/aether-kit-gpt2).
Pin downloads to revision `102d60f338bc5eff7c38aaeb2c0c4e2f76851a3f` and use the `mac/` subdirectory.
[`catalog-macos.json`](catalog-macos.json) is a ready `ModelCatalogEntry` for
Aether's verified preparation/installation flow. GPT-2 remains completion-only;
open the installed bundle through `LanguageGenerationSession`, rather than the
chat facade.

[Publication verification](evidence/huggingface.json) records every hosted file's
size and SHA-256. Small files were downloaded without authentication and checked;
the hosted model's LFS SHA-256 and size match the qualified local bytes. The Hub
repository includes the original license, conversion attribution, model card,
export provenance, and compact qualification evidence. No iOS bundle is published.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "aether-models/aether-kit-gpt2",
    revision="102d60f338bc5eff7c38aaeb2c0c4e2f76851a3f",
    allow_patterns=["mac/**", "LICENSE", "NOTICE"],
    local_dir="gpt2-kit",
)
```

Open the downloaded `gpt2-kit/mac` directory with the completion API above.

Before merging the publication update, the full weight-free Swift Release suite
passed 712 tests across 18 test executables (`sh Scripts/check-cpu.sh --skip-build`).
