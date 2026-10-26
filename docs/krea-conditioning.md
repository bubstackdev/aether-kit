# Krea assets and conditioning — 2026-10-17

This records the conditioning slice. The subsequent [generation slice](krea-generation.md) implements shape registration, denoiser orchestration and VAE decoding and records follow-up native validation. Cache access now works through approved execution outside the agent sandbox; the original validation limitations below describe this slice's earlier run.

## Implemented

`AetherKrea` opens the prototype's dynamic macOS Krea 2 Turbo realism bundle and rejects unsupported schema/recipe/target/function/binding declarations. Asset references stay inside the bundle, including symlink resolution. Six distinct model assets are required.

Opening checks the model hash declarations and small model metadata, hashes tokenizer/configuration files, and compares those identities with provenance when supplied. It does **not** hash the large model streams. `assets.verify()` explicitly streams all six model streams through SHA-256, concurrently and with bounded memory, and rechecks model metadata and tokenizer files. A failure names the first failing asset in serial order. Bundle fingerprints identify the header snapshot, model declarations/metadata and tokenizer bytes. These checks establish content consistency, not publisher authenticity. Keep external files immutable while using them.

Prompt preparation applies the prototype's fixed Qwen chat rendering with no extra special-token insertion or padding. It truncates to 546 tokens while preserving the final five tokens, supplies int32 IDs/running positions/all-one attention masks, and selects conditioning after the first 34 rows. Conditioned inputs are finite fp16 `[1...512,30720]`; supplied float32 references narrow once and reject overflow. Image rotary inputs preserve the original `[imageTokens,2,128]` split-halves layout.

The optional `AetherKreaTokenizers` product loads local tokenizer files through Hugging Face's Tokenizers product from swift-transformers 1.1.0. It never requests a remote model. Core contracts and the Core AI adapter have no dependency on that target, though SwiftPM still resolves package-wide dependencies. The checked-in lock records this checkout's transitive versions; consumer applications manage their own resolution.

`AetherKreaCoreAI` adds an experimental resident `KreaTextEncoder`: load the encoder once, call its `encode` function with typed int32 inputs, select fp16 conditioning rows, and explicitly unload. It uses the shared session's overlap/cancellation/close behavior and requests frequent reshapes. Native execution was blocked during this slice's initial validation; see the generation slice for the subsequent encoded parity gate.

## Use local assets

```swift
import AetherKrea
import AetherKreaTokenizers

let assets = try KreaAssets.open(at: bundleURL)
let tokenizer = try await KreaTokenizer.open(assets: assets)
let prompt = try tokenizer.prepare("a red fox standing in fresh snow at sunrise")
let configuration = try KreaConfiguration(width: 512, height: 512)
let inputs = try prompt.encoderInputs
let rotary = try KreaRotary.image(configuration: configuration)
```

File operations and tokenization are synchronous work; callers should keep preparation off the main actor. Core AI models are not loaded by the code above.

For the experimental native path, add the AetherKreaCoreAI product:

```swift
import AetherKreaCoreAI

let encoder = try await KreaTextEncoder.open(assets: assets)
do {
    let conditioning = try await encoder.encode(prompt)
    await encoder.unload()
    print(conditioning.tensor.shape)
} catch {
    await encoder.unload()
    throw error
}
```

The CLI supports preparation without inference:

```sh
.build/arm64-apple-macosx/release/aether-probe prepare-krea /path/to/macos-any-gpu \
  'a tall clock tower in a snowy old town at night, lanterns glowing' 768 1344
```

`verify-krea BUNDLE` is a separate explicit full-stream integrity check and can read tens of GB.

## Validation

The release test run passes 19 tests including the two external-input checks when all three paths are provided:

```sh
AETHER_KREA_BUNDLE=/path/to/macos-any-gpu \
AETHER_KREA_FIXTURES=/path/to/fixtures/4c12ae019488ff1c \
AETHER_KREA_LAB=/path/to/krea-lab Scripts/check.sh
```

All six retained prompts match every saved token ID and conditioning-row count. A long prompt preserves the five-token suffix. The 768×1344 rotary table matches every retained Swift byte. Other tests cover unsupported recipes, malformed hashes, provenance mismatches, path/symlink escape, aliased model files, post-open corruption, fp16 overflow, row selection and mixed-scalar session calls.

These are CPU preparation and controlled-backend tests. They do not establish text-encoder inference parity, full model-stream integrity of the real external bundle, image generation, or a performance improvement. Full integrity tests use small synthetic bundle streams.

[Recorded evidence](evidence/krea-conditioning-2026-10-17.json) includes the exact source/dependency identities, retained rotary hash and CLI preparation report. The selected prototype reference files still match their scaffold-time hashes.

## Remaining work

- Native typed inputs and Qwen conditioning passed the retained tower's encoded parity gate. Full eight-step independent conditioning/latent/pixel parity also passed at 512×512 and 384×512 with 45- and 75-token prompts. Broaden coverage to additional seeds, lengths and shapes.
- Shape-specialization registration, carry validation, denoiser orchestration and VAE decoding passed the tower's provided/encoded gates. One-step real dynamic fallback passed at 512×512 and 384×512.
- Native model reuse and active unload passed the one-step session checks. With lazy per-session native function reuse, encoder cancellation followed by full eight-step re-encoding recovery and decoder cancellation followed by full eight-step recovery both pass exact independent conditioning/latent/pixel comparisons ([current evidence](evidence/krea-function-reuse-qualification-2026-10-18.json)). Controlled pipeline regressions and isolated native decoder checks remain complementary evidence. Real resident-encoder reuse, broader failure cases and independent GPU queue-drain observation also remain unqualified.
- Compare the direct Swift path with Corsairs before resolving the runtime decision.

Git initialization attempts during this slice were denied by the environment; the evidence retains that historical result. The user subsequently initialized the repository. The existing lab source remains unchanged.
