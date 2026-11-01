# Design note: Qwen3.5-4B image + text on a hybrid recurrent-state decoder (0.0.4)

**Status: design, before code (2026-11-01).** Branch `v0.0.4/qwen3.5-vl`. Goal: run base `Qwen/Qwen3.5-4B` as a
vision-language model in Kit, so a host (Atelier's describer) can drive it the way it drives Qwen3-VL-4B today
through `VisionLanguageSession`. The recurrent-state runtime it needs is built once and serves Granite 4.0-H as well
([roadmap](../roadmap-0.0.4.md), [0.0.3 hybrid spike](hybrid-state-0.0.3.md)).

## Checkpoint

- `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, **Apache-2.0** (model card front matter
  and `LICENSE`), not gated. Already in the local HF cache (8.7 GB bf16, two shards); nothing to download.
- **Text** (`qwen3_5_text`): 32 layers in a fixed 3:1 pattern, 24 Gated DeltaNet (`linear_attention`) and 8 full
  attention (layers 3, 7, …, 31). Hidden 2,560, MLP 9,216, vocab 248,320, tied head. Zero-centred RMSNorm
  (`x̂ · (1 + w)`).
  - Full attention: 16 query / 4 KV heads, head dim 256, q/k RMSNorm, an output gate (`q_proj` emits query and gate;
    output × sigmoid(gate)), partial rotary (64 of 256 dims), interleaved M-RoPE sections [11, 11, 10], θ = 10⁷.
  - Gated DeltaNet: 16 key heads and 32 value heads of 128, causal depthwise conv (kernel 4, SiLU) over the 8,192-wide
    qkv, l2-normed q/k repeated to 32 heads, `g = −exp(A_log) · softplus(a + dt_bias)`, `β = sigmoid(b)`, recurrent
    state [32, 128, 128], output RMSNorm gated by SiLU(z).
  - An MTP head (`mtp.*`, one layer) that upstream ignores at load. Kit ignores it too (a later speculative-decode
    option, not in scope).
- **Vision**: the Qwen3-VL-4B tower exactly (depth 24, hidden 1,024, patch 16, merge 2, 2,304 learned positions,
  out 2,560) with `deepstack_visual_indexes: []`. Same preprocessor (`Qwen2VLImageProcessorFast`, mean/std 0.5).
- **Chat**: Qwen3.5's template thinks by default; the describer renders with `enable_thinking: false` (the screen did
  the same). `LanguageTokenizer.render` already passes `enable_thinking`.

## State contract (generic: Qwen3.5 and Granite 4.0-H)

The 0.0.3 spike's static-width contract, generalised from Granite's two Mamba states to **declared state roles**.
One decoder graph, inputs and outputs by name:

| Name | Kind | Shape (Qwen3.5-4B) | Notes |
|---|---|---|---|
| `input_ids` | input i32 | [1, q] | tail padded with 0 |
| `cache_positions` | input i32 | [1, q] | start…start+n−1, then −1 for padding |
| `rope_positions` | input i32 | [3, q] | M-RoPE; padding rows 0 (Qwen3.5 only; Granite is NoPE) |
| `image_embeds` | input fp16 | [q, 2,560] | used where the token is `<\|image_pad\|>`, zero otherwise (VL only) |
| `keyCache`, `valueCache` | input fp16 | [8, 1, 4, C, 256] | read-only, fixed capacity C per registration |
| `convState` | **state** fp16 | [24, 1, 8,192, 3] | role `recurrent`, written in place |
| `recurrentState` | **state** fp32 | [24, 1, 32, 128, 128] | role `recurrent`, written in place, always fp32 |
| `logits` | output fp16 | [1, 1, 248,320] | the last *valid* row |
| `keyUpdate`, `valueUpdate` | output fp16 | [8, 1, 4, q, 256] | host copies the n valid rows to start…start+n−1 |

- **Roles are declared in `language-model.json`** (`hybrid.states: [{name, role: recurrent, scalarType, shape}]`,
  plus `staticWidth`, `fixedCacheCapacity` and the function layout) and matched against the function descriptor at
  open. Kit allocates one zeroed buffer per declared state, zeroes it per request, enforces the declared scalar type,
  and reports the bytes in cache statistics and resource admission (a fixed term: ~51 MB here, independent of
  context). Granite registers `convState` / `ssmState` the same way; nothing in the runtime is Qwen-specific.
- **Padding changes no state.** For padded rows: DeltaNet `g = 0` (no decay) and `β = 0` (no write); the conv state
  keeps the last 3 *valid* inputs (index-select at n, as Granite's port does); padded keys are masked; logits come
  from row n−1. Granite's padded path already does the Mamba equivalent (dt = 0).
- **No step-back.** Recurrent state can't rewind, so the prefill tail is padded, never re-submitted. Rewind = reset +
  replay. Kit joins every submission before the next, so stop tokens need no state snapshot.
- **KV is fixed capacity** (8 attention layers × 32 KiB per token: 128 MiB at 4,096 rows), allocated once, no growth
  ladder: one KV shape per registration, so Core AI's per-shape-change retention is bounded by construction.
- **Ordering lint at export** (0.0.2 F-P2-20): every read of a state's old contents must feed its write; the Granite
  exporter's `state_order_violations` check is reused.

### Function layout: one weight copy

Core AI wires a separate weight copy per function. For a 4B, S1 (`main` + `decode`) would cost ~7 GiB more (a second fp16 copy of the text weights) and
won't fit beside Krea, so the candidates are both single-function:

- **S2** (qualified on Granite in 0.0.3): `main` at static width W only; decode is a padded W-wide call.
- **D1** (start-scalar dynamic q, the roadmap's 0.0.4 re-qualification): one `main` with dynamic q, positions passed
  as a [1, 1] start and built in-graph (the fix for 0.0.3 defects 1–2), used at exactly two widths, W and 1 (still
  padded at W, so no step-back and only two shapes).

The exporter emits both from the same module. Stage 2 measures and qualifies both; the registration declares which.
D1 is the expected winner (one copy, no padded decode); S2 is the fallback with known-good Core AI behaviour.

## Exporter shape (Forge, locked env)

- `Forge/language/qwen35_hybrid.py`: the eager port (text decoder over the contract above; padding semantics; one
  `forward` for prefill and decode). DeltaNet via coreai-torch's **`GatedDeltaUpdate` composite** (a per-token
  `while_loop` recurrence; padding by g/β as above). If its GPU prefill is too slow or it doesn't lower cleanly, the
  fallback is a single-chunk WY form in plain ops (W×W unit-lower-triangular inverse by repeated squaring, log₂W
  matmuls) — decided in stage 2 on numbers, not up front.
- `Forge/language/export_qwen35_text.py`: weights fp16, DeltaNet maths and the recurrent state fp32, attention and the
  residual as in the Qwen3-VL text exporter (fp16 with fp32 where the Qwen3-VL export needed it). Emits `main` (S2 or
  D1), the state-order lint, and a manifest provenance block (repo, revision, licence, file SHA-256s).
- Vision: `export_qwen3_vl_vision.py` / `qwen3_vl_vision.py` reused with zero DeepStack levels (the same five static
  grid functions; outputs `image_embeds` only). `LanguageVisionEncoder` learns to accept an encoder without
  `deepstack`.
- `register_qwen35_bundle.py` (or a mode of `register_qwen3_vl_bundle.py`): `language-model.json` with the `vision`
  registration plus the `hybrid` block.
- Outputs under `~/.cache/huggingface/aether-kit/qwen3.5-4b/` (references, exports, bundle). Never `/tmp`.

### Reference environment (needs your OK)

Forge pins `transformers==4.57.3`, which has no `qwen3_5`; upstream Qwen3.5 is in 5.x. Bumping Forge's pin would move
every 0.0.2/0.0.3 reference off its recorded version. **Proposal:** a second locked uv project,
`Forge/reference-qwen35/` (`pyproject.toml` + `uv.lock`: the same torch 2.11.0 and torchvision 0.26.0, transformers
5.x), used only for the upstream Qwen3.5 reference scripts. Exporters and the port stay in Forge's main env. Neither
touches aether-engine.

The reference is upstream fp32: teacher-forced rows on the CPU (one forward pass: upstream's chunked torch fallback,
slow but bounded). Greedy continuations on CPU were unusable in the v0.7 screen (no picture in 25 min), so greedy
runs go on MPS fp32 under the GPU lock (MPS fp32 matched CPU exactly in that screen); a CPU teacher-forced spot check
ties the two.

## Swift runtime

`HybridLanguageSession` (actor, `AetherLanguageCoreAI`) beside `LanguageGenerationSession`, conforming to
`LanguageBackend`, `LanguageBackendQualification` and `VisionLanguageDecoding`, so `VisionLanguageSession` drives it
unchanged; `VisionLanguageSession.open(bundle:)` picks it when the manifest has a `hybrid` block. Reused as is:
`LanguageDecodeLoop` (sampling, constraints, delivery), `LanguagePromptPlan` (M-RoPE and image expansion; checked
against Qwen3.5's `get_rope_index`), `LanguageRequestAdmission`, GPU and resource admission, lifecycle and close.
New:

- `LanguageHybridContract`: detection from the descriptor plus declared roles; the padded prefill planner (W-wide
  chunks, tail padded; image rows aligned per chunk as today, so an image may straddle chunks).
- Declared-state buffers (zeroed per request), fixed KV buffers with host row copies, function selection (S2 or D1).
- Statistics: recurrent bytes, KV bytes, wired GiB; resource-admission profile entry for the bundle hashes.

Atelier-facing API is the existing one: `VisionLanguageSession.open(bundle:platform:gpu:resources:)` and
`generate(renderedTokenIDs:images:…)`.

## Qualification (the bar)

Against upstream fp32, on the GPU, release builds:
1. **Text-only** (stage 2): 3 chat cases, teacher-forced rows (cosine, argmax with the near-tie rule), 128-token
   greedy equality; drift to 4,096 tokens.
2. **Image + text** (stage 3): Orion (`orion-woman.jpeg`, snapped to its grid by Kit and the reference alike) with the
   describer's whole-picture prompt, plus encoder parity per grid; teacher-forced rows and greedy equality.
3. **Session** (stage 4): the same through `VisionLanguageSession`; time to first token, decode tok/s, wired GiB, and
   resident memory across 20 repeated image requests.

## Estimate

| Piece | Engineer-days |
|---|---|
| Reference env + upstream reference scripts (text, image) | 1–1.5 |
| Eager port + padded semantics + CPU proof against upstream | 2–3 |
| Text exporter (S2 and D1), state lint, GPU harness, stage-2 parity | 2–3 |
| Vision without DeepStack, encoder change, stage-3 parity | 1–1.5 |
| `HybridLanguageSession`, manifest `hybrid` block, CPU tests, stage-4 numbers | 5–7 |
| Total | ~11–16 (about 2.5–3 weeks), dominated by the runtime |

## Risks

1. **`GatedDeltaUpdate` on the GPU.** It's a `while_loop`; whether Core AI lowers it to a fused kernel or a 64-step
   loop per layer is unknown, and a `torch.cond` already aborted on cpu_only (0.0.3 defect 6). Prefill speed and
   correctness at W = 64 decide between it and the WY fallback.
2. **D1 may not specialize** at 4B depth despite the start-scalar fix (it was shown at Granite 350M with random
   weights only). S2 is the fallback, at a decode-speed cost.
3. **fp16 range.** Qwen3-VL's fp16 text export needed fp32 in places; Qwen3.5's gated attention and zero-centred norms
   may need others. The DeltaNet state stays fp32 by contract.
4. **cpu_only can't self-check static-q graphs** (0.0.3 defect 5), so qualification is native GPU only, as for Granite.
5. **Compile cache size**: the 4B decoder's compiled copy is likely 15–22 GB per namespace (VL-4B decoder: 22.5 GiB);
   purged after each stage. Disk is ~259 GB free.
6. **Reference speed**: upstream's torch DeltaNet fallback is slow on CPU; MPS fp32 for greedy, under the GPU lock.
7. **Tokenizer**: a new 248k vocabulary; Kit's tokenizer must render and round-trip it (checked in stage 2).

## Stages and reports

1. This note. 2. Decoder-only text parity (S2 and D1). 3. Vision + decoder parity on Orion. 4. Swift session with
tok/s and wired GiB. Each stage is a local commit on `v0.0.4/qwen3.5-vl`; no push or PR without the lead's say-so.
