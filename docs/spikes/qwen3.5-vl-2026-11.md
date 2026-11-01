# Design note: Qwen3.5-4B image + text on a hybrid recurrent-state decoder (0.0.4)

**Status: stages 2 (text parity, S2 chosen), 3 (image parity) and 4 (Kit session and bundle) done (2026-11-01;
[stage 2](#stage-2-text-parity), [stage 3](#stage-3-image-parity-orion), [stage 4](#stage-4-kit-session-and-bundle)).** Branch `v0.0.4/qwen3.5-vl`. Goal: run base `Qwen/Qwen3.5-4B` as a
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

### Reference environment (approved 2026-11-01)

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

## Stage 2: text parity

Runs on 2026-11-01, M5 Max, macOS 26B123, quiet host (1-minute load 1.7–2.3), every GPU run under `with-gpu-lock.sh`.
Outputs and the environment of each are listed in `~/.cache/huggingface/aether-kit/qwen3.5-4b/README.md`.

**Reference.** `reference_qwen35.py` in `Forge/reference-qwen35`, upstream fp32 on the CPU for everything (greedy too:
25 min for 3 × 128 tokens plus the 4,096-token pass, so MPS wasn't needed). Teacher-forced argmax equals the greedy
tokens on 32/32 rows per case.

**Port.** `qwen35_hybrid.py`, float32 on the CPU, fed like the host feeds the asset (64-wide padded prefill, then
one token per padded call), against the reference: min cosine 0.99999993, every argmax equal, on the 3 chat cases
(32 rows each) and on 16 rows to 4,096 tokens (`Spikes/qwen35-vl/check_port.py`). On a random tiny model against
upstream: cosine 1.0, and the padded recurrent state equals the unpadded one (`tiny_vs_upstream.py`).

**Delta-rule form.** Three forms are implemented and all three are exact; they differ in speed:

| Form | S2 W = 64 decode | Notes |
|---|---|---|
| `composite`, externalized (`gated_delta_update` composite graph) | **31–33 tok/s** | chosen; also fastest prefill |
| `composite`, traced `while_loop` | ≈ 20 tok/s (tiny: 2.2× slower than externalized) | |
| `chunk` (loop-free WY form, plain ops) | 20.3 tok/s | needs a stable inverse (below) |

The `chunk` form first used the power series (I − N)⁻¹ = Σ Nᵏ by repeated squaring; on real weights it cancels in
float32 (long-context cosine 0.96, then NaN). A blocked doubling inverse (`unit_lower_inverse`) is exact. W = 1
ablations locate the cost: without the DeltaNet layers the 4B decodes at 56.7 tok/s (the weight-bandwidth floor),
with their projections but no delta rule at 44.8, with the plain-ops rule at 21.9 and with the composite at 42.7.

**Layouts.** Both qualify numerically through the Python runtime (forced min cosine ≥ 0.999997, 0 argmax mismatches,
128-token greedy identical on all 3 cases, long rows min cosine ≥ 0.99979 with every argmax equal). Swift harness
(`Spikes/qwen35-vl/harness`, capacity 4,608, warm medians):

| | S2, W = 64 (externalized composite) | D1, W = 64 / q = 1 (externalized composite) |
|---|---|---|
| Numerics, Swift | identical to the Python run | **wrong with `expectFrequentReshapes`** (cosine 0.33–0.65); right without it |
| Time to first token, 64 / 256 / 1,024 / 2,048 tokens | 38 / 129 / 495 / 983 ms | 37 / 130 / 498 / 991 ms (reshapes off) |
| Decode after 256 / 2,048 tokens | 32.8 / 30.5 tok/s | 43.3 / 43.3 tok/s (reshapes off) |
| Metal allocated after both widths ran | **9.4 GiB** (mapped file wired 9.0 GiB) | 18.3 GiB with reshapes off: the q = 1 specialization holds a second copy |

**Decision: S2 at W = 64 with the externalized composite** (`text-s2-w64-composite`, the exporter's default; its own
Python-runtime qualification is `runs/qualify-s2-w64-composite-python.json`; the Swift numbers above are from the
identical graph exported one run earlier, which differs only in the composite's generated graph names). D1 is 32 %
faster at decode but either wrong (reshapes on) or twice the memory (reshapes off), so it isn't usable beside Krea.
For the describer's ~840-token image prompt and ~150-token replies that is about 0.4 s to first token and 4.5–5 s of
decode.

**Core AI defect (Apple Feedback candidate).** A dynamic-q graph containing the `gated_delta_update` composite,
specialized for the GPU with `expectFrequentReshapes`, computes wrong logits (cosine 0.33–0.65 against the reference,
greedy diverging within 4–16 tokens), while the same asset is right through the Python runtime (no reshape flag) and
through Swift without the flag. This looks like F-P3-26 again. Repro: `export_qwen35_text.py --layout D1` (or
`--tiny --layout D1 --static-width 8`), then the harness with and without `--no-reshapes`.

**Export workarounds (labelled in code).** coreai-torch 0.4.2 can't lower `F.pad` by a symbolic amount (D1 uses a
[W, s] selection matrix), and when it externalizes the composite in a dynamic-q graph it rebuilds the token dimension
as a fresh symbol whose range has collapsed to the example width (`widen_collapsed_composite_dims`).

**Open for later stages.** Wider W (128/256) for time to first token; whether S2's decode (30–33 ms per padded 64-wide call, against
23 ms for a static W = 1 call) can get closer to W = 1 without a second weight copy, for example with a smaller W.

## Stage 3: image parity (Orion)

**Reference.** `reference_qwen35_image.py` in `Forge/reference-qwen35`, upstream fp32 on the CPU (16 min): the
describer's whole-picture outfit prompt (system instructions, image, request; `enable_thinking: false`) on
`orion-woman.jpeg`, snapped to grid 48×36 (576×768, 432 image tokens) with the processor's own resize, 838 prompt
tokens; 128 greedy tokens (end of sequence ignored) and 32 forced rows. Its reply equals the v0.7 screen's.

**Ports, fp32 on the CPU, on the reference's pixels** (`Spikes/qwen35-vl/check_port_image.py`): vision tower (the
Qwen3-VL port with zero DeepStack levels) against upstream's image embeddings: cosine 1.0 (row min 0.99999999);
Kit's host M-RoPE positions equal upstream's rope index; text port in padded 64-wide prefill with aligned image rows:
forced min cosine 0.99999999, 32/32 argmax.

**Assets on the GPU** (Python runtime, `gpu_check.py image`): `vision/` (five static grid functions, 638 MB,
`image_embeds` only; its cpu_only self-check worst cosine 1.0 at 7 places) and `text-s2-w64-composite/`.
Encoder: embeddings cosine 0.99998 against upstream (row min 0.9928), 0.15 s per encode. Decoder with those rows:
forced min cosine 0.99975, 0 argmax mismatches; **128-token greedy identical** to upstream.

Kit's own preprocessing (Swift resize of the raw JPEG) and the full `VisionLanguageSession` path are stage 4.

## Stage 4: Kit session and bundle

**Runtime** (`Sources/AetherLanguageCoreAI/HybridLanguageSession.swift`):
- `LanguageHybridRegistration` (`AetherKit`, the `hybrid` block of `language-model.json`): layout `static-width`, the
  width, the declared states (name, role `recurrent`, scalar type, shape) and an optional reshape request (default off).
  Nothing is model-specific; Granite 4.0-H registers `convState`/`ssmState` the same way and its signature (no rotary
  positions, no image rows) is accepted by the same checks (CPU-tested).
- `LanguageHybridContract`: validation of `main` against the registration (required inputs, optional `rope_positions`
  and `image_embeds`, exactly the declared states with their types and shapes, every token input W wide, the KV
  capacity dynamic or static at the registered context), and the padded plan (W-wide chunks, never a step back).
- `HybridLanguageSession`: Kit-owned KV at the registered context, declared-state buffers zeroed per request, one W-wide
  call per chunk and per decode step, valid update rows copied after each join; GPU and resource admission and close
  as the dense session has them. It conforms to `LanguageBackend` (so the `AetherLanguage` facade opens hybrid bundles),
  `LanguageBackendQualification` and `VisionLanguageDecoding`: `VisionLanguageSession.open(bundle:)` picks it when the
  manifest has a `hybrid` block, so a host's call shape is unchanged.
- `LanguageVisionEncoder` accepts an encoder without DeepStack (zero levels). The dense and Qwen3-VL paths are
  unchanged; every CPU suite passes (`swift test --triple arm64-apple-macosx27.0 --no-parallel`) and the frozen API
  matches its baseline.

**Static capacity.** With the KV capacity dynamic, every load's first request took 11.7 s (Core AI specializing the
capacity on first use, never cached). Exported with a static capacity at the registered context
(`--fixed-capacity 4096`), the graph has no dynamic dimension: the first open in a process namespace compiles
(about 21 s, cached on disk), every later open takes 4.8 s and its first request 0.11 s. Most of the 4.8 s is
`LanguageModelManifest.verify` hashing the 7.8 GB `main.mlirb`, as for every Kit language bundle.

**Bundle** `~/.cache/huggingface/aether-kit/qwen3.5-4b-bundle-macos` (the Writer layout: `model.aimodel`,
`vision.aimodel`, `tokenizer/`, `language-model.json`, both export reports), registered by
`register_qwen3_vl_bundle.py` (which now writes the `hybrid` block for a hybrid export): context 4,096, W = 64, stop
tokens `<|endoftext|>` and `<|im_end|>`, the five fixed grids, decoder `modelSHA256` `dccf1ca7…`, encoder `55965fa1…`.

**Through the public API** (`Tests/AetherNativeTests/HybridVisionLanguageSessionTests.swift`,
`runs/session-qwen35-bundle-c4096.json`):
- `AetherLanguageTokenizers` renders every reference prompt exactly (the describer prompt with a system message and an
  image, and the three chat prompts; thinking off).
- Text through `HybridLanguageSession.open(bundle:)`: all three chat cases **identical to upstream** (96, 128 and 99
  tokens, two ending at upstream's end of turn).
- Image through `VisionLanguageSession.open(bundle:)` from the **raw JPEG**: identical to upstream for 12 tokens, then a
  divergence at a reference near tie (margin 0.0016), the same on all three repeats. The reference here is upstream fed
  Kit's own lossless decode of the JPEG (`reference-image-kit-decode-cpu/`), as the Qwen3-VL photos were qualified:
  ImageIO and PIL decode this 366×480 JPEG differently (mean 1.1 levels, max 44), and upstream's reply changes with the
  decode too (it opens with `{` instead of ```` ```json ````). On the same decode Kit's pixels match upstream's at
  cosine 0.9999999, the encoder at 0.99993, and the decoder's 32 forced rows at min cosine 0.9994 with 31/32 argmax,
  the miss being that near tie (`Spikes/qwen35-vl/kitcheck`).
- Speed, warm: image request (preprocess, encode, 838-token prefill) to first token 0.46–0.55 s; decode 32.7–33.8 tok/s;
  text to first token 30 ms (about 30 tokens).
- Memory with the image session open and warm: Metal 10.5 GiB, wired mapped file 9.0 GiB, process footprint 1.1 GiB;
  flat across repeats; released at unload.

## Decode and prefill speed (follow-up, 2026-11-01)

The describe lane measured Qwen3.5 at about 34 ms per token in Atelier against VL-4B's 24 ms. Harness numbers (warm
medians, `--no-reshapes` unless noted; prefill of 1,024 tokens):

| Asset | Decode | Prefill 1,024 | Compiled | Notes |
|---|---|---|---|---|
| S2 W 64, capacity 4,096 (the first bundle) | 29.8 ms (33.6 tok/s) | 482 ms | 9.7 GB | |
| **S2 W 64, capacity 2,048** | 29.2 ms (34.2–34.8 tok/s) | 464 ms | 9.7 GB | new bundle; public API: 27.8 ms (36.0 tok/s) |
| S2 W 32, capacity 2,048 | 26.6 ms | 849 ms | | |
| S2 W 16, capacity 2,048 | 24.0 ms | 1,525 ms | | |
| D1, static-width composite, capacity 2,048 (reshapes on) | 26.0 ms | 505 ms | about 2× (16 GB) | parity on all 3 chat cases |
| D1, static-width composite, capacity 4,096 (reshapes on) | 27.0 ms | 528 ms | about 2× (16 GB) | parity on all 3 chat cases; one resident copy (9.5 GiB Metal) |

- **Where the time goes.** A static q = 1 call costs 23.4 ms, about the weight-bandwidth floor and the same as VL-4B's
  24 ms. Every S2 decode step is a W-wide call, and the extra rows cost about 0.1 ms each: about 6 ms at W = 64. The
  capacity costs about 0.6 ms per step between 4,096 and 2,048.
- **Smaller W** trades decode for prefill almost one for one. For describer calls of about 1,000 prompt tokens, W = 64
  stays best unless a call generates far more tokens than it reads.
- **A second static q = 1 function** would wire a second copy of the weights: Core AI keeps a copy per function it runs
  (Granite S1: 1,446 against 797 MiB; the Qwen3-VL two-function decoder), so it isn't an option.
- **D1 fixed.** The D1 failure under `expectFrequentReshapes` was the dynamic-shape composite: with the composite run
  at the static width on rows placed by a selection matrix, D1 is right with reshapes on and keeps one resident copy.
  It saves about 3 ms per token against S2 at the same capacity, but its compiled package is about 2× (reshape-enabled
  dynamic shapes, as for VL-4B), each load specializes its shapes on first use (1.6 s cold for the first 64-wide call
  in the harness), and `HybridLanguageSession` would need a dynamic-width layout. Not adopted yet.
- The `widen_collapsed_composite_dims` export workaround is gone: the static-width composite has no dynamic dimension.
