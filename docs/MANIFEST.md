# Aether-Kit — Manifest schema v1 and driver catalog

Status: **normative for M1+**, 2026-11-05. DESIGN.md §4 summarises this document; where they differ, this one wins.

Provenance: derived from a contract survey of 38 zoo bundles (graph descriptors from `InferenceFunctionDescriptor`,
bundle metadata, tokenizer files) on macOS 27.2 / M5 Max, then reconciled with the decisions and measurements below.
The survey itself lives in the separate `lab` repo; nothing here depends on it.

## 0. Decisions folded in

| Topic | Rule |
|---|---|
| Multi-asset bundles | **One catalog entry per model.** The entry lists every asset subtree (and any foreign tokenizer source) with per-file sha256. The set installs under `<repo>/<revision>/`, with the generated manifest in `.aether/` at the revision root. One model is one install. |
| State roles | **A rule plus a reviewed override table.** Dynamic sequence axis → `states.sequence`; fully static → `states.recurrent`. Exceptions live in `catalog/zoo-overrides.json`, and per-entry goldens pin every classification. |
| Context | `limits.maxContext` is the declared maximum. **Capacity is derived from the memory budget at load** (DESIGN.md §6.2). |
| Per-arch AOT | Prefer the device's `ios-<arch>/` subtree, else the generic JIT subtree. `unsupportedDevice` only when neither exists. Resolution happens in `ModelRef`, before download. |
| BOS | The chat template owns BOS (measured: 33/33 identical to HF). |
| Stops | Source `generation_config.eos_token_id`, else `config.json` (`text_config.eos_token_id`) when the source has no generation config (Qwen3.5), ∪ tokenizer eos. Pinned per entry as `{source, eosTokenIds, tokens}`, provenance in `provenance.refs["tokenizer.stop"]`. A generic `</s>` in the union is harmless for chat (it never occurs mid-turn) and is kept for fidelity to the source. |
| Compute | `gpu` by default for every class (measured). `neuralEngine` only per entry with a measured win. |
| Tokenizer source | Never an unpinned hub fetch. A missing `tokenizer/` dir means `tokenizer.source.hf` at a pinned revision with hashed files. |

## 1. Driver catalog

Rules that apply to every driver:

- A **driver** is a registry key (§6.3) that owns a closed role set and its own manifest blocks.
- A **graph kind** (`vision-tower@1`, `audio-encoder@1`) is a reusable graph contract that a driver embeds. It
  is not registered on its own and is never the top-level `driver`.
- `ImagePreprocessor` (DESIGN.md §8) is a shared primitive configured by `preprocess.image`. It is not a driver.
  There is no `image-pipeline@1`.
- Roles below are manifest role names. "in", "out" and "state" are graph I/O. Roles marked *static* use
  `binding: "static"` (§2.3).

### 1.1 Text generation

| driver | graphs | required roles | optional roles / blocks | bundles (inspected; *skipped but same contract*) |
|---|---|---|---|---|
| `decoder-lm@1` | `decoder`; optional `prefill` function | in `tokens`, `positions`; out `logits`; ≥ 1 entry in `states.sequence` | `states.recurrent[]`; static inputs from `tables` (Gemma 4 PLE); out `hidden`; `graphs.decoder.functions.prefill` (shares states) | qwen3-0.6b macos, gemma-3-4b-it, minicpm5-1b, youtu-llm-2b (MLA latent), nanbeige4.1-3b, gemma-4-e2b (+PLE tables), qwen3.5-0.8b gpu-pipelined, lfm2.5-1.2b, lfm3-1.7b Mac pf64 (with prefill), granite-4.0-h-1b, nemotron-3-nano-4b; *qwen3-4b/8b, mistral-7b-v0.3, minicpm5-2b, nanbeige4.2-3b, gemma-3-12b, gemma-4-e4b/12b/31b, qwen3.5-2b, lfm2.5-2.6b*; MoE (*glm-4.7-flash, qwen3.6-\*, lfm2.5-8b-a1b*) unverified |
| `static-chunked-lm@1` | `embeddings` (load_embeddings), `gather` (gather_embeddings_{chunk}), `decoder` (function family extend/prompt_opt_{capacity}_{chunk}) | embeddings out `embeddingTable`; gather in `tokens`, `embeddingTable`, out `embeddings`; decoder in `embeddings`, `positions` (uint16), `step`, `mask`, `embeddingTable`, out `logits`, `states.sequence` (seqAxis 4, capacity buckets) | `functionFamily.phases.promptOnly` (prompt_opt, rank-0 logits) | qwen3-0.6b ios; lfm3-1.7b ios-h19p (shapes unverified, h19p only). Merges the `decoder-static-chunks@1` draft name into this id. |
| `decoder-hostcache@1` | prefill + decode graphs | in `tokens`, `positions`, `mask`; `states.host[]` in→out pairs; prefill out `hidden`; decode out `argmaxValue`/`argmaxIndex` | — | qwen3.5-0.8b ios-gpu hc pair. **Reserved, not in v1**: b1 IR, does not load on macOS 27.2, shapes unknown. |

### 1.2 Vision-language and OCR

| driver | graphs | required roles | optional roles / blocks | bundles |
|---|---|---|---|---|
| `vlm-static-inject@1` | `vision` (graph kind `vision-tower@1`), `decoder` (the `decoder-lm@1` contract) | vision in `patches` or `pixels`, out `imageEmbeds`; decoder: everything `decoder-lm@1` requires plus static `imageEmbeds`; `image` block (`tokens`, `feed`) | vision out `deepstackEmbeds`; decoder static `deepstackEmbeds`, `ropeShiftStart`, `ropeShiftAmount`, `imageRC`; `image.rope`; decoder `prefill` function (pf64) | lfm2.5-vl-450m, qwen3-vl-2b, minicpm-v-4.7-1b, glm-ocr, mineru2.5-pro; *qwen3-vl-4b/8b, holo2-4b, minicpm-v-4.6, lfm2.5-vl-3b, north-micro-vision, mineru layout/* |
| `vlm-embeds@1` | `vision`, `decoder` with `prefill` + `decode` functions | vision in `pixels`, out `visualTokens`; decoder prefill in `embeds` [1,L,H], decode in `embeds` [1,1,H] + `position` (scalar); out `logits` (last row); `states.sequence` (static capacity); `tables.embedTokens`; `image.arrange` | `generation` guards (noRepeatNgram, runCap), fixed task prompts | unlimited-ocr |
| graph kind `vision-tower@1` | — | in `patches` [rows,dim] or `pixels` [1,3,H,W]; out ≥ 1 embedding role | in `posTable`, `keyBias`, `patchMask`, `unshuffleIndex` (bound from `tables`) | every VLM tower above; decider-2b-vision g256; d1-3b tower; d1-omni vision; clef g256/g448 (unverified) |

### 1.3 Vision and embeddings (all stateless and fully static)

| driver | required roles | optional roles | bundles |
|---|---|---|---|
| `detector-dense@1` | in `image`; out `preds` [1,A,4+1+C] | — | yolox-s |
| `detector-detr@1` | in `image` (or `features` chained from graph `backbone`); out `boxes`, `classLogits` | out `masks`; graph `backbone` (out `features`) | rf-detr nano, rf-detr nano split, rf-detr seg-nano; *other sizes* |
| `depth-estimator@1` | in `image`; out `depth` | out `confidence` | depth-anything-3-small; *base, fp32* |
| `clip-joint@1` | in `image`, `tokens`; out `imageEmbedding`, `textEmbedding` | in `attentionMask`; `unusedOutputs` | clip-vit-b32 |
| `image-to-image-tiled@1` | in `image`; out `image` | — | adcsr-x4 |
| `text-embedder@1` | in `tokens`, `attentionMask`; out `embedding` | — | embeddinggemma-300m |
| `late-interaction-retriever@1` | graph `queryEncoder` (in `tokens`, `attentionMask`; out `tokenEmbeddings`); graph `docEncoder` (in `image`; out `tokenEmbeddings`) | docEncoder in `pixelMask` (fill `ones`) | colmodernvbert |
| `gliner-classifier@1` | in `tokens`, `attentionMask`, `labelPositions`; out `labelLogits`; sequence `buckets` | — | gliner2.5-decide |
| `yes-no-classifier@1` | in `tokens`, `attentionMask`; out `probs` | — | shieldstral-3b (unverified); a reranker with the same shape (inferred) |

### 1.4 Decision

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `decision-lm@1` | `decoder` (the `decoder-lm@1` contract) | `decoder-lm@1` roles; `outputs.logits.space`; `decision` block | `image` block + `vision` graph + decoder `prefill` function | decider-0.8b, openthai-systemone (slots), qwen3.5-2b-decision, decider-2b-vision; apus-decision-v1-4b, system-one-scorer-4b (both not inspected) |
| `decision-hidden-head@1` | `decoder` (static chunk), optional `vision` | in `tokens`, `positions`; out `hidden` [1,S,H] (no logits); `states.sequence`; `limits.padTokenId`; `decision.head` (a named host reducer: `pointer-head`, `option-rows`) over `tables` | `image` block | kev-0.8b, d1-3b (decode unverified); *kev-4b* |
| `decision-joint-head@1` | `decoder` (hidden-only, S=64), `head` (bucketed functions t512..t4096), towers | decoder out `hidden`; head in `hidden`, `keyValid`, `qAvg`, `oAvg`, `gAvg`, `lexical`, `member`, `typeIds`, `qValid`, `oValid`; out `logits` (`space: options`); `tables.lexical` (about 2 GB raw fp16) | — | clef-flash (head inspected; decode and towers unverified) |
| `encoder-decider@1` | `encoder`; optional `vision`, `audio` | in `tokens`, `attentionMask`, `questionType`; out `markerLogits` or `scores` | in `prefixEmbeds`, `padMask`, `prefixMask`, `keepRight`; out `pooled`; function `act` (in `pooled`, `features`; out `actLogits`); sequence `buckets` | laya-multilingual, d1-omni-600m |
| graph kind `audio-encoder@1` | — | in `mel`, masks; out `prefix`; `preprocess.audio`; clip-length `buckets` | — | d1-omni audio |

Rejected: `gemma4-ple-lm@1`. The PLE gather happens in-graph, so `decoder-lm@1` with `tables` static inputs and
`tokenizer.renderer: "gemma4-turns"` covers it.

### 1.5 Generation

The generative drivers and their blocks are in §5: `latent-diffusion@1` (§5.4), `krea2@1` (§5.5–§5.6),
`flux2-klein@1` (§5.7), the `weights` table (§5.8) and the speech driver `kokoro@1` (§5.9), all normative; the video
driver `minimax-h3@1` (§5.10) is normative since G4.

---

## 2. Manifest schema v1

### 2.1 Principles that resolve most gaps

1. **The manifest declares only what the graph descriptor cannot show.** The driver reads shapes, dtypes, static
   versus dynamic dimensions, `imageSize`, `queries`, static batch and MMAX from `InferenceFunctionDescriptor` at
   load. The manifest names roles and gives meanings: which axis is sequence, which state is recurrent, which logits
   space, what preprocessing applies. This removes the encoder "limits vocabulary" gap (imageSize, queries,
   anchors, maskStride, outputTokens, staticBatch, maxLabels) without adding fields. A shape or dtype written in a
   manifest is an *assertion* the driver checks, never an input.
2. **One manifest per catalog variant**, not per directory. A variant can span several subtrees (`assets`). Its
   driver can differ from other variants of the same id (qwen3-0.6b ios is `static-chunked-lm@1`).
3. **Every role binding has one shape:** `name | {name, binding, source}`. Static inputs, companion tables,
   graph-to-graph chaining and generated fills are the same mechanism.
4. **Driver-specific blocks are top-level and owned by one driver** (`image`, `decision`, `postprocess`, `labels`,
   `generation`). An unknown block is ignored. A missing block that the driver requires fails the load.
5. **Sidecar references.** Where a zoo bundle already ships a per-bundle contract (GLiNER `classifier.json`,
   EmbeddingGemma and Shieldstral `reference.json`, Unlimited-OCR `assets/recipe.json`, d1/laya metadata), a
   driver-block value may be `{"$ref": "<asset>:<path>#<json-pointer>"}`. The adapter resolves and inlines refs
   at install time, so the runtime never follows them. The ref stays in provenance so the golden shows where each
   value came from.
6. **`x-` fields** carry notes and status (`x-notes`, `x-status`, `x-observed`). They are ignored by validation.

### 2.2 Top level

| field | req | type | reason |
|---|---|---|---|
| `schema` | R | `1` | versioning (unchanged) |
| `id` | R | catalog id | unchanged |
| `task` | R | task id | unchanged; includes `image-text-to-text`, `object-detection`, `decision`, … |
| `driver` | R | `name@major` | unchanged; registry key |
| `variant` | R | `{platform: macos\|ios, architectures?: [h18p…]}` | one manifest per catalog variant. iOS variants can use another driver (qwen3-0.6b ios, lfm3 h19p). Per-arch AOT ships as **separate subtrees** (gemma-4 `…_aotc_h18p`, lfm3 `ios-h19p`, d1-omni and gliner `ios-h19p/`, laya `ios-h18p/`), so arch selection happens here and in `ModelRef`, not only in a per-file map. |
| `assets` | R | `{name: {repo?, revision?, subtree, files?: [{path, bytes, sha256}]}}` | Replaces the single bundle root. VLMs are 2–4 sibling subtrees; Gemma 4 needs `ios-frontend/gemma4_qat_gather_raw`; colmodernvbert needs `query/` + `doc/`; decider-2b-vision towers are absent from the catalog. `repo`/`revision` default to the main ones. `subtree` may name a `.aimodel` package (yolox, rf-detr, depth, adcsr), in which case the package is the asset root. `files` limits the download (Gemma 4 needs 2.35 of the 2.81 GB). The first asset is `main`. |
| `graphs` | R | `{graphRole: Graph}` | unchanged as a map; contents in §2.3 |
| `tables` | O | `{name: {asset, file, dtype, shape, layout?, sha256?}}` | non-graph data files: Gemma 4 PLE (`embed_per_layer.i8` int8 [262144×8960], scale f32), unlimited-ocr `embed_tokens.f16` + newline/separator rows + `prompt_input_ids.i32`, kev `head.safetensors`, d1-3b `option_rows` and `position_embedding`, clef `lm_head_fp16.bin`, d1-omni `position_table.f32` and mel filterbank. Sizes and hashes are needed by `ModelStore` and residency. |
| `tokenizer` | O (R for text drivers) | §2.4 | |
| `preprocess` | O | `{image?, audio?}` §2.5 | vision and audio drivers |
| `image` | O | §2.6 | `vlm-*`, `decision-lm@1` with images |
| `decision` | O (R for decision drivers) | §2.7 | |
| `postprocess` | O (R for detector, depth, SR, retriever, classifier) | §2.8 | §8 names a postprocess function but gives it no parameters |
| `labels` | O | `{set: "coco-91"}` or `{file: "<asset>:<path>"}` (a JSON array of names) | COCO-91 is hardcoded in coreai-kit and absent from every bundle. A manifest names a built-in set, or our own export ships a file. |
| `generation` | O | `{noRepeatNgram?, runCap?, prompts?: {task: text}, temperature?}` | unlimited-ocr's gated recipe needs `no_repeat_ngram 35` and run cap 6; OCR task prompts |
| `calibration` | O | `{temperature} \| {byType} \| {byTypeAndOptionCount: {"choice:2":…, "choice:3-5":…}}`, plus `source` | qwen3-0.6b and minicpm5 have catalog temperatures. Decision bundles keep them in metadata, a host-head json (kev) or only on the card (decider-0.8b 1.03). Precedence: app > catalog > manifest > driver default. This matches coreai-kit, so it reproduces the recorded distributions. |
| `precision` | O | free-form `{weights, activations, kv?, logits?, components?}` | informational only. The bf16 KV on Gemma 3 is read from the descriptor; the driver's supported-type set must include bf16 states. |
| `provenance` | O | `{source, sourceRevision?, base?, adapter?, zooRepo, zooRevision, subtrees, zooCompression?, producer?, exportScript?, zooEngineHint?, metadataKind?, gate?, refs?}` | traceability. `zooEngineHint` ("pipelined", "static-shape") is recorded but **never** drives behaviour. |
| `license` | O | SPDX-ish string | the bundle's own claim. The catalog value wins. |

Not part of v1 (earlier draft names): top-level `companions` (now `assets` + `tables`), `bundleRoot` (now `assets`),
`platformVariants` (each variant gets its own manifest), `limits.imageSize`/`queries`/`staticBatch`/… (descriptor),
`config` (replaced by `$ref`), `_fileAlternates` (§2.3 `file` is always one choice; see below).

### 2.3 `graphs.<role>`

| field | req | type | reason / bundles |
|---|---|---|---|
| `file` | R | `"<asset>:<path>"` or `{arch: path, "*": path}` | the asset prefix handles cross-subtree graphs. `"."` means the asset itself is the package. The per-arch map stays for our own exports, which put both in one bundle. A directory holding several graphs (mineru `decoder/` has s1 + pf64) is resolved by naming the file explicitly. The manifest, not inspect order or `assets.main`, picks the file. |
| `function` | R (one of the three) | string | the single-function case (unchanged) |
| `functions` | ↑ | `{phase: name}`; phases `step`, `prefill`, `decode`, `act`, … | lfm3 pf64 and mineru pf64 (`step: main`, `prefill: prefill`, S=64), decider-2b-vision (`prefill` S=16), unlimited-ocr (`prefill` + `decode`, no `main`), laya (`main` + `act`). All phases of one graph share its `states`. |
| `functionFamily` | ↑ | `{pattern, axes: {capacity: [...], chunk: [...]}, phases: {extend: pattern, promptOnly?: pattern}}` | qwen3-0.6b ios (`extend_{capacity}_{chunk}`, `prompt_opt_…`), lfm3 h19p, clef head (`t{tokens}`). The driver validates that every expanded name exists. |
| `buckets` | O | `{axis: "sequence"\|"audioSeconds"\|…, rule: "smallestFitting", entries: [{value, file\|function\|asset}]}` | separate files or folders per shape: GLiNER s256/s512, d1-omni decide L64…L4096 (7) and audio 5/10/20/30 s (4). Per-platform subtree choices (Shieldstral s512 macOS vs s256 iOS) go through `variant`, not buckets. |
| `inputs` | R | `{role: name \| Binding}` | closed role set per driver (§1) |
| `inputsByFunction` | O | `{phase: {role: …}}` | unlimited-ocr prefill (`embeds` [1,115,H]) and decode (`embeds` [1,1,H] + scalar `position`) have different input sets. When absent, `inputs` applies to every phase. |
| `outputs` | R | `{role: name \| {name, rows?, space?}}` | `rows: all\|last\|none` resolves logitsPositions/logitsAt/logitsRows. The adapter infers it from shape (dim 1 static 1 → last; rank 0 → none). It has to be written only for a prefill phase with S>1 that returns [1×1×V] (decider-2b-vision). `space: vocab\|slots\|options\|scalar` (openthai 256 slots, scorer 1, clef head 128). |
| `unusedOutputs` | O | `[name]` | CLIP `logits_per_image`/`_text`, DA3 `depth_conf` when not surfaced, so validation does not flag them as unbound |
| `states` | R for stateful drivers | §2.3.1 | |
| `limits` | O | §2.3.2 | |
| `compute` | R | `gpu\|neuralEngine\|cpu` | Written explicitly so nothing falls back to Core AI defaults. **The adapter default is `gpu` for every graph class** (measured on 11 graphs in 5 classes: the `.neuralEngine` preference gave GPU-identical output on 9, and the only graph that clearly moved, gliner, was 6× slower; CPU fails or is wrong on 6 of 11). `neuralEngine` is allowed only as a per-entry override backed by a measured win (none today, iPhone power not yet measured). |
| `expectFrequentReshapes` | O | bool, default = `compute == gpu` and any dynamic dimension | the M0 4 vs 274 tok/s finding. Decode-only graphs mix static tokens with dynamic KV, so the "shape kind" rule is ambiguous. |
| `warmup` | O | `false` or `{shapes}` | gemma-4 S=1 graph rejects the default S=256 warmup (`noWarmup`) |

**Binding** (input roles):

```jsonc
{ "name": "ple_table",
  "binding": "perCall" | "static",      // static = bound once, rewritten only on events (image change, load)
  "source":  { "table": "ple", "storage": "owned" | "mapped" | "auto" }   // from tables
           | { "from": "backbone.features" }                             // graph chaining
           | { "fill": "zeros" | "ones" | { "value": 1073741824 } }      // generated
}
```

This covers Gemma 4 PLE (`table`, storage `auto`: owned when there is memory headroom, otherwise copy-on-write
mapped), VLM `imageEmbeds`/`deepstackEmbeds`/`rope*` (`static`, written by the image block), the RF-DETR split head and
minicpm's `image_features` → `image_embeds` rename (`from`), colmodernvbert's `pixel_attention_mask` (`fill: ones`),
and `embedding_table` chaining in qwen3-0.6b ios (`from: "embeddings.embeddingTable"`).

**`inputEncodings`** (O): `{role: {dtype?, range?, pad?}}`. Use it only when the host encoding is not the
descriptor's dtype or the default convention. Examples: positions `uint16` (qwen3-0.6b ios), positions
`{range: "0..<pos+S", pad: -1}` (the decoder-lm default, so normally omitted), and `tokens: {extensionIds: {base,
count}}` for ids ≥ vocab (VLMs, decider-2b-vision 248320+k, d1-3b 128000+k).

#### 2.3.1 `states`

```jsonc
"states": {
  "sequence": [   // ≥ 1 for decoder drivers: tensors indexed by sequence position
    { "tensors": ["keyCache", "valueCache"], "kind": "kv" | "latent", "seqAxis": 3,
      "layers": [2,4,…]?, "window": { "rows": [..], "size": 256 }?, "carriesNonAttention": false? }
  ],
  "recurrent": [  // optional; any entry forces rewind by snapshot/restore
    { "name": "convState", "kind": "conv" | "ssm", "init": "zeros", "layers": [...]? }
  ],
  "host": []      // reserved for decoder-hostcache@1 (in→out pairs); rejected by v1 drivers
}
```

| field | req | reason |
|---|---|---|
| `sequence[].tensors` | R | replaces `kv: [k, v]`. Youtu's MLA `kvCacheA`/`kvCacheB` are not a key/value pair, so the "at least one kv pair" rule becomes "at least one sequence state". |
| `sequence[].kind` | O, default `kv` | `latent` for MLA (youtu, glm-4.7-flash inferred) |
| `sequence[].seqAxis` | **R** | it is 3 in every dynamic graph and 4 in qwen3-0.6b ios and lfm3 h19p. §6.2 says "declared axis" but §4 has no field. The adapter writes it from the dynamic dimension, or from the override table for static graphs. |
| `sequence[].layers`, `window`, `carriesNonAttention` | O | informational; a parity check, never used for indexing. Covers lfm3 Mac (11 rows mixing full and sliding-256), lfm3 h19p (conv history in KV rows) and gemma-4 KV sharing (15 of 35 layers). |
| `recurrent[].name` | R if present | the zoo names are fixed (`convState`, `recState`), but the classification must be explicit (§6.2) |
| `recurrent[].kind`, `init`, `layers` | O (`init` default `zeros`) | rewind semantics follow from the presence of the entry. Kind and layer map are for diagnostics. |

Dtype and shape are not manifest fields. bf16 (Gemma 3) and fp16 come from the descriptor.

#### 2.3.2 `limits`

| field | req | reason / bundles |
|---|---|---|
| `maxContext` | R for decoders | the model's **declared** maximum (metadata `max_context_length`). It is **not** a capacity: the engine sizes KV at load as `min(maxContext, kvCapacityCap[platform], what fits the device memory budget after weights)`, grows geometrically, and throws `capacity` when an app asks for more. Gemma 3 4B declares 131072 (≈18.3 GB of bf16 KV), so on a phone the budget, not the manifest, decides. |
| `kvCapacityCap` | O | `{ios: 1024}`: the nanbeige iOS miscompile (bisected on 4.2, applied to 4.1 as inferred) |
| `maxPosition` | O | default `maxContext - 1`. d1-3b allocates 4096 but positions stop at 4095. |
| `staticSequence` | O | int or set (for example `[8,16,64]`), per phase when `functions` is used: `{step: 1, prefill: 64}`. Usually inferred from the descriptor; kept so the golden shows it. |
| `prefillChunk` | O | default 512, and min(512, static S) for static-S prefill functions. M0 correctness limit. |
| `padTail` | O, default false | whether a static prefill function accepts a short last chunk (zoo `language.prefill_pad_tail`). If false, the remainder goes to `step` at S=1. |
| `padTokenId` | O (R for static-chunk hidden backbones) | kev 248044, d1-3b 124893. Padded rows are discarded. |

### 2.4 `tokenizer`

| field | req | reason |
|---|---|---|
| `source` | R | `{dir: "<asset>:tokenizer"}` \| `{file: "<asset>:tokenizer.json"}` (CLIP keeps tokenizer.json at the subtree root; colmodernvbert has it only under `query/`) \| `{hf: "mistralai/…", revision}` (mistral-7b has no tokenizer dir) \| `{embedded: true}`. Replaces `dir`. |
| `chatTemplate` | O, nullable | null for Gemma 4, decision formats and encoders |
| `chatTemplateSha256` | O | lfm3 metadata carries it. Lets goldens detect template drift. |
| `renderer` | O, default `jinja` | `gemma4-turns` (Gemma 4 has no jinja template), plus GLM `[gMASK]<sop>` framing if its template is insufficient |
| `turnFormat` | O | renderer parameters (Gemma 4 `<|turn>`/`<turn|>`, `model` role) |
| `addBos` | O, default `template` | `template\|renderer\|true\|false`. With a chat template (or renderer), the **template owns BOS and the runtime adds none**. Measured: swift-transformers `applyChatTemplate` is token-identical to HF `apply_chat_template` on 11 chat bundles × 3 shapes (33/33), including Gemma 3 and Nanbeige, so there is no double BOS. Plain-text encoding (encoders) follows `addSpecialTokens`. |
| `addSpecialTokens` | O | Shieldstral and decision prompts use `false` |
| `stop` | R for generation | list of strings or ids: the **source model's `generation_config.json` `eos_token_id` list** (pinned in the override table, because zoo bundles don't ship it) unioned with the tokenizer's eos. Gemma 4 is `[1 <eos>, 106 <turn|>, 50 <|tool_response>]` (measured; `<|tool_response>` is the tool-call hand-back, which coreai-kit misses). Gemma 3 `[1 <eos>, 106 <end_of_turn>]` (verified from the gated `google/gemma-3-4b-it@093f9f3` config; 12B identical). Granite `<|end_of_text|>`. |
| `thinking` | O | `{open, close, toggle: {templateVar: "enable_thinking"} \| {emitEmpty: "<|channel>thought\n<channel|>"} \| "none", defaultOn}`. defaultOn is true for LFM3 and Nemotron, false for Qwen3.5. Nanbeige has no toggle. |
| `toolDialect` | O | §7 already names it; values `hermes-json` (Granite `<tool_call>` JSON), `lfm3-xml` |
| `text` | O | `{padId, padSide, padTo, truncation: right\|word\|bodyOnly, prompts: {query, document}, template: {prefix, suffix, body}, unusedBatchRows, maskRule}`. Values: embeddinggemma pad 0 right to 256 with asymmetric prompts; colmodernvbert right-pad 50283 to 32; Shieldstral flattened prefix/suffix with body-only truncation and pad 11; CLIP pad = EOT 49407 to 77, mask through first EOT, dummy rows. |
| `escapes` | O | `[{from: "<|", to: "<¦"}…]` for kev and d1-omni user-text escaping |
| `preTokenize` | O | `icuCodePoints` (openthai Thai) |
| `class` | O | tokenizer class override (GLiNER `XLMRobertaTokenizer` stand-in; CLIP custom lowercase BPE) |
| `markerIds` | O | GLiNER `[P]`, `[L]`, … (normally `$ref` into classifier.json) |

### 2.5 `preprocess`

`image` (O), typed since M4a (`Manifest.ImagePreprocess`; DESIGN.md §8). Every field is optional; the defaults are in
brackets:

```
{resize?: {mode: stretch|letterbox|centerCrop|fitLongSide, size: [width, height], maxSide, anchor: center|topLeft [center],
           padValue: [v] | [r, g, b] [0], filter: vImage|pillowBicubic|torchUint8Bicubic},
 colorSpace: deviceRGB|sRGB [deviceRGB], channelOrder: RGB|BGR [RGB], scale [1], mean [0,0,0], std [1,1,1],
 layout: NCHW|NVCHW|patches [NCHW], views [1], patchify?: {patch, temporal [1], merge [1], rowOrder: blockMajor|rowMajor,
 withinPatch: "C,T,py,px"|"py,px,C"}, hostDType: float32|float16 [float32], orientation: exif [exif]}
```

- **Order:** decode (EXIF orientation applied, alpha by the app's `ImageInput` policy) in `colorSpace` (`deviceRGB` keeps
  stored values, as Pillow decodes; `sRGB` converts), resize on 8-bit RGB, then `(x · scale − mean) / std` in float32
  (multiply, subtract, divide), `mean` and `std` given in RGB order, then `channelOrder`, `layout` and `hostDType`
  (rounded to nearest even).
- **Resize:** no `resize` keeps the decoded size. `stretch` goes to `size`. `letterbox` fits inside `size` (the scaled
  side floored) at `anchor` on an 8-bit `padValue` canvas. `centerCrop` covers `size` (scaled side floored) and cuts
  the centre at `round((scaled − size) / 2)`, ties to even (torchvision). `fitLongSide` scales an image whose longer
  side exceeds `maxSide` down to it (other side floored); smaller images are untouched. `size` goes with the first
  three modes, `maxSide` only with `fitLongSide`, `anchor` and `padValue` only with `letterbox`.
- **Filters:** `vImage` is Accelerate's `vImageScale` (fast, its own kernel, not a parity filter). `pillowBicubic` is
  Pillow's `Image.resize(BICUBIC)` and `torchUint8Bicubic` torch's uint8 `interpolate(bicubic, antialias=True)` (what
  torchvision's `resize` of a uint8 tensor and Hugging Face's fast processors run); both are reproduced bit for bit. A
  filter is added when an export needs one (`pillowBilinear`, `coreGraphicsHigh`, …); until then the name is rejected.
- **Layouts:** `NCHW` is `[1, 3, H, W]`; `NVCHW` repeats the image in `views` views, `[1, V, 3, H, W]`; `patches` is
  `[rows, 3 · temporal · patch²]`, rows block-major (each `merge × merge` block contiguous, Qwen-VL) or row-major
  (SigLIP 2), each row `C,T,py,px` (the frame repeated `temporal` times) or `py,px,C` (`temporal` 1). The image's sides
  must be multiples of `patch · merge`.
- **dtype:** binding converts `hostDType` to the graph input's own type, read from the descriptor (DA3 is fed fp16 from
  an fp32 host), and checks the input's shape against the prepared one.
- `x-` keys are ignored, as everywhere (§2.1.6). Later slices add `tiling` (AdcSR), `crop` tiles and smart sizing (VLM)
  when their drivers land.

Each field has a bundle that needs it: BGR, 0–255 and letterbox top-left pad 114 (yolox); `[-1,1]` (adcsr: scale
1/255, mean 0.5, std 0.5); NVCHW (colmodernvbert, DA3 multi-view); patchify, which fails silently when wrong (qwen3-vl
block-major C,T,py,px; lfm channel-last row-major); fp16 graph fed from fp32 host (DA3); tiling with overlap
`max(4, tile/8)` (adcsr); the white letterbox pad (glm-ocr); Pillow vs torch filters (decision towers, VLMs).
Postprocess-side resampling such as resize-back goes in `postprocess`.

`audio` (O): `{sampleRate: 16000, nFFT: 512, hop: 160, nMels: 128, melScale: slaney, filterbank: {table}, normalize:
perRow, clipBuckets}` (d1-omni).

### 2.6 `image` (VLM and vision-decision)

| field | req | reason / bundles |
|---|---|---|
| `feed` | R | `[{from: "vision.imageEmbeds", to: "decoder.imageEmbeds"}]`. Handles minicpm's rename. |
| `tokens` | R | `{pad, padId, open?, openId?, close?, closeId?, count, idBase}`. The pad ids are rewritten to `idBase + slot`, and `tokens` must accept ids ≥ vocab. |
| `grid` | O | merged grid [h, w]. It cannot be read from the graph (196 rows does not say 14×14). Needed by the rope rule. Glm/mineru 682 = 22×31 is inferred. |
| `rope` | O | `{start: expr, amount: expr\|int, frameTail, textOnly: {start: 1073741824, amount: 0, zeroStatics: true}}`. Covers qwen3-vl, minicpm-v-4.7 (frameTail 2), glm, mineru, decider-2b-vision. The expression grammar is closed: `imageStart`, `count`, `gridH`, `gridW`, `frameTail`, integers, `+ - max`. |
| `deepstack` | O | `{rowsPerToken: 3}` (qwen3-vl, north) |
| `maxImages` | O, default 1 | every export has one static image buffer at a fixed canvas |
| `arrange` | O (`vlm-embeds@1` only) | `{grid: [10,10], rowSuffix: {table: "imageNewline"}, trailer: {table: "viewSeparator"}, placeholderId: 128815, promptIds: {table}}` (unlimited-ocr) |

### 2.7 `decision`

`{promptFormat: decider|slot|decisionFunction|sharedState|scalar|kevRow|d1|d1Omni|clefJoint|deciderVision|encoder,
 layout, labels: {rule: "spacePrefixedAZ"|"bareAJ"|…, ids?}, slot: {rule: lastToken|answerTokenId|tokenPattern, answerTokenId?, pattern?},
 nSlots?, abstainSlot?, maxOptions, multiSlotRows?, rowPerOption?, state: sharedPrefix|freshPerRow,
 noulOrder?, questionTypeOrder?, readoutMarkers?, head?: {reducer: pointerHead|optionRows, tables, temperature?}}`.

All fields are optional except `promptFormat` and `maxOptions`. This replaces directory-name matching
(`contains("decider")`). Values come from bundle `metadata.decision` where present (openthai, qwen3.5-2b-decision,
scorer), and from the override table otherwise (decider-0.8b, apus). Label ids are resolved by the adapter from
vocab and written as ids, so qwen3.5-2b's ` A`=357 … become explicit.

### 2.8 `postprocess`

Typed since M4a (`Manifest.Postprocess`): `kind` picks the variant. The detectors' fields sit beside `kind`; every other
kind's sit in its own sub-object. Interpreters arrive with each driver; decoding and validation are already closed.

| `kind` | driver | fields (defaults in brackets) |
|---|---|---|
| `depth` | `depth-estimator@1` | `depth: {polarity: largerIsNearer\|largerIsFarther, resizeBack [false]}` |
| `denseNms` | `detector-dense@1` | `boxFormat: cxcywh\|xyxy, boxSpace: letterboxPixels\|normalized, activation: applied\|sigmoid [applied], nms: {type: perClass\|classAgnostic, iou ∈ (0, 1]}, defaults: {scoreThreshold ∈ [0, 1], maxDetections ≥ 1}, classIndex: contiguous\|cocoId [contiguous]` |
| `detrSigmoid` | `detector-detr@1` | `boxFormat, boxSpace, defaults, classIndex` as above, `masks?: {logitThreshold [0]}` |
| `embedding` | `clip-joint@1`, `text-embedder@1` | `embedding: {normalized: inGraph\|host\|none, pooling: inGraph\|mean\|cls [inGraph], similarity: cosine\|dot}` |
| `gliner` | `gliner-classifier@1` | `gliner?: {threshold [0.5]}` (multi-label sigmoid threshold) |
| `yesNo` | `yes-no-classifier@1` | `probs: {positiveIndex, threshold [0.5]}` |
| `maxSim` | `late-interaction-retriever@1` | `retrieval: {imageTokenSpan: [start, end), patchGrid: [rows, columns]}`, `rows · columns = end − start` |
| `superRes` | `image-to-image-tiled@1` | `sr: {range: [low, high], colorMatch: host\|none [none]}` |

`letterboxPixels` means pixels of the graph's input, letterboxed or not; the driver maps boxes back to the source
through the prepared image's geometry. DA3 is `largerIsFarther` (its exp head, checked on real frames). `depth`'s
`resizeBack` resamples the map (and confidence) to the source through the geometry, bilinear at pixel centres and
clamped to the content rectangle (M4b, `DepthEstimator`).

Required for every vision/embedding driver, because none of these values is in the bundles. Thresholds are
defaults, and the app can override them.

### 2.9 What the adapter cannot infer: the per-entry override table

`ZooBundleAdapter` reads catalog `kind`, bundle `metadata.json` (where present), the graph descriptors, the tokenizer
config and sidecars. Everything below must come from a checked-in **override table** keyed by
`(catalogId, variant)` at the pinned revision (`aether-kit/catalog/zoo-overrides.json`, proposed). The table is
covered by the per-entry golden tests and is the only place family knowledge lives.

| field | entries needing an override | why it can't be inferred |
|---|---|---|
| `driver` | all vision/embedding entries; VLM vs LM; decision drivers | no bundle-level metadata on vision bundles; VLMs say `kind: llm`; coreai-kit dispatches by catalog id string (yolox-s, colmodernvbert) |
| `assets` beyond the catalog subtree | gemma-4-e2b/e4b (tables), every VLM tower, colmodernvbert query, decider-2b-vision towers, clef towers/head, d1-omni buckets, d1-3b tower | catalog lists one subtree; the companions live only in coreai-kit presets |
| `tables` | gemma-4, unlimited-ocr, kev, d1-3b, clef, d1-omni | file roles are not declared anywhere machine-readable except partly in recipe.json/meta.json |
| `states.sequence` kind/tensors for non-standard names | youtu (`kvCacheA/B`) | name heuristic fails |
| `states.recurrent` | only exceptions to the rule (decided): a state with a dynamic sequence axis is `sequence`, and a fully static state is `recurrent`. Exceptions such as youtu's `kvCacheA/B` (dynamic but MLA latent) are overrides. Goldens pin every result. | no zoo `states` key exists |
| `seqAxis` for fully static graphs | qwen3-0.6b ios, lfm3 h19p, unlimited-ocr | no dynamic dimension to find |
| `limits.maxContext` cap | none: capacity is derived from memory at load (DESIGN.md §6.2). Only bug caps (`kvCapacityCap`) are overrides. | — |
| `limits.kvCapacityCap` | nanbeige4.1/4.2 iOS | bug knowledge |
| `compute` | none today: the measured default is `gpu` for every class. An entry gets `neuralEngine` only with a recorded measured win. | measured, not declared |
| `warmup: false` | gemma-4 S=1 graphs | README only |
| `tokenizer.source` | mistral-7b: `{hf: "mistralai/Mistral-7B-Instruct-v0.3", revision: "c170c708c41dac9275d15a8fff4eca08d52bab71"}`, tokenizer files listed with sha256 as a catalog asset | the bundle has no `tokenizer/`, despite `embedded_tokenizer: true`. The fork then loads unpinned HF `main` at runtime. The adapter must check that the directory exists and never trust the flag. |
| `tokenizer.renderer`/`turnFormat` | gemma-4-\* | no jinja template |
| `tokenizer.addBos` | none (measured: the template owns BOS everywhere tested) | — |
| `tokenizer.stop` | every generating entry: the pinned `generation_config.eos_token_id` list (gemma-4 adds `<|tool_response>`; gemma-3 `<end_of_turn>`) | zoo bundles don't ship generation_config |
| `tokenizer.thinking.toggle`/`defaultOn` | gemma-4 (`emitEmpty`), nanbeige (`none`) | template parsing is not reliable |
| `tokenizer.toolDialect` | granite, lfm3, qwen3/3.5 family | not in metadata |
| `tokenizer.text` | embeddinggemma prompts and pad; colmodernvbert pad side (contradicts tokenizer.json); clip mask rule; shieldstral template (`$ref` reference.json) | hardcoded in coreai-kit |
| `preprocess.image` | every vision, VLM and decision-tower entry | not in any bundle (normalization/filter for glm and mineru come only from coreai-kit presets) |
| `image.*` (`grid`, `rope`, token strings) | all VLMs; minicpm-v-4.6/4.7 and lfm/north are partly in metadata `vlm{}` / `image_tokens` | VLArchitecture constants; mineru/glm presets disagree with graphs |
| `postprocess`, `labels` | all vision/embedding entries | not in bundles |
| `decision.*` | decider-0.8b, apus (no metadata block); all others partially | card + coreai-kit code |
| `calibration` | decider-0.8b (1.03, card only) | card only |
| `graphs.*.file` choice in multi-graph dirs | mineru (s1 vs pf64), shieldstral s512/s256 | metadata `assets.main` and inspect disagree |
| `generation` | unlimited-ocr | recipe |

The adapter infers these for every entry: graph names and roles for the standard zoo LM names (`input_ids`,
`position_ids`, `keyCache`, `valueCache`, `logits`), the sequence axis of dynamic graphs, `staticSequence`, logits
`rows`, `functions` from `language.function_map` (but not for lfm3 h19p, whose `main` map is wrong), `maxContext`
from metadata, `stop` from eos, the chat template path and sha, `thinking` tags when the template uses
`enable_thinking`, and `provenance`.

---

## 3. Contract deviations the spec must handle

| deviation | bundles that show it | how v1 handles it |
|---|---|---|
| **Static token axis, dynamic KV** (`input_ids [1×1]`, `position_ids [1×?]`) | youtu, nanbeige, gemma-4, qwen3.5, lfm2.5, granite, nemotron, every VLM decoder, decision LMs | `staticSequence: 1` (inferred); S=1 prefill; `expectFrequentReshapes` explicit |
| **Static chunk with padding**, hidden-only | kev (S=128), d1-3b, clef decode (S=64) | `padTokenId`; driver discards pad rows; `decision-hidden-head@1` |
| **Last-only logits** | all S=1 decoders; decider-2b-vision prefill [1×16]→[1×1×V]; unlimited-ocr prefill | `outputs.logits.rows: last` |
| **No logits** (rank-0, hidden, argmax partials) | qwen3-0.6b ios `prompt_opt_*`; kev/clef/d1-3b `hidden`; qwen3.5 ios-gpu decode `head_pv/head_pi` | `rows: none`; hidden-only drivers; hostcache reserved |
| **Non-vocabulary logits** | openthai (256 slots), scorer (1), clef head (128) | `outputs.logits.space` |
| **Recurrent states** (conv, SSM) | qwen3.5, lfm2.5, lfm3 Mac, granite, nemotron, minicpm-v-4.7, decider-0.8b, openthai, qwen3.5-2b-decision, decider-2b-vision, kev, d1-3b (conv only) | `states.recurrent[]`; snapshot rewind |
| **Sequence states that are not k/v** | youtu (MLA latent); lfm3 h19p (conv history inside KV rows); lfm3 Mac (sliding + full rows in one tensor); gemma-4 (15 of 35 layers) | `states.sequence[].kind/window/carriesNonAttention` |
| **KV sequence on axis 4, capacity buckets** | qwen3-0.6b ios, lfm3 h19p | `seqAxis` required; `functionFamily.axes.capacity` |
| **bf16 KV** | gemma-3-4b-it (+12b) | descriptor-driven dtype; bf16 in the supported set |
| **Static KV** | unlimited-ocr (2048) | descriptor static axis → fixed capacity, `maxContext` = capacity |
| **Multi-graph bundles across sibling subtrees** | all VLMs, colmodernvbert, rf-detr split, decider-2b-vision, clef, d1-omni, d1-3b, gemma-4 tables | `assets` + `"<asset>:<path>"` |
| **Subtree is the .aimodel package** | yolox-s, rf-detr, rf-detr seg, depth-anything-3, adcsr | asset root = package; manifest stored outside it (§4 edit) |
| **Per-arch AOT `.aimodelc` in its own subtree; no in-bundle fallback; wrong-arch load fails** | gemma-4-e2b `_aotc_h18p`, lfm3 `ios-h19p` (different driver), d1-omni/gliner `ios-h19p`, laya/nemotron `ios-h18p` | `variant.architectures`, resolved in `ModelRef` **before** download; per-file map only for our exports |
| **Prefill as a second function sharing states** | lfm3 pf64, mineru pf64, decider-2b-vision pf16 | `functions: {step, prefill}`, per-phase `staticSequence`, `padTail` |
| **Different inputs per function** | unlimited-ocr | `inputsByFunction` |
| **Bucketed function families in one file** | qwen3-0.6b ios (34 fns), lfm3 h19p, clef head t512…t4096 | `functionFamily` |
| **Bucketed graph files** | gliner s256/s512, d1-omni L64…L4096 + audio buckets | `buckets` |
| **Graph output chained into inputs** | qwen3-0.6b ios `embedding_table`, rf-detr split, every VLM tower→decoder | `source.from` / `image.feed` |
| **Tables and sidecars** | gemma-4 PLE (2.35 GB), unlimited-ocr embed table, kev head, d1-3b option_rows, clef lm_head (2 GB), d1-omni position table/mel; classifier.json, reference.json, recipe.json | `tables`; `$ref` |
| **Ids ≥ vocab in tokens** | all VLMs, decider-2b-vision, d1-3b | `inputEncodings.tokens.extensionIds` |
| **Host-built masks/steps, uint16 positions** | qwen3-0.6b ios, lfm3 h19p | `static-chunked-lm@1` roles + `inputEncodings` |
| **Host-held functional state** | qwen3.5 ios-gpu pair | reserved (`states.host`), not loadable |
| **Static graph not on ANE** | laya, most vision graphs, kev | `compute` required |
| **Metadata drift vs graph** | mineru/glm (VLArchitecture 3072/768 vs graph 2728/682), lfm3 h19p `function_map: main`, openthai `vocab_size: 256`, gliner coreai-kit pin ≠ catalog pin, embeddinggemma comment says fp16 but graph is fp32 | descriptor wins; adapter goldens assert shapes against the descriptor at install |
| **No bundle-level metadata** | every vision/embedding bundle, vlm towers, unlimited-ocr | override table + `$ref` sidecars |

## 4. Resolved during M1 implementation (2026-11-05)

These clarify or amend §1–§3. The implementation and the per-entry goldens follow them.

| # | Topic | Rule |
|---|---|---|
| 1 | `main` asset | Every variant has an asset **named** `main`. JSON object order is not preserved, so "the first asset" means nothing. |
| 2 | `expectFrequentReshapes` | **Required when `compute` is `gpu`, and written by the adapter.** The default ("gpu and anything dynamic") needs the descriptor, which only exists after specialization, so it can't be computed at load. |
| 3 | `outputs.<role>.rows` | May be a single value or a per-phase map (`{"step": "last", "prefill": "all"}`) for graphs with `functions`. |
| 4 | Static table inputs | Role names are camelCase, prefixed by table (`pleTable`, `pleScale` for Gemma 4). |
| 5 | Foreign assets | Assets from another repo (Mistral's tokenizer) install under `<revision>/.aether/assets/<name>/`. The install receipt is `<revision>/.aether/receipt.json`. |
| 6 | `provenance.subtrees` | A map from asset name to subtree. |
| 7 | File sizes | Manifest `assets.files[].bytes` and catalog `files[].size` mean the same thing (decimal bytes). They should be unified (to `size`) in schema 1.1; both are accepted now. |
| 8 | Unknown vs unsupported driver | A driver id **in the catalog** (§1) with no registered driver gives `unsupported`. An id **outside** the catalog gives `unknownDriver`. A graph kind used as a top-level driver is rejected. |
| 9 | `thinking.defaultOn` | Comes from the override table, citing the template text: qwen3 `true`, youtu `true`, qwen3.5 `false`, nemotron `true`. It is left unset where the template emits nothing when undefined (MiniCPM5) or has no toggle (Nanbeige). |
| 10 | `toolDialect` | `hermes-json` for qwen3 and granite. **Qwen3.5 uses an XML dialect** (`<tool_call><function=…><parameter=…>`), to be named `qwen-xml` when M3 implements it. Youtu, Nanbeige and Nemotron also emit `<tool_call>`, but their dialect is unverified, so it stays unset until M3 gates it. |
| 11 | Structured metadata | When a zoo field changes type (nemotron `compression` is `{"scheme":"int8hu"}`), the adapter keeps it as compact JSON in `provenance.zooCompression` and never interprets it. |
| 12 | AOT-only iOS variants | nemotron-3-nano and gemma-4 ship iOS graphs **only** as `ios-h18p` AOT. They resolve to `unsupportedDevice` on other iPhones: intended, but a behaviour change from coreai-kit. |
| 13 | One install per revision | `<repo>/<revision>/` holds exactly one variant, because a device resolves exactly one. A reinstall replaces it atomically. |

## 5. Generative drivers

Status: §5.1–§5.3 (the generic `schedule` and `latent` blocks and per-graph `residency`) are **normative since G0**,
2026-11-06, with `epsilonEuler` and spatial latents since G1; `latent-diffusion@1` (§5.4) is **normative since G1**;
`flux2-klein@1` and its `flux2` block (§5.7) are **normative since G2**; `krea2@1` and its `krea2` block (§5.5–§5.6)
are **normative since G3**. The producers are `Forge/krea2/export_krea2.py`, `Forge/sdturbo/export_sdturbo.py` and
`Forge/flux2/export_flux2.py`; their `--sample-manifest` output is the decoding fixture
(`Tests/AetherCoreTests/Fixtures/krea2/`, `…/sd-turbo/`, `…/flux2/`). Unlike the loose driver blocks of §2.5–§2.8, these blocks are fully typed:
the string-valued rules below are closed vocabularies, and anything else is rejected naming its field.

### 5.1 `schedule`

Any flow-matching or diffusion driver. Implemented by `FlowMatchSchedule` and `EpsilonEulerSchedule` (AetherRuntime).
Each kind takes exactly its own fields: `sigmas` only with `flowMatchEuler`, `tables` and `modelInput` only with
`epsilonEuler`, and each kind's one `timestep` and `update` rule.

| field | req | meaning |
|---|---|---|
| `kind` | R | `flowMatchEuler` (Krea, FLUX.2) or `epsilonEuler` (SD-Turbo: Diffusers' `EulerDiscreteScheduler`, epsilon prediction, gamma 0). H3's solver joins with its driver |
| `steps`, `maxSteps` | R, O | default and maximum step counts (`maxSteps` ≥ `steps`; absent: `steps`) |
| `trainTimesteps` | R | ≥ 1 (1000) |
| `tables` | R (epsilonEuler) | step count (`"1"`…`"maxSteps"`, exactly those) → `{timesteps, sigmas, initNoiseSigma}`: the pinned scheduler's values for that count, captured by the exporter (the runtime never derives an epsilon schedule). `timesteps` decreasing; `sigmas` one longer, decreasing, ending in 0; `initNoiseSigma` > 0 |
| `modelInput` | R (epsilonEuler) | `"x / sqrt(sigma[i]^2 + 1)"`: the denoiser's latent input, in Float |
| `sigmas.base` | R (flowMatchEuler) | `"linspace(1, 1/steps, steps)"`: in Double, each value rounded to Float, the last exactly `1/steps` |
| `sigmas.shift` | R (flowMatchEuler) | `{rule: exponential, mu}` (Krea: μ 1.15, `is_distilled`) · `{rule: empirical, empirical: {lowSteps, lowSlope, lowIntercept, highSteps, highSlope, highIntercept, tokenThreshold}}` (FLUX.2: μ from the generated token count and the steps) · `{rule: none}`. The shifted σ is `e / (e + (1/σ − 1))` in Float, `e = Float(exp(μ))` |
| `sigmas.terminal` | R (flowMatchEuler) | 0, appended after the last step (the only defined value) |
| `timestep` | R | flowMatchEuler `"sigma * trainTimesteps / trainTimesteps"`: what the denoiser's `sigma` input receives, in Float. epsilonEuler `"tables[steps].timesteps[i]"` |
| `update` | R | flowMatchEuler `"x + (sigma[i+1] - sigma[i]) * velocity"`: the host update in Float, the product rounded before the add. epsilonEuler `"x + (x - (x - sigma[i] * epsilon)) / sigma[i] * (sigma[i+1] - sigma[i])"`: Diffusers' order in Float, each product rounded; the initial latents are `noise · initNoiseSigma` |
| `guidance` | R | `{kind: none}` (no `scale`; Krea 2 Turbo is distilled) or `{kind: cfg, scale > 0}` |

### 5.2 `latent`

The image-latent contract.

| field | req | meaning |
|---|---|---|
| `channels` | R | per-token channels the denoiser sees (Krea 64 = 16 VAE channels × 2 × 2; SD-Turbo 4); a multiple of 4 for 2×2 packing |
| `pixelsPerToken` | R | 16 (Krea: VAE 8× × patch 2), 8 (SD-Turbo: VAE 8×); also the multiple width and height must be |
| `packing` | R | `patch2x2ChannelMajor`: packed channel `c·4 + 2·dy + dx` · `spatial`: unpatched `[1, channels, rows, columns]` (U-Nets) |
| `layout`, `decoderLayout` | R | descriptive: denoiser `tokens[rows*columns, 64]`, decoder `[rows, columns, 64]`; spatial `nchw[1, 4, rows, columns]` |
| `noise` | R | `{kind: splitmix64BoxMuller, order, dtype: float32}`: `SeededGaussianNoise`, prefix-stable and bit-identical to 0.0.x's stream. `order` is `packedToken` for packed latents and `nchw` (row-major `[1, C, H, W]`) for spatial ones |
| `maxImageTokens` | O | the export's bound on `rows · columns` (4096) |
| `sizes` | O | `[[width, height], …]`: the sizes a fixed-geometry export supports, the first the default (SD-Turbo `[[512, 512]]`). An assertion the driver checks against the graphs' static shapes; a request for another size is refused before anything runs. Absent: any size the grid rules allow |

### 5.3 `residency` and other field additions

| where | field | meaning |
|---|---|---|
| `graphs.<role>` | `residency` | `perPrompt` (text encoder: load, encode, releasable), `perStep` (denoisers: resident through the steps), `perImage` (decoder). Absent: resident for the model's lifetime. `ResidencyPlan` keeps every graph resident when the measured peak fits, else releases each phase's graphs at its end (only when every graph has a residency) |
| top level | `x-status`, `x-license` | draft marker; license files, `modified: true`, revenue cap, adapter permissions |
| `provenance` | `mergedAdapter` | `{file, sha256, format, merge, multiplier, scale}` for an adapter merged before export (realism) |
| `provenance.refs` | `diffusersCommit`, `forgeLockSHA256`, `exportReport` | the exporter's environment pins |
| `license` | `LicenseRef-Krea-2-Community-License-1.0`, `LicenseRef-Stability-AI-Community-License` | an SPDX `LicenseRef`; the bundle ships the license (Krea: `LICENSE.pdf`, the text, the use policy; SD-Turbo: `LICENSE.md`) and `NOTICE` |

### 5.4 `latent-diffusion@1` (normative since G1)

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `latent-diffusion@1` | `textEncoder` (`encode`), `denoiser` (`denoise`), `vaeDecoder` (`decode`), each one stateless function | textEncoder in `tokens` (int, rank 2; its static axis declared in `limits.staticSequence`); out `conditioning`. denoiser in `latents` (rank 4), `timestep` (rank 1), `conditioning` (`from: textEncoder.<conditioning>`); out `epsilon` (rank 4). vaeDecoder in `latents`; out `image` `[3, H, W]` on [-1, 1] (the decoder applies the VAE scaling factor). Blocks `tokenizer` (`renderer: clip-bpe`, a `dir` with `vocab.json` and `merges.txt`), `schedule` (`epsilonEuler`), `latent` (`spatial`); `guidance: none` | `residency` per graph; `latent.sizes` | sd-turbo (Forge export) |

The name is the family, not the checkpoint: any text-conditioned U-Net latent diffusion model with a CLIP tokenizer,
an epsilon Euler schedule and no classifier-free guidance fits it (SD 2.x-Turbo-like). SDXL's pooled and size
conditioning, CFG and v-prediction would be additions here, each a new rule the driver refuses until it implements
it. **Licensing:** SD-Turbo is under the Stability AI Community License: free for research, non-commercial use and
commercial use under USD 1M annual revenue with registration; distributions ship `LICENSE.md` and the NOTICE line,
and products display "Powered by Stability AI". The manifest records this in `license` and `x-license`.

### 5.5 `krea2@1` (normative since G3)

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `krea2@1` | `textEncoder`, `denoiser0`…`denoiser3` (all `function: segment`), `vaeDecoder` | textEncoder in `tokens`, `positions`; out `conditioning`. denoiser0 in `latents`, `conditioning` (`from: textEncoder.text_embeds`), `sigma`, `imageRotary`; out `carryImage`, `carryText`, `condition`. denoiser1–3 in `carryImage`, `carryText` (`from` the previous segment), `condition` (`from: denoiser0.cond`), `imageRotary`; denoiser1–2 out `carryImage`, `carryText`; denoiser3 out `velocity`. vaeDecoder in `latents`; out `image`. Blocks `schedule`, `latent`, `krea2`. | `residency` per graph | krea-2-turbo, krea-2-turbo-realism (Forge exports) |

Roles are bound once per step from host buffers or chained graph outputs (`source.from`), the §2.3 mechanism; the
driver reads the bindings once at load and refuses a chain that runs forward or a host-fed carry. Segment carries stay
float32; the latents the first segment and the decoder read are fp16 (rounded from the host's float32 latents). A
future `vision` graph and `encodeWithImages` function (reference images) and a VAE encoder (refine) are reserved for
`krea2@1` and absent from the first export. The driver is macOS-only in 0.1.0 (refused on iOS before loading).

### 5.6 `krea2` (normative since G3)

`krea2` is owned by `krea2@1`.

| field | req | meaning |
|---|---|---|
| `prompt` | R | `{systemPrefix, suffix, maxTokens: 546, keepTrailingTokens: 5, dropLeadingRows: 34, padding: none, positions: running}`: the rendered ids are never padded; Diffusers' padded block masks its padding as keys everywhere, so unpadded rows are exactly its real rows |
| `conditioning` | R | `{taps: [2,5,…,35], width: 2560, rows: "1...512"}`; `conditioning` is `[rows, 12·2560]`, tap-major; `rows` is `"lower...upper"` with upper = `maxTokens − dropLeadingRows`, and a prompt outside the range is refused |
| `rotary` | R | `{axes: [32,48,48], theta: 1000, positions: "(0, row, column)", layout: splitHalves, textRotated: false}`; the host builds `imageRotary [N, 2, 128]` |
| `segments` | R | the denoiser graph roles in call order |
| `condRows` | R | 8: six shared block-modulation rows and the final layer's (scale, shift) |
| `contentFilter` | R | `{required: true, reason?}`; `false` is rejected. The Krea 2 Community License §4.2 requires deployments to filter output: loading gives a `Krea2Model`, and a pipeline is only made with the app's `ContentFilter` (no default), which reviews the prompt before any work and the image before it is returned |

`tokenizer.renderer: "krea2-prompt"` names the fixed template above (no chat template); `addSpecialTokens: false`.

### 5.7 `flux2-klein@1` (normative since G2)

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `flux2-klein@1` | `textEncoder` (`encode`), `denoiser0`…`denoiser2` (`segment`; blocks 0–8, 8–16, 16–25 of 5 double + 20 single), `vaeDecoder` (`decode`) | textEncoder in `tokens`, `positions`, `tokenMask`; out `conditioning`. denoiser0 in `latents`, `conditioning` (`from: textEncoder.text_embeds`), `sigma`, `textRotary`, `imageRotary`; out `carryImage`, `carryText`, `condition`. denoiser1–2 in `carryImage`, `carryText` (`from` the previous segment), `condition` (`from: denoiser0.cond`), `textRotary`, `imageRotary`; denoiser1 out `carryImage`, `carryText`; denoiser2 out `velocity`. vaeDecoder in `latents`; out `image`. Blocks `schedule` (`shift.rule: empirical`), `latent` (128 channels), `flux2`. | `vaeEncoder` (`encode`: `referencePixels` → `referenceLatents`; an optional graph role, checked against its contract when present), reference images; `residency` per graph | flux2-klein-4b (Forge export, `Forge/flux2/`) |

The driver wires the segments from the manifest: every input chained with `source.from` must name the text encoder's
conditioning or an earlier segment's output, the others are host-fed (`latents`, `sigma`, the rotary tables), and only
the last segment emits `velocity`. Latents go in as the graph's scalar type (fp16 here), the Euler update stays float32
on the host, and the packed tokens in raster order are the decoder's `[rows, columns, 128]`. There is no default size:
a request names width and height. **macOS only in 0.1.0** (docs/INTEGRATION.md): on iOS the driver refuses with
`unsupportedDevice` before anything loads.

`flux2` is owned by `flux2-klein@1`. It is fully typed: the closed values below are the ones the driver implements,
and anything else is refused naming its field (`maxTokens` must equal `conditioning.rows`; `rotary.axes` has four
entries; the id rules are exactly the text and image rules shown).

| field | req | meaning |
|---|---|---|
| `prompt` | R | `{template: chat, thinking: off, addGenerationPrompt: true, maxTokens: 512, padding: right, padToken: 151643, truncation: keepPrefix, positions: running}`: one user message through the bundle's chat template (`tokenizer.chatTemplate`, `renderer: flux2-prompt`), right-padded with `<|endoftext|>` to exactly 512 ids plus a mask |
| `conditioning` | R | `{taps: [9, 18, 27], width: 7680, rows: 512, paddingRows: conditioning}`: all 512 rows, padding included, condition the denoiser |
| `rotary` | R | `{axes: [32, 32, 32, 32], theta: 2000, textIds: (0,0,0,i), imageIds: (0,row,column,0), referenceIds: (10(k+1),row,column,0), layout: splitHalves}`; the host builds `textRotary [512, 2, 128]` and `imageRotary [N, 2, 128]` |
| `segments`, `condRows` | R | the denoiser roles in call order; 17 modulation rows (image 6, text 6, single 3, output scale and shift) |
| `references` | O | `{encoder: vaeEncoder, kind: packedLatents, positionStride: 10, maxImages: 2, maxSide: 512, maxTotalTokens: 4096}`: reference tokens and rotary rows are appended to every denoiser call; their velocities are discarded; μ uses generated tokens only. **Reserved:** decoded loosely and not implemented by the driver yet |

Precision is recorded in `precision` (int8 per-channel weights on the text encoder and denoisers, the 20 single-block
`proj_out` fp16; both VAE graphs fp16 storage, fp32 compute). License `Apache-2.0`; the bundle ships the checkpoint's
`LICENSE.md` and a modification `NOTICE`. The exporter's parity gate (`check_flux2.py`) gates the real conditioning rows
and the image from the reference's conditioning; the whole pipeline from the graph's own rows is recorded, since the
padding rows move the image under any precision change.

### 5.8 `weights`: block weights as graph inputs (normative since G4 step 2a)

For a graph that serves many blocks with its weights lifted to runtime inputs (H3's body and conditioner layers). The
runtime type is `WeightTable` (AetherRuntime), read by `WeightBanks`.

| where | field | meaning |
|---|---|---|
| `graphs.<role>` | `weights` | `{table: "<asset>:<path>", inputs: [w0, …], blocks?}`: the graph inputs bound per block, the table file, and (a cross-check) its block count |
| table file | `inputs` | the weight inputs, in the order the exporter lifted them |
| table file | `blocks` | per block, each input's extent `{file: "<asset>:<path>", offset, bytes}` in a bundle file |

Each extent holds exactly its input's bytes (the graph's own descriptor, contiguous); a table naming a file outside the
bundle, or a block missing an input, is refused. The files are ordinary bundle assets, verified at install like any
other; the runtime does not re-hash them. A driver running such a graph declares a shape profile for each measured
canvas (DESIGN.md §6.6).

Decoding refuses (naming the field): an empty or repeated `inputs` entry, a weight input that a role also binds (weight
inputs come from the table only), a `table` outside the manifest's assets, and `blocks` < 1. The first producer is
`Forge/h3/export_h3.py`: one file per block (`weights/body/block-NN.bin`, `weights/conditioner/layer-NN.bin`), each
input's bytes contiguous in input order.

### 5.9 `kokoro@1` (normative since G5)

Status: **normative**, 2026-11-06. The producer is `Forge/kokoro/export_kokoro.py`; its `--sample-manifest` output is the
decoding fixture (`Tests/AetherCoreTests/Fixtures/kokoro/`). The driver is `AetherSpeech`'s `KokoroSpeechSession`
(`Registry.speech`); `Manifest.Kokoro` types the block below and refuses, naming the field, any rule it doesn't implement.

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `kokoro@1` | `durations` (`durations`), `decoder` (`decode`) | durations in `tokens` (int32 [1,T], T 3…512), `style` [1,256]; out `durations` (per-token duration sums [1,T], before speed and rounding), `prosody` [1,T,640], `text` [1,512,T]. decoder in `prosody` [1,640,F_b], `text` [1,512,F_b] (aligned on the host, zero-padded to the bucket F_b), `style`, `noise` [1,F_b,600,9], `frames` (int32 [1], the valid count); out `audio` [1,600·F_b], of which the first 600·frames samples are the utterance. Blocks `tables` (one voice per entry), `kokoro`. | — | kokoro-82m (Forge export, `Forge/kokoro/`) |

`kokoro` is owned by `kokoro@1`:

| field | req | meaning |
|---|---|---|
| `sampleRate`, `samplesPerFrame`, `harmonics` | R | 24000, 600, 9 |
| `phonemes` | R | `{vocabulary: {scalar: id}, boundaryToken: 0, maxPhonemes: 510, normalization: none, unknown: refuse}`: the input is a phoneme string in Kokoro's alphabet, read as Unicode scalars without normalization (nasal vowels are `a` + U+0303; `ç` is precomposed), wrapped in boundary tokens. Upstream silently drops unknown symbols; the driver refuses them, so a G2P mismatch shows. No grapheme-to-phoneme step is part of the bundle (see below) |
| `style` | R | `{dimension: 256, row: "phonemeCount - 1", decoder: style[0..<128], prosody: style[128..<256]}`: a voice table's row chosen by phoneme count, as upstream does |
| `durations` | R | `{frames: "max(1, roundHalfToEven(duration / speed))", alignment: repeat}`: per-token frames from the sums in float32 (`torch.round`), and the alignment as a gather (each token's features repeated for its frames) |
| `frames` | R | `{buckets: [64…4096], maxFrames: 4096, cap: {macos: 1024, ios: 512}}`: the smallest bucket that fits; the decoder specializes once per bucket. Metal memory per bucket grows faster than the bucket (0.0.x), so utterances above the platform cap are refused (or split by the caller) |
| `noise` | R | `{kind: splitmix64BoxMuller, shape: "[1, bucketFrames, 600, 9]", order: rowMajor, dtype: float32}`: the G0 stream for the request's seed, drawn for the whole bucket (prefix-stable, so the valid frames get an unpadded run's noise) |
| `voices` | R | `{id: {table, language, gender}}`: language from the id's first letter (`a` en-US, `b` en-GB, `e` es, `f` fr, `h` hi, `i` it, `j` ja, `p` pt-BR, `z` zh), gender from the second |

Each voice is a `tables` entry (`voices/<id>.f32`, float32 [510, 256], row-major little-endian). Placement: `durations`
on the GPU with `expectFrequentReshapes` (dynamic T), `decoder` on the GPU without it (bucketed shapes; 0.0.x saw the
hint abort there). `durations` was on the CPU (0.0.x's and Python's placement), but on macOS 27.2 Swift's Core AI runtime
traps loading it for the CPU (`loadFunction`: "Cannot create NDArray.Storage from unresolved NDArrayDescriptor"); on the
GPU both gates hold, frames exact. License `Apache-2.0`; the bundle ships the license and a modification `NOTICE`.

**Phonemes, not text.** Upstream's G2P (`misaki`) falls back to espeak-ng (GPL-3.0) for out-of-dictionary words, which
an Apache-2.0 SDK can't link, so the driver takes phonemes and the app supplies G2P (0.0.x's `SpeechPhonemizer`
protocol: text → utterances of ≤ 510 scalars). The exporter's gate (`check_kokoro.py`): token ids and frames exact;
waveform rel. L2 ≤ min(0.05, 4× the case's float32-vs-float64 upstream difference); log-mel mean ≤ 0.03 nats; repeats
bit-identical.

### 5.10 `minimax-h3@1` (normative since G4)

MiniMax H3 (FL2VA + the Turbo 8-step LoRA): text (and, reserved, keyframes) to video with audio. macOS only in 0.1.0.
Producer `Forge/h3/export_h3.py`; the driver is `AetherVideo`'s `MiniMaxH3Pipeline` (`Registry.video`).
`Manifest.H3` types the block below and refuses, naming the field, what the driver can't run; `keyframe`, `vision`
and `adapter` stay loose (FL2VA isn't presented yet).

| graph | inputs → outputs | weights | residency |
|---|---|---|---|
| `conditionerLayer` | hidden, cos, sin → hidden (one Qwen3-VL text decoder layer, causal) | `conditioner.json`, 11 inputs × 50 layers | perPrompt |
| `prepareText` | text → refined (context embedder + token refiner + LoRA) | `prepare-text.json`, 1 block | perPrompt |
| `preparePack` | video, audio, text (`from: prepareText.refined`), gather → hidden | — | perStep |
| `body` | hidden (`from: preparePack.hidden`), table [6, 3k, D], tags, cos, sin → hidden | `body.json`, 22 inputs × 50 blocks (base + dynamic LoRA) | perStep |
| `heads` | hidden (`from: body.hidden`), temb, indices, videoIndices, audioIndices → video, audio | — | perStep |
| `videoDecoder` | rows [448, 96] → pixels [1, 3, 22, 256, 256] (bf16 window tile) | — | perImage |
| `videoEncoderTile`, `videoEncoderTail<W>x<H>` | pixels → moments; tiles → condition (FL2VA keyframe) | — | perPrompt |
| `audioDecoder<Na>` | rows [2, Na, 32] → pcm [2, 800·Na] (fixed latent count) | — | perImage |
| `conditionerVision<W>x<H>` (optional) | pixels, posIndex, posWeight, patchRC → imageEmbeds, deepstack | `vision.json`, 1 block | perPrompt |

The `h3` block carries `profiles` (each a canvas and frame count with its rows, keyframe rows, audio latents and the
graphs it uses; the qualified one is 960×544×124), `limits` (rows 64…40960, cardinality 1…3, text tokens 2…4096),
`conditioner` (layers, tap 50, the embedding row source `weights/conditioner/embedding.bin` [151936, 5120] bf16, the
rotary tables), `recipes` (turbo8, turbo4: per task and step the timesteps, the temb file and the per-block AdaLN table
files `tables/<recipe>/<task>/step-<i>/block-NN.raw`), `adapter`, `video` (tile, window, packing and noise), `keyframe`,
`audio` and `platforms`. `license` is `LicenseRef-MiniMax-H3-Community-License`; the bundle ships `LICENSE` and the
required `NOTICE`.

Decoding refuses (naming the field): a profile whose key isn't `<W>x<H>x<frames>`, sides that aren't multiples of 32
over 256 px, frames off the 17n + 5 grid, `latentFrames` or `videoRows` that don't follow from them, an `audioDecoder`
or `keyframeEncoder` naming no graph; bad `limits`; a `tap` other than the last layer; rotary tables shorter than the
text-token limit; a `defaultRecipe` with no `t2va` recipe; an arm without one timestep a step and `steps + 1` sigmas
falling from 1 to 0, or timesteps other than `1 − σ` in float32; a step cardinality outside `limits.cardinality`,
non-ascending step timesteps, a `blocks` template without `{block:02d}`; video windows other than 256 px tiles of 7
latent frames at stride 5; audio other than stereo rows of 32.

`video.noise` is `{kind: splitmix64BoxMuller, order: sourceVideoThenAudio}`: one draw of Kit's stream, the source
`[24, T, H, W]` video tensor first (then packed `patch2x2ChannelMajor`), then the channel-major audio rows, as 0.0.x
drew it (the first export wrote `packedToken`, which named the wrong order; corrected in the exporter and the local
bundle's manifest at G4 step 2b). The driver's host recipe (layout `[text | audio | video]`, positions, row
timesteps, rotary, gather) is Diffusers' `build_packed_sequence`, `build_row_timesteps` and `MiniMaxH3RotaryPosEmbed`;
each step's modulation is padded to the recipe's largest cardinality so the body specializes once a render.
