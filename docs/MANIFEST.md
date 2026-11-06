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
- `ImagePipeline` (§8) is a shared primitive configured by `preprocess.image`. It is not a driver. There is no
  `image-pipeline@1`.
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
| `labels` | O | `{set: "coco-91"\|…}` or `{table}` | COCO-91 is hardcoded in coreai-kit and absent from every bundle. The adapter names a built-in set; our own exports ship a file. |
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

`image` (O):
`{resize: {mode: stretch|letterbox|centerCrop|fitLongSide, size?, anchor: center|topLeft, padValue, filter: vImage|coreGraphicsHigh|pillowBicubic|pillowBilinear|torchUint8Bicubic},
 colorSpace: sRGB|deviceRGB, channelOrder: RGB|BGR, scale (1 | 1/255), mean, std,
 layout: NCHW|NVCHW|patches, patchify?: {patch, temporal, merge, rowOrder: blockMajor|rowMajor, withinPatch: "C,T,py,px"|"py,px,C"},
 hostDType, orientation: exif, tiling?: {tile, overlap, blend: feather, maxInputSide}, crop?: {tiles, thumbnail}}`.

Each field has a bundle that needs it: BGR, 0–255 and letterbox top-left pad 114 (yolox); `[-1,1]` (adcsr: scale
1/255, mean 0.5, std 0.5); NVCHW (colmodernvbert, DA3 multi-view); patchify, which fails silently when wrong (qwen3-vl
block-major C,T,py,px; lfm channel-last row-major); fp16 graph fed from fp32 host (DA3); tiling with overlap
`max(4, tile/8)` (adcsr); the white letterbox pad (glm-ocr); Pillow vs CoreGraphics filters (decision towers).
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

`{kind: denseNms|detrSigmoid|depth|superRes|embedding|maxSim|gliner|yesNo, boxFormat: cxcywh|xyxy, boxSpace: letterboxPixels|normalized,
 activation: applied|sigmoid|softmax, nms?: {type: perClass, iou}, defaults: {scoreThreshold, maxDetections},
 classIndex: contiguous|cocoId, masks?: {output, stride},
 depth?: {polarity: largerIsNearer, resizeBack}, sr?: {range: [-1,1], colorMatch: host|none},
 embedding?: {normalized: inGraph, pooling: inGraph|mean|cls, similarity: cosine|dot},
 retrieval?: {imageTokenSpan: [13,77], patchGrid: [8,8]}, probs?: {positiveIndex: 1, threshold: 0.5}}`.

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
2026-11-06; `krea2@1` and its `krea2` block (§5.4–§5.5) stay **proposed until G3**, and a v1 loader treats them as an
uncatalogued driver (`unknownDriver`) until `krea2@1` is registered. The first producer is
`Forge/krea2/export_krea2.py`; its `--sample-manifest` output is the decoding fixture
(`Tests/AetherCoreTests/Fixtures/krea2/`). Unlike the loose driver blocks of §2.5–§2.8, these blocks are fully typed:
the string-valued rules below are closed vocabularies, and anything else is rejected naming its field.

### 5.1 `schedule`

Any flow-matching or diffusion driver. Implemented by `FlowMatchSchedule` (AetherRuntime).

| field | req | meaning |
|---|---|---|
| `kind` | R | `flowMatchEuler` (Krea, FLUX.2). Other solvers (SD-Turbo's `epsilonEuler`, H3's) join with their drivers |
| `steps`, `maxSteps` | R, O | default and maximum step counts (`maxSteps` ≥ `steps`; absent: `steps`) |
| `trainTimesteps` | R | ≥ 1 (1000) |
| `sigmas.base` | R | `"linspace(1, 1/steps, steps)"`: in Double, each value rounded to Float, the last exactly `1/steps` |
| `sigmas.shift` | R | `{rule: exponential, mu}` (Krea: μ 1.15, `is_distilled`) · `{rule: empirical, empirical: {lowSteps, lowSlope, lowIntercept, highSteps, highSlope, highIntercept, tokenThreshold}}` (FLUX.2: μ from the generated token count and the steps) · `{rule: none}`. The shifted σ is `e / (e + (1/σ − 1))` in Float, `e = Float(exp(μ))` |
| `sigmas.terminal` | R | 0, appended after the last step (the only defined value) |
| `timestep` | R | `"sigma * trainTimesteps / trainTimesteps"`: what the denoiser's `sigma` input receives, in Float |
| `update` | R | `"x + (sigma[i+1] - sigma[i]) * velocity"`: the host update in Float, the product rounded before the add |
| `guidance` | R | `{kind: none}` (no `scale`; Krea 2 Turbo is distilled) or `{kind: cfg, scale > 0}` |

### 5.2 `latent`

The image-latent contract.

| field | req | meaning |
|---|---|---|
| `channels` | R | per-token channels the denoiser sees (Krea 64 = 16 VAE channels × 2 × 2); a multiple of 4 for 2×2 packing |
| `pixelsPerToken` | R | 16 (VAE 8× × patch 2); also the multiple width and height must be |
| `packing` | R | `patch2x2ChannelMajor`: packed channel `c·4 + 2·dy + dx` |
| `layout`, `decoderLayout` | R | descriptive: denoiser `tokens[rows*columns, 64]`, decoder `[rows, columns, 64]` |
| `noise` | R | `{kind: splitmix64BoxMuller, order: packedToken, dtype: float32}`: `SeededGaussianNoise`, prefix-stable and bit-identical to 0.0.x's stream |
| `maxImageTokens` | O | the export's bound on `rows · columns` (4096) |

### 5.3 `residency` and other field additions

| where | field | meaning |
|---|---|---|
| `graphs.<role>` | `residency` | `perPrompt` (text encoder: load, encode, releasable), `perStep` (denoisers: resident through the steps), `perImage` (decoder). Absent: resident for the model's lifetime. `ResidencyPlan` keeps every graph resident when the measured peak fits, else releases each phase's graphs at its end (only when every graph has a residency) |
| top level | `x-status`, `x-license` | draft marker; license files, `modified: true`, revenue cap, adapter permissions |
| `provenance` | `mergedAdapter` | `{file, sha256, format, merge, multiplier, scale}` for an adapter merged before export (realism) |
| `provenance.refs` | `diffusersCommit`, `forgeLockSHA256`, `exportReport` | the exporter's environment pins |
| `license` | `LicenseRef-Krea-2-Community-License-1.0` | an SPDX `LicenseRef`; the bundle ships `LICENSE.pdf`, the text, the use policy and `NOTICE` |

### 5.4 Driver catalog addition (proposed until G3)

| driver | graphs | required roles / blocks | optional | bundles |
|---|---|---|---|---|
| `krea2@1` | `textEncoder`, `denoiser0`…`denoiser3` (all `function: segment`), `vaeDecoder` | textEncoder in `tokens`, `positions`; out `conditioning`. denoiser0 in `latents`, `conditioning` (`from: textEncoder.text_embeds`), `sigma`, `imageRotary`; out `carryImage`, `carryText`, `condition`. denoiser1–3 in `carryImage`, `carryText` (`from` the previous segment), `condition` (`from: denoiser0.cond`), `imageRotary`; denoiser1–2 out `carryImage`, `carryText`; denoiser3 out `velocity`. vaeDecoder in `latents`; out `image`. Blocks `schedule`, `latent`, `krea2`. | `residency` per graph | krea-2-turbo, krea-2-turbo-realism (Forge exports) |

Roles are bound once per step from host buffers or chained graph outputs (`source.from`), the §2.3 mechanism. Segment
carries stay float32. A future `vision` graph and `encodeWithImages` function (reference images) and a VAE encoder
(refine) are reserved for `krea2@1` and absent from the first export.

### 5.5 `krea2` (proposed until G3)

`krea2` is owned by `krea2@1`.

| field | req | meaning |
|---|---|---|
| `prompt` | R | `{systemPrefix, suffix, maxTokens: 546, keepTrailingTokens: 5, dropLeadingRows: 34, padding: none, positions: running}`: the rendered ids are never padded; Diffusers' padded block masks its padding as keys everywhere, so unpadded rows are exactly its real rows |
| `conditioning` | R | `{taps: [2,5,…,35], width: 2560, rows: "1...512"}`; `conditioning` is `[rows, 12·2560]`, tap-major |
| `rotary` | R | `{axes: [32,48,48], theta: 1000, positions: "(0, row, column)", layout: splitHalves, textRotated: false}`; the host builds `imageRotary [N, 2, 128]` |
| `segments` | R | the denoiser graph roles in call order |
| `condRows` | R | 8: six shared block-modulation rows and the final layer's (scale, shift) |
| `contentFilter` | R | `{required: true}`. The Krea 2 Community License §4.2 requires deployments to filter output; the driver takes an explicit `ContentFilter` from the app and has no default |

`tokenizer.renderer: "krea2-prompt"` names the fixed template above (no chat template); `addSpecialTokens: false`.
