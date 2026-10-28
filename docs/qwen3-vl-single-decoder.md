# Qwen3-VL: one decoder function (0.0.3 design)

Status: implemented and qualified, 2026-10-27. This covers two 0.0.3 memory items from the
[roadmap](roadmap-0.0.3.md): the decoder's weights resident twice, and the 4B text export's memory. The design
sections below were written before the re-export, from a toy model (small CPU exports and short GPU probes);
the outcome comes first.

## Outcome

Both Qwen3-VL bundles were re-exported with the contract proposed below and qualified on the real models
([Qwen3-VL guide](qwen3-vl.md#one-decoder-function-003);
[evidence](evidence/qwen3-vl-one-function-2026-10-27.json)):
- **Wired memory.** The 4B decoder wires 8,414 MiB of mapped-file pages at its first request and nothing more
  at the first image request (two functions: 16,086 MiB). Its compiled cache is 15.0 GiB (22.5 GiB), and the
  budget test's warm level is 10,847 MiB (19,566 MiB). The 2B: 6,065 MiB warm (10,113 MiB). Metal after all five
  grids' image requests is 15,344 MiB (15.0 GiB) on the 4B and 9,842 MiB (9.6 GiB) on the 2B, against the
  [expected](#expected-saving) "about 14" and "about 9" GiB.
- **Parity.** Every 0.0.2 gate passes again on the same references, and the 64-token replies of the old and new
  bundles are identical, token for token, for a text and an image prompt on both models.
- **Prefill: one change to the design.** Sending each image as one submission (as proposed below) made the
  function change to the image's width on every image request, and Core AI retained about 360 MiB for it on the
  4B: five budget reloads in 60 alternating requests, against one for the two-function bundle, and the 2B
  missed its bound by 2 MiB. Aligned rows do not need that: Kit now prefills the whole prompt in the fixed
  64-token chunks, each carrying its own image tokens' rows. An image request then retains about 103 MiB and the
  4B reloads twice in 60 requests (the first after request 20, against 34). The `n`-gather fallback would not
  have helped: it needs the same image-width submissions.
- **Timing.** Text time to first token and decode rate are unchanged within host noise. Image requests are
  slower to the first token, the cost of the chunked image prefill: on the 4B 0.35 to 0.45 s at the median and
  0.29 to 0.37 s at the minimum; on the 2B the minimum rises from 0.19 to 0.24 s.
- **Export.** The bounded pipeline below exported the 4B text tower in 109 s at a peak of 16.3 GB RSS (0.0.2:
  31.7 GB, 75 s) and the 2B in 46 s at 7.1 GB, self-checks included.

## Today: two functions over one copy of the weights on disk

`Forge/language/export_qwen3_vl_text.py` converts the text tower twice and saves one asset with two entry points. The
asset stores the weights once (7.5 GiB for the 4B: the converter deduplicates identical blobs).

| | `main` (text prefill and decode) | `withImage` (one image's tokens) |
| --- | --- | --- |
| Token input | `input_ids` [1, q] | `input_ids` [1, q] |
| Positions | `cache_positions` [1, q], `rope_positions` [3, q] | the same |
| Image input | none | `image_embeds` [n, hidden], `deepstack` [levels, n, hidden] |
| Caches | `keyCache`, `valueCache` [layers, 1, kv heads, capacity, head dim], read-only | the same |
| Outputs | `logits` [1, 1, vocab], `keyUpdate`, `valueUpdate` [layers, 1, kv heads, q, head dim] | the same |
| Dynamic dimensions | q 1–512, capacity | q 2–2,048, n 1–2,048, capacity |
| State | none | none |

Both take token IDs; neither takes embeddings. The graphs are the same decoder (embedding lookup, 36 blocks on the
4B, tied head) except that `withImage` replaces the rows whose token is `<|image_pad|>` with the next row of
`image_embeds` (a running count over the pad mask, gathered with an embedding lookup), and adds `deepstack[k]` at
those rows after block k for k < levels (three). M-RoPE positions are host-computed for both, so the rotary code is
identical. Kit never mixes the two in one submission: an image's tokens are exactly one `withImage` submission
(q = n, every token a pad token), and text runs, including the text around an image and every decode step, go
through `main`. So a one-image request uses both functions.

## What Core AI does with two entry points

A toy decoder with the same structure (embedding table tied to the head, 8 blocks of two fp16 matrices, the same
splice and DeepStack adds; 144 MiB of weights) was exported in six layouts and run on the GPU with the language
runtime's options (GPU preferred, `expectFrequentReshapes`), one process per layout
(`Forge/language/probe_function_weights.py`, `Scripts/probe-function-weights.swift`;
[evidence](evidence/core-ai-function-weight-copies-2026-10-27.json)). "Wired" is the process's wired mapped-file pages
from `footprint`, measured after each function's first use; Metal's allocated size moved with it.

| Layout | Compiled cache (`resources.bin`) | Wired after first function | Wired after second function |
| --- | --- | --- | --- |
| `main` only | 288 MiB | 160 MiB | – |
| `withImage` only (text steps sent with one zero image row) | 288 MiB | 160 MiB | – |
| `main` + `withImage` (today's layout) | 432 MiB | 160 MiB | 304 MiB |
| `main` + `main2`, the identical graph under two names | 432 MiB | 160 MiB | 304 MiB |
| `embed` (lookup only) + `main` taking embeddings | 288 MiB | 16 MiB | 160 MiB |
| `main` with aligned image rows (the proposed contract, below) | 288 MiB | 160 MiB | – |

So, in this Core AI (macOS 27.0.1, build 26A434; exported with coreai-torch 0.4.2 and coreai-core 1.0.0b2):

- Each entry point that runs wires its own copy of every weight it uses. Even two identical graphs are not shared.
  `SpecializationOptions` and `AIModel` offer no option to share weights between functions.
- The compiled cache holds one copy of the weights per decoder entry point plus one more: exactly 2× the weights for
  one function, 3× for two. The 4B decoder's 22.5 GiB `resources.bin` is 3 × 7.5 GiB.
- A separate embedding function wires its own copy of the table, even though `main` also holds it as the tied head.
- One function with the image inputs serves text prefill, image prefill and 1-token decode steps; the toy accepted
  q = 1 with q's lower bound exported as 1.

Krea's text encoder has the same two-function layout (`encode` + `encodeWithImages`) and a 20.5 GiB compiled cache,
so it probably wires two copies too when references are used. That asset is outside this item.

## Proposed contract: one `main` that always takes image rows

```
main(input_ids [1, q], cache_positions [1, q], rope_positions [3, q],
     image_embeds [q, hidden], deepstack [levels, q, hidden], keyCache, valueCache)
  -> logits [1, 1, vocab], keyUpdate, valueUpdate
```

- One dynamic token dimension, q from 1 to the image chunk bound (2,048), plus capacity. `n` goes away: image rows
  are aligned with the submission, row i belonging to token i. Rows are used only where `input_ids` is
  `<|image_pad|>` (the splice and all three DeepStack adds are `where` on that mask), so other rows are ignored.
- Kit's image submissions already have q = n with every token a pad token, so their inputs do not change. Text
  submissions and decode steps pass zero rows: zero-copy views of one Kit-owned zero buffer, sized for the largest
  text submission (512 × 2,560 × 4 fp16 values, 10 MiB on the 4B).
- The bucketed contract is unchanged: cache positions, read-only caches, update rows, last-row logits, the fixed
  prefill width with its step-back tail, 256-row cache growth. The function's shape set is today's two shape sets
  together: (C, 1, short runs, 432, 448, 484) × capacity.
- The embedding table stays in the graph, once, shared by the lookup and the tied head.

Kit changes: `LanguageStepContract` recognises a `main` with the image inputs; such a session uses `main` for every
step and loads no `withImage`; the reload path loads one function. Two-function bundles keep working through today's
path, so bundles already installed (Atelier's 4B describer) are unaffected until they are re-exported and
re-registered. An older Kit refuses the new asset (its `main` signature is not a recognised contract), so it fails
closed. The registration script and the export report record the new recipe.

### Alternatives considered

- **Embeddings in, lookup elsewhere** (`main(inputs_embeds, deepstack, …)`). A separate lookup function wires the
  table twice (+0.7 GiB on the 4B; the toy's `embed` row), and a host lookup needs Kit to carry the table outside the
  asset. It also changes the decoder's inputs more than needed. The only gain, no token-derived mask in the graph, is
  small.
- **Keep the `n` gather and send one zero row for text** (the toy's `withImage`-only layout). It works, and it retained
  less per request on the toy (below), but it keeps a second dynamic dimension that is 1 on every text step. Dynamic
  shapes have already produced wrong GPU results in this project (the encoder's dynamic control, engine F-P3-26), and
  a size-1 dimension is the case most likely to be treated differently. It stays the fallback if the aligned rows
  retain materially more on the real decoders: one exporter flag.
- **No re-export: route everything through today's `withImage`.** Its q starts at 2, so 1-token decode steps cannot
  run without stepping back a row each step. Rejected.
- **Cross-function weight sharing in Core AI.** Not available (above). Worth an Apple Feedback, but not something
  to wait for.

### Costs

- **Decode step.** Two more inputs per step (zero-copy views) and four `where` selections on [1, q, hidden]. Expected
  well under 1 % of a step; to be measured (below).
- **Embedding residency.** Unchanged: one copy, in the graph.
- **Shape-change retention.** The decoder is specialized with `expectFrequentReshapes`, and Core AI retains memory on
  each shape change of a function ([native memory retention](language-generation.md#native-memory-retention)). Today
  an image request changes `main` between C and 1 and calls `withImage` separately; with one function, the image
  span is one more shape change inside the same function, and with aligned rows the image inputs change shape on
  every text-width change too. On the toy (a 64-token text chunk, a 48-token image span, an 8-token run and four
  decode steps per request; two processes each, identical results), Metal grew 0.9 MiB per warm request with two
  functions, 1.9 MiB with one function and a one-row image input, and 2.9 MiB with aligned rows: about 1 MiB more for
  each input that changes shape. Those are small next to the 4B writer's measured 58.6 MiB per text request at
  C = 64, but the toy says nothing about how a change to a 484-token width inside the function scales on the real
  decoder. More retention means more frequent budget reloads (each re-wires one copy, about 8 GiB, at idle), not
  wrong output; it must be measured.
- **Shape bounds.** q's upper bound rises from 512 to 2,048 on the text path. Kit still limits text chunks with
  `maximumPrefillChunk`; whether a wider bound changes Core AI's planning is part of the timing check.

## Expected saving

| | VL-4B today | VL-4B, one function | VL-2B today | VL-2B, one function |
| --- | --- | --- | --- | --- |
| Host wired memory added by the decoder after a one-image request | 16.3 GiB (8,598 + 8,132 MiB of first-use host-wired deltas) | about 8.4 GiB | not measured (Metal 8.1 GiB) | about 4 GiB |
| Compiled cache, read into the page cache at open | 22.5 GiB | about 15 GiB | about 9 GiB | about 6 GiB |
| Metal allocated, all five grids used | 22.4 GiB | about 14 GiB | 12.9 GiB | about 9 GiB |

The one-function figures are projections from the toy's rule and today's per-function measurements
([evidence](evidence/qwen3-vl-memory-by-phase-2026-10-27.json)): one function's first use raised host wired memory by 8.4 GiB on the 4B (8,598 MiB; the
process's own mapped-file pages were 8,414 MiB, 8.2 GiB, which is also what the one-function export wires).

## Export memory

The roadmap and [qwen3-vl.md](qwen3-vl.md#limits-and-follow-ups) attribute the 4B export's 33–35 GB to weights held
per ExportedProgram. The toy does not show that. Scaled to 2.03 GiB of weights and measured by stage
(`probe_function_weights.py export-memory`: RSS, its sampled peak within the stage, and `phys_footprint`), converting
two programs over the same weights costs the same as converting one:

| Stage | RSS, two programs | RSS, one program | Footprint (either) |
| --- | --- | --- | --- |
| Eager fp16 module | 2.3 GiB | 2.3 GiB | 2.2 GiB |
| `torch.export` and decompositions | 2.4 GiB | 2.4 GiB | 2.4 GiB |
| `to_coreai` and `optimize` | 4.5 GiB | 4.5 GiB | 2.4 GiB |
| `save_asset` | 6.5 GiB | 6.5 GiB | 4.4 GiB |
| cpu_only load and call (the self-check) | 8.6 GiB, peak 10.6 GiB | 8.5 GiB, peak 10.6 GiB | 4.4 GiB |

The converter adds one copy of the weights however many programs share them (resident but outside the footprint),
saving adds an anonymous copy, and the cpu_only load adds another copy, peaking at two while it compiles. Freeing
objects inside the process does not give the anonymous memory back: deleting the eager module before saving
(`--free-eager`) lowered the footprint by 0.1 GiB, not 2. At the self-check that is about five copies of the weights. The real exporter
adds two things: it builds the text tower in float32 before `.half()` (about 16 GB for the 4B), and it keeps the
float32 vision tower for the self-check. Five copies of the 7.5 GiB tower would be 39 GiB; the observed 33–35 GB fits
that with some clean pages evicted. A single function will not reduce it.

Plan to bound it:
1. Build the text tower directly in fp16 (construct on the meta device and load into fp16 parameters): the float32
   build peak goes away.
2. Compute the eager reference first: the vision tower's rows for the reference image, then the eager text tower's
   logits and cache rows for the self-check plan, saved to the output directory; free the vision tower.
3. Export, convert and save, then exit. Run the cpu_only self-check in a new process that loads only the asset and
   the saved reference.

Estimated peaks for the 4B: about three copies, 23–25 GiB RSS (about 16 GiB footprint), in the export process; about
two copies while the self-check process compiles, roughly 15 GiB RSS of mostly file-backed pages. Today: 35 GB.
Loading the fp16 weights by memory-mapping a one-time fp16 copy of the checkpoint would make the eager copy file-backed
too; it is not needed to stay near 25 GB.

## Export plan

Each model is exported once with the new contract and the bounded pipeline, into a new directory on durable storage,
then registered with `register_qwen3_vl_bundle.py`:
1. Qwen3-VL-2B (3.2 GB of weights): estimated peak about 10 GiB RSS, about 10–15 minutes including the self-check.
   It checks the pipeline's stage memory against the estimate before the larger export.
2. Qwen3-VL-4B describer (8.05 GB): estimated peak 23–25 GiB RSS, about 20–30 minutes. This is a heavy CPU job and
   needs the lead's clearance.

Durations are estimates: no earlier export recorded its wall time.

## Evidence for the change

Export (cpu_only self-check, in the report): an image prefill, forced decode steps, and a text-only prefill, all
through the one function, against the eager tower: cosine ≥ 0.99, argmax equal, no lost cache rows. Today's check
covers text only through decode steps.

Native, both VL-2B and VL-4B, under the existing gates:
- `qwen3VLImageTextQualification`: model parity against the existing references (VL-2B grids; VL-4B Kit-decoded and
  PIL-decoded), Kit end to end on the same decode, and the public session; forced-row cosine ≥ 0.995 and greedy
  identical to the reference end, as in the 0.0.2 records.
- Text-only parity at the fixed prefill width against the upstream text references, and
  `languageOverlappedPrefillMatchesOtherPlans` on text and image prompts.
- Residency: `visionLanguageMemoryByPhase` with residency, expecting one wired step at the first text request and none
  at the first image request, the compiled cache's size, and the host free-memory drop at open.
- Retention: `languageMemoryBudgetBoundsRepeatedRequests` (60 requests alternating text and image), compared with the
  2026-10-27 record: per-request growth, reload count and the request of the first reload.
- Timing: time to first token and decode rate for a text and an image request, old bundle against new, same process
  pattern, on a quiet host if possible.

## Risks

- Retention per image request may rise with one function, so the 4B would reload more often (each reload re-wires
  8 GiB). Correctness is unaffected; latency is.
- The q bound widening on the text path, and `where` on a token mask in every text step, are new to the decoder's GPU
  path. The parity gates cover correctness; the timing check covers cost.
- Re-exporting both bundles is two heavy CPU jobs, and Atelier keeps the old bundle until it installs the new one.
- The compiled-cache and wiring rule is measured on one Core AI build. A later build might share weights across
  functions, which would make the single function unnecessary but not wrong.
