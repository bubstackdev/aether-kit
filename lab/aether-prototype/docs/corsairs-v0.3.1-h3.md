# Corsairs 0.3.1 coordination: H3 quality investigation

No Corsairs wrapper defect has been confirmed. Aether found a concrete export bug: it passed raw per-head-interleaved QKV tensors directly to `convert_transformer_key`, which expects a loaded state dictionary with Q/K/V already reordered. The original shard streamer calls `reorder_interleaved_qkv` first. Aether's CPU checks repeated the omission. This is fixed in Aether; **no Corsairs 0.3.1 wrapper change is requested for this finding**. [Correction record](evidence/h3-qkv-correction.json), [current CPU-qualified exports](evidence/h3-qkv-ready.json).

The corrected native gate and complete H3 Turbo8 render now pass on **published Corsairs 0.3.0** in **708.33 seconds including the gate**. [Retained clip](evidence/h3-qkv-render/clip.mp4), [shared-window/media audit](evidence/h3-qkv-render-media-audit.json). Sampled frames show a usable crafted-bird visual pilot; audio signal checks pass, with subjective listening unassessed. The earlier 701.01-second gray-texture render remains rejected evidence from incorrect weights. GPU work is stopped. [Result and evidence](h3-turbo.md).

## Export finding and qualification handoff

The fix invokes the two original conversion operations in order, replaces 150 body QKV bindings and rebuilds both text-refiner projections. The VAE converter already handles its own QKV normalization. No lab, checkpoint or Corsairs files changed. Rust rejects old H3 manifests without the canonical-layout marker.

The CPU reproducer is `AETHER_H3_LAB=/Users/zakkeown/Code/atelier/lab/minimax-h3-lab /Users/zakkeown/Code/aether-engine/forge/.venv/bin/python -B forge/test_h3_weights.py`. Distinct row sentinels prove the expected Q0/K0/V0/Q1/K1/V1 raw layout independently of neural output. Both body and refiner mappings are covered. [Canonical audit](evidence/h3-qkv-weight-audit.json) checks all 1,100 exported bindings against raw shards after the required normalization.

For 0.3.1 qualification, old H3 tests remain useful for runtime byte/API parity but must not be cited as validation of the trained model's conversion or usable output. Use the corrected Aether fixtures when checkpoint semantics matter. Actual runtime mismatches will still get a separate minimal reproducer and candidate-release regression.

The runtime under investigation is the published crates.io Corsairs **0.3.0** archive, SHA-256 `e7f23287a680009a43aa5a46e8646a66918f5c4c3e2bdb627a18091b21fe3e3a`. No Corsairs checkout or lab files have been changed. Aether's application remains safe Rust.

## Completed CPU isolation

[Legacy CPU references](evidence/h3-turbo-phase-cpu.json) call the pinned upstream modules with the old conversion. All eight native-produced time embeddings and 400 modulation tables are within the predefined 0.02 relative-L2 limit; worst error is 0.000239603. These invariant functions contain no QKV and are unaffected by the identified bug. Corrected [phase references](evidence/h3-qkv-phase-cpu.json) have now been regenerated with canonical body/refiner weights.

The attention oracle computes 64 query rows spread across text, audio and video while retaining all 19,333 keys/values. Hooks select only query outputs and the subsequent row-wise projection/FFN work. Before preparing these references, the method matched an unsliced original 128-row block byte for byte on 32 selected rows. This is a sampled-query numerical oracle with complete production context; it is not a full CPU trajectory comparison. Original outer input preparation and both timestep cardinalities of the output heads have complete-output CPU references.

[Changing-binding CPU check](evidence/h3-turbo-binding-cpu.json) verifies the original zero-projection residual block is exactly the identity on 128 rows. Its production-size case uses zero base/adapter bindings, then restores the real bindings through the same function.

## Bounded native discriminator

The [frozen build](evidence/h3-turbo-phase-build.json) and [exact command](evidence/h3-turbo-phase-ready.json) define a five-minute probe with nine cases and three independent execution routes:

- Owned tensors through `Function::run`.
- Metal inputs with automatically allocated outputs.
- Metal inputs with caller-bound output buffers.

Each route reuses functions with matching source hash/entry point; fresh functions separate the routes. The sequence exercises timestep cardinality 1→2, full-context attention, and real→zero→real weight bindings. Full outputs must remain finite; numerical comparisons use the original CPU references and a predeclared 0.02 relative-L2 limit. Each completed observation is written before proceeding.

The [corrected probe](evidence/h3-turbo-phase-corrected-report.json) passed with a [joined exit 0](evidence/h3-turbo-phase-corrected-controller.json) after 20.80 seconds. Twenty-five supported calls and 31 output comparisons pass the predefined limit; worst relative L2 is 0.00384147. Output hashes match across all tested routes. Two caller-bound dynamic time outputs are skipped because the public API explicitly requires fixed descriptors. The initial probe attempted that unsupported combination, received the documented error and was corrected; this was a probe error, not the production quality failure. [Native capture](evidence/h3-turbo-phase-corrected-capture.json), Evidence `ev_ac644063c2ddb83ad6680def3885b7d55c8f9b85b1d144e14e497ab9b8d979ff`.

That discriminator used the old QKV weights and establishes route parity only for its specific tensors. The [canonical gate](evidence/h3-qkv-phase-summary.json) now passes in 22.25 seconds: 25 supported calls, 31 output comparisons, worst relative L2 0.0110056 against the same 0.02 limit, and identical hashes across all tested routes. Its [report](evidence/h3-qkv-phase-report.json) and [joined controller](evidence/h3-qkv-phase-controller.json) use the [frozen corrected artifacts](evidence/h3-qkv-ready.json). The subsequent complete render exits zero and joins, finishing the shared window in 708.33 seconds. No path-specific Corsairs mismatch was found. GPU work is stopped.

## Fix handoff criterion

A path-specific mismatch will be reduced to the smallest model/input sequence that preserves it. The handoff will include the pinned executable/model/input hashes, CPU expected tensors, failing API and output hashes, exact launch command and regression assertion. A 0.3.1 candidate must first pass that reproducer, then the affected H3 phase checks. A fresh full render remains necessary before accepting quality. A framework-wide or export/source mismatch must be distinguished from a Corsairs wrapper failure before assigning the fix.
