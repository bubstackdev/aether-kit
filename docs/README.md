# Engineering documentation

Research date: 2026-10-17. These records distinguish observed behavior, proposed decisions, and work still needed. They are repository engineering documents, not a promise of implemented features.

1. [Product and architecture](architecture.md) — scope, boundaries, proposed API semantics, decision register.
2. [Prototype evidence](prototype-evidence.md) — what is already demonstrated and what needs requalification.
3. [Research](research.md) — pinned upstream sources, findings, implications, and limitations.
4. [Qualification and release](qualification-and-release.md) — experiments, acceptance gates, and publishing sequence.
   - [v0.0.1 release checklist](release-0.0.1-checklist.md) — scope, requalification, per-product readiness, documentation and tagging steps for the experimental preview.
5. [Implementation status](implementation-status.md) — scaffold, executed checks, limitations and next work.
6. [Krea conditioning slice](krea-conditioning.md) — bundle identity, tokenizer integration, conditioning and retained fixture checks.
7. [Krea generation slice](krea-generation.md) — complete reference orchestration, shape registration, decoding, PNG delivery and parity gate.
8. [Native validation evidence](evidence/krea-native-validation-2026-10-17.json) — all 33 tests and both real-weight tower parity gates passed; shared-host correctness checks, with benchmarking pending a quiet window.
9. [Native session qualification](evidence/krea-native-session-validation-2026-10-17.json) — expanded 34-test suite and 23 native checks passed for model reuse, cancellation/recovery, active unload and two dynamic shapes.
10. [Model cache migration](evidence/model-cache-migration-2026-10-17.json) — model exports moved into `~/.cache/huggingface/aether-kit`, with verified file hashes and compatibility symlinks.
11. [Hugging Face cache validation](evidence/huggingface-cache-validation-2026-10-17.json) — all 34 tests and 23 native checks passed from cache paths; seven outputs match their pre-move hashes exactly.
12. [Dynamic parity and cancellation qualification](evidence/krea-dynamic-qualification-2026-10-17.json) — both additional shapes match all eight independent step/pixel hashes; encoder recovery and scoped native decoder recovery pass. Historical result: the [focused GPU retry](evidence/krea-gpu-retry-2026-10-17-status.json) reached eight-step decoder cancellation but left recovery unqualified.
13. [Function-reuse qualification](evidence/krea-function-reuse-qualification-2026-10-18.json) — lazy per-session native functions pass the full independent dynamic parity, cancellation and eight-step recovery suite; live journals preserve the resource-lifetime comparison.
14. [Krea benchmark workflow](krea-benchmarks.md) — fixed workloads, explicit prepare/run modes, timing and memory scope, and [CPU/preparation readiness](evidence/krea-benchmark-readiness-2026-10-18.json). The [first user-run baseline](evidence/krea-benchmark-baseline-2026-10-18.json) passed correctness; timing drift remains documented.
15. [Benchmark comparison readiness](evidence/krea-benchmark-comparison-readiness-2026-10-18.json) — matched encoded/provided plans, one-time validated conditioning with encoder unload, profile-labelled iterations and CPU/prepare-only validation. The subsequent [native comparison](evidence/krea-benchmark-comparison-2026-10-18.json) passes all 32 independent output checks and records repeat-generation timing, separate setup costs and measurement limitations.

16. [Prepared-prompt workflow](krea-generation.md#prepare-once-render-multiple-seeds) — explicit conditioning ownership, bundle identity and model reuse, with [54 passing CPU tests](evidence/krea-prepared-prompt-readiness-2026-10-18.json). The subsequent [native qualification](evidence/krea-prepared-prompt-qualification-2026-10-18.json) passes 65 checks, exact prepared repeats at both dynamic sizes, cancellation recovery and native model/function lifecycle checks.

17. [Batch-render CLI](krea-generation.md#batch-rendering-from-the-cli) — encode once, render explicit seeds serially, save indexed PNGs and atomic reports. [Validation](evidence/krea-batch-render-validation-2026-10-18.json) records 60 passing CPU tests and three native images with reference parity, repeatability and output protection checks.

18. [MiniMax H3 integration](h3-integration.md) — CPU geometry, packing, captured Turbo8 scheduler and corrected export identity, with [69 passing CPU tests](evidence/h3-cpu-readiness-2026-10-18.json). This records the initial CPU checkpoint; later native block gates follow below.

19. [H3 native binding checkpoint](h3-integration.md#native-block-binding-checkpoint) — BF16 storage, corrected block/fixture validation and owned Metal binding, with [CPU/prepare-only readiness](evidence/h3-binding-readiness-2026-10-18.json). The [first GPU block qualification](evidence/h3-block-qualification-2026-10-18.json) now passes four calls, sampled CPU numerical gates and exact full-output repeats. The subsequent lifecycle gate follows below.

Update the decision register when an experiment changes a recommendation. Each performance result must identify the model assets, execution recipe, hardware, software, and measurement conditions. Preserve failed and superseded results with their scope.

20. [H3 bounded lifecycle qualification](h3-integration.md#lifecycle-qualification-2026-10-18) — stream reuse resolves observed growth in a 22-call probe; four cancellations, retained/reopened recovery and delayed reclamation pass. Includes the failed control, fixed native evidence and 85 passing CPU tests. Full H3 generation remains future work.

21. [H3 three-block chain qualification](h3-integration.md#chain-qualification-2026-10-18) — corrected blocks 0–2 at production row geometry pass 38 submissions, 30 independent CPU block comparisons, repeat/recovery hashes and four cancellations. CPU comparisons use complete captured native input contexts; full CPU trajectory parity remains pending. The fifty-block follow-up is below.

22. [H3 fifty-block backbone qualification](h3-integration.md#backbone-qualification-2026-10-18) — one resident weight set passes 250 submissions, exact repeat/recovery hashes for all 200 full-pass outputs, sampled CPU gates at five checkpoints and midpoint cancellation. Peak footprint is 7.73 GiB; post-unload footprint settles near 0.61 GiB. Includes 95 passing CPU tests. Complete Turbo8 generation remains pending.

23. [H3 outer-phase CPU readiness](h3-integration.md#outer-phase-binding-readiness) — prepare/time/heads bindings with complete original CPU references, strict timestep/index validation, multi-output session ownership and an explicit bounded GPU probe. All 105 CPU tests and real selected-file preparation pass; the native follow-up is below.

24. [H3 outer-phase GPU qualification](h3-integration.md#outer-phase-qualification-2026-10-18) — 30 submissions pass 28 complete-tensor CPU comparisons, repeat/recovery hashes and ten cancellations across prepare/time/heads. Peak footprint is 3.76 GiB; final post-unload footprint is 0.58 GiB. These isolated module probes do not qualify a linked Turbo8 denoising step or complete generation.

25. [H3 linked eight-step denoising qualification](h3-integration.md#denoising-qualification-2026-10-18) — provided-conditioning trajectory passes 824 native submissions, 464 original-CPU comparisons (448 complete tensors and 16 sampled body checkpoints), and all 864 capture hashes. Includes the verifier memory fix and failed control, 112 passing CPU tests, and 10.57 GiB peak / 0.98 GiB post-unload footprint. Full-run lifecycle, conditioning encoder, decoders and media acceptance remain pending.
