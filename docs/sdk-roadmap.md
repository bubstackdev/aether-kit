# SDK implementation sequence

Baseline: remote main `8cad591`, matching Atelier's pin. The older local worktree remains preserved. App implementation and qualification stay with Atelier.

| Stage | Status | Next concrete acceptance |
| --- | --- | --- |
| Baseline reconciliation | Source/API reconciliation recorded in sdk-foundations.md | Consolidate newer profile-specific qualification records before promotion |
| Local asset foundation | Implemented: manifests, verified staged import, immutable IDs, protected readers, model-session lease binding | Weight-free tests and independent consumer; later crash cleanup; external range descriptor leases now implemented |
| Resource foundation | Implemented: transactional reservations and optional ModelSession admission | Krea/H3 adoption implemented with controlled lifetime tests; native-qualified estimates remain required |
| CI foundation | Workflow and CPU-only entry point added | Equipped CI runner plus clean consumer validation |
| Model asset adapters | Krea portable adapter implemented; H3 external weight descriptor leases implemented | Portable H3 graph/table installation and cache lifecycle remain; no automatic Atelier migration |
| Modality resource adoption | Krea model/operation and H3 operation/stage reservations implemented with caller estimates | Reserve load/scratch/state/stage overlap before allocation; deterministic cancellation/growth/pressure tests and separate native qualification |
| Small LLM | Pending | Confirm available Qwen3-0.6B export/license; state/KV ownership, prefill/decode, EOS, sampling, Unicode streaming, context bounds, reset/recovery; optional product |
| Multi-turn and integrations | Pending | Cold/prefix-equivalent responses, abandoned-stream cleanup, tools/guided output and FoundationModels separately qualified |
| Release | Pending | Privacy/resources, DocC, API compatibility, exact release consumer, support matrix, version/tag and install verification |
| Remote assets/catalog | Pending, optional for local-bundle-first release | Immutable revision transport, network policy, staged verification, offline reuse, trust roots and complete certification |
| Optimization and breadth | Pending | Profile current Krea/H3, preserve numerical/lifecycle behavior and measure matched assets; other modalities follow concrete consumers |

The existing aether-engine specifications and failure reproductions are behavioral references. In particular, reproduce admission/load/unload races and certification-completeness failures before reusing their corresponding subsystems. Partial/reference measurements must never count as shipping certification.
