// AetherHost: ModelHost, residency and capability (DESIGN.md §10).
// So far: GPU coordination (SharedGPULease, GPUAdmission, PriorityInferenceQueue, GPUJob) and the first ModelHost slice
// (per-model lifecycle under memory pressure). Residency across models and capability() land in M6.
@_exported public import AetherRuntime
@_exported public import AetherCore
