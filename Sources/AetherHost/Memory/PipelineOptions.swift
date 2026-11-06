public import AetherCore
public import AetherRuntime

/// How a generative pipeline loads its graphs and shares memory and the GPU: one type for every task module (image,
/// speech, and later vision and video), so a host configures them alike.
public struct PipelineOptions: Sendable {
    public var cachePolicy: GraphOptions.CachePolicy
    public var cache: GraphOptions.Cache
    /// The budget a generation's planned peak must fit when there is no coordinator. Absent: the pipeline's own
    /// default (image generation: ``MemoryBudget/imageGeneration``; speech: ``MemoryBudget/platformDefault``).
    public var memory: MemoryBudget?
    /// A shared memory budget: the generation reserves its planned peak here, then keeps what stays resident.
    public var coordinator: ResourceCoordinator?
    /// GPU admission for each generation (in-process and, if configured, the cross-process lease).
    public var gpu: GPUAdmissionOwner?
    /// Measured footprints, shared with other pipelines of the model (and seedable with qualified values).
    public var ledger: FootprintLedger
    /// What phases are measured with. Tests pass a fixed probe, so what they plan never depends on whatever else the
    /// process has allocated.
    public var probe: GPUMemoryProbe

    public init(
        cachePolicy: GraphOptions.CachePolicy = .default, cache: GraphOptions.Cache = .default, memory: MemoryBudget? = nil,
        coordinator: ResourceCoordinator? = nil, gpu: GPUAdmissionOwner? = nil, ledger: FootprintLedger = FootprintLedger(),
        probe: GPUMemoryProbe = .system
    ) {
        self.cachePolicy = cachePolicy
        self.cache = cache
        self.memory = memory
        self.coordinator = coordinator
        self.gpu = gpu
        self.ledger = ledger
        self.probe = probe
    }

    /// The pipeline's graphs and their admission (``ResidentAdmission``), with `defaultMemory` when ``memory`` is
    /// absent.
    public func admission(
        manifest: Manifest, device: Device, defaultMemory: MemoryBudget, unphasedPhase: Manifest.Residency = .perStep,
        maximumShapes: Int = 4, load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64
    ) -> ResidentAdmission {
        ResidentAdmission(
            manifest: manifest, device: device,
            settings: .init(memory: memory ?? defaultMemory, coordinator: coordinator, ledger: ledger, probe: probe,
                            unphasedPhase: unphasedPhase, maximumShapes: maximumShapes),
            load: load, fileBytes: fileBytes)
    }
}
