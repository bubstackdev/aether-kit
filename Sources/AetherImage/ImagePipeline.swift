import AetherCore
public import AetherHost
public import AetherRuntime

// What the image drivers share (DESIGN.md §6.4): their options and request, the admission prologue every generation
// runs, and the load-time check of each graph against its driver contract.

/// How an image pipeline loads its graphs and shares memory and the GPU.
public struct ImagePipelineOptions: Sendable {
    public var cachePolicy: GraphOptions.CachePolicy
    public var cache: GraphOptions.Cache
    /// The budget a generation's planned peak must fit when there is no coordinator (default ``MemoryBudget/imageGeneration``).
    public var memory: MemoryBudget
    /// A shared memory budget: the generation reserves its planned peak here (and fits the available bytes).
    public var coordinator: ResourceCoordinator?
    /// GPU admission for the generation (in-process and, if configured, the cross-process lease).
    public var gpu: GPUAdmissionOwner?
    /// Measured footprints, shared with other pipelines of the model (and seedable with qualified values).
    public var ledger: FootprintLedger
    /// What phases are measured with. Tests pass a fixed probe, so what they plan never depends on whatever else
    /// the process has allocated.
    public var probe: GPUMemoryProbe

    public init(
        cachePolicy: GraphOptions.CachePolicy = .default, cache: GraphOptions.Cache = .default,
        memory: MemoryBudget = .imageGeneration, coordinator: ResourceCoordinator? = nil,
        gpu: GPUAdmissionOwner? = nil, ledger: FootprintLedger = FootprintLedger(), probe: GPUMemoryProbe = .system
    ) {
        self.cachePolicy = cachePolicy
        self.cache = cache
        self.memory = memory
        self.coordinator = coordinator
        self.gpu = gpu
        self.ledger = ledger
        self.probe = probe
    }
}

extension MemoryBudget {
    /// The image pipelines' default budget: on macOS three quarters of RAM, about Metal's recommended working set on
    /// Apple silicon, since these models' graphs are almost all GPU memory (Krea 2 needs about 34 GiB a generation per
    /// phase, measured: more than ``platformDefault``'s half of a 64 GB Mac, which suits a decoder's own buffers); on iOS
    /// what the process may still allocate, less 512 MB, as ``platformDefault``.
    public static var imageGeneration: MemoryBudget {
        #if os(iOS)
        .platformDefault
        #else
        MemoryBudget(.fractionOfRAM(0.75))
        #endif
    }
}

/// One text-to-image generation.
public struct ImageRequest: Sendable, Hashable {
    public var prompt: String
    public var seed: UInt64
    /// Absent: the manifest's default.
    public var steps: Int?
    /// Absent: the export's default size. One side alone makes a square.
    public var width: Int?
    public var height: Int?

    public init(prompt: String, seed: UInt64 = 0, steps: Int? = nil, width: Int? = nil, height: Int? = nil) {
        self.prompt = prompt
        self.seed = seed
        self.steps = steps
        self.width = width
        self.height = height
    }

    /// The requested size, or `fallback` (the export's default) when neither side is given.
    func size(default fallback: (width: Int, height: Int)?) -> (width: Int, height: Int)? {
        switch (width, height) {
        case (let w?, let h?): (w, h)
        case (let w?, nil): (w, w)
        case (nil, let h?): (h, h)
        case (nil, nil): fallback
        }
    }
}

extension ImagePipelineOptions {
    /// The pipeline's graphs and their admission (``ResidentAdmission``): planned from the ledger (or the graphs'
    /// files), reserved from the coordinator, if any, for the generation and then for what stays resident.
    func admission(
        manifest: Manifest, device: Device, load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64,
        maximumShapes: Int = 4
    ) -> ResidentAdmission {
        ResidentAdmission(
            manifest: manifest, device: device,
            settings: .init(memory: memory, coordinator: coordinator, ledger: ledger, probe: probe, maximumShapes: maximumShapes),
            load: load, fileBytes: fileBytes)
    }
}

enum ImageGraphs {
    /// Wraps `loader` so each graph is checked against its manifest role and `contract` before use (MANIFEST.md §2.1.1).
    static func validating(
        _ loader: @escaping GraphResidency.Loader, manifest: Manifest, contract: DriverContract
    ) -> GraphResidency.Loader {
        { role throws(AetherError) in
            let graph = try await loader(role)
            guard let spec = manifest.graphs[role], let graphContract = contract.graph(role) else {
                throw .manifestInvalid(field: "graphs.\(role)", reason: "not a \(contract.driver) graph")
            }
            try SignatureValidator.validate(spec, phase: "step", at: "graphs.\(role)", against: graph.signature,
                                            contract: graphContract, tables: manifest.tables ?? [:])
            return graph
        }
    }
}
