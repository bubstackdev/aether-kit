public import AetherCore
public import AetherRuntime

/// What the stateless drivers share (M4: depth, detection, CLIP, embedders, classifiers, super-resolution): one
/// manifest's fully static graphs, run once per request, with no state between requests.
///
/// Owned by the driver's actor, like ``ResidentAdmission`` (not `Sendable`). Each ``perform(job:profile:_:)``:
/// 1. takes the GPU through the options' admission (by default ``GPUAdmission/shared``) at the request's `GPUJob`
///    priority, before this pipeline's own turn, so priority also orders one pipeline's requests;
/// 2. waits for the pipeline's turn (requests queue, never fail as busy);
/// 3. plans and reserves memory (``ResidentAdmission``: the measured footprint from the ledger, or the graphs' files
///    until measured), refusing with `capacity` before anything loads;
/// 4. runs the driver's body in one measured phase, every graph loaded on demand and checked against the driver's
///    contract; the graphs stay resident for the next request.
///
/// A failure (cancellation included) releases the graphs and the reservation; the pipeline stays usable. Lives in
/// AetherHost, beside ``PipelineOptions`` and ``ResidentAdmission``, so AetherVision and AetherEmbeddings share it
/// without depending on each other.
public final class StatelessPipeline {
    public let manifest: Manifest
    public let options: PipelineOptions
    private let admission: ResidentAdmission
    /// One request at a time, in arrival order (after the GPU grant).
    private let gate = SerialGate()

    /// The pipeline over an installed bundle's graphs. Nothing loads until the first request. `defaultMemory` is the
    /// budget when the options give none.
    public convenience init(
        bundle: InstalledBundle, manifest: Manifest, device: Device, contract: DriverContract, options: PipelineOptions,
        defaultMemory: MemoryBudget = .platformDefault
    ) throws(AetherError) {
        try self.init(
            manifest: manifest, device: device, contract: contract, options: options, defaultMemory: defaultMemory,
            bundle: bundle, load: GraphResidency.bundleLoader(
                bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache),
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The pipeline over any graph loader (tests drive it with fakes). Each loaded graph is checked against
    /// `contract` before use.
    public init(
        manifest: Manifest, device: Device, contract: DriverContract, options: PipelineOptions,
        defaultMemory: MemoryBudget = .platformDefault, bundle: InstalledBundle? = nil,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        try contract.validate(manifest)
        self.manifest = manifest
        self.options = options
        // Static graphs have no residency: everything stays loaded, measured under `perStep`, one shape.
        admission = try options.admission(
            manifest: manifest, device: device, defaultMemory: defaultMemory, unphasedPhase: .perStep, maximumShapes: 1,
            bundle: bundle, load: GraphResidency.validating(load, manifest: manifest, contract: contract), fileBytes: fileBytes)
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { admission.residentRoles }

    /// Runs one request (the steps above). `profile` keys the measured footprint; static graphs have one shape, so the
    /// default `""` fits all but drivers whose request picks among shapes. In `body`, take graphs from the residency
    /// and run them with ``GraphResidency/run(_:_:)``, which stops between graph calls when the task is cancelled.
    public func perform<T>(
        job: GPUJob?, profile: String = "", _ body: (GraphResidency) async throws -> T
    ) async throws -> T {
        try Task.checkCancellation()
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: profile)
            do {
                try await residency.beginPhase(.perStep)
                let result = try await body(residency)
                residency.endPhase(.perStep)
                admission.finish()
                return result
            } catch {
                admission.release()
                throw error
            }
        }
    }

    /// Releases every graph and the reservation, waiting for a running request. The next request loads them again.
    public func unload() async {
        await gate.withTurnIgnoringCancellation {
            admission.unload()
        }
    }
}

extension GraphResidency {
    /// Runs `graph`, samples the open phase's memory, then throws `CancellationError` if the task was cancelled: the
    /// stateless drivers' one graph call.
    public func run(_ graph: any GraphBackend, _ bindings: inout Bindings) async throws {
        try await graph.run(&bindings)
        sample()
        try Task.checkCancellation()
    }
}
