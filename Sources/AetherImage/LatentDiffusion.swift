public import AetherCore
public import AetherHost
public import AetherRuntime
import Foundation

/// `latent-diffusion@1` (MANIFEST.md §1, §5): text-to-image with a CLIP text encoder, an epsilon-predicting U-Net and a
/// VAE decoder, on an `epsilonEuler` schedule (SD-Turbo). Loads a ``LatentDiffusionPipeline``.
public enum LatentDiffusionDriver: ModelDriver {
    public static let id: DriverID = .latentDiffusion

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> LatentDiffusionPipeline {
        try LatentDiffusionPipeline(
            bundle: bundle, manifest: try Manifest.load(fromBundle: bundle.url), device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

extension Registry {
    /// The drivers AetherImage links: `latent-diffusion@1`.
    public static let image = Registry([LatentDiffusionDriver.self])
}

/// One loaded latent diffusion model (DESIGN.md §6.4): prompt → text encoder → `steps` U-Net calls with the host Euler
/// update → VAE decoder → ``GeneratedImage``.
///
/// - **Memory:** before each generation it plans residency from measured footprints (``ResidencyPlan``): every graph
///   resident if that fits, else each phase's graphs released when the phase ends, else `capacity` before anything
///   loads. With a ``Options/coordinator`` it reserves the plan's peak there for the generation.
/// - **One at a time:** generations run in the order requested; ``unload()`` waits for the running one.
/// - **Cancellation:** cancelling the calling task stops between graph calls with `CancellationError`. Any failure
///   releases every graph; the pipeline stays usable.
/// - The seed draws the initial latents from Kit's ``SeededGaussianNoise`` in NCHW order (not PyTorch's generator), so
///   a seed gives the same image here and in the Forge reference.
public actor LatentDiffusionPipeline {
    public struct Options: Sendable {
        public var cachePolicy: GraphOptions.CachePolicy
        public var cache: GraphOptions.Cache
        /// The budget a generation's planned peak must fit when there is no coordinator.
        public var memory: MemoryBudget
        /// A shared memory budget: the generation reserves its planned peak here (and fits the available bytes).
        public var coordinator: ResourceCoordinator?
        /// GPU admission for the generation (in-process and, if configured, the cross-process lease).
        public var gpu: GPUAdmissionOwner?
        /// Measured footprints, shared with other pipelines of the model (and seedable with qualified values).
        public var ledger: FootprintLedger

        public init(
            cachePolicy: GraphOptions.CachePolicy = .default, cache: GraphOptions.Cache = .default,
            memory: MemoryBudget = .platformDefault, coordinator: ResourceCoordinator? = nil,
            gpu: GPUAdmissionOwner? = nil, ledger: FootprintLedger = FootprintLedger()
        ) {
            self.cachePolicy = cachePolicy
            self.cache = cache
            self.memory = memory
            self.coordinator = coordinator
            self.gpu = gpu
            self.ledger = ledger
        }
    }

    public struct Request: Sendable, Hashable {
        public var prompt: String
        public var seed: UInt64
        /// Absent: the manifest's default.
        public var steps: Int?
        /// Absent: the export's default size.
        public var width: Int?
        public var height: Int?

        public init(prompt: String, seed: UInt64 = 0, steps: Int? = nil, width: Int? = nil, height: Int? = nil) {
            self.prompt = prompt
            self.seed = seed
            self.steps = steps
            self.width = width
            self.height = height
        }
    }

    /// The graphs' tensor names, from the manifest.
    struct Names: Sendable {
        var tokens: String, textOut: String
        var latents: String, timestep: String, conditioning: String, epsilon: String
        var decoderIn: String, image: String
    }

    public nonisolated let manifest: Manifest
    public nonisolated let schedule: Manifest.Schedule
    public nonisolated let latent: Manifest.Latent
    let names: Names
    let tokenizer: CLIPTokenizer
    let device: Device
    public nonisolated let options: Options
    private let load: GraphResidency.Loader
    private let fileBytes: @Sendable (String) -> Int64
    private var residency: GraphResidency?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Opens an installed bundle; graphs load on the first generation.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) throws(AetherError) {
        guard let source = manifest.tokenizer?.source, case .dir(let path) = source, let directory = bundle.url(of: path) else {
            throw .manifestInvalid(field: "tokenizer.source", reason: "\(LatentDiffusionDriver.id) needs a tokenizer directory")
        }
        let tokenizer = try CLIPTokenizer(directory: directory)
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        try self.init(
            manifest: manifest, tokenizer: tokenizer, device: device, options: options,
            load: Self.validating(loader, manifest: manifest),
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The pipeline over any graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, tokenizer: CLIPTokenizer, device: Device, options: Options,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping @Sendable (String) -> Int64
    ) throws(AetherError) {
        try DriverContract.latentDiffusion.validate(manifest)
        guard let schedule = manifest.schedule, schedule.kind == .epsilonEuler else {
            throw .manifestInvalid(field: "schedule", reason: "\(LatentDiffusionDriver.id) requires an epsilonEuler schedule")
        }
        guard let latent = manifest.latent, latent.packing == .spatial else {
            throw .manifestInvalid(field: "latent", reason: "\(LatentDiffusionDriver.id) requires spatial latents")
        }
        guard schedule.guidance.kind == .none else {
            throw .manifestInvalid(field: "schedule.guidance", reason: "classifier-free guidance is not supported yet")
        }
        let text = manifest.graphs["textEncoder"]!, unet = manifest.graphs["denoiser"]!, vae = manifest.graphs["vaeDecoder"]!
        let conditioning = unet.inputs["conditioning"]!
        guard case .from(let from)? = conditioning.source, from == "textEncoder.\(text.outputs["conditioning"]!.name)" else {
            throw .manifestInvalid(field: "graphs.denoiser.inputs.conditioning.source",
                                   reason: "must be from textEncoder.\(text.outputs["conditioning"]!.name)")
        }
        names = Names(
            tokens: text.inputs["tokens"]!.name, textOut: text.outputs["conditioning"]!.name,
            latents: unet.inputs["latents"]!.name, timestep: unet.inputs["timestep"]!.name,
            conditioning: conditioning.name, epsilon: unet.outputs["epsilon"]!.name,
            decoderIn: vae.inputs["latents"]!.name, image: vae.outputs["image"]!.name)
        self.manifest = manifest
        self.schedule = schedule
        self.latent = latent
        self.tokenizer = tokenizer
        self.device = device
        self.options = options
        self.load = load
        self.fileBytes = fileBytes
    }

    /// Generates one image.
    public func generate(_ request: Request) async throws -> GeneratedImage {
        try Task.checkCancellation()
        // Validate the request before waiting for a turn or the GPU.
        let table = try EpsilonEulerSchedule(schedule, steps: request.steps)
        guard let size = Self.size(request, latent) else {
            throw AetherError.generation(.invalidParameters("this export has no default size; pass width and height"))
        }
        let grid = try latent.tokenGrid(width: size.width, height: size.height)
        let ids = try tokenizer.encode(request.prompt)

        await turn()
        defer { endTurn() }
        try Task.checkCancellation()
        let grant = try await options.gpu?.admit()
        defer { grant?.release() }
        let profile = "\(size.width)x\(size.height)"
        let inputs = ResidencyPlan.inputs(manifest: manifest, ledger: options.ledger, profile: profile, fileBytes: fileBytes)
        let budget = options.coordinator.map { Int64(clamping: $0.snapshot.availableBytes) } ?? options.memory.bytes(on: device)
        let plan = try ResidencyPlan.choose(inputs, budget: budget)
        let reservation = try options.coordinator?.reserve(bytes: UInt64(max(1, plan.peakBytes)))
        defer { reservation?.release() }
        let residency = residency(for: plan.mode, profile: profile)
        do {
            return try await run(residency, ids: ids, seed: request.seed, schedule: table, grid: grid, size: size)
        } catch {
            residency.releaseAll()
            throw error
        }
    }

    /// Releases every graph (waiting for a running generation). The next generation loads them again.
    public func unload() async {
        await turn()
        residency?.releaseAll()
        residency = nil
        endTurn()
    }

    /// Roles currently loaded (for tests and diagnostics).
    public var residentRoles: [String] { residency?.residentRoles ?? [] }

    // MARK: - The generation

    private func run(
        _ residency: GraphResidency, ids: [Int32], seed: UInt64, schedule: EpsilonEulerSchedule,
        grid: (rows: Int, columns: Int), size: (width: Int, height: Int)
    ) async throws -> GeneratedImage {
        let arena = TensorArena()
        let latentShape = [1, latent.channels, grid.rows, grid.columns]
        let count = latentShape.reduce(1, *)

        // Prompt: the conditioning outlives the text encoder (it may be released before the U-Net loads).
        residency.beginPhase(.perPrompt)
        let text = try await residency.graph("textEncoder")
        let tokens = try text.makeBuffer(for: names.tokens, role: .input, dimensions: [], in: arena)
        let conditioning = try text.makeBuffer(for: names.textOut, role: .output, dimensions: [], in: arena)
        try tokens.write(ids)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(names.tokens, to: tokens)
        try bindings.bindOutput(names.textOut, to: conditioning)
        try await text.run(&bindings)
        residency.sample()
        residency.endPhase(.perPrompt)
        try Task.checkCancellation()

        // Steps: the latents stay on the host; each step writes the scaled input and reads epsilon.
        residency.beginPhase(.perStep)
        let unet = try await residency.graph("denoiser")
        let sample = try unet.makeBuffer(for: names.latents, role: .input, dimensions: [], in: arena)
        let timestep = try unet.makeBuffer(for: names.timestep, role: .input, dimensions: [], in: arena)
        let epsilon = try unet.makeBuffer(for: names.epsilon, role: .output, dimensions: [], in: arena)
        guard sample.shape == latentShape, epsilon.shape == latentShape else {
            throw AetherError.manifestMismatch(
                field: "latent.sizes", expected: "denoiser latents \(latentShape) for \(size.width)x\(size.height)",
                found: "\(sample.shape)")
        }
        var latents = try SeededGaussianNoise.generate(seed: seed, count: count)
        latents.withUnsafeMutableBufferPointer { schedule.scaleInitialNoise($0) }
        var noise = [Float](repeating: 0, count: count)
        for step in 0..<schedule.steps {
            try Task.checkCancellation()
            try sample.write(try schedule.modelInput(step: step, latents: latents))
            try timestep.write([schedule.timesteps[step]])
            var bindings = Bindings(arena: arena)
            try bindings.bindInput(names.latents, to: sample)
            try bindings.bindInput(names.timestep, to: timestep)
            try bindings.bindInput(names.conditioning, to: conditioning)
            try bindings.bindOutput(names.epsilon, to: epsilon)
            try await unet.run(&bindings)
            residency.sample()
            try epsilon.readFloats(0..<count, into: &noise)
            try schedule.advance(step: step, latents: &latents, epsilon: noise)
        }
        residency.endPhase(.perStep)
        try Task.checkCancellation()

        // Image.
        residency.beginPhase(.perImage)
        let vae = try await residency.graph("vaeDecoder")
        let input = try vae.makeBuffer(for: names.decoderIn, role: .input, dimensions: [], in: arena)
        let output = try vae.makeBuffer(for: names.image, role: .output, dimensions: [], in: arena)
        guard output.shape == [3, size.height, size.width] else {
            throw AetherError.manifestMismatch(
                field: "graphs.vaeDecoder.outputs.image", expected: "[3, \(size.height), \(size.width)]", found: "\(output.shape)")
        }
        try input.write(latents)
        bindings = Bindings(arena: arena)
        try bindings.bindInput(names.decoderIn, to: input)
        try bindings.bindOutput(names.image, to: output)
        try await vae.run(&bindings)
        residency.sample()
        residency.endPhase(.perImage)
        return try GeneratedImage(width: size.width, height: size.height, planarRGB: try output.readFloats())
    }

    /// The residency for `mode`, keeping loaded graphs when the mode and profile are unchanged.
    private func residency(for mode: ResidencyPlan.Mode, profile: String) -> GraphResidency {
        if let residency, residency.mode == mode, residency.profile == profile { return residency }
        residency?.releaseAll()
        let fresh = GraphResidency(manifest: manifest, mode: mode, ledger: options.ledger, profile: profile, load: load)
        residency = fresh
        return fresh
    }

    static func size(_ request: Request, _ latent: Manifest.Latent) -> (width: Int, height: Int)? {
        switch (request.width, request.height) {
        case (let w?, let h?): (w, h)
        case (let w?, nil): (w, w)
        case (nil, let h?): (h, h)
        case (nil, nil): latent.defaultSize
        }
    }

    private func turn() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    private func endTurn() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }

    /// Wraps `loader` so each graph is checked against its manifest role before use (MANIFEST.md §2.1.1).
    static func validating(_ loader: @escaping GraphResidency.Loader, manifest: Manifest) -> GraphResidency.Loader {
        { role throws(AetherError) in
            let graph = try await loader(role)
            guard let spec = manifest.graphs[role], let contract = DriverContract.latentDiffusion.graphs[role] else {
                throw .manifestInvalid(field: "graphs.\(role)", reason: "not a \(LatentDiffusionDriver.id) graph")
            }
            try SignatureValidator.validate(spec, phase: "step", at: "graphs.\(role)", against: graph.signature,
                                            contract: contract, tables: manifest.tables ?? [:])
            return graph
        }
    }
}
