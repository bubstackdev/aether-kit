public import AetherCore
public import AetherRuntime
import AetherText
import Foundation

/// `flux2-klein@1` (MANIFEST.md §5.7): FLUX.2 [klein] text-to-image. A Qwen3 text encoder, a flow-matching transformer
/// split into segments chained by their carries, and a VAE decoder, on a `flowMatchEuler` schedule with the empirical
/// shift. Loads a ``Flux2KleinModel``: set its options, then make the ``Flux2KleinPipeline``. macOS only in 0.1.0: on
/// iOS the load is refused before anything opens.
public enum Flux2KleinDriver: ModelDriver {
    public static let id: DriverID = .flux2Klein

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Flux2KleinModel {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        try Flux2KleinPipeline.requireMacOS(manifest, options.device)
        try DriverContract.flux2Klein.validate(manifest)
        return Flux2KleinModel(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened FLUX.2 [klein] bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings.
public struct Flux2KleinModel: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: ImagePipelineOptions

    public func pipeline() async throws -> Flux2KleinPipeline {
        try await Flux2KleinPipeline(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

/// One loaded FLUX.2 [klein] model (DESIGN.md §6.4): prompt → text encoder → per step the denoiser segments in order
/// with the host Euler update → VAE decoder → ``GeneratedImage``.
///
/// - **Prompt:** the bundle's chat template over one user message (generation prompt, thinking off), right-padded
///   with the pad token to `flux2.prompt.maxTokens`, a longer prompt keeping its prefix. Every row, padding included,
///   conditions the denoiser.
/// - **Memory, ordering, cancellation:** as ``LatentDiffusionPipeline``: residency planned from measured footprints
///   before each generation (`capacity` before anything loads), one generation at a time, cancellation between graph
///   calls, and any failure releases every graph.
/// - **Noise:** the seed draws the packed initial latents from Kit's ``SeededGaussianNoise`` (not PyTorch's generator),
///   so a seed gives the same image here and in the Forge reference.
/// - Reference images (the bundle's optional `vaeEncoder`) are not implemented yet.
public actor Flux2KleinPipeline {
    public typealias Options = ImagePipelineOptions
    public typealias Request = ImageRequest
    /// The rendered, unpadded prompt token ids for a prompt.
    typealias PromptEncoder = @Sendable (String) throws -> [Int32]

    public nonisolated let manifest: Manifest
    public nonisolated let schedule: Manifest.Schedule
    public nonisolated let latent: Manifest.Latent
    public nonisolated let flux2: Manifest.Flux2
    public nonisolated let options: Options
    nonisolated let wiring: Wiring
    nonisolated let encodePrompt: PromptEncoder
    let device: Device
    private let load: GraphResidency.Loader
    private let admission: ResidentAdmission
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Opens an installed bundle; graphs load on the first generation.
    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws {
        try Self.requireMacOS(manifest, device)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        let template = try PromptRenderers.chatTemplate(manifest: manifest, bundle: bundle)
        let rule = manifest.flux2?.prompt
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        try self.init(
            manifest: manifest, device: device, options: options,
            encodePrompt: { text in
                try tokenizer.applyChatTemplate(
                    [.user(text)], template: template, context: ["enable_thinking": false],
                    addGenerationPrompt: rule?.addGenerationPrompt ?? true)
            },
            load: ImageGraphs.validating(loader, manifest: manifest, contract: .flux2Klein),
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The pipeline over any prompt encoder and graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, options: Options, encodePrompt: @escaping PromptEncoder,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping @Sendable (String) -> Int64
    ) throws(AetherError) {
        try Self.requireMacOS(manifest, device)
        try DriverContract.flux2Klein.validate(manifest)
        guard let schedule = manifest.schedule, schedule.kind == .flowMatchEuler else {
            throw .manifestInvalid(field: "schedule", reason: "\(Flux2KleinDriver.id) requires a flowMatchEuler schedule")
        }
        guard schedule.guidance.kind == .none else {
            throw .manifestInvalid(field: "schedule.guidance", reason: "FLUX.2 [klein] is distilled without guidance")
        }
        guard let latent = manifest.latent, latent.packing == .patch2x2ChannelMajor else {
            throw .manifestInvalid(field: "latent", reason: "\(Flux2KleinDriver.id) requires patch2x2ChannelMajor latents")
        }
        // These exports have no fixed sizes, so the token bound is what keeps a request's canvas (and its unmeasured
        // memory) finite.
        guard latent.maxImageTokens != nil else {
            throw .manifestInvalid(field: "latent.maxImageTokens", reason: "required by \(Flux2KleinDriver.id)")
        }
        guard let flux2 = manifest.flux2 else {
            throw .manifestInvalid(field: "flux2", reason: "required by \(Flux2KleinDriver.id)")
        }
        guard manifest.tokenizer?.chatTemplate != nil else {
            throw .manifestInvalid(field: "tokenizer.chatTemplate", reason: "\(Flux2KleinDriver.id) renders the prompt with it")
        }
        wiring = try Wiring(manifest: manifest, flux2: flux2)
        self.manifest = manifest
        self.schedule = schedule
        self.latent = latent
        self.flux2 = flux2
        self.device = device
        self.options = options
        self.encodePrompt = encodePrompt
        self.load = load
        admission = options.admission(manifest: manifest, device: device, load: load, fileBytes: fileBytes)
    }

    /// FLUX.2, Krea and H3 are qualified for macOS only in 0.1.0 (docs/INTEGRATION.md).
    static func requireMacOS(_ manifest: Manifest, _ device: Device) throws(AetherError) {
        guard device.platform == .macOS else {
            throw .unsupportedDevice(catalogID: manifest.id, platform: device.platform.rawValue, architecture: device.architecture)
        }
    }

    /// Generates one image. FLUX.2 exports have no default size: pass `width` and `height` (positive multiples of
    /// `latent.pixelsPerToken`, at most `latent.maxImageTokens` tokens).
    public func generate(_ request: Request) async throws -> GeneratedImage {
        try await generate(request, conditioning: nil)
    }

    /// ``generate(_:)``, or, with `conditioning` (the text encoder's rows, `[maxTokens, width]` row-major), from those rows
    /// without the prompt phase: the gate's path from the reference's own rows.
    func generate(_ request: Request, conditioning: [Float]?) async throws -> GeneratedImage {
        try Task.checkCancellation()
        // Validate the request before waiting for a turn or the GPU.
        guard let size = request.size(default: latent.defaultSize) else {
            throw AetherError.generation(.invalidParameters("FLUX.2 has no default size; pass width and height"))
        }
        let grid = try latent.tokenGrid(width: size.width, height: size.height)
        let tokens = grid.rows * grid.columns
        let table = try FlowMatchSchedule(schedule, steps: request.steps, imageTokens: tokens)
        let prompt: Prompt?
        if let conditioning {
            let rows = flux2.conditioning.rows, width = flux2.conditioning.width
            guard conditioning.count == rows * width else {
                throw AetherError.tensor(name: "conditioning", reason: .shapeMismatch(expected: [rows, width], found: [conditioning.count]))
            }
            prompt = nil
        } else {
            prompt = try Prompt(rendered: try encodePrompt(request.prompt), rule: flux2.prompt)
        }

        await turn()
        defer { endTurn() }
        try Task.checkCancellation()
        let grant = try await options.gpu?.admit()
        defer { grant?.release() }
        let residency = try admission.begin(profile: "\(size.width)x\(size.height)")
        do {
            let image = try await run(
                residency, prompt: prompt, conditioning: conditioning, seed: request.seed, schedule: table, grid: grid, size: size)
            admission.finish()
            return image
        } catch {
            admission.release()
            throw error
        }
    }

    /// The text encoder's rows for `prompt`, `[maxTokens, width]` row-major (the gate's text check). Loads the text
    /// encoder for the call and releases it; nothing is measured.
    func conditioningRows(for prompt: String) async throws -> [Float] {
        let prompt = try Prompt(rendered: try encodePrompt(prompt), rule: flux2.prompt)
        await turn()
        defer { endTurn() }
        let text = try await load(wiring.text.role)
        return try await encode(prompt, with: text, in: TensorArena()).readFloats()
    }

    /// Releases every graph (waiting for a running generation). The next generation loads them again.
    public func unload() async {
        await turn()
        admission.release()
        endTurn()
    }

    /// Roles currently loaded (for tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    // MARK: - The generation

    private func run(
        _ residency: GraphResidency, prompt: Prompt?, conditioning provided: [Float]?, seed: UInt64,
        schedule: FlowMatchSchedule, grid: (rows: Int, columns: Int), size: (width: Int, height: Int)
    ) async throws -> GeneratedImage {
        let arena = TensorArena()
        let textRows = flux2.prompt.maxTokens, tokens = grid.rows * grid.columns
        let count = tokens * latent.channels
        let dims = Dimensions(textRows: textRows, imageTokens: tokens)

        // Prompt: the conditioning rows outlive the text encoder (it may be released before the denoisers load).
        var conditioning: TensorBuffer?
        if let prompt {
            residency.beginPhase(.perPrompt)
            conditioning = try await encode(prompt, with: try await residency.graph(wiring.text.role), in: arena)
            residency.sample()
            residency.endPhase(.perPrompt)
            try Task.checkCancellation()
        }

        // Steps: every segment loads, then each step writes the latents and sigma and runs the segments in order.
        residency.beginPhase(.perStep)
        var graphs: [any GraphBackend] = []
        for segment in wiring.segments { graphs.append(try await residency.graph(segment.role)) }
        let first = graphs[0]
        if let provided {
            let rows = try first.makeBuffer(for: wiring.conditioningInput, role: .input, dimensions: [textRows], in: arena)
            try rows.write(Float16.self, count: provided.count) { Float16(provided[$0]) }
            conditioning = rows
        }
        guard let conditioning else { preconditionFailure("the prompt phase or the caller provides the conditioning") }
        let sample = try first.makeBuffer(for: wiring.latents, role: .input, dimensions: [tokens], in: arena)
        let sigma = try first.makeBuffer(for: wiring.sigma, role: .input, dimensions: [], in: arena)
        let textRotary = try first.makeBuffer(for: wiring.textRotary, role: .input, dimensions: [textRows], in: arena)
        let imageRotary = try first.makeBuffer(for: wiring.imageRotary, role: .input, dimensions: [tokens], in: arena)
        guard sample.shape == [tokens, latent.channels] else {
            throw AetherError.manifestMismatch(
                field: "latent.channels", expected: "denoiser latents [\(tokens), \(latent.channels)]", found: "\(sample.shape)")
        }
        try textRotary.write(Flux2Rotary.table(ids: Flux2Rotary.textIDs(count: textRows), rotary: flux2.rotary))
        try imageRotary.write(Flux2Rotary.table(ids: Flux2Rotary.gridIDs(rows: grid.rows, columns: grid.columns), rotary: flux2.rotary))
        var outputs: [String: TensorBuffer] = [:]
        for (segment, graph) in zip(wiring.segments, graphs) {
            for output in segment.outputs {
                outputs[output.ref] = try graph.makeBuffer(
                    for: output.name, role: .output, dimensions: dims.of(output.role), in: arena)
            }
        }
        let velocity = outputs[wiring.velocity]!
        guard velocity.shape == [tokens, latent.channels] else {
            throw AetherError.manifestMismatch(
                field: "latent.channels", expected: "velocity [\(tokens), \(latent.channels)]", found: "\(velocity.shape)")
        }
        func buffer(_ source: Wiring.Source) -> TensorBuffer {
            switch source {
            case .latents: sample
            case .conditioning: conditioning
            case .sigma: sigma
            case .textRotary: textRotary
            case .imageRotary: imageRotary
            case .output(let ref): outputs[ref]!
            }
        }

        var latents = try SeededGaussianNoise.generate(seed: seed, count: count)
        var v = [Float](repeating: 0, count: count)
        for step in 0..<schedule.steps {
            try sample.write(Float16.self, count: count) { Float16(latents[$0]) }
            try sigma.write([schedule.timesteps[step]])
            for (segment, graph) in zip(wiring.segments, graphs) {
                try Task.checkCancellation()
                var bindings = Bindings(arena: arena)
                for input in segment.inputs { try bindings.bindInput(input.name, to: buffer(input.source)) }
                for output in segment.outputs { try bindings.bindOutput(output.name, to: outputs[output.ref]!) }
                try await graph.run(&bindings)
                residency.sample()
            }
            try velocity.readFloats(0..<count, into: &v)
            try schedule.advance(step: step, latents: &latents, velocity: v)
        }
        residency.endPhase(.perStep)
        try Task.checkCancellation()

        // Image: the packed tokens, raster order, are the decoder's [rows, columns, channels].
        residency.beginPhase(.perImage)
        let vae = try await residency.graph(wiring.decoder.role)
        let input = try vae.makeBuffer(for: wiring.decoder.latents, role: .input, dimensions: [grid.rows, grid.columns], in: arena)
        let output = try vae.makeBuffer(for: wiring.decoder.image, role: .output, dimensions: [size.height, size.width], in: arena)
        guard input.shape == [grid.rows, grid.columns, latent.channels], output.shape == [3, size.height, size.width] else {
            throw AetherError.manifestMismatch(
                field: "graphs.\(wiring.decoder.role)", expected: "[\(grid.rows), \(grid.columns), \(latent.channels)] → [3, \(size.height), \(size.width)]",
                found: "\(input.shape) → \(output.shape)")
        }
        try input.write(Float16.self, count: count) { Float16(latents[$0]) }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.decoder.latents, to: input)
        try bindings.bindOutput(wiring.decoder.image, to: output)
        try await vae.run(&bindings)
        residency.sample()
        residency.endPhase(.perImage)
        return try GeneratedImage(width: size.width, height: size.height, planarRGB: try output.readFloats())
    }

    /// Runs the text encoder on `prompt`: its rows, `[maxTokens, width]`, in `arena`.
    private func encode(_ prompt: Prompt, with text: any GraphBackend, in arena: TensorArena) async throws -> TensorBuffer {
        let textRows = flux2.prompt.maxTokens
        let ids = try text.makeBuffer(for: wiring.text.tokens, role: .input, dimensions: [textRows], in: arena)
        let positions = try text.makeBuffer(for: wiring.text.positions, role: .input, dimensions: [textRows], in: arena)
        let mask = try text.makeBuffer(for: wiring.text.mask, role: .input, dimensions: [textRows], in: arena)
        let rows = try text.makeBuffer(for: wiring.text.conditioning, role: .output, dimensions: [textRows], in: arena)
        guard rows.shape == [textRows, flux2.conditioning.width] else {
            throw AetherError.manifestMismatch(
                field: "flux2.conditioning.width", expected: "text rows [\(textRows), \(flux2.conditioning.width)]", found: "\(rows.shape)")
        }
        try ids.write(prompt.ids)
        try positions.write(Int32.self, count: textRows) { Int32($0) }
        try mask.write(prompt.mask)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.text.tokens, to: ids)
        try bindings.bindInput(wiring.text.positions, to: positions)
        try bindings.bindInput(wiring.text.mask, to: mask)
        try bindings.bindOutput(wiring.text.conditioning, to: rows)
        try await text.run(&bindings)
        return rows
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
}

// MARK: - Host pieces

extension Flux2KleinPipeline {
    /// The text encoder's input: rendered ids right-padded with the pad token to `maxTokens`, a longer prompt keeping
    /// its prefix, and the mask of the real tokens.
    struct Prompt: Sendable, Hashable {
        var ids: [Int32]
        var mask: [Int32]
        var truncated: Bool

        init(rendered: [Int32], rule: Manifest.Flux2.Prompt) throws(AetherError) {
            guard !rendered.isEmpty, rendered.allSatisfy({ $0 >= 0 }) else {
                throw .generation(.invalidParameters("the rendered FLUX.2 prompt is empty or has a negative token id"))
            }
            let kept = Array(rendered.prefix(rule.maxTokens))
            let pad = rule.maxTokens - kept.count
            ids = kept + Array(repeating: Int32(rule.padToken), count: pad)
            mask = Array(repeating: 1, count: kept.count) + Array(repeating: 0, count: pad)
            truncated = rendered.count > rule.maxTokens
        }
    }

    /// Which dynamic dimension each role's tensors have: text-side tensors the padded prompt length, image-side ones
    /// the generated token count.
    struct Dimensions {
        var textRows: Int
        var imageTokens: Int

        func of(_ role: String) -> [Int] {
            switch role {
            case "carryText", "textRotary", "conditioning": [textRows]
            case "carryImage", "imageRotary", "latents", "velocity": [imageTokens]
            default: []
            }
        }
    }

    /// The manifest's tensor names and how each segment input is fed, checked once at load: an input chained with
    /// `source.from` must name the text encoder's conditioning or an earlier segment's output, and only the last
    /// segment emits `velocity`.
    struct Wiring: Sendable {
        enum Source: Sendable, Hashable { case latents, conditioning, sigma, textRotary, imageRotary, output(String) }
        struct Input: Sendable { var role: String; var name: String; var source: Source }
        struct Output: Sendable { var role: String; var name: String; var ref: String }
        struct Segment: Sendable { var role: String; var inputs: [Input]; var outputs: [Output] }

        var text: (role: String, tokens: String, positions: String, mask: String, conditioning: String)
        var segments: [Segment]
        var decoder: (role: String, latents: String, image: String)
        /// The first segment's names for the host-fed inputs.
        var latents: String, sigma: String, textRotary: String, imageRotary: String, conditioningInput: String
        /// `role.name` of the last segment's velocity.
        var velocity: String

        init(manifest: Manifest, flux2: Manifest.Flux2) throws(AetherError) {
            let textGraph = manifest.graphs["textEncoder"]!, vae = manifest.graphs["vaeDecoder"]!
            let conditioningRef = "textEncoder.\(textGraph.outputs["conditioning"]!.name)"
            text = ("textEncoder", textGraph.inputs["tokens"]!.name, textGraph.inputs["positions"]!.name,
                    textGraph.inputs["tokenMask"]!.name, textGraph.outputs["conditioning"]!.name)
            decoder = ("vaeDecoder", vae.inputs["latents"]!.name, vae.outputs["image"]!.name)
            let contracted = Set(DriverContract.flux2Klein.graphs.keys.filter { $0.hasPrefix("denoiser") })
            guard Set(flux2.segments) == contracted, flux2.segments.count == contracted.count else {
                throw .manifestInvalid(field: "flux2.segments", reason: "must list \(contracted.sorted()) in call order")
            }
            var produced: Set<String> = []
            var segments: [Segment] = []
            for (index, role) in flux2.segments.enumerated() {
                let graph = manifest.graphs[role]!
                var inputs: [Input] = []
                for (inputRole, binding) in graph.inputs.sorted(by: { $0.key < $1.key }) {
                    let field = "graphs.\(role).inputs.\(inputRole)"
                    let source: Source
                    if case .from(let ref)? = binding.source {
                        if ref == conditioningRef, inputRole == "conditioning" {
                            source = .conditioning
                        } else if produced.contains(ref) {
                            source = .output(ref)
                        } else {
                            throw .manifestInvalid(
                                field: "\(field).source", reason: "'\(ref)' is not \(conditioningRef) or an earlier segment's output")
                        }
                    } else {
                        switch inputRole {
                        case "latents": source = .latents
                        case "sigma": source = .sigma
                        case "textRotary": source = .textRotary
                        case "imageRotary": source = .imageRotary
                        default:
                            throw .manifestInvalid(field: "\(field).source", reason: "must be chained with source.from")
                        }
                    }
                    inputs.append(Input(role: inputRole, name: binding.name, source: source))
                }
                let outputs = graph.outputs.sorted(by: { $0.key < $1.key }).map { Output(role: $0.key, name: $0.value.name, ref: "\(role).\($0.value.name)") }
                let emitsVelocity = outputs.contains { $0.role == "velocity" }
                guard emitsVelocity == (index == flux2.segments.count - 1) else {
                    throw .manifestInvalid(field: "flux2.segments", reason: "only the last segment, \(flux2.segments.last!), emits velocity")
                }
                produced.formUnion(outputs.map(\.ref))
                segments.append(Segment(role: role, inputs: inputs, outputs: outputs))
            }
            self.segments = segments
            let first = manifest.graphs[flux2.segments[0]]!
            guard case .from(conditioningRef)? = first.inputs["conditioning"]?.source else {
                throw .manifestInvalid(field: "graphs.\(flux2.segments[0]).inputs.conditioning.source", reason: "must be from \(conditioningRef)")
            }
            latents = first.inputs["latents"]!.name
            sigma = first.inputs["sigma"]!.name
            textRotary = first.inputs["textRotary"]!.name
            imageRotary = first.inputs["imageRotary"]!.name
            conditioningInput = first.inputs["conditioning"]!.name
            velocity = segments.last!.outputs.first { $0.role == "velocity" }!.ref
        }
    }
}

/// FLUX.2's host rotary tables (MANIFEST.md §5.7 `flux2.rotary`): position ids of 4 axes, `[n, 2, Σ axes]` float32
/// (cos rows, then sin rows) in the split-halves layout (pair `i` rotates dimensions `i` and `i + Σ axes / 2`). Angles,
/// cos and sin are computed in Double and rounded once.
enum Flux2Rotary {
    /// Text token `i`: `(0, 0, 0, i)`.
    static func textIDs(count: Int) -> [Int32] { (0..<count).flatMap { [0, 0, 0, Int32($0)] } }

    /// A `rows × columns` grid in raster order: `(0, row, column, 0)`.
    static func gridIDs(rows: Int, columns: Int) -> [Int32] {
        var ids: [Int32] = []
        ids.reserveCapacity(rows * columns * 4)
        for row in 0..<rows { for column in 0..<columns { ids += [0, Int32(row), Int32(column), 0] } }
        return ids
    }

    static func table(ids: [Int32], rotary: Manifest.Flux2.Rotary) -> [Float] {
        let axes = rotary.axes, width = rotary.dimension, half = width / 2
        precondition(ids.count % axes.count == 0)
        var frequencies: [(axis: Int, value: Double)] = []
        for (axis, dimension) in axes.enumerated() {
            for j in stride(from: 0, to: dimension, by: 2) {
                frequencies.append((axis, 1 / pow(rotary.theta, Double(j) / Double(dimension))))
            }
        }
        let n = ids.count / axes.count
        var values = [Float](repeating: 0, count: n * 2 * width)
        for token in 0..<n {
            let base = token * 2 * width
            for (index, frequency) in frequencies.enumerated() {
                let angle = Double(ids[token * axes.count + frequency.axis]) * frequency.value
                let c = Float(cos(angle)), s = Float(sin(angle))
                values[base + index] = c
                values[base + index + half] = c
                values[base + width + index] = s
                values[base + width + index + half] = s
            }
        }
        return values
    }
}

extension Registry {
    /// The drivers AetherImage links: `latent-diffusion@1`, `flux2-klein@1` and `krea2@1`.
    public static let image = Registry([LatentDiffusionDriver.self, Flux2KleinDriver.self, Krea2Driver.self])
}
