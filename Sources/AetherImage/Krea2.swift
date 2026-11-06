public import AetherCore
public import AetherHost
public import AetherRuntime
import AetherText
import Foundation

/// `krea2@1` (MANIFEST.md §5.5–5.6): Krea 2 Turbo text-to-image, vanilla or with a merged adapter (realism). A Qwen3-VL
/// text encoder whose tapped layers condition a flow-matching transformer split into four segments chained by their
/// carries, then a VAE decoder, on a `flowMatchEuler` schedule with a fixed exponential shift. macOS only in 0.1.0.
///
/// Loading gives a ``Krea2Model``, not a pipeline: the Krea 2 Community License (§4.2) requires a content filter, so a
/// ``Krea2Pipeline`` is only made with the app's ``ContentFilter`` (``Krea2Model/pipeline(filter:)``).
public enum Krea2Driver: ModelDriver {
    public static let id: DriverID = .krea2

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> Krea2Model {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        try Krea2Pipeline.requireMacOS(manifest, options.device)
        try DriverContract.krea2.validate(manifest)
        return Krea2Model(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened Krea 2 bundle, before a content filter is attached. Nothing is loaded yet.
public struct Krea2Model: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: ImagePipelineOptions

    /// The pipeline, reviewing every generation with `filter` (the app's; see ``ContentFilter``).
    public func pipeline(filter: some ContentFilter) async throws -> Krea2Pipeline {
        try await Krea2Pipeline(bundle: bundle, manifest: manifest, device: device, filter: filter, options: options)
    }
}

/// One loaded Krea 2 model (DESIGN.md §6.4): filter the prompt → text encoder → per step the four denoiser segments with
/// the host Euler update → VAE decoder → filter the image → ``GeneratedImage``.
///
/// - **Content filter:** required at construction, no default (``ContentFilter``). The prompt is reviewed before
///   anything runs and the image before it is returned; a refusal is `generation(.contentRefused)` and no image.
/// - **Prompt:** `krea2.prompt.systemPrefix + prompt + suffix`, tokenized without added specials and never padded. A
///   rendering longer than `maxTokens` keeps its first `maxTokens − keepTrailingTokens` ids and its last
///   `keepTrailingTokens` (the suffix's). The text encoder drops the `dropLeadingRows` prefix rows from its output.
/// - **Memory, ordering, cancellation:** as ``LatentDiffusionPipeline``: residency planned from measured footprints
///   before each generation (`capacity` before anything loads), one generation at a time, cancellation between graph
///   calls, and any failure releases every graph.
/// - **Noise:** the seed draws the packed initial latents from Kit's ``SeededGaussianNoise`` (not PyTorch's generator),
///   so a seed gives the same image here and in the Forge reference.
/// - Reference images and refine (reserved for `krea2@1`) are not implemented: the first export has no vision graph
///   or VAE encoder.
public actor Krea2Pipeline {
    public typealias Options = ImagePipelineOptions
    public typealias Request = ImageRequest
    /// The rendered token ids (`systemPrefix + prompt + suffix`, no added specials) for a prompt.
    typealias PromptEncoder = @Sendable (String) throws -> [Int32]

    public nonisolated let manifest: Manifest
    public nonisolated let schedule: Manifest.Schedule
    public nonisolated let latent: Manifest.Latent
    public nonisolated let krea2: Manifest.Krea2
    public nonisolated let options: Options
    nonisolated let filter: any ContentFilter
    nonisolated let wiring: Wiring
    nonisolated let encodePrompt: PromptEncoder
    let device: Device
    private let load: GraphResidency.Loader
    private let admission: ResidentAdmission
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Opens an installed bundle with the app's content filter; graphs load on the first generation.
    public init(
        bundle: InstalledBundle, manifest: Manifest, device: Device, filter: some ContentFilter, options: Options = Options()
    ) async throws {
        try Self.requireMacOS(manifest, device)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        let rule = manifest.krea2?.prompt
        let addSpecialTokens = manifest.tokenizer?.addSpecialTokens ?? false
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        try self.init(
            manifest: manifest, device: device, filter: filter, options: options,
            encodePrompt: { text in
                guard let rule else { throw AetherError.manifestInvalid(field: "krea2", reason: "required by \(Krea2Driver.id)") }
                return tokenizer.encode(rule.systemPrefix + text + rule.suffix, addSpecialTokens: addSpecialTokens)
            },
            bundle: bundle, load: ImageGraphs.validating(loader, manifest: manifest, contract: .krea2),
            fileBytes: { role in GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device) })
    }

    /// The pipeline over any prompt encoder and graph loader (tests drive it with fakes).
    init(
        manifest: Manifest, device: Device, filter: any ContentFilter, options: Options, encodePrompt: @escaping PromptEncoder,
        bundle: InstalledBundle? = nil, load: @escaping GraphResidency.Loader, fileBytes: @escaping @Sendable (String) -> Int64
    ) throws(AetherError) {
        try Self.requireMacOS(manifest, device)
        try DriverContract.krea2.validate(manifest)
        guard let schedule = manifest.schedule, schedule.kind == .flowMatchEuler else {
            throw .manifestInvalid(field: "schedule", reason: "\(Krea2Driver.id) requires a flowMatchEuler schedule")
        }
        guard schedule.guidance.kind == .none else {
            throw .manifestInvalid(field: "schedule.guidance", reason: "Krea 2 Turbo is distilled without guidance")
        }
        guard let latent = manifest.latent, latent.packing == .patch2x2ChannelMajor else {
            throw .manifestInvalid(field: "latent", reason: "\(Krea2Driver.id) requires patch2x2ChannelMajor latents")
        }
        // These exports have no fixed sizes, so the token bound is what keeps a request's canvas (and its unmeasured
        // memory) finite.
        guard latent.maxImageTokens != nil else {
            throw .manifestInvalid(field: "latent.maxImageTokens", reason: "required by \(Krea2Driver.id)")
        }
        guard let krea2 = manifest.krea2 else {
            throw .manifestInvalid(field: "krea2", reason: "required by \(Krea2Driver.id)")
        }
        guard manifest.tokenizer != nil else {
            throw .manifestInvalid(field: "tokenizer", reason: "\(Krea2Driver.id) renders the prompt with it")
        }
        wiring = try Wiring(manifest: manifest, krea2: krea2)
        self.manifest = manifest
        self.schedule = schedule
        self.latent = latent
        self.krea2 = krea2
        self.filter = filter
        self.device = device
        self.options = options
        self.encodePrompt = encodePrompt
        self.load = load
        admission = try options.admission(manifest: manifest, device: device, bundle: bundle, load: load, fileBytes: fileBytes)
    }

    /// Krea is qualified for macOS only in 0.1.0 (docs/INTEGRATION.md).
    static func requireMacOS(_ manifest: Manifest, _ device: Device) throws(AetherError) {
        guard device.platform == .macOS else {
            throw .unsupportedDevice(catalogID: manifest.id, platform: device.platform.rawValue, architecture: device.architecture)
        }
    }

    /// Generates one image, reviewed by the content filter before and after. Krea exports have no default size: pass
    /// `width` and `height` (positive multiples of `latent.pixelsPerToken`, at most `latent.maxImageTokens` tokens).
    public func generate(_ request: Request) async throws -> GeneratedImage {
        try Task.checkCancellation()
        // Validate the request, then the filter's view of the prompt, before waiting for a turn or the GPU.
        guard let size = request.size(default: latent.defaultSize) else {
            throw AetherError.generation(.invalidParameters("Krea 2 has no default size; pass width and height"))
        }
        let grid = try latent.tokenGrid(width: size.width, height: size.height)
        let table = try FlowMatchSchedule(schedule, steps: request.steps, imageTokens: grid.rows * grid.columns)
        let prompt = try Prompt(rendered: try encodePrompt(request.prompt), rule: krea2.prompt,
                                rows: krea2.conditioning.rowRange ?? 1...1)
        try await filter.requireAllowed(prompt: request.prompt)
        try Task.checkCancellation()

        let image = try await render(prompt, seed: request.seed, schedule: table, grid: grid, size: size, job: request.job)
        try await filter.requireAllowed(image: image, prompt: request.prompt)
        return image
    }

    /// The text encoder's conditioning rows for `prompt` (`[rows, taps · width]` row-major; the gate's text check).
    /// Loads the text encoder for the call and releases it; nothing is measured and the filter isn't consulted.
    func conditioningRows(for prompt: String) async throws -> [Float] {
        let prompt = try Prompt(rendered: try encodePrompt(prompt), rule: krea2.prompt, rows: krea2.conditioning.rowRange ?? 1...1)
        await turn()
        defer { endTurn() }
        let text = try await load(wiring.text.role)
        return try await encode(prompt, with: text, in: TensorArena()).readFloats()
    }

    /// Releases every graph (waiting for a running generation). The next generation loads them again.
    public func unload() async {
        await turn()
        admission.unload()
        endTurn()
    }

    /// Roles currently loaded (for tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    // MARK: - The generation

    private func render(
        _ prompt: Prompt, seed: UInt64, schedule: FlowMatchSchedule, grid: (rows: Int, columns: Int),
        size: (width: Int, height: Int), job: GPUJob?
    ) async throws -> GeneratedImage {
        // The GPU first, so a request's priority also orders it among this pipeline's own requests.
        let grant = try await options.gpu.admit(job: job)
        defer { grant.release() }
        await turn()
        defer { endTurn() }
        try Task.checkCancellation()
        // The prompt's row count shapes the text encoder's output and every segment's text carry, and those segments
        // load without the reshape hint, so each new count keeps its own specialization (ResidentAdmission.Settings).
        let residency = try admission.begin(profile: "\(size.width)x\(size.height)", shape: prompt.conditioningRows)
        do {
            let image = try await run(residency, prompt: prompt, seed: seed, schedule: schedule, grid: grid, size: size)
            admission.finish()
            return image
        } catch {
            admission.release()
            throw error
        }
    }

    /// The three phases. The denoisers and the decoder run in their own functions, so their graphs are referenced
    /// only by those functions' locals and the residency: when ``GraphResidency/endPhase(_:)`` releases them (per
    /// phase), nothing else keeps them alive into the next phase.
    private func run(
        _ residency: GraphResidency, prompt: Prompt, seed: UInt64, schedule: FlowMatchSchedule,
        grid: (rows: Int, columns: Int), size: (width: Int, height: Int)
    ) async throws -> GeneratedImage {
        let arena = TensorArena()

        // Prompt: the conditioning rows outlive the text encoder (it may be released before the denoisers load).
        residency.beginPhase(.perPrompt)
        let conditioning = try await encode(prompt, with: try await residency.graph(wiring.text.role), in: arena)
        residency.sample()
        residency.endPhase(.perPrompt)
        try Task.checkCancellation()

        // Steps: every segment loads, then each step writes the latents and sigma and runs the segments in order.
        residency.beginPhase(.perStep)
        let latents = try await denoise(
            conditioning: conditioning, textRows: prompt.conditioningRows, seed: seed, schedule: schedule, grid: grid,
            residency: residency, arena: arena)
        residency.endPhase(.perStep)
        try Task.checkCancellation()

        // Image: the packed tokens, raster order, are the decoder's [rows, columns, channels].
        residency.beginPhase(.perImage)
        let image = try await decode(latents, grid: grid, size: size, residency: residency, arena: arena)
        residency.endPhase(.perImage)
        // A cancellation during the decoder's call: don't report success for a request the caller has given up on.
        try Task.checkCancellation()
        return image
    }

    private func denoise(
        conditioning: TensorBuffer, textRows: Int, seed: UInt64, schedule: FlowMatchSchedule,
        grid: (rows: Int, columns: Int), residency: GraphResidency, arena: TensorArena
    ) async throws -> [Float] {
        let tokens = grid.rows * grid.columns, count = tokens * latent.channels
        let dims = Dimensions(textRows: textRows, imageTokens: tokens)
        var graphs: [any GraphBackend] = []
        for segment in wiring.segments { graphs.append(try await residency.graph(segment.role)) }
        let first = graphs[0]
        let sample = try first.makeBuffer(for: wiring.latents, role: .input, dimensions: [tokens], in: arena)
        let sigma = try first.makeBuffer(for: wiring.sigma, role: .input, dimensions: [], in: arena)
        let imageRotary = try first.makeBuffer(for: wiring.imageRotary, role: .input, dimensions: [tokens], in: arena)
        guard sample.shape == [tokens, latent.channels] else {
            throw AetherError.manifestMismatch(
                field: "latent.channels", expected: "denoiser latents [\(tokens), \(latent.channels)]", found: "\(sample.shape)")
        }
        try imageRotary.write(Krea2Rotary.image(rows: grid.rows, columns: grid.columns, rotary: krea2.rotary))
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
        return latents
    }

    private func decode(
        _ latents: [Float], grid: (rows: Int, columns: Int), size: (width: Int, height: Int), residency: GraphResidency,
        arena: TensorArena
    ) async throws -> GeneratedImage {
        let count = grid.rows * grid.columns * latent.channels
        let vae = try await residency.graph(wiring.decoder.role)
        let input = try vae.makeBuffer(for: wiring.decoder.latents, role: .input, dimensions: [grid.rows, grid.columns], in: arena)
        let output = try vae.makeBuffer(for: wiring.decoder.image, role: .output, dimensions: [size.height, size.width], in: arena)
        guard input.shape == [grid.rows, grid.columns, latent.channels], output.shape == [3, size.height, size.width] else {
            throw AetherError.manifestMismatch(
                field: "graphs.\(wiring.decoder.role)",
                expected: "[\(grid.rows), \(grid.columns), \(latent.channels)] → [3, \(size.height), \(size.width)]",
                found: "\(input.shape) → \(output.shape)")
        }
        try input.write(Float16.self, count: count) { Float16(latents[$0]) }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.decoder.latents, to: input)
        try bindings.bindOutput(wiring.decoder.image, to: output)
        try await vae.run(&bindings)
        residency.sample()
        return try GeneratedImage(width: size.width, height: size.height, planarRGB: try output.readFloats())
    }

    /// Runs the text encoder on `prompt`: its conditioning rows, `[n − dropLeadingRows, taps · width]`, in `arena`.
    private func encode(_ prompt: Prompt, with text: any GraphBackend, in arena: TensorArena) async throws -> TensorBuffer {
        let n = prompt.ids.count
        let ids = try text.makeBuffer(for: wiring.text.tokens, role: .input, dimensions: [n], in: arena)
        let positions = try text.makeBuffer(for: wiring.text.positions, role: .input, dimensions: [n], in: arena)
        let rows = try text.makeBuffer(for: wiring.text.conditioning, role: .output, dimensions: [prompt.conditioningRows], in: arena)
        guard rows.shape == [prompt.conditioningRows, krea2.conditioning.columns] else {
            throw AetherError.manifestMismatch(
                field: "krea2.conditioning", expected: "text rows [\(prompt.conditioningRows), \(krea2.conditioning.columns)]",
                found: "\(rows.shape)")
        }
        try ids.write(prompt.ids)
        try positions.write(Int32.self, count: n) { Int32($0) }
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(wiring.text.tokens, to: ids)
        try bindings.bindInput(wiring.text.positions, to: positions)
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

extension Krea2Pipeline {
    /// The text encoder's input: the rendered ids, unpadded. A rendering longer than `maxTokens` keeps its first
    /// `maxTokens − keepTrailingTokens` ids and its last `keepTrailingTokens`.
    struct Prompt: Sendable, Hashable {
        var ids: [Int32]
        var truncated: Bool
        /// Rows the text encoder emits: the ids after the dropped prefix.
        var conditioningRows: Int

        init(rendered: [Int32], rule: Manifest.Krea2.Prompt, rows: ClosedRange<Int>) throws(AetherError) {
            guard rendered.allSatisfy({ $0 >= 0 }) else {
                throw .generation(.invalidParameters("the rendered Krea 2 prompt has a negative token id"))
            }
            truncated = rendered.count > rule.maxTokens
            ids = truncated
                ? Array(rendered.prefix(rule.maxTokens - rule.keepTrailingTokens)) + rendered.suffix(rule.keepTrailingTokens)
                : rendered
            conditioningRows = ids.count - rule.dropLeadingRows
            guard rows.contains(conditioningRows) else {
                throw .generation(.invalidParameters(
                    "the rendered Krea 2 prompt has \(ids.count) tokens; its \(conditioningRows) conditioning rows must be in \(rows)"))
            }
        }
    }

    /// Which dynamic dimension each role's tensors have: text-side tensors the conditioning rows, image-side ones the
    /// generated token count.
    struct Dimensions {
        var textRows: Int
        var imageTokens: Int

        func of(_ role: String) -> [Int] {
            switch role {
            case "carryText", "conditioning": [textRows]
            case "carryImage", "imageRotary", "latents", "velocity": [imageTokens]
            default: []
            }
        }
    }

    /// The manifest's tensor names and how each segment input is fed, checked once at load: an input chained with
    /// `source.from` must name the text encoder's conditioning or an earlier segment's output, and only the last
    /// segment emits `velocity`.
    struct Wiring: Sendable {
        enum Source: Sendable, Hashable { case latents, conditioning, sigma, imageRotary, output(String) }
        struct Input: Sendable { var role: String; var name: String; var source: Source }
        struct Output: Sendable { var role: String; var name: String; var ref: String }
        struct Segment: Sendable { var role: String; var inputs: [Input]; var outputs: [Output] }

        var text: (role: String, tokens: String, positions: String, conditioning: String)
        var segments: [Segment]
        var decoder: (role: String, latents: String, image: String)
        /// The first segment's names for the host-fed inputs.
        var latents: String, sigma: String, imageRotary: String
        /// `role.name` of the last segment's velocity.
        var velocity: String

        init(manifest: Manifest, krea2: Manifest.Krea2) throws(AetherError) {
            let textGraph = manifest.graphs["textEncoder"]!, vae = manifest.graphs["vaeDecoder"]!
            let conditioningRef = "textEncoder.\(textGraph.outputs["conditioning"]!.name)"
            text = ("textEncoder", textGraph.inputs["tokens"]!.name, textGraph.inputs["positions"]!.name,
                    textGraph.outputs["conditioning"]!.name)
            decoder = ("vaeDecoder", vae.inputs["latents"]!.name, vae.outputs["image"]!.name)
            let contracted = Set(DriverContract.krea2.graphs.keys.filter { $0.hasPrefix("denoiser") })
            guard Set(krea2.segments) == contracted, krea2.segments.count == contracted.count,
                  krea2.segments == contracted.sorted() else {
                throw .manifestInvalid(field: "krea2.segments", reason: "must list \(contracted.sorted()) in call order")
            }
            var produced: Set<String> = []
            var segments: [Segment] = []
            for (index, role) in krea2.segments.enumerated() {
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
                        case "imageRotary": source = .imageRotary
                        default:
                            throw .manifestInvalid(field: "\(field).source", reason: "must be chained with source.from")
                        }
                    }
                    inputs.append(Input(role: inputRole, name: binding.name, source: source))
                }
                let outputs = graph.outputs.sorted(by: { $0.key < $1.key }).map {
                    Output(role: $0.key, name: $0.value.name, ref: "\(role).\($0.value.name)")
                }
                let emitsVelocity = outputs.contains { $0.role == "velocity" }
                guard emitsVelocity == (index == krea2.segments.count - 1) else {
                    throw .manifestInvalid(field: "krea2.segments", reason: "only the last segment, \(krea2.segments.last!), emits velocity")
                }
                produced.formUnion(outputs.map(\.ref))
                segments.append(Segment(role: role, inputs: inputs, outputs: outputs))
            }
            self.segments = segments
            let first = manifest.graphs[krea2.segments[0]]!
            guard case .from(conditioningRef)? = first.inputs["conditioning"]?.source else {
                throw .manifestInvalid(field: "graphs.\(krea2.segments[0]).inputs.conditioning.source", reason: "must be from \(conditioningRef)")
            }
            latents = first.inputs["latents"]!.name
            sigma = first.inputs["sigma"]!.name
            imageRotary = first.inputs["imageRotary"]!.name
            velocity = segments.last!.outputs.first { $0.role == "velocity" }!.ref
        }
    }
}

/// Krea 2's host image rotary table (MANIFEST.md §5.6 `krea2.rotary`): positions `(0, row, column)` in raster order,
/// `[rows · columns, 2, Σ axes]` float32 (cos rows, then sin rows) in the split-halves layout (pair `i` rotates
/// dimensions `i` and `i + Σ axes / 2`). Per axis of dimension `d`, pair `j` turns at `1 / θ^(2j / d)`; angles, cos and
/// sin are computed in Double and rounded once (0.0.x's `KreaRotary.image`).
enum Krea2Rotary {
    static func image(rows: Int, columns: Int, rotary: Manifest.Krea2.Rotary) -> [Float] {
        let width = rotary.dimension, half = width / 2
        var frequencies: [(axis: Int, value: Double)] = []
        for (axis, dimension) in rotary.axes.enumerated() {
            for j in stride(from: 0, to: dimension, by: 2) {
                frequencies.append((axis, 1 / pow(rotary.theta, Double(j) / Double(dimension))))
            }
        }
        var values = [Float](repeating: 0, count: rows * columns * 2 * width)
        for row in 0..<rows {
            for column in 0..<columns {
                let base = (row * columns + column) * 2 * width
                let position = [0, row, column]
                for (index, frequency) in frequencies.enumerated() {
                    let angle = Double(position[frequency.axis]) * frequency.value
                    let c = Float(cos(angle)), s = Float(sin(angle))
                    values[base + index] = c
                    values[base + index + half] = c
                    values[base + width + index] = s
                    values[base + width + index + half] = s
                }
            }
        }
        return values
    }
}
