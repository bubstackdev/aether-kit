public import AetherCore
public import AetherHost
public import AetherRuntime
import AetherText
import Foundation

/// `minimax-h3@1` (MANIFEST.md §5.10): MiniMax H3 (FL2VA weights with the Turbo LoRA) text to video with a stereo
/// soundtrack. macOS only in 0.1.0. Loading opens the bundle and its tokenizer; graphs load on the first generation.
public enum MiniMaxH3Driver: ModelDriver {
    public static let id: DriverID = .minimaxH3

    public static func load(_ bundle: InstalledBundle, options: LoadOptions) async throws -> MiniMaxH3Model {
        let manifest = try Manifest.load(fromBundle: bundle.url)
        try MiniMaxH3Pipeline.requireMacOS(manifest, options.device)
        try DriverContract.minimaxH3.validate(manifest)
        return MiniMaxH3Model(
            bundle: bundle, manifest: manifest, device: options.device,
            options: .init(cachePolicy: options.cachePolicy, cache: options.cache))
    }
}

/// An opened H3 bundle. Nothing is loaded yet; ``options`` start from the `LoadOptions` cache settings: set the
/// coordinator, GPU admission and ledger there, then make the ``MiniMaxH3Pipeline``.
public struct MiniMaxH3Model: Sendable {
    public let bundle: InstalledBundle
    public let manifest: Manifest
    public let device: Device
    public var options: VideoPipelineOptions

    public func pipeline() async throws -> MiniMaxH3Pipeline {
        try await MiniMaxH3Pipeline(bundle: bundle, manifest: manifest, device: device, options: options)
    }
}

extension Registry {
    /// The drivers AetherVideo links: `minimax-h3@1`.
    public static let video = Registry([MiniMaxH3Driver.self])
}

/// How the video pipeline loads its graphs and shares memory and the GPU: AetherHost's ``PipelineOptions``, whose absent
/// `memory` means ``MemoryBudget/videoGeneration`` here. Its ledger is seeded with the qualified profiles' footprints
/// (``MiniMaxH3Pipeline/qualifiedFootprints``).
public typealias VideoPipelineOptions = PipelineOptions

extension MemoryBudget {
    /// The video pipeline's default budget: three quarters of RAM (macOS; H3 is macOS-only in 0.1.0), about Metal's
    /// recommended working set, as for image generation.
    public static var videoGeneration: MemoryBudget { MemoryBudget(.fractionOfRAM(0.75)) }
}

/// One text-to-video-and-audio generation.
public struct VideoRequest: Sendable, Hashable {
    public var prompt: String
    public var seed: UInt64
    /// A qualified profile's key (`960x544x124`); absent, the manifest's first.
    public var profile: String?
    /// A recipe (`turbo8`, `turbo4`); absent, the manifest's default.
    public var recipe: String?
    /// Whether to decode the soundtrack.
    public var audio: Bool

    public init(prompt: String, seed: UInt64 = 0, profile: String? = nil, recipe: String? = nil, audio: Bool = true) {
        self.prompt = prompt
        self.seed = seed
        self.profile = profile
        self.recipe = recipe
        self.audio = audio
    }
}

/// A generated clip: interleaved RGB8 frames at 24 fps and, if asked for, the 32 kHz stereo soundtrack.
public struct GeneratedVideo: Sendable {
    public static let framesPerSecond = 24
    public let width: Int
    public let height: Int
    /// Each frame `width · height · 3` bytes, row-major RGB.
    public let frames: [[UInt8]]
    public let audio: GeneratedAudio?
    public var duration: Duration { .seconds(Double(frames.count) / Double(Self.framesPerSecond)) }
}

/// Progress through a generation.
public enum VideoProgress: Sendable, Hashable {
    case conditionerLayer(Int, of: Int)
    case step(Int, of: Int)
    case window(Int, of: Int)
}

/// One loaded MiniMax H3 model (DESIGN.md §6.8): the prompt's token embeddings through 50 conditioner layers chained on
/// the GPU (their weights streamed from the bundle into two banks), the refiner, then per step pack → the 50-block
/// chain (weights streamed the same way, each block's AdaLN table bound from the step's files) → heads → both scheduler
/// arms on the host, then the video decoded per temporal window in 256 px tiles and stitched, and the audio decoded.
///
/// - **Memory:** residency planned per profile before each generation from the ledger, which starts with the qualified
///   profiles' measured footprints; `capacity` before anything loads. Each phase's graphs, banks and buffers are
///   released when it ends when the plan says so. A profile pins the body's shapes (``ShapeProfile``); the modulation
///   is padded to the recipe's largest cardinality so the body specializes once per render.
/// - **Ordering, cancellation:** one generation at a time (later ones wait), cancellation between graph calls, and any
///   failure releases every graph.
/// - **Noise:** Kit's seeded stream in 0.0.x's order (`h3.video.noise`), so a seed gives the same clip as 0.0.x.
/// - FL2VA (keyframes) is not implemented yet: the bundle carries its graphs, but the driver presents text only.
public actor MiniMaxH3Pipeline {
    public typealias Options = VideoPipelineOptions

    public nonisolated let manifest: Manifest
    public nonisolated let h3: Manifest.H3
    public nonisolated let options: Options
    let device: Device
    let bundle: InstalledBundle?
    private let tokenize: @Sendable (String) -> [Int32]
    private let load: GraphResidency.Loader
    private let admission: ResidentAdmission
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws {
        try Self.requireMacOS(manifest, device)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        let contract = DriverContract.minimaxH3
        try self.init(
            manifest: manifest, device: device, bundle: bundle, options: options,
            tokenize: { tokenizer.encode($0, addSpecialTokens: false) },
            load: { role throws(AetherError) in
                let graph = try await loader(role)
                guard let spec = manifest.graphs[role], let graphContract = contract.graph(role) else {
                    throw .manifestInvalid(field: "graphs.\(role)", reason: "not a \(contract.driver) graph")
                }
                try SignatureValidator.validate(spec, phase: "step", at: "graphs.\(role)", against: graph.signature,
                                                contract: graphContract, tables: manifest.tables ?? [:])
                return graph
            })
    }

    init(
        manifest: Manifest, device: Device, bundle: InstalledBundle?, options: Options,
        tokenize: @escaping @Sendable (String) -> [Int32], load: @escaping GraphResidency.Loader
    ) throws(AetherError) {
        try Self.requireMacOS(manifest, device)
        try DriverContract.minimaxH3.validate(manifest)
        guard let h3 = manifest.h3 else { throw .manifestInvalid(field: "h3", reason: "required by \(MiniMaxH3Driver.id)") }
        guard manifest.tokenizer != nil else {
            throw .manifestInvalid(field: "tokenizer", reason: "\(MiniMaxH3Driver.id) tokenizes the prompt with it")
        }
        for role in ["conditionerLayer", "body"] where manifest.graphs[role]?.weights == nil {
            throw .manifestInvalid(field: "graphs.\(role).weights", reason: "\(MiniMaxH3Driver.id) streams this graph's weights")
        }
        self.manifest = manifest
        self.h3 = h3
        self.device = device
        self.bundle = bundle
        self.options = options
        self.tokenize = tokenize
        self.load = load
        let fileBytes = { [manifest, device] (role: String) in Self.fileBytes(role, bundle: bundle, manifest: manifest, device: device) }
        admission = options.admission(
            manifest: manifest, device: device, defaultMemory: .videoGeneration, load: load, fileBytes: fileBytes)
        Self.seedQualified(options.ledger, model: admission.modelKey)
    }

    /// H3 is qualified for macOS only in 0.1.0 (docs/INTEGRATION.md).
    static func requireMacOS(_ manifest: Manifest, _ device: Device) throws(AetherError) {
        guard device.platform == .macOS else {
            throw .unsupportedDevice(catalogID: manifest.id, platform: device.platform.rawValue, architecture: device.architecture)
        }
    }

    /// Generates one clip. `onProgress` is called as layers, steps and windows finish.
    public func generate(
        _ request: VideoRequest, onProgress: (@Sendable (VideoProgress) -> Void)? = nil
    ) async throws -> GeneratedVideo {
        try await generate(request, onProgress: onProgress, stepLimit: nil, trace: nil)
    }

    /// The generation, optionally stopped after `stepLimit` steps (no decode) and traced (the gates).
    func generate(
        _ request: VideoRequest, onProgress: (@Sendable (VideoProgress) -> Void)?, stepLimit: Int?, trace: Trace?
    ) async throws -> GeneratedVideo {
        var plan = try plan(request)
        plan.stepLimit = stepLimit
        await turn()
        defer { endTurn() }
        try Task.checkCancellation()
        let grant = try await options.gpu?.admit()
        defer { grant?.release() }
        let residency = try admission.begin(profile: plan.profile.key)
        do {
            let video = try await render(plan, residency: residency, onProgress: onProgress, trace: trace)
            admission.finish()
            return video
        } catch {
            admission.release()
            throw error
        }
    }

    /// Releases every graph (waiting for a running generation).
    public func unload() async {
        await turn()
        admission.release()
        endTurn()
    }

    /// Roles currently loaded (tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    // MARK: - Planning

    struct Plan: Sendable {
        var ids: [Int32]
        var profile: Manifest.H3.Profile
        var geometry: H3Geometry
        var recipe: Manifest.H3.Recipe
        /// The recipe's largest step cardinality: the modulation is padded to it.
        var cardinality: Int
        var seed: UInt64
        var audio: Bool
        /// Gate only: stop after this many steps.
        var stepLimit: Int?
    }

    /// Validates the request before waiting for a turn: the profile, recipe and prompt length, and the packed rows.
    func plan(_ request: VideoRequest) throws(AetherError) -> Plan {
        guard let profile = request.profile.map({ h3.profile($0) }) ?? h3.profiles.first else {
            throw .generation(.invalidParameters(
                "no qualified profile '\(request.profile ?? "")'; the bundle has \(h3.profiles.map(\.key).joined(separator: ", "))"))
        }
        let name = (request.recipe ?? h3.defaultRecipe) + "/t2va"
        guard let recipe = h3.recipes[name] else {
            throw .generation(.invalidParameters("no recipe '\(name)'; the bundle has \(h3.recipes.keys.sorted().joined(separator: ", "))"))
        }
        let ids = tokenize(request.prompt)
        let limits = h3.limits.textTokens[0]...h3.limits.textTokens[1]
        guard limits.contains(ids.count) else {
            throw .generation(.invalidParameters("the prompt has \(ids.count) tokens; H3 takes \(limits)"))
        }
        let geometry = H3Geometry(profile)
        let rows = ids.count + geometry.audioRows + geometry.videoRows
        guard (h3.limits.rows[0]...h3.limits.rows[1]).contains(rows) else {
            throw .generation(.invalidParameters("\(rows) packed rows are outside \(h3.limits.rows)"))
        }
        let cardinality = recipe.steps.map(\.timesteps.count).max() ?? 1
        return Plan(ids: ids, profile: profile, geometry: geometry, recipe: recipe, cardinality: cardinality,
                    seed: request.seed, audio: request.audio, stepLimit: nil)
    }

    /// An unmeasured graph's stand-in: its file, plus two banks of its largest weight block (they dwarf the graph).
    static func fileBytes(_ role: String, bundle: InstalledBundle?, manifest: Manifest, device: Device) -> Int64 {
        guard let bundle else { return 0 }
        var bytes = GraphResidency.fileBytes(bundle: bundle, manifest: manifest, role: role, device: device)
        if let weights = manifest.graphs[role]?.weights, let url = bundle.url(of: weights.table),
           let table = try? WeightTable(contentsOf: url) {
            bytes += 2 * (table.blocks.map { $0.values.reduce(0) { $0 + $1.bytes } }.max() ?? 0)
        }
        return bytes
    }

    private func turn() async {
        if busy { await withCheckedContinuation { waiters.append($0) } } else { busy = true }
    }

    private func endTurn() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }

    // MARK: - The generation

    /// Gate-only boundary tensors.
    final class Trace: @unchecked Sendable {
        var ids: [Int32] = []
        /// The tap `[1, T, 5120]` bf16 bits.
        var tap: [UInt8] = []
        /// After each step: the head velocities and the advanced latents.
        var velocities: [(video: [Float], audio: [Float])] = []
        var latents: [(video: [Float], audio: [Float])] = []
    }

    func render(
        _ plan: Plan, residency: GraphResidency, onProgress: (@Sendable (VideoProgress) -> Void)?, trace: Trace?
    ) async throws -> GeneratedVideo {
        let g = plan.geometry
        trace?.ids = plan.ids
        // Prompt: conditioner chain, then the refiner. The refined text outlives the phase (a host buffer).
        residency.beginPhase(.perPrompt)
        let host = TensorArena()
        let refined = try await conditionAndRefine(plan, residency: residency, host: host, onProgress: onProgress, trace: trace)
        residency.sample()
        residency.endPhase(.perPrompt)
        try Task.checkCancellation()

        residency.beginPhase(.perStep)
        var latents = try g.initialLatents(seed: plan.seed)
        try await denoise(plan, latents: &latents, refined: refined, residency: residency, host: host,
                          onProgress: onProgress, trace: trace)
        residency.sample()
        residency.endPhase(.perStep)
        try Task.checkCancellation()

        residency.beginPhase(.perImage)
        if let limit = plan.stepLimit, limit < plan.recipe.steps.count {
            residency.endPhase(.perImage)
            return GeneratedVideo(width: g.width, height: g.height, frames: [], audio: nil)
        }
        let frames = try await decodeVideo(latents.video, geometry: g, residency: residency, host: host, onProgress: onProgress)
        let audio = plan.audio ? try await decodeAudio(latents.audio, plan: plan, residency: residency, host: host) : nil
        residency.sample()
        residency.endPhase(.perImage)
        return GeneratedVideo(width: g.width, height: g.height, frames: frames, audio: audio)
    }

    // MARK: Prompt

    private func url(_ path: AssetPath) throws(AetherError) -> URL {
        guard let url = bundle?.url(of: path) else { throw .manifestInvalid(field: "h3", reason: "'\(path)' is not in the bundle") }
        return url
    }

    private func names(_ role: String) -> (inputs: [String: String], outputs: [String: String]) {
        let graph = manifest.graphs[role]!
        return (graph.inputs.mapValues(\.name), graph.outputs.mapValues(\.name))
    }

    private func weightBanks(_ role: String, graph: any GraphBackend, arena: TensorArena) throws(AetherError) -> WeightBanks {
        let table = try WeightTable(contentsOf: try url(manifest.graphs[role]!.weights!.table))
        return try WeightBanks(table: table, graph: graph, arena: arena, resolve: { [bundle] in bundle?.url(of: $0) })
    }

    private func conditionAndRefine(
        _ plan: Plan, residency: GraphResidency, host: TensorArena, onProgress: (@Sendable (VideoProgress) -> Void)?, trace: Trace?
    ) async throws -> TensorBuffer {
        let c = h3.conditioner, t = plan.ids.count, width = c.embedding.width
        guard let metal = TensorArena.metal() else { throw AetherError.unsupportedDevice(catalogID: manifest.id, platform: "macOS", architecture: "no GPU") }
        guard let layer = try await residency.graph("conditionerLayer") as? Graph else {
            throw AetherError.manifestInvalid(field: "graphs.conditionerLayer", reason: "the conditioner chains on the GPU")
        }
        let n = names("conditionerLayer")
        let chain = try GraphChain(graph: layer, link: (n.inputs["hidden"]!, n.outputs["hidden"]!), dimensions: [t], arena: metal)
        let banks = try weightBanks("conditionerLayer", graph: layer, arena: metal)
        guard banks.table.blockCount == c.layers else {
            throw AetherError.manifestMismatch(field: "graphs.conditionerLayer.weights", expected: "\(c.layers) layers", found: "\(banks.table.blockCount)")
        }
        let cos = try layer.makeBuffer(for: n.inputs["cos"]!, role: .input, dimensions: [t], in: metal)
        let sin = try layer.makeBuffer(for: n.inputs["sin"]!, role: .input, dimensions: [t], in: metal)
        let rowBytes = cos.storageByteCount / t
        try writePrefix(c.ropeTables.cos, bytes: t * rowBytes, into: cos)
        try writePrefix(c.ropeTables.sin, bytes: t * rowBytes, into: sin)
        // The prompt's embedding rows, gathered from the row source (bf16 bits).
        let embedding = try FileHandle(forReadingFrom: try url(c.embedding.file))
        defer { try? embedding.close() }
        var rows = [UInt8]()
        rows.reserveCapacity(t * width * 2)
        for id in plan.ids {
            guard (0..<Int32(c.embedding.rows)).contains(id) else {
                throw AetherError.generation(.invalidParameters("token id \(id) is outside the embedding's \(c.embedding.rows) rows"))
            }
            try embedding.seek(toOffset: UInt64(id) * UInt64(width * 2))
            guard let row = try embedding.read(upToCount: width * 2), row.count == width * 2 else {
                throw AetherError.integrity(file: c.embedding.file.description, expected: "row \(id)", actual: "a short read")
            }
            rows.append(contentsOf: row)
        }
        try chain.head.writeBytes(rows)
        try await chain.run { chain in
            try banks.load(block: 0, into: 0, chain: chain)
            for index in 0..<c.layers {
                try await chain.call(layer, as: "conditionerLayer") { bindings throws(AetherError) in
                    try bindings.bindInput(n.inputs["cos"]!, to: cos)
                    try bindings.bindInput(n.inputs["sin"]!, to: sin)
                    try banks.bind(bank: index % 2, into: &bindings)
                }
                if index + 1 < c.layers { try banks.load(block: index + 1, into: (index + 1) % 2, chain: chain) }
                await chain.synchronize()
                residency.sample()
                onProgress?(.conditionerLayer(index + 1, of: c.layers))
            }
        }
        try Task.checkCancellation()

        // The refiner: one block of weights, run on the host arena.
        let text = try await residency.graph("prepareText")
        let r = names("prepareText")
        let input = try text.makeBuffer(for: r.inputs["text"]!, role: .input, dimensions: [t], in: host)
        try input.copyBytes(from: chain.head)
        if let trace { trace.tap = try input.readBytes() }
        let refined = try text.makeBuffer(for: r.outputs["refined"]!, role: .output, dimensions: [t], in: host)
        let refiner = try weightBanks("prepareText", graph: text, arena: host)
        try refiner.load(block: 0, into: 0)
        var bindings = Bindings(arena: host)
        try bindings.bindInput(r.inputs["text"]!, to: input)
        try refiner.bind(bank: 0, into: &bindings)
        try bindings.bindOutput(r.outputs["refined"]!, to: refined)
        try await text.run(&bindings)
        return refined
    }

    private func writePrefix(_ path: AssetPath, bytes count: Int, into buffer: TensorBuffer) throws(AetherError) {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: try url(path)) } catch let error as AetherError { throw error } catch { throw .runtime(underlying: error) }
        defer { try? handle.close() }
        let data: Data?
        do { data = try handle.read(upToCount: count) } catch { throw .runtime(underlying: error) }
        guard let data, data.count == count else { throw .integrity(file: path.description, expected: "\(count) bytes", actual: "a short read") }
        try buffer.writeBytes([UInt8](data))
    }

    private func readFile(_ path: AssetPath) throws(AetherError) -> [UInt8] {
        do { return [UInt8](try Data(contentsOf: try url(path))) } catch let error as AetherError { throw error } catch { throw .runtime(underlying: error) }
    }

    // MARK: Steps

    private func denoise(
        _ plan: Plan, latents: inout (video: [Float], audio: [Float]), refined: TensorBuffer, residency: GraphResidency,
        host: TensorArena, onProgress: (@Sendable (VideoProgress) -> Void)?, trace: Trace?
    ) async throws {
        let g = plan.geometry, t = plan.ids.count
        let layout = H3Layout(textTokens: t, geometry: g)
        let rows = layout.rows, c = plan.cardinality, hidden = 5376
        guard let metal = TensorArena.metal() else { throw AetherError.unsupportedDevice(catalogID: manifest.id, platform: "macOS", architecture: "no GPU") }
        let pack = try await residency.graph("preparePack")
        guard let body = try await residency.graph("body") as? Graph else {
            throw AetherError.manifestInvalid(field: "graphs.body", reason: "the body chains on the GPU")
        }
        let heads = try await residency.graph("heads")
        let p = names("preparePack"), b = names("body"), h = names("heads")

        // Host buffers: the pack's inputs and output (also the heads' hidden input), and the heads' others.
        let videoIn = try pack.makeBuffer(for: p.inputs["video"]!, role: .input, dimensions: [g.videoRows], in: host)
        let audioIn = try pack.makeBuffer(for: p.inputs["audio"]!, role: .input, dimensions: [g.audioRows], in: host)
        let gather = try pack.makeBuffer(for: p.inputs["gather"]!, role: .input, dimensions: [rows], in: host)
        let packed = try pack.makeBuffer(for: p.outputs["hidden"]!, role: .output, dimensions: [rows], in: host)
        try gather.write(layout.gather)
        let temb = try heads.makeBuffer(for: h.inputs["temb"]!, role: .input, dimensions: [c], in: host)
        let indices = try heads.makeBuffer(for: h.inputs["indices"]!, role: .input, dimensions: [rows], in: host)
        let videoIndices = try heads.makeBuffer(for: h.inputs["videoIndices"]!, role: .input, dimensions: [g.videoRows], in: host)
        let audioIndices = try heads.makeBuffer(for: h.inputs["audioIndices"]!, role: .input, dimensions: [g.audioRows], in: host)
        try videoIndices.write(layout.videoIndices)
        try audioIndices.write(layout.audioIndices)
        let videoOut = try heads.makeBuffer(for: h.outputs["video"]!, role: .output, dimensions: [g.videoRows], in: host)
        let audioOut = try heads.makeBuffer(for: h.outputs["audio"]!, role: .output, dimensions: [g.audioRows], in: host)

        // The body chain on the GPU: rotary tables, tags and one AdaLN table per block, written before each step's run.
        let profile = ShapeProfile(name: plan.profile.key, shapes: ["body": [
            b.inputs["hidden"]!: [[1, rows, hidden]], b.inputs["table"]!: [[6, 3 * c, hidden]]]])
        let chain = try GraphChain(graph: body, link: (b.inputs["hidden"]!, b.outputs["hidden"]!), dimensions: [rows],
                                   arena: metal, profile: profile)
        let banks = try weightBanks("body", graph: body, arena: metal)
        let blocks = banks.table.blockCount
        let rotary = layout.rotary()
        let cos = try body.makeBuffer(for: b.inputs["cos"]!, role: .input, dimensions: [rows], in: metal)
        let sin = try body.makeBuffer(for: b.inputs["sin"]!, role: .input, dimensions: [rows], in: metal)
        let tags = try body.makeBuffer(for: b.inputs["tags"]!, role: .input, dimensions: [rows], in: metal)
        try cos.write(rotary.cos)
        try sin.write(rotary.sin)
        var tables: [TensorBuffer] = []
        for _ in 0..<blocks { tables.append(try body.makeBuffer(for: b.inputs["table"]!, role: .input, dimensions: [3 * c], in: metal)) }

        let video = H3Arm(plan.recipe.video), audio = H3Arm(plan.recipe.audio)
        let steps = min(plan.recipe.steps.count, plan.stepLimit ?? .max)
        for (index, step) in plan.recipe.steps.prefix(steps).enumerated() {
            try Task.checkCancellation()
            let row = layout.timesteps(video: video.timesteps[index], audio: audio.timesteps[index])
            guard row.values.map(\.bitPattern) == step.timesteps.map(\.bitPattern) else {
                throw AetherError.manifestMismatch(field: "h3.recipes.steps.\(index).timesteps", expected: "\(row.values)", found: "\(step.timesteps)")
            }
            let k = row.values.count
            try tags.write(row.adaln)
            try indices.write(row.indices)
            for block in 0..<blocks {
                let padded = H3Padding.table(try readFile(try step.table(block)), k: k, cardinality: c, width: hidden)
                try tables[block].writeBytes(padded)
            }
            let embedding = try readFile(step.temb).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            try temb.write(H3Padding.embedding(embedding, k: k, cardinality: c))

            try videoIn.write(latents.video)
            try audioIn.write(latents.audio)
            var bindings = Bindings(arena: host)
            try bindings.bindInput(p.inputs["video"]!, to: videoIn)
            try bindings.bindInput(p.inputs["audio"]!, to: audioIn)
            try bindings.bindInput(p.inputs["text"]!, to: refined)
            try bindings.bindInput(p.inputs["gather"]!, to: gather)
            try bindings.bindOutput(p.outputs["hidden"]!, to: packed)
            try await pack.run(&bindings)
            try chain.reset()
            try chain.head.copyBytes(from: packed)

            try await chain.run { chain in
                try banks.load(block: 0, into: 0, chain: chain)
                for block in 0..<blocks {
                    try await chain.call(body, as: "body") { bindings throws(AetherError) in
                        try bindings.bindInput(b.inputs["table"]!, to: tables[block])
                        try bindings.bindInput(b.inputs["tags"]!, to: tags)
                        try bindings.bindInput(b.inputs["cos"]!, to: cos)
                        try bindings.bindInput(b.inputs["sin"]!, to: sin)
                        try banks.bind(bank: block % 2, into: &bindings)
                    }
                    if block + 1 < blocks { try banks.load(block: block + 1, into: (block + 1) % 2, chain: chain) }
                    await chain.synchronize()
                    residency.sample()
                }
            }
            try Task.checkCancellation()

            try packed.copyBytes(from: chain.head)
            var head = Bindings(arena: host)
            try head.bindInput(h.inputs["hidden"]!, to: packed)
            try head.bindInput(h.inputs["temb"]!, to: temb)
            try head.bindInput(h.inputs["indices"]!, to: indices)
            try head.bindInput(h.inputs["videoIndices"]!, to: videoIndices)
            try head.bindInput(h.inputs["audioIndices"]!, to: audioIndices)
            try head.bindOutput(h.outputs["video"]!, to: videoOut)
            try head.bindOutput(h.outputs["audio"]!, to: audioOut)
            try await heads.run(&head)
            let vv = try videoOut.readFloats(), va = try audioOut.readFloats()
            video.advance(step: index, sample: &latents.video, velocity: vv)
            audio.advance(step: index, sample: &latents.audio, velocity: va)
            guard latents.video.allSatisfy(\.isFinite), latents.audio.allSatisfy(\.isFinite) else {
                throw AetherError.tensor(name: "latents", reason: .unsupported("a non-finite value after step \(index)"))
            }
            trace?.velocities.append((vv, va))
            trace?.latents.append(latents)
            residency.sample()
            onProgress?(.step(index + 1, of: steps))
        }
    }

    // MARK: Decode

    private func decodeVideo(
        _ packed: [Float], geometry g: H3Geometry, residency: GraphResidency, host: TensorArena,
        onProgress: (@Sendable (VideoProgress) -> Void)?
    ) async throws -> [[UInt8]] {
        let decoder = try await residency.graph("videoDecoder")
        let d = names("videoDecoder")
        let rows = try decoder.makeBuffer(for: d.inputs["rows"]!, role: .input, dimensions: [], in: host)
        let pixels = try decoder.makeBuffer(for: d.outputs["pixels"]!, role: .output, dimensions: [], in: host)
        guard rows.shape == [7 * 64, 96], pixels.shape == [1, 3, 22, H3Tiling.side, H3Tiling.side] else {
            throw AetherError.manifestMismatch(field: "graphs.videoDecoder", expected: "[448, 96] → [1, 3, 22, 256, 256]",
                                               found: "\(rows.shape) → \(pixels.shape)")
        }
        let tiling = H3Tiling(width: g.width, height: g.height)
        var temporal = H3TemporalStitcher(width: g.width, height: g.height)
        var frames: [[UInt8]] = []
        for window in 0..<g.videoWindows {
            let windowRows = g.windowRows(packed, index: window)
            var spatial = H3SpatialStitcher(tiling: tiling, width: g.width, height: g.height)
            for tile in 0..<tiling.count {
                try Task.checkCancellation()
                try rows.write(tiling.extract(windowRows, width: g.width, height: g.height, index: tile))
                var bindings = Bindings(arena: host)
                try bindings.bindInput(d.inputs["rows"]!, to: rows)
                try bindings.bindOutput(d.outputs["pixels"]!, to: pixels)
                try await decoder.run(&bindings)
                spatial.push(try pixels.readFloats())
            }
            residency.sample()
            frames += H3Pixels.rgb8(temporal.push(spatial.output), width: g.width, height: g.height)
            onProgress?(.window(window + 1, of: g.videoWindows))
        }
        frames += H3Pixels.rgb8(temporal.finish(), width: g.width, height: g.height)
        guard frames.count == g.frames else {
            throw AetherError.tensor(name: "video", reason: .unsupported("decoded \(frames.count) frames, not \(g.frames)"))
        }
        return frames
    }

    private func decodeAudio(
        _ latents: [Float], plan: Plan, residency: GraphResidency, host: TensorArena
    ) async throws -> GeneratedAudio {
        let role = plan.profile.audioDecoder
        let decoder = try await residency.graph(role)
        let a = names(role)
        let rows = try decoder.makeBuffer(for: a.inputs["rows"]!, role: .input, dimensions: [], in: host)
        let pcm = try decoder.makeBuffer(for: a.outputs["pcm"]!, role: .output, dimensions: [], in: host)
        let samples = h3.audio.hop * plan.geometry.audioLatents
        guard rows.shape == [2, plan.geometry.audioLatents, 32], pcm.shape == [2, samples] else {
            throw AetherError.manifestMismatch(field: "graphs.\(role)", expected: "[2, Na, 32] → [2, hop · Na]", found: "\(rows.shape) → \(pcm.shape)")
        }
        // Channel-major rows: [1, 2 Na, 32] and [2, Na, 32] are the same values.
        try rows.write(latents)
        var bindings = Bindings(arena: host)
        try bindings.bindInput(a.inputs["rows"]!, to: rows)
        try bindings.bindOutput(a.outputs["pcm"]!, to: pcm)
        try await decoder.run(&bindings)
        let values = try pcm.readFloats()
        return try GeneratedAudio(channels: [Array(values[..<samples]), Array(values[samples...])], sampleRate: h3.audio.sampleRate)
    }
}

extension MiniMaxH3Pipeline {
    /// The qualified profiles' measured phase footprints (the memory gate, `H3GateTests.memory`; docs/INTEGRATION.md),
    /// seeded into the ledger so the first generation plans from measured bytes, not files. Keyed by profile.
    /// Measured 2026-11-06 on an M5 Max (64 GB), turbo8 T2VA: peaks 4,261 / 9,893 / 1,445 MiB, each with 0.0.x's
    /// 256 MiB sampling margin well inside the 18 GiB bar for this region. 544×960 has the same rows and is unmeasured.
    public static let qualifiedFootprints: [String: [Manifest.Residency: PhaseFootprint]] = [
        "960x544x124": [
            .perPrompt: PhaseFootprint(peakBytes: 4261 << 20, heldBytes: 2395 << 20),
            .perStep: PhaseFootprint(peakBytes: 9893 << 20, heldBytes: 8387 << 20),
            .perImage: PhaseFootprint(peakBytes: 1445 << 20, heldBytes: 1151 << 20),
        ],
    ]

    /// Seeds `ledger` under `model`, the pipeline's ledger key (``ResidentAdmission/modelKey``: the manifest id, the
    /// weights' source revision and the device), so the seeded entries are the ones its plans read.
    static func seedQualified(_ ledger: FootprintLedger, model: String) {
        for (profile, phases) in qualifiedFootprints {
            for (phase, footprint) in phases {
                let key = FootprintLedger.Key(model: model, profile: profile, phase: phase)
                if ledger.footprint(key) == nil { ledger.record(key, footprint) }
            }
        }
    }
}
