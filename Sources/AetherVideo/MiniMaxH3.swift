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
    /// How the pipeline admits a profile it has no measured envelope for (default: it refuses it).
    public var admissionPolicy: VideoAdmissionPolicy = .measuredOnly

    public func pipeline() async throws -> MiniMaxH3Pipeline {
        try await MiniMaxH3Pipeline(bundle: bundle, manifest: manifest, device: device, options: options, admissionPolicy: admissionPolicy)
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

/// One text-to-video-and-audio generation, optionally anchored to keyframes (the FL2VA task).
public struct VideoRequest: Sendable, Hashable {
    public var prompt: String
    public var seed: UInt64
    /// A qualified profile's key (`960x544x124`); absent, the manifest's first.
    public var profile: String?
    /// A recipe (`turbo8`, `turbo4`); absent, the manifest's default.
    public var recipe: String?
    /// Whether to decode the soundtrack.
    public var audio: Bool
    /// The request's GPU priority (absent: `.medium`). Keep the job to change it while the request waits.
    public var job: GPUJob?
    /// The clip's first frame, last frame or both, each RGB8 at the profile's canvas; absent, text to video.
    public var keyframes: VideoKeyframes?

    public init(
        prompt: String, seed: UInt64 = 0, profile: String? = nil, recipe: String? = nil, audio: Bool = true,
        job: GPUJob? = nil
    ) {
        self.prompt = prompt
        self.seed = seed
        self.profile = profile
        self.recipe = recipe
        self.audio = audio
        self.job = job
        keyframes = nil
    }

    /// A clip anchored to `keyframes` (the FL2VA task).
    public init(
        prompt: String, keyframes: VideoKeyframes?, seed: UInt64 = 0, profile: String? = nil, recipe: String? = nil,
        audio: Bool = true, job: GPUJob? = nil
    ) {
        self.init(prompt: prompt, seed: seed, profile: profile, recipe: recipe, audio: audio, job: job)
        self.keyframes = keyframes
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
/// - **Keyframes (FL2VA):** a request with ``VideoRequest/keyframes`` encodes each keyframe with the tiled VAE encoder
///   (`videoEncoderTile` per 256 px tile, then the canvas's `videoEncoderTail<W>x<H>`) into condition rows held at
///   t = 0.999 through every step, and presents it to the conditioner as Diffusers does: a `"<Picture i>: "` label and
///   a vision block whose rows come from the canvas's `conditionerVision<W>x<H>` tower, at Qwen3-VL's M-RoPE positions,
///   with DeepStack added after the first layers. First, last, or first and last, in that packed order.
public actor MiniMaxH3Pipeline {
    public typealias Options = VideoPipelineOptions

    public nonisolated let manifest: Manifest
    public nonisolated let h3: Manifest.H3
    public nonisolated let options: Options
    /// The export's FL2VA rules; nil when it doesn't carry them.
    nonisolated let keyframeRules: H3KeyframeRules?
    let device: Device
    let bundle: InstalledBundle?
    private let tokenize: @Sendable (String) -> [Int32]
    private let load: GraphResidency.Loader
    private let admission: ResidentAdmission
    /// One request at a time, in arrival order (after the GPU grant).
    private let gate = SerialGate()
    /// How generations are admitted.
    public nonisolated let admissionPolicy: VideoAdmissionPolicy
    /// The measured envelopes that hold for this export and device, by admission key (``measuredEnvelopes(manifest:device:)``).
    public nonisolated let envelopes: [VideoAdmissionKey: VideoMemoryEnvelope]

    public init(bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options()) async throws {
        try await self.init(bundle: bundle, manifest: manifest, device: device, options: options, admissionPolicy: .measuredOnly)
    }

    /// A pipeline admitting generations under `admissionPolicy`.
    public init(
        bundle: InstalledBundle, manifest: Manifest, device: Device, options: Options = Options(),
        admissionPolicy: VideoAdmissionPolicy
    ) async throws {
        try Self.requireMacOS(manifest, device)
        let tokenizer = try await TokenizerAdapter(bundle: bundle, manifest: manifest, forGeneration: false)
        let loader = GraphResidency.bundleLoader(
            bundle: bundle, manifest: manifest, device: device, cachePolicy: options.cachePolicy, cache: options.cache)
        let contract = DriverContract.minimaxH3
        try self.init(
            manifest: manifest, device: device, bundle: bundle, options: options, admissionPolicy: admissionPolicy,
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
        admissionPolicy: VideoAdmissionPolicy = .measuredOnly,
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
        keyframeRules = try H3KeyframeRules(h3)
        self.device = device
        self.bundle = bundle
        self.options = options
        self.admissionPolicy = admissionPolicy
        envelopes = Self.measuredEnvelopes(manifest: manifest, device: device)
        self.tokenize = tokenize
        self.load = load
        let fileBytes = { [manifest, device] (role: String) in Self.fileBytes(role, bundle: bundle, manifest: manifest, device: device) }
        admission = try options.admission(
            manifest: manifest, device: device, defaultMemory: .videoGeneration, bundle: bundle, load: load,
            fileBytes: fileBytes)
        // The gate measures from nothing: a seeded entry would be kept as the larger value of what it measures.
        if admissionPolicy == .measuredOnly {
            // The manifest's envelopes first: they win over the qualified footprints for a key both measured (the
            // G4 export keeps its digest when envelopes are recorded into it).
            Self.seed(options.ledger, model: admission.modelKey, envelopes: envelopes)
            Self.seedQualified(options.ledger, model: admission.modelKey, manifest: manifest, device: device)
        }
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
        try admit(plan)
        plan.stepLimit = stepLimit
        // The GPU first, so a request's priority also orders it among this pipeline's own requests.
        let grant = try await options.gpu.admit(job: request.job)
        defer { grant.release() }
        return try await gate.withTurn {
            let residency = try admission.begin(profile: plan.admissionKey.description)
            do {
                let video = try await render(plan, residency: residency, onProgress: onProgress, trace: trace)
                admission.finish()
                return video
            } catch {
                admission.release()
                throw error
            }
        }
    }

    /// Releases every graph (waiting for a running generation).
    public func unload() async {
        await gate.withTurnIgnoringCancellation {
            admission.unload()
        }
    }

    /// Roles currently loaded (tests and diagnostics).
    public var residentRoles: [String] { admission.residentRoles }

    // MARK: - Planning

    struct Plan: Sendable {
        /// The prompt's ids.
        var ids: [Int32]
        var profile: Manifest.H3.Profile
        var geometry: H3Geometry
        var recipe: Manifest.H3.Recipe
        /// The recipe's name (`turbo8`), without its task.
        var recipeName: String = ""
        /// The recipe's largest step cardinality: the modulation is padded to it.
        var cardinality: Int
        var seed: UInt64
        var audio: Bool
        /// The keyframes in packed order and their anchors (FL2VA), with the presentation the conditioner sees.
        var keyframes: [VideoKeyframe] = []
        var anchors: [H3Anchor] = []
        var presentation: H3Presentation?
        /// Gate only: stop after this many steps.
        var stepLimit: Int?

        /// The conditioner's ids and the text rows' tags: the presentation's, or the prompt's (all text).
        var conditioningIDs: [Int32] { presentation?.ids ?? ids }
        var textTags: [Int32] { presentation?.tags ?? [Int32](repeating: H3KeyframeRules.textTag, count: ids.count) }
        /// The ledger key the generation is admitted under: a keyframed clip has its own footprints.
        var admissionKey: VideoAdmissionKey { VideoAdmissionKey(profile: profile.key, keyframeCount: anchors.count) }
    }

    /// Validates the request before waiting for a turn: the profile, recipe, keyframes and prompt length, and the
    /// packed rows.
    func plan(_ request: VideoRequest) throws(AetherError) -> Plan {
        guard let profile = request.profile.map({ h3.profile($0) }) ?? h3.profiles.first else {
            throw .generation(.invalidParameters(
                "no qualified profile '\(request.profile ?? "")'; the bundle has \(h3.profiles.map(\.key).joined(separator: ", "))"))
        }
        let keyframes = request.keyframes?.frames ?? []
        let recipeName = request.recipe ?? h3.defaultRecipe
        let name = recipeName + (keyframes.isEmpty ? "/t2va" : "/fl2va")
        guard let recipe = h3.recipes[name] else {
            throw .generation(.invalidParameters("no recipe '\(name)'; the bundle has \(h3.recipes.keys.sorted().joined(separator: ", "))"))
        }
        let ids = tokenize(request.prompt)
        let geometry = H3Geometry(profile)
        var presentation: H3Presentation?
        if !keyframes.isEmpty {
            let canvas = "\(profile.width)x\(profile.height)"
            guard let rules = keyframeRules, let encoder = profile.keyframeEncoder, manifest.graphs[encoder] != nil,
                  manifest.graphs["videoEncoderTile"] != nil, manifest.graphs[Self.visionRole(canvas)] != nil else {
                throw .unsupportedOperation("this H3 export has no keyframe encoder and vision tower for \(canvas)")
            }
            for frame in keyframes where (frame.width, frame.height) != (profile.width, profile.height) {
                throw .generation(.invalidParameters("a \(frame.width)x\(frame.height) keyframe for the \(canvas) canvas; keyframes are not resized"))
            }
            presentation = try H3Presentation(prompt: ids, canvases: keyframes.map { ($0.width, $0.height) }, rules: rules,
                                              tokenize: { [tokenize] label throws(AetherError) in tokenize(label) })
        }
        let textTokens = presentation?.count ?? ids.count
        let limits = h3.limits.textTokens[0]...h3.limits.textTokens[1]
        guard limits.contains(textTokens) else {
            let presented = textTokens == ids.count ? "" : " (\(textTokens) with the keyframes' presentation)"
            throw .generation(.invalidParameters("the prompt has \(ids.count) tokens\(presented); H3 takes \(limits)"))
        }
        let rows = textTokens + keyframes.count * geometry.rowsPerLatentFrame + geometry.audioRows + geometry.videoRows
        guard (h3.limits.rows[0]...h3.limits.rows[1]).contains(rows) else {
            throw .generation(.invalidParameters("\(rows) packed rows are outside \(h3.limits.rows)"))
        }
        let cardinality = recipe.steps.map(\.timesteps.count).max() ?? 1
        return Plan(ids: ids, profile: profile, geometry: geometry, recipe: recipe, recipeName: recipeName, cardinality: cardinality,
                    seed: request.seed, audio: request.audio, keyframes: keyframes,
                    anchors: request.keyframes?.anchors ?? [], presentation: presentation, stepLimit: nil)
    }

    /// Admission before anything waits or loads: the profile offers the recipe, the recipe's modulation is inside the
    /// profile's shapes, and (``VideoAdmissionPolicy/measuredOnly``) a measured envelope covers the admission key and
    /// recipe, on a device with the profile's minimum physical memory.
    func admit(_ plan: Plan) throws(AetherError) {
        let p = plan.profile, key = plan.admissionKey
        if !Self.runs(p, recipe: plan.recipeName, h3: h3, keyframes: !plan.keyframes.isEmpty) {
            throw VideoRefusal.recipeNotOffered(
                profile: p.key, recipe: plan.recipeName, offered: Self.offeredRecipes(p, h3: h3, keyframes: !plan.keyframes.isEmpty)).error
        }
        guard admissionPolicy == .measuredOnly else { return }
        guard let envelope = envelopes[key], envelope.recipes.contains(plan.recipeName) else {
            throw VideoRefusal.unmeasured(admissionKey: key, recipe: plan.recipeName).error
        }
        if UInt64(clamping: envelope.minimumPhysicalMemoryBytes) > device.physicalMemory {
            throw VideoRefusal.insufficientPhysicalMemory(
                admissionKey: key, neededBytes: envelope.minimumPhysicalMemoryBytes,
                availableBytes: Int64(clamping: device.physicalMemory)).error
        }
    }

    /// Whether `profile` runs `recipe` for its task: the profile's `recipes` (absent, every recipe) name it, the export
    /// has it for the task, and its modulation tables are inside the profile's body shapes.
    static func runs(_ profile: Manifest.H3.Profile, recipe name: String, h3: Manifest.H3, keyframes: Bool) -> Bool {
        if let recipes = profile.recipes, !recipes.contains(name) { return false }
        guard let recipe = h3.recipes[name + (keyframes ? "/fl2va" : "/t2va")] else { return false }
        let cardinality = recipe.steps.map(\.timesteps.count).max() ?? 1
        if let tables = profile.shapes?["body"]?["table"], !tables.contains([6, 3 * cardinality, 5376]) { return false }
        return true
    }

    /// The recipes `profile` runs for its task (``runs(_:recipe:h3:keyframes:)``), sorted.
    static func offeredRecipes(_ profile: Manifest.H3.Profile, h3: Manifest.H3, keyframes: Bool) -> [String] {
        let task = keyframes ? "/fl2va" : "/t2va"
        let names = h3.recipes.keys.filter { $0.hasSuffix(task) }.map { String($0.dropLast(task.count)) }
        return Set(names).filter { runs(profile, recipe: $0, h3: h3, keyframes: keyframes) }.sorted()
    }

    /// The envelope a request would be admitted under (its profile, recipe and keyframe count resolved as
    /// ``generate(_:onProgress:)`` resolves them), or nil when it has none: under ``VideoAdmissionPolicy/measuredOnly``
    /// such a request is refused. Its recipe must be one the envelope measured.
    public nonisolated func envelope(for request: VideoRequest) -> VideoMemoryEnvelope? {
        guard let profile = request.profile.map({ h3.profile($0) }) ?? h3.profiles.first else { return nil }
        let count = request.keyframes?.frames.count ?? 0
        guard let envelope = envelopes[VideoAdmissionKey(profile: profile.key, keyframeCount: count)],
              envelope.recipes.contains(request.recipe ?? h3.defaultRecipe) else { return nil }
        return envelope
    }

    /// The body chain's ``ShapeProfile``: the profile's manifest shapes for the body (its modulation tables), else the
    /// render's own, and the render's packed rows. `inputs` maps the contract's input names to the graph's tensors.
    static func bodyShapes(_ profile: Manifest.H3.Profile, rows: Int, cardinality: Int, inputs: [String: String]) -> ShapeProfile {
        var body: [String: Set<[Int]>] = [:]
        for (input, shapes) in profile.shapes?["body"] ?? ["table": [[6, 3 * cardinality, 5376]]] {
            if let name = inputs[input] { body[name] = Set(shapes) }
        }
        body[inputs["hidden"] ?? "hidden"] = [[1, rows, 5376]]
        return ShapeProfile(name: profile.key, shapes: ["body": body])
    }

    /// The vision tower's role for a canvas (`conditionerVision960x544`).
    static func visionRole(_ canvas: String) -> String { "conditionerVision\(canvas)" }

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

    // MARK: - The generation

    /// Gate-only boundary tensors.
    final class Trace: @unchecked Sendable {
        var ids: [Int32] = []
        /// The conditioner's ids and the packed rows' tags (the presentation with keyframes).
        var conditioningIDs: [Int32] = []
        var tags: [Int32] = []
        /// The noised keyframe condition rows that lead the video latents `[conditionRows, 96]`.
        var conditions: [Float] = []
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
        trace?.conditioningIDs = plan.conditioningIDs
        // Prompt: the keyframes' conditions and vision rows, the conditioner chain, then the refiner. The refined text
        // outlives the phase (a host buffer).
        try await residency.beginPhase(.perPrompt)
        let host = TensorArena()
        var conditions: [[Float]] = [], vision: [H3VisionRows] = []
        if let rules = keyframeRules, !plan.keyframes.isEmpty {
            conditions = try await encodeKeyframes(plan, rules: rules, residency: residency)
            vision = try await embedKeyframes(plan, rules: rules, residency: residency)
        }
        let refined = try await conditionAndRefine(plan, vision: vision, residency: residency, host: host,
                                                   onProgress: onProgress, trace: trace)
        residency.sample()
        residency.endPhase(.perPrompt)
        try Task.checkCancellation()

        try await residency.beginPhase(.perStep)
        var latents = try g.initialLatents(seed: plan.seed, conditions: conditions,
                                           timestep: keyframeRules?.keyframe.conditionTimestep ?? 0.999)
        if let trace { trace.conditions = Array(latents.video.prefix(plan.anchors.count * g.rowsPerLatentFrame * 96)) }
        try await denoise(plan, latents: &latents, refined: refined, residency: residency, host: host,
                          onProgress: onProgress, trace: trace)
        residency.sample()
        residency.endPhase(.perStep)
        try Task.checkCancellation()

        try await residency.beginPhase(.perImage)
        if let limit = plan.stepLimit, limit < plan.recipe.steps.count {
            residency.endPhase(.perImage)
            return GeneratedVideo(width: g.width, height: g.height, frames: [], audio: nil)
        }
        let generated = Array(latents.video.dropFirst(plan.anchors.count * g.rowsPerLatentFrame * 96))
        let frames = try await decodeVideo(generated, geometry: g, residency: residency, host: host, onProgress: onProgress)
        let audio = plan.audio ? try await decodeAudio(latents.audio, plan: plan, residency: residency, host: host) : nil
        residency.sample()
        residency.endPhase(.perImage)
        // A cancellation during the last decoder call (the audio decode, or the last video tile): don't report success
        // for a request the caller has given up on; `generate` releases the residency.
        try Task.checkCancellation()
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
        _ plan: Plan, vision: [H3VisionRows], residency: GraphResidency, host: TensorArena,
        onProgress: (@Sendable (VideoProgress) -> Void)?, trace: Trace?
    ) async throws -> TensorBuffer {
        let c = h3.conditioner, ids = plan.conditioningIDs, t = ids.count, width = c.embedding.width
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
        if let presentation = plan.presentation, let rules = keyframeRules {
            // M-RoPE: each column's axis position, gathered from the text tables.
            for (path, buffer) in [(c.ropeTables.cos, cos), (c.ropeTables.sin, sin)] {
                let table = try readFile(path).withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                let rows = try presentation.rotary(table, width: rowBytes / 2, section: rules.mropeSection)
                try buffer.writeBytes(rows.withUnsafeBytes { [UInt8]($0) })
            }
        } else {
            try writePrefix(c.ropeTables.cos, bytes: t * rowBytes, into: cos)
            try writePrefix(c.ropeTables.sin, bytes: t * rowBytes, into: sin)
        }
        // The prompt's embedding rows, gathered from the row source (bf16 bits).
        let embedding = try FileHandle(forReadingFrom: try url(c.embedding.file))
        defer { try? embedding.close() }
        var rows = [UInt8]()
        rows.reserveCapacity(t * width * 2)
        for id in ids {
            guard (0..<Int32(c.embedding.rows)).contains(id) else {
                throw AetherError.generation(.invalidParameters("token id \(id) is outside the embedding's \(c.embedding.rows) rows"))
            }
            try embedding.seek(toOffset: UInt64(id) * UInt64(width * 2))
            guard let row = try embedding.read(upToCount: width * 2), row.count == width * 2 else {
                throw AetherError.integrity(file: c.embedding.file.description, expected: "row \(id)", actual: "a short read")
            }
            rows.append(contentsOf: row)
        }
        // The keyframes' vision rows replace their image pads' embeddings; DeepStack level k is added at the image rows
        // after layer k, on the host between chain runs.
        let images = plan.presentation?.images ?? []
        H3ImageRows.scatter(vision.map(\.embeddings), into: &rows, width: width, at: images)
        try chain.head.writeBytes(rows)
        let levels = vision.isEmpty ? 0 : (keyframeRules?.vision.deepstackLevels ?? 0)
        var segments = (0..<min(levels, c.layers)).map { $0..<($0 + 1) }
        if levels < c.layers { segments.append(levels..<c.layers) }
        for (segment, layers) in segments.enumerated() {
            try await chain.run { chain in
                if segment == 0 { try banks.load(block: 0, into: 0, chain: chain) }
                for index in layers {
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
            if layers.upperBound <= levels {
                var hidden = try chain.head.readBytes()
                H3ImageRows.add(vision.map(\.deepstack), level: layers.lowerBound, into: &hidden, width: width, at: images)
                try chain.head.writeBytes(hidden)
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
        let g = plan.geometry
        let layout = H3Layout(textTags: plan.textTags, geometry: g, anchors: plan.anchors)
        let rows = layout.rows, c = plan.cardinality, hidden = 5376
        let videoRows = layout.videoIndices.count, conditionValues = layout.conditionRows * 96
        trace?.tags = layout.tags
        guard let metal = TensorArena.metal() else { throw AetherError.unsupportedDevice(catalogID: manifest.id, platform: "macOS", architecture: "no GPU") }
        let pack = try await residency.graph("preparePack")
        guard let body = try await residency.graph("body") as? Graph else {
            throw AetherError.manifestInvalid(field: "graphs.body", reason: "the body chains on the GPU")
        }
        let heads = try await residency.graph("heads")
        let p = names("preparePack"), b = names("body"), h = names("heads")

        // Host buffers: the pack's inputs and output (also the heads' hidden input), and the heads' others.
        let videoIn = try pack.makeBuffer(for: p.inputs["video"]!, role: .input, dimensions: [videoRows], in: host)
        let audioIn = try pack.makeBuffer(for: p.inputs["audio"]!, role: .input, dimensions: [g.audioRows], in: host)
        let gather = try pack.makeBuffer(for: p.inputs["gather"]!, role: .input, dimensions: [rows], in: host)
        let packed = try pack.makeBuffer(for: p.outputs["hidden"]!, role: .output, dimensions: [rows], in: host)
        try gather.write(layout.gather)
        let temb = try heads.makeBuffer(for: h.inputs["temb"]!, role: .input, dimensions: [c], in: host)
        let indices = try heads.makeBuffer(for: h.inputs["indices"]!, role: .input, dimensions: [rows], in: host)
        let videoIndices = try heads.makeBuffer(for: h.inputs["videoIndices"]!, role: .input, dimensions: [videoRows], in: host)
        let audioIndices = try heads.makeBuffer(for: h.inputs["audioIndices"]!, role: .input, dimensions: [g.audioRows], in: host)
        try videoIndices.write(layout.videoIndices)
        try audioIndices.write(layout.audioIndices)
        let videoOut = try heads.makeBuffer(for: h.outputs["video"]!, role: .output, dimensions: [videoRows], in: host)
        let audioOut = try heads.makeBuffer(for: h.outputs["audio"]!, role: .output, dimensions: [g.audioRows], in: host)

        // The body chain on the GPU: rotary tables, tags and one AdaLN table per block, written before each step's run.
        let profile = Self.bodyShapes(plan.profile, rows: rows, cardinality: c, inputs: b.inputs)
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
            let row = layout.timesteps(video: video.timesteps[index], audio: audio.timesteps[index],
                                       condition: keyframeRules?.keyframe.conditionTimestep ?? 0.999)
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
            // Only the generated rows advance: the keyframe conditions ride through every step unchanged.
            video.advance(step: index, sample: &latents.video, velocity: vv, from: conditionValues)
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
            .perStep: PhaseFootprint(peakBytes: 9893 << 20, heldBytes: 8614 << 20),
            .perImage: PhaseFootprint(peakBytes: 1445 << 20, heldBytes: 1151 << 20),
        ],
    ]

    /// What the footprints above were measured on: the export (its ``Manifest/contentDigest``, the bundle in
    /// `forge-outputs/minimax-h3`) and the Core AI architecture of the M5 Max. Another export (a requantization, another recipe) or another chip doesn't use them
    /// as measured; it plans from its files until it has measured its own. `MiniMaxH3GateTests` checks that the gate's
    /// export is this one.
    public static let qualifiedExportDigest = "46dc5fb190adb038fd5715559a238f9a32afd685b2793629794c166503d66584"
    public static let qualifiedArchitecture = "h17c"

    /// The envelopes that hold for `manifest` on `device`, by admission key: the manifest's `h3.envelopes` measured on
    /// this export (``Manifest/contentDigest``, which leaves the envelopes out) and the device's Core AI architecture,
    /// with their profiles' bars;
    /// and, for the export and chip ``qualifiedFootprints`` were measured on, those (turbo8, 18 GiB) where the manifest
    /// has none for the key.
    public static func measuredEnvelopes(manifest: Manifest, device: Device) -> [VideoAdmissionKey: VideoMemoryEnvelope] {
        guard device.platform == .macOS, let h3 = manifest.h3 else { return [:] }
        var out: [VideoAdmissionKey: VideoMemoryEnvelope] = [:]
        let digest = manifest.contentDigest
        for e in h3.envelopes ?? [] where e.export == digest && e.architecture == device.architecture {
            guard let memory = h3.profile(e.profile)?.memory,
                  let key = VideoAdmissionKey(e.key) else { continue }
            out[key] = VideoMemoryEnvelope(
                key: key, recipes: e.recipes,
                phases: [.perPrompt: PhaseFootprint(peakBytes: e.perPrompt.peakBytes, heldBytes: e.perPrompt.heldBytes),
                         .perStep: PhaseFootprint(peakBytes: e.perStep.peakBytes, heldBytes: e.perStep.heldBytes),
                         .perImage: PhaseFootprint(peakBytes: e.perImage.peakBytes, heldBytes: e.perImage.heldBytes)],
                marginBytes: e.marginBytes, barBytes: memory.barBytes, minimumPhysicalMemoryBytes: memory.minimumPhysicalMemoryBytes)
        }
        if device.architecture == qualifiedArchitecture, manifest.contentDigest == qualifiedExportDigest {
            for (profile, phases) in qualifiedFootprints where h3.profile(profile) != nil {
                let key = VideoAdmissionKey(profile: profile)
                guard out[key] == nil else { continue }
                out[key] = VideoMemoryEnvelope(key: key, recipes: ["turbo8"], phases: phases, marginBytes: 256 << 20,
                                               barBytes: 18 << 30, minimumPhysicalMemoryBytes: 0)
            }
        }
        return out
    }

    /// Seeds `ledger` under `model` with each envelope's phases, keyed by its admission key.
    static func seed(_ ledger: FootprintLedger, model: String, envelopes: [VideoAdmissionKey: VideoMemoryEnvelope]) {
        for (key, envelope) in envelopes {
            for (phase, footprint) in envelope.phases {
                let entry = FootprintLedger.Key(model: model, profile: key.description, phase: phase)
                if ledger.footprint(entry) == nil { ledger.record(entry, footprint) }
            }
        }
    }

    /// Seeds `ledger` under `model`, the pipeline's ledger key (``ResidentAdmission/modelKey``: the manifest id, the
    /// weights' source revision, the export and the device), so the seeded entries are the ones its plans read; only
    /// for the export and the architecture they were measured on (``qualifiedExportDigest``, ``qualifiedArchitecture``).
    static func seedQualified(
        _ ledger: FootprintLedger, model: String, manifest: Manifest, device: Device,
        qualifiedDigest: String = qualifiedExportDigest
    ) {
        guard device.platform == .macOS, device.architecture == qualifiedArchitecture,
              manifest.contentDigest == qualifiedDigest else { return }
        for (profile, phases) in qualifiedFootprints {
            for (phase, footprint) in phases {
                let key = FootprintLedger.Key(model: model, profile: profile, phase: phase)
                if ledger.footprint(key) == nil { ledger.record(key, footprint) }
            }
        }
    }
}

/// One keyframe's vision-tower rows (bf16 bits): the merger rows `[tokens, width]` and DeepStack `[levels, tokens, width]`.
struct H3VisionRows: Sendable, Hashable {
    var embeddings: [UInt8]
    var deepstack: [UInt8]
}

extension MiniMaxH3Pipeline {
    // MARK: Keyframes

    /// Each keyframe's normalized condition latents `[1, 24, 1, H/16, W/16]`, in packed order.
    func encodeKeyframes(_ plan: Plan, rules: H3KeyframeRules, residency: GraphResidency) async throws -> [[Float]] {
        guard let tailRole = plan.profile.keyframeEncoder, rules.keyframe.tileSize == H3Tiling.side else {
            throw AetherError.manifestMismatch(field: "h3.keyframe.tileSize", expected: "\(H3Tiling.side) and a keyframeEncoder",
                                               found: "\(rules.keyframe.tileSize)")
        }
        let tile = try await residency.graph("videoEncoderTile"), tail = try await residency.graph(tailRole)
        let t = names("videoEncoderTile"), u = names(tailRole)
        let arena = TensorArena()
        var conditions: [[Float]] = []
        for frame in plan.keyframes {
            try Task.checkCancellation()
            conditions.append(try await Self.encodeKeyframe(
                frame, tile: (tile, t.inputs["pixels"]!, t.outputs["moments"]!),
                tail: (tail, u.inputs["tiles"]!, u.outputs["condition"]!), arena: arena))
            residency.sample()
        }
        return conditions
    }

    /// The tiled first-frame encode (`encode_vae_condition` on one frame): each 256 px tile of the planar 0…255 pixels
    /// through the tile graph, in row-major tile order (the source VAE's `_split_tiles`), then the canvas's tail graph
    /// (the stitch, the fixed-seed posterior draw, the float16 rounding and the latent normalization).
    static func encodeKeyframe(
        _ frame: VideoKeyframe, tile: (graph: any GraphBackend, pixels: String, moments: String),
        tail: (graph: any GraphBackend, tiles: String, condition: String), arena: TensorArena
    ) async throws -> [Float] {
        let side = H3Tiling.side, m = side / 16, tiling = H3Tiling(width: frame.width, height: frame.height)
        let pixels = try tile.graph.makeBuffer(for: tile.pixels, role: .input, dimensions: [], in: arena)
        let moments = try tile.graph.makeBuffer(for: tile.moments, role: .output, dimensions: [], in: arena)
        let tiles = try tail.graph.makeBuffer(for: tail.tiles, role: .input, dimensions: [], in: arena)
        let condition = try tail.graph.makeBuffer(for: tail.condition, role: .output, dimensions: [], in: arena)
        guard pixels.shape == [1, 3, 1, side, side], moments.shape == [1, 48, 1, m, m], tiles.shape == [tiling.count, 48, 1, m, m],
              condition.shape == [1, 24, 1, frame.height / 16, frame.width / 16] else {
            throw AetherError.manifestMismatch(
                field: "graphs.videoEncoderTile", expected: "[1, 3, 1, \(side), \(side)] → [1, 48, 1, \(m), \(m)]; \(tiling.count) tiles → [1, 24, 1, H/16, W/16]",
                found: "\(pixels.shape) → \(moments.shape); \(tiles.shape) → \(condition.shape)")
        }
        var gathered: [Float] = []
        gathered.reserveCapacity(tiles.count)
        var values = [Float](repeating: 0, count: 3 * side * side)
        for index in 0..<tiling.count {
            try Task.checkCancellation()
            let y0 = tiling.yStarts[index / tiling.xStarts.count], x0 = tiling.xStarts[index % tiling.xStarts.count]
            for c in 0..<3 {
                for y in 0..<side {
                    let source = ((y0 + y) * frame.width + x0) * 3 + c, target = (c * side + y) * side
                    for x in 0..<side { values[target + x] = Float(frame.rgb[source + x * 3]) }
                }
            }
            try pixels.write(values)
            var bindings = Bindings(arena: arena)
            try bindings.bindInput(tile.pixels, to: pixels)
            try bindings.bindOutput(tile.moments, to: moments)
            try await tile.graph.run(&bindings)
            gathered += try moments.readFloats()
        }
        try tiles.write(gathered)
        var bindings = Bindings(arena: arena)
        try bindings.bindInput(tail.tiles, to: tiles)
        try bindings.bindOutput(tail.condition, to: condition)
        try await tail.graph.run(&bindings)
        return try condition.readFloats()
    }

    /// Each keyframe through the canvas's vision tower (its weights from the bundle, one block on the host), in
    /// packed order.
    func embedKeyframes(_ plan: Plan, rules: H3KeyframeRules, residency: GraphResidency) async throws -> [H3VisionRows] {
        guard let presentation = plan.presentation else { return [] }
        let role = Self.visionRole("\(plan.profile.width)x\(plan.profile.height)")
        let graph = try await residency.graph(role)
        let arena = TensorArena()
        let banks = try weightBanks(role, graph: graph, arena: arena)
        try banks.load(block: 0, into: 0)
        var rows: [H3VisionRows] = []
        for (frame, image) in zip(plan.keyframes, presentation.images) {
            try Task.checkCancellation()
            rows.append(try await Self.embedKeyframe(
                frame, rules: rules, graph: graph, names: names(role), tokens: image.tokens, width: h3.conditioner.embedding.width,
                arena: arena, bindWeights: { bindings throws(AetherError) in try banks.bind(bank: 0, into: &bindings) }))
            residency.sample()
        }
        return rows
    }

    /// One keyframe's tower call: the processor's bf16 pixels and the host tables in, the merger rows and DeepStack out.
    static func embedKeyframe(
        _ frame: VideoKeyframe, rules: H3KeyframeRules, graph: any GraphBackend,
        names: (inputs: [String: String], outputs: [String: String]), tokens: Int, width: Int, arena: TensorArena,
        bindWeights: (inout Bindings) throws(AetherError) -> Void
    ) async throws -> H3VisionRows {
        let inputs = try H3VisionInputs(frame, rules: rules), v = rules.vision
        let rowLength = 3 * v.temporalPatchSize * v.patchSize * v.patchSize, n = inputs.patches
        func buffer(_ role: String, _ side: TensorRole) throws(AetherError) -> TensorBuffer {
            let name = side == .input ? names.inputs[role] : names.outputs[role]
            guard let name else { throw .manifestInvalid(field: "graphs.conditionerVision.\(role)", reason: "required") }
            return try graph.makeBuffer(for: name, role: side, dimensions: [], in: arena)
        }
        let pixels = try buffer("pixels", .input), index = try buffer("posIndex", .input)
        let weight = try buffer("posWeight", .input), rc = try buffer("patchRC", .input)
        let embeds = try buffer("imageEmbeds", .output), deep = try buffer("deepstack", .output)
        guard pixels.shape == [n, rowLength], pixels.scalarType == .bfloat16, index.shape == [n, 4], weight.shape == [n, 4],
              rc.shape == [n, 2], embeds.shape == [tokens, width], embeds.scalarType == .bfloat16,
              deep.shape == [v.deepstackLevels, tokens, width], deep.scalarType == .bfloat16 else {
            throw AetherError.manifestMismatch(
                field: "graphs.conditionerVision", expected: "pixels [\(n), \(rowLength)] bf16 → [\(tokens), \(width)], [\(v.deepstackLevels), \(tokens), \(width)] bf16",
                found: "\(pixels.shape) \(pixels.scalarType) → \(embeds.shape), \(deep.shape)")
        }
        try pixels.writeBytes(inputs.pixels.withUnsafeBytes { [UInt8]($0) })
        try index.write(inputs.positionIndex)
        try weight.write(inputs.positionWeight)
        try rc.write(inputs.patchCoordinates)
        var bindings = Bindings(arena: arena)
        for (role, buffer) in [("pixels", pixels), ("posIndex", index), ("posWeight", weight), ("patchRC", rc)] {
            try bindings.bindInput(names.inputs[role]!, to: buffer)
        }
        try bindWeights(&bindings)
        try bindings.bindOutput(names.outputs["imageEmbeds"]!, to: embeds)
        try bindings.bindOutput(names.outputs["deepstack"]!, to: deep)
        try await graph.run(&bindings)
        return H3VisionRows(embeddings: try embeds.readBytes(), deepstack: try deep.readBytes())
    }
}
