import AetherKit
import AetherAssets
import CryptoKit
import Foundation

/// A schema-3 exported H3 profile (`Scripts/export-h3-profile.py finalize`): one dynamic body graph for all 50 blocks
/// and one dynamic conditioner-layer graph for its 50 layers, both with lifted weights bound in place from the pinned
/// checkpoint and Turbo adapter; a per-render text graph and a per-step pack graph; dynamic heads; a windowed tile
/// video decoder; an audio decoder; and per-schedule precomputed timestep embeddings and AdaLN tables instead of
/// modulation graphs. Opening validates structure only (no model load); `verifyModels()` / `verifyTables()` /
/// `verifyParameters()` hash the referenced bytes. Incomplete profiles open but report `missing`.
public struct H3Profile: Sendable {
    public enum Task: String, Sendable, CaseIterable { case t2va, fl2va }
    public struct Step: Sendable {
        /// The distinct row timesteps of this step, sorted, exactly as the layout derives them.
        public let timesteps: [Float]
        public let embedding: H3TensorFile
        /// One `[6, 3k, 5376]` bf16 table per block.
        public let tables: [H3TensorFile]
    }
    public struct Recipe: Sendable {
        public let schedule: H3Schedule
        public let steps: [Task: [Step]]
    }
    public struct Conditioner: Sendable {
        /// The Qwen3-VL token embedding, gathered on the CPU straight from the checkpoint (a pure row lookup).
        public let embedding: H3EmbeddingTable
        /// One decoder-layer graph; `parameters[i]` are layer i's 11 bf16 weights, in the graph's `w0...w10` order.
        public let layer: H3ModelAsset
        public let parameters: [[H3TensorFile]]
        public let tap: Int
        /// Text rotary tables `[4096, 128]` bf16 from the original module; a T-token prompt uses rows 0..<T.
        public let ropeCos: H3TensorFile
        public let ropeSin: H3TensorFile
    }
    public struct Prepare: Sendable {
        /// Once per prompt: text [1, T, 5120] bf16 + `textParameters` -> refined [1, T, 5376] bf16.
        public let text: H3ModelAsset
        public let textParameters: [H3TensorFile]
        /// Every step: video [1, Nv, 96] f32, audio [1, Na, 32] f32, refined text, gather [R] i32 -> hidden [1, R, 5376].
        public let pack: H3ModelAsset
    }
    public struct AudioDecoder: Sendable {
        /// Channel-major rows [2, Na, 32] f32 -> planar stereo PCM [2, hop * Na] f32 in [-1, 1], Na dynamic. The CPU
        /// export check uses it; on the GPU it aborts in MPSGraph (unresolved dynamic memref.alloc), so native runs use `fixed`.
        public let model: H3ModelAsset
        /// The same decoder with a static latent count Na, per Na the profile was exported for.
        public let fixed: [Int: H3ModelAsset]
        /// The static decoder for `latents` (Na per channel); a clip length without one is unsupported.
        public func model(latents: Int) throws -> H3ModelAsset {
            guard let model = fixed[latents] else {
                throw AetherError.unsupported("H3 profile has no audio decoder for \(latents) latents (has \(fixed.keys.sorted()))")
            }
            return model
        }
        public let sampleRate: Int
        public let hop: Int
        public let channels: Int
    }
    public struct VideoDecoder: Sendable {
        /// The default window-tile decoder (bfloat16 weights in current profiles; see `precision`).
        public let model: H3ModelAsset
        public let tile: Int
        public let precision: String
        /// The float32 decoder, kept for parity and diagnostics only; hosts never pick it.
        public let float32: H3ModelAsset?
    }

    public let header: H3ExportHeader?
    public let manifestSHA256: String
    /// Every H3 clip model identity starts with this prefix (and nothing else does), so hosts can key the licence
    /// credit on it.
    public static let modelIdentityPrefix = "minimax-h3."
    /// The clip model identity hosts record: the profile family plus its manifest fingerprint.
    public var modelIdentity: String { "\(Self.modelIdentityPrefix)fl2va-turbo/1:\(manifestSHA256)" }
    public let complete: Bool
    public let missing: [String]
    public let rowLimits: ClosedRange<Int>
    public let cardinalityLimits: ClosedRange<Int>
    public let textTokenLimits: ClosedRange<Int>
    public let tokenizer: URL
    public private(set) var conditioner: Conditioner?
    public private(set) var prepare: Prepare?
    public let body: H3ModelAsset
    public let heads: H3ModelAsset
    public private(set) var parameters: [[H3TensorFile]]
    public let recipes: [String: Recipe]
    public let videoDecoder: VideoDecoder
    /// First-frame (FL2VA keyframe) encoders, one static graph per canvas, keyed "WIDTHxHEIGHT".
    public let keyframeEncoders: [String: H3ModelAsset]
    public func keyframeEncoder(width: Int, height: Int) throws -> H3ModelAsset {
        guard let model = keyframeEncoders["\(width)x\(height)"] else {
            throw AetherError.unsupported("H3 profile has no first-frame encoder for \(width)x\(height)")
        }
        return model
    }
    /// The first-frame encoder split as the pipeline's own tiled encode: one 256 px tile graph (pixels -> moments
    /// `[1,48,1,16,16]`), run per source tile, and one tail graph per canvas (the source tile blend, posterior with
    /// the fixed keyframe draw, float16 rounding and latent normalization). Each call holds one tile's scratch
    /// instead of the whole canvas's.
    public struct TiledKeyframeEncoder: Sendable {
        public let tile: H3ModelAsset
        public let tileSize: Int
        public let tails: [String: H3ModelAsset]
    }
    public let tiledKeyframeEncoder: TiledKeyframeEncoder?
    /// How a canvas's first frame is encoded. A profile with tiled entries for the canvas uses them; otherwise the
    /// single whole-frame graph. Selection follows the manifest alone.
    public enum KeyframeEncoding: Sendable {
        case whole(H3ModelAsset)
        case tiled(tile: H3ModelAsset, tail: H3ModelAsset, tiling: H3SpatialTiling)
    }
    public func keyframeEncoding(width: Int, height: Int) throws -> KeyframeEncoding? {
        let canvas = "\(width)x\(height)"
        if let tiled = tiledKeyframeEncoder, let tail = tiled.tails[canvas] {
            return .tiled(tile: tiled.tile, tail: tail, tiling: try H3SpatialTiling.source(width: width, height: height))
        }
        return keyframeEncoders[canvas].map { .whole($0) }
    }
    public let audioDecoder: AudioDecoder?

    public init(url: URL) throws {
        guard url.isFileURL else { throw AetherError.invalidModel("H3 profiles must be local") }
        let data = try Data(contentsOf: url)
        manifestSHA256 = h3SHA256(data)
        let m: Manifest
        do { m = try JSONDecoder().decode(Manifest.self, from: data) } catch let error as DecodingError {
            throw AetherError.invalidModel("H3 profile manifest is malformed: \(error)")
        }
        guard m.schema == 3, m.snapshot == H3ExportHeader.snapshot, m.diffusers_commit == H3ExportHeader.diffusersCommit,
              m.transformer_weight_layout == H3ExportHeader.transformerWeightLayout else {
            throw AetherError.invalidModel("H3 profile requires schema 3, pinned provenance and the corrected per-head QKV layout")
        }
        guard m.complete == m.missing.isEmpty else { throw AetherError.invalidModel("H3 profile completeness is inconsistent") }
        complete = m.complete; missing = m.missing
        // The adapter identity is the same pinned dynamic LoRA AetherKit already accepts.
        header = try m.adapter.map { adapter in
            let record: [String: Any] = ["schema": 1, "complete": true, "snapshot": m.snapshot, "diffusers_commit": m.diffusers_commit,
                                         "recipe": "turbo8", "transformer_weight_layout": m.transformer_weight_layout,
                                         "adapter": ["repository": adapter.repository, "revision": adapter.revision, "sha256": adapter.sha256,
                                                     "execution": adapter.execution, "rank": adapter.rank, "alpha": adapter.alpha]]
            return try H3ExportHeader(data: JSONSerialization.data(withJSONObject: record))
        }
        guard m.limits.rows.count == 2, m.limits.cardinality == [1, 3], m.limits.text_tokens.count == 2,
              m.limits.rows[0] >= 1, m.limits.rows[0] <= m.limits.rows[1], m.limits.text_tokens[0] >= 1,
              m.limits.text_tokens[0] <= m.limits.text_tokens[1] else {
            throw AetherError.invalidModel("H3 profile dynamic-shape limits are invalid")
        }
        rowLimits = m.limits.rows[0]...m.limits.rows[1]; cardinalityLimits = 1...3
        textTokenLimits = m.limits.text_tokens[0]...m.limits.text_tokens[1]
        func path(_ value: String) throws -> URL {
            guard value.hasPrefix("/"), !value.utf8.contains(0) else { throw AetherError.invalidModel("H3 profile paths must be absolute") }
            return URL(fileURLWithPath: value)
        }
        tokenizer = try path(m.tokenizer)
        if let c = m.conditioner {
            guard c.parameters.count == 50, c.tap == 50 else { throw AetherError.invalidModel("H3 conditioner must preserve the unnormalized 50-layer tap") }
            guard c.parameters.allSatisfy({ $0.count == 11 && $0.enumerated().allSatisfy { $1.name == "w\($0)" && $1.dtype == "bfloat16" } }) else {
                throw AetherError.invalidModel("H3 conditioner must bind 11 bf16 weights for each of 50 layers")
            }
            let table = c.embedding_table
            guard table.dtype == "bfloat16", table.shape.count == 2 else { throw AetherError.invalidModel("H3 embedding table must be a bf16 matrix") }
            conditioner = Conditioner(
                embedding: try H3EmbeddingTable(url: path(table.path), offset: table.offset, rows: table.shape[0], width: table.shape[1],
                                                byteCount: table.length, sha256: table.sha256),
                layer: try c.layer.asset(path), parameters: try c.parameters.map { try $0.map { try $0.file(path) } }, tap: c.tap,
                ropeCos: try c.rope_tables.cos.file(path), ropeSin: try c.rope_tables.sin.file(path))
            guard [conditioner!.ropeCos, conditioner!.ropeSin].allSatisfy({ $0.spec.scalarType == .bfloat16 && $0.spec.shape == [4096, 128] }) else {
                throw AetherError.invalidModel("H3 conditioner rotary tables must be [4096, 128] bf16")
            }
        } else {
            guard m.missing.contains("conditioner") else { throw AetherError.invalidModel("H3 profile omits its conditioner without declaring it missing") }
            conditioner = nil
        }
        prepare = try m.prepare.map { p in
            guard p.text_parameters.enumerated().allSatisfy({ $1.name == "w\($0)" }), !p.text_parameters.isEmpty else {
                throw AetherError.invalidModel("H3 text graph bindings are out of order")
            }
            return Prepare(text: try p.text.asset(path), textParameters: try p.text_parameters.map { try $0.file(path) }, pack: try p.pack.asset(path))
        }
        guard prepare != nil || m.missing.contains("prepare") else { throw AetherError.invalidModel("H3 profile omits prepare without declaring it missing") }
        body = try m.body.asset(path); heads = try m.heads.asset(path)
        guard m.parameters.isEmpty || (m.parameters.count == 50 && m.parameters.allSatisfy {
            $0.count == 22 && $0.enumerated().allSatisfy { $1.name == "w\($0)" && $1.dtype == "bfloat16" }
        }) else { throw AetherError.invalidModel("H3 profile must bind 22 bf16 weights for each of 50 blocks") }
        guard !m.parameters.isEmpty || m.missing.contains("parameters") else {
            throw AetherError.invalidModel("H3 profile omits block weights without declaring them missing")
        }
        parameters = try m.parameters.map { try $0.map { try $0.file(path) } }
        guard m.video_decoder.tile == 256, m.video_decoder.window_latents == 7, m.video_decoder.window_frames == 22,
              m.video_decoder.window_stride == 5 else { throw AetherError.invalidModel("H3 profile lacks the source decoder window contract") }
        guard ["float32", "bfloat16"].contains(m.video_decoder.precision ?? "float32") else { throw AetherError.invalidModel("unknown H3 video decoder precision") }
        keyframeEncoders = try (m.video_encoder_first ?? [:]).mapValues { try $0.asset(path) }
        guard keyframeEncoders.keys.allSatisfy({ $0.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil }) else {
            throw AetherError.invalidModel("invalid H3 first-frame encoder canvas")
        }
        tiledKeyframeEncoder = try m.video_encoder_first_tiled.map { t in
            guard t.tile_size == 256, t.moments_channels == 48, !t.tails.isEmpty,
                  t.tails.keys.allSatisfy({ $0.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil }) else {
                throw AetherError.invalidModel("H3 tiled first-frame encoder lacks the 256 px source tile contract")
            }
            for canvas in t.tails.keys {
                let sides = canvas.split(separator: "x").map { Int($0)! }
                guard sides[0] % 16 == 0, sides[1] % 16 == 0 else { throw AetherError.invalidModel("invalid H3 tiled encoder canvas \(canvas)") }
                _ = try H3SpatialTiling.source(width: sides[0], height: sides[1])
            }
            return TiledKeyframeEncoder(tile: try t.tile.asset(path), tileSize: t.tile_size, tails: try t.tails.mapValues { try $0.asset(path) })
        }
        videoDecoder = VideoDecoder(model: try m.video_decoder.model.asset(path), tile: 256, precision: m.video_decoder.precision ?? "float32",
                                    float32: try m.video_decoder.float32?.asset(path))
        audioDecoder = try m.audio_decoder.map { a in
            guard a.sample_rate == 32000, a.hop == 800, a.channels == 2 else { throw AetherError.invalidModel("H3 audio decoder lacks the 32 kHz stereo contract") }
            var fixed: [Int: H3ModelAsset] = [:]
            for (key, asset) in a.fixed ?? [:] {
                guard let na = Int(key), (2...2048).contains(na), String(na) == key else { throw AetherError.invalidModel("invalid H3 audio decoder length \(key)") }
                fixed[na] = try asset.asset(path)
            }
            return AudioDecoder(model: try a.model.asset(path), fixed: fixed, sampleRate: a.sample_rate, hop: a.hop, channels: a.channels)
        }
        guard audioDecoder != nil || m.missing.contains("audio_decoder") else {
            throw AetherError.invalidModel("H3 profile omits its audio decoder without declaring it missing")
        }
        guard !m.recipes.isEmpty else { throw AetherError.invalidModel("H3 profile has no schedules") }
        var recipes: [String: Recipe] = [:]
        for (name, recipe) in m.recipes {
            let count = recipe.schedule.video.timesteps.count
            let schedule = try H3Schedule.turbo(steps: count, videoShift: recipe.schedule.video.shift, audioShift: recipe.schedule.audio.shift)
            // The recorded grid must be the generated one, bit for bit.
            guard schedule.sigmas(.video).map(\.bitPattern) == recipe.schedule.video.sigmas.map(\.bitPattern),
                  schedule.sigmas(.audio).map(\.bitPattern) == recipe.schedule.audio.sigmas.map(\.bitPattern) else {
                throw AetherError.invalidModel("H3 profile schedule \(name) differs from its shifted grid")
            }
            var steps: [Task: [Step]] = [:]
            for (taskName, entries) in recipe.tasks {
                guard let task = Task(rawValue: taskName), entries.count == count else { throw AetherError.invalidModel("H3 profile tables for \(name)/\(taskName) differ") }
                steps[task] = try entries.map { entry in
                    let k = entry.timesteps.count
                    guard (1...3).contains(k), entry.timesteps == entry.timesteps.sorted(), entry.blocks.count == 50 else {
                        throw AetherError.invalidModel("H3 profile step tables are incomplete")
                    }
                    let embedding = try entry.temb.file(path)
                    guard embedding.spec.scalarType == .float32, embedding.spec.shape == [k, 2688] else { throw AetherError.invalidModel("H3 timestep embedding shape differs") }
                    let tables = try entry.blocks.map { try $0.file(path) }
                    guard tables.allSatisfy({ $0.spec.scalarType == .bfloat16 && $0.spec.shape == [6, 3 * k, 5376] }) else {
                        throw AetherError.invalidModel("H3 AdaLN table shape differs")
                    }
                    return Step(timesteps: entry.timesteps, embedding: embedding, tables: tables)
                }
            }
            recipes[name] = Recipe(schedule: schedule, steps: steps)
        }
        self.recipes = recipes
    }

    /// Pins checkpoint/adapter weight sources without copying or hashing the complete checkpoint.
    /// Every later tensor/embedding read uses these descriptors and the existing range hashes.
    /// Profile graphs, tokenizer and schedule tables keep their existing caller-owned immutability contract.
    public func protectingWeights() throws -> Self {
        let weights = parameters.flatMap { $0 } + (conditioner?.parameters.flatMap { $0 } ?? []) + (prepare?.textParameters ?? [])
        let urls = weights.flatMap { $0.segments.map { $0.url } } + (conditioner.map { [$0.embedding.url] } ?? [])
        let lease = try ExternalFileLease(files: urls)
        var result = self
        result.parameters = try parameters.map { try $0.map { try $0.holding(lease) } }
        if let c = conditioner {
            result.conditioner = Conditioner(embedding: try c.embedding.holding(lease), layer: c.layer,
                                             parameters: try c.parameters.map { try $0.map { try $0.holding(lease) } },
                                             tap: c.tap, ropeCos: c.ropeCos, ropeSin: c.ropeSin)
        }
        if let prepare {
            result.prepare = Prepare(text: prepare.text, textParameters: try prepare.textParameters.map { try $0.holding(lease) }, pack: prepare.pack)
        }
        return result
    }

    /// The step tables for `layout` under `recipe`. The layout's own row timesteps must equal the recorded ones for
    /// every step, so a table can never be applied to rows it was not computed for.
    public func steps(recipe name: String, layout: H3Layout) throws -> [Step] {
        guard let recipe = recipes[name] else { throw AetherError.unsupported("H3 profile has no \(name) schedule") }
        let task: Task = layout.conditionVideoRows > 0 ? .fl2va : .t2va
        guard let steps = recipe.steps[task] else { throw AetherError.unsupported("H3 profile has no \(name) tables for \(task.rawValue)") }
        guard rowLimits.contains(layout.rows) else { throw AetherError.unsupported("H3 clip needs \(layout.rows) rows; the profile supports \(rowLimits)") }
        for (step, entry) in zip(recipe.schedule.steps, steps) {
            guard layout.timesteps(for: step).values.map(\.bitPattern) == entry.timesteps.map(\.bitPattern) else {
                throw AetherError.invalidModel("H3 profile tables do not match the layout's row timesteps")
            }
        }
        return steps
    }

    /// Hashes every distinct model stream.
    public func verifyModels() throws {
        var seen = Set<String>()
        let all = [body, heads, videoDecoder.model] + (videoDecoder.float32.map { [$0] } ?? []) + (prepare.map { [$0.text, $0.pack] } ?? []) + (conditioner.map { [$0.layer] } ?? [])
            + keyframeEncoders.keys.sorted().map { keyframeEncoders[$0]! }
            + (tiledKeyframeEncoder.map { t in [t.tile] + t.tails.keys.sorted().map { t.tails[$0]! } } ?? [])
            + (audioDecoder.map { audio in [audio.model] + audio.fixed.keys.sorted().map { audio.fixed[$0]! } } ?? [])
        for model in all where seen.insert(model.url.path).inserted { try model.verify() }
        try conditioner?.embedding.verify()
        if let conditioner { for file in [conditioner.ropeCos, conditioner.ropeSin] { _ = try file.read() } }
    }

    /// Hashes every in-place weight binding (≈ 90 GB of reads for a complete profile).
    public func verifyParameters() throws {
        let all = parameters + (conditioner?.parameters ?? []) + [prepare?.textParameters ?? []]
        for group in all { for file in group { try autoreleasepool { _ = try file.read() } } }
    }

    /// Hashes every table and embedding file.
    public func verifyTables() throws {
        for recipe in recipes.values {
            for steps in recipe.steps.values {
                for step in steps { for file in [step.embedding] + step.tables { try autoreleasepool { _ = try file.read() } } }
            }
        }
    }

    private struct Asset: Decodable {
        let path: String; let function: String; let sha256: String
        func asset(_ resolve: (String) throws -> URL) throws -> H3ModelAsset { try H3ModelAsset(url: resolve(path), function: function, sha256: sha256) }
    }
    private struct Segment: Decodable { let path: String; let offset: Int; let length: Int }
    /// A whole raw file (`path`) or byte ranges read in place from pinned files (`segments`), never both.
    private struct Tensor: Decodable {
        let name: String; let path: String?; let segments: [Segment]?; let dtype: String; let shape: [Int]; let sha256: String
        func file(_ resolve: (String) throws -> URL) throws -> H3TensorFile {
            guard let type = H3ScalarType(rawValue: dtype), type != .int32 else { throw AetherError.invalidModel("unsupported H3 tensor dtype \(dtype)") }
            let spec = try H3TensorSpec(scalarType: type, shape: shape)
            switch (path, segments) {
            case let (path?, nil): return try H3TensorFile(name: name, url: resolve(path), spec: spec, sha256: sha256)
            case let (nil, segments?):
                return try H3TensorFile(name: name, segments: segments.map { try .init(url: resolve($0.path), offset: $0.offset, length: $0.length) },
                                        spec: spec, sha256: sha256)
            default: throw AetherError.invalidModel("H3 tensor \(name) needs exactly one of a path or segments")
            }
        }
    }
    private struct Adapter: Decodable { let repository: String; let revision: String; let sha256: String; let execution: String; let rank: Int; let alpha: Float }
    private struct Limits: Decodable { let rows: [Int]; let cardinality: [Int]; let text_tokens: [Int] }
    private struct Table: Decodable { let path: String; let offset: Int; let length: Int; let dtype: String; let shape: [Int]; let sha256: String }
    private struct RopeTables: Decodable { let cos: Tensor; let sin: Tensor }
    private struct ConditionerEntry: Decodable { let embedding_table: Table; let layer: Asset; let parameters: [[Tensor]]; let tap: Int; let rope_tables: RopeTables }
    private struct PrepareEntry: Decodable { let text: Asset; let text_parameters: [Tensor]; let pack: Asset }
    private struct AudioEntry: Decodable {
        let path: String; let function: String; let sha256: String; let sample_rate: Int; let hop: Int; let channels: Int
        let fixed: [String: Asset]?
        var model: Asset { Asset(path: path, function: function, sha256: sha256) }
    }
    private struct Arm: Decodable { let shift: Float; let timesteps: [Float]; let sigmas: [Float] }
    private struct Schedule: Decodable { let video: Arm; let audio: Arm }
    private struct StepEntry: Decodable { let timesteps: [Float]; let temb: Tensor; let blocks: [Tensor] }
    private struct RecipeEntry: Decodable { let schedule: Schedule; let tasks: [String: [StepEntry]] }
    private struct Decoder: Decodable {
        let path: String; let function: String; let sha256: String; let tile: Int; let window_latents: Int; let window_frames: Int; let window_stride: Int
        let precision: String?; let float32: Asset?
        var model: Asset { Asset(path: path, function: function, sha256: sha256) }
    }
    private struct Manifest: Decodable {
        let schema: Int; let complete: Bool; let missing: [String]; let snapshot: String; let diffusers_commit: String
        let transformer_weight_layout: String; let adapter: Adapter?; let limits: Limits; let tokenizer: String
        let conditioner: ConditionerEntry?; let prepare: PrepareEntry?; let body: Asset; let heads: Asset; let parameters: [[Tensor]]
        let recipes: [String: RecipeEntry]; let video_decoder: Decoder; let audio_decoder: AudioEntry?
        let video_encoder_first: [String: Asset]?
        let video_encoder_first_tiled: TiledEncoderEntry?
    }
    private struct TiledEncoderEntry: Decodable { let tile: Asset; let tile_size: Int; let moments_channels: Int; let tails: [String: Asset] }
}

/// A bf16 `[rows, width]` matrix inside an immutable file, read row by row in place (the conditioner's token
/// embedding). `verify()` hashes the whole range once; `rows(_:)` then gathers rows without re-hashing.
public struct H3EmbeddingTable: Sendable {
    public let url: URL
    public let offset: Int
    public let rows: Int
    public let width: Int
    public let sha256: String
    private var lease: ExternalFileLease? = nil
    public init(url: URL, offset: Int, rows: Int, width: Int, byteCount: Int, sha256: String) throws {
        let stride = width.multipliedReportingOverflow(by: 2)
        let extent = rows.multipliedReportingOverflow(by: stride.partialValue)
        guard url.isFileURL, offset >= 0, rows > 0, width > 0, !stride.overflow,
              !extent.overflow, !offset.addingReportingOverflow(byteCount).overflow, extent.partialValue == byteCount,
              h3IsSHA256(sha256) else { throw AetherError.invalidModel("invalid H3 embedding table") }
        self.url = url.standardizedFileURL; self.offset = offset; self.rows = rows; self.width = width; self.sha256 = sha256
    }
    public func holding(_ lease: ExternalFileLease) throws -> Self {
        let size = try lease.byteCount(of: url), count = UInt64(rows * width * 2)
        guard UInt64(offset) <= size, count <= size - UInt64(offset) else { throw AetherError.invalidModel("H3 leased embedding extent differs") }
        var result = self; result.lease = lease; return result
    }

    public func verify() throws {
        if let lease {
            var hasher = SHA256(), consumed = 0
            while consumed < rows * width * 2 {
                try autoreleasepool {
                    var chunk = Data(count: min(8 << 20, rows * width * 2 - consumed))
                    try chunk.withUnsafeMutableBytes { try lease.read(into: $0, from: url, offset: UInt64(offset + consumed)) }
                    hasher.update(data: chunk); consumed += chunk.count
                }
            }
            guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else { throw AetherError.invalidModel("H3 embedding table hash mismatch") }
            return
        }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(offset))
        var hasher = SHA256(), remaining = rows * width * 2
        while remaining > 0 {
            try Task.checkCancellation()
            guard let chunk = try autoreleasepool(invoking: { try handle.read(upToCount: min(remaining, 8 << 20)) }), !chunk.isEmpty else {
                throw AetherError.invalidModel("H3 embedding table is short")
            }
            hasher.update(data: chunk); remaining -= chunk.count
        }
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == sha256 else { throw AetherError.invalidModel("H3 embedding table hash mismatch") }
    }
    /// `[1, ids.count, width]` bf16, exactly the table's rows (an embedding lookup is a byte gather).
    public func rows(_ ids: [Int32]) throws -> H3TensorData {
        guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && Int($0) < rows }) else { throw AetherError.invalidTensor("H3 token id outside the embedding table") }
        let stride = width * 2
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, ids.count, width])
        if let lease {
            var bytes = Data(count: spec.byteCount)
            try bytes.withUnsafeMutableBytes { raw in
                for (index, id) in ids.enumerated() {
                    let row = UnsafeMutableRawBufferPointer(rebasing: raw[(index * stride)..<((index + 1) * stride)])
                    try lease.read(into: row, from: url, offset: UInt64(offset + Int(id) * stride))
                }
            }
            return try H3TensorData(spec: spec, bytes: bytes)
        }
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        var bytes = Data(capacity: spec.byteCount)
        for id in ids {
            try handle.seek(toOffset: UInt64(offset + Int(id) * stride))
            guard let row = try handle.read(upToCount: stride), row.count == stride else { throw AetherError.invalidTensor("H3 embedding row is short") }
            bytes.append(row)
        }
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, ids.count, width]), bytes: bytes)
    }
}
