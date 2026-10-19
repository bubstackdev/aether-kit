import AetherKit
import Foundation

/// A schema-2 exported H3 profile (`Scripts/export-h3-profile.py finalize`): one dynamic body graph for all 50 blocks,
/// dynamic heads and conditioner layers, a windowed tile video decoder, and per-schedule precomputed timestep
/// embeddings and AdaLN tables instead of modulation graphs. Opening validates structure only (no model load);
/// `verifyModels()` / `verifyTables()` hash the referenced files. Incomplete profiles open but report `missing`.
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
        public let embedding: H3ModelAsset
        public let layers: [H3ModelAsset]
        public let tap: Int
    }
    public struct VideoDecoder: Sendable {
        public let model: H3ModelAsset
        public let tile: Int
    }

    public let header: H3ExportHeader?
    public let manifestSHA256: String
    public let complete: Bool
    public let missing: [String]
    public let rowLimits: ClosedRange<Int>
    public let cardinalityLimits: ClosedRange<Int>
    public let textTokenLimits: ClosedRange<Int>
    public let tokenizer: URL
    public let conditioner: Conditioner?
    public let prepare: H3ModelAsset?
    public let body: H3ModelAsset
    public let heads: H3ModelAsset
    public let parameters: [[H3TensorFile]]
    public let recipes: [String: Recipe]
    public let videoDecoder: VideoDecoder

    public init(url: URL) throws {
        guard url.isFileURL else { throw AetherError.invalidModel("H3 profiles must be local") }
        let data = try Data(contentsOf: url)
        manifestSHA256 = h3SHA256(data)
        let m: Manifest
        do { m = try JSONDecoder().decode(Manifest.self, from: data) } catch let error as DecodingError {
            throw AetherError.invalidModel("H3 profile manifest is malformed: \(error)")
        }
        guard m.schema == 2, m.snapshot == H3ExportHeader.snapshot, m.diffusers_commit == H3ExportHeader.diffusersCommit,
              m.transformer_weight_layout == H3ExportHeader.transformerWeightLayout else {
            throw AetherError.invalidModel("H3 profile requires schema 2, pinned provenance and the corrected per-head QKV layout")
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
        if let c = m.conditioner, let embedding = c.embedding, !c.layers.isEmpty {
            guard c.layers.count == 50, c.tap == 50 else { throw AetherError.invalidModel("H3 conditioner must preserve the unnormalized 50-layer tap") }
            conditioner = Conditioner(embedding: try embedding.asset(path), layers: try c.layers.map { try $0.asset(path) }, tap: c.tap)
        } else {
            guard m.missing.contains("conditioner") else { throw AetherError.invalidModel("H3 profile omits its conditioner without declaring it missing") }
            conditioner = nil
        }
        prepare = try m.prepare.map { try $0.asset(path) }
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
        videoDecoder = VideoDecoder(model: try m.video_decoder.model.asset(path), tile: 256)
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
        let all = [body, heads, videoDecoder.model] + (prepare.map { [$0] } ?? []) + (conditioner.map { [$0.embedding] + $0.layers } ?? [])
        for model in all where seen.insert(model.url.path).inserted { try model.verify() }
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
    private struct Tensor: Decodable {
        let name: String; let path: String; let dtype: String; let shape: [Int]; let sha256: String
        func file(_ resolve: (String) throws -> URL) throws -> H3TensorFile {
            guard let type = H3ScalarType(rawValue: dtype), type != .int32 else { throw AetherError.invalidModel("unsupported H3 tensor dtype \(dtype)") }
            return try H3TensorFile(name: name, url: resolve(path), spec: H3TensorSpec(scalarType: type, shape: shape), sha256: sha256)
        }
    }
    private struct Adapter: Decodable { let repository: String; let revision: String; let sha256: String; let execution: String; let rank: Int; let alpha: Float }
    private struct Limits: Decodable { let rows: [Int]; let cardinality: [Int]; let text_tokens: [Int] }
    private struct ConditionerEntry: Decodable { let embedding: Asset?; let layers: [Asset]; let tap: Int }
    private struct Arm: Decodable { let shift: Float; let timesteps: [Float]; let sigmas: [Float] }
    private struct Schedule: Decodable { let video: Arm; let audio: Arm }
    private struct StepEntry: Decodable { let timesteps: [Float]; let temb: Tensor; let blocks: [Tensor] }
    private struct RecipeEntry: Decodable { let schedule: Schedule; let tasks: [String: [StepEntry]] }
    private struct Decoder: Decodable {
        let path: String; let function: String; let sha256: String; let tile: Int; let window_latents: Int; let window_frames: Int; let window_stride: Int
        var model: Asset { Asset(path: path, function: function, sha256: sha256) }
    }
    private struct Manifest: Decodable {
        let schema: Int; let complete: Bool; let missing: [String]; let snapshot: String; let diffusers_commit: String
        let transformer_weight_layout: String; let adapter: Adapter?; let limits: Limits; let tokenizer: String
        let conditioner: ConditionerEntry?; let prepare: Asset?; let body: Asset; let heads: Asset; let parameters: [[Tensor]]
        let recipes: [String: RecipeEntry]; let video_decoder: Decoder
    }
}
