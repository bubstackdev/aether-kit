import AetherKit
import AetherH3
import Foundation

// The H3VideoSession stages over a schema-3 profile. The control flow (token embedding, layer streaming, packing,
// the 50-block trajectory, tile decoding, audio) is plain Swift over small kernel protocols, so CPU tests drive it
// with fakes; the native kernels below run the profile's Core AI graphs. Each stage opens its graphs on first use
// and drops them in `release()`.

/// Conditioner graph calls: one decoder layer (`hidden [1,T,5120]`, `cos`/`sin [1,T,128]` bf16 -> hidden).
protocol H3ConditionerKernel: Sendable {
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData
    func release() async
}
/// Denoiser graph calls.
protocol H3DenoiserKernels: Sendable {
    func refine(_ text: H3TensorData) async throws -> H3TensorData
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData)
    func release() async
}
/// One 256 px spatial tile of one temporal window: rows `[448, 96]` f32 -> pixels `[1,3,22,256,256]` f32.
protocol H3TileKernel: Sendable {
    func decode(_ rows: H3TensorData) async throws -> H3TensorData
    func release() async
}
/// Channel-major rows `[2, Na, 32]` f32 -> planar PCM `[2, 800 Na]` f32.
protocol H3AudioKernel: Sendable {
    func decode(_ rows: H3TensorData) async throws -> H3TensorData
    func release() async
}

func h3Tensor(_ type: H3ScalarType, _ shape: [Int], _ values: [Int32]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: type, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}
func h3Tensor(_ shape: [Int], _ values: [Float]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}

/// Reads token embeddings in place and streams the 50 conditioner layers; returns the unnormalized tap.
actor H3ProfileConditioner: H3PromptConditioner {
    private let conditioner: H3Profile.Conditioner
    private let limits: ClosedRange<Int>
    private let tokenize: @Sendable (String) throws -> [Int32]
    private let kernel: any H3ConditionerKernel
    private var rope: (cos: H3TensorData, sin: H3TensorData)?
    private var embeddingVerified = false

    init(profile: H3Profile, tokenize: @escaping @Sendable (String) throws -> [Int32], kernel: any H3ConditionerKernel) throws {
        guard let conditioner = profile.conditioner else { throw AetherError.invalidModel("H3 profile has no conditioner") }
        self.init(conditioner: conditioner, limits: profile.textTokenLimits, tokenize: tokenize, kernel: kernel)
    }
    init(conditioner: H3Profile.Conditioner, limits: ClosedRange<Int>, tokenize: @escaping @Sendable (String) throws -> [Int32],
         kernel: any H3ConditionerKernel) {
        self.conditioner = conditioner; self.limits = limits; self.tokenize = tokenize; self.kernel = kernel
    }

    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        let ids = try tokenize(prompt)
        guard limits.contains(ids.count) else { throw AetherError.invalidRequest("H3 prompt has \(ids.count) tokens; the profile supports \(limits)") }
        if !embeddingVerified { try conditioner.embedding.verify(); embeddingVerified = true }
        if rope == nil { rope = (try conditioner.ropeCos.read(), try conditioner.ropeSin.read()) }
        let rowBytes = 128 * 2, t = ids.count
        func prefix(_ table: H3TensorData) throws -> H3TensorData {
            try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, t, 128]), bytes: table.bytes.prefix(t * rowBytes))
        }
        let cos = try prefix(rope!.cos), sin = try prefix(rope!.sin)
        var hidden = try conditioner.embedding.rows(ids)
        let spec = hidden.spec
        for index in 0..<conditioner.parameters.count {
            try Task.checkCancellation()
            hidden = try await kernel.layer(index, hidden: hidden, cos: cos, sin: sin)
            guard hidden.spec == spec else {
                throw AetherError.invalidTensor("H3 conditioner layer output shape differs")
            }
            onLayer(index + 1, conditioner.parameters.count)
        }
        return hidden
    }
    func release() async { await kernel.release(); rope = nil }
}

/// Packed-sequence helpers shared by the denoiser and its tests.
enum H3Packing {
    /// `gather[r]` = the source row of packed row `r` in `[text; video; audio]`.
    static func gather(_ layout: H3Layout) throws -> [Int32] {
        var gather = [Int32](repeating: -1, count: layout.rows)
        let t = layout.textIndices.count, v = layout.videoIndices.count
        for (j, r) in layout.textIndices.enumerated() { gather[Int(r)] = Int32(j) }
        for (j, r) in layout.videoIndices.enumerated() { gather[Int(r)] = Int32(t + j) }
        for (j, r) in layout.audioIndices.enumerated() { gather[Int(r)] = Int32(t + v + j) }
        guard !gather.contains(-1) else { throw AetherError.invalidModel("H3 layout rows are not a permutation of text, video and audio") }
        return gather
    }
}

/// The Turbo trajectory over the profile: text refined once, then per step pack -> 50 blocks -> heads -> update.
actor H3ProfileDenoiser: H3Denoiser {
    nonisolated let steps: Int
    private let schedule: H3Schedule
    private let tables: @Sendable (H3Layout) throws -> [H3Profile.Step]
    private let blocks: Int
    private let kernels: any H3DenoiserKernels

    init(profile: H3Profile, recipe: String, kernels: any H3DenoiserKernels) throws {
        guard let entry = profile.recipes[recipe] else { throw AetherError.unsupported("H3 profile has no \(recipe) schedule") }
        guard profile.parameters.count == 50, profile.prepare != nil else { throw AetherError.invalidModel("H3 profile lacks block weights or prepare graphs") }
        self.init(schedule: entry.schedule, blocks: 50, tables: { try profile.steps(recipe: recipe, layout: $0) }, kernels: kernels)
    }
    /// `tables(layout)` returns each step's embedding and per-block AdaLN tables, already matched to the layout.
    init(schedule: H3Schedule, blocks: Int, tables: @escaping @Sendable (H3Layout) throws -> [H3Profile.Step], kernels: any H3DenoiserKernels) {
        self.schedule = schedule; self.steps = schedule.count; self.blocks = blocks; self.tables = tables; self.kernels = kernels
    }

    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry,
                 onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        guard conditioning.spec.scalarType == .bfloat16, conditioning.spec.shape.count == 3, conditioning.spec.shape[0] == 1,
              conditioning.spec.shape[2] == 5120 else { throw AetherError.invalidTensor("H3 conditioning must be the [1,T,5120] bf16 tap") }
        let t = conditioning.spec.shape[1]
        let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: t))
        guard layout.conditionVideoRows == 0, initial.video.spec.shape == [1, layout.videoIndices.count, 96],
              initial.audio.spec.shape == [1, layout.audioIndices.count, 32] else {
            throw AetherError.invalidTensor("H3 initial latents differ from the layout")
        }
        let stepTables = try tables(layout)
        guard stepTables.count == steps, stepTables.allSatisfy({ $0.tables.count == blocks }) else { throw AetherError.invalidModel("H3 step tables differ from the schedule") }
        let rotary = try layout.rotary()
        let cos = try h3Tensor([layout.rows, 96], rotary.cos.values), sin = try h3Tensor([layout.rows, 96], rotary.sin.values)
        let gather = try h3Tensor(.int32, [layout.rows], H3Packing.gather(layout))
        let videoIndices = try h3Tensor(.int32, [layout.videoIndices.count], layout.videoIndices)
        let audioIndices = try h3Tensor(.int32, [layout.audioIndices.count], layout.audioIndices)
        try Task.checkCancellation()
        let refined = try await kernels.refine(conditioning)
        guard refined.spec == (try H3TensorSpec(scalarType: .bfloat16, shape: [1, t, 5376])) else { throw AetherError.invalidTensor("H3 refined text shape differs") }
        var latents = initial
        for (step, entry) in zip(schedule.steps, stepTables) {
            try Task.checkCancellation()
            let rows = layout.timesteps(for: step)
            let tags = try h3Tensor(.int32, [layout.rows], rows.adalnIndices)
            let indices = try h3Tensor(.int32, [layout.rows], rows.indices)
            var hidden = try await kernels.pack(video: latents.video, audio: latents.audio, text: refined, gather: gather)
            let hiddenSpec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, layout.rows, 5376])
            guard hidden.spec == hiddenSpec else { throw AetherError.invalidTensor("H3 packed sequence shape differs") }
            for block in 0..<blocks {
                try Task.checkCancellation()
                hidden = try await kernels.block(block, hidden: hidden, table: try entry.tables[block].read(), tags: tags, cos: cos, sin: sin)
                guard hidden.spec == hiddenSpec else { throw AetherError.invalidTensor("H3 block output shape differs") }
            }
            try Task.checkCancellation()
            let velocity = try await kernels.heads(hidden: hidden, temb: try entry.embedding.read(), indices: indices, videoIndices: videoIndices,
                                                   audioIndices: audioIndices, video: latents.video.spec, audio: latents.audio.spec)
            latents = try latents.advanced(schedule: schedule, step: step.index, videoVelocity: velocity.video, audioVelocity: velocity.audio)
            onStep(step.index + 1, steps)
        }
        return latents
    }
    func release() async { await kernels.release() }
}

/// One temporal window, decoded as the source's 256 px spatial tiles and blended back.
actor H3ProfileVideoDecoder: H3WindowDecoder {
    private let kernel: any H3TileKernel
    init(kernel: any H3TileKernel) { self.kernel = kernel }
    func decode(windowRows: [Float], index: Int, geometry: H3Geometry) async throws -> [Float] {
        let tiling = try H3SpatialTiling.source(width: geometry.width, height: geometry.height)
        var stitcher = try H3SpatialStitcher(tiling: tiling, width: geometry.width, height: geometry.height)
        for tile in 0..<tiling.tileCount {
            try Task.checkCancellation()
            let rows = try tiling.extract(width: geometry.width, height: geometry.height, packed: windowRows, index: tile)
            let pixels = try await kernel.decode(try h3Tensor([448, 96], rows))
            guard pixels.spec == (try H3TensorSpec(scalarType: .float32, shape: [1, 3, 22, 256, 256])) else {
                throw AetherError.invalidTensor("H3 tile decoder output shape differs")
            }
            try stitcher.push(try pixels.floatValues())
        }
        return try stitcher.finish()
    }
    func release() async { await kernel.release() }
}

actor H3ProfileAudioDecoder: H3AudioDecoder {
    private let kernel: any H3AudioKernel
    private let hop: Int
    init(kernel: any H3AudioKernel, hop: Int = 800) { self.kernel = kernel; self.hop = hop }
    func decode(_ latents: H3TensorData, geometry: H3Geometry) async throws -> H3AudioPCM {
        let na = geometry.audioLatentsPerChannel
        guard latents.spec == (try H3TensorSpec(scalarType: .float32, shape: [1, 2 * na, 32])) else {
            throw AetherError.invalidTensor("H3 audio latents differ from the geometry")
        }
        // Channel-major rows: [1, 2 Na, 32] and [2, Na, 32] are the same bytes.
        let rows = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [2, na, 32]), bytes: latents.bytes)
        let pcm = try await kernel.decode(rows)
        guard pcm.spec == (try H3TensorSpec(scalarType: .float32, shape: [2, hop * na])) else { throw AetherError.invalidTensor("H3 audio decoder output shape differs") }
        let values = try pcm.floatValues()
        return try H3AudioPCM(channels: [Array(values[..<(hop * na)]), Array(values[(hop * na)...])])
    }
    func release() async { await kernel.release() }
}

// MARK: Native kernels

actor NativeH3ConditionerKernel: H3ConditionerKernel {
    private let conditioner: H3Profile.Conditioner
    private var graph: NativeH3Graph?
    init(_ conditioner: H3Profile.Conditioner) { self.conditioner = conditioner }
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        if graph == nil { graph = try await NativeH3Graph.open(conditioner.layer) }
        var inputs: [String: NativeH3Graph.Input] = ["hidden": .data(hidden), "cos": .data(cos), "sin": .data(sin)]
        for file in conditioner.parameters[index] { inputs[file.name] = .file(file) }
        return try await graph!.run(inputs, outputs: ["hidden_out": hidden.spec])["hidden_out"]!
    }
    func release() async { graph = nil }
}

actor NativeH3DenoiserKernels: H3DenoiserKernels {
    private let profile: H3Profile
    private let prepare: H3Profile.Prepare
    private var packing: NativeH3Graph?, body: NativeH3Graph?, output: NativeH3Graph?
    init(_ profile: H3Profile) throws {
        guard let prepare = profile.prepare else { throw AetherError.invalidModel("H3 profile has no prepare graphs") }
        self.profile = profile; self.prepare = prepare
    }
    func refine(_ text: H3TensorData) async throws -> H3TensorData {
        // Once per render: the text graph is dropped right after, so its ~1.6 GB of refiner slots never co-reside with the body.
        let graph = try await NativeH3Graph.open(prepare.text)
        var inputs: [String: NativeH3Graph.Input] = ["text": .data(text)]
        for file in prepare.textParameters { inputs[file.name] = .file(file) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, text.spec.shape[1], 5376])
        return try await graph.run(inputs, outputs: ["refined": spec])["refined"]!
    }
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        if packing == nil { packing = try await NativeH3Graph.open(prepare.pack) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, gather.spec.shape[0], 5376])
        return try await packing!.run(["video": .data(video), "audio": .data(audio), "text": .data(text), "gather": .data(gather)],
                                      outputs: ["hidden": spec])["hidden"]!
    }
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        if body == nil { body = try await NativeH3Graph.open(profile.body) }
        var inputs: [String: NativeH3Graph.Input] = ["hidden": .data(hidden), "temb": .data(table), "tags": .data(tags), "cos": .data(cos), "sin": .data(sin)]
        for file in profile.parameters[index] { inputs[file.name] = .file(file) }
        return try await body!.run(inputs, outputs: ["hidden_out": hidden.spec])["hidden_out"]!
    }
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        if output == nil { output = try await NativeH3Graph.open(profile.heads) }
        let result = try await output!.run(["hidden": .data(hidden), "temb": .data(temb), "indices": .data(indices),
                                            "video_indices": .data(videoIndices), "audio_indices": .data(audioIndices)],
                                           outputs: ["video": video, "audio": audio])
        return (result["video"]!, result["audio"]!)
    }
    func release() async { packing = nil; body = nil; output = nil }
}

actor NativeH3TileKernel: H3TileKernel {
    private let asset: H3ModelAsset
    private var graph: NativeH3Graph?
    init(_ asset: H3ModelAsset) { self.asset = asset }
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        if graph == nil { graph = try await NativeH3Graph.open(asset) }
        return try await graph!.run(["rows": .data(rows)], outputs: ["pixels": try H3TensorSpec(scalarType: .float32, shape: [1, 3, 22, 256, 256])])["pixels"]!
    }
    func release() async { graph = nil }
}

actor NativeH3AudioKernel: H3AudioKernel {
    private let asset: H3ModelAsset
    private let hop: Int
    private var graph: NativeH3Graph?
    init(_ asset: H3ModelAsset, hop: Int) { self.asset = asset; self.hop = hop }
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        if graph == nil { graph = try await NativeH3Graph.open(asset) }
        let spec = try H3TensorSpec(scalarType: .float32, shape: [2, hop * rows.spec.shape[1]])
        return try await graph!.run(["rows": .data(rows)], outputs: ["pcm": spec])["pcm"]!
    }
    func release() async { graph = nil }
}

/// Native stages for `H3VideoSession` over a complete schema-3 profile. `tokenize` turns the structured prompt into
/// the conditioner's token ids (`AetherH3Tokenizers.H3Tokenizer.tokenIDs`).
public enum H3ProfileEngine {
    public static func stages(profile: H3Profile, recipe: String = "turbo8",
                              tokenize: @escaping @Sendable (String) throws -> [Int32]) throws -> H3VideoStages {
        guard profile.complete, let conditioner = profile.conditioner, let audio = profile.audioDecoder else {
            throw AetherError.invalidModel("H3 profile is incomplete: \(profile.missing.joined(separator: ", "))")
        }
        return H3VideoStages(
            conditioner: try H3ProfileConditioner(profile: profile, tokenize: tokenize, kernel: NativeH3ConditionerKernel(conditioner)),
            denoiser: try H3ProfileDenoiser(profile: profile, recipe: recipe, kernels: try NativeH3DenoiserKernels(profile)),
            video: H3ProfileVideoDecoder(kernel: NativeH3TileKernel(profile.videoDecoder.model)),
            audio: H3ProfileAudioDecoder(kernel: NativeH3AudioKernel(audio.model, hop: audio.hop), hop: audio.hop))
    }
}
