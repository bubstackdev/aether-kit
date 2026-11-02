import AetherKit
import AetherH3
import Foundation

// The H3VideoSession stages over a schema-3 profile. The control flow (token embedding, layer streaming, packing,
// the 50-block trajectory, tile decoding, audio) is plain Swift over small kernel protocols, so CPU tests drive it
// with fakes; the native kernels below run the profile's Core AI graphs. Each stage opens its graphs on first use
// and drops them in `release()`.

/// Conditioner graph calls: one decoder layer (`hidden [1,T,5120]`, `cos`/`sin [1,T,128]` bf16 -> hidden).
@available(macOS 27, iOS 27, *)
protocol H3ConditionerKernel: Sendable {
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData
    func release() async
}
/// Keyframe vision-tower calls (`H3Profile.ConditionerVision`): one keyframe's inputs at its canvas -> the merger rows
/// `[n/4, 5120]` and DeepStack `[levels, n/4, 5120]`, both bf16.
@available(macOS 27, iOS 27, *)
protocol H3VisionKernel: Sendable {
    func encode(_ inputs: H3VisionInputs, width: Int, height: Int) async throws -> (embeddings: H3TensorData, deepstack: H3TensorData)
    func release() async
}
/// Denoiser graph calls.
@available(macOS 27, iOS 27, *)
protocol H3DenoiserKernels: Sendable {
    func refine(_ text: H3TensorData) async throws -> H3TensorData
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData)
    /// One step's blocks 0..<tables.count in order; returns the last block's output. Each table is padded to
    /// `cardinality` timesteps (`H3ModulationPadding`). `onBlock`, when given, receives every block's output (readback
    /// only then).
    func blocks(hidden: H3TensorData, tables: [H3TensorFile], cardinality: Int, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData,
                onBlock: (@Sendable (Int, H3TensorData) async throws -> Void)?) async throws -> H3TensorData
    func release() async
}
@available(macOS 27, iOS 27, *)
extension H3DenoiserKernels {
    /// One `block` call per block, each output returned to the host.
    func blocks(hidden: H3TensorData, tables: [H3TensorFile], cardinality: Int, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData,
                onBlock: (@Sendable (Int, H3TensorData) async throws -> Void)?) async throws -> H3TensorData {
        var hidden = hidden
        let spec = hidden.spec
        for (index, table) in tables.enumerated() {
            try Task.checkCancellation()
            let padded = try H3ModulationPadding.table(try table.read(), cardinality: cardinality)
            hidden = try await block(index, hidden: hidden, table: padded, tags: tags, cos: cos, sin: sin)
            guard hidden.spec == spec else { throw AetherError.invalidTensor("H3 block output shape differs") }
            try await onBlock?(index, hidden)
        }
        return hidden
    }
}
/// One 256 px spatial tile of one temporal window: rows `[448, 96]` f32 -> pixels `[1,3,22,256,256]` f32.
@available(macOS 27, iOS 27, *)
protocol H3TileKernel: Sendable {
    func decode(_ rows: H3TensorData) async throws -> H3TensorData
    func release() async
}
/// Channel-major rows `[2, Na, 32]` f32 -> planar PCM `[2, 800 Na]` f32.
@available(macOS 27, iOS 27, *)
protocol H3AudioKernel: Sendable {
    /// Whether a decoder exists for `latents` (Na per channel).
    func supports(latents: Int) -> Bool
    func decode(_ rows: H3TensorData) async throws -> H3TensorData
    func release() async
}

@available(macOS 27, iOS 27, *)
extension H3AudioKernel {
    func supports(latents: Int) -> Bool { true }
}

@available(macOS 27, iOS 27, *)
func h3Tensor(_ type: H3ScalarType, _ shape: [Int], _ values: [Int32]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: type, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}
@available(macOS 27, iOS 27, *)
func h3Tensor(_ shape: [Int], _ values: [Float]) throws -> H3TensorData {
    try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
}

/// Reads token embeddings in place and streams the 50 conditioner layers; returns the unnormalized tap. With keyframes,
/// the presentation is upstream's (`H3Presentation`): each keyframe's vision-tower rows replace its image-pad embeddings,
/// the rotary tables follow the M-RoPE positions, and DeepStack level k is added at the image rows after layer k.
@available(macOS 27, iOS 27, *)
actor H3ProfileConditioner: H3PromptConditioner {
    private let conditioner: H3Profile.Conditioner
    private let limits: ClosedRange<Int>
    private let tokenize: @Sendable (String) throws -> [Int32]
    private let kernel: any H3ConditionerKernel
    private let vision: (entry: H3Profile.ConditionerVision, kernel: any H3VisionKernel)?
    /// The denoiser's packed-row limits (`H3Profile.rowLimits`), checked before any stage runs; nil checks none.
    private let rowLimits: ClosedRange<Int>?
    private var rope: (cos: H3TensorData, sin: H3TensorData)?
    private var embeddingVerified = false

    init(profile: H3Profile, tokenize: @escaping @Sendable (String) throws -> [Int32], kernel: any H3ConditionerKernel,
         vision: (any H3VisionKernel)? = nil) throws {
        guard let conditioner = profile.conditioner else { throw AetherError.invalidModel("H3 profile has no conditioner") }
        self.init(conditioner: conditioner, limits: profile.textTokenLimits, tokenize: tokenize, kernel: kernel,
                  vision: profile.conditionerVision.flatMap { entry in vision.map { (entry, $0) } }, rowLimits: profile.rowLimits)
    }
    init(conditioner: H3Profile.Conditioner, limits: ClosedRange<Int>, tokenize: @escaping @Sendable (String) throws -> [Int32],
         kernel: any H3ConditionerKernel, vision: (entry: H3Profile.ConditionerVision, kernel: any H3VisionKernel)? = nil,
         rowLimits: ClosedRange<Int>? = nil) {
        self.conditioner = conditioner; self.limits = limits; self.tokenize = tokenize; self.kernel = kernel; self.vision = vision
        self.rowLimits = rowLimits
    }

    /// Keyframes need the vision tower, with a graph for the canvas.
    nonisolated func supports(_ geometry: H3Geometry, keyframes: [H3Keyframe]) -> Bool {
        keyframes.isEmpty || vision?.entry.functions["\(geometry.width)x\(geometry.height)"] != nil
    }

    /// The presentation's length against the profile's text limits, and the packed rows it implies (text, condition
    /// rows of each anchor, video and audio) against the denoiser's row limits. Tokenizes only.
    nonisolated func preflight(_ request: H3VideoRequest) throws {
        let ids = try tokenize(request.prompt)
        var count = ids.count
        if request.hasKeyframes {
            guard let vision else { throw AetherError.unsupported("H3 profile has no keyframe vision encoder for FL2VA's presentation") }
            count = try H3Presentation(prompt: ids, keyframes: request.keyframes.map { ($0.width, $0.height) }, vision: vision.entry.vision,
                                       tokenize: tokenize).count
        }
        guard limits.contains(count) else {
            throw AetherError.invalidRequest("H3 presentation has \(count) tokens (prompt \(ids.count)); the profile supports \(limits)")
        }
        if let rowLimits {
            let rows = try H3Layout(geometry: request.geometry, textTags: Array(repeating: 1, count: count), anchors: request.anchors).rows
            guard rowLimits.contains(rows) else {
                throw AetherError.unsupported("H3 clip needs \(rows) packed rows (\(count) conditioning, \(request.anchors.count) keyframe(s)); "
                                              + "the profile supports \(rowLimits)")
            }
        }
    }

    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        try await encode(prompt, keyframes: [], geometry: geometry, onLayer: onLayer).tap
    }

    func encode(_ prompt: String, keyframes: [H3Keyframe], geometry: H3Geometry,
                onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Conditioning {
        let ids = try tokenize(prompt)
        guard !keyframes.isEmpty else { return try H3Conditioning(text: try await tap(ids, onLayer: onLayer), ids: ids) }
        guard let vision else { throw AetherError.unsupported("H3 profile has no keyframe vision encoder for FL2VA's presentation") }
        let presentation = try H3Presentation(prompt: ids, keyframes: keyframes.map { ($0.width, $0.height) }, vision: vision.entry.vision,
                                              tokenize: tokenize)
        guard limits.contains(presentation.count) else {
            throw AetherError.invalidRequest("H3 presentation has \(presentation.count) tokens (prompt \(ids.count)); the profile supports \(limits)")
        }
        // Every keyframe's tower call first, then the tower is released before the layers stream.
        var images: [(embeddings: H3TensorData, deepstack: H3TensorData)] = []
        do {
            for (keyframe, image) in zip(keyframes, presentation.images) {
                try Task.checkCancellation()
                let inputs = try H3VisionInputs(rgb: keyframe.rgb, width: keyframe.width, height: keyframe.height, vision: vision.entry.vision)
                let result = try await vision.kernel.encode(inputs, width: keyframe.width, height: keyframe.height)
                let hidden = vision.entry.hiddenSize, levels = vision.entry.deepstackLevels
                guard result.embeddings.spec == (try H3TensorSpec(scalarType: .bfloat16, shape: [image.tokens, hidden])),
                      result.deepstack.spec == (try H3TensorSpec(scalarType: .bfloat16, shape: [levels, image.tokens, hidden])) else {
                    throw AetherError.invalidTensor("H3 keyframe vision output shape differs")
                }
                images.append(result)
            }
        } catch { await vision.kernel.release(); throw error }
        await vision.kernel.release()
        let tap = try await tap(presentation.ids, presentation: presentation, images: images, levels: vision.entry.deepstackLevels, onLayer: onLayer)
        return try H3Conditioning(tap: tap, tags: presentation.tags, ids: presentation.ids)
    }

    /// `presentation` nil: a text-only prompt (rotary rows 0..<T). Otherwise the image rows and M-RoPE tables.
    private func tap(_ ids: [Int32], presentation: H3Presentation? = nil, images: [(embeddings: H3TensorData, deepstack: H3TensorData)] = [],
                     levels: Int = 0, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        guard limits.contains(ids.count) else { throw AetherError.invalidRequest("H3 prompt has \(ids.count) tokens; the profile supports \(limits)") }
        if !embeddingVerified { try conditioner.embedding.verify(); embeddingVerified = true }
        if rope == nil { rope = (try conditioner.ropeCos.read(), try conditioner.ropeSin.read()) }
        let rowBytes = 128 * 2, t = ids.count
        func prefix(_ table: H3TensorData) throws -> H3TensorData {
            try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, t, 128]), bytes: table.bytes.prefix(t * rowBytes))
        }
        let section = H3Presentation.Vision().mropeSection
        let cos = try presentation.map { try $0.rotary(rope!.cos, section: section) } ?? prefix(rope!.cos)
        let sin = try presentation.map { try $0.rotary(rope!.sin, section: section) } ?? prefix(rope!.sin)
        var hidden = try conditioner.embedding.rows(ids)
        let spec = hidden.spec
        if let presentation {
            hidden = try H3ImageRows.scatter(images.map(\.embeddings), into: hidden, at: presentation.images)
        }
        for index in 0..<conditioner.parameters.count {
            try Task.checkCancellation()
            hidden = try await kernel.layer(index, hidden: hidden, cos: cos, sin: sin)
            guard hidden.spec == spec else {
                throw AetherError.invalidTensor("H3 conditioner layer output shape differs")
            }
            if let presentation, index < levels {
                hidden = try H3ImageRows.add(images.map(\.deepstack), level: index, levels: levels, to: hidden, at: presentation.images)
            }
            onLayer(index + 1, conditioner.parameters.count)
        }
        return hidden
    }
    func release() async { await kernel.release(); await vision?.kernel.release(); rope = nil }
}

/// Host-side image rows of the presentation, in bf16 as upstream computes them: the merger rows replace the image-pad
/// embeddings (`masked_scatter`), and DeepStack adds in float32 rounded to nearest-even bf16 (torch's bf16 add).
@available(macOS 27, iOS 27, *)
enum H3ImageRows {
    static func scatter(_ embeddings: [H3TensorData], into hidden: H3TensorData, at images: [H3Presentation.Image]) throws -> H3TensorData {
        let width = hidden.spec.shape[2], rowBytes = width * 2
        var bytes = hidden.bytes
        for (rows, image) in zip(embeddings, images) {
            guard rows.spec.shape == [image.tokens, width] else { throw AetherError.invalidTensor("H3 image rows differ from their span") }
            bytes.replaceSubrange((bytes.startIndex + image.start * rowBytes)..<(bytes.startIndex + (image.start + image.tokens) * rowBytes), with: rows.bytes)
        }
        return try H3TensorData(spec: hidden.spec, bytes: bytes)
    }
    static func add(_ deepstack: [H3TensorData], level: Int, levels: Int, to hidden: H3TensorData, at images: [H3Presentation.Image]) throws -> H3TensorData {
        let width = hidden.spec.shape[2]
        var bytes = hidden.bytes
        for (features, image) in zip(deepstack, images) {
            guard features.spec.shape == [levels, image.tokens, width] else { throw AetherError.invalidTensor("H3 DeepStack rows differ from their span") }
            features.bytes.withUnsafeBytes { source in
                bytes.withUnsafeMutableBytes { target in
                    let add = source.bindMemory(to: UInt16.self), out = target.bindMemory(to: UInt16.self)
                    let from = level * image.tokens * width, to = image.start * width
                    for i in 0..<(image.tokens * width) {
                        let sum = Float(bitPattern: UInt32(out[to + i]) << 16) + Float(bitPattern: UInt32(add[from + i]) << 16)
                        out[to + i] = h3BFloat16Bits(sum)
                    }
                }
            }
        }
        return try H3TensorData(spec: hidden.spec, bytes: bytes)
    }
}

/// Packed-sequence helpers shared by the denoiser and its tests.
@available(macOS 27, iOS 27, *)
public enum H3Packing {
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

/// Pads a step's modulation inputs to the trajectory's largest timestep cardinality, so the body and heads graphs see
/// one shape for the whole render. Their `temb` inputs are dynamic in the cardinality k (the step's distinct row
/// timesteps: T2VA has k = 1 at step 0 and 2 afterwards, FL2VA 2 and then 3). Core AI specializes a reshape-enabled
/// function once per input shape and keeps every specialization until the model is released, so a k change mid-render
/// added a second body specialization: about 2.5 GiB of Metal at 960×544×124. This is a workaround for that retention,
/// not a fix. Padding is exact: the block reads its table only through `index_select` on the AdaLN indices
/// (`tags` < 3k) and the heads read `temb` rows only through `indices` (< k), so the appended rows, copies of existing
/// rows, are never selected, and the qualified boundary hashes replay unchanged.
@available(macOS 27, iOS 27, *)
enum H3ModulationPadding {
    /// A block's AdaLN table `[6, 3k, D]` bf16 -> `[6, 3 cardinality, D]`, each slice's rows followed by copies of them.
    static func table(_ table: H3TensorData, cardinality: Int) throws -> H3TensorData {
        guard table.spec.shape.count == 3, table.spec.shape[0] == 6, table.spec.shape[1] % 3 == 0 else {
            throw AetherError.invalidTensor("H3 modulation table must be [6, 3k, D]")
        }
        return try repeatingRows(table, slices: 6, cardinality: cardinality, rowsPerTimestep: 3)
    }
    /// A step's timestep embedding `[k, C]` -> `[cardinality, C]`, its rows followed by copies of them.
    static func embedding(_ embedding: H3TensorData, cardinality: Int) throws -> H3TensorData {
        guard embedding.spec.shape.count == 2 else { throw AetherError.invalidTensor("H3 timestep embedding must be [k, C]") }
        return try repeatingRows(embedding, slices: 1, cardinality: cardinality, rowsPerTimestep: 1)
    }
    private static func repeatingRows(_ tensor: H3TensorData, slices: Int, cardinality: Int, rowsPerTimestep: Int) throws -> H3TensorData {
        let shape = tensor.spec.shape, axis = shape.count - 2, rows = shape[axis]
        guard rows > 0, rows <= rowsPerTimestep * cardinality else { throw AetherError.invalidTensor("H3 modulation cardinality exceeds the padding target") }
        if rows == rowsPerTimestep * cardinality { return tensor }
        let rowBytes = tensor.spec.byteCount / (slices * rows), target = rowsPerTimestep * cardinality
        var bytes = Data(capacity: slices * target * rowBytes)
        for slice in 0..<slices {
            let base = tensor.bytes.startIndex + slice * rows * rowBytes
            for row in 0..<target {
                let source = base + (row % rows) * rowBytes
                bytes.append(tensor.bytes[source..<(source + rowBytes)])
            }
        }
        var padded = shape; padded[axis] = target
        return try H3TensorData(spec: H3TensorSpec(scalarType: tensor.spec.scalarType, shape: padded), bytes: bytes)
    }
}

/// One boundary of the profile denoiser's trajectory, for qualification. Every value is a tensor the trajectory
/// already holds on the CPU at that point; observing adds no native call or readback.
@available(macOS 27, iOS 27, *)
public enum H3DenoiserObservation: Sendable {
    /// The refined text, once per trajectory.
    case refined(H3TensorData)
    /// The latents entering step `step` (keyframe anchor rows first for FL2VA).
    case stepStarted(step: Int, latents: H3Latents)
    /// The packed sequence entering block 0.
    case packed(step: Int, hidden: H3TensorData)
    /// Block `index`'s output, which is block `index + 1`'s input.
    case block(step: Int, index: Int, output: H3TensorData)
    /// Both head velocities.
    case heads(step: Int, video: H3TensorData, audio: H3TensorData)
    /// The latents after the step's scheduler update.
    case advanced(step: Int, latents: H3Latents)
}
/// Opt-in observer of `H3DenoiserObservation`s (qualification). It is awaited at each boundary, and a thrown error
/// fails the trajectory like a kernel error. Without an observer the denoiser constructs no observation.
@available(macOS 27, iOS 27, *)
public typealias H3DenoiserObserver = @Sendable (H3DenoiserObservation) async throws -> Void

/// The Turbo trajectory over the profile: text refined once, then per step pack -> 50 blocks -> heads -> update.
@available(macOS 27, iOS 27, *)
actor H3ProfileDenoiser: H3Denoiser {
    nonisolated let steps: Int
    private let schedule: H3Schedule
    private let tables: @Sendable (H3Layout) throws -> [H3Profile.Step]
    private let blocks: Int
    private let kernels: any H3DenoiserKernels
    private let observer: H3DenoiserObserver?

    init(profile: H3Profile, recipe: String, kernels: any H3DenoiserKernels, observer: H3DenoiserObserver? = nil) throws {
        guard let entry = profile.recipes[recipe] else { throw AetherError.unsupported("H3 profile has no \(recipe) schedule") }
        guard profile.parameters.count == 50, profile.prepare != nil else { throw AetherError.invalidModel("H3 profile lacks block weights or prepare graphs") }
        self.init(schedule: entry.schedule, blocks: 50, tables: { try profile.steps(recipe: recipe, layout: $0) }, kernels: kernels, observer: observer)
    }
    /// `tables(layout)` returns each step's embedding and per-block AdaLN tables, already matched to the layout.
    init(schedule: H3Schedule, blocks: Int, tables: @escaping @Sendable (H3Layout) throws -> [H3Profile.Step], kernels: any H3DenoiserKernels,
         observer: H3DenoiserObserver? = nil) {
        self.schedule = schedule; self.steps = schedule.count; self.blocks = blocks; self.tables = tables; self.kernels = kernels
        self.observer = observer
    }

    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry,
                 onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        try await denoise(initial, conditioning: try H3Conditioning(text: conditioning), geometry: geometry, onStep: onStep)
    }

    func denoise(_ initial: H3Latents, conditioning tagged: H3Conditioning, geometry: H3Geometry,
                 onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        let conditioning = tagged.tap
        guard conditioning.spec.scalarType == .bfloat16, conditioning.spec.shape.count == 3, conditioning.spec.shape[0] == 1,
              conditioning.spec.shape[2] == 5120 else { throw AetherError.invalidTensor("H3 conditioning must be the [1,T,5120] bf16 tap") }
        let t = conditioning.spec.shape[1]
        // Keyframe anchors lead the video rows: none (T2VA), or first, last or both (FL2VA), as the latents declare them.
        let layout = try H3Layout(geometry: geometry, textTags: tagged.tags, anchors: initial.anchors)
        guard initial.video.spec.shape == [1, layout.videoIndices.count, 96],
              initial.audio.spec.shape == [1, layout.audioIndices.count, 32] else {
            throw AetherError.invalidTensor("H3 initial latents differ from the layout")
        }
        let stepTables = try tables(layout)
        guard stepTables.count == steps, stepTables.allSatisfy({ $0.tables.count == blocks }) else { throw AetherError.invalidModel("H3 step tables differ from the schedule") }
        // One modulation shape for the whole trajectory (see H3ModulationPadding).
        let cardinality = schedule.steps.map { layout.timesteps(for: $0).values.count }.max() ?? 1
        let rotary = try layout.rotary()
        let cos = try h3Tensor([layout.rows, 96], rotary.cos.values), sin = try h3Tensor([layout.rows, 96], rotary.sin.values)
        let gather = try h3Tensor(.int32, [layout.rows], H3Packing.gather(layout))
        let videoIndices = try h3Tensor(.int32, [layout.videoIndices.count], layout.videoIndices)
        let audioIndices = try h3Tensor(.int32, [layout.audioIndices.count], layout.audioIndices)
        try Task.checkCancellation()
        let refined = try await kernels.refine(conditioning)
        guard refined.spec == (try H3TensorSpec(scalarType: .bfloat16, shape: [1, t, 5376])) else { throw AetherError.invalidTensor("H3 refined text shape differs") }
        if let observer { try await observer(.refined(refined)) }
        var latents = initial
        for (step, entry) in zip(schedule.steps, stepTables) {
            try Task.checkCancellation()
            let rows = layout.timesteps(for: step)
            let tags = try h3Tensor(.int32, [layout.rows], rows.adalnIndices)
            let indices = try h3Tensor(.int32, [layout.rows], rows.indices)
            if let observer { try await observer(.stepStarted(step: step.index, latents: latents)) }
            var hidden = try await kernels.pack(video: latents.video, audio: latents.audio, text: refined, gather: gather)
            let hiddenSpec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, layout.rows, 5376])
            guard hidden.spec == hiddenSpec else { throw AetherError.invalidTensor("H3 packed sequence shape differs") }
            if let observer { try await observer(.packed(step: step.index, hidden: hidden)) }
            // Blocks chain on the device; each block's output reaches the host only for an observer.
            let stepIndex = step.index
            var onBlock: (@Sendable (Int, H3TensorData) async throws -> Void)? = nil
            if let observer {
                onBlock = { @Sendable (index: Int, output: H3TensorData) async throws -> Void in
                    try await observer(.block(step: stepIndex, index: index, output: output))
                }
            }
            try Task.checkCancellation()
            hidden = try await kernels.blocks(hidden: hidden, tables: Array(entry.tables.prefix(blocks)), cardinality: cardinality, tags: tags,
                                              cos: cos, sin: sin, onBlock: onBlock)
            guard hidden.spec == hiddenSpec else { throw AetherError.invalidTensor("H3 block output shape differs") }
            try Task.checkCancellation()
            let temb = try H3ModulationPadding.embedding(try entry.embedding.read(), cardinality: cardinality)
            let velocity = try await kernels.heads(hidden: hidden, temb: temb, indices: indices, videoIndices: videoIndices,
                                                   audioIndices: audioIndices, video: latents.video.spec, audio: latents.audio.spec)
            if let observer { try await observer(.heads(step: step.index, video: velocity.video, audio: velocity.audio)) }
            latents = try latents.advanced(schedule: schedule, step: step.index, videoVelocity: velocity.video, audioVelocity: velocity.audio,
                                           conditionVideoRows: layout.conditionVideoRows)
            if let observer { try await observer(.advanced(step: step.index, latents: latents)) }
            onStep(step.index + 1, steps)
        }
        return latents
    }
    func release() async { await kernels.release() }
}

/// One temporal window, decoded as the source's 256 px spatial tiles and blended back.
@available(macOS 27, iOS 27, *)
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

@available(macOS 27, iOS 27, *)
actor H3ProfileAudioDecoder: H3AudioDecoder {
    private let kernel: any H3AudioKernel
    private let hop: Int
    init(kernel: any H3AudioKernel, hop: Int = 800) { self.kernel = kernel; self.hop = hop }
    nonisolated func supports(_ geometry: H3Geometry) -> Bool { kernel.supports(latents: geometry.audioLatentsPerChannel) }
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

/// First-frame encoder graph calls. `whole`: pixels `[1,3,1,H,W]` -> condition `[1,24,1,H/16,W/16]`. `tile`: one 256 px
/// tile `[1,3,1,256,256]` -> moments `[1,48,1,16,16]`. `tail`: the stacked tile moments `[N,48,1,16,16]` (row-major
/// over the source split) -> condition.
@available(macOS 27, iOS 27, *)
protocol H3KeyframeKernels: Sendable {
    func whole(_ model: H3ModelAsset, pixels: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData
    func tile(_ model: H3ModelAsset, pixels: H3TensorData) async throws -> H3TensorData
    func tail(_ model: H3ModelAsset, tiles: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData
    func release() async
}

/// RGB8 keyframe -> planar f32 pixels in 0...255 -> the profile's first-frame encoding for the canvas: the tiled
/// graphs when the manifest declares them for that canvas (one tile's scratch at a time), else the whole-frame graph.
@available(macOS 27, iOS 27, *)
actor H3ProfileKeyframeEncoder: H3KeyframeEncoder {
    private let encoding: @Sendable (Int, Int) -> H3Profile.KeyframeEncoding?
    private let kernels: any H3KeyframeKernels
    init(profile: H3Profile, observer: H3NativeCallObserver? = nil) {
        self.init(encoding: { try? profile.keyframeEncoding(width: $0, height: $1) }, kernels: NativeH3KeyframeKernels(observer: observer))
    }
    init(encoding: @escaping @Sendable (Int, Int) -> H3Profile.KeyframeEncoding?, kernels: any H3KeyframeKernels) {
        self.encoding = encoding; self.kernels = kernels
    }
    nonisolated func supports(_ geometry: H3Geometry) -> Bool { encoding(geometry.width, geometry.height) != nil }
    func encode(_ keyframe: H3Keyframe, geometry: H3Geometry) async throws -> H3TensorData {
        guard keyframe.width == geometry.width, keyframe.height == geometry.height else { throw AetherError.invalidRequest("H3 keyframe must match the clip size") }
        guard let encoding = encoding(geometry.width, geometry.height) else {
            throw AetherError.unsupported("H3 profile has no first-frame encoder for \(geometry.width)x\(geometry.height)")
        }
        let width = geometry.width, height = geometry.height, area = width * height
        var planar = [Float](repeating: 0, count: 3 * area)
        for p in 0..<area { for c in 0..<3 { planar[c * area + p] = Float(keyframe.rgb[p * 3 + c]) } }
        let condition = try H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, geometry.latentHeight, geometry.latentWidth])
        switch encoding {
        case .whole(let model):
            return try await kernels.whole(model, pixels: try h3Tensor([1, 3, 1, height, width], planar), condition: condition)
        case .tiled(let tileModel, let tail, let tiling):
            let side = tiling.tileHeight
            var moments = Data(capacity: tiling.tileCount * 48 * (side / 16) * (side / 16) * 4)
            for y in tiling.yStarts { for x in tiling.xStarts {
                try Task.checkCancellation()
                var tile = [Float](repeating: 0, count: 3 * side * side)
                for c in 0..<3 { for r in 0..<side {
                    let source = c * area + (y + r) * width + x, target = (c * side + r) * side
                    tile[target..<(target + side)] = planar[source..<(source + side)]
                } }
                let value = try await kernels.tile(tileModel, pixels: try h3Tensor([1, 3, 1, side, side], tile))
                guard value.spec == (try H3TensorSpec(scalarType: .float32, shape: [1, 48, 1, side / 16, side / 16])) else {
                    throw AetherError.invalidTensor("H3 first-frame tile moments differ")
                }
                moments.append(value.bytes)
            } }
            let tiles = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [tiling.tileCount, 48, 1, side / 16, side / 16]), bytes: moments)
            return try await kernels.tail(tail, tiles: tiles, condition: condition)
        }
    }
    func release() async { await kernels.release() }
}

/// Native first-frame graphs, each opened on first use and kept until `release()`.
@available(macOS 27, iOS 27, *)
actor NativeH3KeyframeKernels: H3KeyframeKernels {
    private var graphs: [String: NativeH3Graph] = [:]
    private let observer: H3NativeCallObserver?
    init(observer: H3NativeCallObserver? = nil) { self.observer = observer }
    private func graph(_ model: H3ModelAsset) async throws -> NativeH3Graph {
        if let graph = graphs[model.url.path] { return graph }
        let graph = try await NativeH3Graph.open(model, role: "keyframe", observer: observer); graphs[model.url.path] = graph; return graph
    }
    func whole(_ model: H3ModelAsset, pixels: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData {
        try await graph(model).run(["pixels": .data(pixels)], outputs: ["condition": condition])["condition"]!
    }
    func tile(_ model: H3ModelAsset, pixels: H3TensorData) async throws -> H3TensorData {
        let side = pixels.spec.shape[3]
        let spec = try H3TensorSpec(scalarType: .float32, shape: [1, 48, 1, side / 16, side / 16])
        return try await graph(model).run(["pixels": .data(pixels)], outputs: ["moments": spec])["moments"]!
    }
    func tail(_ model: H3ModelAsset, tiles: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData {
        try await graph(model).run(["tiles": .data(tiles)], outputs: ["condition": condition])["condition"]!
    }
    func release() async { graphs = [:] }
}

// MARK: Native kernels

/// Per-call weights, double-buffered: call `i` runs from bank `i % 2` and loads call `i+1`'s weights into the other bank
/// while its GPU work runs (see `NativeH3Graph.run(_:outputs:bank:prefetch:)`).
@available(macOS 27, iOS 27, *)
func h3Banked(_ parameters: [[H3TensorFile]], _ index: Int) -> (files: [String: H3TensorFile], bank: Int) {
    (Dictionary(uniqueKeysWithValues: parameters[index].map { ($0.name, $0) }), index % 2)
}

@available(macOS 27, iOS 27, *)
actor NativeH3ConditionerKernel: H3ConditionerKernel {
    private let conditioner: H3Profile.Conditioner
    private let observer: H3NativeCallObserver?
    private var graph: NativeH3Graph?
    init(_ conditioner: H3Profile.Conditioner, observer: H3NativeCallObserver? = nil) { self.conditioner = conditioner; self.observer = observer }
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        if graph == nil { graph = try await NativeH3Graph.open(conditioner.layer, role: "conditioner", observer: observer) }
        var inputs: [String: NativeH3Graph.Input] = ["hidden": .data(hidden), "cos": .data(cos), "sin": .data(sin)]
        for file in conditioner.parameters[index] { inputs[file.name] = .file(file) }
        // The last layer prefetches nothing: the conditioner is released after it.
        let next = index + 1 < conditioner.parameters.count ? h3Banked(conditioner.parameters, index + 1) : nil
        return try await graph!.run(inputs, outputs: ["hidden_out": hidden.spec], bank: index % 2, prefetch: next)["hidden_out"]!
    }
    func release() async { graph = nil }
}

/// The keyframe vision tower: the canvas's static graph, weights bound in place, kept until `release()`.
@available(macOS 27, iOS 27, *)
actor NativeH3VisionKernel: H3VisionKernel {
    private let vision: H3Profile.ConditionerVision
    private let observer: H3NativeCallObserver?
    private var graph: (canvas: String, graph: NativeH3Graph)?
    init(_ vision: H3Profile.ConditionerVision, observer: H3NativeCallObserver? = nil) { self.vision = vision; self.observer = observer }
    func encode(_ inputs: H3VisionInputs, width: Int, height: Int) async throws -> (embeddings: H3TensorData, deepstack: H3TensorData) {
        let canvas = "\(width)x\(height)"
        if graph?.canvas != canvas {
            graph = nil
            graph = (canvas, try await NativeH3Graph.open(try vision.function(width: width, height: height), role: "vision", observer: observer,
                                                          frequentReshapes: false))
        }
        var bound: [String: NativeH3Graph.Input] = ["pixels": .data(inputs.pixels), "pos_index": .data(inputs.positionIndex),
                                                    "pos_weight": .data(inputs.positionWeight), "patch_rc": .data(inputs.patchCoordinates)]
        for file in vision.parameters { bound[file.name] = .file(file) }
        let tokens = inputs.gridHeight * inputs.gridWidth / (vision.vision.mergeSize * vision.vision.mergeSize)
        let result = try await graph!.graph.run(bound, outputs: [
            "image_embeds": try H3TensorSpec(scalarType: .bfloat16, shape: [tokens, vision.hiddenSize]),
            "deepstack": try H3TensorSpec(scalarType: .bfloat16, shape: [vision.deepstackLevels, tokens, vision.hiddenSize])])
        return (result["image_embeds"]!, result["deepstack"]!)
    }
    func release() async { graph = nil }
}

@available(macOS 27, iOS 27, *)
actor NativeH3DenoiserKernels: H3DenoiserKernels {
    private let profile: H3Profile
    private let prepare: H3Profile.Prepare
    private var packing: NativeH3Graph?, body: NativeH3Graph?, output: NativeH3Graph?
    private let observer: H3NativeCallObserver?
    init(_ profile: H3Profile, observer: H3NativeCallObserver? = nil) throws {
        guard let prepare = profile.prepare else { throw AetherError.invalidModel("H3 profile has no prepare graphs") }
        self.profile = profile; self.prepare = prepare; self.observer = observer
    }
    func refine(_ text: H3TensorData) async throws -> H3TensorData {
        // Once per render: the text graph is dropped right after, so its ~1.6 GB of refiner slots never co-reside with the body.
        let graph = try await NativeH3Graph.open(prepare.text, role: "text", observer: observer)
        var inputs: [String: NativeH3Graph.Input] = ["text": .data(text)]
        for file in prepare.textParameters { inputs[file.name] = .file(file) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, text.spec.shape[1], 5376])
        return try await graph.run(inputs, outputs: ["refined": spec])["refined"]!
    }
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        if packing == nil { packing = try await NativeH3Graph.open(prepare.pack, role: "pack", observer: observer) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [1, gather.spec.shape[0], 5376])
        return try await packing!.run(["video": .data(video), "audio": .data(audio), "text": .data(text), "gather": .data(gather)],
                                      outputs: ["hidden": spec])["hidden"]!
    }
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        if body == nil { body = try await NativeH3Graph.open(profile.body, role: "body", observer: observer) }
        var inputs: [String: NativeH3Graph.Input] = ["hidden": .data(hidden), "temb": .data(table), "tags": .data(tags), "cos": .data(cos), "sin": .data(sin)]
        for file in profile.parameters[index] { inputs[file.name] = .file(file) }
        // Block 49 prefetches block 0 for the next step.
        return try await body!.run(inputs, outputs: ["hidden_out": hidden.spec], bank: index % 2,
                                   prefetch: h3Banked(profile.parameters, (index + 1) % profile.parameters.count))["hidden_out"]!
    }
    /// The step's blocks on one body graph: block 0 copies `hidden` in, every later block binds its predecessor's output
    /// on the device (two alternating buffers), and only the last output (or every output, for `onBlock`) is read back.
    func blocks(hidden: H3TensorData, tables: [H3TensorFile], cardinality: Int, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData,
                onBlock: (@Sendable (Int, H3TensorData) async throws -> Void)?) async throws -> H3TensorData {
        if body == nil { body = try await NativeH3Graph.open(profile.body, role: "body", observer: observer) }
        var last: H3TensorData?
        for (index, table) in tables.enumerated() {
            try Task.checkCancellation()
            let temb = try H3ModulationPadding.table(try table.read(), cardinality: cardinality)
            var inputs: [String: NativeH3Graph.Input] = ["hidden": index == 0 ? .data(hidden) : .chained(hidden.spec), "temb": .data(temb),
                                                         "tags": .data(tags), "cos": .data(cos), "sin": .data(sin)]
            for file in profile.parameters[index] { inputs[file.name] = .file(file) }
            let readback = onBlock != nil || index == tables.count - 1
            // Block 49 prefetches block 0 for the next step.
            let result = try await body!.run(inputs, outputs: ["hidden_out": hidden.spec], bank: index % 2,
                                             prefetch: h3Banked(profile.parameters, (index + 1) % profile.parameters.count),
                                             chain: NativeH3Graph.Chain(output: "hidden_out", readback: readback))
            if let output = result["hidden_out"] {
                try await onBlock?(index, output)
                last = output
            }
        }
        guard let last else { throw AetherError.invalidRequest("H3 step has no blocks") }
        return last
    }
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        if output == nil { output = try await NativeH3Graph.open(profile.heads, role: "heads", observer: observer) }
        let result = try await output!.run(["hidden": .data(hidden), "temb": .data(temb), "indices": .data(indices),
                                            "video_indices": .data(videoIndices), "audio_indices": .data(audioIndices)],
                                           outputs: ["video": video, "audio": audio])
        return (result["video"]!, result["audio"]!)
    }
    func release() async { packing = nil; body = nil; output = nil }
}

@available(macOS 27, iOS 27, *)
actor NativeH3TileKernel: H3TileKernel {
    private let asset: H3ModelAsset
    private let observer: H3NativeCallObserver?
    private var graph: NativeH3Graph?
    init(_ asset: H3ModelAsset, observer: H3NativeCallObserver? = nil) { self.asset = asset; self.observer = observer }
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        if graph == nil { graph = try await NativeH3Graph.open(asset, role: "videoTile", observer: observer) }
        return try await graph!.run(["rows": .data(rows)], outputs: ["pixels": try H3TensorSpec(scalarType: .float32, shape: [1, 3, 22, 256, 256])])["pixels"]!
    }
    func release() async { graph = nil }
}

/// Runs the static-length decoder for the rows' Na (the dynamic asset aborts in MPSGraph on the GPU).
@available(macOS 27, iOS 27, *)
actor NativeH3AudioKernel: H3AudioKernel {
    private let decoder: H3Profile.AudioDecoder
    private let observer: H3NativeCallObserver?
    private var graph: (latents: Int, graph: NativeH3Graph)?
    init(_ decoder: H3Profile.AudioDecoder, observer: H3NativeCallObserver? = nil) { self.decoder = decoder; self.observer = observer }
    /// Native runs use only the static-length decoders.
    nonisolated func supports(latents: Int) -> Bool { decoder.fixed[latents] != nil }
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        let na = rows.spec.shape[1], hop = decoder.hop
        if graph?.latents != na { graph = nil; graph = (na, try await NativeH3Graph.open(try decoder.model(latents: na), role: "audio", observer: observer)) }
        let spec = try H3TensorSpec(scalarType: .float32, shape: [2, hop * rows.spec.shape[1]])
        return try await graph!.graph.run(["rows": .data(rows)], outputs: ["pcm": spec])["pcm"]!
    }
    func release() async { graph = nil }
}

/// Native stages for `H3VideoSession` over a complete schema-3 profile. `tokenize` turns the structured prompt into
/// the conditioner's token ids (`AetherH3Tokenizers.H3Tokenizer.tokenIDs`).
@available(macOS 27, iOS 27, *)
public enum H3ProfileEngine {
    /// Includes exact profile declarations and selected schedule; no model is loaded.
    public static func resourceIdentity(profile: H3Profile, recipe: String = "turbo8") -> String {
        "h3.profile/1:" + profile.manifestSHA256 + ":" + recipe
    }

    /// `observer` (qualification only) receives the denoiser's boundary tensors, and `nativeObserver` (qualification only)
    /// every native submission of every stage's graphs; omit both for generation.
    public static func stages(profile: H3Profile, recipe: String = "turbo8",
                              tokenize: @escaping @Sendable (String) throws -> [Int32],
                              observer: H3DenoiserObserver? = nil, nativeObserver: H3NativeCallObserver? = nil) throws -> H3VideoStages {
        guard profile.complete, let conditioner = profile.conditioner, let audio = profile.audioDecoder else {
            throw AetherError.invalidModel("H3 profile is incomplete: \(profile.missing.joined(separator: ", "))")
        }
        return H3VideoStages(
            conditioner: try H3ProfileConditioner(profile: profile, tokenize: tokenize,
                                                  kernel: NativeH3ConditionerKernel(conditioner, observer: nativeObserver),
                                                  vision: profile.conditionerVision.map { NativeH3VisionKernel($0, observer: nativeObserver) }),
            keyframes: profile.keyframeEncoders.isEmpty && profile.tiledKeyframeEncoder == nil ? nil
                : H3ProfileKeyframeEncoder(profile: profile, observer: nativeObserver),
            denoiser: try H3ProfileDenoiser(profile: profile, recipe: recipe, kernels: try NativeH3DenoiserKernels(profile, observer: nativeObserver),
                                            observer: observer),
            video: H3ProfileVideoDecoder(kernel: NativeH3TileKernel(profile.videoDecoder.model, observer: nativeObserver)),
            audio: H3ProfileAudioDecoder(kernel: NativeH3AudioKernel(audio, observer: nativeObserver), hop: audio.hop),
            resourceIdentity: resourceIdentity(profile: profile, recipe: recipe))
    }
}

/// Per-graph native runs over a profile, for qualification against CPU references (each call opens what it needs and
/// keeps it until `release()`). Inputs are the graphs' own boundary tensors; weights are bound in place.
@available(macOS 27, iOS 27, *)
public actor H3ProfileQualifier {
    private let profile: H3Profile
    private let conditioner: NativeH3ConditionerKernel
    private let denoiser: NativeH3DenoiserKernels
    private let tile: NativeH3TileKernel
    private let audio: NativeH3AudioKernel
    /// `videoDecoder` replaces the profile's tile decoder (decoder experiments; same [448,96] -> [1,3,22,256,256] contract).
    public init(profile: H3Profile, videoDecoder: H3ModelAsset? = nil) throws {
        guard profile.complete, let conditioner = profile.conditioner, let audio = profile.audioDecoder else {
            throw AetherError.invalidModel("H3 profile is incomplete: \(profile.missing.joined(separator: ", "))")
        }
        self.profile = profile; self.conditioner = NativeH3ConditionerKernel(conditioner)
        self.denoiser = try NativeH3DenoiserKernels(profile)
        self.tile = NativeH3TileKernel(videoDecoder ?? profile.videoDecoder.model); self.audio = NativeH3AudioKernel(audio)
    }
    /// The unnormalized tap for `ids`, layer by layer, exactly as the session's conditioner stage computes it.
    public func tap(ids: [Int32], onLayer: @escaping @Sendable (Int, Int) -> Void = { _, _ in }) async throws -> H3TensorData {
        let stage = try H3ProfileConditioner(profile: profile, tokenize: { _ in ids }, kernel: conditioner)
        return try await stage.encode("ids", geometry: try H3Geometry(), onLayer: onLayer)
    }
    public func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        try await conditioner.layer(index, hidden: hidden, cos: cos, sin: sin)
    }
    /// The keyframe vision tower's merger rows and DeepStack for one keyframe at its canvas (opened per call, released after).
    public func vision(_ keyframe: H3Keyframe) async throws -> (embeddings: H3TensorData, deepstack: H3TensorData) {
        guard let entry = profile.conditionerVision else { throw AetherError.unsupported("H3 profile has no keyframe vision encoder") }
        let kernel = NativeH3VisionKernel(entry)
        do {
            let inputs = try H3VisionInputs(rgb: keyframe.rgb, width: keyframe.width, height: keyframe.height, vision: entry.vision)
            let result = try await kernel.encode(inputs, width: keyframe.width, height: keyframe.height)
            await kernel.release(); return result
        } catch { await kernel.release(); throw error }
    }
    /// Upstream's FL2VA presentation of `keyframes` and the prompt `ids`, exactly as the session's conditioner stage
    /// computes it; `tokenize` tokenizes the keyframe labels.
    /// `layers` names decoder layers whose outputs (before any DeepStack add, as upstream's layer hooks record them)
    /// are returned in `outputs`, for locating where a native tap departs from the oracle's.
    public func conditioning(ids: [Int32], keyframes: [H3Keyframe], tokenize: @escaping @Sendable (String) throws -> [Int32],
                             layers: Set<Int> = [], onLayer: @escaping @Sendable (Int, Int) -> Void = { _, _ in })
        async throws -> (conditioning: H3Conditioning, outputs: [Int: H3TensorData]) {
        let prompt = "\u{0}ids"
        let recorder = H3LayerRecorder(inner: conditioner, layers: layers)
        let stage = try H3ProfileConditioner(profile: profile, tokenize: { $0 == prompt ? ids : try tokenize($0) }, kernel: recorder,
                                             vision: profile.conditionerVision.map { NativeH3VisionKernel($0) })
        let conditioning = try await stage.encode(prompt, keyframes: keyframes, geometry: try H3Geometry(), onLayer: onLayer)
        return (conditioning, await recorder.store.outputs)
    }
    public func refine(_ tap: H3TensorData) async throws -> H3TensorData { try await denoiser.refine(tap) }
    public func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        try await denoiser.pack(video: video, audio: audio, text: text, gather: gather)
    }
    public func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        try await denoiser.block(index, hidden: hidden, table: table, tags: tags, cos: cos, sin: sin)
    }
    public func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
                      video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        try await denoiser.heads(hidden: hidden, temb: temb, indices: indices, videoIndices: videoIndices, audioIndices: audioIndices, video: video, audio: audio)
    }
    public func tile(_ rows: H3TensorData) async throws -> H3TensorData { try await tile.decode(rows) }
    /// The first-frame encoder for `keyframe` at `geometry`'s canvas (opened per call, released after).
    public func keyframe(_ keyframe: H3Keyframe, geometry: H3Geometry) async throws -> H3TensorData {
        let encoder = H3ProfileKeyframeEncoder(profile: profile)
        do { let condition = try await encoder.encode(keyframe, geometry: geometry); await encoder.release(); return condition }
        catch { await encoder.release(); throw error }
    }
    public func audio(_ rows: H3TensorData) async throws -> H3TensorData { try await audio.decode(rows) }
    public func release() async { await conditioner.release(); await denoiser.release(); await tile.release(); await audio.release() }
}

@available(macOS 27, iOS 27, *)
extension H3Packing {
    /// Public for probes: the gather permutation of `layout` (see `H3Packing.gather`).
    public static func gatherIndices(_ layout: H3Layout) throws -> [Int32] { try gather(layout) }
}

/// Passes conditioner layers through and keeps the outputs of the listed ones (qualification only).
@available(macOS 27, iOS 27, *)
private struct H3LayerRecorder: H3ConditionerKernel {
    actor Store {
        private(set) var outputs: [Int: H3TensorData] = [:]
        func keep(_ index: Int, _ hidden: H3TensorData) { outputs[index] = hidden }
    }
    let inner: any H3ConditionerKernel
    let layers: Set<Int>
    let store = Store()
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        let output = try await inner.layer(index, hidden: hidden, cos: cos, sin: sin)
        if layers.contains(index) { await store.keep(index, output) }
        return output
    }
    func release() async { await inner.release() }
}
