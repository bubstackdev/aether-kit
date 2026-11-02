@testable import AetherH3
import AetherKit
import Foundation
import Testing
@testable import AetherH3CoreAI

private let zeroHash = String(repeating: "0", count: 64)

private func scratch() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    return root
}
private func bf16(_ values: [Float]) -> Data {
    let bits = values.map { UInt16($0.bitPattern >> 16) }
    return bits.withUnsafeBytes { Data($0) }
}
private func file(_ root: URL, _ name: String, _ data: Data, _ type: H3ScalarType, _ shape: [Int]) throws -> H3TensorFile {
    let url = root.appendingPathComponent(name); try data.write(to: url)
    return try H3TensorFile(name: name, url: url, spec: H3TensorSpec(scalarType: type, shape: shape), sha256: h3SHA256(data))
}
private func floats(_ tensor: H3TensorData) throws -> [Float] { try tensor.floatValues() }

private actor Calls {
    private(set) var log: [String] = []
    private(set) var released = 0
    func add(_ entry: String) { log.append(entry) }
    func release() { released += 1 }
}

// MARK: Conditioner

private struct LayerKernel: H3ConditionerKernel {
    let calls: Calls
    /// Each layer adds 1.0 to every value (exact in bf16 for small integers) and checks the rotary prefix it was given.
    let expectedCos: Data
    func layer(_ index: Int, hidden: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        await calls.add("layer \(index)")
        #expect(cos.bytes == expectedCos && cos.spec.shape == [1, hidden.spec.shape[1], 128] && sin.spec == cos.spec)
        return try H3TensorData(spec: hidden.spec, bytes: bf16(try hidden.floatValues().map { $0 + 1 }))
    }
    func release() async { await calls.release() }
}

@Test func h3ProfileConditionerGathersEmbeddingRowsInPlaceAndStreamsEveryLayer() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    // A 6-row table behind a 16-byte header, as embed_tokens sits inside a safetensors shard.
    let rows = (0..<6).map { r in [Float](repeating: Float(r), count: 5120) }
    let table = bf16(rows.flatMap { $0 })
    let shard = root.appendingPathComponent("shard.safetensors"); try (Data(repeating: 7, count: 16) + table).write(to: shard)
    let ropeValues = (0..<(4096 * 128)).map { Float($0 % 7) }
    let cos = try file(root, "cos", bf16(ropeValues), .bfloat16, [4096, 128]), sin = try file(root, "sin", bf16(ropeValues.map { -$0 }), .bfloat16, [4096, 128])
    let placeholder = try H3TensorFile(name: "w0", url: shard, spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), sha256: zeroHash)
    let conditioner = H3Profile.Conditioner(
        embedding: try H3EmbeddingTable(url: shard, offset: 16, rows: 6, width: 5120, byteCount: table.count, sha256: h3SHA256(table)),
        layer: try H3ModelAsset(url: root, function: "main", sha256: zeroHash),
        parameters: Array(repeating: [placeholder], count: 50), tap: 50, ropeCos: cos, ropeSin: sin)
    let calls = Calls()
    let ids: [Int32] = [5, 0, 3]
    let stage = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: { _ in ids },
                                     kernel: LayerKernel(calls: calls, expectedCos: bf16(Array(ropeValues.prefix(3 * 128)))))
    let progress = Calls()
    let tap = try await stage.encode("p", geometry: try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)) { layer, total in
        Task { await progress.add("\(layer)/\(total)") }
    }
    #expect(tap.spec.shape == [1, 3, 5120])
    #expect(try floats(tap) == ids.flatMap { [Float](repeating: Float($0) + 50, count: 5120) })
    #expect(await calls.log == (0..<50).map { "layer \($0)" })
    await stage.release()
    #expect(await calls.released == 1)
    // Out-of-range ids, too few tokens and a tampered table fail closed.
    let bad = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: { _ in [6, 0] }, kernel: LayerKernel(calls: calls, expectedCos: Data()))
    await #expect(throws: AetherError.self) { _ = try await bad.encode("p", geometry: try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)) { _, _ in } }
    let short = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: { _ in [1] }, kernel: LayerKernel(calls: calls, expectedCos: Data()))
    await #expect(throws: AetherError.self) { _ = try await short.encode("p", geometry: try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)) { _, _ in } }
    try (Data(repeating: 7, count: 16) + bf16(rows.flatMap { $0 }.map { $0 + 1 })).write(to: shard)
    let tampered = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: { _ in ids }, kernel: LayerKernel(calls: calls, expectedCos: Data()))
    await #expect(throws: AetherError.self) { _ = try await tampered.encode("p", geometry: try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)) { _, _ in } }
}

private struct VisionKernel: H3VisionKernel {
    let calls: Calls
    /// Merger rows of 10 and DeepStack level k of k + 1, exact in bf16.
    func encode(_ inputs: H3VisionInputs, width: Int, height: Int) async throws -> (embeddings: H3TensorData, deepstack: H3TensorData) {
        await calls.add("vision \(width)x\(height)")
        let tokens = inputs.gridHeight * inputs.gridWidth / 4
        let embeddings = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [tokens, 5120]),
                                          bytes: bf16([Float](repeating: 10, count: tokens * 5120)))
        let deepstack = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [3, tokens, 5120]),
                                         bytes: bf16((0..<3).flatMap { [Float](repeating: Float($0 + 1), count: tokens * 5120) }))
        return (embeddings, deepstack)
    }
    func release() async { await calls.add("vision release") }
}

@Test func h3ProfileConditionerPresentsKeyframesWithImageRowsMRoPEAndDeepStack() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let rows = (0..<9).map { r in [Float](repeating: Float(r), count: 5120) }
    let table = bf16(rows.flatMap { $0 })
    let shard = root.appendingPathComponent("shard.safetensors"); try (Data(repeating: 7, count: 16) + table).write(to: shard)
    let ropeValues: [Float] = (0..<(4096 * 128)).map { (i: Int) -> Float in Float((i / 128) % 97) + Float(i % 128) / 128 }
    let cos = try file(root, "cos", bf16(ropeValues), .bfloat16, [4096, 128]), sin = try file(root, "sin", bf16(ropeValues.map { -$0 }), .bfloat16, [4096, 128])
    let placeholder = try H3TensorFile(name: "w0", url: shard, spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), sha256: zeroHash)
    let conditioner = H3Profile.Conditioner(
        embedding: try H3EmbeddingTable(url: shard, offset: 16, rows: 9, width: 5120, byteCount: table.count, sha256: h3SHA256(table)),
        layer: try H3ModelAsset(url: root, function: "main", sha256: zeroHash),
        parameters: Array(repeating: [placeholder], count: 50), tap: 50, ropeCos: cos, ropeSin: sin)
    // Small stand-in ids for the vision tokens, so the test table stays small.
    var vision = H3Presentation.Vision()
    vision.visionStart = 6; vision.imagePad = 7; vision.visionEnd = 8
    let entry = H3Profile.ConditionerVision(functions: [:], parameters: [], vision: vision, deepstackLevels: 3, hiddenSize: 5120)
    let tokenize: @Sendable (String) throws -> [Int32] = { $0 == H3Presentation.Vision.label(1) ? [4, 5] : [1, 2, 3] }
    let presentation = try H3Presentation(prompt: [1, 2, 3], keyframes: [(288, 288)], vision: vision, tokenize: tokenize)
    let expectedCos = try presentation.rotary(try cos.read(), section: vision.mropeSection).bytes
    let calls = Calls()
    let stage = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: tokenize,
                                     kernel: LayerKernel(calls: calls, expectedCos: expectedCos), vision: (entry, VisionKernel(calls: calls)))
    let keyframe = try H3Keyframe(width: 288, height: 288, rgb: [UInt8](repeating: 9, count: 288 * 288 * 3))
    let geometry = try H3Geometry(width: 288, height: 288, requestedFrames: 39, diagnostic: true)
    let result = try await stage.encode("p", keyframes: [keyframe], geometry: geometry) { _, _ in }
    #expect(result.tags == presentation.tags)
    #expect(result.tap.spec.shape == [1, presentation.count, 5120])
    // The tower runs and is released before the first layer.
    #expect(await Array(calls.log.prefix(3)) == ["vision 288x288", "vision release", "layer 0"])
    // Image rows: 10, then +1 per layer and +k+1 after layer k < 3: 10 + 50 + 1 + 2 + 3. Text rows: their id + 50.
    let image = try #require(presentation.images.first)
    var expected: [Float] = []
    for (row, id) in presentation.ids.enumerated() {
        let value: Float = (image.start..<(image.start + image.tokens)).contains(row) ? 66 : Float(id) + 50
        expected += [Float](repeating: value, count: 5120)
    }
    #expect(try floats(result.tap) == expected)
    // Without a vision entry, keyframes are refused rather than presented as text.
    let textOnly = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: tokenize, kernel: LayerKernel(calls: calls, expectedCos: Data()))
    await #expect(throws: AetherError.self) { _ = try await textOnly.encode("p", keyframes: [keyframe], geometry: geometry) { _, _ in } }
    // A presentation longer than the profile's text limit is refused before any native call.
    let before = await calls.log.count
    let tight = H3ProfileConditioner(conditioner: conditioner, limits: 2...64, tokenize: tokenize, kernel: LayerKernel(calls: calls, expectedCos: Data()),
                                     vision: (entry, VisionKernel(calls: calls)))
    await #expect(throws: AetherError.self) { _ = try await tight.encode("p", keyframes: [keyframe], geometry: geometry) { _, _ in } }
    #expect(await calls.log.count == before)
}

@Test func h3ProfileConditionerPresentsFirstAndLastKeyframesAtTheirOwnOffsets() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let rows = (0..<9).map { r in [Float](repeating: Float(r), count: 5120) }
    let table = bf16(rows.flatMap { $0 })
    let shard = root.appendingPathComponent("shard.safetensors"); try (Data(repeating: 7, count: 16) + table).write(to: shard)
    let ropeValues: [Float] = (0..<(4096 * 128)).map { (i: Int) -> Float in Float((i / 128) % 97) + Float(i % 128) / 128 }
    let cos = try file(root, "cos", bf16(ropeValues), .bfloat16, [4096, 128]), sin = try file(root, "sin", bf16(ropeValues), .bfloat16, [4096, 128])
    let placeholder = try H3TensorFile(name: "w0", url: shard, spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), sha256: zeroHash)
    let conditioner = H3Profile.Conditioner(
        embedding: try H3EmbeddingTable(url: shard, offset: 16, rows: 9, width: 5120, byteCount: table.count, sha256: h3SHA256(table)),
        layer: try H3ModelAsset(url: root, function: "main", sha256: zeroHash),
        parameters: Array(repeating: [placeholder], count: 50), tap: 50, ropeCos: cos, ropeSin: sin)
    var vision = H3Presentation.Vision()
    vision.visionStart = 6; vision.imagePad = 7; vision.visionEnd = 8
    let entry = H3Profile.ConditionerVision(functions: [:], parameters: [], vision: vision, deepstackLevels: 3, hiddenSize: 5120)
    let tokenize: @Sendable (String) throws -> [Int32] = {
        switch $0 { case H3Presentation.Vision.label(1): [4, 5]; case H3Presentation.Vision.label(2): [4, 3]; default: [1, 2] }
    }
    let presentation = try H3Presentation(prompt: [1, 2], keyframes: [(288, 288), (288, 288)], vision: vision, tokenize: tokenize)
    #expect(presentation.images.map(\.start) == [3, 3 + 81 + 1 + 3])
    let calls = Calls()
    let stage = H3ProfileConditioner(conditioner: conditioner, limits: 2...4096, tokenize: tokenize,
                                     kernel: LayerKernel(calls: calls, expectedCos: try presentation.rotary(try cos.read(), section: vision.mropeSection).bytes),
                                     vision: (entry, VisionKernel(calls: calls)))
    let frame = try H3Keyframe(width: 288, height: 288, rgb: [UInt8](repeating: 9, count: 288 * 288 * 3))
    let result = try await stage.encode("p", keyframes: [frame, frame], geometry: try H3Geometry(width: 288, height: 288, requestedFrames: 39, diagnostic: true)) { _, _ in }
    #expect(await Array(calls.log.prefix(4)) == ["vision 288x288", "vision 288x288", "vision release", "layer 0"])
    #expect(result.tags == presentation.tags)
    // Both images' rows carry the scatter and all three DeepStack adds; the label and text rows between them do not.
    let image = Set(presentation.images.flatMap { $0.start..<($0.start + $0.tokens) })
    let expected = presentation.ids.enumerated().flatMap { row, id in [Float](repeating: image.contains(row) ? 66 : Float(id) + 50, count: 5120) }
    #expect(try floats(result.tap) == expected)
}

@Test func h3ImageRowsAddDeepStackInFloat32RoundedToNearestEvenBF16() throws {
    // Ties at 256 (spacing 2): 256 + 1 rounds to even 256, 256 + 3 to even 260, as torch's bf16 add does.
    let hidden = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 3, 2]), bytes: bf16([256, 256, 0.5, 2, 9, 9]))
    let deepstack = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 2, 2]), bytes: bf16([1, 3, 0.25, -2]))
    let image = H3Presentation.Image(start: 0, tokens: 2, gridHeight: 2, gridWidth: 4)
    let out = try H3ImageRows.add([deepstack], level: 0, levels: 1, to: hidden, at: [image])
    #expect(try floats(out) == [256, 260, 0.75, 0, 9, 9])
}

// MARK: Packing and denoising

@Test func h3ModulationPaddingAppendsCopiesThatTheAdaLNSelectionNeverReads() throws {
    // [6, 3, 4] (k = 1) padded to k = 2: each slice keeps its rows and repeats them; selecting any tag < 3 from the
    // padded table returns the original row, which is why padding leaves block outputs unchanged.
    let values = (0..<(6 * 3 * 4)).map { Float($0) }
    let table = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [6, 3, 4]), bytes: bf16(values))
    let padded = try H3ModulationPadding.table(table, cardinality: 2)
    #expect(padded.spec.shape == [6, 6, 4])
    let p = try padded.floatValues()
    for slice in 0..<6 { for row in 0..<6 { for d in 0..<4 {
        #expect(p[(slice * 6 + row) * 4 + d] == values[(slice * 3 + row % 3) * 4 + d])
    } } }
    // Already at the target, the tensor is returned unchanged; a larger cardinality than the target is refused.
    let unchanged = try H3ModulationPadding.table(padded, cardinality: 2)
    #expect(unchanged.bytes == padded.bytes)
    #expect(throws: AetherError.self) { _ = try H3ModulationPadding.table(padded, cardinality: 1) }
    #expect(throws: AetherError.self) {
        _ = try H3ModulationPadding.table(try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [6, 4, 4]), bytes: Data(count: 6 * 4 * 4 * 2)), cardinality: 2)
    }
    // Embedding [2, 3] -> [3, 3]: rows 0, 1, then a copy of row 0.
    let embedding = try h3Tensor([2, 3], [1, 2, 3, 4, 5, 6])
    let wide = try H3ModulationPadding.embedding(embedding, cardinality: 3)
    let wideValues = try wide.floatValues()
    #expect(wide.spec.shape == [3, 3] && wideValues == [1, 2, 3, 4, 5, 6, 1, 2, 3])
}

@Test func h3PackingGatherIsTheInversePermutationOfTheLayoutRows() throws {
    let layout = try H3Layout(geometry: try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true), textTags: Array(repeating: 1, count: 5))
    let gather = try H3Packing.gather(layout)
    #expect(Set(gather) == Set(0..<Int32(layout.rows)))
    let t = layout.textIndices.count, v = layout.videoIndices.count
    for (j, r) in layout.textIndices.enumerated() { #expect(gather[Int(r)] == Int32(j)) }
    for (j, r) in layout.videoIndices.enumerated() { #expect(gather[Int(r)] == Int32(t + j)) }
    for (j, r) in layout.audioIndices.enumerated() { #expect(gather[Int(r)] == Int32(t + v + j)) }
}

private struct TrajectoryKernels: H3DenoiserKernels {
    let calls: Calls
    let layout: H3Layout
    /// The trajectory's largest timestep cardinality: every step's table and embedding arrive padded to it.
    var cardinality = 2
    /// Per step index, the AdaLN rows the block must receive as `tags` (checked when present).
    var adaln: [Int: [Int32]] = [:]
    func refine(_ text: H3TensorData) async throws -> H3TensorData {
        await calls.add("refine")
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, text.spec.shape[1], 5376]), bytes: Data(count: text.spec.shape[1] * 5376 * 2))
    }
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        await calls.add("pack")
        #expect(gather.bytes == (try H3Packing.gather(layout)).withUnsafeBytes { Data($0) })
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, layout.rows, 5376]), bytes: Data(count: layout.rows * 5376 * 2))
    }
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        // The table's first value names its step; tags are the layout's AdaLN rows for that step.
        let step = Int(try table.floatValues()[0])
        await calls.add("block \(step).\(index)")
        #expect(cos.spec.shape == [layout.rows, 96] && sin.spec.shape == [layout.rows, 96] && tags.spec.shape == [layout.rows])
        #expect(table.spec.shape == [6, 3 * cardinality, 5376])
        if let expected = adaln[step] { #expect(tags.bytes == expected.withUnsafeBytes { Data($0) }, "step \(step)") }
        return hidden
    }
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        let step = try temb.floatValues()[0]
        await calls.add("heads \(Int(step))")
        #expect(temb.spec.shape == [cardinality, 2688])
        #expect(videoIndices.bytes == layout.videoIndices.withUnsafeBytes { Data($0) })
        return (try h3Tensor(video.shape, [Float](repeating: step + 1, count: video.byteCount / 4)),
                try h3Tensor(audio.shape, [Float](repeating: -step - 1, count: audio.byteCount / 4)))
    }
    func release() async { await calls.release() }
}

@Test func h3ProfileDenoiserRefinesOnceThenPacksAndStreamsEveryBlockForEveryStep() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 4))
    let schedule = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    var steps: [H3Profile.Step] = []
    for step in schedule.steps {
        let k = layout.timesteps(for: step).values.count
        let table = try file(root, "table-\(step.index)", bf16([Float](repeating: Float(step.index), count: 6 * 3 * k * 5376)), .bfloat16, [6, 3 * k, 5376])
        let temb = try file(root, "temb-\(step.index)", [Float](repeating: Float(step.index), count: k * 2688).withUnsafeBytes { Data($0) }, .float32, [k, 2688])
        steps.append(H3Profile.Step(timesteps: layout.timesteps(for: step).values, embedding: temb, tables: Array(repeating: table, count: 3)))
    }
    let calls = Calls()
    let frozen = steps
    let denoiser = H3ProfileDenoiser(schedule: schedule, blocks: 3, tables: { _ in frozen }, kernels: TrajectoryKernels(calls: calls, layout: layout))
    let initial = try geometry.initialLatents(seed: 9)
    let conditioning = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 2))
    let result = try await denoiser.denoise(initial, conditioning: conditioning, geometry: geometry) { _, _ in }
    var expected = ["refine"]
    for s in 0..<4 { expected += ["pack"] + (0..<3).map { "block \(s).\($0)" } + ["heads \(s)"] }
    #expect(await calls.log == expected)
    var latents = initial
    for s in 0..<4 {
        latents = try latents.advanced(schedule: schedule, step: s, videoVelocity: try h3Tensor(initial.video.spec.shape, [Float](repeating: Float(s + 1), count: initial.video.spec.byteCount / 4)),
                                       audioVelocity: try h3Tensor(initial.audio.spec.shape, [Float](repeating: Float(-s - 1), count: initial.audio.spec.byteCount / 4)))
    }
    #expect(result.video.bytes == latents.video.bytes && result.audio.bytes == latents.audio.bytes)
    // Wrong conditioning precision or latents from another geometry are refused before any kernel runs.
    let before = await calls.log.count
    let wide = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 4))
    await #expect(throws: AetherError.self) { _ = try await denoiser.denoise(initial, conditioning: wide, geometry: geometry) { _, _ in } }
    await #expect(throws: AetherError.self) {
        _ = try await denoiser.denoise(try H3Geometry(width: 96, height: 32, requestedFrames: 39, diagnostic: true).initialLatents(seed: 1),
                                       conditioning: conditioning, geometry: geometry) { _, _ in }
    }
    #expect(await calls.log.count == before)
}

@Test func h3ProfileDenoiserLaysOutTheConditioningsVisionRowsWithTheVideoTag() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    // Text rows 1 and 2 are a keyframe's vision block: tag 0, as upstream's FL2VA step tags them.
    let tags: [Int32] = [1, 0, 0, 1]
    let layout = try H3Layout(geometry: geometry, textTags: tags), text = try H3Layout(geometry: geometry, textTags: [1, 1, 1, 1])
    let schedule = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    var steps: [H3Profile.Step] = [], adaln: [Int: [Int32]] = [:]
    for step in schedule.steps {
        let rows = layout.timesteps(for: step), k = rows.values.count
        adaln[step.index] = rows.adalnIndices
        // The vision rows read the video modulation row of their timestep; text-only tags would read the text one.
        #expect(rows.adalnIndices[1] == rows.indices[1] * 3 && rows.adalnIndices[2] == rows.indices[2] * 3)
        #expect(rows.adalnIndices != text.timesteps(for: step).adalnIndices)
        let table = try file(root, "table-\(step.index)", bf16([Float](repeating: Float(step.index), count: 6 * 3 * k * 5376)), .bfloat16, [6, 3 * k, 5376])
        let temb = try file(root, "temb-\(step.index)", [Float](repeating: Float(step.index), count: k * 2688).withUnsafeBytes { Data($0) }, .float32, [k, 2688])
        steps.append(H3Profile.Step(timesteps: rows.values, embedding: temb, tables: Array(repeating: table, count: 3)))
    }
    let calls = Calls(), frozen = steps
    var kernels = TrajectoryKernels(calls: calls, layout: layout); kernels.adaln = adaln
    let denoiser = H3ProfileDenoiser(schedule: schedule, blocks: 3, tables: { _ in frozen }, kernels: kernels)
    let tap = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 2))
    _ = try await denoiser.denoise(try geometry.initialLatents(seed: 9), conditioning: try H3Conditioning(tap: tap, tags: tags), geometry: geometry) { _, _ in }
    #expect(await calls.log.filter { $0.hasPrefix("block") }.count == 4 * 3)
}

/// Checks every block receives the anchored layout's rotary tables (a last anchor sits at another rotary time than a
/// first one with the same row count).
private struct RotaryCheckingKernels: H3DenoiserKernels {
    let inner: TrajectoryKernels
    let cos: Data, sin: Data
    func refine(_ text: H3TensorData) async throws -> H3TensorData { try await inner.refine(text) }
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        try await inner.pack(video: video, audio: audio, text: text, gather: gather)
    }
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        #expect(cos.bytes == self.cos && sin.bytes == self.sin)
        return try await inner.block(index, hidden: hidden, table: table, tags: tags, cos: cos, sin: sin)
    }
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        try await inner.heads(hidden: hidden, temb: temb, indices: indices, videoIndices: videoIndices, audioIndices: audioIndices, video: video, audio: audio)
    }
    func release() async { await inner.release() }
}

@Test(arguments: [[H3Anchor.first], [.last], [.first, .last]])
func h3ProfileDenoiserLaysOutTheAnchorsTheLatentsDeclareAndNeverStepsTheirRows(_ anchors: [H3Anchor]) async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 4), anchors: anchors)
    let schedule = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    var steps: [H3Profile.Step] = []
    for step in schedule.steps {
        let k = layout.timesteps(for: step).values.count
        let table = try file(root, "table-\(step.index)", bf16([Float](repeating: Float(step.index), count: 6 * 3 * k * 5376)), .bfloat16, [6, 3 * k, 5376])
        let temb = try file(root, "temb-\(step.index)", [Float](repeating: Float(step.index), count: k * 2688).withUnsafeBytes { Data($0) }, .float32, [k, 2688])
        steps.append(H3Profile.Step(timesteps: layout.timesteps(for: step).values, embedding: temb, tables: Array(repeating: table, count: 3)))
    }
    let frozen = steps
    let condition = try h3Tensor([1, 24, 1, geometry.latentHeight, geometry.latentWidth],
                                 [Float](repeating: 0.5, count: 24 * geometry.latentHeight * geometry.latentWidth))
    let initial = try geometry.initialLatents(seed: 9, conditions: anchors.map { ($0, condition) })
    let rotary = try layout.rotary()
    let calls = Calls()
    // Anchored layouts have two distinct row timesteps at step 0 and three afterwards.
    let kernels = RotaryCheckingKernels(inner: TrajectoryKernels(calls: calls, layout: layout, cardinality: 3),
                                        cos: rotary.cos.values.withUnsafeBytes { Data($0) }, sin: rotary.sin.values.withUnsafeBytes { Data($0) })
    let denoiser = H3ProfileDenoiser(schedule: schedule, blocks: 3, tables: { got in
        #expect(got.positions.values == layout.positions.values && got.conditionVideoRows == layout.conditionVideoRows)
        return frozen
    }, kernels: kernels)
    let conditioning = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 2))
    let result = try await denoiser.denoise(initial, conditioning: conditioning, geometry: geometry) { _, _ in }
    #expect(result.anchors == anchors)
    let anchorBytes = layout.conditionVideoRows * 96 * 4
    #expect(result.video.bytes.prefix(anchorBytes) == initial.video.bytes.prefix(anchorBytes))
    #expect(result.video.bytes.dropFirst(anchorBytes) != initial.video.bytes.dropFirst(anchorBytes))
    // The same rows declared with no anchors no longer match the layout and are refused before any kernel runs.
    let before = await calls.log.count
    let undeclared = try H3Latents(video: initial.video, audio: initial.audio)
    await #expect(throws: AetherError.self) { _ = try await denoiser.denoise(undeclared, conditioning: conditioning, geometry: geometry) { _, _ in } }
    #expect(await calls.log.count == before)
}

private actor Observations {
    private(set) var events: [String] = []
    private(set) var first: H3Latents?, last: H3Latents?
    func add(_ event: H3DenoiserObservation) {
        switch event {
        case .refined: events.append("refined")
        case .stepStarted(let step, let latents): events.append("start \(step)"); if step == 0 { first = latents }
        case .packed(let step, _): events.append("packed \(step)")
        case .block(let step, let index, _): events.append("block \(step).\(index)")
        case .heads(let step, let video, _): events.append("heads \(step) \(Int(try! video.floatValues()[0]))")
        case .advanced(let step, let latents): events.append("advanced \(step)"); last = latents
        }
    }
}
private struct ObserverFailure: Error {}

/// Kernels that run a step's blocks in one `blocks` call (as the native kernels chain them on the device), recording
/// whether per-block outputs were requested.
private struct ChainedKernels: H3DenoiserKernels {
    let inner: TrajectoryKernels
    let calls: Calls
    func refine(_ text: H3TensorData) async throws -> H3TensorData { try await inner.refine(text) }
    func pack(video: H3TensorData, audio: H3TensorData, text: H3TensorData, gather: H3TensorData) async throws -> H3TensorData {
        try await inner.pack(video: video, audio: audio, text: text, gather: gather)
    }
    func block(_ index: Int, hidden: H3TensorData, table: H3TensorData, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData) async throws -> H3TensorData {
        Issue.record("the denoiser must run a step's blocks through blocks(...)"); return hidden
    }
    func blocks(hidden: H3TensorData, tables: [H3TensorFile], cardinality: Int, tags: H3TensorData, cos: H3TensorData, sin: H3TensorData,
                onBlock: (@Sendable (Int, H3TensorData) async throws -> Void)?) async throws -> H3TensorData {
        #expect(cardinality == inner.cardinality)
        let step = Int(try tables[0].read().floatValues()[0])
        await calls.add("blocks \(step) x\(tables.count) readback=\(onBlock != nil)")
        for index in tables.indices { try await onBlock?(index, hidden) }
        return hidden
    }
    func heads(hidden: H3TensorData, temb: H3TensorData, indices: H3TensorData, videoIndices: H3TensorData, audioIndices: H3TensorData,
               video: H3TensorSpec, audio: H3TensorSpec) async throws -> (video: H3TensorData, audio: H3TensorData) {
        try await inner.heads(hidden: hidden, temb: temb, indices: indices, videoIndices: videoIndices, audioIndices: audioIndices, video: video, audio: audio)
    }
    func release() async { await inner.release() }
}

@Test func h3ProfileDenoiserRunsEachStepsBlocksInOneChainAndReadsBackOnlyForAnObserver() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 4))
    let schedule = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    var steps: [H3Profile.Step] = []
    for step in schedule.steps {
        let k = layout.timesteps(for: step).values.count
        let table = try file(root, "table-\(step.index)", bf16([Float](repeating: Float(step.index), count: 6 * 3 * k * 5376)), .bfloat16, [6, 3 * k, 5376])
        let temb = try file(root, "temb-\(step.index)", [Float](repeating: Float(step.index), count: k * 2688).withUnsafeBytes { Data($0) }, .float32, [k, 2688])
        steps.append(H3Profile.Step(timesteps: layout.timesteps(for: step).values, embedding: temb, tables: Array(repeating: table, count: 3)))
    }
    let frozen = steps
    let initial = try geometry.initialLatents(seed: 9)
    let conditioning = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 2))
    func run(_ observer: H3DenoiserObserver?) async throws -> [String] {
        let calls = Calls()
        let kernels = ChainedKernels(inner: TrajectoryKernels(calls: calls, layout: layout), calls: calls)
        let denoiser = H3ProfileDenoiser(schedule: schedule, blocks: 3, tables: { _ in frozen }, kernels: kernels, observer: observer)
        _ = try await denoiser.denoise(initial, conditioning: conditioning, geometry: geometry) { _, _ in }
        return await calls.log
    }
    var expected = ["refine"]
    for s in 0..<4 { expected += ["pack", "blocks \(s) x3 readback=false", "heads \(s)"] }
    #expect(try await run(nil) == expected)
    let seen = Observations()
    let observed = try await run { event in await seen.add(event) }
    #expect(observed == expected.map { $0.replacingOccurrences(of: "readback=false", with: "readback=true") })
    #expect(await seen.events.filter { $0.hasPrefix("block ") }.count == 12)
}

@Test func h3ProfileDenoiserObserverIsOptInAndLeavesTheTrajectoryUnchanged() async throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let layout = try H3Layout(geometry: geometry, textTags: Array(repeating: 1, count: 4))
    let schedule = try H3Schedule.turbo(steps: 4, videoShift: 12, audioShift: 3)
    var steps: [H3Profile.Step] = []
    for step in schedule.steps {
        let k = layout.timesteps(for: step).values.count
        let table = try file(root, "table-\(step.index)", bf16([Float](repeating: Float(step.index), count: 6 * 3 * k * 5376)), .bfloat16, [6, 3 * k, 5376])
        let temb = try file(root, "temb-\(step.index)", [Float](repeating: Float(step.index), count: k * 2688).withUnsafeBytes { Data($0) }, .float32, [k, 2688])
        steps.append(H3Profile.Step(timesteps: layout.timesteps(for: step).values, embedding: temb, tables: Array(repeating: table, count: 3)))
    }
    let frozen = steps
    let initial = try geometry.initialLatents(seed: 9)
    let conditioning = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 5120]), bytes: Data(count: 4 * 5120 * 2))
    func run(_ observer: H3DenoiserObserver?) async throws -> (log: [String], result: H3Latents) {
        let calls = Calls()
        let denoiser = H3ProfileDenoiser(schedule: schedule, blocks: 3, tables: { _ in frozen }, kernels: TrajectoryKernels(calls: calls, layout: layout),
                                         observer: observer)
        let result = try await denoiser.denoise(initial, conditioning: conditioning, geometry: geometry) { _, _ in }
        return (await calls.log, result)
    }
    // The default path (no observer) and an observed run issue exactly the same kernel calls, so observing adds no
    // native call or readback, and they produce identical latents.
    let plain = try await run(nil)
    let seen = Observations()
    let observed = try await run { event in await seen.add(event) }
    #expect(observed.log == plain.log)
    #expect(observed.result.video.bytes == plain.result.video.bytes && observed.result.audio.bytes == plain.result.audio.bytes)
    var expected = ["refined"]
    for s in 0..<4 { expected += ["start \(s)", "packed \(s)"] + (0..<3).map { "block \(s).\($0)" } + ["heads \(s) \(s + 1)", "advanced \(s)"] }
    #expect(await seen.events == expected)
    #expect(await seen.first?.video.bytes == initial.video.bytes)
    let last = await seen.last
    #expect(last?.video.bytes == plain.result.video.bytes && last?.audio.bytes == plain.result.audio.bytes)
    // An observer error fails the trajectory at that boundary.
    await #expect(throws: ObserverFailure.self) {
        _ = try await run { event in if case .block(1, 2, _) = event { throw ObserverFailure() } }
    }
}

// MARK: First-frame encoder

private struct KeyframeKernels: H3KeyframeKernels {
    let calls: Calls
    func whole(_ model: H3ModelAsset, pixels: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData {
        await calls.add("whole \(model.url.lastPathComponent) \(pixels.spec.shape)")
        return try h3Tensor(condition.shape, [Float](repeating: 7, count: condition.byteCount / 4))
    }
    func tile(_ model: H3ModelAsset, pixels: H3TensorData) async throws -> H3TensorData {
        // Each tile's moments carry its first (channel 0, top-left) pixel, so the tail can check order and crop.
        let origin = try pixels.floatValues()[0]
        await calls.add("tile \(model.url.lastPathComponent) \(pixels.spec.shape) \(Int(origin))")
        return try h3Tensor([1, 48, 1, 16, 16], [Float](repeating: origin, count: 48 * 256))
    }
    func tail(_ model: H3ModelAsset, tiles: H3TensorData, condition: H3TensorSpec) async throws -> H3TensorData {
        let values = try tiles.floatValues()
        let firsts = stride(from: 0, to: values.count, by: 48 * 256).map { Int(values[$0]) }
        await calls.add("tail \(model.url.lastPathComponent) \(tiles.spec.shape) \(firsts)")
        return try h3Tensor(condition.shape, [Float](repeating: 9, count: condition.byteCount / 4))
    }
    func release() async { await calls.release() }
}

private func asset(_ name: String) throws -> H3ModelAsset {
    try H3ModelAsset(url: URL(fileURLWithPath: "/nonexistent/\(name).aimodel"), function: "main", sha256: zeroHash)
}

@Test func h3KeyframeEncoderRunsTheWholeFrameGraphWithoutTiledEntries() async throws {
    let geometry = try H3Geometry(width: 288, height: 288, requestedFrames: 22, diagnostic: true)
    let whole = try asset("whole")
    let calls = Calls()
    let encoder = H3ProfileKeyframeEncoder(encoding: { w, h in w == 288 && h == 288 ? .whole(whole) : nil }, kernels: KeyframeKernels(calls: calls))
    #expect(encoder.supports(geometry))
    #expect(!encoder.supports(try H3Geometry(width: 320, height: 288, requestedFrames: 22, diagnostic: true)))
    let rgb = (0..<(288 * 288 * 3)).map { UInt8($0 % 251) }
    let condition = try await encoder.encode(try H3Keyframe(width: 288, height: 288, rgb: rgb), geometry: geometry)
    #expect(try condition.spec.shape == [1, 24, 1, 18, 18] && condition.floatValues()[0] == 7)
    #expect(await calls.log == ["whole whole.aimodel [1, 3, 1, 288, 288]"])
    await encoder.release()
    #expect(await calls.released == 1)
}

@Test func h3KeyframeEncoderRunsEverySourceTileThenTheTailWhenTheManifestDeclaresThem() async throws {
    let geometry = try H3Geometry(width: 288, height: 288, requestedFrames: 22, diagnostic: true)
    let tiling = try H3SpatialTiling.source(width: 288, height: 288)
    #expect(tiling.yStarts == [0, 32] && tiling.xStarts == [0, 32])
    let tile = try asset("tile256"), tail = try asset("tail-288x288")
    let calls = Calls()
    let encoder = H3ProfileKeyframeEncoder(encoding: { _, _ in .tiled(tile: tile, tail: tail, tiling: tiling) }, kernels: KeyframeKernels(calls: calls))
    // Channel 0 at (y, x) is (7y + 3x) mod 251; channels 1 and 2 differ, so a channel mix-up would show.
    var rgb = [UInt8](repeating: 0, count: 288 * 288 * 3)
    for y in 0..<288 { for x in 0..<288 { for c in 0..<3 { rgb[(y * 288 + x) * 3 + c] = UInt8((7 * y + 3 * x + 50 * c) % 251) } } }
    let condition = try await encoder.encode(try H3Keyframe(width: 288, height: 288, rgb: rgb), geometry: geometry)
    #expect(try condition.spec.shape == [1, 24, 1, 18, 18] && condition.floatValues()[0] == 9)
    // Row-major over the source split; each tile is the 256 px crop at its start.
    let origins = tiling.yStarts.flatMap { y in tiling.xStarts.map { x in (7 * y + 3 * x) % 251 } }
    #expect(await calls.log == origins.map { "tile tile256.aimodel [1, 3, 1, 256, 256] \($0)" } + ["tail tail-288x288.aimodel [4, 48, 1, 16, 16] \(origins)"])
}

// MARK: Decoders

private struct TileKernel: H3TileKernel {
    let calls: Calls
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        await calls.add("tile")
        // A tile is a constant equal to its first latent value, so the stitched window traces the tile grid.
        let value = try rows.floatValues()[0]
        return try h3Tensor([1, 3, 22, 256, 256], [Float](repeating: value, count: 3 * 22 * 256 * 256))
    }
    func release() async { await calls.release() }
}

@Test func h3ProfileVideoDecoderDecodesEverySourceTileAndBlendsLikeTheStitcher() async throws {
    let geometry = try H3Geometry(width: 288, height: 288, requestedFrames: 22, diagnostic: true)
    let tiling = try H3SpatialTiling.source(width: 288, height: 288)
    let packed = (0..<(7 * geometry.rowsPerLatentFrame * 96)).map { Float($0 % 97) / 97 }
    let calls = Calls()
    let window = try await H3ProfileVideoDecoder(kernel: TileKernel(calls: calls)).decode(windowRows: packed, index: 0, geometry: geometry)
    #expect(await calls.log.count == tiling.tileCount)
    var reference = try H3SpatialStitcher(tiling: tiling, width: 288, height: 288)
    for tile in 0..<tiling.tileCount {
        let rows = try tiling.extract(width: 288, height: 288, packed: packed, index: tile)
        try reference.push([Float](repeating: rows[0], count: 3 * 22 * 256 * 256))
    }
    #expect(window == (try reference.finish()))
}

private struct AudioKernel: H3AudioKernel {
    let calls: Calls
    func decode(_ rows: H3TensorData) async throws -> H3TensorData {
        await calls.add("audio \(rows.spec.shape)")
        let n = 800 * rows.spec.shape[1]
        return try h3Tensor([2, n], [Float](repeating: 0.25, count: n) + [Float](repeating: -0.5, count: n))
    }
    func release() async { await calls.release() }
}

@Test func h3ProfileAudioDecoderReshapesChannelMajorRowsAndSplitsPlanarStereo() async throws {
    let geometry = try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true)
    let na = geometry.audioLatentsPerChannel, calls = Calls()
    let latents = try h3Tensor([1, 2 * na, 32], (0..<(2 * na * 32)).map(Float.init))
    let pcm = try await H3ProfileAudioDecoder(kernel: AudioKernel(calls: calls)).decode(latents, geometry: geometry)
    #expect(await calls.log == ["audio [2, \(na), 32]"])
    #expect(pcm.sampleCount == 800 * na && pcm.channels[0].allSatisfy { $0 == 0.25 } && pcm.channels[1].allSatisfy { $0 == -0.5 })
    await #expect(throws: AetherError.self) {
        _ = try await H3ProfileAudioDecoder(kernel: AudioKernel(calls: calls)).decode(try h3Tensor([1, 2 * na - 2, 32], [Float](repeating: 0, count: (2 * na - 2) * 32)), geometry: geometry)
    }
}

/// Review L6: native audio decoding supports exactly the profile's static lengths, so the session can refuse others first.
@Test func h3NativeAudioDecodingSupportsOnlyTheProfilesStaticLengths() throws {
    let fixed = try [37: asset("audio-na37"), 207: asset("audio-na207")]
    let profileDecoder = H3Profile.AudioDecoder(model: try asset("audio"), fixed: fixed, sampleRate: 32000, hop: 800, channels: 2)
    let decoder = H3ProfileAudioDecoder(kernel: NativeH3AudioKernel(profileDecoder))
    let fiveSeconds = try H3Geometry(width: 960, height: 544, requestedFrames: 120)
    let longer = try H3Geometry(width: 960, height: 544, requestedFrames: 125)
    #expect(fiveSeconds.audioLatentsPerChannel == 207 && longer.frames == 141 && longer.audioLatentsPerChannel == 235)
    #expect(decoder.supports(fiveSeconds) && !decoder.supports(longer))
}

/// Opt-in, CPU only: with `AETHER_H3_PROFILE` set to a measured profile on its measured device, the session's
/// admission comes from the recorded memory measurements, one envelope per entry; elsewhere it is refused.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"] != nil))
func h3MeasuredAdmissionForTheExportedProfile() throws {
    let profile = try H3Profile(url: URL(fileURLWithPath: ProcessInfo.processInfo.environment["AETHER_H3_PROFILE"]!))
    let identity = H3ProfileEngine.resourceIdentity(profile: profile)
    let measured = try? H3MeasuredResources.bundledEntries(executionIdentity: identity, deviceArchitecture: H3ProfileEngine.deviceArchitecture)
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes)
    if measured == nil {
        #expect(throws: AetherError.self) { _ = try H3ProfileEngine.measuredAdmission(profile: profile, coordinator: budget) }
    } else {
        let admission = try H3ProfileEngine.measuredAdmission(profile: profile, coordinator: budget)
        #expect(admission.executionIdentity == identity && admission.envelopes.count == measured!.count)
    }
}

@Test func h3ProfileConditionerPreflightRefusesWhatItCannotPresentBeforeLoading() throws {
    let root = try scratch(); defer { try? FileManager.default.removeItem(at: root) }
    let placeholder = try H3TensorFile(name: "w0", url: root, spec: H3TensorSpec(scalarType: .bfloat16, shape: [1]), sha256: zeroHash)
    let conditioner = H3Profile.Conditioner(
        embedding: try H3EmbeddingTable(url: root, offset: 0, rows: 9, width: 5120, byteCount: 9 * 5120 * 2, sha256: zeroHash),
        layer: try H3ModelAsset(url: root, function: "main", sha256: zeroHash),
        parameters: Array(repeating: [placeholder], count: 50), tap: 50, ropeCos: placeholder, ropeSin: placeholder)
    var vision = H3Presentation.Vision()
    vision.visionStart = 6; vision.imagePad = 7; vision.visionEnd = 8
    let entry = H3Profile.ConditionerVision(functions: ["288x288": try H3ModelAsset(url: root, function: "main", sha256: zeroHash)], parameters: [],
                                            vision: vision, deepstackLevels: 3, hiddenSize: 5120)
    let tokenize: @Sendable (String) throws -> [Int32] = { $0.hasPrefix("<Picture") ? [4, 5] : [1, 2, 3] }
    let calls = Calls()
    func stage(limits: ClosedRange<Int> = 2...4096, rows: ClosedRange<Int>? = nil, vision withVision: Bool = true) -> H3ProfileConditioner {
        H3ProfileConditioner(conditioner: conditioner, limits: limits, tokenize: tokenize, kernel: LayerKernel(calls: calls, expectedCos: Data()),
                             vision: withVision ? (entry, VisionKernel(calls: calls)) : nil, rowLimits: rows)
    }
    let g = try H3Geometry(width: 288, height: 288, requestedFrames: 39, diagnostic: true)
    let frame = try H3Keyframe(width: 288, height: 288, rgb: [UInt8](repeating: 9, count: 288 * 288 * 3))
    let first = try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: frame)
    let both = try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: frame, lastFrame: frame)
    // Canvases: a keyframe needs a tower graph for the canvas; T2VA needs none.
    #expect(stage().supports(g, keyframes: [frame]))
    #expect(stage().supports(try H3Geometry(width: 544, height: 960, requestedFrames: 39, diagnostic: true), keyframes: []))
    let other = try H3Geometry(width: 544, height: 960, requestedFrames: 39, diagnostic: true)
    #expect(!stage().supports(other, keyframes: [try H3Keyframe(width: 544, height: 960, rgb: [UInt8](repeating: 9, count: 544 * 960 * 3))]))
    #expect(!stage(vision: false).supports(g, keyframes: [frame]))
    // The presentation's length: 3 prompt + (2 + 1 + 81 + 1) per 288x288 keyframe.
    #expect(throws: Never.self) { try stage().preflight(first) }
    #expect(throws: AetherError.self) { try stage(limits: 2...64).preflight(first) }
    #expect(throws: AetherError.self) { try stage(vision: false).preflight(first) }
    // Packed rows: the layout of the presentation, its anchors, video and audio, against the denoiser's limit.
    let needed = try H3Layout(geometry: g, textTags: Array(repeating: 1, count: 3 + 2 * 85), anchors: [.first, .last]).rows
    #expect(throws: Never.self) { try stage(rows: 1...needed).preflight(both) }
    #expect(throws: AetherError.self) { try stage(rows: 1...(needed - 1)).preflight(both) }
    #expect(throws: Never.self) { try stage(rows: 1...(needed - 1)).preflight(first) }
}
