@preconcurrency import CoreAI
import AetherVision
import CryptoKit
import Foundation
import Metal
import Testing

/// Opt-in native qualification of the Qwen3-VL vision encoder asset (`encode`) against the upstream
/// CPU reference (`Forge/language/reference_qwen3_vl_vision.py`). Pixels come from the reference;
/// the per-patch tables are computed by `AetherVision.Qwen3VLVisionTables`.
///
/// AETHER_NATIVE_QWEN3VL_VISION=1
/// AETHER_QWEN3VL_VISION_MODEL=<vision.aimodel>
/// AETHER_QWEN3VL_VISION_REFERENCE=<reference dir with reference.json>
/// AETHER_QWEN3VL_VISION_OUTPUT=<report .json>
private struct VisionReference: Decodable {
    struct Image: Decodable {
        let name: String
        let grid: [Int]
        let patches: Int
        let tokens: Int
        let files: [String: File]
    }
    struct File: Decodable { let file: String; let shape: [Int]?; let sha256: String }
    let images: [Image]
}

private struct EncodeResult: Codable {
    let image: String
    let grid: [Int]
    let imageEmbedsCosine: Double
    let deepstackCosine: [Double]
    let firstEncodeSeconds: Double
    let repeatEncodeSeconds: Double
}

private struct VisionReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = AIModel.deviceArchitectureName
    var modelFileSHA256 = ""
    var referenceSHA256 = ""
    var gate = "imageEmbeds and every DeepStack level cosine >= 0.999 vs the float32 upstream reference, GPU-preferred, specialized without expectFrequentReshapes"
    var hostLoad = "Shared host; encode timings are single observations, not benchmark records"
    var reshapesOff: [EncodeResult] = []
    var reshapesOnDiagnostic: [EncodeResult] = []
}

private func sha256File(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
}

private func floats(_ url: URL) throws -> [Float] {
    try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func cosine(_ a: [Float], _ b: ArraySlice<Float>) -> Double {
    var dot = 0.0, aa = 0.0, bb = 0.0
    for (x, y) in zip(a, b) { dot += Double(x) * Double(y); aa += Double(x) * Double(x); bb += Double(y) * Double(y) }
    return dot / (aa.squareRoot() * bb.squareRoot())
}

private func readFloat16(_ array: NDArray) -> [Float] {
    array.view(as: Float16.self).withUnsafePointer { pointer, shape, strides in
        // Row-major traversal over any strides.
        var out: [Float] = []
        let dims = (0..<shape.count).map { shape[$0] }, step = (0..<strides.count).map { strides[$0] }
        out.reserveCapacity(dims.reduce(1, *))
        var index = [Int](repeating: 0, count: dims.count)
        for _ in 0..<dims.reduce(1, *) {
            out.append(Float(pointer[zip(index, step).reduce(0) { $0 + $1.0 * $1.1 }]))
            var d = dims.count - 1
            while d >= 0 { index[d] += 1; if index[d] < dims[d] { break }; index[d] = 0; d -= 1 }
        }
        return out
    }
}

private func encodeAll(model url: URL, reshapes: Bool, reference: VisionReference, directory: URL) async throws -> [EncodeResult] {
    var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
    options.expectFrequentReshapes = reshapes
    let model = try await AIModel(contentsOf: url, options: options)
    let function = try #require(try model.loadFunction(named: "encode"))
    let queue = try #require(MTLCreateSystemDefaultDevice()?.makeCommandQueue())
    let stream = ComputeStream(commandQueue: queue)
    var results: [EncodeResult] = []
    for image in reference.images {
        let pixels = try floats(directory.appendingPathComponent(image.files["pixels"]!.file)).map { Float16($0) }
        let tables = try Qwen3VLVisionTables(t: image.grid[0], h: image.grid[1], w: image.grid[2])
        let n = tables.patchCount
        let inputs: [String: InferenceFunction.AsyncValue] = [
            "pixels": .init(NDArray(scalars: pixels, shape: [n, pixels.count / n])),
            "pos_index": .init(NDArray(scalars: tables.positionIndex, shape: [n, 4])),
            "pos_weight": .init(NDArray(scalars: tables.positionWeight, shape: [n, 4])),
            "patch_rc": .init(NDArray(scalars: tables.patchCoordinates, shape: [n, 2])),
        ]
        var seconds: [Double] = []
        var outputs: [String: InferenceFunction.AsyncValue] = [:]
        for _ in 0..<2 {
            let start = ContinuousClock.now
            outputs = try function.encode(inputs: inputs, states: InferenceFunction.AsyncMutableViews(), to: stream)
            await stream.currentWorkCompleted()
            let elapsed = start.duration(to: .now)
            seconds.append(Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18)
        }
        let embeds = readFloat16(try #require(try await outputs["image_embeds"]?.ndArray))
        let deepstack = readFloat16(try #require(try await outputs["deepstack"]?.ndArray))
        let refEmbeds = try floats(directory.appendingPathComponent(image.files["embeds"]!.file))
        let refDeep = try floats(directory.appendingPathComponent(image.files["deepstack"]!.file))
        guard embeds.count == refEmbeds.count, deepstack.count == refDeep.count, refDeep.count % 3 == 0 else {
            throw NativeFailure(description: "encoder output shape mismatch for \(image.name)")
        }
        let level = refDeep.count / 3
        results.append(EncodeResult(image: image.name, grid: image.grid,
            imageEmbedsCosine: cosine(embeds, refEmbeds[...]),
            deepstackCosine: (0..<3).map { cosine(Array(deepstack[($0 * level)..<(($0 + 1) * level)]), refDeep[($0 * level)..<(($0 + 1) * level)]) },
            firstEncodeSeconds: seconds[0], repeatEncodeSeconds: seconds[1]))
    }
    return results
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN3VL_VISION"] == "1"), .timeLimit(.minutes(20)))
func qwen3VLVisionEncoderQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = VisionReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let model = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_MODEL"]))
        let directory = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_REFERENCE"]))
        let referenceURL = directory.appendingPathComponent("reference.json")
        let reference = try JSONDecoder().decode(VisionReference.self, from: Data(contentsOf: referenceURL))
        report.modelFileSHA256 = try sha256File(model.appendingPathComponent("main.mlirb"))
        report.referenceSHA256 = try sha256File(referenceURL)
        for image in reference.images {
            for (_, file) in image.files where try sha256File(directory.appendingPathComponent(file.file)) != file.sha256 {
                throw NativeFailure(description: "reference file hash mismatch: \(file.file)")
            }
        }
        report.stage = "reshapes off"
        report.reshapesOff = try await encodeAll(model: model, reshapes: false, reference: reference, directory: directory)
        report.stage = "reshapes on (diagnostic)"
        report.reshapesOnDiagnostic = try await encodeAll(model: model, reshapes: true, reference: reference, directory: directory)
        report.passed = report.reshapesOff.count == reference.images.count
            && report.reshapesOff.allSatisfy { $0.imageEmbedsCosine >= 0.999 && $0.deepstackCosine.allSatisfy { $0 >= 0.999 } }
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
