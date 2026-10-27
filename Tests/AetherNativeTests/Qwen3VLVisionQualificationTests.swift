import AetherKit
import AetherLanguageCoreAI
import AetherVision
import CryptoKit
import Darwin
import Foundation
import Testing

/// Opt-in native qualification of the Qwen3-VL vision encoder asset: one static-shape function per
/// fixed patch grid, driven through `LanguageVisionEncoder` (GPU-preferred, specialized without
/// `expectFrequentReshapes`).
///
/// Parity (`qwen3VLVisionEncoderQualification`) against the upstream CPU reference
/// (`Forge/language/reference_qwen3_vl_vision.py`, one synthetic scene per grid at the grid's exact
/// pixel size, so upstream and Kit see the same pixels):
///
/// AETHER_NATIVE_QWEN3VL_VISION=1
/// AETHER_QWEN3VL_BUNDLE=<bundle with a vision registration listing the grids>
/// AETHER_QWEN3VL_VISION_REFERENCE=<reference dir with reference.json>
/// AETHER_QWEN3VL_VISION_OUTPUT=<report .json>
///
/// Stability (`qwen3VLVisionEncoderStability`): the same photo resized to every grid, encoded
/// round-robin across the grids, each output compared bit for bit with that grid's first encode.
/// A dynamic-shape export failed this way on 2.5 % of encodes (always the same wrong bits).
///
/// AETHER_NATIVE_QWEN3VL_STABILITY=1
/// AETHER_QWEN3VL_BUNDLE=<bundle>
/// AETHER_QWEN3VL_STABILITY_PHOTO=<image file>
/// AETHER_QWEN3VL_STABILITY_ENCODES=<encodes per grid, default 200>
/// AETHER_QWEN3VL_STABILITY_OUTPUT=<report .json>
private struct VisionReference: Decodable {
    struct Image: Decodable {
        let name: String
        let grid: [Int]
        let tokens: Int
        let files: [String: File]
    }
    struct File: Decodable { let file: String; let shape: [Int]?; let sha256: String }
    let images: [Image]
}

private struct ParityResult: Codable {
    let image: String
    let grid: [Int]
    let function: String
    let imageEmbedsCosine: Double
    let deepstackCosine: [Double]
    let firstEncodeSeconds: Double
    let repeatEncodeSeconds: Double
}

private struct FootprintSample: Codable {
    let phase: String
    let bytes: UInt64
}

private struct ParityReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = LanguageGenerationSession.deviceArchitectureName
    var modelFileSHA256 = ""
    var referenceSHA256 = ""
    var grids: [String] = []
    var gate = "per fixed grid: imageEmbeds and every DeepStack level cosine >= 0.999 vs the float32 upstream reference on the same pixels; every registered grid covered"
    var hostLoad = "Shared host; encode timings are single observations, not benchmark records"
    var results: [ParityResult] = []
    var footprint: [FootprintSample] = []
}

private struct GridStability: Codable {
    let function: String
    let encodes: Int
    let differingFromFirst: Int
    let distinctOutputs: Int
    let firstEmbedsHash: String
    let firstDeepstackHash: String
    let differing: [Differing]
    let meanEncodeSeconds: Double
    struct Differing: Codable {
        let encode: Int
        let embedsHash: String
        let deepstackHash: String
        let embedsCosine: Double
        let deepstackCosine: Double
    }
}

private struct StabilityReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = LanguageGenerationSession.deviceArchitectureName
    var modelFileSHA256 = ""
    var photoSHA256 = ""
    var gate = "every encode of every fixed grid bit-identical to that grid's first encode (FNV-1a 64 over the fp16 bits, then full comparison)"
    var order = "round-robin across the grids, one process"
    var grids: [GridStability] = []
    var footprint: [FootprintSample] = []
}

private func sha256File(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
}

private func floats(_ url: URL) throws -> [Float] {
    try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

private func cosine<A: Collection, B: Collection>(_ a: A, _ b: B) -> Double
    where A.Element: BinaryFloatingPoint, B.Element: BinaryFloatingPoint {
    var dot = 0.0, aa = 0.0, bb = 0.0
    for (x, y) in zip(a, b) { let p = Double(x), q = Double(y); dot += p * q; aa += p * p; bb += q * q }
    return dot / (aa.squareRoot() * bb.squareRoot())
}

private func fnv(_ values: [Float16]) -> String {
    var h: UInt64 = 0xcbf2_9ce4_8422_2325
    for v in values { h = (h ^ UInt64(v.bitPattern)) &* 0x100_0000_01b3 }
    return String(h, radix: 16)
}

private func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}

private func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }

/// The bundle's registration, its verified vision asset and the encoder over its grids.
private func openEncoder(bundle: URL) throws -> (LanguageVisionRegistration, URL, [Qwen3VLImagePreprocessor.Grid]) {
    let manifest = try LanguageModelManifest.load(from: bundle)
    let registration = try #require(manifest.vision)
    try registration.validate()
    let visionModel = try #require(try manifest.verifyVision(in: bundle))
    let grids = (registration.imageGrids ?? []).map { Qwen3VLImagePreprocessor.Grid(height: $0.height, width: $0.width) }
    return (registration, visionModel, grids)
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN3VL_VISION"] == "1"), .timeLimit(.minutes(20)))
func qwen3VLVisionEncoderQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = ParityReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_BUNDLE"]))
        let directory = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_REFERENCE"]))
        let referenceURL = directory.appendingPathComponent("reference.json")
        let reference = try JSONDecoder().decode(VisionReference.self, from: Data(contentsOf: referenceURL))
        let (registration, visionModel, grids) = try openEncoder(bundle: bundle)
        report.modelFileSHA256 = try sha256File(visionModel.appendingPathComponent("main.mlirb"))
        report.referenceSHA256 = try sha256File(referenceURL)
        report.grids = grids.map { LanguageVisionEncoder.functionName(for: $0) }
        for image in reference.images {
            for (_, file) in image.files where try sha256File(directory.appendingPathComponent(file.file)) != file.sha256 {
                throw NativeFailure(description: "reference file hash mismatch: \(file.file)")
            }
        }
        report.footprint.append(.init(phase: "before open", bytes: footprint()))
        report.stage = "open"
        let encoder = try await LanguageVisionEncoder.open(at: visionModel,
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide, imageGrids: grids))
        report.footprint.append(.init(phase: "after open", bytes: footprint()))
        var warmed = false
        for image in reference.images {
            report.stage = "image \(image.name)"
            let pixels = try floats(directory.appendingPathComponent(image.files["pixels"]!.file)).map { Float16($0) }
            let grid = (image.grid[0], image.grid[1], image.grid[2])
            var durations: [Double] = []
            var features: LanguageImageFeatures?
            for _ in 0..<2 {
                let start = ContinuousClock.now
                features = try await encoder.encode(pixels: pixels, grid: grid)
                durations.append(seconds(start.duration(to: .now)))
            }
            let f = try #require(features)
            let refEmbeds = try floats(directory.appendingPathComponent(image.files["embeds"]!.file))
            let refDeep = try floats(directory.appendingPathComponent(image.files["deepstack"]!.file))
            guard f.embeddings.count == refEmbeds.count, f.deepstack.count == refDeep.count, f.deepstackLevels > 0 else {
                throw NativeFailure(description: "encoder output shape mismatch for \(image.name)")
            }
            let level = refDeep.count / f.deepstackLevels
            report.results.append(ParityResult(image: image.name, grid: image.grid,
                function: LanguageVisionEncoder.functionName(for: .init(height: image.grid[1], width: image.grid[2])),
                imageEmbedsCosine: cosine(f.embeddings, refEmbeds),
                deepstackCosine: (0..<f.deepstackLevels).map {
                    cosine(f.deepstack[($0 * level)..<(($0 + 1) * level)], refDeep[($0 * level)..<(($0 + 1) * level)])
                },
                firstEncodeSeconds: durations[0], repeatEncodeSeconds: durations[1]))
            warmed = Set(report.results.map(\.function)).count == grids.count
            if warmed && !report.footprint.contains(where: { $0.phase == "all grids warmed" }) {
                report.footprint.append(.init(phase: "all grids warmed", bytes: footprint()))
            }
        }
        await encoder.unload()
        report.footprint.append(.init(phase: "after unload", bytes: footprint()))
        let covered = Set(report.results.map(\.function)) == Set(report.grids)
        report.passed = covered && warmed
            && report.results.allSatisfy { $0.imageEmbedsCosine >= 0.999 && $0.deepstackCosine.allSatisfy { $0 >= 0.999 } }
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN3VL_STABILITY"] == "1"), .timeLimit(.minutes(30)))
func qwen3VLVisionEncoderStability() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = StabilityReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_STABILITY_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_BUNDLE"]))
        let photo = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_STABILITY_PHOTO"]))
        let count = Int(nativeEnvironment["AETHER_QWEN3VL_STABILITY_ENCODES"] ?? "200") ?? 200
        let (registration, visionModel, grids) = try openEncoder(bundle: bundle)
        report.modelFileSHA256 = try sha256File(visionModel.appendingPathComponent("main.mlirb"))
        report.photoSHA256 = try sha256File(photo)
        let image = try VisionRGBImage(contentsOf: photo)
        report.stage = "preprocess"
        var inputs: [(grid: (t: Int, h: Int, w: Int), pixels: [Float16])] = []
        for grid in grids {
            var configuration = VisionLanguageSession.preprocessing(for: registration)
            configuration.imageGrids = [grid]                                     // force this grid
            let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                 configuration: configuration)
            inputs.append((processed.grid, processed.pixels.map { Float16($0) }))
        }
        report.footprint.append(.init(phase: "before open", bytes: footprint()))
        report.stage = "open"
        let encoder = try await LanguageVisionEncoder.open(at: visionModel,
            configuration: .init(mergeSize: registration.mergeSize, positionTableSide: registration.positionTableSide, imageGrids: grids))
        report.footprint.append(.init(phase: "after open", bytes: footprint()))
        var first: [LanguageImageFeatures?] = Array(repeating: nil, count: grids.count)
        var differing: [[GridStability.Differing]] = Array(repeating: [], count: grids.count)
        var outputs: [Set<String>] = Array(repeating: [], count: grids.count)
        var elapsed = [Double](repeating: 0, count: grids.count)
        for i in 0..<count {
            report.stage = "encode \(i)"
            for (g, input) in inputs.enumerated() {
                let start = ContinuousClock.now
                let f = try await encoder.encode(pixels: input.pixels, grid: input.grid)
                elapsed[g] += seconds(start.duration(to: .now))
                let embedsHash = fnv(f.embeddings), deepstackHash = fnv(f.deepstack)
                outputs[g].insert(embedsHash + "/" + deepstackHash)
                guard let reference = first[g] else { first[g] = f; continue }
                if f.embeddings != reference.embeddings || f.deepstack != reference.deepstack {
                    differing[g].append(.init(encode: i, embedsHash: embedsHash, deepstackHash: deepstackHash,
                                              embedsCosine: cosine(f.embeddings, reference.embeddings),
                                              deepstackCosine: cosine(f.deepstack, reference.deepstack)))
                }
            }
            if i == 0 { report.footprint.append(.init(phase: "all grids warmed", bytes: footprint())) }
        }
        report.footprint.append(.init(phase: "after all encodes", bytes: footprint()))
        await encoder.unload()
        report.grids = grids.indices.map { g in
            GridStability(function: LanguageVisionEncoder.functionName(for: grids[g]), encodes: count,
                          differingFromFirst: differing[g].count, distinctOutputs: outputs[g].count,
                          firstEmbedsHash: fnv(first[g]?.embeddings ?? []), firstDeepstackHash: fnv(first[g]?.deepstack ?? []),
                          differing: differing[g], meanEncodeSeconds: elapsed[g] / Double(max(count, 1)))
        }
        report.passed = count > 0 && report.grids.allSatisfy { $0.differingFromFirst == 0 && $0.distinctOutputs == 1 }
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
