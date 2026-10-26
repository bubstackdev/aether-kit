import AetherKit
import AetherLanguageCoreAI
import Foundation
import Testing

/// Opt-in native qualification of Qwen3-VL image + text generation through Kit: the vision encoder
/// session and the bucketed decoder's `withImage` path against upstream Transformers
/// (`Forge/language/reference_qwen3_vl.py`) on the reference's own pixel tensors.
///
/// AETHER_NATIVE_QWEN3VL=1
/// AETHER_QWEN3VL_TEXT_MODEL=<text model.aimodel, export-report.json beside it>
/// AETHER_QWEN3VL_VISION_MODEL=<vision.aimodel>
/// AETHER_QWEN3VL_REFERENCE=<reference dir with reference.json>
/// AETHER_QWEN3VL_OUTPUT=<report .json>
private struct VLReference: Decodable {
    struct File: Decodable { let file: String; let shape: [Int]?; let sha256: String }
    struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int; let sha256: String }
    struct Case: Decodable {
        let name: String
        let promptTokens: [Int]
        let grid: [Int]
        let greedyTokens: [Int]
        let forcedTokens: [Int]
        let pixels: File
        let forcedLogits: Logits
    }
    let cases: [Case]
}

private struct VLCaseReport: Codable {
    let name: String
    let promptTokens: Int
    let imageTokens: Int
    let forcedRows: Int
    let forcedMinimumCosine: Double
    let forcedArgmaxMismatches: Int
    let forcedMismatchesAtNearTies: Int
    let greedyTokens: [Int]
    let referenceGreedyTokens: [Int]
    let greedyFirstDivergence: Int?
    let greedyDivergenceReferenceMargin: Double?
    let greedyAccepted: Bool
    let encodeSeconds: Double
    let timeToFirstTokenSeconds: Double
}

private struct VLReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = LanguageGenerationSession.deviceArchitectureName
    var gate = "per forced row cosine >= 0.995 against the float32 upstream reference; argmax equal unless a near-tie (reference top-1 minus top-2 probability < 0.1, native token in the reference top-5); greedy identical up to the reference's end, or first diverging at such a near-tie"
    var hostLoad = "Shared host; timings are single observations"
    var cases: [VLCaseReport] = []
}

private func f32(_ url: URL) throws -> [Float] { try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }

/// (cosine, argmax equal, reference top-1 minus top-2 probability, native token's rank in the reference)
private func compareRow(_ native: [Float], _ reference: ArraySlice<Float>) -> (Double, Bool, Double, Int) {
    var dot = 0.0, nn = 0.0, rr = 0.0
    for (a, b) in zip(native, reference) { dot += Double(a) * Double(b); nn += Double(a) * Double(a); rr += Double(b) * Double(b) }
    let nativeBest = native.indices.max { native[$0] < native[$1] }!
    let referenceBest = reference.indices.max { reference[$0] < reference[$1] }!
    let sorted = reference.map(Double.init).sorted(by: >)
    let denominator = reference.reduce(0.0) { $0 + exp(Double($1) - sorted[0]) }
    let margin = (1 - exp(sorted[1] - sorted[0])) / denominator
    let value = Double(reference[nativeBest])
    return (dot / (nn.squareRoot() * rr.squareRoot()), nativeBest == referenceBest, margin, reference.filter { Double($0) > value }.count)
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_QWEN3VL"] == "1"), .timeLimit(.minutes(30)))
func qwen3VLImageTextQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = VLReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let textModel = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_TEXT_MODEL"]))
        let visionModel = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_VISION_MODEL"]))
        let directory = URL(filePath: try #require(nativeEnvironment["AETHER_QWEN3VL_REFERENCE"]))
        let reference = try JSONDecoder().decode(VLReference.self, from: Data(contentsOf: directory.appendingPathComponent("reference.json")))
        struct Export: Decodable { let exportBounds: LanguageExportBounds }
        let bounds = try JSONDecoder().decode(Export.self, from: Data(contentsOf:
            textModel.deletingLastPathComponent().appendingPathComponent("export-report.json"))).exportBounds
        let tokens = LanguageImageTokens(visionStart: 151652, imagePad: 151655, visionEnd: 151653)
        report.stage = "open"
        let encoder = try await LanguageVisionEncoder.open(at: visionModel)
        let session = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: 2048,
                                                               stopTokenIDs: [151645, 151643], exportBounds: bounds)
        #expect(session.acceptsImages && session.takesRopePositions && session.contract == .bucketedPositions)
        for c in reference.cases {
            report.stage = "case \(c.name)"
            let pixels = try f32(directory.appendingPathComponent(c.pixels.file)).map { Float16($0) }
            let start = ContinuousClock.now
            let features = try await encoder.encode(pixels: pixels, grid: (c.grid[0], c.grid[1], c.grid[2]))
            let encodeDuration = start.duration(to: .now)
            var rendered: [Int] = []
            for id in c.promptTokens where !(id == tokens.imagePad && rendered.last == tokens.imagePad) { rendered.append(id) }
            let prompt = LanguageVisionPrompt(renderedTokenIDs: rendered, images: [features])

            let forced = try f32(directory.appendingPathComponent(c.forcedLogits.file))
            let vocab = c.forcedLogits.vocab
            let rows = try await session.teacherForcedLogits(prompt: prompt, imageTokens: tokens, continuation: Array(c.forcedTokens.dropLast()))
            let compared = rows.enumerated().map { i, row in compareRow(row, forced[(i * vocab)..<((i + 1) * vocab)]) }
            let mismatches = compared.filter { !$0.1 }
            let nearTieMismatches = mismatches.filter { $0.2 < 0.1 && $0.3 < 5 }

            let reply = try await session.generate(prompt: prompt, imageTokens: tokens, maximumTokens: c.greedyTokens.count)
            let divergence = zip(reply.tokens, c.greedyTokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? (reply.tokens.count < c.greedyTokens.count ? reply.tokens.count : nil)
            var margin: Double?
            var accepted = divergence == nil
            if let d = divergence, d < c.forcedLogits.rows, d < reply.tokens.count, Array(c.forcedTokens.prefix(d)) == Array(c.greedyTokens.prefix(d)) {
                var onehot = [Float](repeating: 0, count: vocab)
                onehot[reply.tokens[d]] = 1
                let row = compareRow(onehot, forced[(d * vocab)..<((d + 1) * vocab)])
                margin = row.2
                accepted = row.2 < 0.1 && row.3 < 5
            }
            report.cases.append(VLCaseReport(name: c.name, promptTokens: c.promptTokens.count, imageTokens: features.tokenCount,
                forcedRows: rows.count, forcedMinimumCosine: compared.map(\.0).min() ?? 0, forcedArgmaxMismatches: mismatches.count,
                forcedMismatchesAtNearTies: nearTieMismatches.count, greedyTokens: reply.tokens, referenceGreedyTokens: c.greedyTokens,
                greedyFirstDivergence: divergence, greedyDivergenceReferenceMargin: margin, greedyAccepted: accepted,
                encodeSeconds: Double(encodeDuration.components.seconds) + Double(encodeDuration.components.attoseconds) / 1e18,
                timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds))
        }
        await session.unload()
        await encoder.unload()
        report.passed = !report.cases.isEmpty && report.cases.allSatisfy {
            $0.forcedMinimumCosine >= 0.995 && $0.forcedArgmaxMismatches == $0.forcedMismatchesAtNearTies && $0.greedyAccepted
        }
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
