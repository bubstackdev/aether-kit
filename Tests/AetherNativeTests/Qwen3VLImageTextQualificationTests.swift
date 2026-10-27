import AetherKit
import AetherLanguageCoreAI
import AetherVision
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
/// AETHER_QWEN3VL_PREPROCESS=swift (optional: decode and preprocess the reference images with Kit)
/// AETHER_QWEN3VL_INPUTS=same-decode (with swift: the reference was made from Kit's own decode of each photo, saved as a
///   lossless PNG, so both sides see identical RGB and the run is gated at the model-parity bar. Otherwise the reference
///   decoded the JPEGs itself (PIL, libjpeg-turbo): that run measures the decoder difference and is reported, not gated.)
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
        let image: File
    }
    let cases: [Case]
    let maximumPixels: Int?
    /// The fixed patch grids [h, w] the reference snapped images to (`--grids`).
    let imageGrids: [[Int]]
    var grids: [Qwen3VLImagePreprocessor.Grid] { imageGrids.map { .init(height: $0[0], width: $0[1]) } }
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
    /// "reference": the reference processor's pixel tensor; "swift": Kit's decoding and preprocessing.
    let preprocessing: String
    let swiftGridMatches: Bool?
    let swiftPixelCosine: Double?
}

private struct VLReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = LanguageGenerationSession.deviceArchitectureName
    var gate = "reference pixels (model parity): per forced row cosine >= 0.995 against the float32 upstream reference, argmax equal unless a near-tie (reference top-1 minus top-2 probability < 0.1, native token in the reference top-5), greedy identical up to the reference end (its final stop token excluded) or first diverging at such a near-tie. Kit preprocessing on the same decode (end to end): grid equal, pixel cosine >= 0.9999 against the reference processor (resampler only), then the model-parity rule. Kit preprocessing on an independent decode: reported, not gated; passed records grid equal, pixel cosine >= 0.999 and forced rows cosine >= 0.98"
    /// Kit preprocessing only: "same-decode" (the reference saw Kit's decode) or "independent-decode".
    var inputs: String?
    /// False for an independent-decode run: its `passed` is reported, not asserted, and it cannot back a session run.
    var gated = true
    var hostLoad = "Shared host; timings are single observations"
    var cases: [VLCaseReport] = []
}

private func f32(_ url: URL) throws -> [Float] { try Data(contentsOf: url).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) } }

/// (cosine, argmax equal, reference top-1 minus top-2 probability, native token's rank in the reference)
private func compareRow(_ native: [Float], _ slice: ArraySlice<Float>) -> (Double, Bool, Double, Int) {
    let reference = Array(slice)                      // zero-based, like `native`
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
        let swiftPreprocessing = nativeEnvironment["AETHER_QWEN3VL_PREPROCESS"] == "swift"
        let sameDecode = swiftPreprocessing && nativeEnvironment["AETHER_QWEN3VL_INPUTS"] == "same-decode"
        if swiftPreprocessing {
            report.inputs = sameDecode ? "same-decode" : "independent-decode"
            report.gated = sameDecode
        }
        report.stage = "open"
        let encoder = try await LanguageVisionEncoder.open(at: visionModel, configuration: .init(imageGrids: reference.grids))
        let session = try await LanguageGenerationSession.open(at: textModel, maximumContextLength: 2048,
                                                               stopTokenIDs: [151645, 151643], exportBounds: bounds)
        #expect(session.acceptsImages && session.takesRopePositions && session.contract == .bucketedPositions)
        for c in reference.cases {
            report.stage = "case \(c.name)"
            let referencePixels = try f32(directory.appendingPathComponent(c.pixels.file))
            var pixels = referencePixels.map { Float16($0) }
            var grid = (c.grid[0], c.grid[1], c.grid[2])
            var gridMatches: Bool?, pixelCosine: Double?
            if swiftPreprocessing {
                // Kit's own path: ImageIO decoding, snapping to the nearest fixed grid, resampling, patchify.
                let image = try VisionRGBImage(contentsOf: directory.appendingPathComponent(c.image.file))
                var configuration = Qwen3VLImagePreprocessor.Configuration()
                if let cap = reference.maximumPixels { configuration.maximumPixels = cap }
                configuration.imageGrids = reference.grids
                let processed = try Qwen3VLImagePreprocessor.process(rgb: image.rgb, width: image.width, height: image.height,
                                                                     configuration: configuration)
                gridMatches = [processed.grid.t, processed.grid.h, processed.grid.w] == c.grid
                if gridMatches == true {
                    var dot = 0.0, aa = 0.0, bb = 0.0
                    for (a, b) in zip(processed.pixels, referencePixels) { dot += Double(a) * Double(b); aa += Double(a * a); bb += Double(b * b) }
                    pixelCosine = dot / (aa.squareRoot() * bb.squareRoot())
                }
                pixels = processed.pixels.map { Float16($0) }
                grid = processed.grid
            }
            let start = ContinuousClock.now
            let features = try await encoder.encode(pixels: pixels, grid: grid)
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

            // Upstream `generate` keeps its final stop token; Kit's reply ends before it.
            var referenceGreedy = c.greedyTokens
            while let last = referenceGreedy.last, [151645, 151643].contains(last) { referenceGreedy.removeLast() }
            let reply = try await session.generate(prompt: prompt, imageTokens: tokens, maximumTokens: c.greedyTokens.count)
            let divergence = zip(reply.tokens, referenceGreedy).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                ?? (reply.tokens.count != referenceGreedy.count ? min(reply.tokens.count, referenceGreedy.count) : nil)
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
                timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, preprocessing: swiftPreprocessing ? "swift" : "reference",
                swiftGridMatches: gridMatches, swiftPixelCosine: pixelCosine))
        }
        await session.unload()
        await encoder.unload()
        report.passed = !report.cases.isEmpty && report.cases.allSatisfy {
            // Model parity on identical inputs.
            let modelParity = $0.forcedMinimumCosine >= 0.995 && $0.forcedArgmaxMismatches == $0.forcedMismatchesAtNearTies
                && $0.greedyAccepted
            if $0.preprocessing == "reference" { return modelParity }
            // Kit end to end on the decode the reference saw: only Kit's resampler differs from the reference processor
            // (within one level), so the logits must meet the model-parity bar.
            if sameDecode { return $0.swiftGridMatches == true && ($0.swiftPixelCosine ?? 0) >= 0.9999 && modelParity }
            // An independent decode (ImageIO against libjpeg-turbo) moves pixels by up to tens of levels: recorded as
            // pixel parity plus logits within that input band, with tokens compared; reported, not asserted.
            return $0.swiftGridMatches == true && ($0.swiftPixelCosine ?? 0) >= 0.999 && $0.forcedMinimumCosine >= 0.98
        }
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    if report.gated { #expect(report.passed) }
}
