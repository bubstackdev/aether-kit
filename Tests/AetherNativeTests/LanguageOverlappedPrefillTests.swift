import AetherKit
@testable import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation
import Synchronization
import Testing

/// Opt-in native check that the bucketed prefill's step-back tail is exact on real models, and that other prefill plans
/// differ from it only by rounding. A 449-token prompt (by default one 256-token chunk, then a last chunk stepping back
/// 63 rows, with the cache sized for the prompt before the prefill) is prefilled four ways: the default plan, one chunk
/// for the whole prompt, 1-token steps, and the 0.0.2 width (64-token chunks). The graph attends only to cache rows
/// before a chunk's first position, so the rows a step-back rewrites are recomputed from the same prefix. Every
/// comparison needs the default plan's last-row logits at cosine >= 0.9999 with the same argmax, then:
/// - step-back (default against one chunk, the exactness claim): identical 32-token greedy continuations;
/// - plan rounding (default against 1-token steps, and against the 0.0.2 width: other kernel shapes): identical greedy,
///   or a first divergence at an upstream near-tie: the two plans' tokens are the upstream reference's top two at that
///   row, its top-1 minus top-2 probability is under 0.1 (the parity gate's threshold), and the two plans'
///   teacher-forced rows agree at cosine >= 0.99999 on every row up to and including the divergence (the other plan's
///   row r: the prompt and r generated tokens prefilled at its width). Judged only where an upstream reference with
///   this prompt covers the row; otherwise any divergence fails.
/// A Qwen3-VL bundle also runs three image prompts through `VisionLanguageSession`, two with one image and one with two:
/// greedy identical across the three plans, with no near-tie allowance. A one-function decoder prefills image tokens in the same chunks as text, so the
/// default plan's chunks (256 wide: the image prompts are longer than 256 tokens) cross the image span's edges in both,
/// and the cases are chosen for the step-back tail: after a long question it lands in the text after the image; after
/// a short one it steps back into the image span and re-covers image rows (the report records the image span and the
/// default plan's chunk starts, and requires both).
///
/// AETHER_NATIVE_LANGUAGE_OVERLAP=1
/// AETHER_OVERLAP_MODELS=<name=path,...: text or Qwen3-VL bundles, or a raw bucketed model.aimodel>
/// AETHER_OVERLAP_REFERENCES=<optional name=dir,...: upstream text references (reference.json, forced logits) of the
///   same prompt, for the plan-rounding near-tie rule>
/// AETHER_OVERLAP_OUTPUT=<report .json>
/// A plan-rounding divergence judged against the upstream reference.
private struct OverlapNearTie: Codable {
    var row = 0
    var tokens: [Int] = []                 // default plan, 1-token steps
    var upstreamTopTwo: [Int] = []
    var upstreamMargin = 0.0               // top-1 minus top-2 probability
    var marginSource = ""
    var rowCosines: [Double] = []          // the two plans' teacher-forced rows 0...row
}

private struct OverlapComparison: Codable {
    var plan = ""
    var rule = ""
    var cosine: Double?
    var argmaxEqual: Bool?
    var greedyIdentical = false
    var firstDivergence: Int?
    var greedy: [Int] = []
    var nearTie: OverlapNearTie?
    /// At a divergence, reported and never gating: each plan's own row there (the default plan's teacher-forced row,
    /// the other plan's prefill of the prompt and prefix), as [Kit's top-1, top-2] tokens and top-1 minus top-2 probability.
    var kitRowsAtDivergence: [String: KitRow]?
    var acceptedAtNearTie = false
    var passed = false
}

private struct KitRow: Codable {
    var topTwo: [Int] = []
    var margin = 0.0
    init(_ logits: [Float]) {
        let order = logits.indices.sorted { logits[$0] > logits[$1] }
        let top = Double(logits[order[0]])
        let denominator = logits.reduce(0.0) { $0 + exp(Double($1) - top) }
        topTwo = [order[0], order[1]]
        margin = (1 - exp(Double(logits[order[1]]) - top)) / denominator
    }
}

/// An upstream text reference: row r of `logits` follows prompt + forcedTokens[:r].
private struct OverlapReference {
    struct Record: Decodable {
        let promptTokens: [Int]
        let forcedTokens: [Int]
        let forcedLogits: Logits
        struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int }
    }
    let name: String
    let record: Record
    let logits: [Float]
    init(directory: URL) throws {
        name = directory.lastPathComponent
        record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: directory.appendingPathComponent("reference.json")))
        logits = try Data(contentsOf: directory.appendingPathComponent(record.forcedLogits.file))
            .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard logits.count == record.forcedLogits.rows * record.forcedLogits.vocab else { throw NativeFailure(description: "reference logits shape") }
    }
    func row(_ r: Int) -> ArraySlice<Float> { logits[(r * record.forcedLogits.vocab)..<((r + 1) * record.forcedLogits.vocab)] }
}

private struct OverlapModel: Codable {
    var name = ""
    var promptTokens = 0
    var submissions: [String: Int] = [:]
    var defaultGreedy: [Int] = []
    var comparisons: [OverlapComparison] = []
    var imageCases: [OverlapImageCase] = []
    var imageGreedyIdentical: Bool?
    var passed = false
}

/// An image prompt prefilled three ways. `imageSpan` is [start, end) in the expanded prompt; `defaultChunkStarts` are the
/// default plan's chunk starts on a one-function decoder (a 0.0.2 decoder sends the image span as one submission).
private struct OverlapImageCase: Codable {
    var name = ""
    var promptTokens = 0
    var imageSpan: [Int] = []
    var defaultChunkStarts: [Int] = []
    var chunksCrossImageEdges = false
    var tailReCoversImageRows = false
    var greedy: [String: [Int]] = [:]
    var greedyIdentical = false
}

private struct OverlapReport: Codable {
    var gate = "449-token prompt, default plan (width 256 above 256 prompt tokens, last chunk stepping back 63 rows, cache sized for the prompt before the prefill) against three plans; last-row logits cosine >= 0.9999 with equal argmax for each. Rule step-back (against one whole-prompt chunk, the exactness claim): identical 32-token greedy. Rule plan-rounding (against 1-token steps and against the 0.0.2 width of 64, other kernel shapes): identical greedy, or a first divergence where the two plans' tokens are the upstream reference's top two at that row, its top-1 minus top-2 probability < 0.1 (the parity threshold), and the two plans' teacher-forced rows have cosine >= 0.99999 on every row up to and including it (acceptedAtNearTie, margin and source recorded); any other divergence fails. Qwen3-VL image prompts (a long and a short question about one image, and two images) under the default plan, one 512-token-wide plan and 1-token steps: identical greedy, no near-tie allowance"
    var device = LanguageGenerationSession.deviceArchitectureName
    var models: [OverlapModel] = []
    var error: String?
    var passed = false
}

private func cosine(_ a: [Float], _ b: [Float]) -> Double {
    var dot = 0.0, na = 0.0, nb = 0.0
    for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
    return dot / (na.squareRoot() * nb.squareRoot())
}

private func argmax(_ v: [Float]) -> Int { v.indices.max { v[$0] < v[$1] }! }

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_OVERLAP"] == "1"), .timeLimit(.minutes(30)))
func languageOverlappedPrefillMatchesOtherPlans() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_OVERLAP_OUTPUT"]))
    var report = OverlapReport()
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    let text = String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40)
    do {
        var references: [String: OverlapReference] = [:]
        for entry in (nativeEnvironment["AETHER_OVERLAP_REFERENCES"] ?? "").split(separator: ",") {
            let parts = entry.split(separator: "=", maxSplits: 1)
            references[String(parts[0])] = try OverlapReference(directory: URL(filePath: String(parts[1])))
        }
        let entries = try #require(nativeEnvironment["AETHER_OVERLAP_MODELS"]).split(separator: ",").map { $0.split(separator: "=", maxSplits: 1) }
        for entry in entries {
            var model = OverlapModel()
            model.name = String(entry[0])
            let path = URL(filePath: String(entry[1]))
            let session: LanguageGenerationSession
            let tokenizer: LanguageTokenizer
            var vision = false
            if path.pathExtension == "aimodel" {
                struct Export: Decodable { let exportBounds: LanguageExportBounds? }
                let directory = path.deletingLastPathComponent()
                let bounds = try JSONDecoder().decode(Export.self, from: Data(contentsOf: directory.appendingPathComponent("export-report.json"))).exportBounds
                tokenizer = try await LanguageTokenizer.open(directory: directory.appendingPathComponent("tokenizer"))
                session = try await LanguageGenerationSession.open(at: path, maximumContextLength: 2048, exportBounds: bounds)
            } else {
                tokenizer = try await LanguageTokenizer.open(bundle: path)
                session = try await LanguageGenerationSession.open(bundle: path, platform: .macos)
                vision = try LanguageModelManifest.load(from: path).vision != nil
            }
            let prompt = try tokenizer.render([.user(text)]).tokenIDs
            model.promptTokens = prompt.count
            // nil: the default plan (256 wide for this prompt); the whole prompt in one chunk; 1-token steps; the 0.0.2 width.
            let plans: [(String, Int?)] = [("default", nil), ("single chunk", prompt.count), ("1-token steps", 1),
                                           ("0.0.2 width", LanguageStepContract.defaultBucketedPrefillChunk)]
            var logits: [String: [Float]] = [:], greedy: [String: [Int]] = [:]
            for (name, chunk) in plans {
                let submissions = Counter()
                logits[name] = try await session.prefillLogits(prompt: prompt, chunkSize: chunk)
                let reply = try await session.generate(prompt: prompt, maximumTokens: 32, prefillChunkSize: chunk,
                                                       observer: { if case .submitted = $0 { submissions.add() } })
                greedy[name] = reply.tokens
                // The prefill's submissions: every generated token is stepped, except the last at the output limit.
                model.submissions[name] = submissions.value - reply.tokens.count + (reply.stopReason == "eos" ? 0 : 1)
            }
            model.defaultGreedy = greedy["default"]!
            for (name, chunk) in plans.dropFirst() {
                let a = greedy["default"]!, b = greedy[name]!
                var comparison = OverlapComparison(plan: name, rule: name == "single chunk" ? "step-back" : "plan-rounding",
                    cosine: cosine(logits["default"]!, logits[name]!),
                    argmaxEqual: argmax(logits["default"]!) == argmax(logits[name]!), greedyIdentical: a == b,
                    firstDivergence: zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset, greedy: b)
                if let d = comparison.firstDivergence {
                    let own = try await session.teacherForcedLogits(prompt: prompt, continuation: Array(a.prefix(d)))[d]
                    let other = try await session.prefillLogits(prompt: prompt + a.prefix(d), chunkSize: chunk)
                    comparison.kitRowsAtDivergence = ["default": KitRow(own), name: KitRow(other)]
                }
                if comparison.rule == "plan-rounding", let d = comparison.firstDivergence, let reference = references[model.name],
                   reference.record.promptTokens == prompt, d < reference.record.forcedLogits.rows,
                   Array(reference.record.forcedTokens.prefix(d)) == Array(a.prefix(d)) {
                    var tie = OverlapNearTie(row: d, tokens: [a[d], b[d]], marginSource: "\(reference.name) forced row \(d)")
                    let row = reference.row(d).map(Double.init)
                    let order = row.indices.sorted { row[$0] > row[$1] }
                    let denominator = row.reduce(0.0) { $0 + exp($1 - row[order[0]]) }
                    tie.upstreamTopTwo = [order[0], order[1]]
                    tie.upstreamMargin = (1 - exp(row[order[1]] - row[order[0]])) / denominator
                    // Default plan: its prefill, then 1-token steps; the other plan: the prompt and r tokens prefilled at
                    // its width (for 1-token steps, 1-token steps throughout).
                    let defaultRows = try await session.teacherForcedLogits(prompt: prompt, continuation: Array(a.prefix(d)))
                    for r in 0...d {
                        let stepRow = try await session.prefillLogits(prompt: prompt + a.prefix(r), chunkSize: chunk)
                        tie.rowCosines.append(cosine(defaultRows[r], stepRow))
                    }
                    comparison.acceptedAtNearTie = Set(tie.tokens) == Set(tie.upstreamTopTwo) && tie.upstreamMargin < 0.1
                        && tie.rowCosines.allSatisfy { $0 >= 0.99999 }
                    comparison.nearTie = tie
                }
                comparison.passed = (comparison.cosine ?? 0) >= 0.9999 && comparison.argmaxEqual == true
                    && (comparison.greedyIdentical || comparison.acceptedAtNearTie)
                model.comparisons.append(comparison)
            }
            await session.unload()
            if vision {
                let registration = try #require(try LanguageModelManifest.load(from: path).vision)
                let grids = try #require(registration.imageGrids).map { Qwen3VLImagePreprocessor.Grid(height: $0.height, width: $0.width) }
                let grid = try Qwen3VLImagePreprocessor.nearestGrid(width: 512, height: 384, in: grids)
                let imageTokens = grid.height * grid.width / (registration.mergeSize * registration.mergeSize)
                let image = try VisionRGBImage(rgb: (0..<(512 * 384)).flatMap { i -> [UInt8] in
                    let x = i % 512, y = i / 512
                    return [UInt8(x * 255 / 511), UInt8(y * 255 / 383), UInt8((x + y) * 127 / 893)]
                }, width: 512, height: 384)
                let vl = try await VisionLanguageSession.open(bundle: path, platform: .macos)
                // The long question puts the default plan's stepping-back tail after the image at width 256.
                let questions = [("long question", "Describe the colours in this picture. "
                                    + String(repeating: "Mention every shade you can see, from the brightest to the darkest. ", count: 20)),
                                 ("short question", "Describe this picture.")]
                for (caseName, question) in questions {
                    let rendered = try tokenizer.render([.user(question, images: 1)]).tokenIDs
                    var imageCase = OverlapImageCase()
                    imageCase.name = caseName
                    let start = try #require(rendered.firstIndex(of: registration.imageTokenID))
                    let total = rendered.count - 1 + imageTokens
                    imageCase.promptTokens = total
                    imageCase.imageSpan = [start, start + imageTokens]
                    let width = try #require(LanguageStepContract.bucketedPositions.defaultPrefillChunk(promptCount: total))
                    imageCase.defaultChunkStarts = Array(stride(from: 0, through: total - width, by: width))
                        + (total % width == 0 || total <= width ? [] : [total - width])
                    func inside(_ edge: Int) -> Bool { imageCase.defaultChunkStarts.contains { $0 < edge && edge < $0 + width } }
                    imageCase.chunksCrossImageEdges = inside(start) && inside(start + imageTokens)
                    imageCase.tailReCoversImageRows = total % width != 0 && total - width < start + imageTokens
                    for (name, chunk) in [("default", nil), ("single chunk", 512), ("1-token steps", 1)] as [(String, Int?)] {
                        imageCase.greedy[name] = try await vl.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: 32,
                                                                        prefillChunkSize: chunk).tokens
                    }
                    imageCase.greedyIdentical = Set(imageCase.greedy.values).count == 1
                    model.imageCases.append(imageCase)
                }
                // Two images in one prompt: both spans' edges and the gap between them fall wherever the chunks put them.
                let second = try VisionRGBImage(rgb: (0..<(512 * 384)).flatMap { i -> [UInt8] in
                    let x = i % 512, y = i / 512
                    return [UInt8((x + y) * 127 / 893), UInt8(x * 255 / 511), UInt8(y * 255 / 383)]
                }, width: 512, height: 384)
                let pair = try tokenizer.render([.user("What differs between these two pictures?", images: 2)]).tokenIDs
                var pairCase = OverlapImageCase()
                pairCase.name = "two images"
                let pads = pair.indices.filter { pair[$0] == registration.imageTokenID }
                let first = pads[0], next = pads[1] + imageTokens - 1
                pairCase.promptTokens = pair.count - 2 + 2 * imageTokens
                pairCase.imageSpan = [first, first + imageTokens, next, next + imageTokens]
                let total = pairCase.promptTokens
                let width = try #require(LanguageStepContract.bucketedPositions.defaultPrefillChunk(promptCount: total))
                pairCase.defaultChunkStarts = Array(stride(from: 0, through: total - width, by: width))
                    + (total % width == 0 || total <= width ? [] : [total - width])
                pairCase.chunksCrossImageEdges = pairCase.imageSpan.allSatisfy { edge in
                    pairCase.defaultChunkStarts.contains { $0 < edge && edge < $0 + width }
                }
                pairCase.tailReCoversImageRows = total % width != 0 && total - width < next + imageTokens
                for (name, chunk) in [("default", nil), ("single chunk", 512), ("1-token steps", 1)] as [(String, Int?)] {
                    pairCase.greedy[name] = try await vl.generate(renderedTokenIDs: pair, images: [image, second], maximumTokens: 32,
                                                                   prefillChunkSize: chunk).tokens
                }
                pairCase.greedyIdentical = Set(pairCase.greedy.values).count == 1
                model.imageCases.append(pairCase)
                await vl.unload()
                // Each single-image case must exercise what it is for: image edges inside chunks, and the short
                // question's tail stepping back into the image. Every case's greedy output must agree across plans.
                let geometry = model.imageCases.filter { $0.name != "two images" }.allSatisfy(\.chunksCrossImageEdges)
                    && model.imageCases.contains { $0.name == "short question" && $0.tailReCoversImageRows }
                    && model.imageCases.contains { $0.name == "long question" && !$0.tailReCoversImageRows }
                model.imageGreedyIdentical = geometry && model.imageCases.allSatisfy(\.greedyIdentical)
            }
            model.passed = model.comparisons.allSatisfy(\.passed) && (model.imageGreedyIdentical ?? true)
            report.models.append(model)
            #expect(model.passed, "\(model.name): \(model.comparisons), image identical \(String(describing: model.imageGreedyIdentical))")
        }
        report.passed = report.models.allSatisfy(\.passed)
    } catch {
        report.error = String(describing: error)
        throw error
    }
}

private final class Counter: Sendable {
    private let count = Mutex(0)
    func add() { count.withLock { $0 += 1 } }
    var value: Int { count.withLock { $0 } }
}
