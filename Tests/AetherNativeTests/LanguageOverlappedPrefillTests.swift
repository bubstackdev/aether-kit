import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation
import Synchronization
import Testing

/// Opt-in native check that the bucketed prefill's step-back tail is exact on real models. A 449-token prompt (seven
/// 64-token chunks, then a last chunk stepping back 63 rows) is prefilled three ways: the default width (64, stepping
/// back), one chunk for the whole prompt, and 1-token steps. The graph attends only to cache rows before a chunk's
/// first position, so the rows a step-back rewrites are recomputed from the same prefix. Every comparison needs the
/// default plan's last-row logits at cosine >= 0.9999 with the same argmax, then:
/// - step-back (default against one chunk, the exactness claim): identical 32-token greedy continuations;
/// - plan rounding (default against 1-token steps, another kernel shape): identical greedy, or a first divergence at an
///   upstream near-tie: the two plans' tokens are the upstream reference's top two at that row, its top-1 minus top-2
///   probability is under 0.1 (the parity gate's threshold), and the two plans' teacher-forced rows agree at cosine
///   >= 0.99999 on every row up to and including the divergence. Judged only where an upstream reference with this
///   prompt covers the row; otherwise any divergence fails.
/// A Qwen3-VL bundle also runs two image prompts through `VisionLanguageSession`: greedy identical across the three
/// plans, with no near-tie allowance. A one-function decoder prefills image tokens in the same chunks as text, so the
/// default plan's chunks cross the image span's edges in both, and the cases are chosen for the step-back tail: after
/// a long question it lands in the text after the image; after a short one it steps back into the image span and
/// re-covers image rows (the report records the image span and the default plan's chunk starts, and requires both).
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
    var acceptedAtNearTie = false
    var passed = false
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
    var gate = "449-token prompt, default width (64, last chunk stepping back 63 rows) against two plans; last-row logits cosine >= 0.9999 with equal argmax for both. Rule step-back (against one whole-prompt chunk, the exactness claim): identical 32-token greedy. Rule plan-rounding (against 1-token steps, another kernel shape): identical greedy, or a first divergence where the two plans' tokens are the upstream reference's top two at that row, its top-1 minus top-2 probability < 0.1 (the parity threshold), and the two plans' teacher-forced rows have cosine >= 0.99999 on every row up to and including it (acceptedAtNearTie, margin and source recorded); any other divergence fails. Qwen3-VL image prompt with a stepping-back question: identical greedy across the three plans, no near-tie allowance"
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
            // nil: the default width (64); the whole prompt in one chunk; 1-token steps.
            let plans: [(String, Int?)] = [("default", nil), ("single chunk", prompt.count), ("1-token steps", 1)]
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
            for (name, _) in plans.dropFirst() {
                let a = greedy["default"]!, b = greedy[name]!
                var comparison = OverlapComparison(plan: name, rule: name == "single chunk" ? "step-back" : "plan-rounding",
                    cosine: cosine(logits["default"]!, logits[name]!),
                    argmaxEqual: argmax(logits["default"]!) == argmax(logits[name]!), greedyIdentical: a == b,
                    firstDivergence: zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset, greedy: b)
                if comparison.rule == "plan-rounding", let d = comparison.firstDivergence, let reference = references[model.name],
                   reference.record.promptTokens == prompt, d < reference.record.forcedLogits.rows,
                   Array(reference.record.forcedTokens.prefix(d)) == Array(a.prefix(d)) {
                    var tie = OverlapNearTie(row: d, tokens: [a[d], b[d]], marginSource: "\(reference.name) forced row \(d)")
                    let row = reference.row(d).map(Double.init)
                    let order = row.indices.sorted { row[$0] > row[$1] }
                    let denominator = row.reduce(0.0) { $0 + exp($1 - row[order[0]]) }
                    tie.upstreamTopTwo = [order[0], order[1]]
                    tie.upstreamMargin = (1 - exp(row[order[1]] - row[order[0]])) / denominator
                    // Default plan: its prefill, then 1-token steps; 1-token plan: 1-token steps throughout.
                    let defaultRows = try await session.teacherForcedLogits(prompt: prompt, continuation: Array(a.prefix(d)))
                    for r in 0...d {
                        let stepRow = try await session.prefillLogits(prompt: prompt + a.prefix(r), chunkSize: 1)
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
                let questions = [("long question", "Describe the colours in this picture. "
                                    + String(repeating: "Mention every shade you can see, from the brightest to the darkest. ", count: 8)),
                                 ("short question", "Describe this picture.")]
                for (caseName, question) in questions {
                    let rendered = try tokenizer.render([.user(question, images: 1)]).tokenIDs
                    var imageCase = OverlapImageCase()
                    imageCase.name = caseName
                    let start = try #require(rendered.firstIndex(of: registration.imageTokenID))
                    let total = rendered.count - 1 + imageTokens
                    imageCase.promptTokens = total
                    imageCase.imageSpan = [start, start + imageTokens]
                    let width = 64
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
                await vl.unload()
                // Each case must exercise what it is for: image edges inside chunks, and the short question's tail
                // stepping back into the image.
                let geometry = model.imageCases.allSatisfy(\.chunksCrossImageEdges)
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
