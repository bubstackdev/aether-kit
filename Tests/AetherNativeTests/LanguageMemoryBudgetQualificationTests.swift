import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import AetherVision
import Foundation
import Metal
import Testing

/// Opt-in native check of the memory budget: repeated identical requests on one session keep Metal's allocated size
/// within the post-warm baseline plus the budget and the largest warm request's growth (a request of a kind that already
/// ran since the last reload), the budget reloads the model at least once,
/// and every reply of a kind is token-for-token the first one, across reloads. Each model is opened as an application
/// would: a text bundle through `LanguageGenerationSession.open(bundle:)`, a Qwen3-VL bundle through
/// `VisionLanguageSession.open(bundle:)` (alternating text-only and image requests, the image a synthetic gradient),
/// and a raw bucketed `model.aimodel` (Forge export, `export-report.json` and `tokenizer/` beside it) through the
/// raw-asset entry point.
///
/// AETHER_NATIVE_LANGUAGE_MEMORY_BUDGET=1
/// AETHER_MEMORY_BUDGET_MODELS=<name=path,...: bundles or raw model.aimodel>
/// AETHER_MEMORY_BUDGET_OUTPUT=<report .json>
/// AETHER_MEMORY_BUDGET_MIB=<budget, default the Kit default (2048)>
/// AETHER_MEMORY_BUDGET_REQUESTS=<requests per model after warming, default 60>
private struct BudgetSample: Codable { let request: Int, kind: String, metalMiB: Int, retainedMiB: Int, reloads: Int }

private struct BudgetModelReport: Codable {
    var name = ""
    var promptTokens: [String: Int] = [:]
    var replies: [String: [Int]] = [:]
    var warmMetalMiB = 0
    var maximumMetalMiB = 0
    var maximumRequestGrowthMiB = 0
    var boundMiB = 0
    var reloads = 0
    var idleReloads = 0
    var requestReloads = 0
    var mismatchedReplies = 0
    var unloadedMetalMiB = 0
    var seconds = 0.0
    var samples: [BudgetSample] = []
    var passed = false
}

private struct BudgetReport: Codable {
    var gate = "Per model: at least 60 identical-shape requests after warming; Metal after every request at most the post-warm baseline plus the budget plus the largest warm request's growth (its kind already ran since the last reload); at least one reload; every reply of a kind identical to the first"
    var device = LanguageGenerationSession.deviceArchitectureName
    var budgetMiB = 0
    var requests = 0
    var models: [BudgetModelReport] = []
    var error: String?
    var passed = false
}

/// A smooth synthetic RGB gradient.
private func budgetGradient(width: Int, height: Int) throws -> VisionRGBImage {
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        for x in 0..<width {
            let i = 3 * (y * width + x)
            rgb[i] = UInt8(x * 255 / max(1, width - 1))
            rgb[i + 1] = UInt8(y * 255 / max(1, height - 1))
            rgb[i + 2] = UInt8((x + y) * 127 / max(1, width + height - 2))
        }
    }
    return try VisionRGBImage(rgb: rgb, width: width, height: height)
}

/// One request kind: a name and how to run it, returning the reply's tokens.
private struct BudgetRequest {
    let kind: String
    let run: () async throws -> [Int]
}

private struct BudgetSubject {
    let requests: [BudgetRequest]
    let statistics: () async -> LanguageGenerationSession.CacheStatistics?
    let unload: () async -> Void
}

private let budgetPrompt = String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40)

private func openSubject(path: URL, budget: Int, report: inout BudgetModelReport) async throws -> BudgetSubject {
    if path.pathExtension == "aimodel" {
        struct Export: Decodable { let exportBounds: LanguageExportBounds? }
        let directory = path.deletingLastPathComponent()
        let bounds = try JSONDecoder().decode(Export.self, from: Data(contentsOf: directory.appendingPathComponent("export-report.json"))).exportBounds
        let tokenizer = try await LanguageTokenizer.open(directory: directory.appendingPathComponent("tokenizer"))
        let long = try tokenizer.render([.user(budgetPrompt)]).tokenIDs
        report.promptTokens = ["text": long.count]
        let session = try await LanguageGenerationSession.open(at: path, maximumContextLength: 2048, memoryGrowthBudgetBytes: budget,
                                                               exportBounds: bounds)
        return BudgetSubject(requests: [BudgetRequest(kind: "text") { try await session.generate(prompt: long, maximumTokens: 8).tokens }],
                             statistics: { await session.cacheStatistics() }, unload: { await session.unload() })
    }
    let tokenizer = try await LanguageTokenizer.open(bundle: path)
    let long = try tokenizer.render([.user(budgetPrompt)]).tokenIDs
    if try LanguageModelManifest.load(from: path).vision != nil {
        let withImage = try tokenizer.render([.user("Describe the colours in this picture.", images: 1)]).tokenIDs
        report.promptTokens = ["text": long.count, "image": withImage.count]
        let image = try budgetGradient(width: 512, height: 384)
        let session = try await VisionLanguageSession.open(bundle: path, platform: .macos, memoryGrowthBudgetBytes: budget)
        return BudgetSubject(requests: [
            BudgetRequest(kind: "text") { try await session.generate(renderedTokenIDs: long, images: [], maximumTokens: 8).tokens },
            BudgetRequest(kind: "image") { try await session.generate(renderedTokenIDs: withImage, images: [image], maximumTokens: 8).tokens },
        ], statistics: { await session.decoderStatistics() }, unload: { await session.unload() })
    }
    report.promptTokens = ["text": long.count]
    let session = try await LanguageGenerationSession.open(bundle: path, platform: .macos, memoryGrowthBudgetBytes: budget)
    return BudgetSubject(requests: [BudgetRequest(kind: "text") { try await session.generate(prompt: long, maximumTokens: 8).tokens }],
                         statistics: { await session.cacheStatistics() }, unload: { await session.unload() })
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_MEMORY_BUDGET"] == "1"), .timeLimit(.minutes(60)))
func languageMemoryBudgetBoundsRepeatedRequests() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_MEMORY_BUDGET_OUTPUT"]))
    var report = BudgetReport()
    report.budgetMiB = Int(nativeEnvironment["AETHER_MEMORY_BUDGET_MIB"] ?? "") ?? LanguageGenerationSession.defaultMemoryGrowthBudgetBytes >> 20
    report.requests = Int(nativeEnvironment["AETHER_MEMORY_BUDGET_REQUESTS"] ?? "") ?? 60
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    let metalMiB = { device.currentAllocatedSize >> 20 }
    do {
        let models = try #require(nativeEnvironment["AETHER_MEMORY_BUDGET_MODELS"]).split(separator: ",").map { $0.split(separator: "=", maxSplits: 1) }
        for entry in models {
            var model = BudgetModelReport()
            model.name = String(entry[0])
            let start = ContinuousClock.now
            let subject = try await openSubject(path: URL(filePath: String(entry[1])), budget: report.budgetMiB << 20, report: &model)
            // Warm every kind once (first use of each function and shape), then the baseline.
            for request in subject.requests { model.replies[request.kind] = try await request.run() }
            model.warmMetalMiB = metalMiB()
            var previous = model.warmMetalMiB
            // Kinds run since the last reload: the first request of each kind after one copies its weights again.
            var warmKinds = Set(subject.requests.map(\.kind))
            for index in 0..<report.requests {
                let request = subject.requests[index % subject.requests.count]
                let tokens = try await request.run()
                if tokens != model.replies[request.kind] { model.mismatchedReplies += 1 }
                let now = metalMiB()
                let statistics = await subject.statistics()
                model.samples.append(BudgetSample(request: index, kind: request.kind, metalMiB: now,
                                                  retainedMiB: (statistics?.retainedMemoryBytes ?? 0) >> 20, reloads: statistics?.modelReloadCount ?? 0))
                // Growth across a warm request: no reload before it, and its kind already ran since the last one.
                if (statistics?.modelReloadCount ?? 0) != model.reloads { warmKinds = [] }
                if warmKinds.contains(request.kind) { model.maximumRequestGrowthMiB = max(model.maximumRequestGrowthMiB, now - previous) }
                warmKinds.insert(request.kind)
                model.reloads = statistics?.modelReloadCount ?? 0
                model.maximumMetalMiB = max(model.maximumMetalMiB, now)
                previous = now
            }
            // Reloads at idle (after a request, under its own grant) and in front of a request (the fallback).
            let final = await subject.statistics()
            model.idleReloads = final?.idleMemoryReloads ?? 0
            model.requestReloads = final?.requestMemoryReloads ?? 0
            await subject.unload()
            model.unloadedMetalMiB = metalMiB()
            let d = start.duration(to: .now)
            model.seconds = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            model.boundMiB = model.warmMetalMiB + report.budgetMiB + model.maximumRequestGrowthMiB
            model.passed = model.maximumMetalMiB <= model.boundMiB && model.reloads >= 1 && model.mismatchedReplies == 0
            report.models.append(model)
            #expect(model.passed, "\(model.name): max \(model.maximumMetalMiB) MiB, bound \(model.boundMiB) MiB, reloads \(model.reloads), mismatched \(model.mismatchedReplies)")
        }
        report.passed = report.models.allSatisfy(\.passed)
    } catch {
        report.error = String(describing: error)
        throw error
    }
}
