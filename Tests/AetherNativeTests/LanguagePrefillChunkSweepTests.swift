import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import Foundation
import Metal
import Testing

/// Opt-in sweep for choosing the bucketed prefill width C on a text bundle. For each C, on a fresh session without a
/// memory budget: Metal retained per warm request (a 449-token prompt repeated, then alternating with a 12-token one),
/// and the median time to first token for each prompt; then, on a fresh session at the default budget, the reloads over
/// 60 of the long requests. Every request generates 8 tokens.
///
/// AETHER_NATIVE_LANGUAGE_CHUNK_SWEEP=1
/// AETHER_CHUNK_SWEEP_BUNDLE=<text bundle>
/// AETHER_CHUNK_SWEEP_OUTPUT=<report .json>
/// AETHER_CHUNK_SWEEP_WIDTHS=<default 32,64,128,256>
private struct ChunkRow: Codable {
    var chunk = 0
    var retainedPerLongRequestMiB = 0.0
    var retainedPerMixedRequestMiB = 0.0
    var medianTimeToFirstTokenLongSeconds = 0.0
    var medianTimeToFirstTokenShortSeconds = 0.0
    var reloadsPer60AtDefaultBudget = 0
    var secondsFor60AtDefaultBudget = 0.0
    var longReply: [Int] = []
    var shortReply: [Int] = []
    var repliesStable = true
}

private struct ChunkSweepReport: Codable {
    var bundle = ""
    var device = LanguageGenerationSession.deviceArchitectureName
    var promptTokens: [String: Int] = [:]
    var defaultBudgetMiB = LanguageGenerationSession.defaultMemoryGrowthBudgetBytes >> 20
    var rows: [ChunkRow] = []
    var error: String?
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_CHUNK_SWEEP"] == "1"), .timeLimit(.minutes(60)))
func languagePrefillChunkSweep() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let device = try #require(MTLCreateSystemDefaultDevice())
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_CHUNK_SWEEP_OUTPUT"]))
    var report = ChunkSweepReport()
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
    let mib = { Double(device.currentAllocatedSize) / Double(1 << 20) }
    do {
        let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_CHUNK_SWEEP_BUNDLE"]))
        report.bundle = bundle.lastPathComponent
        let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        let long = try tokenizer.render([.user(String(repeating: "The lighthouse keeper paints the railing a calm teal. ", count: 40))]).tokenIDs
        let short = try tokenizer.render([.user("Name three colours.")]).tokenIDs
        report.promptTokens = ["long": long.count, "short": short.count]
        let widths = (nativeEnvironment["AETHER_CHUNK_SWEEP_WIDTHS"] ?? "32,64,128,256").split(separator: ",").compactMap { Int($0) }
        for width in widths {
            var row = ChunkRow()
            row.chunk = width
            let session = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos, memoryGrowthBudgetBytes: nil)
            row.longReply = try await session.generate(prompt: long, maximumTokens: 8, prefillChunkSize: width).tokens
            row.shortReply = try await session.generate(prompt: short, maximumTokens: 8, prefillChunkSize: width).tokens
            _ = try await session.generate(prompt: long, maximumTokens: 8, prefillChunkSize: width)
            var longTimes: [Double] = [], shortTimes: [Double] = []
            var start = mib()
            for _ in 0..<20 {
                let reply = try await session.generate(prompt: long, maximumTokens: 8, prefillChunkSize: width)
                longTimes.append(reply.timeToFirstTokenSeconds)
                if reply.tokens != row.longReply { row.repliesStable = false }
            }
            row.retainedPerLongRequestMiB = (mib() - start) / 20
            start = mib()
            for index in 0..<20 {
                let prompt = index.isMultiple(of: 2) ? short : long
                let reply = try await session.generate(prompt: prompt, maximumTokens: 8, prefillChunkSize: width)
                if prompt == short { shortTimes.append(reply.timeToFirstTokenSeconds) }
                if reply.tokens != (prompt == short ? row.shortReply : row.longReply) { row.repliesStable = false }
            }
            row.retainedPerMixedRequestMiB = (mib() - start) / 20
            row.medianTimeToFirstTokenLongSeconds = median(longTimes)
            row.medianTimeToFirstTokenShortSeconds = median(shortTimes)
            await session.unload()

            let budgeted = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
            _ = try await budgeted.generate(prompt: long, maximumTokens: 8, prefillChunkSize: width)
            let clock = ContinuousClock.now
            for _ in 0..<60 {
                if try await budgeted.generate(prompt: long, maximumTokens: 8, prefillChunkSize: width).tokens != row.longReply {
                    row.repliesStable = false
                }
            }
            let d = clock.duration(to: .now)
            row.secondsFor60AtDefaultBudget = Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
            row.reloadsPer60AtDefaultBudget = await budgeted.cacheStatistics().modelReloadCount
            await budgeted.unload()
            report.rows.append(row)
        }
    } catch {
        report.error = String(describing: error)
        throw error
    }
}
