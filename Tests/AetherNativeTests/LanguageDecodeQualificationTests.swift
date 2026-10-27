import AetherCoreAI
import AetherKit
import AetherLanguageCoreAI
import CryptoKit
import Darwin
import Foundation
import Synchronization
import Testing

/// Opt-in native qualification of the bucketed language decode contract against an independent
/// CPU reference (`Forge/language/reference_qwen3.py`), with a before/after comparison against a
/// running-contract export of the same checkpoint and precision.
///
/// AETHER_NATIVE_LANGUAGE_QUALIFICATION=1
/// AETHER_LANGUAGE_REFERENCE=<dir with reference.json and forced-logits.f32>
/// AETHER_LANGUAGE_BUCKETED_MODEL=<bucketed model.aimodel>
/// AETHER_LANGUAGE_RUNNING_MODEL=<running model.aimodel> (optional: before/after comparison)
/// AETHER_LANGUAGE_OUTPUT=<report .json>
/// AETHER_LANGUAGE_DECODE_STEPS=<long-run decode steps, default 1024>
/// AETHER_LANGUAGE_TIMING_ROUNDS=<paired before/after timing rounds, default 6>
/// AETHER_LANGUAGE_LONG_CONTEXT=<optional context for a long-prompt memory/timing probe, e.g. 8192>
private struct LanguageReference: Decodable {
    let promptTokens: [Int]
    let greedyTokens: [Int]
    let forcedTokens: [Int]
    let forcedLogits: Logits
    struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int; let sha256: String }
}

private struct RowComparison: Codable {
    let row: Int
    let cosine: Double
    let maxAbs: Double
    let argmaxEqual: Bool
    let referenceMargin: Double
    let nativeRankInReference: Int
    var nearTie: Bool { !argmaxEqual && referenceMargin < 0.1 && nativeRankInReference < 5 }
}

private struct Timing: Codable {
    let promptTokens: Int
    let generatedTokens: Int
    let timeToFirstTokenSeconds: Double
    let totalSeconds: Double
    var decodeTokensPerSecond: Double { Double(generatedTokens - 1) / max(totalSeconds - timeToFirstTokenSeconds, 1e-9) }
}

private struct FootprintSample: Codable { let generated: Int; let physicalFootprintBytes: UInt64 }

private struct ContractReport: Codable {
    var contract = ""
    var modelFileSHA256 = ""
    var forcedRows: [RowComparison] = []
    var forcedMinimumCosine = 0.0
    var forcedMaximumAbs = 0.0
    var forcedArgmaxMismatches = 0
    var forcedNearTies = 0
    var greedyTokens: [Int] = []
    var greedyFirstDivergence: Int?
    /// The reference's teacher-forced row at the first divergence (same prefix on both sides).
    var greedyDivergenceRow: RowComparison?
    var greedyAccepted: Bool {
        greedyFirstDivergence == nil || (greedyDivergenceRow.map { $0.referenceMargin < 0.1 && $0.nativeRankInReference < 5 } ?? false)
    }
    var longRunSteps = 0
    var longRunTokens: [Int] = []
    var longRunModelReloads = 0
    var longRunDistinctStepShapes = 0
    var longRunAllocatedCapacity = 0
    var longRunFootprint: [FootprintSample] = []
}

private struct LanguageQualificationReport: Codable {
    var passed = false
    var stage = "preparation"
    var error: String?
    var device = LanguageGenerationSession.deviceArchitectureName
    var referenceSHA256: [String: String] = [:]
    var tolerance = "per forced row: cosine >= 0.995 against float32 reference; argmax equal unless a near-tie (reference top-1 minus top-2 probability < 0.1 and native token within reference top-5); greedy continuation identical to the reference's, or first diverging at a reference near-tie by the same rule (judged on the reference's teacher-forced row at that index, where both prefixes are equal)"
    var stopTokenNote = "Long runs register stop id 151935 (an unused padding row of the 151936-row head) so decoding is not cut short by EOS"
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; timings are paired and interleaved observations, not benchmark records"
    var bucketed = ContractReport()
    var running: ContractReport?
    var bucketedLongRunMatchesRunning: Int?
    /// "<contract>-<prompt>" -> one warm 256-token request per round (after one warm-up each).
    var pairedTimings: [String: [Timing]] = [:]
    var pairedLoadAverages: [Double] = []
    var longContext: [String: LongContextReport] = [:]
}

private struct LongContextReport: Codable {
    let promptTokens: Int
    let timing: Timing
    let allocatedCapacity: Int
    let allocatedBytes: Int
    let modelReloads: Int
    let distinctStepShapes: Int
    let footprint: [FootprintSample]
}

private func footprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}

private func sha256(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
}

private func compare(row: Int, native: [Float], reference: ArraySlice<Float>) -> RowComparison {
    var dot = 0.0, nn = 0.0, rr = 0.0, maxAbs = 0.0
    var refBest = (index: 0, value: -Double.infinity)
    for (n, r) in zip(native, reference) {
        let a = Double(n), b = Double(r)
        dot += a * b; nn += a * a; rr += b * b
        maxAbs = max(maxAbs, abs(a - b))
    }
    for (i, r) in reference.enumerated() where Double(r) > refBest.value { refBest = (i, Double(r)) }
    let nativeBest = native.indices.max { native[$0] < native[$1] }!
    // Softmax top-1 minus top-2 probability of the reference row.
    let sorted = reference.map(Double.init).sorted(by: >)
    let denominator = reference.reduce(0.0) { $0 + exp(Double($1) - sorted[0]) }
    let margin = (1 - exp(sorted[1] - sorted[0])) / denominator
    let nativeValue = Double(reference[reference.startIndex + nativeBest])
    let rank = reference.filter { Double($0) > nativeValue }.count
    return RowComparison(row: row, cosine: dot / (nn.squareRoot() * rr.squareRoot()), maxAbs: maxAbs,
                         argmaxEqual: nativeBest == refBest.index, referenceMargin: margin, nativeRankInReference: rank)
}

private final class TokenClock: Sendable {
    private let samples = Mutex<[FootprintSample]>([])
    private let count = Mutex(0)
    func record() {
        let n = count.withLock { $0 += 1; return $0 }
        if n == 1 || n % 128 == 0 { samples.withLock { $0.append(FootprintSample(generated: n, physicalFootprintBytes: footprint())) } }
    }
    var recorded: [FootprintSample] { samples.withLock { $0 } }
}

private func qualify(model: URL, reference: LanguageReference, forced: [Float], decodeSteps: Int) async throws -> ContractReport {
    var report = ContractReport()
    report.modelFileSHA256 = try sha256(model.appendingPathComponent("main.mlirb"))
    let vocab = reference.forcedLogits.vocab

    // Teacher-forced rows and the short greedy continuation, with registered EOS.
    let bounds = try exportBounds(for: model)
    let session = try await LanguageGenerationSession.open(at: model, maximumContextLength: 2048, exportBounds: bounds)
    report.contract = session.contract.rawValue
    let rows = try await session.teacherForcedLogits(prompt: reference.promptTokens, continuation: Array(reference.forcedTokens.dropLast()))
    report.forcedRows = rows.enumerated().map { index, row in
        compare(row: index, native: row, reference: forced[(index * vocab)..<((index + 1) * vocab)])
    }
    report.forcedMinimumCosine = report.forcedRows.map(\.cosine).min() ?? 0
    report.forcedMaximumAbs = report.forcedRows.map(\.maxAbs).max() ?? 0
    report.forcedArgmaxMismatches = report.forcedRows.filter { !$0.argmaxEqual }.count
    report.forcedNearTies = report.forcedRows.filter(\.nearTie).count
    await session.unload()

    let open = { try await LanguageGenerationSession.open(at: model, maximumContextLength: 2048, stopTokenIDs: [151935], exportBounds: bounds) }
    let greedy = try await open()
    report.greedyTokens = try await greedy.generate(prompt: reference.promptTokens, maximumTokens: reference.greedyTokens.count).tokens
    report.greedyFirstDivergence = zip(report.greedyTokens, reference.greedyTokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        ?? (report.greedyTokens.count == reference.greedyTokens.count ? nil : min(report.greedyTokens.count, reference.greedyTokens.count))
    if let d = report.greedyFirstDivergence, d < reference.forcedLogits.rows, d < report.greedyTokens.count,
       reference.forcedTokens.prefix(d) == reference.greedyTokens.prefix(d) {
        // A one-hot "native row" for the chosen token yields the reference margin and that token's rank.
        var chosen = [Float](repeating: 0, count: vocab)
        chosen[report.greedyTokens[d]] = 1
        report.greedyDivergenceRow = compare(row: d, native: chosen, reference: forced[(d * vocab)..<((d + 1) * vocab)])
    }

    // Long run in a fresh request on the same session: no recycling for the bucketed contract.
    let clock = TokenClock()
    let long = try await greedy.generate(prompt: reference.promptTokens, maximumTokens: decodeSteps, onToken: { _ in clock.record() })
    let statistics = await greedy.cacheStatistics()
    report.longRunSteps = long.tokens.count
    report.longRunTokens = long.tokens
    report.longRunModelReloads = statistics.modelReloadCount
    report.longRunDistinctStepShapes = statistics.distinctStepShapes
    report.longRunAllocatedCapacity = statistics.allocatedCapacity
    report.longRunFootprint = clock.recorded
    await greedy.unload()
    return report
}

/// The bounds the Forge exporter recorded next to the asset (`export-report.json`).
private func exportBounds(for model: URL) throws -> LanguageExportBounds? {
    struct Report: Decodable { let exportBounds: LanguageExportBounds? }
    let url = model.deletingLastPathComponent().appendingPathComponent("export-report.json")
    return try JSONDecoder().decode(Report.self, from: Data(contentsOf: url)).exportBounds
}

/// The sessions compared for timing: the bucketed export, the running export with its default
/// recycling (what 0.0.1 consumers get), and the running export with recycling and memory-budget
/// reloads disabled (its pure per-step cost; the native shape cache then grows for the whole run).
private func timingSessions(bucketed: URL, running: URL, context: Int) async throws -> [(String, LanguageGenerationSession)] {
    [("bucketed", try await LanguageGenerationSession.open(at: bucketed, maximumContextLength: context, stopTokenIDs: [151935],
                                                         exportBounds: try exportBounds(for: bucketed))),
     ("running", try await LanguageGenerationSession.open(at: running, maximumContextLength: context, stopTokenIDs: [151935],
                                                        exportBounds: try exportBounds(for: running))),
     ("runningNoRecycle", try await LanguageGenerationSession.open(at: running, maximumContextLength: context, stopTokenIDs: [151935],
                                                                 maximumSubmissionsPerModel: .max, memoryGrowthBudgetBytes: nil,
                                                                 exportBounds: try exportBounds(for: running)))]
}

/// Before/after timing on a shared host: all sessions stay resident and every round runs each
/// contract on each prompt, rotating which goes first, so host drift hits them alike.
private func pairedTiming(bucketed: URL, running: URL, prompts: [String: [Int]], rounds: Int,
                          into report: inout LanguageQualificationReport) async throws {
    let sessions = try await timingSessions(bucketed: bucketed, running: running, context: 2048)
    let labels = prompts.keys.sorted()
    for (_, session) in sessions { for label in labels { _ = try await session.generate(prompt: prompts[label]!, maximumTokens: 256) } }
    for round in 0..<rounds {
        var load = [0.0, 0.0, 0.0]
        _ = getloadavg(&load, 3)
        report.pairedLoadAverages.append(load[0])
        let shift = round % sessions.count
        for (name, session) in Array(sessions[shift...] + sessions[..<shift]) {
            for label in labels {
                let reply = try await session.generate(prompt: prompts[label]!, maximumTokens: 256)
                report.pairedTimings["\(name)-\(label)", default: []].append(Timing(promptTokens: prompts[label]!.count,
                    generatedTokens: reply.tokens.count, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds,
                    totalSeconds: reply.totalSeconds))
            }
        }
    }
    for (_, session) in sessions { await session.unload() }
}

/// Long context: one request per session with a prompt filling `context - 1024` tokens, then 512
/// decoded tokens, sampling the process footprint (all sessions resident; the bucketed one first).
private func longContextProbe(bucketed: URL, running: URL, prompt: [Int], context: Int,
                              into report: inout LanguageQualificationReport) async throws {
    let sessions = try await timingSessions(bucketed: bucketed, running: running, context: context)
    let long = Array(Array(repeating: prompt, count: (context - 1024) / prompt.count + 1).joined().prefix(context - 1024))
    for (name, session) in sessions {
        let clock = TokenClock()
        let reply = try await session.generate(prompt: long, maximumTokens: 512, onToken: { _ in clock.record() })
        let statistics = await session.cacheStatistics()
        report.longContext[name] = LongContextReport(promptTokens: long.count, timing: Timing(promptTokens: long.count,
            generatedTokens: reply.tokens.count, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, totalSeconds: reply.totalSeconds),
            allocatedCapacity: statistics.allocatedCapacity, allocatedBytes: statistics.allocatedBytes,
            modelReloads: statistics.modelReloadCount, distinctStepShapes: statistics.distinctStepShapes, footprint: clock.recorded)
    }
    for (_, session) in sessions { await session.unload() }
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_QUALIFICATION"] == "1"), .timeLimit(.minutes(60)))
func languageBucketedDecodeQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    var report = LanguageQualificationReport()
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_OUTPUT"]))
    defer {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: output)
    }
    do {
        let referenceDirectory = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_REFERENCE"]))
        let referenceJSON = referenceDirectory.appendingPathComponent("reference.json")
        let reference = try JSONDecoder().decode(LanguageReference.self, from: Data(contentsOf: referenceJSON))
        let logitsURL = referenceDirectory.appendingPathComponent(reference.forcedLogits.file)
        report.referenceSHA256 = ["reference.json": try sha256(referenceJSON), reference.forcedLogits.file: try sha256(logitsURL)]
        guard report.referenceSHA256[reference.forcedLogits.file] == reference.forcedLogits.sha256 else {
            throw NativeFailure(description: "reference logits hash mismatch")
        }
        let forced = try Data(contentsOf: logitsURL).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard forced.count == reference.forcedLogits.rows * reference.forcedLogits.vocab,
              reference.forcedLogits.rows == reference.forcedTokens.count else { throw NativeFailure(description: "reference logits shape") }
        let steps = Int(nativeEnvironment["AETHER_LANGUAGE_DECODE_STEPS"] ?? "") ?? 1024

        report.stage = "bucketed"
        report.bucketed = try await qualify(model: URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_BUCKETED_MODEL"])),
                                            reference: reference, forced: forced, decodeSteps: steps)
        if let running = nativeEnvironment["AETHER_LANGUAGE_RUNNING_MODEL"] {
            report.stage = "running"
            let before = try await qualify(model: URL(filePath: running), reference: reference, forced: forced, decodeSteps: steps)
            report.running = before
            report.bucketedLongRunMatchesRunning = zip(report.bucketed.longRunTokens, before.longRunTokens).prefix { $0 == $1 }.count
            report.stage = "paired timing"
            let longPrompt = Array(Array(repeating: reference.promptTokens, count: 1000 / reference.promptTokens.count + 1).joined().prefix(1000))
            try await pairedTiming(bucketed: URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_BUCKETED_MODEL"])),
                                   running: URL(filePath: running),
                                   prompts: ["short": reference.promptTokens, "long": longPrompt],
                                   rounds: Int(nativeEnvironment["AETHER_LANGUAGE_TIMING_ROUNDS"] ?? "") ?? 6, into: &report)
            if let context = Int(nativeEnvironment["AETHER_LANGUAGE_LONG_CONTEXT"] ?? "") {
                report.stage = "long context"
                try await longContextProbe(bucketed: URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_BUCKETED_MODEL"])),
                                           running: URL(filePath: running), prompt: reference.promptTokens, context: context,
                                           into: &report)
            }
        }
        report.stage = "verdict"
        let b = report.bucketed
        report.passed = b.contract == LanguageStepContract.bucketedPositions.rawValue
            && b.forcedMinimumCosine >= 0.995
            && b.forcedRows.allSatisfy { $0.argmaxEqual || $0.nearTie }
            && b.greedyAccepted
            && b.longRunSteps >= 1000 && b.longRunModelReloads == 0
        report.stage = "complete"
    } catch {
        report.error = String(describing: error)
        throw error
    }
    #expect(report.passed)
}
