import AetherKit
import AetherLanguageCoreAI
import Foundation
import Synchronization
import Testing

/// Opt-in measurement of time to first token on raw language assets (for example a bucketed and a running export of the
/// same checkpoint), at several prompt lengths. For each asset and length: the first request's time (first use of its
/// shapes) and the median of warm rounds interleaved across assets, each split by the native submissions an observer
/// timestamps: how many, their widths, the GPU time (submit to join) and the host time around them (cache growth and
/// zeroing, update-row copies, logits reads). Prompts are a fixed synthetic token pattern; one token is generated. The
/// bucketed asset is also measured at other prefill widths.
///
/// AETHER_NATIVE_LANGUAGE_TTFT=1
/// AETHER_LANGUAGE_TTFT_MODELS=<label=model.aimodel,...> (each beside its export-report.json)
/// AETHER_LANGUAGE_TTFT_OUTPUT=<report .json>
/// AETHER_LANGUAGE_TTFT_LENGTHS=<default 64,256,1024,2048>
/// AETHER_LANGUAGE_TTFT_ROUNDS=<default 5>
/// AETHER_LANGUAGE_TTFT_WIDTHS=<bucketed prefill widths, default 64,128,256,512>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_TTFT"] == "1"), .timeLimit(.minutes(60)))
func languageTimeToFirstToken() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_TTFT_OUTPUT"]))
    let lengths = (nativeEnvironment["AETHER_LANGUAGE_TTFT_LENGTHS"] ?? "64,256,1024,2048").split(separator: ",").compactMap { Int($0) }
    let rounds = Int(nativeEnvironment["AETHER_LANGUAGE_TTFT_ROUNDS"] ?? "") ?? 5
    let widths = (nativeEnvironment["AETHER_LANGUAGE_TTFT_WIDTHS"] ?? "64,128,256,512").split(separator: ",").compactMap { Int($0) }
    var report = TimingReport()
    report.lengths = lengths
    var sessions: [(label: String, session: LanguageGenerationSession)] = []
    for entry in try #require(nativeEnvironment["AETHER_LANGUAGE_TTFT_MODELS"]).split(separator: ",") {
        let parts = entry.split(separator: "=", maxSplits: 1).map(String.init)
        let model = URL(filePath: parts[1])
        struct Export: Decodable { let exportBounds: LanguageExportBounds? }
        let bounds = try JSONDecoder().decode(Export.self, from: Data(contentsOf: model.deletingLastPathComponent()
            .appendingPathComponent("export-report.json"))).exportBounds
        // No memory budget and no recycling: reloads would land inside the timed requests.
        let session = try await LanguageGenerationSession.open(at: model, maximumContextLength: 4096, maximumSubmissionsPerModel: .max,
                                                               memoryGrowthBudgetBytes: nil, exportBounds: bounds)
        sessions.append((parts[0], session))
        report.contracts[parts[0]] = session.contract.rawValue
    }
    let prompts = Dictionary(uniqueKeysWithValues: lengths.map { length in (length, (0..<length).map { 1000 + ($0 * 37) % 5000 }) })

    func timed(_ session: LanguageGenerationSession, _ length: Int, width: Int? = nil) async throws -> Sample {
        let events = Mutex<[(Bool, Int, ContinuousClock.Instant)]>([])
        let start = ContinuousClock.now
        let reply = try await session.generate(prompt: prompts[length]!, maximumTokens: 1, prefillChunkSize: width,
                                               observer: { event in
            let now = ContinuousClock.now
            events.withLock { recorded in
                switch event {
                case .submitted(_, let count): recorded.append((true, count, now))
                case .completed(_, let count): recorded.append((false, count, now))
                }
            }
        })
        let wall = seconds(start, .now)
        let recorded = events.withLock { $0 }
        var gpu = 0.0, widths: [Int] = [], lastSubmit = start
        for (submitted, count, at) in recorded {
            if submitted { lastSubmit = at; widths.append(count) } else { gpu += seconds(lastSubmit, at) }
        }
        return Sample(timeToFirstToken: reply.timeToFirstTokenSeconds, wall: wall, gpu: gpu, host: reply.timeToFirstTokenSeconds - gpu,
                      submissions: widths.count, widths: Array(Set(widths)).sorted(), firstToken: reply.tokens.first ?? -1)
    }

    // First use of each length's shapes, then warm rounds interleaved across assets.
    for (label, session) in sessions {
        for length in lengths { report.cold[label, default: [:]][String(length)] = try await timed(session, length) }
    }
    var warm: [String: [String: [Sample]]] = [:]
    for round in 0..<rounds {
        let order = round % 2 == 0 ? sessions : sessions.reversed()
        for (label, session) in order {
            for length in lengths { warm[label, default: [:]][String(length), default: []].append(try await timed(session, length)) }
        }
    }
    report.warm = warm.mapValues { $0.mapValues(median) }
    // Other bucketed prefill widths (warmed first).
    for (label, session) in sessions where session.contract == .bucketedPositions {
        for width in widths {
            for length in lengths { _ = try await timed(session, length, width: width) }
            var samples: [String: [Sample]] = [:]
            for _ in 0..<rounds { for length in lengths { samples[String(length), default: []].append(try await timed(session, length, width: width)) } }
            report.widths["\(label)@\(width)"] = samples.mapValues(median)
        }
    }
    // Greedy first tokens agree across assets and widths.
    report.firstTokensAgree = lengths.allSatisfy { length in
        Set(report.warm.values.compactMap { $0[String(length)]?.firstToken }).count == 1
    }
    for (_, session) in sessions { await session.unload() }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(report).write(to: output)
}

private func seconds(_ start: ContinuousClock.Instant, _ end: ContinuousClock.Instant) -> Double {
    let elapsed = end - start
    return Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
}

/// The sample with the median time to first token.
private func median(_ samples: [Sample]) -> Sample { samples.sorted { $0.timeToFirstToken < $1.timeToFirstToken }[samples.count / 2] }

private struct Sample: Codable {
    let timeToFirstToken: Double, wall: Double, gpu: Double, host: Double
    let submissions: Int, widths: [Int], firstToken: Int
}

private struct TimingReport: Codable {
    var device = LanguageGenerationSession.deviceArchitectureName
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; timings are observations"
    var lengths: [Int] = []
    var contracts: [String: String] = [:]
    var cold: [String: [String: Sample]] = [:]
    var warm: [String: [String: Sample]] = [:]
    var widths: [String: [String: Sample]] = [:]
    var firstTokensAgree = false
}
