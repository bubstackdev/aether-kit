import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import Foundation
import Testing

/// Opt-in native qualification of the language runtime seam on a real model: the `LanguageBackend` protocol's default
/// request against the existing API, seeded random sampling (repeats, including in a freshly loaded session, and
/// differs by seed), an allowed-set constraint (digits, then only end of sequence) on every token, and the per-token
/// cost of each sampling mode and of slow constraint processors against greedy. Prompts render through the model's
/// own template. Run under the shared GPU lease (`with-gpu-lock.sh`), release build:
///
/// AETHER_NATIVE_LANGUAGE_SAMPLING=1
/// AETHER_LANGUAGE_SAMPLING_BUNDLE=<registered bundle with language-model.json>, or
///   AETHER_LANGUAGE_SAMPLING_MODEL=<bucketed model.aimodel beside its export-report.json> and
///   AETHER_LANGUAGE_SAMPLING_TOKENIZER=<its tokenizer directory>
/// AETHER_LANGUAGE_SAMPLING_OUTPUT=<report .json>
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_LANGUAGE_SAMPLING"] == "1"), .timeLimit(.minutes(30)))
func languageSamplingQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let output = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_SAMPLING_OUTPUT"]))
    let open: @Sendable () async throws -> LanguageGenerationSession
    let tokenizer: LanguageTokenizer
    var report = SamplingReport()
    if let bundle = nativeEnvironment["AETHER_LANGUAGE_SAMPLING_BUNDLE"].map({ URL(filePath: $0) }) {
        tokenizer = try await LanguageTokenizer.open(bundle: bundle)
        report.model = try LanguageModelManifest.load(from: bundle).sourceRepository
        open = { try await LanguageGenerationSession.open(bundle: bundle, platform: .macos) }
    } else {
        let model = URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_SAMPLING_MODEL"]))
        tokenizer = try await LanguageTokenizer.open(directory: URL(filePath: try #require(nativeEnvironment["AETHER_LANGUAGE_SAMPLING_TOKENIZER"])))
        struct Export: Decodable { let exportBounds: LanguageExportBounds? }
        let bounds = try JSONDecoder().decode(Export.self, from: Data(contentsOf: model.deletingLastPathComponent()
            .appendingPathComponent("export-report.json"))).exportBounds
        report.model = model.deletingLastPathComponent().lastPathComponent
        open = { try await LanguageGenerationSession.open(at: model, maximumContextLength: 2048, stopTokenIDs: [151643, 151645],
                                                          exportBounds: bounds) }
    }
    let prompts = try [
        ("rewrite", [LanguageChatMessage.system("Rewrite the picture description as one vivid sentence for an image generator. Reply with the sentence only."),
                     .user("A lighthouse on a cliff at dusk, gulls overhead, warm light in the lantern room.")]),
        ("story", [.system("You write short, gentle stories."), .user("Tell a story about a cartographer who maps clouds.")]),
    ].map { (name: $0.0, tokens: try tokenizer.render($0.1, thinking: false).tokenIDs) }
    let session = try await open()
    report.vocabularySize = session.vocabularySize

    // 1. The protocol's default request is the existing greedy API, token for token.
    for prompt in prompts {
        let old = try await session.generate(prompt: prompt.tokens, maximumTokens: 96)
        let delivered = DeliveryCount()
        let new = try await session.generate(LanguageGenerationRequest(prompt: prompt.tokens, maximumTokens: 96)) { _ in
            delivered.add(); return .continue
        }
        report.defaultPath.append(.init(prompt: prompt.name, tokens: old.tokens, protocolTokens: new.tokens,
                                        deliveries: delivered.value, identical: old.tokens == new.tokens && old.stopReason == new.stopReason.rawValue))
        #expect(old.tokens == new.tokens && delivered.value == new.tokens.count, "\(prompt.name): protocol default equals the existing API")
    }

    // 2. Seeded random sampling repeats (also after other requests, and in a freshly loaded model) and differs by seed.
    let random = LanguageSampling.random(temperature: 0.8, topK: 40, topP: 0.95, seed: 1234)
    let story = prompts[1].tokens
    let first = try await session.generate(LanguageGenerationRequest(prompt: story, maximumTokens: 64, sampling: random), onToken: nil)
    _ = try await session.generate(prompt: prompts[0].tokens, maximumTokens: 32)
    let again = try await session.generate(LanguageGenerationRequest(prompt: story, maximumTokens: 64, sampling: random), onToken: nil)
    var other = random; other.mode = .random(temperature: 0.8, topK: 40, topP: 0.95, seed: 1235)
    let different = try await session.generate(LanguageGenerationRequest(prompt: story, maximumTokens: 64, sampling: other), onToken: nil)
    let fresh = try await open()
    let reloaded = try await fresh.generate(LanguageGenerationRequest(prompt: story, maximumTokens: 64, sampling: random), onToken: nil)
    await fresh.unload()
    report.seeded = .init(seed: 1234, tokens: first.tokens, repeatedAfterOtherRequests: again.tokens == first.tokens,
                          repeatedInFreshSession: reloaded.tokens == first.tokens, otherSeedTokens: different.tokens,
                          otherSeedDiffers: different.tokens != first.tokens, reportedSeed: first.seed)
    #expect(first.seed == 1234 && again.tokens == first.tokens && reloaded.tokens == first.tokens && different.tokens != first.tokens)

    // 3. A constraint holds on every token: single-character digits, then only end of sequence.
    let digits = Set((0...9).compactMap { tokenizer.encode(String($0)).count == 1 ? tokenizer.encode(String($0))[0] : nil })
    let numbers = try tokenizer.render([.user("List some numbers.")], thinking: false).tokenIDs
    let processor = DigitsThenStop(digits: LanguageTokenSet(digits, vocabularySize: session.vocabularySize),
                                   stop: LanguageTokenSet(session.stopTokenIDs, vocabularySize: session.vocabularySize), limit: 12)
    let constrained = try await session.generate(LanguageGenerationRequest(prompt: numbers, maximumTokens: 64, constraint: processor),
                                                 onToken: nil)
    report.constrained = .init(allowedDigitTokens: digits.sorted(), tokens: constrained.tokens, text: tokenizer.decode(constrained.tokens),
                               stopReason: constrained.stopReason.rawValue, everyTokenAllowed: constrained.tokens.allSatisfy(digits.contains))
    #expect(digits.count == 10 && constrained.tokens.allSatisfy(digits.contains) && constrained.tokens.count <= 12
            && constrained.stopReason == .endOfSequence)

    // 4. Per-token decode cost by mode (warm; tokens after the first). The host is shared, so these are observations.
    let modes: [(String, LanguageSampling, (any LanguageLogitsProcessor)?)] = [
        ("greedy", .greedy, nil),
        ("random-temperature", .random(temperature: 0.8, seed: 1), nil),
        ("random-top-k-40-top-p-0.95", random, nil),
        ("random-top-p-0.9", .random(temperature: 0.8, topP: 0.9, seed: 1), nil),
        ("greedy-repetition-penalty-1.1", LanguageSampling(repetitionPenalty: 1.1), nil),
        ("greedy-allow-all-processor", .greedy, AllowAll(vocabulary: session.vocabularySize, delay: nil)),
        ("greedy-processor-5ms", .greedy, AllowAll(vocabulary: session.vocabularySize, delay: .milliseconds(5))),
        ("greedy-processor-40ms", .greedy, AllowAll(vocabulary: session.vocabularySize, delay: .milliseconds(40))),
    ]
    for round in 0..<3 {
        for (name, sampling, processor) in modes {
            let reply = try await session.generate(LanguageGenerationRequest(prompt: story, maximumTokens: 96, sampling: sampling,
                                                                            constraint: processor), onToken: nil)
            guard round > 0, reply.tokens.count > 8 else { continue }
            report.decodeMillisecondsPerToken[name, default: []].append(
                (reply.totalSeconds - reply.timeToFirstTokenSeconds) * 1000 / Double(reply.tokens.count - 1))
        }
    }
    let statistics = await session.cacheStatistics()
    report.modelReloads = statistics.modelReloadCount
    report.distinctStepShapes = statistics.distinctStepShapes
    await session.unload()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: output)
}

private final class DeliveryCount: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func add() { lock.withLock { count += 1 } }
    var value: Int { lock.withLock { count } }
}

/// Digits for up to `limit` tokens, then only the stop tokens.
private actor DigitsThenStop: LanguageLogitsProcessor {
    let digits: LanguageTokenSet, stop: LanguageTokenSet, limit: Int
    private var accepted = 0
    init(digits: LanguageTokenSet, stop: LanguageTokenSet, limit: Int) { self.digits = digits; self.stop = stop; self.limit = limit }
    func begin(prompt: [Int]) -> LanguageTokenConstraint { accepted = 0; return .allowed(digits) }
    func accept(_ token: Int) -> LanguageTokenConstraint {
        accepted += 1
        return .allowed(accepted < limit ? digits : stop)
    }
}

/// Allows every token, optionally after a delay standing in for a grammar matcher's work.
private struct AllowAll: LanguageLogitsProcessor {
    let vocabulary: Int, delay: Duration?
    func begin(prompt: [Int]) async throws -> LanguageTokenConstraint { .unconstrained }
    func accept(_ token: Int) async throws -> LanguageTokenConstraint {
        if let delay { try await Task.sleep(for: delay) }
        return .allowed(LanguageTokenSet(range: 0..<vocabulary, vocabularySize: vocabulary))
    }
}

private struct SamplingReport: Encodable {
    struct DefaultPath: Encodable { let prompt: String; let tokens: [Int]; let protocolTokens: [Int]; let deliveries: Int; let identical: Bool }
    struct Seeded: Encodable {
        let seed: UInt64; let tokens: [Int]; let repeatedAfterOtherRequests: Bool; let repeatedInFreshSession: Bool
        let otherSeedTokens: [Int]; let otherSeedDiffers: Bool; let reportedSeed: UInt64?
    }
    struct Constrained: Encodable {
        let allowedDigitTokens: [Int]; let tokens: [Int]; let text: String; let stopReason: String; let everyTokenAllowed: Bool
    }
    var device = LanguageGenerationSession.deviceArchitectureName
    var model = ""
    var vocabularySize = 0
    var hostLoad = "Shared host: GPU jobs of cooperating agents are serialized by a lock, other host work is not; timings are observations"
    var defaultPath: [DefaultPath] = []
    var seeded: Seeded?
    var constrained: Constrained?
    /// Warm requests of up to 96 tokens on the story prompt, rounds 2 and 3.
    var decodeMillisecondsPerToken: [String: [Double]] = [:]
    var modelReloads = 0
    var distinctStepShapes = 0
}
