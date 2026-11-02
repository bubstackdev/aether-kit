import AetherKit
import AetherLanguageCoreAI
import AetherLanguageTokenizers
import Foundation
import Testing

/// Real-checkpoint completion port gate. No chat template is added to a base model.
/// AETHER_NATIVE_PHI2=1 AETHER_PHI2_BUNDLE=... AETHER_PHI2_REFERENCE=... AETHER_PHI2_OUTPUT=...
@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_PHI2"] == "1"), .timeLimit(.minutes(20)))
func phi2CompletionQualification() async throws {
    let gpu = try await nativeGPUGrant(); defer { gpu?.release() }
    let bundle = URL(filePath: try #require(nativeEnvironment["AETHER_PHI2_BUNDLE"]))
    let referenceDirectory = URL(filePath: try #require(nativeEnvironment["AETHER_PHI2_REFERENCE"]))
    let reference = try JSONDecoder().decode(Phi2Reference.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("reference.json")))
    let data = try Data(contentsOf: referenceDirectory.appendingPathComponent(reference.forcedLogits.file))
    let expected = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    #expect(expected.count == reference.forcedLogits.rows * reference.forcedLogits.vocab)
    let manifest = try LanguageModelManifest.load(from: bundle)
    #expect(manifest.maximumContextLength == 2048)
    let tokenizer = try await LanguageTokenizer.open(bundle: bundle)
    #expect(!tokenizer.hasChatTemplate)
    #expect(tokenizer.encode(reference.prompt) == reference.promptTokens)
    let session = try await LanguageGenerationSession.open(bundle: bundle, platform: .macos)
    let rows = try await session.teacherForcedLogits(prompt: reference.promptTokens, continuation: Array(reference.forcedTokens.dropLast()))
    var minimumCosine = 1.0, maximumAbs = 0.0, mismatches = 0, nearTies = 0
    for (index, row) in rows.enumerated() {
        let target = expected[(index * reference.forcedLogits.vocab)..<((index + 1) * reference.forcedLogits.vocab)].map(Double.init)
        let actual = row.map(Double.init)
        let dot = zip(actual, target).reduce(0.0) { $0 + $1.0 * $1.1 }
        let norm = (actual.reduce(0) { $0 + $1 * $1 } * target.reduce(0) { $0 + $1 * $1 }).squareRoot()
        minimumCosine = min(minimumCosine, dot / norm)
        maximumAbs = max(maximumAbs, zip(actual, target).map { abs($0 - $1) }.max() ?? 0)
        let a = actual.indices.max { actual[$0] < actual[$1] }
        let b = target.indices.max { target[$0] < target[$1] }
        if a != b {
            mismatches += 1
            if let a, phi2NearTie(token: a, logits: target) { nearTies += 1 }
        }
    }
    #expect(rows.count == reference.forcedLogits.rows)
    #expect(minimumCosine >= 0.995 && mismatches == nearTies)
    let first = try await session.generate(prompt: reference.promptTokens, maximumTokens: reference.greedyTokens.count, prefillChunkSize: 7)
    let again = try await session.generate(prompt: reference.promptTokens, maximumTokens: reference.greedyTokens.count, prefillChunkSize: 16)
    let divergence = zip(first.tokens, reference.greedyTokens).enumerated().first { $0.element.0 != $0.element.1 }?.offset
    let identical = first.tokens == reference.greedyTokens
    var divergenceAccepted = identical
    if let divergence, divergence < reference.forcedLogits.rows {
        let target = expected[(divergence * reference.forcedLogits.vocab)..<((divergence + 1) * reference.forcedLogits.vocab)].map(Double.init)
        divergenceAccepted = phi2NearTie(token: first.tokens[divergence], logits: target)
    }
    #expect(divergenceAccepted)
    #expect(again.tokens == first.tokens)
    if identical { #expect(tokenizer.decode(first.tokens) == reference.greedyText) }
    let statistics = await session.cacheStatistics()
    let requiresGrowth = reference.promptTokens.count <= 256 && reference.promptTokens.count + reference.greedyTokens.count - 1 > 256
    let growthVerified = !requiresGrowth || statistics.growthCount > 0
    #expect(growthVerified)
    await session.unload()
    let report = Phi2Report(passed: minimumCosine >= 0.995 && mismatches == nearTies && divergenceAccepted && again.tokens == first.tokens && growthVerified,
        device: LanguageGenerationSession.deviceArchitectureName, sourceRepository: manifest.sourceRepository,
        sourceRevision: manifest.sourceRevision, modelSHA256: manifest.modelSHA256,
        forcedRows: rows.count, minimumCosine: minimumCosine, maximumAbs: maximumAbs, argmaxMismatches: mismatches, acceptedNearTies: nearTies,
        greedyIdentical: identical, greedyFirstDivergence: divergence, greedyDivergenceAccepted: divergenceAccepted,
        greedyTokens: first.tokens, text: tokenizer.decode(first.tokens),
        allocatedCapacity: statistics.allocatedCapacity, cacheGrowths: statistics.growthCount, modelReloads: statistics.modelReloadCount)
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: URL(filePath: try #require(nativeEnvironment["AETHER_PHI2_OUTPUT"])))
}

private struct Phi2Reference: Decodable {
    let prompt: String, greedyText: String
    let promptTokens: [Int], greedyTokens: [Int], forcedTokens: [Int]
    let forcedLogits: Logits
    struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int }
}

private struct Phi2Report: Encodable {
    let passed: Bool
    let device: String, sourceRepository: String, sourceRevision: String, modelSHA256: String
    let forcedRows: Int, minimumCosine: Double, maximumAbs: Double, argmaxMismatches: Int
    let acceptedNearTies: Int
    let greedyIdentical: Bool, greedyFirstDivergence: Int?, greedyDivergenceAccepted: Bool
    let greedyTokens: [Int], text: String
    let allocatedCapacity: Int, cacheGrowths: Int, modelReloads: Int
}

// Same fp16 policy as LanguageDecodeQualificationTests: reference probability margin
// below 0.1, and the native choice within reference top five. Never compares diverged prefixes.
private func phi2NearTie(token: Int, logits: [Double]) -> Bool {
    let top = logits.indices.sorted { logits[$0] > logits[$1] }.prefix(5)
    guard top.count >= 2, top.contains(token), let first = top.first else { return false }
    let total = logits.reduce(0.0) { $0 + exp($1 - logits[first]) }
    let second = top.dropFirst().first!
    return (1.0 - exp(logits[second] - logits[first])) / total < 0.1
}
