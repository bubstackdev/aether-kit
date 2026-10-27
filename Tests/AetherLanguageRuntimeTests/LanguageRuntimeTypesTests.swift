@testable @_spi(Qualification) import AetherLanguageRuntime
import Foundation
import Testing

private struct GeneratorFixture: Decodable {
    struct Case: Decodable { let seed: String; let splitmix64: [String]; let xoshiro256starstar: [String] }
    let cases: [Case]
}

struct LanguageRuntimeTypesTests {
    /// Known answers from `Forge/language/reference_sampling.py`'s own implementation of both published algorithms.
    @Test func randomGeneratorMatchesIndependentReference() throws {
        let url = try #require(Bundle.module.url(forResource: "random-generator", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(GeneratorFixture.self, from: Data(contentsOf: url))
        #expect(fixture.cases.count == 5)
        for entry in fixture.cases {
            let seed = try #require(UInt64(entry.seed))
            var mix = SplitMix64(state: seed)
            #expect((0..<8).map { _ in mix.next() } == entry.splitmix64.compactMap(UInt64.init))
            var generator = LanguageRandomGenerator(seed: seed)
            #expect((0..<8).map { _ in generator.next() } == entry.xoshiro256starstar.compactMap(UInt64.init))
        }
        var a = LanguageRandomGenerator(seed: 9), b = LanguageRandomGenerator(seed: 9)
        let draws = (0..<1000).map { _ in a.nextUnitInterval() }
        #expect(draws == (0..<1000).map { _ in b.nextUnitInterval() })
        #expect(draws.allSatisfy { $0 >= 0 && $0 < 1 })
    }

    @Test func tokenSetOperations() {
        var set = LanguageTokenSet([0, 5, 63, 64, 129, 129, -1, 200], vocabularySize: 130)
        #expect(set.ids == [0, 5, 63, 64, 129] && set.count == 5 && !set.isEmpty)
        #expect(set.contains(63) && !set.contains(62) && !set.contains(-1) && !set.contains(130))
        set.remove(5); set.insert(6); set.insert(130)
        #expect(set.ids == [0, 6, 63, 64, 129])
        let range = LanguageTokenSet(range: 60..<129, vocabularySize: 130)
        #expect(range.ids == Array(60..<129))
        #expect(LanguageTokenSet(range: 0..<1000, vocabularySize: 130).ids == Array(0..<130))
        #expect(LanguageTokenSet(range: 64..<64, vocabularySize: 130).isEmpty)
        #expect(set.intersection(range).ids == [63, 64])
        #expect(LanguageTokenConstraint.unconstrained.intersection(.allowed(range)) == .allowed(range))
        #expect(LanguageTokenConstraint.allowed(set).intersection(.allowed(range)) == .allowed(set.intersection(range)))
        var logits = [Float](repeating: 1, count: 130)
        set.mask(&logits)
        #expect(logits.indices.filter { logits[$0] == 1 } == set.ids)
        #expect(logits.allSatisfy { $0 == 1 || $0 == -.infinity })
    }

    @Test func samplingValidation() throws {
        try LanguageSampling.greedy.validate()
        try LanguageSampling.random(temperature: 0.7, topK: 40, topP: 0.9, seed: 1).validate()
        #expect(LanguageSampling.greedy.isPlainGreedy && LanguageSampling().isPlainGreedy)
        #expect(!LanguageSampling(repetitionPenalty: 1.1).isPlainGreedy && !LanguageSampling.random().isPlainGreedy)
        for invalid in [LanguageSampling.random(temperature: 0), .random(temperature: .infinity), .random(topK: 0),
                        .random(topP: 0), .random(topP: 1.01), .random(topP: .nan),
                        LanguageSampling(repetitionPenalty: 0), LanguageSampling(repetitionPenalty: 1.2, repetitionWindow: 0)] {
            #expect(throws: LanguageGenerationError.self, "\(invalid)") { try invalid.validate() }
        }
    }

    /// The protocol can be adopted by an actor with `nonisolated let` properties, as a backend would.
    @Test func protocolIsAdoptableByAnActor() async throws {
        let backend: any LanguageBackend = EchoBackend()
        let collector = Collector()
        let result = try await backend.generate(LanguageGenerationRequest(prompt: [1, 2], maximumTokens: 3)) { token in
            await collector.add(token.id)
            return .continue
        }
        let delivered = await collector.ids
        #expect(result.tokens == [2, 1, 2] && delivered == result.tokens && result.stopReason == .outputLimit && result.seed == nil)
        let rows = try await (backend as! any LanguageBackendQualification).teacherForcedLogits(prompt: [1], continuation: [2])
        #expect(rows.count == 2)
    }
}

private actor Collector {
    var ids: [Int] = []
    func add(_ id: Int) { ids.append(id) }
}

private actor EchoBackend: LanguageBackendQualification {
    nonisolated let vocabularySize = 4
    nonisolated let maximumContextLength = 16
    nonisolated let stopTokenIDs: Set<Int> = [0]

    func generate(_ request: LanguageGenerationRequest,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult {
        let tokens = (0..<request.maximumTokens).map { request.prompt.reversed()[$0 % request.prompt.count] }
        for (index, token) in tokens.enumerated() { _ = try await onToken?(LanguageGeneratedToken(id: token, index: index)) }
        return LanguageGenerationResult(tokens: tokens, stopReason: .outputLimit, seed: nil, timeToFirstTokenSeconds: 0, totalSeconds: 0)
    }

    func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]] {
        Array(repeating: [Float](repeating: 0, count: vocabularySize), count: continuation.count + 1)
    }

    func unload() async {}
}
