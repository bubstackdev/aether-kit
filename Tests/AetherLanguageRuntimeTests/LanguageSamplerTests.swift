@testable import AetherLanguageRuntime
import Foundation
import Testing

private struct WarperFixture: Decodable {
    struct Case: Decodable {
        let name: String
        let logits: [Float]
        let history: [Int]
        let repetitionPenalty: Double
        let allowed: [Int]?
        let temperature: Double
        let topK: Int?
        let topP: Double?
        let greedy: Int
        let kept: [Int]
        let probabilities: [Double]
    }
    let transformers: String
    let cases: [Case]
}

/// The runtime's greedy before the sampler existed, verbatim: the oracle for the default path.
private func originalGreedy(_ logits: [Float]) throws -> Int {
    var best = -Float.infinity
    var token: Int?
    for (i, score) in logits.enumerated() {
        guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
        if score.isFinite && score > best { best = score; token = i }
    }
    guard let token else { throw LanguageGenerationError.invalid("No finite logits") }
    return token
}

private func result(_ body: () throws -> Int) -> Result<Int, LanguageGenerationError> {
    do { return .success(try body()) } catch { return .failure(error as! LanguageGenerationError) }
}

struct LanguageSamplerTests {
    @Test func plainGreedyMatchesTheOriginalOnFuzzedRows() throws {
        var generator = LanguageRandomGenerator(seed: 5)
        let specials: [Float] = [.infinity, -.infinity, .nan, 0, -0.0, .greatestFiniteMagnitude, -.greatestFiniteMagnitude]
        var rows: [[Float]] = [[], [.nan], [.infinity, -.infinity], [-.infinity, -.infinity], [1, 1, 1], [.infinity, 2, 2]]
        for _ in 0..<2000 {
            let count = Int(generator.next() % 300) + 1
            var row = (0..<count).map { _ in Float(Int(generator.next() % 64)) / 8 - 4 }  // coarse values: many ties
            for _ in 0..<Int(generator.next() % 4) where generator.next() % 3 == 0 {
                row[Int(generator.next() % UInt64(count))] = specials[Int(generator.next() % UInt64(specials.count))]
            }
            rows.append(row)
        }
        for row in rows {
            var sampler = try LanguageSampler(sampling: .greedy, prompt: [0], vocabularySize: row.count)
            var copy = row
            #expect(result { try sampler.sample(&copy, constraint: .unconstrained) } == result { try originalGreedy(row) }, "\(row)")
            #expect(copy.elementsEqual(row) { $0 == $1 || ($0.isNaN && $1.isNaN) }, "the default path leaves the row alone")
            #expect(sampler.seed == nil)
        }
    }

    /// Upstream Transformers' processors and warpers on the same rows (`Forge/language/reference_sampling.py`).
    @Test func matchesTransformersProcessorsAndWarpers() throws {
        let url = try #require(Bundle.module.url(forResource: "transformers-warpers", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(WarperFixture.self, from: Data(contentsOf: url))
        #expect(fixture.cases.count == 18)
        for entry in fixture.cases {
            let vocabulary = entry.logits.count
            let constraint = entry.allowed.map { LanguageTokenConstraint.allowed(LanguageTokenSet($0, vocabularySize: vocabulary)) }
                ?? .unconstrained
            var greedy = try LanguageSampler(sampling: LanguageSampling(repetitionPenalty: entry.repetitionPenalty),
                                             prompt: entry.history, vocabularySize: vocabulary)
            var row = entry.logits
            #expect(try greedy.sample(&row, constraint: constraint) == entry.greedy, "\(entry.name) greedy")

            let random = LanguageSampling(mode: .random(temperature: entry.temperature, topK: entry.topK, topP: entry.topP, seed: 1),
                                          repetitionPenalty: entry.repetitionPenalty)
            let sampler = try LanguageSampler(sampling: random, prompt: entry.history, vocabularySize: vocabulary)
            row = entry.logits
            try sampler.narrow(&row, constraint: constraint)
            let candidates = try LanguageSampler.distribution(row, temperature: entry.temperature, topK: entry.topK, topP: entry.topP)
            #expect(candidates.ids == entry.kept, "\(entry.name) kept set")
            guard candidates.ids == entry.kept else { continue }
            let worst = zip(candidates.probabilities, entry.probabilities).map { abs($0 - $1) / max($1, 1e-3) }.max() ?? 0
            #expect(worst < 1e-4, "\(entry.name) probabilities: worst relative difference \(worst)")
        }
    }

    @Test func seededDrawsReplayAndFollowTheDistribution() throws {
        let row: [Float] = [2, 1, 0, -1, -.infinity, 0.5]
        let expected = try LanguageSampler.distribution(row, temperature: 1, topK: nil, topP: nil)
        #expect(expected.ids == [0, 1, 2, 3, 5])
        func run(seed: UInt64?, count: Int) throws -> (tokens: [Int], seed: UInt64?) {
            var sampler = try LanguageSampler(sampling: .random(seed: seed), prompt: [0], vocabularySize: row.count)
            return (try (0..<count).map { _ in var copy = row; return try sampler.sample(&copy, constraint: .unconstrained) }, sampler.seed)
        }
        let first = try run(seed: 77, count: 20_000)
        let replay = try run(seed: 77, count: 20_000), other = try run(seed: 78, count: 20_000)
        #expect(first.seed == 77 && first.tokens == replay.tokens && first.tokens != other.tokens)
        for (id, probability) in zip(expected.ids, expected.probabilities) {
            let frequency = Double(first.tokens.filter { $0 == id }.count) / 20_000
            #expect(abs(frequency - probability) < 0.01, "token \(id): \(frequency) vs \(probability)")
        }
        #expect(!first.tokens.contains(4))
        // No seed: one is drawn and reported, and replays the same tokens.
        let drawn = try run(seed: nil, count: 64)
        let seed = try #require(drawn.seed)
        let replayed = try run(seed: seed, count: 64)
        #expect(replayed.tokens == drawn.tokens)
    }

    /// Exactly one draw per random token, even when the constraint leaves one candidate, so replays never drift.
    @Test func oneDrawPerRandomToken() throws {
        let row: [Float] = (0..<32).map { Float($0 % 7) / 3 }
        var constrained = try LanguageSampler(sampling: .random(seed: 3), prompt: [0], vocabularySize: 32)
        var free = try LanguageSampler(sampling: .random(seed: 3), prompt: [0], vocabularySize: 32)
        var copy = row
        #expect(try constrained.sample(&copy, constraint: .allowed(LanguageTokenSet([9], vocabularySize: 32))) == 9)
        copy = row
        _ = try free.sample(&copy, constraint: .unconstrained)
        for _ in 0..<50 {
            var a = row, b = row
            #expect(try constrained.sample(&a, constraint: .unconstrained) == free.sample(&b, constraint: .unconstrained))
        }
    }

    @Test func repetitionPenaltyCountsSampledTokensWithinItsWindow() throws {
        // Token 0 leads; once sampled, its penalty hands the lead to token 1, and so on.
        let row: [Float] = [4, 3.9, 3.8, 1]
        var sampler = try LanguageSampler(sampling: LanguageSampling(repetitionPenalty: 2), prompt: [3], vocabularySize: 4)
        var tokens: [Int] = []
        for _ in 0..<4 { var copy = row; tokens.append(try sampler.sample(&copy, constraint: .unconstrained)) }
        #expect(tokens == [0, 1, 2, 0])
        // A window of one: only the latest token is penalized.
        var windowed = try LanguageSampler(sampling: LanguageSampling(repetitionPenalty: 2, repetitionWindow: 1), prompt: [3],
                                           vocabularySize: 4)
        tokens = []
        for _ in 0..<4 { var copy = row; tokens.append(try windowed.sample(&copy, constraint: .unconstrained)) }
        #expect(tokens == [0, 1, 0, 1])
        // Negative logits are multiplied.
        var negative: [Float] = [-1, -1.5]
        let penalized = try LanguageSampler(sampling: LanguageSampling(repetitionPenalty: 2), prompt: [0], vocabularySize: 2)
        try penalized.narrow(&negative, constraint: .unconstrained)
        #expect(negative == [-2, -1.5])
    }

    @Test func constraintErrors() throws {
        var sampler = try LanguageSampler(sampling: .greedy, prompt: [0], vocabularySize: 4)
        var row: [Float] = [1, 2, -.infinity, 3]
        #expect(throws: LanguageGenerationError.invalid("The constraint allows no token")) {
            try sampler.sample(&row, constraint: .allowed(LanguageTokenSet([2], vocabularySize: 4)))
        }
        row = [1, 2, .nan, 3]
        #expect(throws: LanguageGenerationError.invalid("NaN logits")) {
            try sampler.sample(&row, constraint: .allowed(LanguageTokenSet([0], vocabularySize: 4)))
        }
        row = [1, 2, 0, 3]
        #expect(throws: LanguageGenerationError.invalid("The constraint's vocabulary does not match the model's")) {
            try sampler.sample(&row, constraint: .allowed(LanguageTokenSet([0], vocabularySize: 5)))
        }
        row = [1, 2, 0, 3]
        #expect(try sampler.sample(&row, constraint: .allowed(LanguageTokenSet([0, 2], vocabularySize: 4))) == 0)
        #expect(throws: LanguageGenerationError.self) { try LanguageSampler(sampling: .random(temperature: 0), prompt: [0], vocabularySize: 4) }
    }

    @Test func selectionHelpers() {
        var generator = LanguageRandomGenerator(seed: 11)
        for _ in 0..<300 {
            let values = (0..<Int(generator.next() % 200 + 1)).map { _ in Float(generator.next() % 20) }
            let k = Int(generator.next() % UInt64(values.count)) + 1
            #expect(LanguageSampler.kthLargest(values, k, divisor: 1) == values.sorted(by: >)[k - 1])
            #expect(LanguageSampler.kthLargest(values + [.infinity, -.infinity], k, divisor: 0.5) == values.sorted(by: >)[k - 1] * 2)
            // The nucleus against a sort-based reference with the same tie rule.
            let weights = values.map { exp(Double($0) / 4) }
            let total = weights.reduce(0, +)
            let probabilities = weights.map { $0 / total }
            let mass = Double(generator.next() % 1000 + 1) / 1000
            let order = values.indices.sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1 }
            var reference: [Int] = [], above = 0.0
            for position in order where above < mass { reference.append(position); above += probabilities[position] }
            #expect(LanguageSampler.nucleus(values: values, probabilities: probabilities, mass: mass) == reference.sorted())
        }
    }
}
