import Foundation
import Testing
@testable import AetherRuntime

/// The sampler and RNG ported from 0.0.x (`LanguageSamplerTests`), on its recorded fixtures:
/// - `transformers-warpers.json`: Transformers 4.57.3's own repetition-penalty, prefix-constraint, temperature, top-k
///   and top-p processors on 18 seeded rows (`Forge/language/reference_sampling.py` in 0.0.7);
/// - `random-generator.json`: SplitMix64 and xoshiro256** known answers from the same script.
@Suite("Sampler (0.0.x fixtures)")
struct SamplerTests {
    struct WarperFixture: Decodable {
        struct Case: Decodable {
            let name: String
            let logits: [Float]
            let history: [Int32]
            let repetitionPenalty: Float
            let allowed: [Int]?
            let temperature: Float
            let topK: Int?
            let topP: Double?
            let greedy: Int32
            let kept: [Int]
            let probabilities: [Double]
        }
        let transformers: String
        let cases: [Case]
    }

    static func fixture(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try Data(contentsOf: url)
    }

    /// The same greedy token, the same kept set, and probabilities within a relative 1e-4 on every recorded row.
    @Test func matchesTransformersProcessorsAndWarpers() throws {
        let fixture = try JSONDecoder().decode(WarperFixture.self, from: Self.fixture("transformers-warpers"))
        #expect(fixture.cases.count == 18 && fixture.transformers == "4.57.3")
        for entry in fixture.cases {
            let vocabulary = entry.logits.count
            let constraint = entry.allowed.map { TokenConstraint.allowed(TokenSet($0, vocabularySize: vocabulary)) } ?? .unconstrained
            var greedy = Sampler(SamplingParameters(repetitionPenalty: entry.repetitionPenalty), prompt: entry.history)
            var row = entry.logits
            #expect(try greedy.sample(&row, constraint: constraint) == entry.greedy, "\(entry.name) greedy")

            let random = SamplingParameters(temperature: entry.temperature, topK: entry.topK, topP: entry.topP,
                                            repetitionPenalty: entry.repetitionPenalty, seed: 1)
            row = entry.logits
            try Sampler(random, prompt: entry.history).narrow(&row, constraint: constraint)
            let candidates = try Sampler.distribution(row, temperature: entry.temperature, topK: entry.topK, topP: entry.topP)
            #expect(candidates.ids == entry.kept, "\(entry.name) kept set")
            guard candidates.ids == entry.kept else { continue }
            let worst = zip(candidates.probabilities, entry.probabilities).map { abs($0 - $1) / max($1, 1e-3) }.max() ?? 0
            #expect(worst < 1e-4, "\(entry.name) probabilities: worst relative difference \(worst)")
        }
    }

    @Test func generatorsMatchTheirKnownAnswers() throws {
        struct Fixture: Decodable {
            struct Case: Decodable { let seed: String; let splitmix64: [String]; let xoshiro256starstar: [String] }
            let cases: [Case]
        }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Self.fixture("random-generator"))
        #expect(fixture.cases.count == 5)
        for entry in fixture.cases {
            let seed = try #require(UInt64(entry.seed))
            var split = SplitMix64(seed: seed)
            var xoshiro = SeededRandom(seed: seed)
            #expect(entry.splitmix64.map { UInt64($0)! } == (0..<8).map { _ in split.next() }, "splitmix64 \(seed)")
            #expect(entry.xoshiro256starstar.map { UInt64($0)! } == (0..<8).map { _ in xoshiro.next() }, "xoshiro \(seed)")
        }
    }

    @Test func seededDrawsReplayAndFollowTheDistribution() throws {
        let row: [Float] = [2, 1, 0, -1, -.infinity, 0.5]
        let expected = try Sampler.distribution(row, temperature: 1, topK: nil, topP: nil)
        #expect(expected.ids == [0, 1, 2, 3, 5])
        func run(seed: UInt64, count: Int) throws -> [Int32] {
            var sampler = Sampler(SamplingParameters(temperature: 1, seed: seed))
            return try (0..<count).map { _ in
                var copy = row
                return try sampler.sample(&copy)
            }
        }
        let first = try run(seed: 77, count: 20_000)
        let replay = try run(seed: 77, count: 20_000), other = try run(seed: 78, count: 20_000)
        #expect(first == replay && first != other)
        for (id, probability) in zip(expected.ids, expected.probabilities) {
            let frequency = Double(first.filter { $0 == id }.count) / 20_000
            #expect(abs(frequency - probability) < 0.01, "token \(id): \(frequency) vs \(probability)")
        }
        #expect(!first.contains(4))
    }

    /// Exactly one draw per random token, even when the constraint leaves one candidate, so replays never drift.
    @Test func oneDrawPerRandomToken() throws {
        let row: [Float] = (0..<32).map { Float($0 % 7) / 3 }
        var constrained = Sampler(SamplingParameters(temperature: 1, seed: 3))
        var free = Sampler(SamplingParameters(temperature: 1, seed: 3))
        var copy = row
        #expect(try constrained.sample(&copy, constraint: .allowed(TokenSet([9], vocabularySize: 32))) == 9)
        copy = row
        _ = try free.sample(&copy)
        for _ in 0..<50 {
            var a = row, b = row
            let x = try constrained.sample(&a), y = try free.sample(&b)
            #expect(x == y)
        }
    }

    @Test func repetitionPenaltyCountsChosenTokensWithinItsWindow() throws {
        // Token 0 leads; once chosen, its penalty hands the lead to token 1, and so on.
        let row: [Float] = [4, 3.9, 3.8, 1]
        func run(_ parameters: SamplingParameters) throws -> [Int32] {
            var sampler = Sampler(parameters, prompt: [3])
            return try (0..<4).map { _ in
                var copy = row
                return try sampler.sample(&copy)
            }
        }
        #expect(try run(SamplingParameters(repetitionPenalty: 2)) == [0, 1, 2, 0])
        // A window of one: only the latest token is penalized.
        #expect(try run(SamplingParameters(repetitionPenalty: 2, repetitionWindow: 1)) == [0, 1, 0, 1])
        // Negative logits are multiplied.
        var negative: [Float] = [-1, -1.5]
        try Sampler(SamplingParameters(repetitionPenalty: 2), prompt: [0]).narrow(&negative, constraint: .unconstrained)
        #expect(negative == [-2, -1.5])
    }

    @Test func constraintErrors() throws {
        var sampler = Sampler(.greedy)
        var row: [Float] = [1, 2, -.infinity, 3]
        #expect(throws: AetherError.self) { try sampler.sample(&row, constraint: .allowed(TokenSet([2], vocabularySize: 4))) }
        row = [1, 2, .nan, 3]
        #expect(throws: AetherError.self) { try sampler.sample(&row, constraint: .allowed(TokenSet([0], vocabularySize: 4))) }
        row = [1, 2, 0, 3]
        #expect(throws: AetherError.self) { try sampler.sample(&row, constraint: .allowed(TokenSet([0], vocabularySize: 5))) }
        row = [1, 2, 0, 3]
        #expect(try sampler.sample(&row, constraint: .allowed(TokenSet([0, 2], vocabularySize: 4))) == 0)
        #expect(throws: AetherError.self) { try SamplingParameters(temperature: 1, topP: 0).validate() }
        #expect(throws: AetherError.self) { try SamplingParameters(repetitionPenalty: 0).validate() }
        #expect(throws: AetherError.self) { try SamplingParameters(temperature: .nan).validate() }
        try SamplingParameters(temperature: 0.7, topK: 40, topP: 1, repetitionPenalty: 1.1, repetitionWindow: 64).validate()
    }

    @Test func tokenSetsAndConstraints() {
        let range = TokenSet(range: 60..<130, vocabularySize: 200)
        #expect(range.ids == Array(60..<130) && range.count == 70)
        #expect(TokenSet(range: 0..<500, vocabularySize: 70).count == 70)
        let a = TokenConstraint.allowed(TokenSet([1, 2, 3], vocabularySize: 10))
        #expect(a.intersection(.unconstrained) == a && TokenConstraint.unconstrained.intersection(a) == a)
        #expect(a.intersection(.allowed(TokenSet([3, 4], vocabularySize: 10))) == .allowed(TokenSet([3], vocabularySize: 10)))
    }

    @Test func selectionHelpers() {
        var generator = SeededRandom(seed: 11)
        for _ in 0..<300 {
            let values = (0..<Int(generator.next() % 200 + 1)).map { _ in Float(generator.next() % 20) }
            let k = Int(generator.next() % UInt64(values.count)) + 1
            #expect(Sampler.kthLargest(values, k, divisor: 1) == values.sorted(by: >)[k - 1])
            #expect(Sampler.kthLargest(values + [.infinity, -.infinity], k, divisor: 0.5) == values.sorted(by: >)[k - 1] * 2)
            // The nucleus against a sort-based reference with the same tie rule.
            let weights = values.map { exp(Double($0) / 4) }
            let total = weights.reduce(0, +)
            let probabilities = weights.map { $0 / total }
            let mass = Double(generator.next() % 1000 + 1) / 1000
            let order = values.indices.sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1 }
            var reference: [Int] = [], above = 0.0
            for position in order where above < mass {
                reference.append(position)
                above += probabilities[position]
            }
            #expect(Sampler.nucleus(values: values, probabilities: probabilities, mass: mass) == reference.sorted())
        }
    }

    /// Review #11: a temperature small enough to overflow `score / T` is the greedy limit, not a NaN distribution
    /// and a blind draw (which picked the last candidate).
    @Test func aTinyTemperatureIsTheGreedyLimit() throws {
        let row: [Float] = [3, 1, 2, -1]
        let candidates = try Sampler.distribution(row, temperature: .leastNonzeroMagnitude, topK: nil, topP: nil)
        #expect(candidates.ids == [0] && candidates.probabilities == [1])
        var sampler = Sampler(SamplingParameters(temperature: .leastNonzeroMagnitude, seed: 5))
        for _ in 0..<20 {
            var copy = row
            #expect(try sampler.sample(&copy) == 0)
        }
        let negative = try Sampler.distribution([-3, -1, -2], temperature: .leastNonzeroMagnitude, topK: nil, topP: nil)
        #expect(negative.ids == [1])
    }
}

