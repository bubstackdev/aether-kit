import Foundation

/// Chooses each token of one request from its steps' logits (``LanguageSampling``). Holds the request's generator and
/// repetition history, so it lives in the decode loop, never in a session: model reloads and recycling never meet it.
///
/// Plain greedy with no constraint calls ``greedy(_:)`` on the row and nothing else: the default path stays
/// token-identical to the runtime before sampling existed.
package struct LanguageSampler {
    package let sampling: LanguageSampling
    package let vocabularySize: Int
    /// The seed random sampling uses (the request's, or one drawn from the system); nil when greedy.
    package let seed: UInt64?
    private var generator: LanguageRandomGenerator?
    /// Prompt and sampled tokens, kept only while the repetition penalty is on.
    private var history: [Int]

    package init(sampling: LanguageSampling, prompt: [Int], vocabularySize: Int) throws {
        try sampling.validate()
        self.sampling = sampling
        self.vocabularySize = vocabularySize
        if case .random(_, _, _, let requested) = sampling.mode {
            let seed = requested ?? UInt64.random(in: .min ... .max)
            self.seed = seed
            generator = LanguageRandomGenerator(seed: seed)
        } else {
            seed = nil
            generator = nil
        }
        history = sampling.repetitionPenalty == 1 ? [] : prompt
    }

    /// The next token from `logits` (one step's last row, which this may modify), under `constraint`.
    package mutating func sample(_ logits: inout [Float], constraint: LanguageTokenConstraint) throws -> Int {
        if sampling.isPlainGreedy, case .unconstrained = constraint { return try Self.greedy(logits) }
        guard logits.count == vocabularySize else { throw LanguageGenerationError.invalid("Unexpected logits row") }
        try narrow(&logits, constraint: constraint)
        let token: Int
        switch sampling.mode {
        case .greedy:
            token = try Self.greedy(logits)
        case .random(let temperature, let topK, let topP, _):
            let candidates = try Self.distribution(logits, temperature: temperature, topK: topK, topP: topP)
            token = Self.draw(candidates, unit: generator!.nextUnitInterval())
        }
        if sampling.repetitionPenalty != 1 { history.append(token) }
        return token
    }

    /// The repetition penalty, then the constraint (Transformers' processor order).
    func narrow(_ logits: inout [Float], constraint: LanguageTokenConstraint) throws {
        if case .allowed(let allowed) = constraint {
            guard allowed.vocabularySize == vocabularySize
            else { throw LanguageGenerationError.invalid("The constraint's vocabulary does not match the model's") }
            // Masking would hide a NaN the unconstrained path reports.
            guard !logits.contains(where: \.isNaN) else { throw LanguageGenerationError.invalid("NaN logits") }
        }
        if sampling.repetitionPenalty != 1 {
            let penalty = Float(sampling.repetitionPenalty)
            let recent = sampling.repetitionWindow.map { history.suffix($0) } ?? history[...]
            var seen = Set<Int>(minimumCapacity: recent.count)
            for token in recent where token >= 0 && token < vocabularySize && seen.insert(token).inserted {
                logits[token] = logits[token] < 0 ? logits[token] * penalty : logits[token] / penalty
            }
        }
        if case .allowed(let allowed) = constraint {
            allowed.mask(&logits)
            guard logits.contains(where: \.isFinite)
            else { throw LanguageGenerationError.invalid("The constraint allows no token") }
        }
    }

    /// The first index of the largest finite logit; infinities are skipped and NaN throws.
    package static func greedy(_ logits: [Float]) throws -> Int {
        var best = -Float.infinity
        var token: Int?
        for (i, score) in logits.enumerated() {
            guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
            if score.isFinite && score > best { best = score; token = i }
        }
        guard let token else { throw LanguageGenerationError.invalid("No finite logits") }
        return token
    }

    /// The tokens a random draw can pick and their probabilities (ascending by ID, summing to 1): finite logits
    /// divided by `temperature`, every token at or above the k-th largest (ties at the boundary kept, as Transformers
    /// does), then the smallest set in descending probability whose preceding mass stays below `topP` (ties in
    /// ascending ID order). Selection is linear on average; nothing sorts the vocabulary.
    static func distribution(_ logits: [Float], temperature: Double, topK: Int?, topP: Double?) throws
        -> (ids: [Int], probabilities: [Double]) {
        var ids: [Int] = []
        var values: [Double] = []
        ids.reserveCapacity(logits.count)
        values.reserveCapacity(logits.count)
        for (i, score) in logits.enumerated() {
            guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
            // Transformers divides float32 scores; the quotient is rounded the same way here.
            if score.isFinite { ids.append(i); values.append(Double(score / Float(temperature))) }
        }
        guard !ids.isEmpty else { throw LanguageGenerationError.invalid("No finite logits") }
        if let topK, topK < values.count {
            let threshold = kthLargest(values, topK)
            let kept = values.indices.filter { values[$0] >= threshold }
            ids = kept.map { ids[$0] }
            values = kept.map { values[$0] }
        }
        let maximum = values.max()!
        var probabilities = values.map { exp($0 - maximum) }
        normalize(&probabilities)
        if let topP, topP < 1 {
            let kept = nucleus(values: values, probabilities: probabilities, mass: topP)
            ids = kept.map { ids[$0] }
            probabilities = kept.map { probabilities[$0] }
            normalize(&probabilities)
        }
        return (ids, probabilities)
    }

    /// Inverse CDF over the candidates in ascending ID order: the first whose cumulative probability exceeds `unit`.
    static func draw(_ candidates: (ids: [Int], probabilities: [Double]), unit: Double) -> Int {
        var cumulative = 0.0
        for (id, probability) in zip(candidates.ids, candidates.probabilities) {
            cumulative += probability
            if unit < cumulative { return id }
        }
        return candidates.ids.last!
    }

    private static func normalize(_ probabilities: inout [Double]) {
        let total = probabilities.reduce(0, +)
        for i in probabilities.indices { probabilities[i] /= total }
    }

    /// The k-th largest value (k ≥ 1), by quickselect.
    static func kthLargest(_ values: [Double], _ k: Int) -> Double {
        precondition(k >= 1 && k <= values.count)
        var pool = values
        var rank = k
        while true {
            let pivot = median(pool[0], pool[pool.count / 2], pool[pool.count - 1])
            let greater = pool.filter { $0 > pivot }
            if rank <= greater.count { pool = greater; continue }
            let equal = pool.reduce(0) { $0 + ($1 == pivot ? 1 : 0) }
            if rank <= greater.count + equal { return pivot }
            rank -= greater.count + equal
            pool = pool.filter { $0 < pivot }
        }
    }

    /// Positions (ascending) of the nucleus: in descending value order, ties by ascending position, every candidate
    /// whose preceding mass is below `mass`. Selection by partitioning on value, so expected linear time.
    static func nucleus(values: [Double], probabilities: [Double], mass: Double) -> [Int] {
        var kept: [Int] = []
        var budget = mass  // `mass` minus everything kept so far, all of which ranks above the remaining pool
        var pool = Array(values.indices)
        while !pool.isEmpty {
            let pivot = median(values[pool[0]], values[pool[pool.count / 2]], values[pool[pool.count - 1]])
            let greater = pool.filter { values[$0] > pivot }
            let greaterMass = greater.reduce(0.0) { $0 + probabilities[$1] }
            if greaterMass >= budget { pool = greater; continue }
            kept += greater
            budget -= greaterMass
            // Tied at the pivot: ascending position (= token ID) order, while the preceding mass stays below `mass`.
            for position in pool where values[position] == pivot {
                guard budget > 0 else { return kept.sorted() }
                kept.append(position)
                budget -= probabilities[position]
            }
            guard budget > 0 else { break }
            pool = pool.filter { values[$0] < pivot }
        }
        return kept.sorted()
    }

    private static func median(_ a: Double, _ b: Double, _ c: Double) -> Double {
        max(min(a, b), min(max(a, b), c))
    }
}
