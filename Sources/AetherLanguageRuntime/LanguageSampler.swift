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
        // Transformers divides the float32 scores by the temperature before truncating; the quotients round alike here.
        let divisor = Float(temperature)
        var threshold = -Float.infinity
        if let topK {
            var finite = 0
            for score in logits {
                guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
                if score.isFinite { finite += 1 }
            }
            if topK < finite { threshold = kthLargest(logits, topK, divisor: divisor) }
        }
        var ids: [Int] = [], values: [Float] = []
        ids.reserveCapacity(threshold == -.infinity ? logits.count : 2 * topK!)
        values.reserveCapacity(ids.capacity)
        try logits.withUnsafeBufferPointer { row in
            for id in row.indices {
                let score = row[id]
                guard !score.isNaN else { throw LanguageGenerationError.invalid("NaN logits") }
                guard score.isFinite else { continue }
                let value = score / divisor
                if value >= threshold {
                    ids.append(id)
                    values.append(value)
                }
            }
        }
        guard !ids.isEmpty else { throw LanguageGenerationError.invalid("No finite logits") }
        // Float32 weights, as Transformers' softmax; normalized in double precision.
        var maximum = -Float.infinity
        for value in values { maximum = max(maximum, value) }
        var probabilities = [Double](repeating: 0, count: values.count)
        var total = 0.0
        values.withUnsafeBufferPointer { values in
            probabilities.withUnsafeMutableBufferPointer { probabilities in
                for i in values.indices {
                    let weight = Double(exp(values[i] - maximum))
                    probabilities[i] = weight
                    total += weight
                }
                for i in probabilities.indices { probabilities[i] /= total }
            }
        }
        if let topP, topP < 1 {
            // Exact pruning: candidates at or below (1 - p) / n hold at most 1 - p together and rank below every other
            // candidate, so the others' mass reaches p before any of them could be kept.
            let floor = (1 - topP) / Double(probabilities.count)
            let pool = probabilities.indices.filter { probabilities[$0] > floor }
            let kept = nucleus(values: pool.map { values[$0] }, probabilities: pool.map { probabilities[$0] }, mass: topP)
            ids = kept.map { ids[pool[$0]] }
            probabilities = kept.map { probabilities[pool[$0]] }
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

    /// The k-th largest finite logit divided by `divisor` (1 ≤ k ≤ the finite count), by one selection scan that keeps
    /// at most 2k values: whatever is not above the current k-th largest can't change it.
    static func kthLargest(_ logits: [Float], _ k: Int, divisor: Float) -> Float {
        var kept: [Float] = []
        kept.reserveCapacity(2 * k)
        var floor = -Float.infinity
        for score in logits where score.isFinite && score / divisor > floor {
            kept.append(score / divisor)
            if kept.count == 2 * k {
                kept.sort(by: >)
                kept.removeLast(k)
                floor = kept[k - 1]
            }
        }
        kept.sort(by: >)
        return kept[k - 1]
    }

    /// Positions (ascending) of the nucleus: in descending value order, ties by ascending position, every candidate
    /// whose preceding mass is below `mass`. In-place three-way partitioning on value, so expected linear time.
    static func nucleus(values: [Float], probabilities: [Double], mass: Double) -> [Int] {
        var order = Array(values.indices)
        var kept = [Bool](repeating: false, count: values.count)
        var budget = mass  // `mass` minus everything kept so far, all of which ranks above the remaining pool
        var low = 0, high = order.count  // the pool: order[low..<high]
        while low < high {
            let pivot = median(values[order[low]], values[order[(low + high) / 2]], values[order[high - 1]])
            // order[low..<greater] > pivot, order[greater..<less] == pivot, order[less..<high] < pivot.
            var greater = low, index = low, less = high
            while index < less {
                let value = values[order[index]]
                if value > pivot {
                    order.swapAt(greater, index)
                    greater += 1
                    index += 1
                } else if value < pivot {
                    less -= 1
                    order.swapAt(index, less)
                } else {
                    index += 1
                }
            }
            var greaterMass = 0.0
            for position in order[low..<greater] { greaterMass += probabilities[position] }
            if greaterMass >= budget { high = greater; continue }
            for position in order[low..<greater] { kept[position] = true }
            budget -= greaterMass
            // Tied at the pivot: ascending position (= token ID) order, while the preceding mass stays below `mass`.
            for position in order[greater..<less].sorted() {
                guard budget > 0 else { break }
                kept[position] = true
                budget -= probabilities[position]
            }
            guard budget > 0 else { break }
            low = less
        }
        return kept.indices.filter { kept[$0] }
    }

    private static func median(_ a: Float, _ b: Float, _ c: Float) -> Float {
        max(min(a, b), min(max(a, b), c))
    }
}
