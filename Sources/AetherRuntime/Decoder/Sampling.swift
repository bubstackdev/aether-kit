#if canImport(Darwin)
import Darwin
#endif

/// CPU sampling over ``Logits`` (DESIGN.md §6.2), ported from AetherKit 0.0.x and checked against Transformers'
/// own processors (18 recorded rows). Per step, in Transformers' `generate` order:
/// 1. the repetition penalty (over the prompt and the output so far, optionally a recent window);
/// 2. the token constraint (disallowed tokens become −∞, so truncation only ever sees allowed ones);
/// 3. greedy argmax, or temperature → top-k (ties at the k-th value kept) → top-p (exact, no full-vocabulary sort);
/// 4. one seeded draw per sampled token (an inverse CDF in ascending token order), even when one candidate remains.
///
/// A GPU sampler would be an optimization behind the same interface; the greedy GPU argmax of the pipelined decoder
/// is the one that exists, and plain greedy (no penalty, no constraint) never comes here.
public struct SamplingParameters: Sendable, Hashable {
    /// 0 is greedy (argmax; the RNG is not used). Transformers divides the float32 scores by it.
    public var temperature: Float
    /// Keep every token at or above the `topK`-th largest value; nil keeps all.
    public var topK: Int?
    /// Keep the smallest highest-probability set whose mass reaches `topP`; nil (or 1) keeps all.
    public var topP: Double?
    /// HF semantics: each distinct seen token's positive logit is divided by it, a negative one multiplied. 1 is off.
    /// Applies in greedy mode too (then it is not the plain-greedy path).
    public var repetitionPenalty: Float
    /// Only the most recent tokens (prompt included) count for the penalty; nil counts the whole request.
    public var repetitionWindow: Int?
    /// The RNG seed. The same seed, logits and parameters always give the same tokens, on every OS and device.
    public var seed: UInt64

    public init(
        temperature: Float = 0, topK: Int? = nil, topP: Double? = nil, repetitionPenalty: Float = 1,
        repetitionWindow: Int? = nil, seed: UInt64 = 0
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
        self.seed = seed
    }

    public static let greedy = SamplingParameters()

    /// Greedy with no penalty: the case the pipelined GPU argmax can take.
    public var isPlainGreedy: Bool { isGreedy && repetitionPenalty == 1 }

    public var isGreedy: Bool { temperature <= 0 }

    /// Refuses values with no defined meaning before a request starts.
    public func validate() throws(AetherError) {
        guard repetitionPenalty.isFinite, repetitionPenalty > 0, (repetitionWindow ?? 1) > 0 else {
            throw .generation(.invalidParameters("repetition penalty \(repetitionPenalty), window \(repetitionWindow.map(String.init) ?? "nil")"))
        }
        guard temperature.isFinite, (topK ?? 1) > 0, topP.map({ $0 > 0 && $0 <= 1 }) ?? true else {
            throw .generation(.invalidParameters("temperature \(temperature), topK \(topK.map(String.init) ?? "nil"), topP \(topP.map { "\($0)" } ?? "nil")"))
        }
    }
}

/// SplitMix64 (Steele, Lea and Flood, 2014): the seeder of ``SeededRandom``, and on its own the stream the seeded
/// Gaussian noise of the generative pipelines draws from (stream-compatible with 0.0.x).
public struct SplitMix64: RandomNumberGenerator, Sendable, Hashable {
    public private(set) var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// The sampler's RNG: xoshiro256** (Blackman and Vigna, 2018) with its state filled by four ``SplitMix64`` outputs
/// of the 64-bit seed, as its authors recommend, and as 0.0.x seeded it, so a 0.0.x seed draws the same units.
/// Specified here rather than taken from the system, so a seed replays the same draws on every OS and device.
public struct SeededRandom: RandomNumberGenerator, Sendable, Hashable {
    private var s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64

    public init(seed: UInt64) {
        var mix = SplitMix64(seed: seed)
        s0 = mix.next()
        s1 = mix.next()
        s2 = mix.next()
        s3 = mix.next()
    }

    public mutating func next() -> UInt64 {
        let result = Self.rotate(s1 &* 5, 7) &* 9
        let t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = Self.rotate(s3, 45)
        return result
    }

    /// A uniform draw in [0, 1) from the top 53 bits of one output.
    public mutating func nextUnit() -> Double { Double(next() >> 11) * 0x1p-53 }

    private static func rotate(_ x: UInt64, _ k: UInt64) -> UInt64 { (x << k) | (x >> (64 - k)) }
}

/// A set of token ids over one vocabulary, as a bitset (19 KiB for a 151,936-row head).
public struct TokenSet: Sendable, Hashable {
    public let vocabularySize: Int
    private var words: [UInt64]

    /// The empty set.
    public init(vocabularySize: Int) {
        precondition(vocabularySize >= 0)
        self.vocabularySize = vocabularySize
        words = [UInt64](repeating: 0, count: (vocabularySize + 63) / 64)
    }

    /// Ids outside `0..<vocabularySize` are ignored.
    public init(_ ids: some Sequence<Int>, vocabularySize: Int) {
        self.init(vocabularySize: vocabularySize)
        for id in ids { insert(id) }
    }

    /// The ids in `range` (clamped to the vocabulary), for example the tokenizer's own rows of a padded head.
    public init(range: Range<Int>, vocabularySize: Int) {
        self.init(vocabularySize: vocabularySize)
        let range = range.clamped(to: 0..<vocabularySize)
        guard !range.isEmpty else { return }
        for word in (range.lowerBound >> 6)...((range.upperBound - 1) >> 6) {
            let low = max(range.lowerBound - word << 6, 0), high = min(range.upperBound - word << 6, 64)
            words[word] = (high == 64 ? ~0 : (1 << UInt64(high)) - 1) & ~((1 << UInt64(low)) - 1)
        }
    }

    public func contains(_ id: Int) -> Bool {
        id >= 0 && id < vocabularySize && words[id >> 6] & (1 << UInt64(id & 63)) != 0
    }

    public mutating func insert(_ id: Int) {
        guard id >= 0, id < vocabularySize else { return }
        words[id >> 6] |= 1 << UInt64(id & 63)
    }

    public mutating func remove(_ id: Int) {
        guard id >= 0, id < vocabularySize else { return }
        words[id >> 6] &= ~(1 << UInt64(id & 63))
    }

    public mutating func formUnion(_ other: TokenSet) {
        for i in 0..<min(words.count, other.words.count) { words[i] |= other.words[i] }
    }

    public var count: Int { words.reduce(0) { $0 + $1.nonzeroBitCount } }
    public var isEmpty: Bool { words.allSatisfy { $0 == 0 } }

    /// The ids in ascending order.
    public var ids: [Int] {
        var out: [Int] = []
        for (index, word) in words.enumerated() {
            var bits = word
            while bits != 0 {
                out.append(index << 6 + bits.trailingZeroBitCount)
                bits &= bits - 1
            }
        }
        return out
    }

    /// Sets over different vocabularies intersect over the smaller one.
    public func intersection(_ other: TokenSet) -> TokenSet {
        var out = TokenSet(vocabularySize: min(vocabularySize, other.vocabularySize))
        for index in out.words.indices { out.words[index] = words[index] & other.words[index] }
        return out
    }

    /// Sets `logits[i]` to −∞ for every row not in the set. `logits.count` must equal `vocabularySize`.
    func mask(_ logits: inout [Float]) {
        precondition(logits.count == vocabularySize)
        logits.withUnsafeMutableBufferPointer { row in
            for (index, word) in words.enumerated() where word != ~0 {
                let base = index << 6
                for bit in 0..<min(64, vocabularySize - base) where word & (1 << UInt64(bit)) == 0 {
                    row[base + bit] = -.infinity
                }
            }
        }
    }
}

/// Which tokens may be chosen next.
public enum TokenConstraint: Sendable, Hashable {
    case unconstrained
    /// Only these tokens may be chosen; every other logit becomes −∞ before temperature and truncation.
    case allowed(TokenSet)

    /// Both constraints at once (for example a grammar within the tokenizer's own rows of a padded head).
    public func intersection(_ other: TokenConstraint) -> TokenConstraint {
        switch (self, other) {
        case (.unconstrained, _): other
        case (_, .unconstrained): self
        case (.allowed(let a), .allowed(let b)): .allowed(a.intersection(b))
        }
    }
}

/// The hook constrained decoding plugs into (DESIGN.md §7): for every chosen token, which tokens may follow (a
/// grammar or a JSON-schema matcher). The constraint on token t+1 depends only on tokens up to t, all known once t is
/// fed, so the Generator runs `accept` concurrently with the step that computes t+1's logits and applies its result
/// when both are done; `begin` runs concurrently with the prompt's prefill. Work up to the step's duration is hidden.
///
/// A processor holds one request's state, and `begin` resets it; the Generator serves one request at a time. Any
/// processor forces the host-logits path (depth 1). A throwing call fails the request.
public protocol LogitsProcessor: Sendable {
    /// The constraint on the first chosen token.
    func begin(prompt: [Int32]) async throws -> TokenConstraint
    /// Records `token` (just chosen, now being fed) and returns the constraint on the token after it. Not called for
    /// a stop token or for the last token at the output limit.
    func accept(_ token: Int32) async throws -> TokenConstraint
    /// The constrained text is a value (structured output), not prose: the Generator parses no tags in it after a
    /// leading thinking block, so a string value containing a tag's text stays in the value. Default false.
    var isStructuredOutput: Bool { get }
}

extension LogitsProcessor {
    public var isStructuredOutput: Bool { false }
}

/// Picks tokens from logits rows with ``SamplingParameters``. Holds one request's RNG and repetition history.
public struct Sampler: Sendable {
    public let parameters: SamplingParameters
    private var rng: SeededRandom
    /// Prompt and chosen tokens, kept only while the repetition penalty is on.
    private var history: [Int32]

    public init(_ parameters: SamplingParameters, prompt: [Int32] = []) {
        self.parameters = parameters
        rng = SeededRandom(seed: parameters.seed)
        history = parameters.repetitionPenalty == 1 ? [] : prompt
    }

    /// The next token from `logits` (one row, which this may modify) under `constraint`. Records the token for the
    /// repetition penalty.
    public mutating func sample(_ logits: inout [Float], constraint: TokenConstraint = .unconstrained) throws(AetherError) -> Int32 {
        try narrow(&logits, constraint: constraint)
        let token: Int32
        if parameters.isGreedy {
            token = try Self.greedy(logits)
        } else if let forced = logits.firstIndex(of: .infinity) {
            // +inf forces the token (the lowest such index): its probability is 1, and exp(inf − inf) would be NaN.
            // The draw is still consumed, so a replay never drifts.
            _ = rng.next()
            token = Int32(forced)
        } else {
            let candidates = try Self.distribution(logits, temperature: parameters.temperature, topK: parameters.topK, topP: parameters.topP)
            token = Int32(Self.draw(candidates, unit: rng.nextUnit()))
        }
        if parameters.repetitionPenalty != 1 { history.append(token) }
        return token
    }

    /// The repetition penalty, then the constraint (Transformers' processor order).
    func narrow(_ logits: inout [Float], constraint: TokenConstraint) throws(AetherError) {
        if case .allowed(let allowed) = constraint {
            guard allowed.vocabularySize == logits.count else {
                throw .generation(.constraintExhausted("a \(allowed.vocabularySize)-token constraint on a \(logits.count)-row head"))
            }
            // Masking would hide a NaN the unconstrained path reports.
            guard !logits.contains(where: \.isNaN) else { throw .generation(.invalidLogits("NaN")) }
        }
        if parameters.repetitionPenalty != 1 {
            let penalty = parameters.repetitionPenalty
            let recent = parameters.repetitionWindow.map { history.suffix($0) } ?? history[...]
            var seen = Set<Int32>(minimumCapacity: recent.count)
            for token in recent where token >= 0 && Int(token) < logits.count && seen.insert(token).inserted {
                let v = logits[Int(token)]
                logits[Int(token)] = v < 0 ? v * penalty : v / penalty
            }
        }
        if case .allowed(let allowed) = constraint {
            allowed.mask(&logits)
            guard logits.contains(where: { $0 > -.infinity }) else { throw .generation(.constraintExhausted("every logit is masked")) }
        }
    }

    /// The first index of the largest value (+inf ranks first, −inf is never chosen); NaN throws.
    static func greedy(_ logits: [Float]) throws(AetherError) -> Int32 {
        var best = -Float.infinity
        var token: Int?
        for (i, score) in logits.enumerated() {
            guard !score.isNaN else { throw .generation(.invalidLogits("NaN")) }
            if score > best {
                best = score
                token = i
            }
        }
        guard let token else { throw .generation(.invalidLogits("no finite logit")) }
        return Int32(token)
    }

    /// The tokens a random draw can pick and their probabilities (ascending by id, summing to 1): finite logits
    /// divided by `temperature`, every token at or above the k-th largest (ties at the boundary kept, as Transformers
    /// does), then the smallest set in descending probability whose preceding mass stays below `topP` (ties in
    /// ascending id order). Linear on average; nothing sorts the vocabulary.
    static func distribution(_ logits: [Float], temperature: Float, topK: Int?, topP: Double?) throws(AetherError)
        -> (ids: [Int], probabilities: [Double])
    {
        // Transformers divides the float32 scores by the temperature before truncating; the quotients round alike here.
        let divisor = temperature
        var threshold = -Float.infinity
        var finite = 0
        for score in logits {
            guard !score.isNaN else { throw .generation(.invalidLogits("NaN")) }
            if score.isFinite { finite += 1 }
        }
        if let topK, topK < finite { threshold = kthLargest(logits, topK, divisor: divisor) }
        var ids: [Int] = [], values: [Float] = []
        ids.reserveCapacity(threshold == -.infinity ? finite : 2 * (topK ?? 1))
        values.reserveCapacity(ids.capacity)
        logits.withUnsafeBufferPointer { row in
            for id in row.indices {
                let score = row[id]
                guard score.isFinite else { continue }
                let value = score / divisor
                if value >= threshold {
                    ids.append(id)
                    values.append(value)
                }
            }
        }
        guard !ids.isEmpty else { throw .generation(.invalidLogits("no finite logit")) }
        // Float32 weights, as Transformers' softmax; normalized in double precision.
        var maximum = -Float.infinity
        for value in values { maximum = max(maximum, value) }
        // A temperature small enough to overflow the quotients is the greedy limit: the largest logits share the mass
        // (exp(inf − inf) would be NaN and the draw blind).
        if !maximum.isFinite {
            let best = ids.map { logits[$0] }.max()!
            let top = ids.filter { logits[$0] == best }
            return (top, [Double](repeating: 1 / Double(top.count), count: top.count))
        }
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
            let sum = probabilities.reduce(0, +)
            for i in probabilities.indices { probabilities[i] /= sum }
        }
        return (ids, probabilities)
    }

    /// Inverse CDF over the candidates in ascending id order: the first whose cumulative probability exceeds `unit`.
    static func draw(_ candidates: (ids: [Int], probabilities: [Double]), unit: Double) -> Int {
        var cumulative = 0.0
        for (id, probability) in zip(candidates.ids, candidates.probabilities) {
            cumulative += probability
            if unit < cumulative { return id }
        }
        return candidates.ids.last!
    }

    /// The k-th largest finite logit divided by `divisor` (1 ≤ k ≤ the finite count), by one selection scan that keeps
    /// at most 2k values: whatever is not above the current k-th largest can't change it.
    ///
    /// Selects on the raw logits and divides only the result (dividing by a positive temperature keeps the order): a
    /// tiny temperature turns negative quotients into `-inf`, which the scan used to skip, keeping fewer than `k`
    /// values and then indexing past them.
    static func kthLargest(_ logits: [Float], _ k: Int, divisor: Float) -> Float {
        var kept: [Float] = []
        kept.reserveCapacity(2 * k)
        var floor = -Float.infinity
        for score in logits where score.isFinite && (kept.count < k || score > floor) {
            kept.append(score)
            if kept.count == 2 * k {
                kept.sort(by: >)
                kept.removeLast(k)
                floor = kept[k - 1]
            }
        }
        kept.sort(by: >)
        return kept[min(k, kept.count) - 1] / divisor
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
            if greaterMass >= budget {
                high = greater
                continue
            }
            for position in order[low..<greater] { kept[position] = true }
            budget -= greaterMass
            // Tied at the pivot: ascending position (= token id) order, while the preceding mass stays below `mass`.
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
