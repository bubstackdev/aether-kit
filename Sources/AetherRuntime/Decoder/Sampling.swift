/// CPU sampling over ``Logits`` (DESIGN.md §6.2): greedy, temperature, top-k, top-p and repetition penalty, with a
/// seeded RNG. A GPU sampler would be an optimization behind the same interface; the greedy GPU argmax of the
/// pipelined decoder is the one that exists.
public struct SamplingParameters: Sendable, Hashable {
    /// 0 is greedy (argmax; the RNG is not used).
    public var temperature: Float
    /// Keep the `topK` most likely tokens; nil keeps all.
    public var topK: Int?
    /// Keep the smallest set whose probability mass reaches `topP`; nil (or ≥ 1) keeps all.
    public var topP: Float?
    /// HF semantics: a seen token's positive logit is divided by it, a negative one multiplied. 1 is off.
    public var repetitionPenalty: Float
    /// The RNG seed. The same seed, logits and parameters always give the same tokens.
    public var seed: UInt64

    public init(temperature: Float = 0, topK: Int? = nil, topP: Float? = nil, repetitionPenalty: Float = 1, seed: UInt64 = 0) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    public static let greedy = SamplingParameters()

    /// Greedy with no penalty: the case the pipelined GPU argmax can take.
    public var isPlainGreedy: Bool { temperature == 0 && repetitionPenalty == 1 }
}

/// The one RNG (DESIGN.md §1.2): SplitMix64. Small, fast, and stable across platforms and releases, so a seed
/// reproduces a sample everywhere.
public struct SeededRandom: RandomNumberGenerator, Sendable, Hashable {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A uniform value in [0, 1) with 24 bits of precision.
    public mutating func nextUnit() -> Float {
        Float(next() >> 40) * (1.0 / Float(1 << 24))
    }
}

/// What a logits processor sees: the prompt and the tokens generated so far.
public struct LogitsContext: Sendable {
    public var prompt: [Int32]
    public var generated: [Int32]

    public init(prompt: [Int32], generated: [Int32]) {
        self.prompt = prompt
        self.generated = generated
    }
}

/// A hook that rewrites the next-token logits before sampling (DESIGN.md §7): constrained decoding (M3) is a
/// processor that masks the tokens its grammar forbids. Applied by the Generator in order, on the CPU, after the
/// repetition penalty and before temperature. Any processor forces the host-logits path (depth 1).
public protocol LogitsProcessor: Sendable {
    func process(_ logits: inout [Float], context: LogitsContext)
}

/// Picks tokens from logits rows with ``SamplingParameters`` and its own RNG. Non-`Sendable` state (the RNG)
/// belongs to one generation.
public struct Sampler {
    public let parameters: SamplingParameters
    private var rng: SeededRandom
    private var scratch: [(index: Int32, value: Float)] = []

    public init(_ parameters: SamplingParameters) {
        self.parameters = parameters
        rng = SeededRandom(seed: parameters.seed)
    }

    /// Applies the repetition penalty over `seen` (prompt and generated tokens, as HF does) in place.
    public static func applyRepetitionPenalty(_ penalty: Float, seen: some Sequence<Int32>, to logits: inout [Float]) {
        guard penalty != 1 else { return }
        var done = Set<Int32>()
        for token in seen where token >= 0 && Int(token) < logits.count && done.insert(token).inserted {
            let v = logits[Int(token)]
            logits[Int(token)] = v > 0 ? v / penalty : v * penalty
        }
    }

    /// Samples one token from `logits` (already penalized and processed).
    public mutating func sample(_ logits: [Float]) -> Int32 {
        if parameters.temperature <= 0 { return Logits.argmax(logits[...]) }
        // +inf forces the token (the lowest such index): its probability is 1, and exp(inf − inf) would be NaN.
        if let forced = logits.firstIndex(of: .infinity) { return Int32(forced) }
        let inverse = 1 / parameters.temperature
        scratch.removeAll(keepingCapacity: true)
        scratch.reserveCapacity(logits.count)
        for (i, v) in logits.enumerated() where v.isFinite {
            scratch.append((Int32(i), v * inverse))
        }
        guard !scratch.isEmpty else { return Logits.argmax(logits[...]) }
        // Descending by value, ties by index: a total order, so top-k/top-p cut deterministically.
        scratch.sort { $0.value != $1.value ? $0.value > $1.value : $0.index < $1.index }
        if let k = parameters.topK, k > 0, k < scratch.count { scratch.removeSubrange(k...) }
        let maxValue = scratch[0].value
        var probs = scratch.map { exp($0.value - maxValue) }
        var total = probs.reduce(0, +)
        if let p = parameters.topP, p > 0, p < 1 {
            var mass: Float = 0
            var keep = probs.count
            for (i, q) in probs.enumerated() {
                mass += q / total
                if mass >= p {
                    keep = i + 1
                    break
                }
            }
            probs.removeSubrange(keep...)
            total = probs.reduce(0, +)
        }
        let target = rng.nextUnit() * total
        var running: Float = 0
        for (i, q) in probs.enumerated() {
            running += q
            if target < running { return scratch[i].index }
        }
        return scratch[probs.count - 1].index
    }
}

#if canImport(Darwin)
import Darwin
#endif
