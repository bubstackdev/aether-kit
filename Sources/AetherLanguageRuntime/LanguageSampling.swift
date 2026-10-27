/// How each token is chosen from a step's logits. The order per step follows Transformers' `generate`: repetition
/// penalty, the request's constraint, then (random mode) temperature, top-k, top-p, normalization and one draw.
///
/// Greedy with no repetition penalty and no constraint is today's argmax, unchanged.
public struct LanguageSampling: Sendable, Hashable {
    public enum Mode: Sendable, Hashable {
        /// The first index of the largest finite logit.
        case greedy
        /// A seeded draw. `temperature` > 0; `topK` keeps every token tied at the k-th logit; `topP` keeps the
        /// smallest highest-probability set whose mass reaches it. A nil seed is drawn from the system and reported in
        /// the result, so any reply can be replayed.
        case random(temperature: Double, topK: Int?, topP: Double?, seed: UInt64?)
    }

    public var mode: Mode
    /// Transformers' rule: each distinct token already in the request (prompt included) has a positive logit divided
    /// by the penalty and a negative one multiplied by it. 1: off. Applies in greedy mode too.
    public var repetitionPenalty: Double
    /// Only the most recent tokens (prompt included) count for the penalty. nil: the whole request.
    public var repetitionWindow: Int?

    public init(mode: Mode = .greedy, repetitionPenalty: Double = 1, repetitionWindow: Int? = nil) {
        self.mode = mode
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
    }

    public static let greedy = LanguageSampling()

    public static func random(temperature: Double = 1, topK: Int? = nil, topP: Double? = nil,
                              seed: UInt64? = nil) -> LanguageSampling {
        LanguageSampling(mode: .random(temperature: temperature, topK: topK, topP: topP, seed: seed))
    }

    /// Greedy with no repetition penalty: the default path, which must stay token-identical to plain argmax.
    var isPlainGreedy: Bool { mode == .greedy && repetitionPenalty == 1 }

    /// Refuses values with no defined meaning before a request starts.
    public func validate() throws {
        guard repetitionPenalty.isFinite, repetitionPenalty > 0, (repetitionWindow ?? 1) > 0
        else { throw LanguageGenerationError.invalid("Invalid repetition penalty") }
        if case .random(let temperature, let topK, let topP, _) = mode {
            guard temperature.isFinite, temperature > 0, (topK ?? 1) > 0, topP.map({ $0 > 0 && $0 <= 1 }) ?? true
            else { throw LanguageGenerationError.invalid("Invalid sampling parameters") }
        }
    }
}
