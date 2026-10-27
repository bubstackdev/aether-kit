/// A token-level language model runtime the `AetherLanguage` facade can drive without knowing how it runs: token IDs in,
/// token IDs out. Tokenization, chat templates, reasoning and text belong to the facade. No backend type appears in
/// this surface, so the protocol builds below the Core AI floor (`LanguageGenerationSession` is the Core AI backend).
public protocol LanguageBackend: Sendable {
    /// Rows of the logits head. Prompt and sampled token IDs are in `0..<vocabularySize`.
    var vocabularySize: Int { get }
    /// The registered context: prompt plus output.
    var maximumContextLength: Int { get }
    /// The registration's end-of-sequence IDs (`stopTokenIDs` in the manifest).
    var stopTokenIDs: Set<Int> { get }

    /// One request. Each emitted token goes to `onToken` while the following step runs; at most one delivery is in
    /// flight, and every delivery finishes before this returns or throws. Returning `.stop` ends the request after the
    /// overlapping step joins (its result is discarded). Cancelling the calling task stops at the next step, and
    /// submitted native work joins before the error is thrown. One request at a time: overlap throws `busy`.
    func generate(_ request: LanguageGenerationRequest,
                  onToken: (@Sendable (LanguageGeneratedToken) async throws -> LanguageTokenDisposition)?) async throws
        -> LanguageGenerationResult

    /// Terminal close: an active request finishes first, and later requests throw `closed`.
    func unload() async
}

public struct LanguageGenerationRequest: Sendable {
    public var prompt: [Int]
    public var maximumTokens: Int
    public var sampling: LanguageSampling
    /// Narrows every sampled token (structured output). It runs beside the decode steps (``LanguageLogitsProcessor``).
    public var constraint: (any LanguageLogitsProcessor)?

    public init(prompt: [Int], maximumTokens: Int, sampling: LanguageSampling = .greedy,
                constraint: (any LanguageLogitsProcessor)? = nil) {
        self.prompt = prompt
        self.maximumTokens = maximumTokens
        self.sampling = sampling
        self.constraint = constraint
    }
}

/// One emitted token, as delivered while the next step runs. A struct, so per-token data (for example
/// log-probabilities) can be added without changing the callback's type.
public struct LanguageGeneratedToken: Sendable, Hashable {
    public let id: Int
    /// Its index in the reply.
    public let index: Int

    public init(id: Int, index: Int) {
        self.id = id
        self.index = index
    }
}

/// What the consumer wants after a delivered token: stop strings and enough-is-enough end a request this way.
public enum LanguageTokenDisposition: Sendable, Hashable {
    case `continue`, stop
}

public enum LanguageStopReason: String, Sendable, Hashable {
    /// A registered end-of-sequence token was sampled (it is not part of the reply).
    case endOfSequence = "eos"
    case outputLimit = "output-limit"
    /// The consumer returned ``LanguageTokenDisposition/stop``.
    case stopped
}

public struct LanguageGenerationResult: Sendable {
    public let tokens: [Int]
    public let stopReason: LanguageStopReason
    /// The seed the request's random sampling used (drawn from the system when the request gave none); nil when greedy.
    public let seed: UInt64?
    public let timeToFirstTokenSeconds: Double
    public let totalSeconds: Double

    public init(tokens: [Int], stopReason: LanguageStopReason, seed: UInt64?, timeToFirstTokenSeconds: Double, totalSeconds: Double) {
        self.tokens = tokens
        self.stopReason = stopReason
        self.seed = seed
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.totalSeconds = totalSeconds
    }
}

/// Qualification entry points a backend can adopt so the same parity harness runs against every backend. Not part of
/// the app-facing surface.
@_spi(Qualification)
public protocol LanguageBackendQualification: LanguageBackend {
    /// Last-row logits after the prompt's prefill, then after each continuation token, in a fresh request.
    func teacherForcedLogits(prompt: [Int], continuation: [Int]) async throws -> [[Float]]
}

public enum LanguageGenerationError: Error, Sendable, Equatable {
    case invalid(String), busy, closed, contextLimit(limit: Int)
    case memoryBudget(requiredBytes: UInt64, availableBytes: UInt64)
}
