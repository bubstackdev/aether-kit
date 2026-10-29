import AetherLanguageRuntime
import AetherLanguageTokenizers

/// Per-request options. The names follow FoundationModels' `GenerationOptions` where the meaning is the same
/// (`samplingMode`, `temperature`, `maximumResponseTokens`); the rest are what a local model adds.
public struct AetherGenerationOptions: Sendable, Equatable {
    public struct SamplingMode: Sendable, Equatable {
        @nonexhaustive public enum Kind: Sendable, Equatable {
            case greedy
            case randomTopK(Int, seed: UInt64?)
            case randomProbabilityThreshold(Double, seed: UInt64?)
        }
        public let kind: Kind

        /// The most likely token every step: the same reply every time.
        public static let greedy = SamplingMode(kind: .greedy)
        /// A seeded draw among the `k` most likely tokens.
        public static func random(top k: Int, seed: UInt64? = nil) -> SamplingMode { SamplingMode(kind: .randomTopK(k, seed: seed)) }
        /// A seeded draw among the most likely tokens whose probabilities add up to `probabilityThreshold`.
        public static func random(probabilityThreshold: Double, seed: UInt64? = nil) -> SamplingMode {
            SamplingMode(kind: .randomProbabilityThreshold(probabilityThreshold, seed: seed))
        }
    }

    @nonexhaustive public enum Reasoning: Sendable, Equatable {
        /// Render with the template's thinking switch off; the model answers directly.
        case off
        /// Let the model think first. Its thinking is returned separately (`Response.reasoning`) and is not rendered back
        /// into later turns. Only for models that declare it (``AetherLanguageModel/supportsReasoning``).
        case on
    }

    /// nil: greedy. Apps opt into sampling per request.
    public var samplingMode: SamplingMode?
    /// Divides the logits before a random draw; ignored when greedy, as FoundationModels ignores it. nil: 1.
    public var temperature: Double?
    /// nil: the context left after the prompt, capped at ``defaultMaximumResponseTokens``.
    public var maximumResponseTokens: Int?
    /// The reply ends before the first of these that appears in its text (not in its reasoning).
    public var stopSequences: [String]
    /// Transformers' repetition penalty over the whole request, prompt included; nil or 1: off.
    public var repetitionPenalty: Double?
    /// nil: the bundle's default, which is off for every registration today.
    public var reasoning: Reasoning?

    /// The output limit when a request names none (and the context allows it).
    public static let defaultMaximumResponseTokens = 4096

    public init(samplingMode: SamplingMode? = nil, temperature: Double? = nil, maximumResponseTokens: Int? = nil,
                stopSequences: [String] = [], repetitionPenalty: Double? = nil, reasoning: Reasoning? = nil) {
        self.samplingMode = samplingMode
        self.temperature = temperature
        self.maximumResponseTokens = maximumResponseTokens
        self.stopSequences = stopSequences
        self.repetitionPenalty = repetitionPenalty
        self.reasoning = reasoning
    }

    /// The runtime's sampling for these options.
    var sampling: LanguageSampling {
        var sampling = LanguageSampling.greedy
        switch samplingMode?.kind ?? .greedy {
        case .greedy: break
        case .randomTopK(let k, let seed): sampling = .random(temperature: temperature ?? 1, topK: k, seed: seed)
        case .randomProbabilityThreshold(let p, let seed): sampling = .random(temperature: temperature ?? 1, topP: p, seed: seed)
        }
        sampling.repetitionPenalty = repetitionPenalty ?? 1
        return sampling
    }
}

/// A conversation: what renders into each request, in order. Reasoning is kept for the app but not rendered back, as
/// thinking models' templates expect.
public struct AetherTranscript: Sendable, Equatable, Codable {
    @nonexhaustive public enum Entry: Sendable, Equatable, Codable {
        case instructions(String)
        case prompt(String)
        case reasoning(String)
        case response(String)
    }

    public var entries: [Entry]

    public init(entries: [Entry] = []) { self.entries = entries }

    /// Chat messages for the model's template: instructions as the system message, prompts as user messages and
    /// responses as assistant messages.
    var messages: [LanguageChatMessage] {
        entries.compactMap { entry in
            switch entry {
            case .instructions(let text): .system(text)
            case .prompt(let text): .user(text)
            case .response(let text): .assistant(text)
            case .reasoning: nil
            }
        }
    }
}
