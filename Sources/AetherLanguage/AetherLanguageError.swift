/// Failures of the language facade.
@nonexhaustive public enum AetherLanguageError: Error, Sendable, CustomStringConvertible {
    /// The model can't serve requests here (``AetherLanguageModel/availability``).
    case unavailable(AetherLanguageModel.UnavailableReason)
    /// Verifying or loading the bundle failed; the underlying error says why. The next request tries again.
    case loading(any Error & Sendable)
    /// The rendered transcript plus the output limit doesn't fit the registered context. The transcript isn't trimmed:
    /// shorten it, or lower ``AetherGenerationOptions/maximumResponseTokens``.
    case contextWindowExceeded(promptTokens: Int, maximumResponseTokens: Int, limit: Int)
    /// The session already has a request in progress.
    case busy
    /// The model was unloaded while this request ran.
    case closed
    /// The device lacks the memory to grow the model's cache (iOS), or memory pressure is critical, so the model won't
    /// load until it eases (docs/model-lifecycle.md).
    case insufficientMemory
    /// The request is malformed (for example reasoning asked of a model without it).
    case invalidRequest(String)
    /// A generable type's schema uses something the JSON-schema constraint doesn't support (for example a `pattern`).
    /// Such a request is refused rather than run unconstrained.
    case unsupportedSchema(String)
    /// The constrained reply didn't decode into the requested type (for example it reached the output limit first).
    case decodingFailure(String)

    public var description: String {
        switch self {
        case .unavailable(let reason): "The language model is unavailable: \(reason)"
        case .loading(let error): "The language model failed to load: \(error)"
        case .contextWindowExceeded(let prompt, let output, let limit):
            "The prompt (\(prompt) tokens) and output limit (\(output)) exceed the model's context (\(limit))"
        case .busy: "This session already has a request in progress"
        case .closed: "The language model was unloaded"
        case .insufficientMemory: "Not enough memory to load the model or continue generating"
        case .invalidRequest(let reason): "Invalid request: \(reason)"
        case .unsupportedSchema(let reason): "Unsupported schema: \(reason)"
        case .decodingFailure(let reason): "The reply did not decode: \(reason)"
        }
    }
}
