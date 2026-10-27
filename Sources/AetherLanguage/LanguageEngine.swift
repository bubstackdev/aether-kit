import AetherLanguageRuntime
import AetherLanguageTokenizers
import os

/// Why a reply ended.
public enum AetherStopReason: Sendable, Equatable {
    /// The model ended its turn (a registered end-of-sequence token).
    case endOfTurn
    /// It reached `maximumResponseTokens`.
    case outputLimit
    /// One of the request's `stopSequences` appeared; the reply ends before it.
    case stopSequence
}

/// One generated reply, before it becomes a session's `Response` or a FoundationModels channel's events.
struct LanguageEngineReply: Sendable {
    let content: String
    let reasoning: String
    let promptTokens: Int
    let outputTokens: Int
    let stopReason: AetherStopReason
    let seed: UInt64?
}

/// The `LanguageTextStream` a request's deliveries share. Deliveries run one at a time, but in successive child tasks.
private final class TextStreamBox: Sendable {
    private let stream: OSAllocatedUnfairLock<LanguageTextStream>
    init(_ stream: LanguageTextStream) { self.stream = OSAllocatedUnfairLock(uncheckedState: stream) }
    func append(_ token: Int) -> (LanguageTextStream.Delta, stopped: Bool) {
        stream.withLockUnchecked { stream in (stream.append(token), stream.stopped) }
    }
    func finish() -> (LanguageTextStream.Delta, stopped: Bool, content: String, reasoning: String) {
        stream.withLockUnchecked { stream in (stream.finish(), stream.stopped, stream.content, stream.reasoning) }
    }
}

extension AetherLanguageModel {
    /// The engine both surfaces share: renders `messages` with the model's template, generates under `options`, and
    /// streams text deltas to `onDelta` (each awaited before the next, while the following decode step runs).
    func generate(messages: [LanguageChatMessage], options: AetherGenerationOptions,
                  onDelta: @escaping @Sendable (LanguageTextStream.Delta) async -> Void) async throws -> LanguageEngineReply {
        let supportsReasoning = self.supportsReasoning
        if options.reasoning == .on, !supportsReasoning {
            throw AetherLanguageError.invalidRequest("this model has no thinking mode")
        }
        if let limit = options.maximumResponseTokens, limit <= 0 {
            throw AetherLanguageError.invalidRequest("maximumResponseTokens must be positive")
        }
        let sampling = options.sampling
        do { try sampling.validate() } catch { throw AetherLanguageError.invalidRequest("invalid sampling options") }
        return try await resources.withModel { model in
            let thinking = options.reasoning == .on
            let prompt: [Int]
            do { prompt = try model.codec.render(messages, thinking: thinking) }
            catch { throw AetherLanguageError.invalidRequest("the chat template failed: \(error)") }
            let limit = model.backend.maximumContextLength
            let maximum = options.maximumResponseTokens ?? min(limit - prompt.count, AetherGenerationOptions.defaultMaximumResponseTokens)
            guard maximum > 0, prompt.count <= limit, maximum <= limit - prompt.count else {
                throw AetherLanguageError.contextWindowExceeded(promptTokens: prompt.count, maximumResponseTokens: max(maximum, 0), limit: limit)
            }
            var delimiters: LanguageTextStream.ReasoningDelimiters?
            if supportsReasoning, let start = model.codec.tokenID("<think>"), let end = model.codec.tokenID("</think>") {
                delimiters = .init(start: start, end: end)
            }
            // A prompt that leaves a thinking block open (some templates open it for the model) starts the reply inside it.
            let startsInReasoning = delimiters.map { (prompt.lastIndex(of: $0.start) ?? -1) > (prompt.lastIndex(of: $0.end) ?? -1) } ?? false
            let text = TextStreamBox(LanguageTextStream(bytes: model.codec.bytes, delimiters: delimiters,
                                                        startsInReasoning: startsInReasoning, stopSequences: options.stopSequences))
            let request = LanguageGenerationRequest(prompt: prompt, maximumTokens: maximum, sampling: sampling)
            let result: LanguageGenerationResult
            do {
                result = try await model.backend.generate(request) { token in
                    let (delta, stopped) = text.append(token.id)
                    if !delta.isEmpty { await onDelta(delta) }
                    return stopped ? .stop : .continue
                }
            } catch let error as LanguageGenerationError {
                throw Self.facadeError(error, promptTokens: prompt.count, maximum: maximum, limit: limit)
            }
            let (last, stopped, content, reasoning) = text.finish()
            if !last.isEmpty { await onDelta(last) }
            let stopReason: AetherStopReason = stopped ? .stopSequence
                : result.stopReason == .endOfSequence ? .endOfTurn : .outputLimit
            return LanguageEngineReply(content: content, reasoning: reasoning, promptTokens: prompt.count,
                                       outputTokens: result.tokens.count, stopReason: stopReason, seed: result.seed)
        }
    }

    private static func facadeError(_ error: LanguageGenerationError, promptTokens: Int, maximum: Int, limit: Int) -> AetherLanguageError {
        switch error {
        case .busy: .busy
        case .closed: .closed
        case .contextLimit(let backendLimit): .contextWindowExceeded(promptTokens: promptTokens, maximumResponseTokens: maximum, limit: backendLimit)
        case .memoryBudget: .insufficientMemory
        case .invalid(let reason): .invalidRequest(reason)
        }
    }
}
