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
package struct LanguageEngineReply: Sendable {
    package let content: String
    package let reasoning: String
    package let promptTokens: Int
    package let outputTokens: Int
    /// Of `outputTokens`, those inside the thinking block (delimiters included).
    package let reasoningTokens: Int
    package let stopReason: AetherStopReason
    package let seed: UInt64?
    /// The JSON-schema constraint's cost, for structured requests.
    package let constraintStatistics: LanguageJSONSchemaConstraint.Statistics?
}

/// A request for JSON matching a schema (structured output).
package struct LanguageStructuredOutput: Sendable {
    package let schema: AetherGenerationSchema
    /// Also describe the schema to the model, in the system message, as FoundationModels does by default.
    package let includeSchemaInPrompt: Bool

    package init(schema: AetherGenerationSchema, includeSchemaInPrompt: Bool) {
        self.schema = schema
        self.includeSchemaInPrompt = includeSchemaInPrompt
    }

    /// The system-message text that describes the schema.
    package var instruction: String {
        "Respond with only a JSON object that matches this JSON schema:\n\(schema.json)"
    }

    /// `messages` with the schema described in the system message (added first when there's none).
    package func messages(_ messages: [LanguageChatMessage]) -> [LanguageChatMessage] {
        guard includeSchemaInPrompt else { return messages }
        if let first = messages.first, first.role == .system, case .text(let text)? = first.parts.first {
            return [.system(text + "\n\n" + instruction)] + messages.dropFirst()
        }
        return [.system(instruction)] + messages
    }
}

/// The `LanguageTextStream` a request's deliveries share. Deliveries run one at a time, but in successive child tasks.
private final class TextStreamBox: Sendable {
    private let stream: OSAllocatedUnfairLock<LanguageTextStream>
    init(_ stream: LanguageTextStream) { self.stream = OSAllocatedUnfairLock(uncheckedState: stream) }
    /// The delta, whether a stop sequence matched, and whether the token belongs to the thinking block.
    func append(_ token: Int) -> (LanguageTextStream.Delta, stopped: Bool, reasoning: Bool) {
        stream.withLockUnchecked { stream in
            let before = stream.inReasoning
            let delta = stream.append(token)
            return (delta, stream.stopped, before || stream.inReasoning)
        }
    }
    func finish() -> (LanguageTextStream.Delta, stopped: Bool, content: String, reasoning: String) {
        stream.withLockUnchecked { stream in (stream.finish(), stream.stopped, stream.content, stream.reasoning) }
    }
}

extension AetherLanguageModel {
    /// The engine both surfaces share: renders `messages` with the model's template, generates under `options`, and
    /// streams text deltas to `onDelta` (each awaited before the next, while the following decode step runs).
    /// `onDelta` also gets how many tokens produced the delta (tokens whose text was held back count with the delta
    /// that releases it).
    /// With `structured`, the reply is constrained to JSON matching its schema (refused up front if the schema uses
    /// something the constraint doesn't support).
    package func generate(messages: [LanguageChatMessage], options: AetherGenerationOptions,
                          structured: LanguageStructuredOutput? = nil,
                          onDelta: @escaping @Sendable (LanguageTextStream.Delta, _ tokens: Int) async -> Void)
        async throws -> LanguageEngineReply {
        let messages = structured?.messages(messages) ?? messages
        // Compile the schema before loading anything: an unsupported one fails fast.
        if let structured { _ = try LanguageJSONGrammar(schema: structured.schema) }
        let resources = self.resources
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
            var constraint: LanguageJSONSchemaConstraint?
            if let structured {
                let excluded = Set([delimiters?.start, delimiters?.end].compactMap { $0 })
                let vocabulary = await resources.jsonVocabulary(model, excluded: excluded)
                constraint = try LanguageJSONSchemaConstraint(schema: structured.schema, vocabulary: vocabulary,
                    stopTokenIDs: model.backend.stopTokenIDs, reasoningEnd: thinking ? delimiters?.end : nil)
            }
            let request = LanguageGenerationRequest(prompt: prompt, maximumTokens: maximum, sampling: sampling, constraint: constraint)
            let counts = OSAllocatedUnfairLock(initialState: (pending: 0, reasoning: 0))
            let result: LanguageGenerationResult
            do {
                result = try await model.backend.generate(request) { token in
                    let (delta, stopped, reasoning) = text.append(token.id)
                    let tokens = counts.withLock { counts in
                        counts.pending += 1
                        if reasoning { counts.reasoning += 1 }
                        guard !delta.isEmpty else { return 0 }
                        defer { counts.pending = 0 }
                        return counts.pending
                    }
                    if !delta.isEmpty { await onDelta(delta, tokens) }
                    return stopped ? .stop : .continue
                }
            } catch let error as LanguageGenerationError {
                throw Self.facadeError(error, promptTokens: prompt.count, maximum: maximum, limit: limit)
            }
            let (last, stopped, content, reasoning) = text.finish()
            let (pending, reasoningTokens) = counts.withLock { ($0.pending, $0.reasoning) }
            if !last.isEmpty { await onDelta(last, pending) }
            let stopReason: AetherStopReason = stopped ? .stopSequence
                : result.stopReason == .endOfSequence ? .endOfTurn : .outputLimit
            return LanguageEngineReply(content: content, reasoning: reasoning, promptTokens: prompt.count,
                                       outputTokens: result.tokens.count, reasoningTokens: reasoningTokens, stopReason: stopReason,
                                       seed: result.seed, constraintStatistics: await constraint?.statistics)
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
