import AetherLanguage
import AetherLanguageTokenizers
import Foundation
import FoundationModels

/// An installed Kit language model as a FoundationModels `LanguageModel` (OS 27), so an app keeps its
/// `LanguageModelSession` code and changes one line:
///
/// ```swift
/// let session = LanguageModelSession(model: AetherLanguageModel(bundle: installedBundleURL), instructions: "…")
/// ```
///
/// The same engine as ``AetherLanguageSession`` renders the transcript with the model's own template, generates with
/// the requested sampling (greedy unless the request opts in), and streams UTF-8-safe text, and the model's thinking
/// when a `reasoningLevel` asks for it. Tool calling, attachments and guided generation (`@Generable`) aren't declared
/// yet, so FoundationModels refuses those requests itself.
@available(macOS 27, iOS 27, *)
extension AetherLanguageModel: LanguageModel {
    public typealias Executor = AetherLanguageExecutor

    public var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities(supportsReasoning ? [.reasoning] : [])
    }

    public var executorConfiguration: AetherLanguageExecutor.Configuration { .init(model: self) }
}

@available(macOS 27, iOS 27, *)
public struct AetherLanguageExecutor: LanguageModelExecutor {
    public typealias Model = AetherLanguageModel

    /// Copies of one model share its loaded backend, so they're the same configuration.
    public struct Configuration: Hashable, Sendable {
        let model: AetherLanguageModel
        public static func == (lhs: Self, rhs: Self) -> Bool { lhs.model.identity == rhs.model.identity }
        public func hash(into hasher: inout Hasher) { hasher.combine(model.identity) }
    }

    public init(configuration: Configuration) throws {}

    public func respond(to request: LanguageModelExecutorGenerationRequest, model: AetherLanguageModel,
                        streamingInto channel: LanguageModelExecutorGenerationChannel) async throws {
        let options = Self.options(request.generationOptions, request.contextOptions, supportsReasoning: model.supportsReasoning)
        let reply: LanguageEngineReply
        do {
            reply = try await model.generate(messages: try Self.messages(request.transcript), options: options) { delta, tokens in
                if !delta.reasoning.isEmpty {
                    await channel.send(.reasoning(action: .appendText(delta.reasoning, tokenCount: delta.content.isEmpty ? tokens : 0)))
                }
                if !delta.content.isEmpty { await channel.send(.response(action: .appendText(delta.content, tokenCount: tokens))) }
            }
        } catch AetherLanguageError.contextWindowExceeded(let prompt, let output, let limit) {
            throw LanguageModelError.contextSizeExceeded(.init(contextSize: limit, tokenCount: prompt + output,
                debugDescription: "The prompt (\(prompt) tokens) and output limit (\(output)) exceed the model's context (\(limit))"))
        }
        await channel.send(.response(action: .updateUsage(
            input: .init(totalTokenCount: reply.promptTokens, cachedTokenCount: 0),
            output: .init(totalTokenCount: reply.outputTokens, reasoningTokenCount: reply.reasoningTokens))))
    }

    /// The transcript as chat messages for the model's template: instructions as the system message, prompts as user
    /// messages, responses as assistant messages. Earlier reasoning isn't rendered back.
    static func messages(_ transcript: Transcript) throws -> [LanguageChatMessage] {
        try transcript.compactMap { entry -> LanguageChatMessage? in
            switch entry {
            case .instructions(let instructions): .system(try text(instructions.segments))
            case .prompt(let prompt): .user(try text(prompt.segments))
            case .response(let response): .assistant(try text(response.segments))
            case .reasoning: nil
            default: throw AetherLanguageError.invalidRequest("this model doesn't take tool calls")
            }
        }
    }

    private static func text(_ segments: [Transcript.Segment]) throws -> String {
        try segments.map { segment in
            switch segment {
            case .text(let text): text.content
            case .structure(let structure): structure.content.jsonString
            default: throw AetherLanguageError.invalidRequest("this model doesn't take attachments")
            }
        }.joined()
    }

    /// FoundationModels' options in the facade's terms. Any reasoning level turns thinking on (the model has one
    /// switch); no level leaves the bundle's default.
    static func options(_ generation: GenerationOptions, _ context: ContextOptions, supportsReasoning: Bool) -> AetherGenerationOptions {
        var options = AetherGenerationOptions(temperature: generation.temperature, maximumResponseTokens: generation.maximumResponseTokens)
        switch generation.samplingMode?.kind {
        case nil: break
        case .greedy?: options.samplingMode = .greedy
        case .randomTopK(let k, let seed)?: options.samplingMode = .random(top: k, seed: seed)
        case .randomProbabilityThreshold(let p, let seed)?: options.samplingMode = .random(probabilityThreshold: p, seed: seed)
        @unknown default: break
        }
        if context.reasoningLevel != nil, supportsReasoning { options.reasoning = .on }
        return options
    }
}
