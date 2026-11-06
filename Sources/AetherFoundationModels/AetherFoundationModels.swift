// AetherFoundationModels: the FoundationModels `LanguageModel` bridge (DESIGN.md §7, M3), on the Generator.
@_exported public import AetherText
public import FoundationModels
import Foundation

/// A loaded Aether chat model as a FoundationModels `LanguageModel`, so an app keeps its `LanguageModelSession` code:
///
/// ```swift
/// let model = try await AetherLanguageModel.load(bundle: bundle, manifest: manifest, device: .current)
/// let session = LanguageModelSession(model: model, instructions: "…")
/// ```
///
/// It runs on the same ``Generator`` as ``ChatSession`` (ported from 0.0.x's adapter, re-expressed on it):
/// - the transcript renders with the model's own template; prefix reuse makes each turn cost its new tokens;
/// - options map as below; the reply streams as response and reasoning deltas, then the exact usage;
/// - guided generation (`@Generable`) is constrained by the JSON-schema constraint inside the decode loop, and a
///   schema outside its subset (a regex guide) is refused with `LanguageModelError.unsupportedGenerationGuide`;
/// - tool calling, when the manifest declares a tool dialect this build implements (`hermes-json`: qwen3, granite);
///   other dialects don't declare the capability, so FoundationModels refuses tools for them itself.
/// Capabilities come from the manifest (DESIGN §7): tools if a dialect is implemented, reasoning if thinking tags
/// are declared, and guided generation always.
public struct AetherLanguageModel: LanguageModel {
    public typealias Executor = AetherLanguageModelExecutor

    public let generator: Generator

    public init(generator: Generator) { self.generator = generator }

    /// Loads a `decoder-lm@1` chat model from an installed bundle.
    public static func load(
        bundle: InstalledBundle, manifest: Manifest, device: Device, engine options: DecoderEngine.Options = .init()
    ) async throws -> AetherLanguageModel {
        AetherLanguageModel(generator: try await ChatSession.generator(bundle: bundle, manifest: manifest, device: device, engine: options))
    }

    public var supportsReasoning: Bool { generator.profile.thinkingTag != nil }

    /// The manifest's tool dialect, when this build implements it.
    public var toolDialect: (any ToolDialect)? { try? ToolDialects.make(generator.profile.toolDialect) }

    public var capabilities: LanguageModelCapabilities {
        // Guided generation needs the token byte table (a byte-level tokenizer) to walk the vocabulary.
        var capabilities: [LanguageModelCapabilities.Capability] = generator.supportsStructuredOutput ? [.guidedGeneration] : []
        if supportsReasoning { capabilities.append(.reasoning) }
        if toolDialect != nil { capabilities.append(.toolCalling) }
        return LanguageModelCapabilities(capabilities)
    }

    public var executorConfiguration: AetherLanguageModelExecutor.Configuration { .init(generator: generator) }
}

public struct AetherLanguageModelExecutor: LanguageModelExecutor {
    public typealias Model = AetherLanguageModel

    /// Copies of one model share its Generator, so they are the same configuration.
    public struct Configuration: Hashable, Sendable {
        let generator: Generator
        public static func == (lhs: Self, rhs: Self) -> Bool { lhs.generator === rhs.generator }
        public func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(generator)) }
    }

    /// The output limit when a request names none (and the context allows it), as in 0.0.x.
    public static let defaultMaximumResponseTokens = 4096

    public init(configuration: Configuration) throws {}

    public func respond(
        to request: LanguageModelExecutorGenerationRequest, model: AetherLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        var schemaName: String?
        do {
            try await generate(request, model: model, channel: channel, schemaName: &schemaName)
        } catch {
            throw Self.mapped(error, schemaName: schemaName)
        }
    }

    private func generate(
        _ request: LanguageModelExecutorGenerationRequest, model: AetherLanguageModel, channel: LanguageModelExecutorGenerationChannel,
        schemaName: inout String?
    ) async throws {
        let generator = model.generator
        var messages = try Self.messages(request.transcript)
        var parameters = Self.parameters(request.generationOptions, request.contextOptions, supportsReasoning: model.supportsReasoning)
        let dialect = request.enabledToolDefinitions.isEmpty ? nil : model.toolDialect
        if dialect != nil { parameters.tools = try request.enabledToolDefinitions.map(Self.tool) }
        var constraint: JSONSchemaConstraint?
        if let schema = request.schema {
            guard generator.supportsStructuredOutput else {
                throw LanguageModelError.unsupportedCapability(.init(
                    capability: .guidedGeneration, debugDescription: "Guided generation needs a byte-level tokenizer"))
            }
            let output = try OutputSchema(json: try JSONEncoder().encode(schema))
            schemaName = output.name
            // Thinking only when a reasoning level asks for it; then the value is constrained after the thinking.
            let thinking = parameters.thinking == true
            parameters.thinking = thinking
            let json = try await generator.jsonConstraint(output, allowThinking: thinking)
            constraint = json
            parameters.logitsProcessors.append(json)
            // In the current prompt, not the system message: turns that alternate guided and plain keep their prefix.
            if request.contextOptions.includeSchemaInPrompt ?? true {
                messages = Self.addingToPrompt(ChatSession.schemaInstruction(output), to: messages)
            }
        }

        // The rendered prompt plus the output limit must fit the model's context; it is never trimmed.
        let prompt = try generator.renderer.render(messages, options: RenderOptions(thinking: parameters.thinking, tools: parameters.tools))
        let limit = generator.engine.maxCapacity
        let maximum = request.generationOptions.maximumResponseTokens ?? min(limit - prompt.tokens.count, Self.defaultMaximumResponseTokens)
        guard maximum > 0, prompt.tokens.count + maximum <= limit else {
            throw LanguageModelError.contextSizeExceeded(.init(
                contextSize: limit, tokenCount: prompt.tokens.count + max(maximum, 0),
                debugDescription: "The prompt (\(prompt.tokens.count) tokens) and output limit (\(maximum)) exceed the model's context (\(limit))"))
        }
        parameters.maxTokens = maximum

        var summary: GenerationSummary?
        var calledTools = false
        var text = ""
        let tag = generator.profile.tags.first { $0.kind == .toolCall }
        for try await event in generator.generate(messages, parameters: parameters) {
            switch event {
            case .text(let delta):
                text += delta
                await channel.send(.response(action: .appendText(delta, tokenCount: 0)))
            case .thinking(let delta):
                await channel.send(.reasoning(action: .appendText(delta, tokenCount: 0)))
            case .toolCall(let body):
                // No tools offered, or a body that doesn't parse (two calls run together, a call cut at the limit):
                // the model's text, as it wrote it, rather than an error after text already streamed.
                guard let dialect, let calls = try? dialect.parse(body) else {
                    let raw = (tag?.open ?? "") + body + (tag?.close ?? "")
                    text += raw
                    await channel.send(.response(action: .appendText(raw, tokenCount: 0)))
                    continue
                }
                calledTools = true
                for call in calls {
                    await channel.send(.toolCalls(action: .toolCall(id: call.id, name: call.name, action: .appendArguments(call.arguments, tokenCount: 0))))
                    // The arguments as generated, to render back byte for byte (FoundationModels re-serializes them).
                    await channel.send(.toolCalls(action: .toolCall(id: call.id, name: call.name, action: .updateMetadata([Self.rawArguments: call.arguments]))))
                }
            case .finished(let s):
                summary = s
            }
        }
        guard let summary else { throw CancellationError() }
        // A guided reply must end with its value: one cut at the output limit or the context isn't handed over as done.
        if let constraint, !constraint.completes(text) {
            if case .capacity = summary.stopReason {
                throw LanguageModelError.contextSizeExceeded(.init(
                    contextSize: limit, tokenCount: summary.promptTokens + summary.tokens.count,
                    debugDescription: "The context filled before the structured value ended"))
            }
            throw GeneratedContent.ParsingError(rawContent: text, debugDescription: "The reply reached its output limit before the value ended")
        }
        let input = LanguageModelExecutorGenerationChannel.Usage.Input(totalTokenCount: summary.promptTokens, cachedTokenCount: summary.reusedTokens)
        let output = LanguageModelExecutorGenerationChannel.Usage.Output(totalTokenCount: summary.tokens.count, reasoningTokenCount: summary.thinkingTokens)
        if calledTools {
            await channel.send(.toolCalls(action: .updateUsage(input: input, output: output)))
        } else {
            await channel.send(.response(action: .updateUsage(input: input, output: output)))
        }
    }

    /// The tool-call metadata key holding the arguments as the model generated them.
    static let rawArguments = "aether.arguments"

    /// Aether's errors as FoundationModels' where they correspond; the rest pass through.
    static func mapped(_ error: any Error, schemaName: String?) -> any Error {
        guard let error = error as? AetherError else { return error }
        switch error {
        case .generation(.unsupportedSchema(let reason)):
            return LanguageModelError.unsupportedGenerationGuide(.init(schemaName: schemaName, debugDescription: "The schema isn't supported: \(reason)"))
        case .generation(.decodingFailed(let reason)), .generation(.constraintViolated(let reason)),
             .generation(.constraintExhausted(let reason)), .generation(.toolCall(let reason)):
            return GeneratedContent.ParsingError(rawContent: "", debugDescription: reason)
        case .capacity(let what, let needed, let available):
            return LanguageModelError.contextSizeExceeded(.init(contextSize: Int(available), tokenCount: Int(needed), debugDescription: "Insufficient \(what)"))
        default:
            return error
        }
    }

    // MARK: - Mapping

    /// The transcript as chat messages for the model's template: instructions as the system message, prompts as user
    /// messages, responses as assistant messages, tool calls as the assistant's `tool_calls` and tool outputs as tool
    /// messages. Earlier reasoning isn't rendered back, as thinking models' templates expect.
    static func messages(_ transcript: Transcript) throws -> [ChatMessage] {
        var messages: [ChatMessage] = []
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                messages.append(.system(try text(instructions.segments, in: entry)))
            case .prompt(let prompt):
                messages.append(.user(try text(prompt.segments, in: entry)))
            case .response(let response):
                messages.append(.assistant(try text(response.segments, in: entry)))
            case .toolCalls(let calls):
                let mapped = calls.map { call in
                    let raw = call.metadata[rawArguments].flatMap { try? $0.value(String.self) }
                    return ToolCall(id: call.id, name: call.toolName, arguments: raw ?? call.arguments.jsonString)
                }
                // A reply that wrote text and then called tools is one assistant turn.
                if let last = messages.last, last.role == .assistant, last.toolCalls.isEmpty {
                    messages[messages.count - 1].toolCalls = mapped
                } else {
                    messages.append(.assistant("", toolCalls: mapped))
                }
            case .toolOutput(let output):
                messages.append(.tool(try text(output.segments, in: entry), name: output.toolName))
            case .reasoning:
                continue
            default:
                throw LanguageModelError.unsupportedTranscriptContent(.init(unsupportedContent: [entry], debugDescription: "This model takes no data entries"))
            }
        }
        return messages
    }

    private static func text(_ segments: [Transcript.Segment], in entry: Transcript.Entry) throws -> String {
        try segments.map { segment in
            switch segment {
            case .text(let text): text.content
            case .structure(let structure): structure.content.jsonString
            default: throw LanguageModelError.unsupportedTranscriptContent(.init(unsupportedContent: [entry], debugDescription: "This model takes no attachments"))
            }
        }.joined()
    }

    /// `messages` with `text` after the last user prompt (or as a new one).
    static func addingToPrompt(_ text: String, to messages: [ChatMessage]) -> [ChatMessage] {
        guard let last = messages.lastIndex(where: { $0.role == .user }) else { return messages + [.user(text)] }
        var messages = messages
        messages[last].content += "\n\n" + text
        return messages
    }

    static func tool(_ definition: Transcript.ToolDefinition) throws -> ToolDefinition {
        let parameters = try JSONDecoder().decode(JSONValue.self, from: try JSONEncoder().encode(definition.parameters))
        return ToolDefinition(name: definition.name, description: definition.description, parameters: parameters)
    }

    /// FoundationModels' options in the Generator's terms (0.0.x's mapping): no sampling mode is greedy, and the
    /// temperature applies only to a random mode (default 1), as FoundationModels ignores it when greedy; a random
    /// mode without a seed draws one. Any reasoning level turns thinking on (the model has one switch); no level
    /// turns it off, so a hybrid thinks only when asked.
    static func parameters(_ generation: GenerationOptions, _ context: ContextOptions, supportsReasoning: Bool) -> GenerationParameters {
        var parameters = GenerationParameters()
        let temperature = Float(generation.temperature ?? 1)
        switch generation.samplingMode?.kind {
        case .randomTopK(let k, let seed)?:
            parameters.sampling = SamplingParameters(temperature: temperature, topK: k, seed: seed ?? UInt64.random(in: .min ... .max))
        case .randomProbabilityThreshold(let p, let seed)?:
            parameters.sampling = SamplingParameters(temperature: temperature, topP: p, seed: seed ?? UInt64.random(in: .min ... .max))
        default:
            parameters.sampling = .greedy
        }
        if let maximum = generation.maximumResponseTokens { parameters.maxTokens = maximum }
        if supportsReasoning { parameters.thinking = context.reasoningLevel != nil }
        return parameters
    }
}
