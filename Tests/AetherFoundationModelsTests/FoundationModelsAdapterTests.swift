import Foundation
import FoundationModels
import Synchronization
import Testing
@testable import AetherFoundationModels
import AetherTestSupport

@Generable private struct Look {
    @Guide(description: "a short look") var look: String
}

@Generable private struct Tagged {
    @Guide(.pattern(/[a-z]+/)) var slug: String
}

/// A tool that records what it was called with.
private struct Weather: Tool {
    final class Calls: Sendable {
        let cities = Mutex<[String]>([])
    }
    let calls = Calls()
    let name = "weather"
    let description = "The weather in a city"
    @Generable struct Arguments {
        @Guide(description: "The city") var city: String
    }
    func call(arguments: Arguments) async throws -> String {
        calls.cities.withLock { $0.append(arguments.city) }
        return "sunny, 21 C"
    }
}

/// The adapter through a real FoundationModels `LanguageModelSession` over the fake byte-level Generator (0.0.x
/// `AetherLanguageFoundationModelsTests`, plus tool calling).
@Suite("FoundationModels adapter (fake)")
struct FoundationModelsAdapterTests {
    static func model(capacity: Int = 4096, profile: OutputProfile = ByteVocab.profile, log: PromptLog? = nil,
                      _ script: @escaping @Sendable ([Int32]) -> [Int32]) async throws -> AetherLanguageModel {
        AetherLanguageModel(generator: try await ByteChat.generator(capacity: capacity, profile: profile) { prompt in
            log?.append(prompt)
            return script(prompt)
        })
    }

    static func render(_ messages: [ChatMessage], thinking: Bool? = false, tools: [ToolDefinition] = []) throws -> [Int32] {
        try ByteRenderer().render(messages, options: RenderOptions(thinking: thinking, tools: tools)).tokens
    }

    @Test func languageModelSessionRunsOnTheGenerator() async throws {
        let log = PromptLog()
        let model = try await Self.model(log: log) { _ in ByteVocab.tokens("Teal, with a clasp 🌙.") }
        let session = LanguageModelSession(model: model, instructions: "Describe looks.")
        let reply = try await session.respond(to: "Her coat?")
        #expect(reply.content == "Teal, with a clasp 🌙.")
        #expect(log.all.last == (try Self.render([.system("Describe looks."), .user("Her coat?")])), "the transcript renders with the model's template")

        // The next turn renders the first, and streaming delivers the same text.
        var last = ""
        for try await snapshot in session.streamResponse(to: "And her boots?") { last = snapshot.content }
        #expect(last == "Teal, with a clasp 🌙.")
        let second = try Self.render([.system("Describe looks."), .user("Her coat?"), .assistant("Teal, with a clasp 🌙."), .user("And her boots?")])
        #expect(log.all.last == second)
    }

    @Test func optionsMapAndReasoningLevelsTurnThinkingOn() async throws {
        let topK = AetherLanguageModelExecutor.parameters(
            GenerationOptions(samplingMode: .random(top: 40, seed: 7), temperature: 0.7, maximumResponseTokens: 32), ContextOptions(),
            supportsReasoning: true)
        #expect(topK.sampling == SamplingParameters(temperature: 0.7, topK: 40, seed: 7) && topK.maxTokens == 32 && topK.thinking == false)
        let topP = AetherLanguageModelExecutor.parameters(
            GenerationOptions(samplingMode: .random(probabilityThreshold: 0.9, seed: 3)), ContextOptions(), supportsReasoning: true)
        #expect(topP.sampling == SamplingParameters(temperature: 1, topP: 0.9, seed: 3))
        #expect(AetherLanguageModelExecutor.parameters(GenerationOptions(temperature: 2), ContextOptions(), supportsReasoning: true).sampling == .greedy)
        let reasoning = AetherLanguageModelExecutor.parameters(GenerationOptions(), ContextOptions(reasoningLevel: .light), supportsReasoning: true)
        #expect(reasoning.thinking == true)
        #expect(AetherLanguageModelExecutor.parameters(GenerationOptions(), ContextOptions(reasoningLevel: .deep), supportsReasoning: false).thinking == nil)

        let log = PromptLog()
        let model = try await Self.model(log: log) { _ in
            [ByteVocab.thinkStart] + ByteVocab.tokens("\nweighing it\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens("\n\nYes.")
        }
        #expect(model.capabilities.contains(.reasoning) && model.capabilities.contains(.guidedGeneration) && model.capabilities.contains(.toolCalling))
        let thinking = try await LanguageModelSession(model: model)
            .respond(to: "q", options: GenerationOptions(), contextOptions: ContextOptions(reasoningLevel: .moderate))
        #expect(thinking.content == "Yes.")
        #expect(log.all.last == (try Self.render([.user("q")], thinking: true)), "a reasoning level renders thinking on")
    }

    /// @Generable through the adapter: the type's schema constrains generation, FoundationModels decodes the JSON,
    /// and the schema is described in the system message.
    @Test func generableRequestsAreConstrainedAndDecode() async throws {
        let log = PromptLog()
        let model = try await Self.model(log: log) { _ in ByteVocab.tokens(#"{"look": "teal coat"}"#) }
        let session = LanguageModelSession(model: model, instructions: "Be brief.")
        let reply = try await session.respond(to: "q", generating: Look.self)
        #expect(reply.content.look == "teal coat")
        let schema = try OutputSchema(json: try JSONEncoder().encode(Look.generationSchema))
        #expect(log.all.last == (try Self.render([.system("Be brief."), .user(ChatSession.withSchemaInstruction("q", schema))])),
                "the schema is described in the prompt, not the system message (review #6)")
        var last: Look.PartiallyGenerated?
        for try await snapshot in session.streamResponse(to: "again", generating: Look.self) { last = snapshot.content }
        #expect(last?.look == "teal coat")
    }

    /// A model that strays from the schema is held to it: the decoded value is valid for the type.
    @Test func aStrayingModelIsHeldToTheSchema() async throws {
        let model = try await Self.model { _ in ByteVocab.tokens(#"{"look":"teal"}"#) }
        let reply = try await LanguageModelSession(model: model).respond(to: "q", generating: Look.self)
        #expect(!reply.content.look.contains("\u{FFFD}"))
    }

    @Test func unsupportedGuidesBecomeFoundationModelsError() async throws {
        let log = PromptLog()
        let model = try await Self.model(log: log) { _ in ByteVocab.tokens(#"{"slug": "x"}"#) }
        do {
            _ = try await LanguageModelSession(model: model).respond(to: "q", generating: Tagged.self)
            Issue.record("expected an unsupported guide")
        } catch let error as LanguageModelError {
            guard case .unsupportedGenerationGuide = error else { Issue.record("unexpected \(error)"); return }
        }
        #expect(log.all.isEmpty, "refused before generating")
    }

    @Test func contextOverflowBecomesFoundationModelsError() async throws {
        let model = try await Self.model(capacity: 64) { _ in ByteVocab.tokens("x") }
        do {
            _ = try await LanguageModelSession(model: model).respond(to: String(repeating: "long ", count: 20))
            Issue.record("expected a context error")
        } catch let error as LanguageModelError {
            guard case .contextSizeExceeded(let detail) = error else { Issue.record("unexpected \(error)"); return }
            #expect(detail.contextSize == 64)
        }
    }

    /// A Hermes tool call round trip: the model calls the tool, FoundationModels runs it and hands the output back,
    /// and the model answers from it. The tools render into the template; the call and its output render back.
    @Test func hermesToolCallRoundTrip() async throws {
        let log = PromptLog()
        let call = #"{"name": "weather", "arguments": {"city": "Paris"}}"#
        let model = try await Self.model(log: log) { prompt in
            ByteChat.text(prompt).contains("<tool_response>sunny, 21 C</tool_response>")
                ? ByteVocab.tokens("It is sunny in Paris.")
                : ByteVocab.tokens("Checking.") + [ByteVocab.toolOpen] + ByteVocab.tokens("\n\(call)\n") + [ByteVocab.toolClose]
        }
        let weather = Weather()
        let session = LanguageModelSession(model: model, tools: [weather], instructions: "Use tools.")
        let reply = try await session.respond(to: "Weather in Paris?")
        #expect(reply.content == "It is sunny in Paris.")
        #expect(weather.calls.cities.withLock { $0 } == ["Paris"])
        let text = ByteChat.text(log.all.last ?? [])
        #expect(text.contains("tools: weather"), "the tool is rendered into the prompt")
        #expect(text.contains(#"assistant\#nChecking.<tool_call>{"name": "weather", "arguments": {"city": "Paris"}}</tool_call>"#),
                "the text and the call render back as one assistant turn")
        #expect(text.contains("<|im_start|>tool\n<tool_response>sunny, 21 C</tool_response>"), "the output renders back")
        let entries = session.transcript.map { entry -> String in
            switch entry {
            case .instructions: "instructions"
            case .prompt: "prompt"
            case .toolCalls: "toolCalls"
            case .toolOutput: "toolOutput"
            case .response: "response"
            default: "other"
            }
        }
        #expect(entries.contains("toolCalls") && entries.contains("toolOutput") && entries.last == "response")
    }

    /// Tools are declared only for a dialect this build implements; FoundationModels then refuses tools itself.
    @Test func toolCallingFollowsTheManifestDialect() async throws {
        let tags = ByteVocab.profile.tags
        let none = try await Self.model(profile: OutputProfile(tags: tags)) { _ in [] }
        let xml = try await Self.model(profile: OutputProfile(tags: tags, toolDialect: "qwen-xml")) { _ in [] }
        let plain = try await Self.model(profile: .plain) { _ in [] }
        #expect(!none.capabilities.contains(.toolCalling) && !xml.capabilities.contains(.toolCalling))
        #expect(!plain.capabilities.contains(.reasoning) && plain.capabilities.contains(.guidedGeneration))
        #expect(throws: AetherError.self) { try ToolDialects.make("qwen-xml") }
        #expect(throws: AetherError.self) { try ToolDialects.make(nil) }
    }

    @Test func copiesOfAModelShareAConfiguration() async throws {
        let model = try await Self.model { _ in [] }
        let copy = model
        #expect(model.executorConfiguration == copy.executorConfiguration)
        #expect(model.executorConfiguration != (try await Self.model { _ in [] }).executorConfiguration)
    }

    // MARK: Review regressions

    /// Review #4: a guided reply cut at the output limit is an error, not truncated JSON handed over as done; one
    /// that completes exactly at the limit is fine.
    @Test func aGuidedReplyCutAtTheLimitThrows() async throws {
        let json = #"{"look": "teal coat"}"#
        let model = try await Self.model { _ in ByteVocab.tokens(json) }
        do {
            _ = try await LanguageModelSession(model: model).respond(to: "q", generating: Look.self, options: GenerationOptions(maximumResponseTokens: 8))
            Issue.record("expected a parsing error")
        } catch let error as GeneratedContent.ParsingError {
            #expect(error.debugDescription.contains("output limit"))
        }
        let exact = try await LanguageModelSession(model: model)
            .respond(to: "q", generating: Look.self, options: GenerationOptions(maximumResponseTokens: json.utf8.count))
        #expect(exact.content.look == "teal coat")
    }

    /// Review #4: without a byte table (a SentencePiece tokenizer) guided generation is not advertised.
    @Test func guidedGenerationNeedsATokenByteTable() async throws {
        struct NoTable: TokenDecoding {
            func decode(_ ids: [Int32]) -> String { ByteDecoding().decode(ids) }
        }
        let model = AetherLanguageModel(generator: try await ByteChat.generator(decoder: NoTable()) { _ in [] })
        #expect(!model.capabilities.contains(.guidedGeneration))
        await #expect(throws: (any Error).self) { try await LanguageModelSession(model: model).respond(to: "q", generating: Look.self) }
    }

    /// Review #4: the remaining generation errors map to FoundationModels' where one corresponds.
    @Test func errorsMapToFoundationModels() {
        #expect(AetherLanguageModelExecutor.mapped(AetherError.generation(.decodingFailed("x")), schemaName: nil) is GeneratedContent.ParsingError)
        #expect(AetherLanguageModelExecutor.mapped(AetherError.generation(.constraintViolated("x")), schemaName: nil) is GeneratedContent.ParsingError)
        guard case .unsupportedGenerationGuide? = AetherLanguageModelExecutor.mapped(AetherError.generation(.unsupportedSchema("x")), schemaName: "S") as? LanguageModelError
        else { Issue.record("unsupportedSchema"); return }
        guard case .contextSizeExceeded? = AetherLanguageModelExecutor.mapped(AetherError.capacity(what: "kv", needed: 9, available: 8), schemaName: nil) as? LanguageModelError
        else { Issue.record("capacity"); return }
    }

    /// Review #5: two calls in one `<tool_call>` both run; a body cut off (here at the end of the reply) comes back as
    /// the model's text instead of failing a reply that already streamed.
    @Test func parallelAndMalformedToolCalls() async throws {
        let two = #"{"name": "weather", "arguments": {"city": "Paris"}}"# + "\n" + #"{"name": "weather", "arguments": {"city": "Rome"}}"#
        let model = try await Self.model { prompt in
            ByteChat.text(prompt).contains("<tool_response>")
                ? ByteVocab.tokens("Both sunny.")
                : [ByteVocab.toolOpen] + ByteVocab.tokens("\n\(two)\n") + [ByteVocab.toolClose]
        }
        let weather = Weather()
        let reply = try await LanguageModelSession(model: model, tools: [weather]).respond(to: "Paris and Rome?")
        #expect(reply.content == "Both sunny.")
        #expect(weather.calls.cities.withLock { $0 }.sorted() == ["Paris", "Rome"])

        let cut = try await Self.model { _ in ByteVocab.tokens("Sure.") + [ByteVocab.toolOpen] + ByteVocab.tokens(#"{"name": "weather", "argu"#) }
        let text = try await LanguageModelSession(model: cut, tools: [Weather()]).respond(to: "q")
        #expect(text.content == #"Sure.<tool_call>{"name": "weather", "argu</tool_call>"#)
    }

    /// Review #7: the arguments render back exactly as the model wrote them (FoundationModels re-serializes them), so
    /// the next turn reuses the generated tokens.
    @Test func toolArgumentsRenderBackAsGenerated() async throws {
        let log = PromptLog()
        let call = #"{"name": "weather", "arguments": {"city":"Paris"}}"#
        let model = try await Self.model(log: log) { prompt in
            ByteChat.text(prompt).contains("<tool_response>")
                ? ByteVocab.tokens("Sunny.")
                : [ByteVocab.toolOpen] + ByteVocab.tokens(call) + [ByteVocab.toolClose]
        }
        _ = try await LanguageModelSession(model: model, tools: [Weather()]).respond(to: "Paris?")
        #expect(ByteChat.text(log.all.last ?? []).contains(#""arguments": {"city":"Paris"}}"#))
    }
}

