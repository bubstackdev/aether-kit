@testable import AetherLanguage
@testable import AetherLanguageFoundationModels
import AetherLanguageRuntime
import Foundation
import FoundationModels
import Testing

@Generable private struct Look {
    @Guide(description: "a short look") var look: String
}

@Generable private struct Tagged {
    @Guide(.pattern(/[a-z]+/)) var slug: String
}

/// The adapter through a real FoundationModels `LanguageModelSession` over the fake backend.
struct AetherLanguageFoundationModelsTests {
    @Test func languageModelSessionRunsOnAKitModel() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("Teal, with a clasp 🌙.") }
        let model = FakeModel.make(supportsReasoning: false, backend: backend)
        let session = LanguageModelSession(model: model, instructions: "Describe looks.")
        let reply = try await session.respond(to: "Her coat?")
        #expect(reply.content == "Teal, with a clasp 🌙.")
        let expected = try FakeCodec().render([.system("Describe looks."), .user("Her coat?")], thinking: false)
        #expect(await backend.prompts == [expected], "the transcript renders with the model's template")
        #expect(await backend.requests.first?.sampling == .greedy)

        // The next turn renders the first, and streaming delivers the same text.
        var last = ""
        for try await snapshot in session.streamResponse(to: "And her boots?") { last = snapshot.content }
        #expect(last == "Teal, with a clasp 🌙.")
        let second = try FakeCodec().render([.system("Describe looks."), .user("Her coat?"), .assistant("Teal, with a clasp 🌙."),
                                             .user("And her boots?")], thinking: false)
        #expect(await backend.prompts.last == second)
    }

    @Test func optionsMapAndReasoningLevelsTurnThinkingOn() async throws {
        let backend = FakeBackend { _ in
            [FakeVocabulary.thinkStart] + FakeVocabulary.tokens("\nweighing it\n") + [FakeVocabulary.thinkEnd] + FakeVocabulary.tokens("\n\nYes.")
        }
        let model = FakeModel.make(backend: backend)
        #expect(model.capabilities.contains(.reasoning) && model.capabilities.contains(.guidedGeneration))
        let session = LanguageModelSession(model: model)
        _ = try await session.respond(to: "q", options: GenerationOptions(samplingMode: .random(top: 40, seed: 7), temperature: 0.7,
                                                                           maximumResponseTokens: 32))
        _ = try await session.respond(to: "q", options: GenerationOptions(samplingMode: .random(probabilityThreshold: 0.9, seed: 3)))
        let requests = await backend.requests
        #expect(requests.map(\.sampling) == [.random(temperature: 0.7, topK: 40, seed: 7), .random(temperature: 1, topP: 0.9, seed: 3)])
        #expect(requests.map(\.maximumTokens).first == 32)

        let mapped = AetherLanguageExecutor.options(GenerationOptions(), ContextOptions(reasoningLevel: .light), supportsReasoning: true)
        #expect(mapped.reasoning == .on && mapped.samplingMode == nil)
        #expect(AetherLanguageExecutor.options(GenerationOptions(), ContextOptions(), supportsReasoning: true).reasoning == nil)
        #expect(AetherLanguageExecutor.options(GenerationOptions(), ContextOptions(reasoningLevel: .deep), supportsReasoning: false).reasoning == nil)

        let thinking = try await LanguageModelSession(model: model)
            .respond(to: "q", options: GenerationOptions(), contextOptions: ContextOptions(reasoningLevel: .moderate))
        #expect(thinking.content == "Yes.")
        let prompt = await backend.prompts.last ?? []
        #expect(prompt == (try FakeCodec().render([.user("q")], thinking: true)), "a reasoning level renders thinking on")
    }

    /// FoundationModels' @Generable through the adapter: the type's schema constrains generation, FoundationModels
    /// decodes the JSON, and the schema is described in the system message.
    @Test func generableRequestsAreConstrainedAndDecode() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens(#"{"look": "teal coat"}"#) }
        let session = LanguageModelSession(model: FakeModel.make(supportsReasoning: false, backend: backend), instructions: "Be brief.")
        let reply = try await session.respond(to: "q", generating: Look.self)
        #expect(reply.content.look == "teal coat")
        let request = try #require(await backend.requests.first)
        #expect(request.constraint != nil)
        let schema = try AetherGenerationSchema(json: try JSONEncoder().encode(Look.generationSchema))
        let structured = LanguageStructuredOutput(schema: schema, includeSchemaInPrompt: true)
        #expect(request.prompt == (try FakeCodec().render([.system("Be brief.\n\n" + structured.instruction), .user("q")], thinking: false)))
        var last: Look.PartiallyGenerated?
        for try await snapshot in session.streamResponse(to: "again", generating: Look.self) { last = snapshot.content }
        #expect(last?.look == "teal coat")
    }

    @Test func unsupportedGuidesBecomeFoundationModelsError() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens(#"{"slug": "x"}"#) }
        let session = LanguageModelSession(model: FakeModel.make(backend: backend))
        do {
            _ = try await session.respond(to: "q", generating: Tagged.self)
            Issue.record("expected an unsupported guide")
        } catch let error as LanguageModelError {
            guard case .unsupportedGenerationGuide = error else { Issue.record("unexpected \(error)"); return }
        }
        #expect(await backend.requests.isEmpty)
    }

    @Test func contextOverflowBecomesFoundationModelsError() async throws {
        let backend = FakeBackend(context: 40) { _ in FakeVocabulary.tokens("x") }
        let session = LanguageModelSession(model: FakeModel.make(context: 40, backend: backend))
        do {
            _ = try await session.respond(to: String(repeating: "long ", count: 20))
            Issue.record("expected a context error")
        } catch let error as LanguageModelError {
            guard case .contextSizeExceeded(let detail) = error else { Issue.record("unexpected \(error)"); return }
            #expect(detail.contextSize == 40)
        }
    }

    @Test func copiesOfAModelShareAConfiguration() {
        let model = FakeModel.make()
        let copy = model
        #expect(model.executorConfiguration == copy.executorConfiguration)
        #expect(model.executorConfiguration != FakeModel.make().executorConfiguration)
    }
}
