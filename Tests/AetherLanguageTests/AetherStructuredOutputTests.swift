@testable import AetherLanguage
import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation
import Testing

struct Slug: AetherGenerable {
    var slug: String
    static var generationSchema: AetherGenerationSchema {
        try! AetherGenerationSchema(json: Data(#"{"type": "object", "title": "Slug", "properties": {"slug": {"type": "string", "pattern": "[a-z]+"}}, "required": ["slug"], "x-order": ["slug"], "additionalProperties": false}"#.utf8))
    }
}

struct AetherStructuredOutputTests {
    @Test func respondDecodesIntoTheTypeAndKeepsTheJSON() async throws {
        let json = #"{"item": "scarf", "warm": true}"#
        let backend = FakeBackend { _ in FakeVocabulary.tokens(json) }
        let session = AetherLanguageSession(model: FakeModel.make(supportsReasoning: false, backend: backend), instructions: "Dress warmly.")
        let reply = try await session.respond(to: "One layer?", generating: Layer.self)
        #expect(reply.content == Layer(item: "scarf", warm: true) && reply.stopReason == .endOfTurn)
        #expect(await session.transcript.entries == [.instructions("Dress warmly."), .prompt("One layer?"), .response(json)])
        let structured = LanguageStructuredOutput(schema: Layer.generationSchema, includeSchemaInPrompt: true)
        let expected = try FakeCodec().render([.system("Dress warmly.\n\n" + structured.instruction), .user("One layer?")], thinking: false)
        #expect(await backend.prompts == [expected], "the schema is described in the system message")
        #expect(await backend.requests.first?.constraint != nil)
        // Without the schema in the prompt, and without instructions.
        let bare = AetherLanguageSession(model: FakeModel.make(supportsReasoning: false, backend: backend))
        _ = try await bare.respond(to: "q", generating: Layer.self, includeSchemaInPrompt: false)
        #expect(await backend.prompts.last == (try FakeCodec().render([.user("q")], thinking: false)))
    }

    /// A model that strays from the layout (no spaces, a wrong key, an out-of-range number) is held to the schema: the
    /// reply is valid for the type, or, if it runs out of tokens first, a decoding failure, never invalid JSON.
    @Test func aStrayingModelIsHeldToTheSchema() async throws {
        for script in [#"{"item":"scarf","warm":true}"#, #"{"thing": "scarf", "warm": maybe}"#, #"["scarf"]"#] {
            let backend = FakeBackend { _ in FakeVocabulary.tokens(script) }
            let session = AetherLanguageSession(model: FakeModel.make(supportsReasoning: false, backend: backend))
            do {
                let reply = try await session.respond(to: "q", generating: Layer.self, options: .init(maximumResponseTokens: 120))
                #expect(!reply.content.item.contains("\u{FFFD}"), "\(script)")
            } catch AetherLanguageError.decodingFailure {
                #expect(await session.transcript.entries.isEmpty, "a reply that didn't decode isn't kept")
            }
        }
    }

    @Test func unsupportedSchemasAreRefusedBeforeGenerating() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens("x") }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        do {
            _ = try await session.respond(to: "q", generating: Slug.self)
            Issue.record("expected unsupportedSchema")
        } catch AetherLanguageError.unsupportedSchema(let reason) {
            #expect(reason.contains("pattern"))
        }
        #expect(await backend.requests.isEmpty)
    }

    @Test func reachingTheOutputLimitIsADecodingFailure() async throws {
        let backend = FakeBackend { _ in FakeVocabulary.tokens(#"{"item": "a very long scarf indeed", "warm": true}"#) }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        await #expect(throws: AetherLanguageError.self) {
            try await session.respond(to: "q", generating: Layer.self, options: .init(maximumResponseTokens: 10))
        }
        #expect(await session.transcript.entries.isEmpty)
    }

    @Test func reasoningComesFirstThenTheConstrainedValue() async throws {
        let backend = FakeBackend { _ in
            [FakeVocabulary.thinkStart] + FakeVocabulary.tokens("\nIt is cold.\n") + [FakeVocabulary.thinkEnd]
                + FakeVocabulary.tokens(#"{"item": "coat", "warm": true}"#)
        }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let reply = try await session.respond(to: "q", generating: Layer.self, options: .init(reasoning: .on))
        #expect(reply.content == Layer(item: "coat", warm: true) && reply.reasoning == "It is cold.")
    }

    @Test func nestedTypesArraysEnumsAndGuidesDecode() async throws {
        let json = #"{"name": "Mara", "size": 7, "price": 12.5, "colours": ["red", "blue"], "layers": [{"item": "coat", "warm": true}, {"item": "vest", "warm": false}], "note": null, "grade": "A"}"#
        let backend = FakeBackend { _ in FakeVocabulary.tokens(json) }
        let session = AetherLanguageSession(model: FakeModel.make(backend: backend))
        let outfit = try await session.respond(to: "q", generating: Outfit.self).content
        #expect(outfit == Outfit(name: "Mara", size: 7, price: 12.5, colours: [.red, .blue],
                                 layers: [Layer(item: "coat", warm: true), Layer(item: "vest", warm: false)], note: nil, grade: "A"))
    }
}
