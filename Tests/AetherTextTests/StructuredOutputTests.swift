import Foundation
import Testing
@testable import AetherText
import AetherRuntime
import AetherTestSupport

struct Slug: StructuredOutput {
    var slug: String
    static var outputSchema: OutputSchema {
        try! OutputSchema(json: Data(#"{"type": "object", "title": "Slug", "properties": {"slug": {"type": "string", "pattern": "[a-z]+"}}, "required": ["slug"], "x-order": ["slug"], "additionalProperties": false}"#.utf8))
    }
}

/// `ChatSession.respond(to:generating:)` on the byte-level fake (0.0.x `AetherStructuredOutputTests`): the reply is
/// constrained by the JSON-schema processor inside the Generator's loop and decoded into the type.
@Suite("Structured output (0.0.x)")
struct StructuredOutputTests {
    static func session(thinking: Bool? = nil, maxTokens: Int = 256, system: String? = nil,
                        _ script: @escaping @Sendable ([Int32]) -> [Int32]) async throws -> ChatSession {
        ChatSession(generator: try await ByteChat.generator(script),
                    configuration: .init(systemPrompt: system, parameters: GenerationParameters(maxTokens: maxTokens, thinking: thinking)))
    }

    @Test func respondDecodesIntoTheTypeAndKeepsTheJSON() async throws {
        let json = #"{"item": "scarf", "warm": true}"#
        let prompts = PromptLog()
        let session = try await Self.session(system: "Dress warmly.") { prompt in
            prompts.append(prompt)
            return ByteVocab.tokens(json)
        }
        let reply = try await session.respond(to: "One layer?", generating: Layer.self)
        #expect(reply == Layer(item: "scarf", warm: true))
        #expect(await session.history == [.user(ChatSession.withSchemaInstruction("One layer?", Layer.outputSchema)), .assistant(json)])
        let turn = ChatSession.withSchemaInstruction("One layer?", Layer.outputSchema)
        let expected = try ByteRenderer().render([.system("Dress warmly."), .user(turn)], options: RenderOptions(thinking: false))
        #expect(prompts.all.last == expected.tokens, "the schema is described in the user turn, thinking off")
        // Without the schema in the prompt, and without instructions.
        let bare = try await Self.session { prompt in
            prompts.append(prompt)
            return ByteVocab.tokens(json)
        }
        _ = try await bare.respond(to: "q", generating: Layer.self, includeSchemaInPrompt: false)
        #expect(prompts.all.last == (try ByteRenderer().render([.user("q")], options: RenderOptions(thinking: false)).tokens))
    }

    /// A model that strays from the layout (no spaces, a wrong key, an out-of-range value) is held to the schema: the
    /// reply is valid for the type, or, if it runs out of tokens first, a decoding failure, never invalid JSON.
    @Test func aStrayingModelIsHeldToTheSchema() async throws {
        for script in [#"{"item":"scarf","warm":true}"#, #"{"thing": "scarf", "warm": maybe}"#, #"["scarf"]"#] {
            let session = try await Self.session(maxTokens: 120) { _ in ByteVocab.tokens(script) }
            do {
                let reply = try await session.respond(to: "q", generating: Layer.self)
                #expect(!reply.item.contains("\u{FFFD}"), "\(script)")
            } catch AetherError.generation(.decodingFailed) {
                #expect(await session.history.isEmpty, "a reply that didn't decode isn't kept")
            }
        }
    }

    @Test func unsupportedSchemasAreRefusedBeforeGenerating() async throws {
        let prompts = PromptLog()
        let session = try await Self.session { prompt in
            prompts.append(prompt)
            return ByteVocab.tokens("x")
        }
        do {
            _ = try await session.respond(to: "q", generating: Slug.self)
            Issue.record("expected unsupportedSchema")
        } catch AetherError.generation(.unsupportedSchema(let reason)) {
            #expect(reason.contains("pattern"))
        }
        #expect(prompts.all.isEmpty)
    }

    @Test func reachingTheOutputLimitIsADecodingFailure() async throws {
        let session = try await Self.session(maxTokens: 10) { _ in ByteVocab.tokens(#"{"item": "a very long scarf indeed", "warm": true}"#) }
        await #expect(throws: AetherError.self) { try await session.respond(to: "q", generating: Layer.self) }
        #expect(await session.history.isEmpty)
    }

    @Test func reasoningComesFirstThenTheConstrainedValue() async throws {
        let session = try await Self.session(thinking: true) { _ in
            [ByteVocab.thinkStart] + ByteVocab.tokens("\nIt is cold.\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens(#"{"item": "coat", "warm": true}"#)
        }
        let reply = try await session.respond(to: "q", generating: Layer.self)
        #expect(reply == Layer(item: "coat", warm: true))
        #expect(await session.history.last == .assistant(#"{"item": "coat", "warm": true}"#))
    }

    @Test func nestedTypesArraysEnumsAndGuidesDecode() async throws {
        let json = #"{"name": "Mara", "size": 7, "price": 12.5, "colours": ["red", "blue"], "layers": [{"item": "coat", "warm": true}, {"item": "vest", "warm": false}], "grade": "A"}"#
        let session = try await Self.session { _ in ByteVocab.tokens(json) }
        let outfit = try await session.respond(to: "q", generating: Outfit.self)
        #expect(outfit == Outfit(name: "Mara", size: 7, price: 12.5, colours: [.red, .blue],
                                 layers: [Layer(item: "coat", warm: true), Layer(item: "vest", warm: false)], note: nil, grade: "A"))
    }

    /// A plain `Decodable` with a hand-written schema, and a sampled (not greedy) constrained reply.
    @Test func decodableWithASchemaAndSampling() async throws {
        struct Point: Decodable, Equatable { var x: Int; var y: Int }
        let schema = OutputSchema.object(name: "Point", properties: [.init(name: "x", type: Int.self, guides: [.range(0...9)]),
                                                                     .init(name: "y", type: Int.self, guides: [.range(0...9)])])
        let session = ChatSession(generator: try await ByteChat.generator { _ in ByteVocab.tokens(#"{"x": 3, "y": 12}"#) },
                                  configuration: .init(parameters: GenerationParameters(maxTokens: 64, sampling: SamplingParameters(temperature: 0.5, seed: 4))))
        let point = try await session.respond(to: "q", generating: Point.self, schema: schema)
        #expect((0...9).contains(point.x) && (0...9).contains(point.y))
    }

    // MARK: Review regressions

    /// Review #1: tags are not parsed inside the constrained value. A string holding `<think>…</think>` or a
    /// `<tool_call>` (spelled in bytes, or as Qwen's non-special `<tool_call>` token) stays in the value.
    @Test func tagTextInsideAValueStaysInTheValue() async throws {
        for item in ["a<think>x</think>b", "<tool_call>x</tool_call>", "<think>"] {
            let json = #"{"item": ""# + item + #"", "warm": true}"#
            let session = try await Self.session { _ in ByteVocab.tokens(json) }
            #expect(try await session.respond(to: "q", generating: Layer.self) == Layer(item: item, warm: true), "\(item)")
        }
        let token = try await Self.session { _ in
            ByteVocab.tokens(#"{"item": "a"#) + [ByteVocab.toolOpen] + ByteVocab.tokens(#"b", "warm": false}"#)
        }
        #expect(try await token.respond(to: "q", generating: Layer.self) == Layer(item: "a<tool_call>b", warm: false))
    }

    /// Review #2a: a template that opens thinking regardless (`toggle: none`) starts the reply inside a block; the
    /// value is constrained after it, not sent to `.thinking` whole.
    @Test func aPromptThatOpensThinkingIsFollowed() async throws {
        let generator = try await ByteChat.generator(renderer: ByteRenderer(alwaysThinks: true)) { _ in
            ByteVocab.tokens("It is cold.\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens(#"{"item":"coat","warm":maybe}"#)
        }
        let session = ChatSession(generator: generator, configuration: .init(parameters: GenerationParameters(maxTokens: 120)))
        let reply = try await session.respond(to: "q", generating: Layer.self)
        #expect(!reply.item.isEmpty)
    }

    /// Review #2b: with thinking on, a model that answers without opening `<think>` is still constrained (its
    /// script's `maybe` would not decode unconstrained).
    @Test func thinkingAllowedButSkippedIsStillConstrained() async throws {
        let session = try await Self.session(thinking: true, maxTokens: 120) { _ in ByteVocab.tokens(#"{"item": "scarf", "warm": maybe}"#) }
        let reply = try await session.respond(to: "q", generating: Layer.self)
        #expect(reply.item == "scarf")
    }

    /// Review #3: a value that completes exactly at `maxTokens` is a value, not a decoding failure.
    @Test func aValueEndingExactlyAtTheLimitDecodes() async throws {
        let json = #"{"item": "scarf", "warm": true}"#
        let session = try await Self.session(maxTokens: json.utf8.count) { _ in ByteVocab.tokens(json) }
        #expect(try await session.respond(to: "q", generating: Layer.self) == Layer(item: "scarf", warm: true))
    }

    /// Review #6: the schema goes in the user turn, so a plain turn after a structured one reuses the whole history.
    @Test func structuredAndPlainTurnsKeepPrefixReuse() async throws {
        let session = try await Self.session(system: "Be brief.") { prompt in
            ByteChat.text(prompt).hasSuffix("JSON schema") ? [] : ByteVocab.tokens(#"{"item": "hat", "warm": true}"#)
        }
        _ = try await session.respond(to: "one", generating: Layer.self)
        var summary: GenerationSummary?
        for try await event in await session.respond(to: "two") { if case .finished(let s) = event { summary = s } }
        let history = await session.history
        // Everything up to turn 1's reply (its generation prompt's empty think block is not in the history).
        let previous = try ByteRenderer().render([.system("Be brief."), history[0]], options: RenderOptions())
        #expect((summary?.reusedTokens ?? 0) >= previous.tokens.count, "turn 2 reuses turn 1's prompt")
    }
}
