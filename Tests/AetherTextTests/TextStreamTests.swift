import Foundation
import Testing
@testable import AetherText
import AetherRuntime
import AetherTestSupport

/// The 0.0.x text-stream corpus (`LanguageTextStreamTests`), run through the Generator's own output stage (token
/// bytes → grapheme hold → tag parser → stop sequences on the response text), on the byte-level fake vocabulary.
@Suite("Text stream (0.0.x corpus)")
struct TextStreamTests {
    struct Result {
        var content = ""
        var reasoning = ""
        var events: [GenerationEvent] = []
        var stop: GenerationSummary.StopReason?
        var stopped: Bool { if case .stopSequence? = stop { true } else { false } }
    }

    /// Feeds `tokens` one at a time (no stop token, no limit), then finishes.
    static func stream(_ tokens: [Int32], profile: OutputProfile = .plain, startsInThinking: Bool = false, stops: [String] = [],
                       decoder: any TokenDecoding = ByteDecoding()) async throws -> Result {
        let (events, continuation) = AsyncThrowingStream.makeStream(of: GenerationEvent.self)
        var state = Generator.LoopState(
            text: TokenText(decoder: decoder, keepSpecial: Set(profile.tags.flatMap { [$0.open, $0.close] })),
            parser: TagParser(profile: profile, startsInThinking: startsInThinking), stopper: StopMatcher(stops),
            stopTokens: [], maxTokens: .max, continuation: continuation)
        for token in tokens where state.accept(token) { break }
        state.finish()
        continuation.finish()
        var result = Result(stop: state.reason)
        for try await event in events {
            result.events.append(event)
            switch event {
            case .text(let s): result.content += s
            case .thinking(let s): result.reasoning += s
            default: break
            }
        }
        return result
    }

    /// Characters emitted so far must be the final text's first characters, unchanged; no delta shows U+FFFD.
    static func expectStable(_ result: Result, final: String, sourceLocation: SourceLocation = #_sourceLocation) {
        var emitted = ""
        for case .text(let delta) in result.events {
            #expect(!delta.contains("\u{FFFD}"), sourceLocation: sourceLocation)
            emitted += delta
            #expect(Array(emitted) == Array(final.prefix(emitted.count)), "emitted characters changed", sourceLocation: sourceLocation)
        }
    }

    static let thinking = OutputProfile(tags: [.init(open: "<think>", close: "</think>", kind: .thinking)])

    @Test func multiByteScalarsSplitAcrossTokensNeverLeak() async throws {
        let text = "Café — naïve 日本語 🌙 done"
        let result = try await Self.stream(ByteVocab.tokens(text))
        #expect(result.content == text && result.reasoning.isEmpty && !result.stopped)
        Self.expectStable(result, final: text)
    }

    @Test func graphemeClustersWaitUntilTheyEnd() async throws {
        // e + combining acute; a ZWJ family; two flags; Hangul jamo L V T; a skin-tone modifier.
        let clusters = ["e\u{301}", "👨\u{200D}👩\u{200D}👧", "🇺🇸", "🇫🇷", "\u{1100}\u{1161}\u{11A8}", "👋\u{1F3FD}", "x"]
        let text = clusters.joined()
        let result = try await Self.stream(ByteVocab.tokens(text))
        #expect(result.content == text)
        Self.expectStable(result, final: text)
        // The accent arrives after "e" was decoded: "e" must not have been emitted on its own.
        let accented = try await Self.stream(ByteVocab.tokens("e\u{301}!"))
        #expect(!accented.events.contains(.text("e")) && accented.content == "e\u{301}!")
        // Each flag is emitted whole: never one regional indicator alone.
        for case .text(let delta) in result.events {
            #expect(!delta.unicodeScalars.contains { $0 == "\u{1F1FA}" } || delta.contains("🇺🇸"))
        }
    }

    @Test func specialTokensAndPaddingRowsAddNoText() async throws {
        let tokens = ByteVocab.tokens("Hi") + [ByteVocab.imEnd, ByteVocab.imStart, 262] + ByteVocab.tokens("!")
        #expect(try await Self.stream(tokens).content == "Hi!")
    }

    @Test func malformedBytesDecodeAsTheTokenizerDoes() async throws {
        let bytes: [UInt8] = [0x41, 0x80, 0xE6, 0x97, 0x42, 0xF0, 0x9F]  // stray continuation, truncated 3- and 4-byte sequences
        let result = try await Self.stream(bytes.map(Int32.init))
        #expect(result.content == String(decoding: bytes, as: UTF8.self))
    }

    @Test func stopSequencesEndTheReplyBeforeThemAndNeverFlash() async throws {
        let result = try await Self.stream(ByteVocab.tokens("The answer is 42.\n\nUser: next"), stops: ["\n\nUser:"])
        #expect(result.content == "The answer is 42." && result.stopped)
        #expect(!result.content.contains("\n"), "the stop's prefix was never emitted")
        // A prefix that never completes is released.
        let partial = try await Self.stream(ByteVocab.tokens("a STOx and STOP"), stops: ["STOP"])
        #expect(partial.content == "a STOx and " && partial.stopped)
        let unfinished = try await Self.stream(ByteVocab.tokens("ends with STO"), stops: ["STOP"])
        #expect(unfinished.content == "ends with STO" && !unfinished.stopped)
        // The earliest of several stops wins; text after a stop is ignored.
        let several = try await Self.stream(ByteVocab.tokens("one|two#three"), stops: ["#", "|"])
        #expect(several.content == "one")
    }

    @Test func reasoningGoesToItsOwnChannelWithoutItsFraming() async throws {
        let tokens = [ByteVocab.thinkStart] + ByteVocab.tokens("\nThe user wants a colour.\n\n") + [ByteVocab.thinkEnd]
            + ByteVocab.tokens("\n\nTeal.")
        let result = try await Self.stream(tokens, profile: Self.thinking)
        #expect(result.reasoning == "The user wants a colour." && result.content == "Teal.")
        // A reply that begins inside a block the prompt opened.
        let open = try await Self.stream(ByteVocab.tokens("\nhmm\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens("\nOK"),
                                         profile: Self.thinking, startsInThinking: true)
        #expect(open.reasoning == "hmm" && open.content == "OK")
        // Newlines inside the reasoning stay; only the framing goes.
        let inner = try await Self.stream([ByteVocab.thinkStart] + ByteVocab.tokens("\na\n\nb\n") + [ByteVocab.thinkEnd] + ByteVocab.tokens("c"),
                                          profile: Self.thinking)
        #expect(inner.reasoning == "a\n\nb" && inner.content == "c")
        // Without delimiters (a model without reasoning) the markers are just text, as the tokenizer decodes them.
        #expect(try await Self.stream([ByteVocab.thinkStart] + ByteVocab.tokens("x")).content == "<think>x")
        // Stop sequences apply to the answer, not to reasoning.
        let stopped = try await Self.stream(
            [ByteVocab.thinkStart] + ByteVocab.tokens("a#b") + [ByteVocab.thinkEnd] + ByteVocab.tokens("c#d"), profile: Self.thinking, stops: ["#"])
        #expect(stopped.reasoning == "a#b" && stopped.content == "c")
    }

    @Test func toolCallsAreCutOutWhole() async throws {
        let tokens = ByteVocab.tokens("Let me check.") + [ByteVocab.toolOpen] + ByteVocab.tokens("\n{\"name\": \"w\", \"arguments\": {}}\n")
            + [ByteVocab.toolClose]
        let result = try await Self.stream(tokens, profile: ByteVocab.profile)
        #expect(result.content == "Let me check.")
        #expect(result.events.contains(.toolCall("\n{\"name\": \"w\", \"arguments\": {}}\n")))
    }

    @Test func fuzzedStreamsConcatenateToTheWholeText() async throws {
        let pieces = ["a", "Z", " ", "\n", "é", "e\u{301}", "日", "🌙", "👨\u{200D}👩\u{200D}👧", "🇯🇵", "\u{1100}\u{1161}", "\r\n", "👋\u{1F3FD}"]
        var generator = SeededRandom(seed: 2026)
        for _ in 0..<300 {
            let text = (0..<Int(generator.next() % 24 + 1)).map { _ in pieces[Int(generator.next() % UInt64(pieces.count))] }.joined()
            let result = try await Self.stream(ByteVocab.tokens(text))
            #expect(result.content == text, "\(text.unicodeScalars.map { String($0.value, radix: 16) })")
            Self.expectStable(result, final: text)
        }
    }

    /// Without a byte table (a SentencePiece tokenizer) the detokenizer path holds graphemes back the same way.
    @Test func theDetokenizerPathIsGraphemeSafeToo() async throws {
        struct NoTable: TokenDecoding {
            func decode(_ ids: [Int32]) -> String { String(decoding: ids.map { UInt8($0) }, as: UTF8.self) }
        }
        let text = "e\u{301} 🇺🇸🇫🇷 👨\u{200D}👩\u{200D}👧!"
        let result = try await Self.stream(ByteVocab.tokens(text), decoder: NoTable())
        #expect(result.content == text)
        Self.expectStable(result, final: text)
        #expect(!result.events.contains(.text("e")))
    }

    @Test func completeLengthFindsTheLastWholeScalar() {
        #expect(TokenText.completeLength([]) == 0)
        #expect(TokenText.completeLength([0x41]) == 1)
        #expect(TokenText.completeLength([0x41, 0xE6]) == 1)
        #expect(TokenText.completeLength([0x41, 0xE6, 0x97]) == 1)
        #expect(TokenText.completeLength([0x41, 0xE6, 0x97, 0xA5]) == 4)
        #expect(TokenText.completeLength([0xF0, 0x9F, 0x8C]) == 0)
        #expect(TokenText.completeLength([0x80, 0x80]) == 2)
    }
}

/// The 0.0.x token byte table tests (`LanguageTokenBytesTests`), on synthetic `tokenizer.json` files.
@Suite("Token bytes (0.0.x)")
struct TokenBytesTests {
    @Test func byteLevelTokensDecodeToTheirBytes() throws {
        let moon = Array("🌙".utf8)  // F0 9F 8C 99
        let pieces: [[UInt8]] = [Array("Hello".utf8), Array(" world".utf8), [0x0A], Array("é".utf8), Array(" 🌙".utf8),
                                 Array(moon.prefix(2)), Array(moon.suffix(2))]  // a scalar split across two tokens
        var vocab: [String: Int] = [:]
        for (id, piece) in pieces.enumerated() { vocab[ByteVocab.spelled(piece)] = id }
        let table = try TokenBytes(tokenizerJSON: ByteVocab.tokenizerJSON(vocab: vocab, added: [(8, "<|im_end|>", true), (9, "<think>", false)]))
        #expect(table.count == 10)
        for (id, piece) in pieces.enumerated() { #expect(table.bytes(of: id).map(Array.init) == piece, "token \(id)") }
        #expect(table.bytes(of: 7) == nil && table.bytes(of: 10) == nil && table.bytes(of: -1) == nil)
        #expect(table.bytes(of: 8).map(Array.init) == Array("<|im_end|>".utf8) && table.isSpecial(8))
        #expect(table.bytes(of: 9).map(Array.init) == Array("<think>".utf8) && !table.isSpecial(9))
        #expect(String(decoding: table.bytes(of: [0, 1, 8, 2]), as: UTF8.self) == "Hello world\n")
        #expect(String(decoding: table.bytes(of: [5, 6]), as: UTF8.self) == "🌙")
        #expect(String(decoding: table.bytes(of: [9, 0, 8], skipSpecialTokens: false), as: UTF8.self) == "<think>Hello<|im_end|>")
    }

    @Test func everyByteRoundTrips() throws {
        var vocab: [String: Int] = [:]
        for byte in 0...255 { vocab[ByteVocab.spelled([UInt8(byte)])] = byte }
        let table = try TokenBytes(tokenizerJSON: ByteVocab.tokenizerJSON(vocab: vocab, added: [], decoder: [
            "type": "Sequence", "decoders": [["type": "Replace"], ["type": "ByteLevel"]],
        ]))
        #expect((0...255).allSatisfy { table.bytes(of: $0).map(Array.init) == [UInt8($0)] })
        #expect(TokenBytes.unicodeToByte.count == 256)
    }

    @Test func nonByteLevelTokenizersAreRefused() throws {
        let data = try ByteVocab.tokenizerJSON(vocab: ["▁Hello": 0], added: [], decoder: ["type": "Metaspace"])
        #expect(throws: AetherError.self) { try TokenBytes(tokenizerJSON: data) }
    }
}

@Suite("AetherText review regressions (port-language)")
struct PortReviewRegressionTests {
    /// Review #12: without a byte table, special tokens add no text either (the manifest's tags excepted).
    @Test func theDetokenizerPathSkipsSpecialTokens() async throws {
        struct Specials: TokenDecoding {
            func decode(_ ids: [Int32]) -> String {
                ids.map { $0 == 300 ? "<eos>" : $0 == 301 ? "<think>" : $0 == 302 ? "</think>" : String(decoding: [UInt8($0)], as: UTF8.self) }.joined()
            }
            func isSpecial(_ id: Int32) -> Bool { id >= 300 }
            func tokenID(_ text: String) -> Int32? { ["<think>": 301, "</think>": 302][text] }
        }
        let tokens = ByteVocab.tokens("Hi") + [300] + ByteVocab.tokens("!")
        #expect(try await TextStreamTests.stream(tokens, decoder: Specials()).content == "Hi!")
        let kept = try await TextStreamTests.stream([301] + ByteVocab.tokens("x") + [302] + ByteVocab.tokens("y"), profile: TextStreamTests.thinking, decoder: Specials())
        #expect(kept.reasoning == "x" && kept.content == "y", "a special that is a manifest tag stays visible to the tag parser")
    }

    /// Review #7: tool arguments go in as text where the template writes them verbatim, else as an object, so
    /// `| tojson` doesn't encode a string twice.
    @Test func toolArgumentsMatchWhatTheTemplateExpects() throws {
        let message = ChatMessage.assistant("", toolCalls: [ToolCall(id: "1", name: "f", arguments: #"{"b": 2, "a": 1}"#)])
        func arguments(_ asText: Bool) -> (any Sendable)? {
            let calls = message.templateMessage(argumentsAsText: asText)["tool_calls"] as? [[String: any Sendable]]
            return (calls?.first?["function"] as? [String: any Sendable])?["arguments"]
        }
        #expect(arguments(true) as? String == #"{"b": 2, "a": 1}"#)
        let object = try #require(arguments(false) as? [String: any Sendable])
        #expect(object["a"] as? Int == 1 && object["b"] as? Int == 2)
    }

    /// Review #7: Hermes calls keep their arguments byte for byte, and a body may hold several calls.
    @Test func hermesKeepsRawArgumentsAndParallelCalls() throws {
        let dialect = HermesJSONDialect()
        let one = try dialect.parse("\n{\"name\": \"weather\", \"arguments\": {\"city\":\"Paris\",  \"days\": 2}}\n")
        #expect(one.map(\.name) == ["weather"] && one[0].arguments == #"{"city":"Paris",  "days": 2}"#)
        let two = try dialect.parse(#"{"name": "a", "arguments": {}}"# + "\n" + #"{"name": "b", "arguments": {"x": 1}}"#)
        #expect(two.map(\.name) == ["a", "b"] && two[1].arguments == #"{"x": 1}"#)
        let array = try dialect.parse(#"[{"name": "a"}, {"name": "b", "arguments": "{\"y\": 2}"}]"#)
        #expect(array.map(\.name) == ["a", "b"] && array[0].arguments == "{}" && array[1].arguments == #"{"y": 2}"#)
        for bad in ["", "not json", #"{"arguments": {}}"#, #"{"name": "f", "arguments": [1]}"#, #"{"name": "weather", "argu"#] {
            #expect(throws: AetherError.self, "\(bad)") { try dialect.parse(bad) }
        }
    }

    /// Qwen3.5/3.6's XML calls, as their template renders them: values as JSON strings in order, several functions,
    /// and the same body through `hermes-json` (what their older exports declare).
    @Test func qwenXMLCallsAndTheHermesFallback() throws {
        let body = "\n<function=replace_phrase>\n<parameter=id>\natelier.seed.x\n</parameter>\n<parameter=new_phrase>\n"
            + "a \"black\" sleeve\nover two lines\n</parameter>\n</function>\n"
        for dialect in [QwenXMLDialect() as any ToolDialect, HermesJSONDialect()] {
            let calls = try dialect.parse(body)
            #expect(calls.map(\.name) == ["replace_phrase"])
            #expect(calls[0].arguments == #"{"id": "atelier.seed.x", "new_phrase": "a \"black\" sleeve\nover two lines"}"#)
        }
        let two = try QwenXMLDialect().parse("<function=a>\n</function>\n<function=b>\n<parameter=x>\n1\n</parameter>\n</function>")
        #expect(two.map(\.name) == ["a", "b"] && two[0].arguments == "{}" && two[1].arguments == #"{"x": "1"}"#)
        for bad in ["", "{\"name\": \"f\"}", "<function=f>\n<parameter=x>\n1\n</function>", "<function=f>", "<function=>\n</function>"] {
            #expect(throws: AetherError.self, "\(bad)") { try QwenXMLDialect().parse(bad) }
        }
        #expect(try ToolDialects.make("qwen-xml").id == "qwen-xml")
    }
}

