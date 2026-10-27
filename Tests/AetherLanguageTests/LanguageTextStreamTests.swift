@testable import AetherLanguage
import AetherLanguageRuntime
import Testing

private let delimiters = LanguageTextStream.ReasoningDelimiters(start: FakeVocabulary.thinkStart, end: FakeVocabulary.thinkEnd)

/// Feeds `tokens` one at a time and checks the streaming invariants after every token: no replacement characters
/// unless the input has malformed bytes, and emitted text only ever grows by whole characters that never change.
private func stream(_ tokens: [Int], delimiters: LanguageTextStream.ReasoningDelimiters? = nil, startsInReasoning: Bool = false,
                    stops: [String] = [], expectWellFormed: Bool = true, sourceLocation: SourceLocation = #_sourceLocation)
    -> (content: String, reasoning: String, deltas: [LanguageTextStream.Delta], stopped: Bool) {
    var decoder = LanguageTextStream(bytes: FakeVocabulary.bytes, delimiters: delimiters, startsInReasoning: startsInReasoning,
                                     stopSequences: stops)
    var deltas: [LanguageTextStream.Delta] = []
    var content = "", reasoning = ""
    for token in tokens {
        let delta = decoder.append(token)
        deltas.append(delta)
        if expectWellFormed {
            #expect(!delta.content.contains("\u{FFFD}") && !delta.reasoning.contains("\u{FFFD}"), sourceLocation: sourceLocation)
        }
        content += delta.content
        reasoning += delta.reasoning
        #expect(content == decoder.content && reasoning == decoder.reasoning, sourceLocation: sourceLocation)
    }
    let last = decoder.finish()
    deltas.append(last)
    return (content + last.content, reasoning + last.reasoning, deltas, decoder.stopped)
}

/// Characters emitted so far must be the final text's first characters, unchanged.
private func expectStable(_ deltas: [LanguageTextStream.Delta], final: String, sourceLocation: SourceLocation = #_sourceLocation) {
    var emitted = ""
    for delta in deltas {
        emitted += delta.content
        #expect(Array(emitted) == Array(final.prefix(emitted.count)), "emitted characters changed", sourceLocation: sourceLocation)
    }
}

struct LanguageTextStreamTests {
    @Test func multiByteScalarsSplitAcrossTokensNeverLeak() {
        let text = "Café — naïve 日本語 🌙 done"
        let result = stream(FakeVocabulary.tokens(text))
        #expect(result.content == text && result.reasoning.isEmpty && !result.stopped)
        expectStable(result.deltas, final: text)
    }

    @Test func graphemeClustersWaitUntilTheyEnd() {
        // e + combining acute; a ZWJ family; two flags; Hangul jamo L V T; a skin-tone modifier.
        let clusters = ["e\u{301}", "👨\u{200D}👩\u{200D}👧", "🇺🇸", "🇫🇷", "\u{1100}\u{1161}\u{11A8}", "👋\u{1F3FD}", "x"]
        let text = clusters.joined()
        let result = stream(FakeVocabulary.tokens(text))
        #expect(result.content == text)
        expectStable(result.deltas, final: text)
        // The accent arrives after "e" was decoded: "e" must not have been emitted on its own.
        let accented = stream(FakeVocabulary.tokens("e\u{301}!"))
        #expect(accented.deltas.allSatisfy { $0.content != "e" } && accented.content == "e\u{301}!")
        // Each flag is emitted whole: never one regional indicator alone.
        #expect(result.deltas.allSatisfy { !$0.content.unicodeScalars.contains { $0 == "\u{1F1FA}" } || $0.content.contains("🇺🇸") })
    }

    @Test func specialTokensAndPaddingRowsAddNoText() {
        let tokens = FakeVocabulary.tokens("Hi") + [FakeVocabulary.imEnd, FakeVocabulary.imStart, 262] + FakeVocabulary.tokens("!")
        #expect(stream(tokens).content == "Hi!")
    }

    @Test func malformedBytesDecodeAsTheTokenizerDoes() {
        let bytes: [UInt8] = [0x41, 0x80, 0xE6, 0x97, 0x42, 0xF0, 0x9F]  // stray continuation, truncated 3- and 4-byte sequences
        let result = stream(bytes.map(Int.init), expectWellFormed: false)
        #expect(result.content == String(decoding: bytes, as: UTF8.self))
    }

    @Test func stopSequencesEndTheReplyBeforeThemAndNeverFlash() {
        let result = stream(FakeVocabulary.tokens("The answer is 42.\n\nUser: next"), stops: ["\n\nUser:"])
        #expect(result.content == "The answer is 42." && result.stopped)
        #expect(!result.deltas.map(\.content).joined().contains("\n"), "the stop's prefix was never emitted")
        // A prefix that never completes is released.
        let partial = stream(FakeVocabulary.tokens("a STOx and STOP"), stops: ["STOP"])
        #expect(partial.content == "a STOx and " && partial.stopped)
        let unfinished = stream(FakeVocabulary.tokens("ends with STO"), stops: ["STOP"])
        #expect(unfinished.content == "ends with STO" && !unfinished.stopped)
        // The earliest of several stops wins; text after a stop is ignored.
        let several = stream(FakeVocabulary.tokens("one|two#three"), stops: ["#", "|"])
        #expect(several.content == "one")
    }

    @Test func reasoningGoesToItsOwnChannelWithoutItsFraming() {
        let tokens = [FakeVocabulary.thinkStart] + FakeVocabulary.tokens("\nThe user wants a colour.\n\n") + [FakeVocabulary.thinkEnd]
            + FakeVocabulary.tokens("\n\nTeal.")
        let result = stream(tokens, delimiters: delimiters)
        #expect(result.reasoning == "The user wants a colour." && result.content == "Teal.")
        // A reply that begins inside a block the prompt opened.
        let open = stream(FakeVocabulary.tokens("\nhmm\n") + [FakeVocabulary.thinkEnd] + FakeVocabulary.tokens("\nOK"),
                          delimiters: delimiters, startsInReasoning: true)
        #expect(open.reasoning == "hmm" && open.content == "OK")
        // Without delimiters (a model without reasoning) the markers are just text, as the tokenizer decodes them.
        #expect(stream([FakeVocabulary.thinkStart] + FakeVocabulary.tokens("x")).content == "<think>x")
        // Stop sequences apply to the answer, not to reasoning.
        let stopped = stream([FakeVocabulary.thinkStart] + FakeVocabulary.tokens("a#b") + [FakeVocabulary.thinkEnd] + FakeVocabulary.tokens("c#d"),
                             delimiters: delimiters, stops: ["#"])
        #expect(stopped.reasoning == "a#b" && stopped.content == "c")
    }

    @Test func fuzzedStreamsConcatenateToTheWholeText() {
        let pieces = ["a", "Z", " ", "\n", "é", "e\u{301}", "日", "🌙", "👨\u{200D}👩\u{200D}👧", "🇯🇵", "\u{1100}\u{1161}", "\r\n", "👋\u{1F3FD}"]
        var generator = LanguageRandomGenerator(seed: 2026)
        for _ in 0..<300 {
            let text = (0..<Int(generator.next() % 24 + 1)).map { _ in pieces[Int(generator.next() % UInt64(pieces.count))] }.joined()
            let result = stream(FakeVocabulary.tokens(text))
            #expect(result.content == text, "\(text.unicodeScalars.map { String($0.value, radix: 16) })")
            expectStable(result.deltas, final: text)
        }
    }

    @Test func completeLengthFindsTheLastWholeScalar() {
        #expect(LanguageTextStream.completeLength([]) == 0)
        #expect(LanguageTextStream.completeLength([0x41]) == 1)
        #expect(LanguageTextStream.completeLength([0x41, 0xE6]) == 1)
        #expect(LanguageTextStream.completeLength([0x41, 0xE6, 0x97]) == 1)
        #expect(LanguageTextStream.completeLength([0x41, 0xE6, 0x97, 0xA5]) == 4)
        #expect(LanguageTextStream.completeLength([0xF0, 0x9F, 0x8C]) == 0)
        #expect(LanguageTextStream.completeLength([0x80, 0x80]) == 2)
    }
}
