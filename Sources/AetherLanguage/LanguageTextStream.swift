import AetherLanguageTokenizers

/// Turns generated token IDs into text deltas, from the tokens' bytes rather than by re-decoding:
/// - a delta never holds part of a UTF-8 scalar, so it never shows a replacement character that later turns into text;
/// - a delta never ends inside a grapheme cluster the next scalar could extend (combining marks, ZWJ emoji, flags,
///   Hangul jamo): the last character waits for the next one, so emitted characters never change;
/// - text that could be the start of a stop sequence waits until it completes (the reply ends before it) or can't;
/// - reasoning (between the registration's thinking delimiters, which are not text) goes to its own channel, without
///   the newlines that frame it.
/// Special tokens contribute no text, as `decode(skipSpecialTokens: true)`.
package struct LanguageTextStream {
    package struct Delta: Sendable, Equatable {
        package var content = ""
        package var reasoning = ""
        package var isEmpty: Bool { content.isEmpty && reasoning.isEmpty }
    }

    struct ReasoningDelimiters: Sendable, Equatable {
        let start: Int
        let end: Int
    }

    private let bytes: LanguageTokenBytes
    private let delimiters: ReasoningDelimiters?
    private let stopSequences: [String]
    private(set) var inReasoning: Bool
    private var pending: [UInt8] = []
    /// Decoded text not yet emitted on the current channel.
    private var tail = ""
    /// Newlines right after a delimiter are framing, not text.
    private var skippingNewlines = false
    /// A stop sequence matched: the reply ends, and nothing more is emitted.
    private(set) var stopped = false
    private(set) var content = ""
    private(set) var reasoning = ""

    /// `startsInReasoning`: the prompt left a thinking block open, so the reply begins inside it.
    init(bytes: LanguageTokenBytes, delimiters: ReasoningDelimiters?, startsInReasoning: Bool = false, stopSequences: [String] = []) {
        self.bytes = bytes
        self.delimiters = delimiters
        self.stopSequences = stopSequences.filter { !$0.isEmpty }
        inReasoning = delimiters != nil && startsInReasoning
        skippingNewlines = inReasoning
    }

    mutating func append(_ token: Int) -> Delta {
        guard !stopped else { return Delta() }
        if let delimiters, token == delimiters.start || token == delimiters.end {
            let entering = token == delimiters.start
            guard entering != inReasoning else { return Delta() }
            // Text can't continue across a delimiter: the channel ending now releases everything it held.
            let delta = flush(finalChannel: true)
            inReasoning = entering
            skippingNewlines = true
            return delta
        }
        guard !bytes.isSpecial(token), let tokenBytes = bytes.bytes(of: token) else { return Delta() }
        pending += tokenBytes
        let complete = Self.completeLength(pending)
        guard complete > 0 else { return Delta() }
        var text = String(decoding: pending[..<complete], as: UTF8.self)
        pending.removeFirst(complete)
        if skippingNewlines {
            text = String(text.drop { $0 == "\n" || $0 == "\r\n" })
            if !text.isEmpty { skippingNewlines = false }
        }
        tail += text
        return emit(final: false)
    }

    /// The rest of the text, at the end of the reply: incomplete bytes decode with replacement characters, as
    /// `String(decoding:as:)` and the tokenizer's decode do, and held text is released unless a stop sequence matched.
    mutating func finish() -> Delta {
        guard !stopped else { return Delta() }
        return flush(finalChannel: true)
    }

    private mutating func flush(finalChannel: Bool) -> Delta {
        if !pending.isEmpty {
            tail += String(decoding: pending, as: UTF8.self)
            pending.removeAll()
        }
        return emit(final: finalChannel)
    }

    /// Moves what can be emitted from `tail` to the current channel.
    private mutating func emit(final: Bool) -> Delta {
        var out = Delta()
        if inReasoning {
            // Trailing newlines may be the closing frame of the block: they wait, and are dropped at its end.
            var held = final ? 0 : 1
            if !final { held = max(held, Self.trailingCount(of: tail) { $0 == "\n" || $0 == "\r\n" }) }
            var text = String(tail.dropLast(held))
            if final { text = String(text.reversed().drop { $0 == "\n" || $0 == "\r\n" }.reversed()) }
            tail = final ? "" : String(tail.suffix(held))
            reasoning += text
            out.reasoning = text
            return out
        }
        if let match = stopSequences.compactMap({ tail.firstRange(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
            out.content = String(tail[..<match.lowerBound])
            content += out.content
            tail = ""
            stopped = true
            return out
        }
        let held = final ? 0 : max(1, longestStopPrefixSuffix(of: tail))
        out.content = String(tail.dropLast(held))
        tail = final ? "" : String(tail.suffix(held))
        content += out.content
        return out
    }

    /// The length, in characters, of the longest suffix of `text` that is a proper prefix of a stop sequence.
    private func longestStopPrefixSuffix(of text: String) -> Int {
        var longest = 0
        for stop in stopSequences {
            for length in stride(from: min(stop.count - 1, text.count), to: longest, by: -1)
            where text.suffix(length).elementsEqual(stop.prefix(length)) {
                longest = length
                break
            }
        }
        return longest
    }

    private static func trailingCount(of text: String, where predicate: (Character) -> Bool) -> Int {
        var count = 0
        for character in text.reversed() {
            guard predicate(character) else { break }
            count += 1
        }
        return count
    }

    /// How many leading bytes form complete UTF-8 sequences: an incomplete multi-byte sequence at the end waits for the
    /// next token. Malformed bytes count as complete (they decode to replacement characters, as the tokenizer's do).
    static func completeLength(_ bytes: [UInt8]) -> Int {
        var index = bytes.count - 1
        while index >= 0, bytes.count - index <= 4 {
            let byte = bytes[index]
            if byte & 0xC0 != 0x80 {
                let needed = byte >= 0xF0 && byte < 0xF8 ? 4 : byte >= 0xE0 && byte < 0xF0 ? 3 : byte >= 0xC0 && byte < 0xE0 ? 2 : 1
                return bytes.count - index < needed ? index : bytes.count
            }
            index -= 1
        }
        return bytes.count
    }
}
