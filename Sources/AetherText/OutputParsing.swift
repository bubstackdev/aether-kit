public import AetherCore
import Foundation

/// Which tags structure a model's output (DESIGN.md §7), from the manifest only: `tokenizer.thinking` gives the
/// thinking tags, `tokenizer.toolDialect` the tool-call tags. Nothing is probed from the vocabulary.
public struct OutputProfile: Sendable, Hashable {
    public enum Kind: Sendable, Hashable { case thinking, toolCall }

    public struct Tag: Sendable, Hashable {
        public var open: String
        public var close: String
        public var kind: Kind

        public init(open: String, close: String, kind: Kind) {
            self.open = open
            self.close = close
            self.kind = kind
        }
    }

    public var tags: [Tag]
    /// The manifest's `tokenizer.toolDialect`: how a tool call's body is written (``ToolDialects``).
    public var toolDialect: String?

    public init(tags: [Tag], toolDialect: String? = nil) {
        self.tags = tags
        self.toolDialect = toolDialect
    }

    public var thinkingTag: Tag? { tags.first { $0.kind == .thinking } }

    public static let plain = OutputProfile(tags: [])

    /// Tool-call tags per dialect. `hermes-json` (Qwen3, Granite) and `qwen-xml` (Qwen3.5) both wrap a call in
    /// `<tool_call>…</tool_call>`; the body's dialect is parsed by M3's `ToolDialect`.
    static let toolTags: [String: (String, String)] = [
        "hermes-json": ("<tool_call>", "</tool_call>"),
        "qwen-xml": ("<tool_call>", "</tool_call>"),
    ]

    public init(manifest tokenizer: Manifest.Tokenizer?) {
        var tags: [Tag] = []
        if let t = tokenizer?.thinking { tags.append(Tag(open: t.open, close: t.close, kind: .thinking)) }
        if let dialect = tokenizer?.toolDialect, let (open, close) = Self.toolTags[dialect] {
            tags.append(Tag(open: open, close: close, kind: .toolCall))
        }
        self.tags = tags
        toolDialect = tokenizer?.toolDialect
    }
}

/// Generated tokens to text deltas (ported from 0.0.x `LanguageTextStream`):
/// - from the tokens' bytes when the tokenizer is byte-level (``TokenBytes``), otherwise through the incremental
///   detokenizer; either way a delta never holds part of a UTF-8 scalar, so it never shows a replacement character
///   that later turns into text;
/// - a delta never ends inside a grapheme cluster the next scalar could extend (combining marks, ZWJ emoji, flags,
///   Hangul jamo): the last character waits for the next one, so emitted characters never change;
/// - special tokens contribute no text (as `decode(skipSpecialTokens: true)`), except those in `keepSpecial` (the
///   manifest's tags, which the tag parser must see).
struct TokenText {
    private enum Source {
        case bytes(TokenBytes, keep: Set<Int>)
        case decoder(any TokenDecoding, IncrementalDetokenizer, keep: Set<Int32>)
    }

    private var source: Source
    private var pending: [UInt8] = []
    /// The last character, held until the next one shows it can't grow.
    private var held = ""

    init(decoder: any TokenDecoding, keepSpecial: Set<String> = []) {
        if let bytes = decoder.tokenBytes {
            source = .bytes(bytes, keep: Set(keepSpecial.compactMap { decoder.tokenID($0) }.map(Int.init)))
        } else {
            source = .decoder(decoder, IncrementalDetokenizer(), keep: Set(keepSpecial.compactMap { decoder.tokenID($0) }))
        }
    }

    mutating func append(_ token: Int32) -> String {
        let text: String
        switch source {
        case .bytes(let table, let keep):
            let id = Int(token)
            guard !table.isSpecial(id) || keep.contains(id), let bytes = table.bytes(of: id) else { return "" }
            pending += bytes
            let complete = Self.completeLength(pending)
            guard complete > 0 else { return "" }
            text = String(decoding: pending[..<complete], as: UTF8.self)
            pending.removeFirst(complete)
        case .decoder(let decoder, var detokenizer, let keep):
            guard keep.contains(token) || !decoder.isSpecial(token) else { return "" }
            text = detokenizer.append(token, decoder)
            source = .decoder(decoder, detokenizer, keep: keep)
        }
        return hold(text)
    }

    /// The rest, at the end of the reply: incomplete bytes decode with replacement characters, as the tokenizer's
    /// decode does.
    mutating func finish() -> String {
        var text = held
        held = ""
        switch source {
        case .bytes:
            text += String(decoding: pending, as: UTF8.self)
            pending.removeAll()
        case .decoder(let decoder, var detokenizer, let keep):
            text += detokenizer.finish(decoder)
            source = .decoder(decoder, detokenizer, keep: keep)
        }
        return text
    }

    private mutating func hold(_ text: String) -> String {
        guard !text.isEmpty else { return "" }
        let all = held + text
        guard let last = all.indices.last else { return "" }
        held = String(all[last...])
        return String(all[..<last])
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

/// A streaming tag parser: text in, typed segments out. Text that could still be the start of a tag is held back
/// until it is decided, so a tag split across tokens is never emitted as response text.
///
/// Thinking is framed as 0.0.x framed it: newlines right after the open tag and before the close tag are framing,
/// not reasoning, and newlines right after the close tag are framing, not response text.
struct TagParser {
    enum Segment: Equatable {
        case text(String)
        case thinking(String)
        case toolCall(String)
    }

    let profile: OutputProfile
    private var inside: Int?
    private var pending = ""
    private var toolBody = ""
    /// Trailing newlines of the thinking so far: framing if the close tag follows, reasoning if more text does.
    private var thinkingNewlines = ""
    /// Drop leading newlines of the next thinking (after an open tag) or text (after a thinking close tag).
    private var skipNewlines = false

    /// A structured reply (`valueOnly`): only a leading thinking block is parsed; once it closes, or once any text
    /// arrives, everything is text, so a JSON string holding a tag's text stays in the value.
    private let valueOnly: Bool
    private var sealed = false

    init(profile: OutputProfile, startsInThinking: Bool = false, valueOnly: Bool = false) {
        self.profile = valueOnly ? OutputProfile(tags: profile.tags.filter { $0.kind == .thinking }, toolDialect: profile.toolDialect) : profile
        self.valueOnly = valueOnly
        inside = startsInThinking ? self.profile.tags.firstIndex { $0.kind == .thinking } : nil
        skipNewlines = inside != nil
    }

    /// Whether the parser is inside a thinking block.
    var inThinking: Bool { inside.map { profile.tags[$0].kind == .thinking } ?? false }

    mutating func feed(_ text: String) -> [Segment] {
        pending += text
        var out: [Segment] = []
        while true {
            if let i = inside {
                let tag = profile.tags[i]
                if let r = pending.range(of: tag.close) {
                    emit(String(pending[..<r.lowerBound]), in: tag.kind, to: &out)
                    if tag.kind == .toolCall {
                        out.append(.toolCall(toolBody))
                        toolBody = ""
                    } else {
                        thinkingNewlines = ""
                        skipNewlines = true
                        if valueOnly { sealed = true }
                    }
                    pending = String(pending[r.upperBound...])
                    inside = nil
                    continue
                }
                let keep = Self.partialSuffix(of: pending, matching: [tag.close])
                emit(String(pending.dropLast(keep)), in: tag.kind, to: &out)
                pending = String(pending.suffix(keep))
                return out
            }
            if sealed {
                emitText(pending, to: &out)
                pending = ""
                return out
            }
            let opens = profile.tags.enumerated().compactMap { i, tag in pending.range(of: tag.open).map { (i, $0) } }
            if let (i, r) = opens.min(by: { $0.1.lowerBound < $1.1.lowerBound }),
               !(valueOnly && !pending[..<r.lowerBound].allSatisfy(\.isNewline)) {
                emitText(String(pending[..<r.lowerBound]), to: &out)
                pending = String(pending[r.upperBound...])
                inside = i
                skipNewlines = profile.tags[i].kind == .thinking
                thinkingNewlines = ""
                continue
            }
            let keep = Self.partialSuffix(of: pending, matching: profile.tags.map(\.open))
            emitText(String(pending.dropLast(keep)), to: &out)
            pending = String(pending.suffix(keep))
            return out
        }
    }

    /// Flushes held-back text at the end of generation. An unterminated tool call is returned as a tool call.
    mutating func finish() -> [Segment] {
        var out: [Segment] = []
        if let i = inside {
            let kind = profile.tags[i].kind
            emit(pending, in: kind, to: &out)
            if kind == .toolCall { out.append(.toolCall(toolBody)) }
        } else {
            emitText(pending, to: &out)
        }
        pending = ""
        toolBody = ""
        thinkingNewlines = ""
        inside = nil
        return out
    }

    private mutating func emitText(_ s: String, to out: inout [Segment]) {
        var s = Substring(s)
        if skipNewlines {
            s = s.drop { $0 == "\n" || $0 == "\r\n" }
            if s.isEmpty { return }
            skipNewlines = false
        }
        guard !s.isEmpty else { return }
        out.append(.text(String(s)))
        if valueOnly { sealed = true }
    }

    private mutating func emit(_ s: String, in kind: OutputProfile.Kind, to out: inout [Segment]) {
        guard !s.isEmpty else { return }
        switch kind {
        case .toolCall:
            toolBody += s
        case .thinking:
            var text = Substring(thinkingNewlines + s)
            if skipNewlines {
                text = text.drop { $0 == "\n" || $0 == "\r\n" }
                if text.isEmpty {
                    thinkingNewlines = ""
                    return
                }
                skipNewlines = false
            }
            let body = text.reversed().drop { $0 == "\n" || $0 == "\r\n" }.count
            let end = text.index(text.startIndex, offsetBy: body)
            thinkingNewlines = String(text[end...])
            if body > 0 { out.append(.thinking(String(text[..<end]))) }
        }
    }

    /// The length of the longest suffix of `s` that is a proper prefix of one of `markers`.
    static func partialSuffix(of s: String, matching markers: [String]) -> Int {
        var best = 0
        for marker in markers {
            var n = min(marker.count - 1, s.count)
            while n > best {
                if s.hasSuffix(marker.prefix(n)) {
                    best = n
                    break
                }
                n -= 1
            }
        }
        return best
    }
}

/// Text-level stop sequences on the response text (not on thinking or tool calls), with the same hold-back rule: text
/// that could be the start of a stop waits until it completes (the reply ends before it) or can't.
struct StopMatcher {
    let sequences: [String]
    private var pending = ""

    init(_ sequences: [String]) { self.sequences = sequences.filter { !$0.isEmpty } }

    /// Returns the text that is safe to emit, and the stop sequence if one completed (text after it is dropped).
    mutating func feed(_ text: String) -> (String, String?) {
        guard !sequences.isEmpty else { return (text, nil) }
        pending += text
        let hits = sequences.compactMap { seq in pending.range(of: seq).map { (seq, $0) } }
        if let (seq, r) = hits.min(by: { $0.1.lowerBound < $1.1.lowerBound }) {
            let out = String(pending[..<r.lowerBound])
            pending = ""
            return (out, seq)
        }
        let keep = TagParser.partialSuffix(of: pending, matching: sequences)
        let out = String(pending.dropLast(keep))
        pending = String(pending.suffix(keep))
        return (out, nil)
    }

    /// Releases held text: at the end of the reply, or when a tag ends the text run (a stop can't span a tag).
    mutating func finish() -> String {
        defer { pending = "" }
        return pending
    }
}

/// Incremental detokenization for tokenizers without a byte table, without re-decoding the whole output (no O(n²)):
/// decode a short window from `prefix` to the end, and emit only what extends the window's already-emitted text.
/// Holds back while the tail is an incomplete UTF-8 sequence (U+FFFD), so multi-token characters come out whole.
struct IncrementalDetokenizer {
    private(set) var tokens: [Int32] = []
    private var prefix = 0
    private var read = 0

    mutating func append(_ token: Int32, _ decoder: any TokenDecoding) -> String {
        tokens.append(token)
        return step(decoder, final: false)
    }

    mutating func finish(_ decoder: any TokenDecoding) -> String { step(decoder, final: true) }

    private mutating func step(_ decoder: any TokenDecoding, final: Bool) -> String {
        guard read < tokens.count else { return "" }
        let before = decoder.decode(Array(tokens[prefix..<read]))
        let after = decoder.decode(Array(tokens[prefix...]))
        guard final || !after.hasSuffix("\u{FFFD}") else { return "" }
        let a = Array(after.utf8), b = Array(before.utf8)
        guard a.count >= b.count, a.starts(with: b) else {
            // The window re-segmented: its text is no longer a prefix of the new decode. Hold everything back (do not
            // advance) until the end, then emit what follows the longest common prefix, cut at a scalar boundary so
            // no UTF-8 sequence is split.
            guard final else { return "" }
            var common = zip(a, b).prefix { $0 == $1 }.count
            while common > 0, common < a.count, a[common] & 0xC0 == 0x80 { common -= 1 }
            prefix = read
            read = tokens.count
            return String(decoding: a.dropFirst(common), as: UTF8.self)
        }
        let delta = String(decoding: a.dropFirst(b.count), as: UTF8.self)
        prefix = read
        read = tokens.count
        return delta
    }
}
