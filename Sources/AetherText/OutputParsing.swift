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

    public init(tags: [Tag]) { self.tags = tags }

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
    }
}

/// A streaming tag parser: text in, typed segments out. Text that could still be the start of a tag is held back
/// until it is decided, so a tag split across tokens is never emitted as response text.
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

    init(profile: OutputProfile, startsInThinking: Bool = false) {
        self.profile = profile
        inside = startsInThinking ? profile.tags.firstIndex { $0.kind == .thinking } : nil
    }

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
            let opens = profile.tags.enumerated().compactMap { i, tag in pending.range(of: tag.open).map { (i, $0) } }
            if let (i, r) = opens.min(by: { $0.1.lowerBound < $1.1.lowerBound }) {
                if r.lowerBound > pending.startIndex { out.append(.text(String(pending[..<r.lowerBound]))) }
                pending = String(pending[r.upperBound...])
                inside = i
                continue
            }
            let keep = Self.partialSuffix(of: pending, matching: profile.tags.map(\.open))
            let ready = String(pending.dropLast(keep))
            if !ready.isEmpty { out.append(.text(ready)) }
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
        } else if !pending.isEmpty {
            out.append(.text(pending))
        }
        pending = ""
        toolBody = ""
        inside = nil
        return out
    }

    private mutating func emit(_ s: String, in kind: OutputProfile.Kind, to out: inout [Segment]) {
        guard !s.isEmpty else { return }
        switch kind {
        case .thinking: out.append(.thinking(s))
        case .toolCall: toolBody += s
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

/// Text-level stop sequences, with the same hold-back rule.
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

    mutating func finish() -> String {
        defer { pending = "" }
        return pending
    }
}

/// Incremental detokenization without re-decoding the whole output (no O(n²)): decode a short window from
/// `prefix` to the end, and emit only what extends the window's already-emitted text. Holds back while the tail is
/// an incomplete UTF-8 sequence (U+FFFD), so multi-token characters come out whole.
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
