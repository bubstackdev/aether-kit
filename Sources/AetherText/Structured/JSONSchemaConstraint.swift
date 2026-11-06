public import AetherRuntime
import Foundation

/// A model's tokens indexed for JSON constraints, built once per loaded model: a trie of every token's bytes (special
/// tokens, reasoning delimiters and byteless rows left out), the set of tokens that are plain string text (allowed
/// anywhere inside a string body, between whole scalars), and a second trie of the rest.
public final class JSONVocabulary: Sendable {
    let headRows: Int
    let bytes: TokenBytes
    let all: Trie
    let nonPlain: Trie
    let plain: TokenSet

    struct Trie: Sendable {
        /// Per node: its first child in `children`, its child count, its first token in `tokens`, its token count.
        var childStart: [Int32] = [0], childCount: [Int32] = [0], tokenStart: [Int32] = [0], tokenCount: [Int32] = [0]
        var childByte: [UInt8] = [], childNode: [Int32] = []
        var tokens: [Int32] = []
    }

    public init(bytes: TokenBytes, headRows: Int, excluded: Set<Int> = []) {
        self.headRows = headRows
        self.bytes = bytes
        var usable: [(id: Int, bytes: [UInt8])] = []
        for id in 0..<min(bytes.count, headRows) where !excluded.contains(id) && !bytes.isSpecial(id) {
            guard let tokenBytes = bytes.bytes(of: id), !tokenBytes.isEmpty else { continue }
            usable.append((id, Array(tokenBytes)))
        }
        var plain = TokenSet(vocabularySize: headRows)
        var nonPlain: [(id: Int, bytes: [UInt8])] = []
        for token in usable {
            if Self.isPlainString(token.bytes, pending: 0) { plain.insert(token.id) } else { nonPlain.append(token) }
        }
        self.plain = plain
        all = Self.trie(usable)
        self.nonPlain = Self.trie(nonPlain)
    }

    /// Plain string text: no quote, backslash or control byte, and valid UTF-8 (RFC 3629) from a scalar boundary.
    static func isPlainString(_ bytes: [UInt8], pending: Int) -> Bool {
        var pending = pending, low: UInt8 = 0x80, high: UInt8 = 0xBF
        for byte in bytes {
            if pending > 0 {
                guard byte >= low, byte <= high else { return false }
                pending -= 1
                (low, high) = (0x80, 0xBF)
            } else if byte == UInt8(ascii: "\"") || byte == UInt8(ascii: "\\") || byte < 0x20 {
                return false
            } else if byte >= 0x80 {
                guard let next = JSONMatcher.continuation(lead: byte) else { return false }
                (pending, low, high) = (next.count, next.low, next.high)
            }
        }
        return true
    }

    private static func trie(_ tokens: [(id: Int, bytes: [UInt8])]) -> Trie {
        // Sorting by bytes makes each node's children and subtree contiguous, so the trie builds in one pass.
        let sorted = tokens.sorted { $0.bytes.lexicographicallyPrecedes($1.bytes) }
        final class Builder {
            var children: [[(UInt8, Int)]] = [[]]
            var tokens: [[Int32]] = [[]]
        }
        let builder = Builder()
        for token in sorted {
            var node = 0
            for byte in token.bytes {
                if let last = builder.children[node].last, last.0 == byte {
                    node = last.1
                } else {
                    builder.children.append([])
                    builder.tokens.append([])
                    let child = builder.children.count - 1
                    builder.children[node].append((byte, child))
                    node = child
                }
            }
            builder.tokens[node].append(Int32(token.id))
        }
        var trie = Trie(childStart: [], childCount: [], tokenStart: [], tokenCount: [])
        for node in builder.children.indices {
            trie.childStart.append(Int32(trie.childByte.count))
            trie.childCount.append(Int32(builder.children[node].count))
            for (byte, child) in builder.children[node] {
                trie.childByte.append(byte)
                trie.childNode.append(Int32(child))
            }
            trie.tokenStart.append(Int32(trie.tokens.count))
            trie.tokenCount.append(Int32(builder.tokens[node].count))
            trie.tokens += builder.tokens[node]
        }
        return trie
    }
}

/// Constrains a reply to JSON matching a generation schema, as a ``LogitsProcessor``: each call, run beside the step,
/// advances the matcher by the accepted token's bytes and returns the tokens whose bytes the matcher accepts next (only
/// the stop tokens once the value is complete). Allowed sets are cached by matcher state, so a string body or a repeated
/// position costs a lookup after its first visit.
///
/// Thinking, with the model's ``Reasoning`` delimiters (single tokens, matched by id as 0.0.x did):
/// - a prompt that leaves a thinking block open (`lastIndex(open) > lastIndex(close)`, as when the template opens it
///   or ignores the off switch) starts the reply inside it: unconstrained until the close token, then constrained;
/// - with `allowThinking`, a reply may open a block itself: its first token is the open token or a JSON start;
/// - otherwise the reply is constrained from its first token.
public actor JSONSchemaConstraint: LogitsProcessor {
    public struct Statistics: Sendable, Equatable {
        public var calls = 0
        public var cacheHits = 0
        public var totalSeconds = 0.0
        public var maximumSeconds = 0.0
    }

    /// The thinking delimiters, as token ids.
    public struct Reasoning: Sendable, Hashable {
        public var open: Int32
        public var close: Int32
        public init(open: Int32, close: Int32) {
            self.open = open
            self.close = close
        }
    }

    private enum Phase { case thinking, opening, value }

    private let grammar: JSONGrammar
    private let vocabulary: JSONVocabulary
    private let stopTokens: TokenSet
    private let reasoning: Reasoning?
    private let allowThinking: Bool
    private var matcher: JSONMatcher
    private var phase = Phase.value
    private var cache: [JSONMatcher: TokenSet] = [:]
    public private(set) var statistics = Statistics()

    /// The constrained text is a JSON value, not prose: the Generator parses no tags in it (``LogitsProcessor``).
    public nonisolated var isStructuredOutput: Bool { true }

    public init(schema: OutputSchema, vocabulary: JSONVocabulary, stopTokenIDs: Set<Int32>,
                reasoning: Reasoning? = nil, allowThinking: Bool = false) throws {
        grammar = try JSONGrammar(schema: schema)
        self.vocabulary = vocabulary
        stopTokens = TokenSet(stopTokenIDs.map(Int.init), vocabularySize: vocabulary.headRows)
        self.reasoning = reasoning
        self.allowThinking = allowThinking
        matcher = JSONMatcher(grammar: grammar)
    }

    public func begin(prompt: [Int32]) -> TokenConstraint {
        matcher = JSONMatcher(grammar: grammar)
        if let reasoning, (prompt.lastIndex(of: reasoning.open) ?? -1) > (prompt.lastIndex(of: reasoning.close) ?? -1) {
            phase = .thinking
            return .unconstrained
        }
        phase = reasoning != nil && allowThinking ? .opening : .value
        return timed { constraint() }
    }

    public func accept(_ token: Int32) throws -> TokenConstraint {
        switch phase {
        case .thinking:
            guard token == reasoning?.close else { return .unconstrained }
            phase = .value
            return timed { constraint() }
        case .opening where token == reasoning?.open:
            phase = .thinking
            return .unconstrained
        case .opening, .value:
            phase = .value
            return try timed {
                guard let tokenBytes = vocabulary.bytes.bytes(of: Int(token)), let next = matcher.feeding(tokenBytes, grammar) else {
                    throw AetherError.generation(.constraintViolated("token \(token) left the schema's grammar"))
                }
                matcher = next
                return constraint()
            }
        }
    }

    /// Whether `text` is a complete value of the schema (the reply's text after any thinking). Checked from the
    /// text, since the last token at the output limit is never `accept`ed.
    public nonisolated func completes(_ text: String) -> Bool {
        JSONMatcher(grammar: grammar).feeding(Array(text.utf8), grammar)?.canEnd(grammar) ?? false
    }

    private func constraint() -> TokenConstraint {
        guard case .allowed(var set) = allowed() else { return .unconstrained }
        if phase == .opening, let open = reasoning?.open { set.insert(Int(open)) }
        return .allowed(set)
    }

    private func timed<T>(_ body: () throws -> T) rethrows -> T {
        let start = ContinuousClock.now
        defer {
            let elapsed = start.duration(to: .now)
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            statistics.calls += 1
            statistics.totalSeconds += seconds
            statistics.maximumSeconds = max(statistics.maximumSeconds, seconds)
        }
        return try body()
    }

    private func allowed() -> TokenConstraint {
        if let cached = cache[matcher] {
            statistics.cacheHits += 1
            return .allowed(cached)
        }
        let set = Self.allowedTokens(matcher, grammar: grammar, vocabulary: vocabulary, stopTokens: stopTokens)
        cache[matcher] = set
        return .allowed(set)
    }

    /// Every token whose bytes `matcher` accepts, plus the stop tokens when the reply can end.
    static func allowedTokens(_ matcher: JSONMatcher, grammar: JSONGrammar, vocabulary: JSONVocabulary,
                              stopTokens: TokenSet) -> TokenSet {
        var set: TokenSet
        let trie: JSONVocabulary.Trie
        if matcher.stringBodyPending == 0 {
            // Plain string text keeps the matcher in this string; only the other tokens need walking. (Mid-scalar, the
            // full trie prunes at the first byte anyway.)
            set = vocabulary.plain
            trie = vocabulary.nonPlain
        } else {
            set = TokenSet(vocabularySize: vocabulary.headRows)
            trie = vocabulary.all
        }
        if matcher.canEnd(grammar) { set.formUnion(stopTokens) }
        var stack: [(node: Int32, matcher: JSONMatcher)] = [(0, matcher)]
        while let (node, state) = stack.popLast() {
            let start = Int(trie.childStart[Int(node)])
            for index in start..<(start + Int(trie.childCount[Int(node)])) {
                guard let next = state.feeding(trie.childByte[index], grammar) else { continue }
                let child = trie.childNode[index]
                let tokens = Int(trie.tokenStart[Int(child)])
                for position in tokens..<(tokens + Int(trie.tokenCount[Int(child)])) { set.insert(Int(trie.tokens[position])) }
                stack.append((child, next))
            }
        }
        return set
    }
}
