import AetherLanguageRuntime
import AetherLanguageTokenizers
import Foundation

/// A model's tokens indexed for JSON constraints, built once per loaded model: a trie of every token's bytes (special
/// tokens, reasoning delimiters and byteless rows left out), the set of tokens that are plain string text (allowed
/// anywhere inside a string body, between whole scalars), and a second trie of the rest.
package final class LanguageJSONVocabulary: Sendable {
    let headRows: Int
    let bytes: LanguageTokenBytes
    let all: Trie
    let nonPlain: Trie
    let plain: LanguageTokenSet

    struct Trie: Sendable {
        /// Per node: its first child in `children`, its child count, its first token in `tokens`, its token count.
        var childStart: [Int32] = [0], childCount: [Int32] = [0], tokenStart: [Int32] = [0], tokenCount: [Int32] = [0]
        var childByte: [UInt8] = [], childNode: [Int32] = []
        var tokens: [Int32] = []
    }

    package init(bytes: LanguageTokenBytes, headRows: Int, excluded: Set<Int>) {
        self.headRows = headRows
        self.bytes = bytes
        var usable: [(id: Int, bytes: [UInt8])] = []
        for id in 0..<min(bytes.count, headRows) where !excluded.contains(id) && !bytes.isSpecial(id) {
            guard let tokenBytes = bytes.bytes(of: id), !tokenBytes.isEmpty else { continue }
            usable.append((id, Array(tokenBytes)))
        }
        var plain = LanguageTokenSet(vocabularySize: headRows)
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
                guard let next = LanguageJSONMatcher.continuation(lead: byte) else { return false }
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

/// Constrains a reply to JSON matching a generation schema, as a runtime `LanguageLogitsProcessor`: each call, run
/// beside the GPU step, advances the matcher by the accepted token's bytes and returns the tokens whose bytes the
/// matcher accepts next (only the stop tokens once the value is complete). Allowed sets are cached by matcher state, so a
/// string body or a repeated position costs a lookup after its first visit.
///
/// With `reasoningEnd`, the reply may think first: it's unconstrained until that token, then constrained.
package actor LanguageJSONSchemaConstraint: LanguageLogitsProcessor {
    package struct Statistics: Sendable, Equatable {
        package var calls = 0
        package var cacheHits = 0
        package var totalSeconds = 0.0
        package var maximumSeconds = 0.0
    }

    private let grammar: LanguageJSONGrammar
    private let vocabulary: LanguageJSONVocabulary
    private let stopTokens: LanguageTokenSet
    private let reasoningEnd: Int?
    private var matcher: LanguageJSONMatcher
    private var thinking: Bool
    private var cache: [LanguageJSONMatcher: LanguageTokenSet] = [:]
    package private(set) var statistics = Statistics()

    package init(schema: AetherGenerationSchema, vocabulary: LanguageJSONVocabulary, stopTokenIDs: Set<Int>,
                 reasoningEnd: Int? = nil) throws {
        grammar = try LanguageJSONGrammar(schema: schema)
        self.vocabulary = vocabulary
        stopTokens = LanguageTokenSet(stopTokenIDs, vocabularySize: vocabulary.headRows)
        self.reasoningEnd = reasoningEnd
        matcher = LanguageJSONMatcher(grammar: grammar)
        thinking = reasoningEnd != nil
    }

    package func begin(prompt: [Int]) -> LanguageTokenConstraint {
        matcher = LanguageJSONMatcher(grammar: grammar)
        thinking = reasoningEnd != nil
        return thinking ? .unconstrained : timed { allowed() }
    }

    package func accept(_ token: Int) throws -> LanguageTokenConstraint {
        if thinking {
            guard token == reasoningEnd else { return .unconstrained }
            thinking = false
            return timed { allowed() }
        }
        return try timed {
            guard let tokenBytes = vocabulary.bytes.bytes(of: token), let next = matcher.feeding(tokenBytes, grammar) else {
                throw AetherLanguageError.decodingFailure("token \(token) left the schema's grammar")
            }
            matcher = next
            return allowed()
        }
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

    private func allowed() -> LanguageTokenConstraint {
        if let cached = cache[matcher] {
            statistics.cacheHits += 1
            return .allowed(cached)
        }
        let set = Self.allowedTokens(matcher, grammar: grammar, vocabulary: vocabulary, stopTokens: stopTokens)
        cache[matcher] = set
        return .allowed(set)
    }

    /// Every token whose bytes `matcher` accepts, plus the stop tokens when the reply can end.
    static func allowedTokens(_ matcher: LanguageJSONMatcher, grammar: LanguageJSONGrammar, vocabulary: LanguageJSONVocabulary,
                              stopTokens: LanguageTokenSet) -> LanguageTokenSet {
        var set: LanguageTokenSet
        let trie: LanguageJSONVocabulary.Trie
        if matcher.stringBodyPending == 0 {
            // Plain string text keeps the matcher in this string; only the other tokens need walking. (Mid-scalar, the
            // full trie prunes at the first byte anyway.)
            set = vocabulary.plain
            trie = vocabulary.nonPlain
        } else {
            set = LanguageTokenSet(vocabularySize: vocabulary.headRows)
            trie = vocabulary.all
        }
        if matcher.canEnd(grammar) { for id in stopTokens.ids { set.insert(id) } }
        var stack: [(node: Int32, matcher: LanguageJSONMatcher)] = [(0, matcher)]
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
