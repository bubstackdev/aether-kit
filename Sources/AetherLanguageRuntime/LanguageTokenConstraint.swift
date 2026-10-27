/// Computes, for every sampled token, which tokens may follow: a grammar or JSON-schema matcher for structured output.
///
/// The constraint on token t+1 depends only on tokens up to t, all known once t is submitted, so each call runs in a
/// child task concurrently with the GPU step whose logits its result constrains; the decode loop applies the result
/// after that step joins. Work up to the step's duration is hidden; only the excess adds to each token (and keeps the
/// request's GPU grant held). A processor holds one request's state: `begin` resets it.
///
/// Failures follow token delivery's rules: a throwing call has its overlapping step joined first, that step's result
/// is discarded and the request throws; a failing step cancels and awaits the call.
public protocol LanguageLogitsProcessor: Sendable {
    /// The constraint on the first sampled token. Runs concurrently with the prompt's prefill.
    func begin(prompt: [Int]) async throws -> LanguageTokenConstraint
    /// Records `token` (just sampled and now being submitted) and returns the constraint on the token after it. Runs
    /// concurrently with `token`'s decode step. Not called for a stop token or for the last token at the output limit.
    func accept(_ token: Int) async throws -> LanguageTokenConstraint
}

public enum LanguageTokenConstraint: Sendable, Equatable {
    case unconstrained
    /// Only these tokens may be sampled; every other logit becomes −∞ before temperature and truncation.
    case allowed(LanguageTokenSet)

    /// Both constraints at once (for example a structured-output grammar within the tokenizer's vocabulary).
    public func intersection(_ other: LanguageTokenConstraint) -> LanguageTokenConstraint {
        switch (self, other) {
        case (.unconstrained, _): other
        case (_, .unconstrained): self
        case (.allowed(let a), .allowed(let b)): .allowed(a.intersection(b))
        }
    }
}

/// A set of token IDs over one vocabulary, stored as a bitset (19 KiB for a 151,936-row head).
public struct LanguageTokenSet: Sendable, Hashable {
    public let vocabularySize: Int
    private var words: [UInt64]

    /// The empty set.
    public init(vocabularySize: Int) {
        precondition(vocabularySize >= 0)
        self.vocabularySize = vocabularySize
        words = [UInt64](repeating: 0, count: (vocabularySize + 63) / 64)
    }

    /// IDs outside `0..<vocabularySize` are ignored.
    public init(_ ids: some Sequence<Int>, vocabularySize: Int) {
        self.init(vocabularySize: vocabularySize)
        for id in ids { insert(id) }
    }

    /// The IDs in `range` (clamped to the vocabulary), for example the tokenizer's own rows of a padded head.
    public init(range: Range<Int>, vocabularySize: Int) {
        self.init(vocabularySize: vocabularySize)
        let range = range.clamped(to: 0..<vocabularySize)
        guard !range.isEmpty else { return }
        for word in (range.lowerBound >> 6)...((range.upperBound - 1) >> 6) {
            let low = max(range.lowerBound - word << 6, 0), high = min(range.upperBound - word << 6, 64)
            words[word] = (high == 64 ? ~0 : (1 << UInt64(high)) - 1) & ~((1 << UInt64(low)) - 1)
        }
    }

    public func contains(_ id: Int) -> Bool {
        id >= 0 && id < vocabularySize && words[id >> 6] & (1 << UInt64(id & 63)) != 0
    }

    public mutating func insert(_ id: Int) {
        guard id >= 0, id < vocabularySize else { return }
        words[id >> 6] |= 1 << UInt64(id & 63)
    }

    public mutating func remove(_ id: Int) {
        guard id >= 0, id < vocabularySize else { return }
        words[id >> 6] &= ~(1 << UInt64(id & 63))
    }

    public var count: Int { words.reduce(0) { $0 + $1.nonzeroBitCount } }
    public var isEmpty: Bool { words.allSatisfy { $0 == 0 } }

    /// The IDs in ascending order.
    public var ids: [Int] {
        var out: [Int] = []
        for (index, word) in words.enumerated() {
            var bits = word
            while bits != 0 {
                out.append(index << 6 + bits.trailingZeroBitCount)
                bits &= bits - 1
            }
        }
        return out
    }

    /// Sets over different vocabularies intersect over the smaller one.
    public func intersection(_ other: LanguageTokenSet) -> LanguageTokenSet {
        var out = LanguageTokenSet(vocabularySize: min(vocabularySize, other.vocabularySize))
        for index in out.words.indices { out.words[index] = words[index] & other.words[index] }
        return out
    }

    /// Sets `logits[i]` to −∞ for every row not in the set. `logits.count` must equal `vocabularySize`.
    func mask(_ logits: inout [Float]) {
        precondition(logits.count == vocabularySize)
        logits.withUnsafeMutableBufferPointer { row in
            for (index, word) in words.enumerated() where word != ~0 {
                let base = index << 6
                for bit in 0..<min(64, vocabularySize - base) where word & (1 << UInt64(bit)) == 0 {
                    row[base + bit] = -.infinity
                }
            }
        }
    }
}
