/// One embedding vector, as the embedding drivers return it (`text-embedder@1` in AetherEmbeddings, `clip-joint@1` in
/// AetherVision): an image's or a text's. Here, one layer below both, so their vectors are one type and compare with
/// each other's without either module depending on the other (DESIGN.md §8).
public struct Embedding: Sendable, Hashable {
    public var vector: [Float]

    public init(_ vector: [Float]) { self.vector = vector }

    /// The number of components.
    public var dimension: Int { vector.count }

    /// The dot product, in Float with a Double accumulator. Both must have the same dimension.
    public func dot(_ other: Embedding) -> Float {
        precondition(vector.count == other.vector.count, "embeddings of dimension \(vector.count) and \(other.vector.count)")
        var sum = 0.0
        for (a, b) in zip(vector, other.vector) { sum += Double(a) * Double(b) }
        return Float(sum)
    }

    /// The Euclidean length.
    public var norm: Float {
        Float(vector.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot())
    }

    /// Cosine similarity, in [-1, 1]: for unit vectors (every manifest that normalizes, in the graph or on the host,
    /// CLIP's included) the dot product. 0 when either vector is zero. Both must have the same dimension.
    public func similarity(_ other: Embedding) -> Float {
        precondition(vector.count == other.vector.count, "embeddings of dimension \(vector.count) and \(other.vector.count)")
        var dot = 0.0, a = 0.0, b = 0.0
        for (x, y) in zip(vector, other.vector) {
            dot += Double(x) * Double(y)
            a += Double(x) * Double(x)
            b += Double(y) * Double(y)
        }
        let norms = (a * b).squareRoot()
        return norms > 0 ? Float(dot / norms) : 0
    }

    /// The vector scaled to unit length (unchanged when it is zero).
    public var normalized: Embedding {
        let n = norm
        return n > 0 ? Embedding(vector.map { $0 / n }) : self
    }
}

/// A label and its score: a zero-shot classifier's probability over the labels it was given (`ImageTextEncoder`;
/// AetherEmbeddings' classifiers to come).
public struct LabelScore: Sendable, Hashable {
    public let label: String
    public let score: Float

    public init(label: String, score: Float) {
        self.label = label
        self.score = score
    }
}
