/// One embedding vector (DESIGN.md §8): an image's or a text's, from AetherVision's `ImageTextEncoder` (CLIP) or
/// AetherEmbeddings' text embedders. Here, below both modules, so their vectors are one type without either depending
/// on the other.
public struct Embedding: Sendable, Hashable {
    public let vector: [Float]

    public init(_ vector: [Float]) { self.vector = vector }

    public var dimension: Int { vector.count }

    /// The dot product, in Float with a Double accumulator. The vectors must have the same dimension.
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

    /// Cosine similarity, in [-1, 1]; 0 when either vector is zero. For unit-length vectors (CLIP's, normalized in its
    /// graphs) it is the dot product.
    public func similarity(_ other: Embedding) -> Float {
        let norms = norm * other.norm
        return norms > 0 ? dot(other) / norms : 0
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
