/// One embedding vector, as the embedding drivers return it (`text-embedder@1` in AetherEmbeddings, `clip-joint@1` in
/// AetherVision). Here, one layer below both, so their vectors compare with each other's (DESIGN.md §8).
public struct Embedding: Sendable, Hashable {
    public var vector: [Float]

    public init(_ vector: [Float]) { self.vector = vector }

    public var dimensions: Int { vector.count }

    /// Cosine similarity, in [-1, 1]: for unit vectors (every manifest that normalizes, in the graph or on the host),
    /// the dot product. 0 when either vector is zero. Both must have the same dimensions.
    public func similarity(_ other: Embedding) -> Float {
        precondition(vector.count == other.vector.count, "embeddings of \(vector.count) and \(other.vector.count) dimensions")
        var dot = 0.0, a = 0.0, b = 0.0
        for (x, y) in zip(vector, other.vector) {
            dot += Double(x) * Double(y)
            a += Double(x) * Double(x)
            b += Double(y) * Double(y)
        }
        let norms = (a * b).squareRoot()
        return norms > 0 ? Float(dot / norms) : 0
    }
}
