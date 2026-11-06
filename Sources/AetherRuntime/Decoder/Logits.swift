import Accelerate

/// Logits for a contiguous run of positions, `rows × vocab`, row-major (DESIGN.md §6.2). A `Sendable` value: the
/// engine reads only the rows a caller asked for into it, and hands it out of its actor.
public struct Logits: Sendable, Hashable {
    /// The position of row 0 in the sequence.
    public var firstPosition: Int
    public var rows: Int
    public var vocab: Int
    public var values: [Float]

    public init(firstPosition: Int, rows: Int, vocab: Int, values: [Float]) {
        precondition(values.count == rows * vocab, "logits: \(values.count) values for \(rows) × \(vocab)")
        self.firstPosition = firstPosition
        self.rows = rows
        self.vocab = vocab
        self.values = values
    }

    public func row(_ r: Int) -> ArraySlice<Float> { values[(r * vocab)..<((r + 1) * vocab)] }

    /// The last row (the next-token distribution after the last fed token).
    public var last: ArraySlice<Float> { row(rows - 1) }

    /// The highest value's index; ties go to the lowest index (the GPU argmax kernel's rule too).
    public static func argmax(_ row: ArraySlice<Float>) -> Int32 {
        guard !row.isEmpty else { return 0 }
        // vDSP returns the first index of the maximum: the same tie rule, much faster on a 150k-value row (a debug
        // build's element loop cost milliseconds per token). Rows with NaNs are not ranked meaningfully either way.
        var value: Float = 0
        var index: vDSP_Length = 0
        row.withUnsafeBufferPointer { vDSP_maxvi($0.baseAddress!, 1, &value, &index, vDSP_Length($0.count)) }
        return Int32(index)
    }
}

/// Which logits rows a feed should read back. Reading is the expensive part of prefill (converting every row of a
/// 512-token chunk to `[Float]` made `run()` prefill 8× slower in the lab), so the default is the last row only.
public enum LogitsSelection: Sendable, Hashable {
    /// Read nothing (state only: a prompt prefix, a snapshot point).
    case none
    /// The row after the last fed token.
    case last
    /// One row per fed token. Only graphs whose logits keep every row (`rows: all`) or schedules that feed one token
    /// per call can provide it; anything else is a `manifestMismatch` (never a silent misread).
    case all
}
