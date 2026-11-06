public import AetherCore

/// The host steps between Kokoro's durations and decoder graphs (`kokoro.durations`, MANIFEST.md §5.9), matching
/// upstream's `forward_with_tokens`.
public enum KokoroAlignment {
    /// `max(1, round(sum / speed))` per token, ties to even (torch.round's rule), in float32 as the graph emits it.
    /// Refuses a total above `maximumFrames`, checked before converting or adding so an extreme finite duration
    /// throws rather than traps.
    public static func frames(durationSums: [Float], speed: Float, maximumFrames: Int) throws(AetherError) -> [Int] {
        guard !durationSums.isEmpty, durationSums.allSatisfy(\.isFinite), speed.isFinite, speed > 0 else {
            throw .tensor(name: "durations", reason: .unsupported("duration sums must be finite and the speed positive"))
        }
        func tooLong(_ total: String) -> AetherError {
            .generation(.invalidParameters(
                "the utterance needs \(total) frames; at most \(maximumFrames) are supported, split the phonemes"))
        }
        var frames: [Int] = []
        frames.reserveCapacity(durationSums.count)
        var total = 0
        for sum in durationSums {
            let rounded = (sum / speed).rounded(.toNearestOrEven)
            // Every finite Float below 2^63 converts to Int; the division can also overflow to infinity.
            guard rounded < Float(Int.max) else { throw tooLong("more than \(Int.max)") }
            let count = rounded < 1 ? 1 : Int(rounded)
            guard count <= maximumFrames - total else {
                let (needed, overflow) = total.addingReportingOverflow(count)
                throw tooLong(overflow ? "more than \(Int.max)" : "\(needed)")
            }
            total += count
            frames.append(count)
        }
        return frames
    }

    /// Token columns repeated into frame columns, zero-padded to `bucket` frames (upstream's one-hot alignment
    /// matmul, then the bucket's padding).
    /// - prosody `[1, T, C]` (token-major) → `[1, C, bucket]`;
    /// - text `[1, C, T]` (channel-major) → `[1, C, bucket]`.
    public static func align(
        prosody: [Float], prosodyChannels pc: Int, text: [Float], textChannels tc: Int, frames: [Int], bucket: Int
    ) throws(AetherError) -> (prosody: [Float], text: [Float]) {
        let tokens = frames.count
        let total = frames.reduce(0, +)
        guard tokens > 0, frames.allSatisfy({ $0 > 0 }), prosody.count == tokens * pc, text.count == tc * tokens,
              total <= bucket else {
            throw .tensor(name: "alignment", reason: .unsupported(
                "alignment needs prosody [1, T, C], text [1, C, T], T positive frame counts and a bucket that holds them"))
        }
        var index: [Int] = []
        index.reserveCapacity(total)
        for (token, count) in frames.enumerated() { index.append(contentsOf: repeatElement(token, count: count)) }
        var alignedProsody = [Float](repeating: 0, count: pc * bucket)
        var alignedText = [Float](repeating: 0, count: tc * bucket)
        for c in 0..<pc {
            for (f, token) in index.enumerated() { alignedProsody[c * bucket + f] = prosody[token * pc + c] }
        }
        for c in 0..<tc {
            let row = c * tokens
            for (f, token) in index.enumerated() { alignedText[c * bucket + f] = text[row + token] }
        }
        return (alignedProsody, alignedText)
    }
}
