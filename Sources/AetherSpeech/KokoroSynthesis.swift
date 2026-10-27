import AetherKit
import Foundation

/// Per-request synthesis settings.
public struct KokoroSpeechConfiguration: Sendable, Equatable {
    public static let speedRange: ClosedRange<Float> = 0.25...4

    public let voice: KokoroVoice
    /// Divides predicted durations before rounding, as upstream's `speed` does.
    public let speed: Float
    /// Seeds the vocoder's harmonic-source noise. Upstream draws it from PyTorch's global RNG;
    /// a fixed seed here makes output reproducible. Segment `i` of a chunked request uses `seed &+ i`.
    public let seed: UInt64

    public init(voice: KokoroVoice = .afHeart, speed: Float = 1, seed: UInt64 = 0) throws {
        guard speed.isFinite, Self.speedRange.contains(speed) else {
            throw AetherError.invalidRequest("Kokoro speed must be within 0.25...4")
        }
        self.voice = voice
        self.speed = speed
        self.seed = seed
    }
}

/// Host steps between the duration and decoder graphs, matching upstream `forward_with_tokens`.
public enum KokoroAlignment {
    /// `round(sum / speed)` with ties to even (torch.round), at least one frame per token.
    public static func frames(durationSums: [Float], speed: Float, maximumFrames: Int) throws -> [Int] {
        guard !durationSums.isEmpty, durationSums.allSatisfy(\.isFinite), speed.isFinite, speed > 0 else {
            throw AetherError.invalidTensor("duration sums must be finite and speed positive")
        }
        let frames = durationSums.map { max(1, Int(($0 / speed).rounded(.toNearestOrEven))) }
        let total = frames.reduce(0, +)
        guard total <= maximumFrames else {
            throw AetherError.invalidRequest("utterance needs \(total) frames; at most \(maximumFrames) are supported, split the phonemes")
        }
        return frames
    }

    /// Gathers token rows into frame columns, equivalent to upstream's one-hot alignment matmul.
    /// prosody [1,T,C] -> [1,C,F]; text [1,C,T] -> [1,C,F].
    public static func align(prosody: FloatTensor, text: FloatTensor, frames: [Int]) throws -> (prosody: FloatTensor, text: FloatTensor) {
        let tokens = frames.count
        guard prosody.shape.count == 3, prosody.shape[0] == 1, prosody.shape[1] == tokens,
              text.shape.count == 3, text.shape[0] == 1, text.shape[2] == tokens, frames.allSatisfy({ $0 > 0 }) else {
            throw AetherError.invalidTensor("alignment expects prosody [1,T,C], text [1,C,T] and T positive frame counts")
        }
        let total = frames.reduce(0, +)
        var index: [Int] = []
        index.reserveCapacity(total)
        for (token, count) in frames.enumerated() { index.append(contentsOf: repeatElement(token, count: count)) }
        let pc = prosody.shape[2], tc = text.shape[1]
        let alignedProsody = [Float](unsafeUninitializedCapacity: pc * total) { out, initialized in
            prosody.values.withUnsafeBufferPointer { source in
                for c in 0..<pc {
                    for (f, token) in index.enumerated() { out[c * total + f] = source[token * pc + c] }
                }
            }
            initialized = pc * total
        }
        let alignedText = [Float](unsafeUninitializedCapacity: tc * total) { out, initialized in
            text.values.withUnsafeBufferPointer { source in
                for c in 0..<tc {
                    let row = c * tokens
                    for (f, token) in index.enumerated() { out[c * total + f] = source[row + token] }
                }
            }
            initialized = tc * total
        }
        return (try FloatTensor(shape: [1, pc, total], values: alignedProsody),
                try FloatTensor(shape: [1, tc, total], values: alignedText))
    }
}

/// SplitMix64 and double-precision Box-Muller (`SeededGaussianNoise`, the stream `KreaNoise` reads),
/// identical to the Forge reference harness. The same seed does not reproduce PyTorch's generator.
public enum KokoroNoise {
    public static func generate(seed: UInt64, count: Int) throws -> [Float] {
        try SeededGaussianNoise.generate(seed: seed, count: count)
    }
}
