import Foundation

/// The seeded Gaussian stream of the generative drivers (MANIFEST.md §5.2 `latent.noise: splitmix64BoxMuller`):
/// SplitMix64, with double-precision Box–Muller over consecutive pairs of draws, rounded to Float.
///
/// - **0.0.x-compatible**: bit for bit the stream of 0.0.x's Krea, FLUX.2, H3 and Kokoro noise (they shared one
///   generator), so a seed reproduces a 0.0.x image's initial latents.
/// - **Prefix-stable**: value `i` doesn't depend on how many values are drawn, so drivers that draw several tensors from
///   one stream (H3's keyframe noise, then video, then audio) read consecutive ranges of a single call.
/// - It does not reproduce PyTorch's generator for the same seed; parity gates feed the reference's own noise.
public enum SeededGaussianNoise {
    /// `count` values from `seed`.
    public static func generate(seed: UInt64, count: Int) throws(AetherError) -> [Float] {
        guard count >= 0, count <= Int.max / MemoryLayout<Float>.stride else {
            throw .generation(.invalidParameters("noise count must be in 0...\(Int.max / MemoryLayout<Float>.stride), not \(count)"))
        }
        return [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
            fill(buffer, seed: seed)
            initialized = count
        }
    }

    /// Writes the stream's first `buffer.count` values into `buffer` (a latent tensor's storage, say).
    public static func fill(_ buffer: UnsafeMutableBufferPointer<Float>, seed: UInt64) {
        var state = seed
        func next() -> UInt64 {
            state &+= 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z >> 31)
        }
        var i = 0
        while i < buffer.count {
            let u1 = (Double(next() >> 11) + 1) / 9_007_199_254_740_992
            let u2 = Double(next() >> 11) / 9_007_199_254_740_992
            let radius = (-2 * Foundation.log(u1)).squareRoot()
            let angle = 2 * Double.pi * u2
            buffer[i] = Float(radius * cos(angle))
            if i + 1 < buffer.count { buffer[i + 1] = Float(radius * sin(angle)) }
            i += 2
        }
    }
}
