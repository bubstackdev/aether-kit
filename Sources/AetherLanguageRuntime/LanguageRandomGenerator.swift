/// The sampler's seeded generator: xoshiro256** (Blackman and Vigna, 2018) with its state filled by SplitMix64 from
/// the 64-bit seed, as its authors recommend. Specified here rather than taken from the system, so a seed replays the
/// same draws on every OS and device.
public struct LanguageRandomGenerator: RandomNumberGenerator, Sendable {
    private var s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64

    public init(seed: UInt64) {
        var mix = SplitMix64(state: seed)
        s0 = mix.next(); s1 = mix.next(); s2 = mix.next(); s3 = mix.next()
    }

    public mutating func next() -> UInt64 {
        let result = rotate(s1 &* 5, 7) &* 9
        let t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = rotate(s3, 45)
        return result
    }

    /// A uniform draw in [0, 1) from the top 53 bits of one output.
    public mutating func nextUnitInterval() -> Double {
        Double(next() >> 11) * 0x1p-53
    }

    private func rotate(_ x: UInt64, _ k: UInt64) -> UInt64 { (x << k) | (x >> (64 - k)) }
}

/// SplitMix64 (Steele, Lea and Flood, 2014), used only to seed ``LanguageRandomGenerator``.
struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
