// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit

/// SplitMix64 and double-precision Box–Muller in packed-token order (`SeededGaussianNoise`).
/// The same integer seed does not reproduce PyTorch's random generator.
public enum KreaNoise {
    public static func generate(seed: UInt64, count: Int) throws -> [Float] {
        try SeededGaussianNoise.generate(seed: seed, count: count)
    }
}
