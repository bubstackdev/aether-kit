// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// SplitMix64 and double-precision Box–Muller in packed-token order.
/// The same integer seed does not reproduce PyTorch's random generator.
public enum KreaNoise {
    public static func generate(seed: UInt64, count: Int) throws -> [Float] {
        guard count >= 0, count <= Int.max / MemoryLayout<Float>.stride else {
            throw AetherError.invalidRequest("invalid noise element count")
        }
        var state = seed
        func next() -> UInt64 {
            state &+= 0x9e3779b97f4a7c15
            var value = state
            value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
            value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
            return value ^ (value >> 31)
        }
        var values: [Float] = []
        values.reserveCapacity(count)
        while values.count < count {
            let u1 = (Double(next() >> 11) + 1) / 9007199254740992
            let u2 = Double(next() >> 11) / 9007199254740992
            let radius = sqrt(-2 * log(u1))
            let angle = 2 * Double.pi * u2
            values.append(Float(radius * cos(angle)))
            if values.count < count { values.append(Float(radius * sin(angle))) }
        }
        return values
    }
}
