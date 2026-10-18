// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// Reference CPU flow schedule. Arithmetic order preserves the existing fixtures.
public struct KreaSchedule: Sendable {
    public let sigmas: [Float]
    public let timesteps: [Float]

    public init(steps: Int = 8) throws {
        guard (1...50).contains(steps) else {
            throw AetherError.invalidRequest("Krea steps must be in 1...50")
        }
        let stop = 1.0 / Double(steps)
        let delta = steps > 1 ? (stop - 1.0) / Double(steps - 1) : 0.0
        let shift = Float(exp(1.15))
        var sigmas: [Float] = (0..<steps).map { index in
            let base = Float(index == steps - 1 && steps > 1 ? stop : Double(index) * delta + 1.0)
            return shift / (shift + (1.0 / base - 1.0))
        }
        self.timesteps = sigmas.map { $0 * 1000.0 / 1000.0 }
        sigmas.append(0)
        self.sigmas = sigmas
    }

    /// Returns a new latent buffer, leaving input buffers unchanged on any failure.
    public func advance(step: Int, latents: [Float], velocity: [Float]) throws -> [Float] {
        guard timesteps.indices.contains(step), latents.count == velocity.count else {
            throw AetherError.invalidRequest("invalid Euler step or mismatched latent/velocity counts")
        }
        guard latents.allSatisfy(\.isFinite), velocity.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("Euler inputs must be finite")
        }
        let delta = sigmas[step + 1] - sigmas[step]
        let result = zip(latents, velocity).map { $0 + roundedProduct(delta, $1) }
        guard result.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("Euler result is not finite")
        }
        return result
    }
}

// Keep multiplication separate from addition, including in optimized builds.
@inline(never)
func roundedProduct(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }
