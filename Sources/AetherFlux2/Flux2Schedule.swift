// Derived from the Apache-2.0 Aether prototype (11 §B3, F-P3-51). See NOTICE.
import AetherKit
import Foundation

/// The klein flow-matching schedule. μ is the pipeline's empirical rule of the GENERATED token count
/// and the step count (reference tokens never change it); sigmas are `linspace(1, 1/steps, steps)` in
/// Double rounded to Float, shifted in Float by `e/(e + (1/σ − 1))` with `e = Float(e^μ)`, then a
/// final 0. The denoiser's timestep input is `Float(σ·1000)/1000`.
public struct Flux2Schedule: Sendable, Equatable {
    public struct EmpiricalShift: Sendable, Equatable {
        public let lowSteps: Int, lowSlope: Double, lowIntercept: Double
        public let highSteps: Int, highSlope: Double, highIntercept: Double
        public let tokenThreshold: Int
        public static let klein = EmpiricalShift(lowSteps: 10, lowSlope: 8.73809524e-05, lowIntercept: 1.89833333,
                                                 highSteps: 200, highSlope: 0.00016927, highIntercept: 0.45666666,
                                                 tokenThreshold: 4300)

        /// Evaluated in Double in the reference's operation order.
        public func mu(imageTokens n: Int, steps s: Int) -> Double {
            let high = highSlope * Double(n) + highIntercept
            if n > tokenThreshold { return high }
            let low = lowSlope * Double(n) + lowIntercept
            let slope = (high - low) / Double(highSteps - lowSteps)
            let intercept = high - Double(highSteps) * slope
            return slope * Double(s) + intercept
        }
    }

    public let mu: Double
    /// `steps + 1` values ending in 0.
    public let sigmas: [Float]
    /// Denoiser `sigma` inputs.
    public let timesteps: [Float]

    public init(steps: Int, imageTokens: Int, shift: EmpiricalShift = .klein) throws {
        guard (1...50).contains(steps), imageTokens >= 1 else {
            throw AetherError.invalidRequest("FLUX.2 schedule needs 1...50 steps and at least one token")
        }
        mu = shift.mu(imageTokens: imageTokens, steps: steps)
        let stop = 1.0 / Double(steps)
        let delta = steps > 1 ? (stop - 1.0) / Double(steps - 1) : 0.0
        let e = Float(exp(mu))
        var sigmas: [Float] = (0..<steps).map { index in
            let base = Float(index == steps - 1 && steps > 1 ? stop : Double(index) * delta + 1.0)
            return e / (e + (1 / base - 1))
        }
        timesteps = sigmas.map { $0 * 1000 / 1000 }
        sigmas.append(0)
        self.sigmas = sigmas
    }

    /// Euler `x + (σ[i+1] − σ[i])·v` in Float, rounding the product before the sum (no FMA).
    /// Returns a new buffer; inputs are unchanged on failure.
    public func advance(step: Int, latents: [Float], velocity: [Float]) throws -> [Float] {
        guard timesteps.indices.contains(step), latents.count == velocity.count else {
            throw AetherError.invalidRequest("invalid Euler step or mismatched latent/velocity counts")
        }
        let delta = sigmas[step + 1] - sigmas[step]
        var result = [Float](repeating: 0, count: latents.count)
        for index in latents.indices {
            result[index] = latents[index] + separatelyRounded(delta, velocity[index])
        }
        guard result.allSatisfy(\.isFinite) else { throw AetherError.invalidTensor("Euler result is not finite") }
        return result
    }
}

@inline(never)
func separatelyRounded(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }
