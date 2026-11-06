public import AetherCore
import Foundation

/// A `flowMatchEuler` schedule (MANIFEST.md §5.2) for one request: its sigmas, the denoiser's timestep inputs and the
/// host's Euler update. The arithmetic order is the reference's, so the values are bit-identical to 0.0.x's Krea and
/// FLUX.2 schedules:
/// - base `linspace(1, 1/steps, steps)` in Double, each value rounded to Float, the last exactly `1/steps`;
/// - shifted in Float, `e / (e + (1/σ − 1))` with `e = Float(exp(μ))`; then the terminal 0;
/// - timestep `σ · T / T` in Float (`T` = `trainTimesteps`);
/// - update `x + (σ[i+1] − σ[i]) · v` in Float, the product rounded before the add.
public struct FlowMatchSchedule: Sendable, Hashable {
    /// `steps + 1` values, decreasing, ending in 0.
    public let sigmas: [Float]
    /// The denoiser's `sigma` input for each step.
    public let timesteps: [Float]
    /// The μ the sigmas were shifted with (nil for `rule: none`).
    public let mu: Double?

    public var steps: Int { timesteps.count }

    /// The schedule a manifest describes, for `steps` steps (default: the manifest's) and, for the empirical shift,
    /// `imageTokens` generated tokens. Request errors are `generation(.invalidParameters)`.
    public init(_ schedule: Manifest.Schedule, steps: Int? = nil, imageTokens: Int? = nil) throws(AetherError) {
        let steps = steps ?? schedule.steps
        guard (1...schedule.stepLimit).contains(steps) else {
            throw .generation(.invalidParameters("steps must be in 1...\(schedule.stepLimit), not \(steps)"))
        }
        guard schedule.kind == .flowMatchEuler, let shift = schedule.sigmas?.shift else {
            throw .manifestMismatch(field: "schedule.kind", expected: "flowMatchEuler", found: schedule.kind.rawValue)
        }
        let mu: Double?
        switch shift.rule {
        case .exponential:
            mu = shift.mu
        case .empirical:
            guard let empirical = shift.empirical else {
                throw .manifestInvalid(field: "schedule.sigmas.shift.empirical", reason: "rule empirical needs its coefficients")
            }
            guard let imageTokens, imageTokens >= 1 else {
                throw .generation(.invalidParameters("the empirical shift needs the generated image-token count"))
            }
            mu = empirical.mu(imageTokens: imageTokens, steps: steps)
        case .none:
            mu = nil
        }
        try self.init(steps: steps, mu: mu, trainTimesteps: schedule.trainTimesteps)
    }

    /// The schedule for `steps` ≥ 1 steps, shifted at a finite `mu` (unshifted when nil). Refuses other values with
    /// `generation(.invalidParameters)`.
    public init(steps: Int, mu: Double?, trainTimesteps: Int = 1000) throws(AetherError) {
        guard steps >= 1, trainTimesteps >= 1, mu?.isFinite ?? true else {
            throw .generation(.invalidParameters(
                "a flow-match schedule needs steps >= 1, trainTimesteps >= 1 and a finite mu, not \(steps), \(trainTimesteps), \(mu.map { "\($0)" } ?? "nil")"))
        }
        let stop = 1.0 / Double(steps)
        let delta = steps > 1 ? (stop - 1.0) / Double(steps - 1) : 0.0
        let e = mu.map { Float(exp($0)) }
        let t = Float(trainTimesteps)
        var sigmas: [Float] = (0..<steps).map { index in
            let base = Float(index == steps - 1 && steps > 1 ? stop : Double(index) * delta + 1.0)
            guard let e else { return base }
            return e / (e + (1 / base - 1))
        }
        timesteps = sigmas.map { $0 * t / t }
        sigmas.append(0)
        self.sigmas = sigmas
        self.mu = mu
    }

    /// One Euler step in place: `latents[i] += (σ[step+1] − σ[step]) · velocity[i]`. On any failure (a step out of
    /// range, mismatched counts, a non-finite input or result) `latents` is unchanged.
    public func advance(
        step: Int, latents: UnsafeMutableBufferPointer<Float>, velocity: UnsafeBufferPointer<Float>
    ) throws(AetherError) {
        guard timesteps.indices.contains(step) else {
            throw .generation(.invalidParameters("step \(step) is outside 0..<\(steps)"))
        }
        guard latents.count == velocity.count else {
            throw .tensor(name: "velocity", reason: .shapeMismatch(expected: [latents.count], found: [velocity.count]))
        }
        let delta = sigmas[step + 1] - sigmas[step]
        for i in latents.indices {
            let x = latents[i], v = velocity[i]
            guard x.isFinite, v.isFinite, (x + roundedProduct(delta, v)).isFinite else {
                throw .tensor(name: v.isFinite ? "latents" : "velocity", reason: .unsupported("a non-finite value at element \(i) of the Euler step"))
            }
        }
        for i in latents.indices { latents[i] = latents[i] + roundedProduct(delta, velocity[i]) }
    }

    /// ``advance(step:latents:velocity:)`` on arrays.
    public func advance(step: Int, latents: inout [Float], velocity: [Float]) throws(AetherError) {
        try velocity.withUnsafeBufferPointer { v throws(AetherError) in
            try latents.withUnsafeMutableBufferPointer { x throws(AetherError) in try advance(step: step, latents: x, velocity: v) }
        }
    }
}

/// Keeps the product rounded to Float before the sum (no fused multiply-add), including in optimized builds.
@inline(never)
func roundedProduct(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }
