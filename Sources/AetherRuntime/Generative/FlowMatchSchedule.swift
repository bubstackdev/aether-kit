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
    /// `generation(.invalidParameters)`, including a `mu` whose shift `exp(μ)` isn't a normal Float (about
    /// −87.3 < μ < 88.7): the shift is computed in Float for parity with the reference, and past that range it
    /// overflows to infinity (every sigma NaN) or underflows to zero.
    public init(steps: Int, mu: Double?, trainTimesteps: Int = 1000) throws(AetherError) {
        guard steps >= 1, trainTimesteps >= 1, mu?.isFinite ?? true else {
            throw .generation(.invalidParameters(
                "a flow-match schedule needs steps >= 1, trainTimesteps >= 1 and a finite mu, not \(steps), \(trainTimesteps), \(mu.map { "\($0)" } ?? "nil")"))
        }
        let stop = 1.0 / Double(steps)
        let delta = steps > 1 ? (stop - 1.0) / Double(steps - 1) : 0.0
        let e = mu.map { Float(exp($0)) }
        if let e, let mu, !e.isNormal {
            throw .generation(.invalidParameters("a flow-match shift exp(mu) must be a normal Float; mu \(mu) gives \(e)"))
        }
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

    private init(sigmas: [Float], timesteps: [Float], mu: Double?) {
        self.sigmas = sigmas
        self.timesteps = timesteps
        self.mu = mu
    }

    /// This schedule with its last step starting at `sigma` instead (an off-grid image-to-image start; every earlier
    /// step unchanged), its timestep `σ · T / T` as the others'. `sigma` must be finite, in (0, 1), and below the
    /// previous step's sigma (when there is one).
    public func replacingLastSigma(_ sigma: Float, trainTimesteps: Int = 1000) throws(AetherError) -> FlowMatchSchedule {
        let last = steps - 1
        guard steps >= 1, sigma.isFinite, sigma > 0, sigma < 1, last == 0 || sigma < sigmas[last - 1], trainTimesteps > 0 else {
            throw .generation(.invalidParameters("an off-grid last sigma must be in (0, 1) and below the step before it"))
        }
        var sigmas = self.sigmas, timesteps = self.timesteps
        let t = Float(trainTimesteps)
        sigmas[last] = sigma
        timesteps[last] = sigma * t / t
        return FlowMatchSchedule(sigmas: sigmas, timesteps: timesteps, mu: mu)
    }

    /// The flow-matching start of an image-to-image run at `step`: `(1 − σ) · x0 + σ · noise` per element, each product
    /// rounded before the add (0.0.x's `refineStart`). `step` must be at least 1 (step 0 is pure noise).
    public func refineStart(x0: [Float], noise: [Float], step: Int) throws(AetherError) -> [Float] {
        guard step >= 1, step < steps, x0.count == noise.count else {
            throw .generation(.invalidParameters("a refine starts at a step from 1 to \(steps - 1), with as much noise as latents"))
        }
        let sigma = sigmas[step]
        return zip(x0, noise).map { roundedProduct(1 - sigma, $0) + roundedProduct(sigma, $1) }
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

    /// One Euler step from `latents` into `next`, validating as it goes (one pass). On any failure `latents` is
    /// unchanged and `next` is partly written.
    public func advance(
        step: Int, latents: UnsafeBufferPointer<Float>, velocity: UnsafeBufferPointer<Float>,
        into next: UnsafeMutableBufferPointer<Float>
    ) throws(AetherError) {
        guard timesteps.indices.contains(step) else {
            throw .generation(.invalidParameters("step \(step) is outside 0..<\(steps)"))
        }
        guard latents.count == velocity.count, latents.count == next.count else {
            throw .tensor(name: "velocity", reason: .shapeMismatch(expected: [latents.count], found: [velocity.count]))
        }
        let delta = sigmas[step + 1] - sigmas[step]
        for i in latents.indices {
            let x = latents[i], v = velocity[i]
            let value = x + roundedProduct(delta, v)
            guard x.isFinite, v.isFinite, value.isFinite else {
                throw .tensor(name: v.isFinite ? "latents" : "velocity", reason: .unsupported("a non-finite value at element \(i) of the Euler step"))
            }
            next[i] = value
        }
    }

    /// One Euler step of `latents` through `scratch` (resized to fit and reused across steps): one pass, then the two
    /// arrays swap. On any failure `latents` is unchanged.
    public func advance(step: Int, latents: inout [Float], velocity: [Float], scratch: inout [Float]) throws(AetherError) {
        if scratch.count != latents.count { scratch = [Float](repeating: 0, count: latents.count) }
        try latents.withUnsafeBufferPointer { x throws(AetherError) in
            try velocity.withUnsafeBufferPointer { v throws(AetherError) in
                try scratch.withUnsafeMutableBufferPointer { n throws(AetherError) in try advance(step: step, latents: x, velocity: v, into: n) }
            }
        }
        swap(&latents, &scratch)
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
