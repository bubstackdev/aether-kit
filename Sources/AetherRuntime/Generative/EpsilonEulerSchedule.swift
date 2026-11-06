public import AetherCore

/// An `epsilonEuler` schedule (MANIFEST.md §5.2) for one request: Diffusers' `EulerDiscreteScheduler` with an
/// epsilon-predicting denoiser and no stochasticity (gamma 0). The table is the pinned scheduler's own; the arithmetic is
/// in Float in Diffusers' order, each product rounded before it is used (no fused multiply-add):
/// - initial latents `noise · initNoiseSigma`;
/// - model input `x / sqrt(σ[i]² + 1)`;
/// - update: original `x − σ[i]·ε`, derivative `(x − original) / σ[i]`, then `x + derivative · (σ[i+1] − σ[i])`.
public struct EpsilonEulerSchedule: Sendable, Hashable {
    public let timesteps: [Float]
    /// `steps + 1` values, decreasing, ending in 0.
    public let sigmas: [Float]
    public let initNoiseSigma: Float

    public var steps: Int { timesteps.count }

    /// The manifest's table for `steps` steps (default: the manifest's). Request errors are
    /// `generation(.invalidParameters)`.
    public init(_ schedule: Manifest.Schedule, steps: Int? = nil) throws(AetherError) {
        guard schedule.kind == .epsilonEuler else {
            throw .manifestMismatch(field: "schedule.kind", expected: "epsilonEuler", found: schedule.kind.rawValue)
        }
        let table = try schedule.table(steps: steps ?? schedule.steps)
        self.init(unchecked: table)
    }

    /// A schedule from explicit values. Fails unless they form a valid table (as the manifest requires).
    public init(timesteps: [Float], sigmas: [Float], initNoiseSigma: Float) throws(AetherError) {
        let table = Manifest.Schedule.Table(timesteps: timesteps, sigmas: sigmas, initNoiseSigma: initNoiseSigma)
        do {
            try table.validate(at: "schedule", steps: timesteps.count)
        } catch {
            throw .generation(.invalidParameters("not a valid Euler epsilon schedule: \(error)"))
        }
        self.init(unchecked: table)
    }

    private init(unchecked table: Manifest.Schedule.Table) {
        timesteps = table.timesteps
        sigmas = table.sigmas
        initNoiseSigma = table.initNoiseSigma
    }

    /// The initial latents in place: `noise · initNoiseSigma`.
    public func scaleInitialNoise(_ latents: UnsafeMutableBufferPointer<Float>) {
        for i in latents.indices { latents[i] = latents[i] * initNoiseSigma }
    }

    /// The denoiser's input for `step`: `latents / sqrt(σ² + 1)` into `input`.
    public func modelInput(
        step: Int, latents: UnsafeBufferPointer<Float>, into input: UnsafeMutableBufferPointer<Float>
    ) throws(AetherError) {
        try check(step)
        guard latents.count == input.count else {
            throw .tensor(name: "latents", reason: .shapeMismatch(expected: [input.count], found: [latents.count]))
        }
        let sigma = sigmas[step]
        let scale = (roundedProduct(sigma, sigma) + 1).squareRoot()
        for i in latents.indices { input[i] = latents[i] / scale }
    }

    /// One Euler step in place. On any failure (a step out of range, mismatched counts, a non-finite input or result)
    /// `latents` is unchanged.
    public func advance(
        step: Int, latents: UnsafeMutableBufferPointer<Float>, epsilon: UnsafeBufferPointer<Float>
    ) throws(AetherError) {
        try check(step)
        guard latents.count == epsilon.count else {
            throw .tensor(name: "epsilon", reason: .shapeMismatch(expected: [latents.count], found: [epsilon.count]))
        }
        let sigma = sigmas[step], dt = sigmas[step + 1] - sigma
        @inline(__always) func next(_ x: Float, _ e: Float) -> Float {
            let original = x - roundedProduct(sigma, e)
            return x + roundedProduct((x - original) / sigma, dt)
        }
        for i in latents.indices {
            let x = latents[i], e = epsilon[i]
            guard x.isFinite, e.isFinite, next(x, e).isFinite else {
                throw .tensor(name: e.isFinite ? "latents" : "epsilon", reason: .unsupported("a non-finite value at element \(i) of the Euler step"))
            }
        }
        for i in latents.indices { latents[i] = next(latents[i], epsilon[i]) }
    }

    /// ``advance(step:latents:epsilon:)`` on arrays.
    public func advance(step: Int, latents: inout [Float], epsilon: [Float]) throws(AetherError) {
        try epsilon.withUnsafeBufferPointer { e throws(AetherError) in
            try latents.withUnsafeMutableBufferPointer { x throws(AetherError) in try advance(step: step, latents: x, epsilon: e) }
        }
    }

    /// ``modelInput(step:latents:into:)`` on arrays.
    public func modelInput(step: Int, latents: [Float]) throws(AetherError) -> [Float] {
        var out = [Float](repeating: 0, count: latents.count)
        try latents.withUnsafeBufferPointer { x throws(AetherError) in
            try out.withUnsafeMutableBufferPointer { o throws(AetherError) in try modelInput(step: step, latents: x, into: o) }
        }
        return out
    }

    private func check(_ step: Int) throws(AetherError) {
        guard timesteps.indices.contains(step) else {
            throw .generation(.invalidParameters("step \(step) is outside 0..<\(steps)"))
        }
    }
}
