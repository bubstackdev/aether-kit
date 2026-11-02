import AetherKit
import Foundation

/// An explicit Euler epsilon-prediction schedule, captured from an exporter.
/// Flow matching, v-prediction, ancestral noise and classifier-free guidance need other solvers.
public struct EpsilonSchedule: Sendable, Codable, Equatable {
    public let timesteps: [Float]
    public let sigmas: [Float]
    public let initialNoiseSigma: Float

    public init(timesteps: [Float], sigmas: [Float], initialNoiseSigma: Float) throws {
        guard !timesteps.isEmpty, sigmas.count == timesteps.count + 1,
              timesteps.allSatisfy({ $0.isFinite && $0 >= 0 }),
              zip(timesteps, timesteps.dropFirst()).allSatisfy({ $0 > $1 }),
              sigmas.allSatisfy({ $0.isFinite && $0 >= 0 }), sigmas.last == 0,
              zip(sigmas, sigmas.dropFirst()).allSatisfy({ $0 > $1 }),
              initialNoiseSigma.isFinite, initialNoiseSigma > 0 else {
            throw AetherError.invalidRequest("invalid Euler epsilon schedule")
        }
        self.timesteps = timesteps; self.sigmas = sigmas; self.initialNoiseSigma = initialNoiseSigma
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(timesteps: c.decode([Float].self, forKey: .timesteps), sigmas: c.decode([Float].self, forKey: .sigmas),
                      initialNoiseSigma: c.decode(Float.self, forKey: .initialNoiseSigma))
    }

    public func scaledInput(_ latents: FloatTensor, step: Int) throws -> FloatTensor {
        guard timesteps.indices.contains(step) else { throw AetherError.invalidRequest("schedule step out of bounds") }
        let scale = sqrt(sigmas[step] * sigmas[step] + 1)
        return try FloatTensor(shape: latents.shape, values: latents.values.map { $0 / scale })
    }

    public func advance(_ latents: FloatTensor, epsilon: FloatTensor, step: Int) throws -> FloatTensor {
        guard timesteps.indices.contains(step), latents.shape == epsilon.shape else {
            throw AetherError.invalidTensor("invalid Euler step or epsilon shape")
        }
        // Match Diffusers' operation order, including its predicted-original subtraction.
        let sigma = sigmas[step], dt = sigmas[step + 1] - sigma
        return try FloatTensor(shape: latents.shape, values: zip(latents.values, epsilon.values).map { x, e in
            let original = x - sigma * e
            return x + ((x - original) / sigma) * dt
        })
    }
}
