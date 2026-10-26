// Derived from the Apache-2.0 Aether prototype and pinned Diffusers oracle. See NOTICE.
import AetherKit
import Foundation

/// Captured Turbo8 schedule. Custom grids and base50 execution are not exposed.
public struct H3Schedule: Sendable {
    public enum Modality: Sendable { case video, audio }
    public struct Step: Sendable {
        public let index: Int
        public let videoTimestep: Float
        public let audioTimestep: Float
    }
    private struct Arm: Decodable, Sendable { let shift: Float; let timesteps: [Float]; let sigmas: [Float] }
    private struct Record: Decodable { let gridPoints: Int; let expectedForwards: Int; let schedules: [Arm] }
    private let video: Arm
    private let audio: Arm
    public var steps: [Step] {
        (0..<8).map { Step(index: $0, videoTimestep: video.timesteps[$0], audioTimestep: audio.timesteps[$0]) }
    }

    public init() throws {
        guard let url = Bundle.module.url(forResource: "turbo8-schedule", withExtension: "json", subdirectory: "Resources") else {
            throw AetherError.invalidModel("missing H3 Turbo8 schedule resource")
        }
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: url))
        guard record.gridPoints == 9, record.expectedForwards == 8, record.schedules.count == 2,
              record.schedules[0].shift == 12, record.schedules[1].shift == 3 else {
            throw AetherError.invalidModel("invalid H3 Turbo8 schedule")
        }
        for arm in record.schedules {
            guard arm.timesteps.count == 8, arm.sigmas.count == 9, arm.sigmas.first == 1, arm.sigmas.last == 0,
                  arm.sigmas.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  zip(arm.sigmas, arm.sigmas.dropFirst()).allSatisfy({ $0 > $1 }),
                  zip(arm.timesteps, arm.sigmas).allSatisfy({ $0.bitPattern == (1 - $1).bitPattern }) else {
                throw AetherError.invalidModel("invalid H3 sigma/timestep grid")
            }
        }
        video = record.schedules[0]; audio = record.schedules[1]
    }

    /// Original data-ward x0 blend with separately rounded float32 products.
    /// This is different from Krea's signed Euler update. Inputs remain unchanged on failure.
    public func advance(_ modality: Modality, step: Int, sample: [Float], velocity: [Float]) throws -> [Float] {
        guard (0..<8).contains(step), !sample.isEmpty, sample.count == velocity.count else {
            throw AetherError.invalidRequest("invalid H3 step or mismatched sample/velocity counts")
        }
        guard sample.allSatisfy(\.isFinite), velocity.allSatisfy(\.isFinite) else {
            throw AetherError.invalidTensor("H3 scheduler inputs must be finite")
        }
        let arm = modality == .video ? video : audio
        let sigma = 1 - arm.timesteps[step]
        let ratio = arm.sigmas[step + 1] / arm.sigmas[step]
        let result = zip(sample, velocity).map { value, velocity in
            let denoised = value + product(sigma, velocity)
            return product(ratio, value) + product(1 - ratio, denoised)
        }
        guard result.allSatisfy(\.isFinite) else { throw AetherError.invalidTensor("nonfinite H3 scheduler output") }
        return result
    }
}

@inline(never) private func product(_ lhs: Float, _ rhs: Float) -> Float { lhs * rhs }
