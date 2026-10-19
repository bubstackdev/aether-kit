// Derived from the Apache-2.0 Aether prototype and pinned Diffusers oracle. See NOTICE.
import AetherKit
import Foundation

/// Turbo schedules: the captured Turbo8 grid (`init()`), or a generated few-step grid
/// (`turbo(steps:videoShift:audioShift:)`) from the same shifted linear formula, which
/// reproduces the captured grid bit for bit. base50 execution is not exposed.
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
    public var count: Int { video.timesteps.count }
    public var videoShift: Float { video.shift }
    public var audioShift: Float { audio.shift }
    public var steps: [Step] {
        (0..<count).map { Step(index: $0, videoTimestep: video.timesteps[$0], audioTimestep: audio.timesteps[$0]) }
    }
    public func sigmas(_ modality: Modality) -> [Float] { modality == .video ? video.sigmas : audio.sigmas }

    public init() throws {
        guard let url = Bundle.module.url(forResource: "turbo8-schedule", withExtension: "json", subdirectory: "Resources") else {
            throw AetherError.invalidModel("missing H3 Turbo8 schedule resource")
        }
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: url))
        guard record.gridPoints == 9, record.expectedForwards == 8, record.schedules.count == 2,
              record.schedules[0].shift == 12, record.schedules[1].shift == 3 else {
            throw AetherError.invalidModel("invalid H3 Turbo8 schedule")
        }
        guard record.schedules.allSatisfy({ $0.timesteps.count == 8 }) else { throw AetherError.invalidModel("invalid H3 sigma/timestep grid") }
        try self.init(video: record.schedules[0], audio: record.schedules[1])
    }

    /// The Turbo adapters' grid: unshifted q_i = (N − i)/N, sigma = s·q / (1 + (s − 1)·q)
    /// in double precision rounded to float32, then 0; timestep = 1 − sigma in float32.
    /// Published pairs: 544p adapters 12/3, 768p adapters 6/3 (video/audio).
    public static func turbo(steps: Int, videoShift: Float, audioShift: Float) throws -> H3Schedule {
        guard (1...50).contains(steps), videoShift >= 1, audioShift >= 1, videoShift.isFinite, audioShift.isFinite else {
            throw AetherError.invalidRequest("H3 Turbo schedules need 1...50 steps and finite shifts >= 1")
        }
        func arm(_ shift: Float) -> Arm {
            let s = Double(shift)
            let sigmas = (0..<steps).map { i -> Float in
                let q = Double(steps - i) / Double(steps)
                return Float(s * q / (1 + (s - 1) * q))
            } + [0]
            return Arm(shift: shift, timesteps: sigmas.dropLast().map { 1 - $0 }, sigmas: sigmas)
        }
        return try H3Schedule(video: arm(videoShift), audio: arm(audioShift))
    }

    private init(video: Arm, audio: Arm) throws {
        for arm in [video, audio] {
            guard !arm.timesteps.isEmpty, arm.sigmas.count == arm.timesteps.count + 1, arm.sigmas.first == 1, arm.sigmas.last == 0,
                  arm.sigmas.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
                  zip(arm.sigmas, arm.sigmas.dropFirst()).allSatisfy({ $0 > $1 }),
                  zip(arm.timesteps, arm.sigmas).allSatisfy({ $0.bitPattern == (1 - $1).bitPattern }) else {
                throw AetherError.invalidModel("invalid H3 sigma/timestep grid")
            }
        }
        guard video.timesteps.count == audio.timesteps.count else { throw AetherError.invalidModel("H3 video/audio step counts differ") }
        self.video = video; self.audio = audio
    }

    /// Original data-ward x0 blend with separately rounded float32 products.
    /// This is different from Krea's signed Euler update. Inputs remain unchanged on failure.
    public func advance(_ modality: Modality, step: Int, sample: [Float], velocity: [Float]) throws -> [Float] {
        guard (0..<count).contains(step), !sample.isEmpty, sample.count == velocity.count else {
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
