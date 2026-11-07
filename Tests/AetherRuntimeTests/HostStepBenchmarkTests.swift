import Foundation
import Testing
import AetherRuntime

/// The diffusion pipelines' host work per step (second code review, batch C item 3), timed outside the graphs: write the
/// denoiser's input, read its output, apply the Euler update. `before` is the path the pipelines took (a scaled input
/// array allocated per step, a validating pass then an update pass), `after` the one they take now (the input written
/// straight into the buffer, one validating pass into a reused scratch array). Opt-in: `AETHER_BENCH=1`.
@Suite("Host step benchmark", .enabled(if: ProcessInfo.processInfo.environment["AETHER_BENCH"] == "1"))
struct HostStepBenchmarkTests {
    static func time(_ iterations: Int, _ body: () throws -> Void) rethrows -> Duration {
        try body()  // warm
        let started = ContinuousClock.now
        for _ in 0..<iterations { try body() }
        return (ContinuousClock.now - started) / iterations
    }

    /// SD-Turbo at 512 × 512: `[1, 4, 64, 64]` float32 in and out.
    @Test func sdTurbo() throws {
        let count = 4 * 64 * 64
        let arena = TensorArena()
        let sample = try arena.allocate(name: "sample", shape: [1, 4, 64, 64], scalarType: .float32)
        let epsilon = try arena.allocate(name: "epsilon", shape: [1, 4, 64, 64], scalarType: .float32)
        try epsilon.write(try SeededGaussianNoise.generate(seed: 2, count: count).map { $0 * 1e-3 })
        let s = try EpsilonEulerSchedule(timesteps: [999], sigmas: [14.6, 0], initNoiseSigma: 14.7)
        let start = try SeededGaussianNoise.generate(seed: 1, count: count)
        var latents = start, noise = [Float](repeating: 0, count: count), next = noise
        let before = try Self.time(2_000) {
            try sample.write(try s.modelInput(step: 0, latents: latents))
            try epsilon.readFloats(0..<count, into: &noise)
            try s.advance(step: 0, latents: &latents, epsilon: noise)
        }
        let after = try Self.time(2_000) {
            let scale = try s.inputScale(step: 0)
            try latents.withUnsafeBufferPointer { x throws(AetherError) in
                try sample.write(Float.self, count: count) { x[$0] / scale }
            }
            try epsilon.readFloats(0..<count, into: &noise)
            try s.advance(step: 0, latents: &latents, epsilon: noise, scratch: &next)
        }
        print("bench: sd-turbo 512x512 host step: before \(before), after \(after)")
    }

    /// FLUX.2 [klein] and Krea at 1024 × 1024: 4096 tokens × 128 channels, float16 in and out.
    @Test func flowMatch() throws {
        let count = 4096 * 128
        let arena = TensorArena()
        let sample = try arena.allocate(name: "sample", shape: [4096, 128], scalarType: .float16)
        let velocity = try arena.allocate(name: "velocity", shape: [4096, 128], scalarType: .float16)
        try velocity.write(Float16.self, count: count) { Float16(Float($0 % 7) * 0.01) }
        let s = try FlowMatchSchedule(steps: 4, mu: 1.15)
        let start = try SeededGaussianNoise.generate(seed: 1, count: count)
        var latents = start, v = [Float](repeating: 0, count: count), next = v
        let before = try Self.time(100) {
            try sample.write(Float16.self, count: count) { Float16(latents[$0]) }
            try velocity.readFloats(0..<count, into: &v)
            try s.advance(step: 0, latents: &latents, velocity: v)
        }
        let after = try Self.time(100) {
            try latents.withUnsafeBufferPointer { x throws(AetherError) in
                try sample.write(Float16.self, count: count) { Float16(x[$0]) }
            }
            try velocity.readFloats(0..<count, into: &v)
            try s.advance(step: 0, latents: &latents, velocity: v, scratch: &next)
        }
        print("bench: flow-match 1024x1024 host step: before \(before), after \(after)")
    }
}
