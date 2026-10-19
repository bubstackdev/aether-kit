// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// The prototype's one deterministic stream (`krea::seeded_noise`): SplitMix64 and
/// double-precision Box–Muller, the same algorithm as `AetherKrea.KreaNoise`. Kept local
/// so AetherH3 stays independent of Krea. It does not reproduce PyTorch's generator;
/// exact PyTorch replay supplies original latent rows instead.
public enum H3Noise {
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

extension H3Geometry {
    /// Stereo audio latent rows: both channels' `audioLatentsPerChannel` rows of 32.
    public var audioRows: Int { audioLatentsPerChannel * 2 }

    /// Seeded initial latents: one stream, the contiguous source `[24,T,H,W]` video
    /// tensor first (then packed), followed by channel-major audio `[1, audioRows, 32]`.
    public func initialLatents(seed: UInt64) throws -> H3Latents {
        let videoCount = targetVideoRows * 96, audioCount = audioRows * 32
        let noise = try H3Noise.generate(seed: seed, count: videoCount + audioCount)
        let video = try packVideo(FloatTensor(shape: videoLatentShape, values: Array(noise[..<videoCount])))
        func tensor(_ shape: [Int], _ values: [Float]) throws -> H3TensorData {
            try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
        }
        return try H3Latents(video: tensor([1, targetVideoRows, 96], video.values),
                             audio: tensor([1, audioRows, 32], Array(noise[videoCount...])))
    }
}
