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
    /// The keyframe noise level the released model holds its anchors at (`keyframe_noise_aug`).
    public static let keyframeNoiseLevel: Float = 0.999

    /// FL2VA: like `initialLatents(seed:)`, but the stream draws the first keyframe's condition noise `[24,1,H,W]` first
    /// (the pipeline draws each condition before the generated rows), mixes it into the normalized condition latents at
    /// `t = 0.999` (`t * x + (1 - t) * noise`, float32) and packs those rows in front of the generated video rows.
    public func initialLatents(seed: UInt64, firstFrameCondition condition: H3TensorData) throws -> H3Latents {
        let frame = 24 * latentHeight * latentWidth
        guard condition.spec == (try H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, latentHeight, latentWidth])) else {
            throw AetherError.invalidTensor("H3 first-frame condition must be [1,24,1,H,W] float32")
        }
        let videoCount = targetVideoRows * 96, audioCount = audioRows * 32
        let noise = try H3Noise.generate(seed: seed, count: frame + videoCount + audioCount)
        let clean = try condition.floatValues(), t = H3Geometry.keyframeNoiseLevel
        let one = Float(1) - t
        let noised = (0..<frame).map { t * clean[$0] + one * noise[$0] }
        let conditionRows = try packFrames(noised, latentFrames: 1)
        let target = try packVideo(FloatTensor(shape: videoLatentShape, values: Array(noise[frame..<(frame + videoCount)])))
        func tensor(_ shape: [Int], _ values: [Float]) throws -> H3TensorData {
            try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
        }
        return try H3Latents(video: tensor([1, conditionRows.count / 96 + targetVideoRows, 96], conditionRows + target.values),
                             audio: tensor([1, audioRows, 32], Array(noise[(frame + videoCount)...])))
    }

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
