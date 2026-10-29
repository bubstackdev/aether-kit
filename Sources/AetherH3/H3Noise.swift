// Derived from the Apache-2.0 Aether prototype. See NOTICE.
import AetherKit
import Foundation

/// The prototype's one deterministic stream (`krea::seeded_noise`): SplitMix64 and
/// double-precision Box–Muller, read from the shared `AetherKit.SeededGaussianNoise` (the same
/// stream as `AetherKrea.KreaNoise`, without depending on Krea). It does not reproduce PyTorch's
/// generator; exact PyTorch replay supplies original latent rows instead.
public enum H3Noise {
    public static func generate(seed: UInt64, count: Int) throws -> [Float] {
        try SeededGaussianNoise.generate(seed: seed, count: count)
    }
}

extension H3Geometry {
    /// Stereo audio latent rows: both channels' `audioLatentsPerChannel` rows of 32.
    public var audioRows: Int { audioLatentsPerChannel * 2 }

    /// Seeded initial latents: one stream, the contiguous source `[24,T,H,W]` video
    /// tensor first (then packed), followed by channel-major audio `[1, audioRows, 32]`.
    /// The keyframe noise level the released model holds its anchors at (`keyframe_noise_aug`).
    public static let keyframeNoiseLevel: Float = 0.999

    /// FL2VA with a first frame, a last frame or both: `initialLatents(seed:conditions:)` in packed order. With
    /// neither, the T2VA stream.
    public func initialLatents(seed: UInt64, firstFrameCondition first: H3TensorData?,
                               lastFrameCondition last: H3TensorData? = nil) throws -> H3Latents {
        try initialLatents(seed: seed, conditions: [(H3Anchor.first, first), (.last, last)].compactMap { anchor, condition in
            condition.map { (anchor, $0) }
        })
    }

    /// FL2VA: like `initialLatents(seed:)`, but the stream first draws each keyframe's condition noise `[24,1,H,W]` in
    /// packed order, first then last (the pipeline draws each condition before the generated rows). Each draw is mixed
    /// into its normalized condition latents at `t = 0.999` (`t * x + (1 - t) * noise`, float32), and the condition
    /// rows are packed in front of the generated video rows. With a first frame only, the stream is the one the
    /// first-frame path has always drawn; with no conditions, the T2VA stream.
    public func initialLatents(seed: UInt64, conditions: [(H3Anchor, H3TensorData)]) throws -> H3Latents {
        guard !conditions.isEmpty else { return try initialLatents(seed: seed) }
        guard H3Anchor.isPackedOrder(conditions.map(\.0)) else {
            throw AetherError.invalidRequest("H3 keyframe conditions must be ordered first then last")
        }
        let frame = 24 * latentHeight * latentWidth
        let spec = try H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, latentHeight, latentWidth])
        guard conditions.allSatisfy({ $0.1.spec == spec }) else {
            throw AetherError.invalidTensor("H3 keyframe conditions must be [1,24,1,H,W] float32")
        }
        let leading = frame * conditions.count, videoCount = targetVideoRows * 96, audioCount = audioRows * 32
        let noise = try H3Noise.generate(seed: seed, count: leading + videoCount + audioCount)
        let t = H3Geometry.keyframeNoiseLevel, one = Float(1) - t
        var conditionRows: [Float] = []
        for (index, (_, condition)) in conditions.enumerated() {
            let clean = try condition.floatValues(), base = index * frame
            conditionRows += try packFrames((0..<frame).map { t * clean[$0] + one * noise[base + $0] }, latentFrames: 1)
        }
        let target = try packVideo(FloatTensor(shape: videoLatentShape, values: Array(noise[leading..<(leading + videoCount)])))
        func tensor(_ shape: [Int], _ values: [Float]) throws -> H3TensorData {
            try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: shape), bytes: values.withUnsafeBytes { Data($0) })
        }
        return try H3Latents(video: tensor([1, conditionRows.count / 96 + targetVideoRows, 96], conditionRows + target.values),
                             audio: tensor([1, audioRows, 32], Array(noise[(leading + videoCount)...])),
                             anchors: conditions.map(\.0))
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
