import AetherKit
import Foundation

/// An owned RGB8 keyframe for first/last-frame conditioning, already at the clip size.
public struct H3Keyframe: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let rgb: [UInt8]
    public init(width: Int, height: Int, rgb: [UInt8]) throws {
        guard width > 0, height > 0, rgb.count == width * height * 3 else {
            throw AetherError.invalidRequest("H3 keyframe must be interleaved RGB8 at its declared size")
        }
        self.width = width; self.height = height; self.rgb = rgb
    }
}

/// One audio/video generation. `prompt` is the already-structured H3 prompt text; the
/// caller owns its fields. With `audio` false, no audio decoder is acquired.
public struct H3VideoRequest: Sendable {
    public let prompt: String
    public let geometry: H3Geometry
    public let seed: UInt64
    public let firstFrame: H3Keyframe?
    public let lastFrame: H3Keyframe?
    public let audio: Bool
    public init(prompt: String, geometry: H3Geometry, seed: UInt64, firstFrame: H3Keyframe? = nil,
                lastFrame: H3Keyframe? = nil, audio: Bool = true) throws {
        guard !prompt.isEmpty else { throw AetherError.invalidRequest("H3 prompt must not be empty") }
        for frame in [firstFrame, lastFrame].compactMap({ $0 }) where frame.width != geometry.width || frame.height != geometry.height {
            throw AetherError.invalidRequest("H3 keyframes must match the clip size")
        }
        self.prompt = prompt; self.geometry = geometry; self.seed = seed
        self.firstFrame = firstFrame; self.lastFrame = lastFrame; self.audio = audio
    }
}

/// Planar stereo float PCM at 32 kHz, as the source audio VAE returns it.
public struct H3AudioPCM: Sendable, Equatable {
    public static let sampleRate = 32_000
    public let channels: [[Float]]
    public var sampleCount: Int { channels.first?.count ?? 0 }
    public var durationSeconds: Double { Double(sampleCount) / Double(Self.sampleRate) }
    public init(channels: [[Float]]) throws {
        guard channels.count == 2, channels[0].count > 0, channels[0].count == channels[1].count,
              channels.allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw AetherError.invalidTensor("H3 audio must be finite planar stereo with equal channel lengths")
        }
        self.channels = channels
    }
}

/// Consecutive RGB8 frames, delivered in order as each decoder window is stitched.
public struct H3DecodedFrames: Sendable {
    public let startFrame: Int
    public let totalFrames: Int
    public let width: Int
    public let height: Int
    public let rgb: [[UInt8]]
}

public enum H3VideoStage: String, Sendable, Codable { case encoding, encodingKeyframe, denoising, decodingVideo, decodingAudio }

public enum H3VideoEvent: Sendable {
    case stageStarted(H3VideoStage)
    case stageReleased(H3VideoStage)
    case conditionerLayer(Int, of: Int)
    case step(Int, of: Int)
    case frames(H3DecodedFrames)
    case audio(H3AudioPCM)
}
public typealias H3VideoEventHandler = @Sendable (H3VideoEvent) async throws -> Void

public struct H3VideoResult: Sendable {
    public let frames: Int
    public let seed: UInt64
    public let steps: Int
    public let audio: Bool
}

// MARK: Stages. Each acquires its native resources on first use and returns them on
// `release()`, which must be idempotent. The session never has two stages acquired.

public protocol H3PromptConditioner: Sendable {
    /// Returns the conditioner tap for `prompt` at the geometry's text layout.
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData
    func release() async
}
public protocol H3Denoiser: Sendable {
    var steps: Int { get }
    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry,
                 onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents
    func release() async
}
public protocol H3WindowDecoder: Sendable {
    /// Packed `[7*H/32*W/32, 96]` rows of one temporal window → CHW float32 `[3,22,H,W]`.
    func decode(windowRows: [Float], index: Int, geometry: H3Geometry) async throws -> [Float]
    func release() async
}
public protocol H3AudioDecoder: Sendable {
    func decode(_ latents: H3TensorData, geometry: H3Geometry) async throws -> H3AudioPCM
    func release() async
}

/// Encodes a first-frame keyframe (RGB8 at the clip size) into normalized condition latents `[1,24,1,H/16,W/16]` f32.
public protocol H3KeyframeEncoder: Sendable {
    func encode(_ keyframe: H3Keyframe, geometry: H3Geometry) async throws -> H3TensorData
    func release() async
}

public struct H3VideoStages: Sendable {
    public let conditioner: any H3PromptConditioner
    /// FL2VA only; a session without it refuses keyframes.
    public let keyframes: (any H3KeyframeEncoder)?
    public let denoiser: any H3Denoiser
    public let video: any H3WindowDecoder
    public let audio: any H3AudioDecoder
    public init(conditioner: any H3PromptConditioner, keyframes: (any H3KeyframeEncoder)? = nil, denoiser: any H3Denoiser,
                video: any H3WindowDecoder, audio: any H3AudioDecoder) {
        self.conditioner = conditioner; self.keyframes = keyframes; self.denoiser = denoiser; self.video = video; self.audio = audio
    }
}

/// Experimental end-to-end H3 session over injected stages: prompt encoding, seeded
/// latents, denoising, windowed video decoding and optional audio decoding, strictly in
/// sequence so at most one heavyweight stage is resident. One call at a time; cancellation
/// is observed between stages, steps (by the denoiser) and decoder windows, and every
/// acquired stage is released before the call returns or throws.
public actor H3VideoSession {
    public enum State: String, Sendable { case ready, running, closing, closed }
    public private(set) var state: State = .ready
    private var stages: H3VideoStages?
    private var active: Task<H3VideoResult, any Error>?

    public init(stages: H3VideoStages) { self.stages = stages }

    public func generate(_ request: H3VideoRequest, onEvent: @escaping H3VideoEventHandler = { _ in }) async throws -> H3VideoResult {
        try Task.checkCancellation()
        guard state != .closing, state != .closed, let stages else { throw AetherError.sessionClosed }
        guard active == nil else { throw AetherError.sessionBusy }
        guard request.lastFrame == nil else { throw AetherError.unsupported("H3 last-frame conditioning is not implemented") }
        guard request.firstFrame == nil || stages.keyframes != nil else { throw AetherError.unsupported("H3 session has no keyframe encoder") }
        let task = Task { try await Self.run(request, stages: stages, onEvent: onEvent) }
        active = task; state = .running
        defer { active = nil; if state == .running { state = .ready } }
        return try await withTaskCancellationHandler {
            let result = try await task.value; try Task.checkCancellation(); return result
        } onCancel: { task.cancel() }
    }

    /// Waits for any active call, then releases every stage.
    public func unload() async {
        guard state != .closed else { return }
        state = .closing
        if let active { _ = await active.result }
        if let stages { await Self.releaseAll(stages) }
        stages = nil; state = .closed
    }

    private static func run(_ request: H3VideoRequest, stages: H3VideoStages, onEvent: @escaping H3VideoEventHandler) async throws -> H3VideoResult {
        let geometry = request.geometry
        do {
            try Task.checkCancellation()
            try await onEvent(.stageStarted(.encoding))
            let conditioning = try await forwarding({ .conditionerLayer($0, of: $1) }, to: onEvent) { report in
                try await stages.conditioner.encode(request.prompt, geometry: geometry, onLayer: report)
            }
            await stages.conditioner.release(); try await onEvent(.stageReleased(.encoding))

            try Task.checkCancellation()
            var initial = try geometry.initialLatents(seed: request.seed)
            var conditionRows = 0
            if let keyframe = request.firstFrame, let encoder = stages.keyframes {
                try await onEvent(.stageStarted(.encodingKeyframe))
                let condition = try await encoder.encode(keyframe, geometry: geometry)
                await encoder.release(); try await onEvent(.stageReleased(.encodingKeyframe))
                initial = try geometry.initialLatents(seed: request.seed, firstFrameCondition: condition)
                conditionRows = initial.video.spec.shape[1] - geometry.targetVideoRows
            }
            try await onEvent(.stageStarted(.denoising))
            let denoised = try await forwarding({ .step($0, of: $1) }, to: onEvent) { report in
                try await stages.denoiser.denoise(initial, conditioning: conditioning, geometry: geometry, onStep: report)
            }
            await stages.denoiser.release(); try await onEvent(.stageReleased(.denoising))
            guard denoised.video.spec == initial.video.spec, denoised.audio.spec == initial.audio.spec else {
                throw AetherError.invalidTensor("H3 denoiser output geometry differs from its initial latents")
            }

            try Task.checkCancellation()
            try await onEvent(.stageStarted(.decodingVideo))
            // Only the generated rows are decoded; the keyframe anchors lead the video rows (the pipeline's `latents[ncv:]`).
            let packed = Array(try denoised.video.floatValues().dropFirst(conditionRows * 96))
            var stitcher = try H3VideoStitcher(geometry: geometry)
            var emitted = 0
            func deliver(_ normalized: [Float]) async throws {
                let rgb = try H3Pixels.rgb8(frames: normalized, width: geometry.width, height: geometry.height)
                try await onEvent(.frames(H3DecodedFrames(startFrame: emitted, totalFrames: geometry.frames,
                                                          width: geometry.width, height: geometry.height, rgb: rgb)))
                emitted += rgb.count
            }
            for index in 0..<geometry.videoWindows {
                try Task.checkCancellation()
                let rows = try geometry.videoWindowRows(packed, index: index)
                try await deliver(stitcher.push(try await stages.video.decode(windowRows: rows, index: index, geometry: geometry)))
            }
            try await deliver(stitcher.finish())
            guard emitted == geometry.frames else { throw AetherError.invalidTensor("H3 decoded frame count differs") }
            await stages.video.release(); try await onEvent(.stageReleased(.decodingVideo))

            if request.audio {
                try Task.checkCancellation()
                try await onEvent(.stageStarted(.decodingAudio))
                let pcm = try await stages.audio.decode(denoised.audio, geometry: geometry)
                await stages.audio.release(); try await onEvent(.stageReleased(.decodingAudio))
                // One audio latent is 1/40 s; the decoded track must cover the clip to within it.
                guard abs(pcm.durationSeconds - geometry.durationSeconds) <= 1.0 / 40 else {
                    throw AetherError.invalidTensor("H3 audio duration differs from the video")
                }
                try await onEvent(.audio(pcm))
            }
            return H3VideoResult(frames: emitted, seed: request.seed, steps: stages.denoiser.steps, audio: request.audio)
        } catch {
            await releaseAll(stages)
            throw error
        }
    }

    /// Delivers a stage's synchronous progress callbacks to the handler in order and joins
    /// them before returning. Progress handler errors are not propagated.
    private static func forwarding<T: Sendable>(_ event: @escaping @Sendable (Int, Int) -> H3VideoEvent,
                                                to onEvent: @escaping H3VideoEventHandler,
                                                _ body: (@escaping @Sendable (Int, Int) -> Void) async throws -> T) async throws -> T {
        let (stream, continuation) = AsyncStream.makeStream(of: H3VideoEvent.self)
        let forward = Task { for await item in stream { try? await onEvent(item) } }
        do {
            let value = try await body { continuation.yield(event($0, $1)) }
            continuation.finish(); await forward.value
            return value
        } catch {
            continuation.finish(); await forward.value
            throw error
        }
    }

    private static func releaseAll(_ stages: H3VideoStages) async {
        await stages.conditioner.release(); await stages.keyframes?.release(); await stages.denoiser.release()
        await stages.video.release(); await stages.audio.release()
    }
}
