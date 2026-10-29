import AetherKit
import AetherH3
import AetherAssets
import Darwin
import Foundation
import Synchronization
import Testing

/// Records stage acquisition and release so tests can prove at most one stage is resident.
private actor StageLog {
    private(set) var entries: [String] = []
    private var resident: Set<String> = []
    private(set) var maximumResident = 0
    func acquire(_ name: String) { if resident.insert(name).inserted { entries.append("acquire \(name)") }; maximumResident = max(maximumResident, resident.count) }
    func release(_ name: String) { if resident.remove(name) != nil { entries.append("release \(name)") } }
    var residentNow: Set<String> { resident }
    private(set) var notes: [String] = []
    func note(_ text: String) { notes.append(text) }
}

private enum FakeFailure: Error { case planned }

private struct FakeConditioner: H3PromptConditioner {
    let log: StageLog
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        await log.acquire("conditioner")
        for layer in 1...3 { onLayer(layer, 3) }
        return try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1, 4, 8]), bytes: Data(count: 64))
    }
    func release() async { await log.release("conditioner") }
}

private struct FakeDenoiser: H3Denoiser {
    let log: StageLog
    var steps: Int { 4 }
    var failAtStep: Int? = nil
    var waitAtStep: Int? = nil
    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry, onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        await log.acquire("denoiser")
        #expect(await log.residentNow == ["denoiser"])
        for step in 1...steps {
            if step == failAtStep { throw FakeFailure.planned }
            if step == waitAtStep { try await Task.sleep(for: .seconds(30)) }
            try Task.checkCancellation()
            onStep(step, steps)
        }
        return initial
    }
    func release() async { await log.release("denoiser") }
}

private struct FakeWindowDecoder: H3WindowDecoder {
    let log: StageLog
    func decode(windowRows: [Float], index: Int, geometry: H3Geometry) async throws -> [Float] {
        await log.acquire("video")
        #expect(windowRows.count == 7 * geometry.rowsPerLatentFrame * 96)
        // Each window is a constant, so stitched frames identify their source window.
        return [Float](repeating: Float(index) * 0.5, count: 66 * geometry.width * geometry.height)
    }
    func release() async { await log.release("video") }
}

private struct FakeAudioDecoder: H3AudioDecoder {
    let log: StageLog
    var samples: Int? = nil
    func decode(_ latents: H3TensorData, geometry: H3Geometry) async throws -> H3AudioPCM {
        await log.acquire("audio")
        let count = samples ?? Int((geometry.durationSeconds * Double(H3AudioPCM.sampleRate)).rounded())
        return try H3AudioPCM(channels: [[Float](repeating: 0.1, count: count), [Float](repeating: -0.1, count: count)])
    }
    func release() async { await log.release("audio") }
}

private actor EventLog {
    private(set) var events: [String] = []
    private(set) var frames = 0
    private(set) var firstPixels: [UInt8] = []
    func record(_ event: H3VideoEvent) {
        switch event {
        case .stageStarted(let stage): events.append("start \(stage.rawValue)")
        case .stageReleased(let stage): events.append("released \(stage.rawValue)")
        case .conditionerLayer(let layer, let total): events.append("layer \(layer)/\(total)")
        case .step(let step, let total): events.append("step \(step)/\(total)")
        case .frames(let batch):
            if batch.startFrame == 0 { firstPixels = batch.rgb[0] }
            #expect(batch.startFrame == frames); frames += batch.rgb.count; events.append("frames \(batch.rgb.count)")
        case .audio(let pcm): events.append("audio \(pcm.channels.count)")
        }
    }
}

private func geometry() throws -> H3Geometry { try H3Geometry(width: 64, height: 32, requestedFrames: 39, diagnostic: true) }

@Test func h3VideoSessionRunsStagesInSequenceAndReleasesEachBeforeTheNext() async throws {
    let log = StageLog(), events = EventLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let request = try H3VideoRequest(prompt: "integrated_multimodal_description: a quiet room", geometry: geometry(), seed: 3)
    let result = try await session.generate(request) { await events.record($0) }
    #expect(result.frames == 39 && result.steps == 4 && result.audio)
    #expect(await log.entries == ["acquire conditioner", "release conditioner", "acquire denoiser", "release denoiser",
                                  "acquire video", "release video", "acquire audio", "release audio"])
    #expect(await log.maximumResident == 1)
    #expect(await events.events == ["start encoding", "layer 1/3", "layer 2/3", "layer 3/3", "released encoding",
                                    "start denoising", "step 1/4", "step 2/4", "step 3/4", "step 4/4", "released denoising",
                                    "start decodingVideo", "frames 17", "frames 17", "frames 5", "released decodingVideo",
                                    "start decodingAudio", "released decodingAudio", "audio 2"])
    #expect(await events.frames == 39)
    // Window 0 is constant 0 in ImageNet space: the first pixel denormalizes to the channel means.
    #expect(await events.firstPixels.prefix(3) == [124, 116, 104])
    #expect(await session.state == .ready)
}

@Test func h3VideoSessionWithoutAudioNeverAcquiresTheAudioDecoder() async throws {
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let result = try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, audio: false))
    #expect(!result.audio)
    #expect(!(await log.entries.contains("acquire audio")))
}

@Test func h3VideoSessionReleasesEveryStageWhenAStageFails() async throws {
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log, failAtStep: 2),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    await #expect(throws: FakeFailure.self) { try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    #expect(await log.residentNow.isEmpty)
    #expect(!(await log.entries.contains("acquire video")))
    #expect(await session.state == .ready)
    // The session recovers for the next call.
    let ok = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                  video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    #expect(try await ok.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)).frames == 39)
}

@Test func h3VideoSessionRejectsAudioThatDoesNotCoverTheClip() async throws {
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log, samples: 1000)))
    await #expect(throws: AetherError.self) { try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    #expect(await log.residentNow.isEmpty)
}

@Test func h3VideoSessionCancellationStopsWorkReleasesStagesAndKeepsTheSlotUntilJoined() async throws {
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log, waitAtStep: 2),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let request = try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)
    let call = Task { try await session.generate(request) }
    while !(await log.entries.contains("acquire denoiser")) { await Task.yield() }
    await #expect(throws: AetherError.sessionBusy) { try await session.generate(request) }
    call.cancel()
    await #expect(throws: CancellationError.self) { try await call.value }
    #expect(await log.residentNow.isEmpty)
    #expect(!(await log.entries.contains("acquire video")))
    #expect(await session.state == .ready)
}

@Test func h3VideoSessionUnloadWaitsForTheActiveCallThenRefusesNewWork() async throws {
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let request = try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)
    let call = Task { try await session.generate(request) }
    while !(await log.entries.contains("acquire conditioner")) { await Task.yield() }
    await session.unload()
    #expect(try await call.value.frames == 39)
    #expect(await session.state == .closed)
    #expect(await log.residentNow.isEmpty)
    await #expect(throws: AetherError.sessionClosed) { try await session.generate(request) }
}

@Test func h3VideoRequestRejectsEmptyPromptsMismatchedKeyframesAndSessionsWithoutAnEncoder() async throws {
    let g = try geometry()
    #expect(throws: AetherError.self) { try H3VideoRequest(prompt: "", geometry: g, seed: 1) }
    let wrong = try H3Keyframe(width: 32, height: 32, rgb: [UInt8](repeating: 0, count: 32 * 32 * 3))
    #expect(throws: AetherError.self) { try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: wrong) }
    #expect(throws: AetherError.self) { try H3Keyframe(width: 2, height: 2, rgb: [0, 0, 0]) }
    let log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let first = try H3Keyframe(width: 64, height: 32, rgb: [UInt8](repeating: 0, count: 64 * 32 * 3))
    await #expect(throws: AetherError.self) {
        try await session.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: first))
    }
    #expect(await log.entries.isEmpty)
}

private struct FakeKeyframeEncoder: H3KeyframeEncoder {
    let log: StageLog
    var canvases: Set<String>? = nil
    func supports(_ geometry: H3Geometry) -> Bool { canvases.map { $0.contains("\(geometry.width)x\(geometry.height)") } ?? true }
    /// The condition is the keyframe's first byte everywhere, so tests can tell which keyframe landed where.
    func encode(_ keyframe: H3Keyframe, geometry: H3Geometry) async throws -> H3TensorData {
        await log.acquire("keyframes")
        await log.note("encode \(keyframe.rgb[0])")
        let count = 24 * geometry.latentHeight * geometry.latentWidth
        return try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, geometry.latentHeight, geometry.latentWidth]),
                                bytes: [Float](repeating: Float(keyframe.rgb[0]), count: count).withUnsafeBytes { Data($0) })
    }
    func release() async { await log.release("keyframes") }
}

/// Checks the denoiser sees the anchors first, in packed order and declared on the latents, and returns the latents
/// unchanged.
private struct AnchorCheckingDenoiser: H3Denoiser {
    let log: StageLog
    var anchors: [H3Anchor] = [.first]
    var steps: Int { 1 }
    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry, onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        await log.acquire("denoiser")
        let frameRows = (geometry.latentHeight / 2) * (geometry.latentWidth / 2)
        #expect(initial.anchors == anchors)
        #expect(initial.video.spec.shape == [1, anchors.count * frameRows + geometry.targetVideoRows, 96])
        onStep(1, 1)
        return initial
    }
    func release() async { await log.release("denoiser") }
}

@Test func h3VideoSessionEncodesTheFirstFrameBeforeDenoisingAndDecodesOnlyGeneratedRows() async throws {
    let log = StageLog(), events = EventLog(), g = try geometry()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), keyframes: FakeKeyframeEncoder(log: log),
                                                       denoiser: AnchorCheckingDenoiser(log: log), video: FakeWindowDecoder(log: log),
                                                       audio: FakeAudioDecoder(log: log)))
    let keyframe = try H3Keyframe(width: g.width, height: g.height, rgb: [UInt8](repeating: 9, count: g.width * g.height * 3))
    let result = try await session.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 5, firstFrame: keyframe)) { await events.record($0) }
    #expect(result.frames == 39)
    #expect(await log.entries.prefix(6) == ["acquire conditioner", "release conditioner", "acquire keyframes", "release keyframes",
                                            "acquire denoiser", "release denoiser"])
    let recorded = await events.events
    #expect(recorded.contains("start encodingKeyframe") && recorded.contains("released encodingKeyframe"))
    #expect(await log.maximumResident == 1)
}

@Test func h3VideoSessionRefusesKeyframesItCannotEncode() async throws {
    let log = StageLog(), g = try geometry()
    let keyframe = try H3Keyframe(width: g.width, height: g.height, rgb: [UInt8](repeating: 9, count: g.width * g.height * 3))
    let noEncoder = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                         video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    await #expect(throws: AetherError.self) { try await noEncoder.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: keyframe)) }
    await #expect(throws: AetherError.self) { try await noEncoder.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 1, lastFrame: keyframe)) }
    #expect(await log.entries.isEmpty)
}

@Test(arguments: [[H3Anchor.last], [.first, .last]])
func h3VideoSessionEncodesLastAndBothKeyframesInPackedOrderAndDecodesOnlyGeneratedRows(_ anchors: [H3Anchor]) async throws {
    let log = StageLog(), events = EventLog(), g = try geometry()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), keyframes: FakeKeyframeEncoder(log: log),
                                                       denoiser: AnchorCheckingDenoiser(log: log, anchors: anchors), video: FakeWindowDecoder(log: log),
                                                       audio: FakeAudioDecoder(log: log)))
    func keyframe(_ value: UInt8) throws -> H3Keyframe { try H3Keyframe(width: g.width, height: g.height, rgb: [UInt8](repeating: value, count: g.width * g.height * 3)) }
    let request = try H3VideoRequest(prompt: "p", geometry: g, seed: 5, firstFrame: anchors.contains(.first) ? keyframe(3) : nil, lastFrame: keyframe(7))
    let result = try await session.generate(request) { await events.record($0) }
    #expect(result.frames == 39)
    // One encoder stage for both keyframes, first then last, released before denoising.
    #expect(await log.entries.prefix(6) == ["acquire conditioner", "release conditioner", "acquire keyframes", "release keyframes",
                                            "acquire denoiser", "release denoiser"])
    #expect(await log.notes == (anchors.count == 2 ? ["encode 3", "encode 7"] : ["encode 7"]))
    let recorded = await events.events
    #expect(recorded.filter { $0 == "start encodingKeyframe" }.count == 1 && recorded.contains("released encodingKeyframe"))
    #expect(await log.maximumResident == 1)
    #expect(await log.residentNow.isEmpty)
}

@Test func h3FirstFrameLatentsDrawTheConditionNoiseFirstAndKeepTheT2VAStreamShape() throws {
    let g = try geometry()
    let frame = 24 * g.latentHeight * g.latentWidth, frameRows = (g.latentHeight / 2) * (g.latentWidth / 2)
    let clean = (0..<frame).map { Float($0 % 7) - 3 }
    let condition = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, g.latentHeight, g.latentWidth]),
                                     bytes: clean.withUnsafeBytes { Data($0) })
    let latents = try g.initialLatents(seed: 11, firstFrameCondition: condition)
    #expect(latents.video.spec.shape == [1, frameRows + g.targetVideoRows, 96])
    let noise = try H3Noise.generate(seed: 11, count: frame + g.targetVideoRows * 96 + g.audioRows * 32)
    let t: Float = 0.999, one = Float(1) - t
    let expectedCondition = try g.packFrames((0..<frame).map { t * clean[$0] + one * noise[$0] }, latentFrames: 1)
    let video = try latents.video.floatValues()
    #expect(Array(video[..<(frameRows * 96)]) == expectedCondition)
    #expect(Array(video[(frameRows * 96)...]) == (try g.packVideo(FloatTensor(shape: g.videoLatentShape,
                                                                               values: Array(noise[frame..<(frame + g.targetVideoRows * 96)])))).values)
    #expect(try latents.audio.floatValues() == Array(noise[(frame + g.targetVideoRows * 96)...]))
}

@Test func h3KeyframeLatentsDrawEachConditionsNoiseInPackedOrderBeforeTheGeneratedRows() throws {
    let g = try geometry()
    let frame = 24 * g.latentHeight * g.latentWidth, frameRows = (g.latentHeight / 2) * (g.latentWidth / 2)
    let videoCount = g.targetVideoRows * 96
    func condition(_ offset: Float) throws -> (clean: [Float], tensor: H3TensorData) {
        let clean = (0..<frame).map { Float($0 % 7) - 3 + offset }
        return (clean, try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, g.latentHeight, g.latentWidth]),
                                        bytes: clean.withUnsafeBytes { Data($0) }))
    }
    let first = try condition(0), last = try condition(10)
    let t: Float = 0.999, one = Float(1) - t
    // First and last: two condition draws (first, then last), then the generated video rows and the audio.
    let both = try g.initialLatents(seed: 11, first: first.tensor, last: last.tensor)
    #expect(both.anchors == [.first, .last])
    #expect(both.video.spec.shape == [1, 2 * frameRows + g.targetVideoRows, 96])
    var noise = try H3Noise.generate(seed: 11, count: 2 * frame + videoCount + g.audioRows * 32)
    var video = try both.video.floatValues()
    #expect(Array(video[..<(frameRows * 96)]) == (try g.packFrames((0..<frame).map { t * first.clean[$0] + one * noise[$0] }, latentFrames: 1)))
    #expect(Array(video[(frameRows * 96)..<(2 * frameRows * 96)])
            == (try g.packFrames((0..<frame).map { t * last.clean[$0] + one * noise[frame + $0] }, latentFrames: 1)))
    #expect(Array(video[(2 * frameRows * 96)...]) == (try g.packVideo(FloatTensor(shape: g.videoLatentShape,
                                                                                   values: Array(noise[(2 * frame)..<(2 * frame + videoCount)])))).values)
    #expect(try both.audio.floatValues() == Array(noise[(2 * frame + videoCount)...]))
    // Last only: one condition draw, the same stream shape as a first frame, declared as the last anchor.
    let lastOnly = try g.initialLatents(seed: 11, first: nil, last: last.tensor)
    #expect(lastOnly.anchors == [.last])
    noise = try H3Noise.generate(seed: 11, count: frame + videoCount + g.audioRows * 32)
    video = try lastOnly.video.floatValues()
    #expect(Array(video[..<(frameRows * 96)]) == (try g.packFrames((0..<frame).map { t * last.clean[$0] + one * noise[$0] }, latentFrames: 1)))
    #expect(try lastOnly.audio.floatValues() == Array(noise[(frame + videoCount)...]))
    // A first frame alone draws exactly the first-frame stream; no keyframe draws the T2VA stream.
    let firstOnly = try g.initialLatents(seed: 11, first: first.tensor, last: nil)
    let legacy = try g.initialLatents(seed: 11, firstFrameCondition: first.tensor)
    #expect(firstOnly.anchors == [.first] && legacy.anchors == [.first])
    #expect(firstOnly.video.bytes == legacy.video.bytes && firstOnly.audio.bytes == legacy.audio.bytes)
    let none = try g.initialLatents(seed: 11, first: nil, last: nil), t2va = try g.initialLatents(seed: 11)
    #expect(none.anchors.isEmpty && none.video.bytes == t2va.video.bytes && none.audio.bytes == t2va.audio.bytes)
    // A condition at another canvas is refused.
    let small = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, 1, 1]), bytes: Data(count: 96))
    #expect(throws: AetherError.self) { try g.initialLatents(seed: 11, first: nil, last: small) }
}

@Test func h3LatentsKeepTheirAnchorsThroughASchedulerStepAndRefuseMisorderedOnes() throws {
    let schedule = try H3Schedule.turbo(steps: 8, videoShift: 12, audioShift: 3)
    func tensor(_ rows: Int, _ width: Int, _ value: Float) throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, rows, width]),
                         bytes: [Float](repeating: value, count: rows * width).withUnsafeBytes { Data($0) })
    }
    let latents = try H3Latents(video: tensor(5, 96, 1), audio: tensor(2, 32, 1), anchors: [.last])
    let next = try latents.advanced(schedule: schedule, step: 0, videoVelocity: tensor(5, 96, 3), audioVelocity: tensor(2, 32, 3), conditionVideoRows: 1)
    #expect(next.anchors == [.last])
    for anchors: [H3Anchor] in [[.last, .first], [.first, .first], [.first, .last, .last]] {
        #expect(throws: AetherError.self) { try H3Latents(video: tensor(5, 96, 1), audio: tensor(2, 32, 1), anchors: anchors) }
    }
}

@Test func h3SchedulerStepNeverWritesTheKeyframeAnchors() throws {
    let schedule = try H3Schedule.turbo(steps: 8, videoShift: 12, audioShift: 3)
    func tensor(_ rows: Int, _ width: Int, _ value: Float) throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, rows, width]),
                         bytes: [Float](repeating: value, count: rows * width).withUnsafeBytes { Data($0) })
    }
    let latents = try H3Latents(video: tensor(5, 96, 1), audio: tensor(2, 32, 1))
    let next = try latents.advanced(schedule: schedule, step: 0, videoVelocity: tensor(5, 96, 3), audioVelocity: tensor(2, 32, 3), conditionVideoRows: 2)
    let video = try next.video.floatValues()
    #expect(video[..<(2 * 96)].allSatisfy { $0 == 1 })
    #expect(video[(2 * 96)...].allSatisfy { $0 != 1 })
    #expect(try next.audio.floatValues().allSatisfy { $0 != 1 })
}

@Test func h3FirstFrameAtACanvasWithoutAnEncoderIsRefusedBeforeAnyStage() async throws {
    let g = try geometry(), log = StageLog()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log),
                                                       keyframes: FakeKeyframeEncoder(log: log, canvases: ["960x544"]),
                                                       denoiser: FakeDenoiser(log: log), video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    let keyframe = try H3Keyframe(width: g.width, height: g.height, rgb: [UInt8](repeating: 9, count: g.width * g.height * 3))
    await #expect(throws: AetherError.self) { try await session.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: keyframe)) }
    #expect(await log.entries.isEmpty)
    #expect(await session.state == .ready)
}


private func stageEstimates(audio: Bool = true) -> [H3VideoStage: UInt64] {
    var stages: [H3VideoStage: UInt64] = [.encoding: 20, .denoising: 40, .decodingVideo: 30]
    if audio { stages[.decodingAudio] = 15 }
    return stages
}
private func admission(_ budget: ResourceCoordinator, audio: Bool = true) throws -> H3ResourceAdmission {
    try H3ResourceAdmission(coordinator: budget, executionIdentity: "fixture/1", operationBytes: 10,
                             stageBytes: stageEstimates(audio: audio), maximumPixels: 2048, maximumFrames: 39, maximumPromptUTF8Bytes: 4)
}
private func budgetStages(_ log: StageLog, conditioner: (any H3PromptConditioner)? = nil) -> H3VideoStages {
    H3VideoStages(conditioner: conditioner ?? FakeConditioner(log: log), keyframes: FakeKeyframeEncoder(log: log),
                  denoiser: FakeDenoiser(log: log), video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log),
                  resourceIdentity: "fixture/1")
}

@Test func h3AdmissionReservesEachStageBeforeEventsAndReleasesBeforeTheNext() async throws {
    let budget = try ResourceCoordinator(capacityBytes: 100), log = StageLog()
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget))
    let estimates = stageEstimates()
    for _ in 0..<2 {
        _ = try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) { event in
            switch event {
            case .stageStarted(let stage): #expect(budget.snapshot.reservedBytes == 10 + estimates[stage]!)
            case .stageReleased: #expect(budget.snapshot.reservedBytes == 10)
            default: break
            }
        }
        #expect(budget.snapshot.reservedBytes == 0)
    }
    await session.unload()
    #expect(await session.state == .closed)
}

@Test func h3AdmissionRefusesARunThatCannotFitBeforeAnyStage() async throws {
    // Operation 10 + largest stage 40 = 50 > 35: refused up front, nothing reserved or acquired.
    let budget = try ResourceCoordinator(capacityBytes: 35), log = StageLog()
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget))
    await #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: 50, available: 35)) {
        try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1))
    }
    #expect(await log.entries.isEmpty)
    #expect(budget.snapshot.reservedBytes == 0)
    #expect(await session.state == .ready)
    await session.unload()
}

@Test func h3AdmissionStillRefusesALaterStageWhenOthersTakeTheBudgetMidRun() async throws {
    // The run fits when it starts; another holder then takes 55 of 100, leaving 35 for the 40-byte denoiser.
    let budget = try ResourceCoordinator(capacityBytes: 100), log = StageLog()
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget))
    let other = Mutex<ResourceReservation?>(nil)
    await #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: 40, available: 35)) {
        try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) { event in
            if case .stageStarted(.encoding) = event { other.withLock { $0 = try? budget.reserve(bytes: 55) } }
        }
    }
    #expect(await log.entries == ["acquire conditioner", "release conditioner"])
    other.withLock { $0?.release(); $0 = nil }
    #expect(budget.snapshot.reservedBytes == 0)
    await session.unload()
}

@Test func h3AdmissionRejectsWrongIdentityMissingStagesAndOutOfProfileRequests() async throws {
    let budget = try ResourceCoordinator(capacityBytes: 100), log = StageLog()
    let anonymous = H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                    video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log))
    #expect(throws: AetherError.self) { try H3VideoSession(stages: anonymous, resources: admission(budget)) }
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget, audio: false))
    for request in [
        try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1), // Missing audio estimate.
        try H3VideoRequest(prompt: "long prompt", geometry: geometry(), seed: 1, audio: false),
        try H3VideoRequest(prompt: "p", geometry: H3Geometry(width: 96, height: 32, requestedFrames: 39, diagnostic: true), seed: 1, audio: false),
        try H3VideoRequest(prompt: "p", geometry: H3Geometry(width: 64, height: 32, requestedFrames: 40, diagnostic: true), seed: 1, audio: false),
    ] {
        await #expect(throws: AetherError.self) { try await session.generate(request) }
    }
    #expect(await log.entries.isEmpty)
    #expect(budget.snapshot.reservedBytes == 0)
    await session.unload()
}

private actor ResourceGate {
    private var entered = false
    private var completed = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    private var completion: CheckedContinuation<Void, Never>?
    func enter() async {
        if completed { return }
        entered = true; waiting.forEach { $0.resume() }; waiting = []
        await withCheckedContinuation { completion = $0 }
    }
    func waitUntilEntered() async { if entered { return }; await withCheckedContinuation { waiting.append($0) } }
    func complete() { completed = true; completion?.resume(); completion = nil }
}
private struct HeldConditioner: H3PromptConditioner {
    let log: StageLog
    let gate: ResourceGate
    let lease: ExternalFileLease
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        let value = try await FakeConditioner(log: log).encode(prompt, geometry: geometry, onLayer: onLayer)
        await gate.enter()
        withExtendedLifetime(lease) {}
        return value
    }
    func release() async { await log.release("conditioner"); withExtendedLifetime(lease) {} }
}

@Test(.timeLimit(.minutes(1))) func h3SharedAdmissionAndExternalLeaseSurviveCancellationUntilJoinedUnload() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("weight")
    try Data([0,0]).write(to: source)
    let fd = open(source.path, O_RDONLY); defer { close(fd) }
    #expect(fd >= 0)
    let budget = try ResourceCoordinator(capacityBytes: 50), log = StageLog(), gate = ResourceGate()
    let first = try H3VideoSession(stages: budgetStages(log, conditioner: HeldConditioner(log: log, gate: gate, lease: ExternalFileLease(files: [source]))),
                                     resources: admission(budget))
    let secondLog = StageLog()
    let second = try H3VideoSession(stages: budgetStages(secondLog), resources: admission(budget))
    let task = Task { try await first.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    await gate.waitUntilEntered()
    #expect(budget.snapshot.reservedBytes == 30)
    await #expect(throws: ResourceAdmissionError.self) {
        try await second.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1))
    }
    #expect(budget.snapshot.reservedBytes == 30)
    #expect(await secondLog.entries.isEmpty)
    task.cancel()
    let close = Task { await first.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await first.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await first.state == .closing)
    #expect(budget.snapshot.reservedBytes == 30)
    #expect(flock(fd, LOCK_EX | LOCK_NB) == -1)
    await gate.complete()
    await #expect(throws: CancellationError.self) { try await task.value }
    await close.value
    #expect(budget.snapshot.reservedBytes == 0)
    #expect(flock(fd, LOCK_EX | LOCK_NB) == 0)
    flock(fd, LOCK_UN)
    #expect(await log.entries == ["acquire conditioner", "release conditioner"])
    // A competing session is usable once the first session returns all capacity.
    _ = try await second.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1))
    #expect(budget.snapshot.reservedBytes == 0)
    await second.unload()
}


private struct HeldStageRelease: H3PromptConditioner {
    let log: StageLog
    let gate: ResourceGate
    func encode(_ prompt: String, geometry: H3Geometry, onLayer: @escaping @Sendable (Int, Int) -> Void) async throws -> H3TensorData {
        try await FakeConditioner(log: log).encode(prompt, geometry: geometry, onLayer: onLayer)
    }
    func release() async { await gate.enter(); await log.release("conditioner") }
}

@Test(.timeLimit(.minutes(1))) func h3StageAdmissionRemainsChargedUntilReleaseAndEveryUnloadCallerJoins() async throws {
    let log = StageLog(), gate = ResourceGate(), budget = try ResourceCoordinator(capacityBytes: 100)
    let session = try H3VideoSession(stages: budgetStages(log, conditioner: HeldStageRelease(log: log, gate: gate)), resources: admission(budget))
    let task = Task { try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    await gate.waitUntilEntered()
    #expect(budget.snapshot.reservedBytes == 30)
    task.cancel()
    let firstClose = Task { await session.unload() }, secondClose = Task { await session.unload() }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while await session.state != .closing, ContinuousClock.now < deadline { await Task.yield() }
    #expect(await session.state == .closing)
    #expect(budget.snapshot.reservedBytes == 30)
    await gate.complete()
    await #expect(throws: CancellationError.self) { try await task.value }
    await firstClose.value; await secondClose.value
    #expect(await session.state == .closed)
    #expect(budget.snapshot.reservedBytes == 0)
}

@Test func h3StageEventFailureReturnsReservationsWithoutAcquiringTheStage() async throws {
    let log = StageLog(), budget = try ResourceCoordinator(capacityBytes: 100)
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget))
    await #expect(throws: FakeFailure.self) {
        try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) { event in
            if case .stageStarted = event { throw FakeFailure.planned }
        }
    }
    #expect(await log.entries.isEmpty)
    #expect(budget.snapshot.reservedBytes == 0)
    await session.unload()
}

@Test func h3VideoSessionWaitsForSharedGPUAdmissionBeforeItsFirstStage() async throws {
    let log = StageLog()
    let admission = try GPUAdmission()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)),
                                 gpu: admission.owner("h3"))
    let other = try await admission.admit(owner: "krea")
    let run = Task { try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, audio: false)) }
    while admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    #expect(await log.entries.isEmpty)
    other.release()
    #expect(try await run.value.frames == 39)
    #expect(admission.snapshot.running == 0)
    let failFast = try GPUAdmission(waiting: .failFast)
    let quick = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                     video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)),
                               gpu: failFast.owner("h3"))
    let busy = try await failFast.admit(owner: "krea")
    await #expect(throws: GPUAdmissionError.busy) {
        try await quick.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, audio: false))
    }
    #expect(await quick.state == .ready)
    busy.release()
}

@Test func h3GenerationWaitingForTheGPUHoldsNoOperationReservation() async throws {
    let budget = try ResourceCoordinator(capacityBytes: 100), log = StageLog()
    let shared = try GPUAdmission()
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission(budget), gpu: shared.owner("h3"))
    let other = try await shared.admit(owner: "krea")
    let run = Task { try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    while shared.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    #expect(budget.snapshot.reservedBytes == 0)
    other.release()
    _ = try await run.value
    #expect(budget.snapshot.reservedBytes == 0)
    await session.unload()
}

@Test func h3UnloadWithdrawsAGenerationStillWaitingForTheGPU() async throws {
    let log = StageLog()
    let admission = try GPUAdmission()
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)),
                                 gpu: admission.owner("h3"))
    let other = try await admission.admit(owner: "krea")
    let run = Task { try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) }
    while admission.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    await session.unload()
    await #expect(throws: AetherError.sessionClosed) { try await run.value }
    #expect(await session.state == .closed)
    #expect(await log.entries.filter { $0.hasPrefix("acquire") }.isEmpty)
    #expect(admission.snapshot.queued == 0)
    other.release()
}

// MARK: Measured resources and the 18 GiB bar

private let gib: UInt64 = 1 << 30
private func measured(denoising: UInt64 = 13_800 * (1 << 20), operation: UInt64 = 1 << 30) throws -> H3MeasuredResources {
    try H3MeasuredResources(executionIdentity: "fixture/1", deviceArchitecture: "fixture", canvases: ["960x544", "544x960"], maximumFrames: 124,
                            stagePeakBytes: ["encoding": 1_830 * (1 << 20), "encodingKeyframe": 2_050 * (1 << 20), "denoising": denoising,
                                             "decodingVideo": 5_780 * (1 << 20), "decodingAudio": 1_030 * (1 << 20)],
                            operationBytes: operation, evidence: "fixture")
}

@Test func h3MeasuredAdmissionReservesEveryMeasuredStagePeakAndRefusesARunThatWillNotFit() async throws {
    let entry = try measured()
    // An 18 GiB host budget admits the run; every stage reservation covers its measured peak plus the margin.
    let roomy = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes), log = StageLog()
    let session = try H3VideoSession(stages: budgetStages(log), resources: entry.admission(coordinator: roomy))
    _ = try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) { event in
        if case .stageStarted(let stage) = event {
            #expect(roomy.snapshot.reservedBytes == entry.operationBytes + entry.stagePeakBytes[stage.rawValue]! + H3MeasuredResources.marginBytes)
        }
    }
    #expect(roomy.snapshot.reservedBytes == 0)
    await session.unload()
    // A 12 GiB budget cannot hold the measured denoising peak: refused up front, before any stage is acquired.
    let tight = try ResourceCoordinator(capacityBytes: 12 * gib), tightLog = StageLog()
    let refused = try H3VideoSession(stages: budgetStages(tightLog), resources: entry.admission(coordinator: tight))
    await #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: entry.operationBytes + entry.stagePeakBytes["denoising"]! + H3MeasuredResources.marginBytes,
                                                                    available: 12 * gib)) {
        try await refused.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1))
    }
    #expect(await tightLog.entries.isEmpty)
    #expect(tight.snapshot.reservedBytes == 0)
    await refused.unload()
    #expect(entry.maximumFrames == 124 && entry.maximumPixels == 960 * 544)
}

@Test func h3MeasuredAdmissionRefusesRequestsNeedingStagesTheEntryLacksBeforeAnyStage() async throws {
    // A partial entry: no first-frame encoder and no audio decoder measured.
    let partial = try H3MeasuredResources(executionIdentity: "fixture/1", deviceArchitecture: "fixture", canvases: ["960x544"], maximumFrames: 124,
                                          stagePeakBytes: ["encoding": 2 * gib, "denoising": 14 * gib, "decodingVideo": 6 * gib],
                                          operationBytes: gib / 4, evidence: "fixture")
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes), log = StageLog()
    let session = try H3VideoSession(stages: budgetStages(log), resources: partial.admission(coordinator: budget))
    let first = try H3Keyframe(width: 64, height: 32, rgb: [UInt8](repeating: 0, count: 64 * 32 * 3))
    await #expect(throws: AetherError.self) {
        try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, firstFrame: first, audio: false))
    }
    await #expect(throws: AetherError.self) {
        try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, lastFrame: first, audio: false))
    }
    await #expect(throws: AetherError.self) { try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, audio: true)) }
    #expect(await log.entries.isEmpty)
    #expect(budget.snapshot.reservedBytes == 0)
    // What the entry does cover still runs.
    _ = try await session.generate(try H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1, audio: false))
    #expect(budget.snapshot.reservedBytes == 0)
    await session.unload()
}

@Test func h3MeasuredAdmissionReservesOnlyAfterTheSharedGPUGrant() async throws {
    let entry = try measured()
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes), log = StageLog()
    let shared = try GPUAdmission()
    let session = try H3VideoSession(stages: budgetStages(log), resources: entry.admission(coordinator: budget), gpu: shared.owner("h3"))
    let other = try await shared.admit(owner: "krea")
    let run = Task {
        try await session.generate(H3VideoRequest(prompt: "p", geometry: geometry(), seed: 1)) { event in
            // Every measured stage reservation is taken while this generation holds the grant.
            if case .stageStarted(let stage) = event {
                #expect(shared.snapshot.running == 1)
                #expect(budget.snapshot.reservedBytes == entry.operationBytes + entry.stagePeakBytes[stage.rawValue]! + H3MeasuredResources.marginBytes)
            }
        }
    }
    while shared.snapshot.queued == 0 { try await Task.sleep(for: .milliseconds(5)) }
    // Waiting for the GPU holds neither the operation nor any stage reservation.
    #expect(budget.snapshot.reservedBytes == 0)
    #expect(await log.entries.isEmpty)
    other.release()
    _ = try await run.value
    #expect(budget.snapshot.reservedBytes == 0 && shared.snapshot.running == 0)
    await session.unload()
}

@Test func h3MeasuredResourcesOverTheEighteenGiBBarAreRefused() throws {
    #expect(throws: AetherError.self) { _ = try measured(denoising: 17 * gib) }               // 17 GiB + margin + 1 GiB > 18 GiB
    #expect(throws: AetherError.self) { _ = try measured(denoising: 13 * gib, operation: 5 * gib) }
    _ = try measured(denoising: 16 * gib, operation: gib / 2)                                    // 16.75 GiB fits
    #expect(throws: AetherError.self) {
        _ = try H3MeasuredResources(executionIdentity: "fixture/1", deviceArchitecture: "fixture", canvases: ["960x544"], maximumFrames: 124,
                                    stagePeakBytes: ["denoising": gib], operationBytes: gib, evidence: "fixture")   // missing required stages
    }
    // Every recorded entry decodes and is within the bar; an unmeasured identity has no estimate.
    for entry in try H3MeasuredResources.bundled() { try entry.validate() }
    // The recorded entry for the qualified tiled-encoder profile in its Kit-owned home (memory gate, both orientations, T2VA and FL2VA).
    let recorded = try H3MeasuredResources.bundled(
        executionIdentity: "h3.profile/1:5af39db57a6ffc513d9c71f43d6875a44bd44b4ca53e3d29cf9e76cc205992ce:turbo8", deviceArchitecture: "h17c")
    #expect(Set(recorded.stagePeakBytes.keys) == ["encoding", "encodingKeyframe", "denoising", "decodingVideo", "decodingAudio"])
    #expect(recorded.stagePeakBytes.values.max()! + H3MeasuredResources.marginBytes + recorded.operationBytes <= H3MeasuredResources.memoryBarBytes)
    #expect(recorded.maximumFrames == 124 && recorded.canvases == ["960x544", "544x960"])
    #expect(throws: AetherError.self) { _ = try H3MeasuredResources.bundled(executionIdentity: "unmeasured", deviceArchitecture: "fixture") }
}
