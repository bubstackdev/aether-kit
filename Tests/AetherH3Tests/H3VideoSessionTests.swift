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

@Test func h3VideoRequestRejectsEmptyPromptsMismatchedKeyframesAndUnimplementedConditioning() async throws {
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
    func encode(_ keyframe: H3Keyframe, geometry: H3Geometry) async throws -> H3TensorData {
        await log.acquire("keyframes")
        let count = 24 * geometry.latentHeight * geometry.latentWidth
        return try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [1, 24, 1, geometry.latentHeight, geometry.latentWidth]),
                                bytes: [Float](repeating: 2, count: count).withUnsafeBytes { Data($0) })
    }
    func release() async { await log.release("keyframes") }
}

/// Checks the denoiser sees the anchors first and returns the latents unchanged.
private struct AnchorCheckingDenoiser: H3Denoiser {
    let log: StageLog
    var steps: Int { 1 }
    func denoise(_ initial: H3Latents, conditioning: H3TensorData, geometry: H3Geometry, onStep: @escaping @Sendable (Int, Int) -> Void) async throws -> H3Latents {
        await log.acquire("denoiser")
        let frameRows = (geometry.latentHeight / 2) * (geometry.latentWidth / 2)
        #expect(initial.video.spec.shape == [1, frameRows + geometry.targetVideoRows, 96])
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
    let withEncoder = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), keyframes: FakeKeyframeEncoder(log: log),
                                                           denoiser: FakeDenoiser(log: log), video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)))
    await #expect(throws: AetherError.self) { try await withEncoder.generate(try H3VideoRequest(prompt: "p", geometry: g, seed: 1, lastFrame: keyframe)) }
    #expect(await log.entries.isEmpty)
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

private func entry(_ canvases: [String], frames: Int, denoising: UInt64, operation: UInt64 = gib / 4, tasks: Set<H3Profile.Task>? = nil,
                   audio: Bool = true, keyframe: Bool = true, bar: UInt64 = H3MeasuredResources.memoryBarBytes,
                   identity: String = "fixture/1") throws -> H3MeasuredResources {
    var peaks: [String: UInt64] = ["encoding": 2 * gib, "denoising": denoising, "decodingVideo": 6 * gib]
    if audio { peaks["decodingAudio"] = gib }
    if keyframe { peaks["encodingKeyframe"] = 2 * gib }
    return try H3MeasuredResources(executionIdentity: identity, deviceArchitecture: "fixture", canvases: canvases, maximumFrames: frames,
                                   tasks: tasks, stagePeakBytes: peaks, operationBytes: operation, memoryBarBytes: bar, evidence: "fixture")
}
private func request(_ width: Int, _ height: Int, _ frames: Int, first: Bool = false, audio: Bool = false) throws -> H3VideoRequest {
    let g = try H3Geometry(width: width, height: height, requestedFrames: frames, diagnostic: true)
    let keyframe = first ? try H3Keyframe(width: width, height: height, rgb: [UInt8](repeating: 0, count: width * height * 3)) : nil
    return try H3VideoRequest(prompt: "p", geometry: g, seed: 1, firstFrame: keyframe, audio: audio)
}

@Test func h3MeasuredEnvelopesHoldEachRequestToOneCoveringEntryAndRefuseUnmeasuredCombinations() async throws {
    // A small canvas measured at a longer clip, and a larger canvas measured only at a short one (both landscape).
    let long = try entry(["64x32"], frames: 56, denoising: 12 * gib), large = try entry(["96x64"], frames: 39, denoising: 15 * gib)
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes), log = StageLog()
    let admission = try H3MeasuredResources.admission(entries: [long, large], coordinator: budget)
    #expect(admission.envelopes == [try long.envelope(), try large.envelope()])
    #expect(admission.maximumPixels == 96 * 64 && admission.maximumFrames == 56)     // a union, not an admitted combination
    let session = try H3VideoSession(stages: budgetStages(log), resources: admission)
    func run(_ r: H3VideoRequest, expecting chosen: H3MeasuredResources) async throws {
        #expect(try admission.envelope(for: r) == chosen.envelope())
        _ = try await session.generate(r) { event in
            if case .stageStarted(let stage) = event {
                #expect(budget.snapshot.reservedBytes == chosen.operationBytes + chosen.stagePeakBytes[stage.rawValue]! + H3MeasuredResources.marginBytes)
            }
        }
    }
    try await run(request(64, 32, 56), expecting: long)     // at both of long's limits
    try await run(request(64, 32, 39), expecting: long)     // both cover it; the cheaper envelope is used
    try await run(request(96, 64, 39), expecting: large)    // at both of large's limits
    try await run(request(64, 64, 39), expecting: large)    // a square canvas fits a landscape one
    // Just outside: the next valid width, height and length, the portrait of a landscape-only canvas, and the large
    // canvas at the long clip that no single entry measured.
    for outside in [try request(128, 64, 39), try request(96, 96, 39), try request(96, 64, 56), try request(64, 32, 73),
                    try request(64, 96, 39), try request(32, 64, 39)] {
        #expect(!admission.admits(outside))
        #expect(throws: AetherError.self) { _ = try admission.envelope(for: outside) }
    }
    do { _ = try admission.envelope(for: request(96, 64, 56)); Issue.record("expected a refusal") } catch {
        #expect("\(error)".contains("nearest"))
    }
    let before = await log.entries.count
    await #expect(throws: AetherError.self) { try await session.generate(request(96, 64, 56)) }
    #expect(await log.entries.count == before && budget.snapshot.reservedBytes == 0)
    await session.unload()
    // Envelopes come from entries of one identity and device.
    let other = try entry(["64x32"], frames: 39, denoising: 12 * gib, identity: "fixture/2")
    #expect(throws: AetherError.self) { _ = try H3MeasuredResources.admission(entries: [long, other], coordinator: budget) }
    #expect(throws: AetherError.self) { _ = try H3MeasuredResources.admission(entries: [], coordinator: budget) }
}

@Test func h3MeasuredEnvelopeChoiceFollowsTheTaskAndTheStagesTheRequestNeeds() throws {
    // T2VA measured on its own (cheaper), and both tasks measured without the audio decoder.
    let t2va = try entry(["64x32"], frames: 39, denoising: 12 * gib, tasks: [.t2va], keyframe: false)
    let both = try entry(["64x32"], frames: 39, denoising: 14 * gib, audio: false)
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes)
    let admission = try H3MeasuredResources.admission(entries: [t2va, both], coordinator: budget)
    #expect(try admission.envelope(for: request(64, 32, 39)) == t2va.envelope())
    #expect(try admission.envelope(for: request(64, 32, 39, audio: true)) == t2va.envelope())
    #expect(try admission.envelope(for: request(64, 32, 39, first: true)) == both.envelope())
    #expect(throws: AetherError.self) { _ = try admission.envelope(for: request(64, 32, 39, first: true, audio: true)) }
    // An FL2VA entry can be replaced without touching the T2VA one: an admission without it still admits T2VA.
    let t2vaOnly = try H3MeasuredResources.admission(entries: [t2va], coordinator: budget)
    let withAudio = try request(64, 32, 39, audio: true), withFirst = try request(64, 32, 39, first: true)
    #expect(t2vaOnly.admits(withAudio) && !t2vaOnly.admits(withFirst))
    #expect(throws: AetherError.self) { _ = try entry(["64x32"], frames: 39, denoising: gib, tasks: []) }
}

@Test func h3MeasuredBarsAreHeldPerEntryAndTheExtendedBarNeedsSixtyFourGiB() throws {
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.extendedMemoryBarBytes)
    // Today's region keeps 18 GiB; the 24 GiB bar is valid only beyond 960x544's pixels or 124 frames.
    let today = try entry(["960x544", "544x960"], frames: 124, denoising: 11 * gib)
    let canvas = try entry(["1280x704", "704x1280"], frames: 141, denoising: 20 * gib, operation: gib / 8, tasks: [.t2va], keyframe: false,
                           bar: H3MeasuredResources.extendedMemoryBarBytes)
    let clip = try entry(["960x544"], frames: 243, denoising: 20 * gib, tasks: [.t2va], keyframe: false, bar: H3MeasuredResources.extendedMemoryBarBytes)
    #expect(today.minimumPhysicalMemoryBytes == 0 && canvas.minimumPhysicalMemoryBytes == 64 << 30 && clip.isBeyondBaseline)
    #expect(throws: AetherError.self) { _ = try entry(["1280x704"], frames: 141, denoising: 20 * gib) }                 // over 18 GiB
    #expect(throws: AetherError.self) {                                                                                   // invalid 24 GiB claim
        _ = try entry(["960x544"], frames: 124, denoising: 20 * gib, bar: H3MeasuredResources.extendedMemoryBarBytes)
    }
    #expect(throws: AetherError.self) {                                                                                   // over 24 GiB
        _ = try entry(["1280x704"], frames: 141, denoising: 24_100 * (1 << 20), bar: H3MeasuredResources.extendedMemoryBarBytes)   // + 512 MiB > 24 GiB
    }
    #expect(throws: AetherError.self) { _ = try entry(["1280x704"], frames: 141, denoising: gib, bar: 20 << 30) }       // unknown bar
    _ = try entry(["1280x704"], frames: 141, denoising: 23_800 * (1 << 20), bar: H3MeasuredResources.extendedMemoryBarBytes)    // 24,312 MiB fits
    #expect(throws: AetherError.self) { _ = try entry(["1344x800"], frames: 124, denoising: gib) }                     // over the pixel cap
    let large = try request(1280, 704, 141), long = try request(960, 544, 243)
    // Below 64 GiB: today's region still runs; the extended entries are refused, naming the requirement.
    let small = try H3MeasuredResources.admission(entries: [today, canvas, clip], coordinator: budget, physicalMemoryBytes: 32 << 30)
    let landscape = try request(960, 544, 124), portrait = try request(544, 960, 124, first: true)
    #expect(small.admits(landscape) && small.admits(portrait))
    for r in [large, long] {
        do { _ = try small.envelope(for: r); Issue.record("expected a refusal") } catch { #expect("\(error)".contains("64 GiB")) }
    }
    #expect(throws: AetherError.self) {
        _ = try H3MeasuredResources.admission(entries: [canvas], coordinator: budget, physicalMemoryBytes: 32 << 30)
    }
    #expect(try H3MeasuredResources.entries([today, canvas, clip], executionIdentity: "fixture/1", deviceArchitecture: "fixture",
                                            physicalMemoryBytes: 32 << 30) == [today])
    #expect(throws: AetherError.self) {
        _ = try H3MeasuredResources.entries([canvas], executionIdentity: "fixture/1", deviceArchitecture: "fixture", physicalMemoryBytes: 32 << 30)
    }
    // At 64 GiB: the large canvas and the long clip are admitted, each under its own entry; 960x544 at 125...141
    // frames falls under the large canvas; the 18 GiB class is preferred where both cover, even when dearer.
    let big = try H3MeasuredResources.admission(entries: [today, canvas, clip], coordinator: budget, physicalMemoryBytes: 64 << 30)
    #expect(try big.envelope(for: large) == canvas.envelope() && big.envelope(for: request(704, 1280, 141)) == canvas.envelope())
    #expect(try big.envelope(for: long) == clip.envelope() && big.envelope(for: request(960, 544, 141)) == canvas.envelope())
    #expect(try big.envelope(for: request(960, 544, 124)) == today.envelope())
    // A 24 GiB envelope never covers today's region, even alone and even where its canvases would.
    let extendedOnly = try H3MeasuredResources.admission(entries: [canvas, clip], coordinator: budget, physicalMemoryBytes: 64 << 30)
    let canvasEnvelope = try canvas.envelope(), clipEnvelope = try clip.envelope()
    for inside in [try request(960, 544, 124), try request(640, 352, 39), try request(960, 544, 39)] {
        #expect(!canvasEnvelope.covers(inside) && !clipEnvelope.covers(inside) && !extendedOnly.admits(inside))
    }
    #expect(extendedOnly.admits(try request(960, 544, 141)))
    do { _ = try extendedOnly.envelope(for: request(960, 544, 124)); Issue.record("expected a refusal") } catch {
        #expect("\(error)".contains("18 GiB measured entry"))
    }
    // With an 18 GiB entry present, a refused request inside today's region is pointed at it, not at a 24 GiB one.
    let todayT2VA = try entry(["960x544", "544x960"], frames: 124, denoising: 11 * gib, tasks: [.t2va], keyframe: false)
    let mixed = try H3MeasuredResources.admission(entries: [todayT2VA, canvas], coordinator: budget, physicalMemoryBytes: 64 << 30)
    do { _ = try mixed.envelope(for: request(960, 544, 124, first: true)); Issue.record("expected a refusal") } catch {
        #expect("\(error)".contains("nearest: 960x544, 544x960 x 124 frames"))
    }
    #expect(throws: AetherError.self) { _ = try entry(["99999999999999999999x2"], frames: 124, denoising: gib) }      // does not parse
    // Never measured: the long clip in portrait, a canvas wider than any measured, the large canvas at the long clip,
    // and FL2VA where only T2VA was measured.
    for r in [try request(544, 960, 243), try request(1408, 640, 141), try request(1280, 704, 158), try request(1280, 704, 141, first: true)] {
        #expect(!big.admits(r))
    }
}

@Test func h3SingleEntryLookupCountsRecordedEntriesBeforeTheMemoryFilter() throws {
    let today = try entry(["960x544"], frames: 124, denoising: 11 * gib)
    let extended = try entry(["1280x704"], frames: 141, denoising: 20 * gib, bar: H3MeasuredResources.extendedMemoryBarBytes)
    for memory: UInt64 in [64 << 30, 32 << 30] {
        #expect(throws: AetherError.self) {
            _ = try H3MeasuredResources.entry([today, extended], executionIdentity: "fixture/1", deviceArchitecture: "fixture", physicalMemoryBytes: memory)
        }
    }
    #expect(try H3MeasuredResources.entry([today], executionIdentity: "fixture/1", deviceArchitecture: "fixture") == today)
    #expect(throws: AetherError.self) {
        _ = try H3MeasuredResources.entry([extended], executionIdentity: "fixture/1", deviceArchitecture: "fixture", physicalMemoryBytes: 32 << 30)
    }
}

@Test func h3MeasuredEntriesDecodeWithTheDefaultBarAndRefuseLocalEvidencePaths() throws {
    let json = #"{"execution_identity": "fixture/1", "device_architecture": "fixture", "canvases": ["960x544"], "maximum_frames": 124,"#
        + #" "stage_peak_bytes": {"encoding": 1, "denoising": 2, "decodingVideo": 3}, "operation_bytes": 4, "evidence": "docs/evidence/x.json"}"#
    let decoded = try JSONDecoder().decode(H3MeasuredResources.self, from: Data(json.utf8))
    #expect(decoded.memoryBarBytes == H3MeasuredResources.memoryBarBytes && decoded.tasks == nil)
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as! [String: Any]
    #expect(encoded["memory_bar_bytes"] == nil && encoded["tasks"] == nil)
    let extended = try entry(["960x544"], frames: 243, denoising: 20 * gib, tasks: [.t2va], keyframe: false, bar: H3MeasuredResources.extendedMemoryBarBytes)
    #expect(try JSONDecoder().decode(H3MeasuredResources.self, from: JSONEncoder().encode(extended)) == extended)
    for path in ["/Volumes/x/gate.json", "~/gate.json", "unrecorded"] {
        #expect(throws: AetherError.self) {
            _ = try H3MeasuredResources(executionIdentity: "fixture/1", deviceArchitecture: "fixture", canvases: ["960x544"], maximumFrames: 124,
                                        stagePeakBytes: ["encoding": 1, "denoising": 2, "decodingVideo": 3], operationBytes: 4, evidence: path)
        }
    }
}

@Test func h3MeasuredResourcesOverTheEighteenGiBBarAreRefused() throws {
    #expect(throws: AetherError.self) { _ = try measured(denoising: 17 * gib) }               // 17 GiB + margin + 1 GiB > 18 GiB
    #expect(throws: AetherError.self) { _ = try measured(denoising: 13 * gib, operation: 5 * gib) }
    _ = try measured(denoising: 16 * gib, operation: gib / 2)                                    // 16.75 GiB fits
    #expect(throws: AetherError.self) {
        _ = try H3MeasuredResources(executionIdentity: "fixture/1", deviceArchitecture: "fixture", canvases: ["960x544"], maximumFrames: 124,
                                    stagePeakBytes: ["denoising": gib], operationBytes: gib, evidence: "fixture")   // missing required stages
    }
    // Every recorded entry decodes and is within its bar; an unmeasured identity has no estimate.
    for entry in try H3MeasuredResources.bundled() { try entry.validate() }
    #expect(throws: AetherError.self) { _ = try H3MeasuredResources.bundled(executionIdentity: "unmeasured", deviceArchitecture: "fixture") }
}

/// Not opt-in: the bundled entry for the qualified tiled-encoder profile in its Kit-owned home (memory gate, both
/// orientations, T2VA and FL2VA) keeps its 18 GiB bar and reservations on any device memory. Coverage is narrower than
/// before envelopes: a canvas must fit inside 960x544 or 544x960, so 1024x480, 640x800 and 576x576 are now refused.
@Test func h3BundledQualifiedEntryKeepsItsBarAndReservations() throws {
    let identity = "h3.profile/1:5af39db57a6ffc513d9c71f43d6875a44bd44b4ca53e3d29cf9e76cc205992ce:turbo8"
    let recorded = try H3MeasuredResources.bundled(executionIdentity: identity, deviceArchitecture: "h17c", physicalMemoryBytes: 8 << 30)
    #expect(Set(recorded.stagePeakBytes.keys) == ["encoding", "encodingKeyframe", "denoising", "decodingVideo", "decodingAudio"])
    #expect(recorded.maximumFrames == 124 && recorded.canvases == ["960x544", "544x960"] && recorded.tasks == nil)
    #expect(recorded.memoryBarBytes == H3MeasuredResources.memoryBarBytes && recorded.minimumPhysicalMemoryBytes == 0)
    #expect(recorded.evidence == "docs/evidence/h3-memory-gate-review-2026-10-28.json")
    #expect(recorded.stagePeakBytes.values.max()! + H3MeasuredResources.marginBytes + recorded.operationBytes <= H3MeasuredResources.memoryBarBytes)
    let budget = try ResourceCoordinator(capacityBytes: H3MeasuredResources.memoryBarBytes)
    let admission = try H3MeasuredResources.bundledAdmission(executionIdentity: identity, deviceArchitecture: "h17c", coordinator: budget,
                                                             physicalMemoryBytes: 8 << 30)
    #expect(admission.envelopes.count == 1 && admission.operationBytes == recorded.operationBytes)
    let envelope = admission.envelopes[0]
    #expect(envelope.stageBytes.count == 5 && envelope.stageBytes.allSatisfy {
        $0.value == recorded.stagePeakBytes[$0.key.rawValue]! + H3MeasuredResources.marginBytes
    })
    let first = try request(960, 544, 124, first: true, audio: true), portrait = try request(544, 960, 124, audio: true)
    let longer = try request(960, 544, 141), larger = try request(1280, 704, 124)
    #expect(admission.admits(first) && admission.admits(portrait) && !admission.admits(longer) && !admission.admits(larger))
    let wide = try request(1024, 480, 124), tall = try request(640, 800, 124), square = try request(576, 576, 124)
    #expect(!admission.admits(wide) && !admission.admits(tall) && !admission.admits(square))
}

@Test func h3VideoSessionRefusesAGenerationUnderCriticalPressure() async throws {
    let log = StageLog()
    let monitor = MemoryPressureMonitor(systemSource: false)
    let session = H3VideoSession(stages: H3VideoStages(conditioner: FakeConditioner(log: log), denoiser: FakeDenoiser(log: log),
                                                       video: FakeWindowDecoder(log: log), audio: FakeAudioDecoder(log: log)),
                                 pressure: monitor)
    let request = try H3VideoRequest(prompt: "integrated_multimodal_description: a quiet room", geometry: geometry(), seed: 3)
    monitor.deliver(.critical)
    await #expect(throws: AetherError.memoryPressure) { try await session.generate(request) }
    #expect(await log.entries.isEmpty)
    #expect(await session.state == .ready)
    monitor.deliver(.normal)
    _ = try await session.generate(request)
    #expect(await log.entries.first == "acquire conditioner")
}
