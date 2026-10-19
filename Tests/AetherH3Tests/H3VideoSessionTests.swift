import AetherKit
import AetherH3
import Foundation
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
