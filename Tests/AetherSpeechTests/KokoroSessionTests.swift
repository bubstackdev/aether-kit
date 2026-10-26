import AetherCoreAI
import AetherKit
import AetherSpeech
import Foundation
import Synchronization
import Testing
@testable import AetherSpeechCoreAI

private func session(_ factory: FakeFactory, bundle: FakeBundle, resources: KokoroResourceAdmission? = nil) throws -> KokoroSpeechSession {
    KokoroSpeechSession(assets: try KokoroAssets.open(at: bundle.root), factory: factory, resources: resources)
}

@Test func synthesisAlignsFramesSeedsNoiseAndReusesModels() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let log = EventLog()
    let factory = FakeFactory(log: log)
    let speech = try session(factory, bundle: bundle)
    let phonemes = try speech.phonemes("həlˈO")
    let events = EventLog()
    let result = try await speech.synthesize(phonemes, configuration: try KokoroSpeechConfiguration(voice: .bmGeorge, seed: 5)) { event in
        switch event {
        case .framesPredicted(let segment, let frames): events.append("frames:\(segment):\(frames)")
        case .chunk(let chunk): events.append("chunk:\(chunk.index)/\(chunk.count)")
        }
    }
    let tokens = phonemes.tokenIDs.count
    #expect(result.frames == Array(repeating: 2, count: tokens))
    #expect(result.audio.sampleRate == 24_000 && result.audio.frameCount == 2 * tokens * 600)
    #expect(events.snapshot == ["frames:0:\(Array(repeating: 2, count: tokens))", "chunk:0/1"])
    let first = try #require(factory.observed.withLock { $0.first })
    #expect(first.noise == (try KokoroNoise.generate(seed: 5, count: 1))[0])
    #expect(first.style == Float(1000 + phonemes.count - 1))  // bm_george is the second fake voice
    #expect(first.prosody == (0..<tokens).flatMap { [Float($0), Float($0)] })

    // Chunked request: segment i uses seed + i, chunks arrive in order, models stay loaded.
    let segments = try ["ðə", "kwˈɪk", "fˈɑks"].map { try speech.phonemes($0) }
    let chunks = EventLog()
    let parts = try await speech.synthesize(segments: segments, configuration: try KokoroSpeechConfiguration(seed: 100)) { event in
        if case .chunk(let chunk) = event { chunks.append("\(chunk.index):\(chunk.speech.phonemes)") }
    }
    #expect(chunks.snapshot == ["0:ðə", "1:kwˈɪk", "2:fˈɑks"])
    #expect(parts.map(\.configuration.seed) == [100, 101, 102])
    let noises = factory.observed.withLock { $0.dropFirst().map(\.noise) }
    #expect(noises == (try [100, 101, 102].map { try KokoroNoise.generate(seed: $0, count: 1)[0] }))
    let joined = try GeneratedAudio.concatenating(parts.map(\.audio))
    #expect(joined.frameCount == parts.map(\.audio.frameCount).reduce(0, +))
    #expect(log.snapshot.filter { $0.hasPrefix("open") } == ["open:durations", "open:decode"])
    await speech.unload()
    #expect(log.snapshot.suffix(2) == ["close:durations", "close:decode"])
    #expect(await speech.state == .closed)
}

@Test func overlappingRequestsFailBusyAndCancellationRecovers() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let log = EventLog()
    let gate = Gate()
    let speech = try session(FakeFactory(log: log, decoderGate: gate), bundle: bundle)
    let phonemes = try speech.phonemes("ðə")
    let configuration = try KokoroSpeechConfiguration()
    let task = Task { try await speech.synthesize(phonemes, configuration: configuration) }
    await gate.waitUntilEntered()
    await #expect(throws: AetherError.sessionBusy) { try await speech.synthesize(phonemes, configuration: configuration) }
    #expect(await speech.state == .running)
    task.cancel()
    await gate.release()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await speech.state == .ready)
    // The submitted decoder call returned before cancellation was delivered; models are retained.
    let recovered = try await speech.synthesize(phonemes, configuration: configuration)
    #expect(recovered.audio.frameCount == 2 * phonemes.tokenIDs.count * 600)
    #expect(log.snapshot.filter { $0.hasPrefix("open") }.count == 2)
    await speech.unload()
}

@Test func unloadWaitsForActiveWorkThenClosesPermanently() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let log = EventLog()
    let gate = Gate()
    let speech = try session(FakeFactory(log: log, durationsGate: gate), bundle: bundle)
    let phonemes = try speech.phonemes("ðə")
    let configuration = try KokoroSpeechConfiguration()
    let task = Task { try await speech.synthesize(phonemes, configuration: configuration) }
    await gate.waitUntilEntered()
    let unloading = Task { await speech.unload() }
    while await speech.state != .closing { await Task.yield() }
    await #expect(throws: AetherError.sessionClosed) { try await speech.synthesize(phonemes, configuration: configuration) }
    #expect(!log.snapshot.contains("close:durations"))
    await gate.release()
    let completed = try await task.value
    await unloading.value
    #expect(completed.audio.frameCount > 0)
    #expect(await speech.state == .closed)
    let closes = log.snapshot.filter { $0.hasPrefix("close") }
    #expect(closes == ["close:durations", "close:decode"])
    let lastRun = try #require(log.snapshot.lastIndex { $0.hasPrefix("run") })
    let firstClose = try #require(log.snapshot.firstIndex { $0.hasPrefix("close") })
    #expect(lastRun < firstClose)
    await speech.unload()
    #expect(log.snapshot.filter { $0.hasPrefix("close") }.count == 2)
    await #expect(throws: AetherError.sessionClosed) { try await speech.synthesize(phonemes, configuration: configuration) }
}

@Test func contractViolationsAndLimitsFailBeforeDecoding() async throws {
    let bundle = try FakeBundle.make(maximumFrames: 8)
    defer { bundle.remove() }
    let log = EventLog()
    let bad = try session(FakeFactory(log: log, badDurationsSignature: true), bundle: bundle)
    await #expect(throws: AetherError.self) { try await bad.synthesize(try bad.phonemes("ðə"), configuration: try KokoroSpeechConfiguration()) }
    #expect(log.snapshot == ["open:durations", "close:durations"])
    #expect(await bad.state == .ready)
    await bad.unload()

    let limited = EventLog()
    let speech = try session(FakeFactory(log: limited, framesPerToken: 3), bundle: bundle)
    // 4 tokens x 3 frames = 12 > the bundle's 8-frame limit.
    await #expect(throws: AetherError.self) { try await speech.synthesize(try speech.phonemes("ðə"), configuration: try KokoroSpeechConfiguration()) }
    #expect(!limited.snapshot.contains("run:decode"))
    await #expect(throws: AetherError.self) {
        try await speech.synthesize(try speech.phonemes("ðə"), configuration: try KokoroSpeechConfiguration(voice: .afBella))
    }
    await #expect(throws: AetherError.self) {
        try await speech.synthesize(segments: [], configuration: try KokoroSpeechConfiguration())
    }
    await speech.unload()
}

@Test func admissionReservesOperationStorageAndRefusesOversizedRequests() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let assets = try KokoroAssets.open(at: bundle.root)
    let coordinator = try ResourceCoordinator(capacityBytes: 100)
    let durations = try KokoroResourceAdmission.ModelEstimate(asset: assets.durations, residentBytes: 10, runScratchBytes: 1)
    let decoder = try KokoroResourceAdmission.ModelEstimate(asset: assets.decoder, residentBytes: 10, runScratchBytes: 1)
    #expect(throws: AetherError.self) {
        try KokoroResourceAdmission(coordinator: coordinator, operationBytes: 10, maximumFrames: 100, durations: durations, decoder: durations)
    }
    let gate = Gate()
    let admission = try KokoroResourceAdmission(coordinator: coordinator, operationBytes: 60, maximumFrames: 10,
                                                durations: durations, decoder: decoder)
    let speech = KokoroSpeechSession(assets: assets, factory: FakeFactory(log: EventLog(), decoderGate: gate), resources: admission)
    let phonemes = try speech.phonemes("ðə")  // 4 tokens x 2 frames = 8 <= 10
    let task = Task { try await speech.synthesize(phonemes, configuration: try KokoroSpeechConfiguration()) }
    await gate.waitUntilEntered()
    #expect(coordinator.snapshot.reservedBytes == 60)
    let competitor = try coordinator.reserve(bytes: 30)
    await gate.release()
    _ = try await task.value
    #expect(coordinator.snapshot.reservedBytes == 30)
    competitor.release()
    let hog = try coordinator.reserve(bytes: 50)
    await #expect(throws: ResourceAdmissionError.self) { try await speech.synthesize(phonemes, configuration: try KokoroSpeechConfiguration()) }
    hog.release()
    // 5 tokens x 2 frames = 10 is admitted; 6 tokens x 2 = 12 exceeds the profile's maximumFrames.
    _ = try await speech.synthesize(try speech.phonemes("ðəə"), configuration: try KokoroSpeechConfiguration())
    await #expect(throws: AetherError.self) { try await speech.synthesize(try speech.phonemes("ðəəə"), configuration: try KokoroSpeechConfiguration()) }
    #expect(coordinator.snapshot.reservedBytes == 0)
    await speech.unload()
}

private struct ListPhonemizer: SpeechPhonemizer {
    func phonemize(_ text: String, language: KokoroVoice.Language) async throws -> [String] {
        #expect(language == .britishEnglish)
        return text.split(separator: "|").map(String.init)
    }
}

@Test func textSynthesisUsesTheAppPhonemizer() async throws {
    let bundle = try FakeBundle.make()
    defer { bundle.remove() }
    let speech = try session(FakeFactory(log: EventLog()), bundle: bundle)
    let parts = try await speech.synthesize(text: "ðə|kwˈɪk", phonemizer: ListPhonemizer(),
                                            configuration: try KokoroSpeechConfiguration(voice: .bmGeorge))
    #expect(parts.map(\.phonemes.string) == ["ðə", "kwˈɪk"])
    await #expect(throws: AetherError.self) {
        try await speech.synthesize(text: "1", phonemizer: ListPhonemizer(), configuration: try KokoroSpeechConfiguration(voice: .bmGeorge))
    }
    await speech.unload()
}
