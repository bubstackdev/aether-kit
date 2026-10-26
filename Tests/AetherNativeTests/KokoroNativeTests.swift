import AetherCoreAI
import AetherKit
import AetherSpeech
import AetherSpeechCoreAI
import CryptoKit
import Foundation
import Synchronization
import Testing

// Stated before the first native run (docs/kokoro-speech.md#tolerances):
// - phoneme token ids and integer frames per token must match upstream exactly;
// - waveform relative L2 <= 0.05 and log-mel mean absolute difference <= 0.03 nats;
// - waveform relative L2 <= 4x the case's own float32-versus-float64 upstream difference,
//   i.e. native error stays within a small multiple of the model's intrinsic float32 noise.
private let maximumRelativeL2 = 0.05
private let maximumLogMelMean = 0.03
private let precisionFloorMultiple = 4.0

private struct ReferenceCase: Decodable {
    let name: String
    let voice: String
    let speed: Float
    let seed: UInt64
    let phonemes: String
    let input_ids: [Int32]
    let frames: [Int]
    let samples: Int
    let audio_sha256: String
    let perturbed_sha256: String
    let sensitivity_floor: KokoroMetrics.Comparison
    let float64_upstream: KokoroMetrics.Comparison
}

private struct CaseReport: Encodable {
    let name: String
    let voice: String
    let phonemeCount: Int
    let audioSeconds: Double
    let tokensMatch: Bool
    let framesMatch: Bool
    let native: KokoroMetrics.Comparison?
    let swiftSensitivityFloor: KokoroMetrics.Comparison
    let pythonSensitivityFloor: KokoroMetrics.Comparison
    let float64Floor: KokoroMetrics.Comparison
    let firstSynthesisSeconds: Double
    let warmSynthesisSeconds: Double
    let warmRealTimeFactor: Double
    let repeatIdentical: Bool
    let audioSHA256: String
    let decoderFrames: Int
    let footprintMiBAfter: Double
    let metalMiBAfter: Double
    let passed: Bool
}

private struct ShapeTiming: Encodable {
    let phonemes: Int
    let frames: Int
    let firstSeconds: Double
    let repeatSeconds: Double
    let decoderFrames: Int
}

private struct LifecycleReport: Encodable {
    var cancellationError: String = ""
    var stateAfterCancellation: String = ""
    var recoveryMatchesBaseline = false
    var unloadClosedWhileActive = false
    var activeCompletedAfterUnload = false
    var activeMatchesBaseline = false
    var rejectedAfterUnload = false
}

private struct KokoroQualificationReport: Encodable {
    var passed = false
    var compute = ""
    var bundleFingerprint = ""
    var bundleVerified = false
    var references = ""
    var tolerances = ["maximumRelativeL2": maximumRelativeL2, "maximumLogMelMean": maximumLogMelMean,
                      "precisionFloorMultiple": precisionFloorMultiple]
    var host: [String: String] = [:]
    var loadAndVerifySeconds = 0.0
    var cases: [CaseReport] = []
    var novelShapes: [ShapeTiming] = []
    var lifecycle = LifecycleReport()
    var decoderBuckets: [Int] = []
    var memoryMiB: [String: Double] = [:]
    var failures: [String] = []
}

private func floats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
    }
}

private func sha256Hex(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

private func sha(_ values: [Float]) -> String {
    values.withUnsafeBufferPointer { sha256Hex(Data(buffer: $0)) }
}

private func hardwareModel() -> String {
    var size = 0
    sysctlbyname("hw.model", nil, &size, nil, 0)
    var model = [CChar](repeating: 0, count: size)
    sysctlbyname("hw.model", &model, &size, nil, 0)
    return String(decoding: model.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

private final class Signal: Sendable {
    private let fired = Mutex(false)
    func fire() { fired.withLock { $0 = true } }
    var isFired: Bool { fired.withLock { $0 } }
}

@Test(.enabled(if: nativeEnvironment["AETHER_NATIVE_KOKORO_QUALIFICATION"] == "1"), .timeLimit(.minutes(30)))
func kokoroNativeParityTimingAndLifecycle() async throws {
    let output = try nativePath("AETHER_NATIVE_KOKORO_OUTPUT")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: false)
    var report = KokoroQualificationReport()
    func save() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
    }
    func check(_ condition: Bool, _ message: String) {
        if !condition { report.failures.append(message) }
        #expect(condition, Comment(rawValue: message))
    }
    // "recommended" (durations on CPU, decoder on GPU) or one ModelComputePolicy for both graphs.
    let computeName = nativeEnvironment["AETHER_NATIVE_KOKORO_COMPUTE"] ?? "recommended"
    let compute: KokoroComputePlacement
    if let uniform = ModelComputePolicy(rawValue: computeName) { compute = .uniform(uniform) }
    else if computeName == "recommended" { compute = .recommended }
    else { throw NativeFailure(description: "unknown compute \(computeName)") }
    report.compute = "durations=\(compute.durations.rawValue) decoder=\(compute.decoder.rawValue)"
    let info = ProcessInfo.processInfo
    report.host = ["os": info.operatingSystemVersionString, "model": hardwareModel(),
                   "memoryGiB": String(info.physicalMemory >> 30), "activeProcessors": String(info.activeProcessorCount)]
    defer { try? save() }

    let started = ContinuousClock.now
    let assets = try KokoroAssets.open(at: nativePath("AETHER_KOKORO_BUNDLE"))
    try assets.verify()
    report.bundleVerified = true
    report.bundleFingerprint = assets.fingerprint
    report.decoderBuckets = assets.decoderBuckets
    let referenceRoot = try nativePath("AETHER_KOKORO_REFERENCE")
    report.references = sha256Hex(try Data(contentsOf: referenceRoot.appendingPathComponent("references.json")))
    let index = try JSONSerialization.jsonObject(with: Data(contentsOf: referenceRoot.appendingPathComponent("references.json"))) as! [String: Any]
    let names = (index["cases"] as! [[String: Any]]).map { $0["name"] as! String }
    let session = KokoroSpeechSession(assets: assets, compute: compute)
    report.loadAndVerifySeconds = (ContinuousClock.now - started).seconds

    var baselines: [String: [Float]] = [:]
    for name in names {
        let directory = referenceRoot.appendingPathComponent(name)
        let reference = try JSONDecoder().decode(ReferenceCase.self, from: Data(contentsOf: directory.appendingPathComponent("case.json")))
        let expected = try floats(directory.appendingPathComponent("audio.f32"))
        let perturbed = try floats(directory.appendingPathComponent("perturbed.f32"))
        check(sha(expected) == reference.audio_sha256 && sha(perturbed) == reference.perturbed_sha256, "\(name): reference audio hash")
        // Cross-check this metric implementation against the Python one on the same pair.
        let swiftFloor = KokoroMetrics.compare(perturbed, expected)
        let python = reference.sensitivity_floor
        check(abs(swiftFloor.rel_l2 - python.rel_l2) <= 1e-9 + 1e-6 * python.rel_l2
              && abs(swiftFloor.log_mel_mean_abs - python.log_mel_mean_abs) <= 1e-9 + 1e-4 * python.log_mel_mean_abs,
              "\(name): Swift and Python metrics disagree")

        let phonemes = try session.phonemes(reference.phonemes)
        let configuration = try KokoroSpeechConfiguration(voice: KokoroVoice(reference.voice), speed: reference.speed, seed: reference.seed)
        let first = ContinuousClock.now
        let speech = try await session.synthesize(phonemes, configuration: configuration)
        let firstSeconds = (ContinuousClock.now - first).seconds
        let warm = ContinuousClock.now
        let repeated = try await session.synthesize(phonemes, configuration: configuration)
        let warmSeconds = (ContinuousClock.now - warm).seconds
        let tokensMatch = phonemes.tokenIDs == reference.input_ids
        let framesMatch = speech.frames == reference.frames
        check(tokensMatch, "\(name): token ids differ from upstream")
        check(framesMatch, "\(name): frames per token differ from upstream")
        var comparison: KokoroMetrics.Comparison?
        var passed = tokensMatch && framesMatch
        if speech.audio.samples.count == expected.count {
            let value = KokoroMetrics.compare(speech.audio.samples, expected)
            comparison = value
            let bound = min(maximumRelativeL2, precisionFloorMultiple * reference.float64_upstream.rel_l2)
            check(value.rel_l2 <= bound, "\(name): waveform rel L2 \(value.rel_l2) exceeds \(bound)")
            check(value.log_mel_mean_abs <= maximumLogMelMean, "\(name): log-mel mean \(value.log_mel_mean_abs) exceeds \(maximumLogMelMean)")
            passed = passed && value.rel_l2 <= bound && value.log_mel_mean_abs <= maximumLogMelMean
        } else {
            check(false, "\(name): sample count \(speech.audio.samples.count) differs from \(expected.count)")
            passed = false
        }
        try speech.audio.saveWAV(to: output.appendingPathComponent("\(name).wav"))
        baselines[name] = speech.audio.samples
        report.cases.append(CaseReport(
            name: name, voice: reference.voice, phonemeCount: phonemes.count, audioSeconds: speech.audio.durationSeconds,
            tokensMatch: tokensMatch, framesMatch: framesMatch, native: comparison, swiftSensitivityFloor: swiftFloor,
            pythonSensitivityFloor: python, float64Floor: reference.float64_upstream,
            firstSynthesisSeconds: firstSeconds, warmSynthesisSeconds: warmSeconds,
            warmRealTimeFactor: warmSeconds / speech.audio.durationSeconds,
            repeatIdentical: repeated.audio.samples == speech.audio.samples,
            audioSHA256: sha(speech.audio.samples), decoderFrames: assets.decoderFrames(for: speech.frames.reduce(0, +)),
            footprintMiBAfter: Double(kokoroMemorySnapshot().footprint) / 1_048_576,
            metalMiBAfter: Double(kokoroMemorySnapshot().metal) / 1_048_576, passed: passed))
        try save()
    }

    // Latency at utterance lengths not yet seen by this session (new dynamic shapes), then repeated.
    for text in ["ðə sˈʌn ɹˈOz.", "wˈʌn, tˈu, θɹˈi, fˈɔɹ, fˈIv, sˈɪks.", "ʃi sˈɛlz sˈi ʃˈɛlz bI ðə sˈi ʃˈɔɹ ˈɛvɹi sˈʌmɚ mˈɔɹnɪŋ."] {
        let phonemes = try session.phonemes(text)
        let configuration = try KokoroSpeechConfiguration()
        let first = ContinuousClock.now
        let speech = try await session.synthesize(phonemes, configuration: configuration)
        let firstSeconds = (ContinuousClock.now - first).seconds
        let again = ContinuousClock.now
        _ = try await session.synthesize(phonemes, configuration: configuration)
        report.novelShapes.append(ShapeTiming(phonemes: phonemes.count, frames: speech.frames.reduce(0, +),
                                              firstSeconds: firstSeconds, repeatSeconds: (ContinuousClock.now - again).seconds,
                                              decoderFrames: assets.decoderFrames(for: speech.frames.reduce(0, +))))
    }
    try save()

    // Lifecycle on the longest case: cancel once frames are predicted (decoder next), then recover.
    let longName = try #require(names.last)
    let long = try JSONDecoder().decode(ReferenceCase.self, from: Data(contentsOf: referenceRoot.appendingPathComponent(longName).appendingPathComponent("case.json")))
    let phonemes = try session.phonemes(long.phonemes)
    let configuration = try KokoroSpeechConfiguration(voice: KokoroVoice(long.voice), speed: long.speed, seed: long.seed)
    let predicted = Signal()
    let cancelled = Task { try await session.synthesize(phonemes, configuration: configuration) { if case .framesPredicted = $0 { predicted.fire() } } }
    while !predicted.isFired { try await Task.sleep(for: .milliseconds(1)) }
    cancelled.cancel()
    do {
        _ = try await cancelled.value
        report.lifecycle.cancellationError = "completed before cancellation"
    } catch {
        report.lifecycle.cancellationError = String(describing: type(of: error))
    }
    report.lifecycle.stateAfterCancellation = await session.state.rawValue
    check(report.lifecycle.cancellationError == "CancellationError", "cancellation did not surface CancellationError")
    let recovered = try await session.synthesize(phonemes, configuration: configuration)
    report.lifecycle.recoveryMatchesBaseline = recovered.audio.samples == baselines[longName]
    check(report.lifecycle.recoveryMatchesBaseline, "recovery after cancellation differs from the baseline")

    let predictedAgain = Signal()
    let active = Task { try await session.synthesize(phonemes, configuration: configuration) { if case .framesPredicted = $0 { predictedAgain.fire() } } }
    while !predictedAgain.isFired { try await Task.sleep(for: .milliseconds(1)) }
    let beforeUnload = kokoroMemorySnapshot()
    await session.unload()
    let afterUnload = kokoroMemorySnapshot()
    report.memoryMiB = ["footprintBeforeUnload": Double(beforeUnload.footprint) / 1_048_576,
                        "metalBeforeUnload": Double(beforeUnload.metal) / 1_048_576,
                        "footprintAfterUnload": Double(afterUnload.footprint) / 1_048_576,
                        "metalAfterUnload": Double(afterUnload.metal) / 1_048_576]
    report.lifecycle.unloadClosedWhileActive = await session.state == .closed
    let finished = try? await active.value
    report.lifecycle.activeCompletedAfterUnload = finished != nil
    report.lifecycle.activeMatchesBaseline = finished?.audio.samples == baselines[longName]
    do {
        _ = try await session.synthesize(phonemes, configuration: configuration)
    } catch AetherError.sessionClosed {
        report.lifecycle.rejectedAfterUnload = true
    }
    check(report.lifecycle.unloadClosedWhileActive && report.lifecycle.activeCompletedAfterUnload
          && report.lifecycle.activeMatchesBaseline && report.lifecycle.rejectedAfterUnload, "unload lifecycle")
    report.passed = report.failures.isEmpty
    try save()
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
