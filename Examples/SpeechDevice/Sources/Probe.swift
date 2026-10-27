import AetherCoreAI
import AetherKit
import AetherSpeech
import AetherSpeechCoreAI
import Darwin
import Foundation

// Same tolerances as the Mac native qualification (Tests/AetherNativeTests/KokoroNativeTests.swift).
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
    let float64_upstream: KokoroMetrics.Comparison
}

struct CaseResult: Encodable {
    let name: String
    let frames: Int
    let decoderFrames: Int
    let refusedAboveCap: Bool
    let framesMatch: Bool
    let tokensMatch: Bool
    let native: KokoroMetrics.Comparison?
    let bound: Double
    let firstSeconds: Double
    let warmSeconds: Double
    let warmRealTimeFactor: Double
    let footprintMiB: Double
    let passed: Bool
}

struct DeviceReport: Encodable {
    var device: [String: String] = [:]
    var bundleFingerprint = ""
    var decoderPrecision = ""
    var frameCap = 0
    var availableMemoryMiBAtStart = 0.0
    var availableMemoryMiBMinimum = 0.0
    var loadAndVerifySeconds = 0.0
    var cases: [CaseResult] = []
    /// Memory probe at a larger bucket: a prefix of the longest case's phonemes (no reference audio).
    var bucketProbe: [String: Double] = [:]
    var footprintPeakMiB = 0.0
    var footprintAfterUnloadMiB = 0.0
    var passed = false
    var failures: [String] = []
}

private func floats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
    }
}

/// Current and peak physical footprint, the figures jetsam limits are enforced against.
private func footprint() -> (current: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    guard result == KERN_SUCCESS else { return (0, 0) }
    return (Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576)
}

private func availableMemory() -> Double {
    #if os(iOS)
    return Double(os_proc_available_memory()) / 1_048_576
    #else
    return 0
    #endif
}

private func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var value = [UInt8](repeating: 0, count: size)
    sysctlbyname(name, &value, &size, nil, 0)
    return String(decoding: value.prefix { $0 != 0 }, as: UTF8.self)
}

/// One progress line per stage, flushed immediately, so a run that is killed (for example by jetsam)
/// still shows how far it got and the memory state at each step.
private func progress(_ stage: String) {
    let memory = footprint()
    print(String(format: "KOKORO_DEVICE_PROGRESS %@ footprint=%.0fMiB peak=%.0fMiB available=%.0fMiB", stage, memory.current, memory.peak, availableMemory()))
    fflush(stdout)
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}

/// Synthesizes the staged parity cases on the device and reports parity, real-time factor and memory.
func runKokoroDeviceProbe() async -> DeviceReport {
    var report = DeviceReport()
    let info = ProcessInfo.processInfo
    report.device = ["model": sysctlString("hw.machine"), "os": info.operatingSystemVersionString,
                     "memoryGiB": String(format: "%.1f", Double(info.physicalMemory) / 1_073_741_824)]
    report.availableMemoryMiBAtStart = availableMemory()
    report.availableMemoryMiBMinimum = report.availableMemoryMiBAtStart
    do {
        guard let assetsRoot = Bundle.main.url(forResource: "Assets", withExtension: nil) else {
            report.failures.append("no staged Assets folder in the app bundle")
            return report
        }
        let references = assetsRoot.appendingPathComponent("references")
        let capText = (try? String(contentsOf: assetsRoot.appendingPathComponent("frame-cap.txt"), encoding: .utf8)) ?? "1024"
        let cap = Int(capText.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 1024
        let started = ContinuousClock.now
        let assets = try KokoroAssets.open(at: assetsRoot.appendingPathComponent("bundle"))
        try assets.verify()
        report.loadAndVerifySeconds = (ContinuousClock.now - started).seconds
        progress("verified bundle")
        report.bundleFingerprint = assets.fingerprint
        report.decoderPrecision = assets.decoderPrecision
        report.frameCap = cap
        let session = KokoroSpeechSession(assets: assets, frameCap: try KokoroFrameCap(frames: cap))
        // Shortest first, so a memory limit shows up at the smallest bucket that hits it.
        var cases: [(name: String, reference: ReferenceCase, frames: Int)] = []
        for name in try FileManager.default.contentsOfDirectory(atPath: references.path) where !name.hasPrefix(".") {
            let url = references.appendingPathComponent(name).appendingPathComponent("case.json")
            let reference = try JSONDecoder().decode(ReferenceCase.self, from: Data(contentsOf: url))
            cases.append((name, reference, reference.frames.reduce(0, +)))
        }
        cases.sort { $0.frames < $1.frames }
        for (name, reference, _) in cases {
            let directory = references.appendingPathComponent(name)
            progress("start \(name) frames=\(reference.frames.reduce(0, +))")
            let expected = try floats(directory.appendingPathComponent("audio.f32"))
            let phonemes = try session.phonemes(reference.phonemes)
            let configuration = try KokoroSpeechConfiguration(voice: KokoroVoice(reference.voice), speed: reference.speed, seed: reference.seed)
            let frames = reference.frames.reduce(0, +)
            if frames > cap {
                var refused = false
                do { _ = try await session.synthesize(phonemes, configuration: configuration) } catch AetherError.invalidRequest { refused = true }
                if !refused { report.failures.append("\(name): \(frames) frames above the \(cap)-frame cap was not refused") }
                report.cases.append(CaseResult(name: name, frames: frames, decoderFrames: 0, refusedAboveCap: refused, framesMatch: false,
                                               tokensMatch: true, native: nil, bound: 0, firstSeconds: 0, warmSeconds: 0,
                                               warmRealTimeFactor: 0, footprintMiB: footprint().current, passed: refused))
                continue
            }
            let first = ContinuousClock.now
            let speech = try await session.synthesize(phonemes, configuration: configuration)
            let firstSeconds = (ContinuousClock.now - first).seconds
            progress("first \(name) \(String(format: "%.2f", firstSeconds))s")
            report.availableMemoryMiBMinimum = min(report.availableMemoryMiBMinimum, availableMemory())
            let warm = ContinuousClock.now
            _ = try await session.synthesize(phonemes, configuration: configuration)
            let warmSeconds = (ContinuousClock.now - warm).seconds
            report.availableMemoryMiBMinimum = min(report.availableMemoryMiBMinimum, availableMemory())
            let framesMatch = speech.frames == reference.frames
            let tokensMatch = phonemes.tokenIDs == reference.input_ids
            let bound = min(maximumRelativeL2, precisionFloorMultiple * reference.float64_upstream.rel_l2)
            var comparison: KokoroMetrics.Comparison?
            var passed = framesMatch && tokensMatch
            if speech.audio.samples.count == expected.count {
                let value = KokoroMetrics.compare(speech.audio.samples, expected)
                comparison = value
                passed = passed && value.rel_l2 <= bound && value.log_mel_mean_abs <= maximumLogMelMean
            } else {
                passed = false
            }
            if !passed { report.failures.append("\(name) failed parity") }
            progress("done \(name) passed=\(passed) warm=\(String(format: "%.3f", warmSeconds))s")
            report.cases.append(CaseResult(name: name, frames: frames, decoderFrames: assets.decoderFrames(for: frames), refusedAboveCap: false,
                                           framesMatch: framesMatch, tokensMatch: tokensMatch, native: comparison, bound: bound,
                                           firstSeconds: firstSeconds, warmSeconds: warmSeconds,
                                           warmRealTimeFactor: warmSeconds / speech.audio.durationSeconds,
                                           footprintMiB: footprint().current, passed: passed))
        }
        if let longest = cases.last, cap > 256 {
            let prefix = String(String.UnicodeScalarView(longest.reference.phonemes.unicodeScalars.prefix(160)))
            let configuration = try KokoroSpeechConfiguration(voice: KokoroVoice(longest.reference.voice), speed: longest.reference.speed,
                                                              seed: longest.reference.seed)
            progress("start bucket probe")
            let probeStart = ContinuousClock.now
            let speech = try await session.synthesize(try session.phonemes(prefix), configuration: configuration)
            let firstSeconds = (ContinuousClock.now - probeStart).seconds
            let warm = ContinuousClock.now
            _ = try await session.synthesize(try session.phonemes(prefix), configuration: configuration)
            let warmSeconds = (ContinuousClock.now - warm).seconds
            let frames = speech.frames.reduce(0, +)
            report.availableMemoryMiBMinimum = min(report.availableMemoryMiBMinimum, availableMemory())
            report.bucketProbe = ["frames": Double(frames), "decoderFrames": Double(assets.decoderFrames(for: frames)),
                                  "firstSeconds": firstSeconds, "warmSeconds": warmSeconds,
                                  "warmRealTimeFactor": warmSeconds / speech.audio.durationSeconds,
                                  "footprintMiB": footprint().current, "availableMiB": availableMemory()]
            progress("done bucket probe frames=\(frames)")
        }
        report.footprintPeakMiB = footprint().peak
        await session.unload()
        report.footprintAfterUnloadMiB = footprint().current
    } catch {
        report.failures.append("error: \(error)")
        report.footprintPeakMiB = footprint().peak
    }
    report.passed = report.failures.isEmpty
    return report
}
