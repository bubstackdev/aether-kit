import AetherKit
import AetherLanguageCoreAI
import AetherVision
import Darwin
import Foundation

/// The staged run: `Assets/probe.json` names the mode; `Assets/bundle` is an iOS-registered bundle and
/// `Assets/reference` its float32 Transformers reference. Mode "prepare" instead downloads `Assets/entry.json`
/// (PrepareProbe.swift).
private struct ProbeConfiguration: Decodable {
    /// "text": a dense Qwen3 bundle against `reference.json` + `forced-logits.f32`
    /// (`Forge/language/reference_qwen3.py`). "vision": a Qwen3-VL bundle against a reference-grids directory
    /// (`Forge/language/reference_qwen3_vl.py`) made from Kit's own decode of each photo.
    let mode: String
    /// Vision only: the reference cases to run (default: the first).
    let cases: [String]?
    /// Prepare only: stop the first run once this many bytes have downloaded, then prepare again.
    let cutAfterBytes: UInt64?
    /// Prepare and readiness: the store's folder in Application Support (default "Models").
    let store: String?
    /// Readiness only: replaces the entry's memory requirement.
    let memoryBytes: UInt64?
    /// Lifecycle only: the idle unload policy's seconds (default 10).
    let idleSeconds: Int?
}

private struct Logits: Decodable { let file: String; let rows: Int; let vocab: Int }
private struct TextReference: Decodable {
    let promptTokens: [Int]
    let greedyTokens: [Int]
    let forcedTokens: [Int]
    let forcedLogits: Logits
}
private struct VisionReference: Decodable {
    struct File: Decodable { let file: String }
    struct Case: Decodable {
        let name: String
        let promptTokens: [Int]
        let greedyTokens: [Int]
        let forcedTokens: [Int]
        let forcedLogits: Logits
        let image: File
    }
    let cases: [Case]
}

struct CaseResult: Codable {
    let name: String
    let promptTokens: Int
    /// Text mode: teacher-forced rows against the reference.
    let forcedRows: Int?
    let forcedMinimumCosine: Double?
    let forcedArgmaxMismatches: Int?
    let forcedMismatchesAtNearTies: Int?
    let greedyTokens: [Int]
    let referenceGreedyTokens: [Int]
    let greedyFirstDivergence: Int?
    /// At a divergence: the reference's top-1 minus top-2 probability on its teacher-forced row there.
    let greedyDivergenceReferenceMargin: Double?
    let greedyAccepted: Bool
    let timeToFirstTokenSeconds: Double
    let totalSeconds: Double
    let footprintPeakMiB: Double
    let passed: Bool
}

struct DeviceReport: Codable {
    var device: [String: String] = [:]
    var mode = ""
    var bundleModelSHA256 = ""
    var openSeconds = 0.0
    var availableMemoryMiBAtStart = 0.0
    var availableMemoryMiBMinimum = 0.0
    var footprintPeakMiB = 0.0
    var footprintAfterUnloadMiB = 0.0
    var cases: [CaseResult] = []
    /// Mode "prepare": the background preparation run (PrepareProbe.swift).
    var preparation: PreparationResult?
    /// Mode "readiness": the states a `ModelReadiness` passed through (ReadinessProbe.swift).
    var readiness: ReadinessResult?
    /// Mode "lifecycle": idle unload, reload and release under real memory pressure (LifecycleProbe.swift).
    var lifecycle: LifecycleResult?
    var gate = "text: every teacher-forced row cosine >= 0.995 against the float32 reference, argmax equal unless a near-tie (reference top-1 minus top-2 probability < 0.1 and native token in the reference top-5); greedy identical to the reference (its final stop token excluded) or first diverging at such a near-tie. vision: Kit decodes and preprocesses the photo the reference was made from; greedy identical or first diverging at a near-tie of the reference's teacher-forced row there"
    var passed = false
    var failures: [String] = []
}

/// Opens the staged bundle with `platform: .ios`, runs the reference cases and reports parity, timings and memory.
func runLanguageDeviceProbe() async -> DeviceReport {
    var report = DeviceReport()
    let info = ProcessInfo.processInfo
    report.device = ["model": sysctlString("hw.machine"), "os": info.operatingSystemVersionString,
                     "memoryGiB": String(format: "%.1f", Double(info.physicalMemory) / 1_073_741_824)]
    report.availableMemoryMiBAtStart = availableMemory()
    report.availableMemoryMiBMinimum = report.availableMemoryMiBAtStart
    do {
        guard let assets = Bundle.main.url(forResource: "Assets", withExtension: nil) else {
            report.failures.append("no staged Assets folder in the app bundle")
            return report
        }
        let configuration = try JSONDecoder().decode(ProbeConfiguration.self, from: Data(contentsOf: assets.appendingPathComponent("probe.json")))
        report.mode = configuration.mode
        if configuration.mode == "prepare" {
            try await runPreparation(assets: assets, store: configuration.store ?? "Models",
                                     cutAfter: configuration.cutAfterBytes, report: &report)
            report.passed = report.failures.isEmpty && report.preparation?.reply != nil
            return report
        }
        if configuration.mode == "lifecycle" {
            report.lifecycle = try await runLifecycle(assets: assets, store: configuration.store ?? "Models",
                                                      idleSeconds: configuration.idleSeconds ?? 10)
            report.passed = report.failures.isEmpty && report.lifecycle?.releasedUnderPressure == true
            return report
        }
        if configuration.mode == "readiness" {
            try await runReadiness(assets: assets, store: configuration.store ?? "Models",
                                   memoryBytes: configuration.memoryBytes, report: &report)
            // Ready and answering, or, with a replaced memory requirement, refused for memory.
            let final = report.readiness?.final ?? ""
            report.passed = report.failures.isEmpty && (report.readiness?.reply != nil
                || (configuration.memoryBytes != nil && final.hasPrefix("unsupported memory")))
            return report
        }
        let bundle = assets.appendingPathComponent("bundle"), reference = assets.appendingPathComponent("reference")
        let manifest = try LanguageModelManifest.load(from: bundle)
        report.bundleModelSHA256 = manifest.modelSHA256
        switch configuration.mode {
        case "text": try await runText(bundle: bundle, reference: reference, report: &report)
        case "vision": try await runVision(bundle: bundle, reference: reference, names: configuration.cases, report: &report)
        default: report.failures.append("unknown mode \(configuration.mode)")
        }
        report.footprintAfterUnloadMiB = footprint().current
        report.passed = report.failures.isEmpty && !report.cases.isEmpty && report.cases.allSatisfy(\.passed)
        if !report.passed && report.failures.isEmpty { report.failures.append("a case did not meet the gate") }
    } catch {
        report.failures.append(String(describing: error))
    }
    return report
}

private func runText(bundle: URL, reference: URL, report: inout DeviceReport) async throws {
    let ref = try JSONDecoder().decode(TextReference.self, from: Data(contentsOf: reference.appendingPathComponent("reference.json")))
    let forced = try floats(reference.appendingPathComponent(ref.forcedLogits.file))
    let started = ContinuousClock.now
    let session = try await LanguageGenerationSession.open(bundle: bundle, platform: .ios)
    report.openSeconds = (ContinuousClock.now - started).seconds
    progress("session opened")
    let vocab = ref.forcedLogits.vocab
    let rows = try await session.teacherForcedLogits(prompt: ref.promptTokens, continuation: Array(ref.forcedTokens.dropLast()))
    let compared = rows.enumerated().map { i, row in compareRow(row, forced[(i * vocab)..<((i + 1) * vocab)]) }
    progress("teacher forced")
    noteMemory(&report)
    var referenceGreedy = ref.greedyTokens
    while let last = referenceGreedy.last, [151645, 151643].contains(last) { referenceGreedy.removeLast() }
    let reply = try await session.generate(prompt: ref.promptTokens, maximumTokens: ref.greedyTokens.count)
    progress("generated")
    noteMemory(&report)
    let (divergence, margin, accepted) = judgeGreedy(reply.tokens, referenceGreedy, ref.forcedTokens, ref.greedyTokens, forced, vocab, ref.forcedLogits.rows)
    let mismatches = compared.filter { !$0.1 }
    let nearTies = mismatches.filter { $0.2 < 0.1 && $0.3 < 5 }
    let minimum = compared.map(\.0).min() ?? 0
    report.cases.append(CaseResult(name: "text", promptTokens: ref.promptTokens.count, forcedRows: rows.count,
        forcedMinimumCosine: minimum, forcedArgmaxMismatches: mismatches.count, forcedMismatchesAtNearTies: nearTies.count,
        greedyTokens: reply.tokens, referenceGreedyTokens: ref.greedyTokens, greedyFirstDivergence: divergence,
        greedyDivergenceReferenceMargin: margin, greedyAccepted: accepted, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds,
        totalSeconds: reply.totalSeconds, footprintPeakMiB: footprint().peak,
        passed: minimum >= 0.995 && mismatches.count == nearTies.count && accepted))
    await session.unload()
    progress("unloaded")
}

private func runVision(bundle: URL, reference: URL, names: [String]?, report: inout DeviceReport) async throws {
    let ref = try JSONDecoder().decode(VisionReference.self, from: Data(contentsOf: reference.appendingPathComponent("reference.json")))
    let selected = names.map { n in ref.cases.filter { n.contains($0.name) } } ?? Array(ref.cases.prefix(1))
    let started = ContinuousClock.now
    let session = try await VisionLanguageSession.open(bundle: bundle, platform: .ios)
    report.openSeconds = (ContinuousClock.now - started).seconds
    progress("session opened")
    let imagePad = 151655
    for c in selected {
        let forced = try floats(reference.appendingPathComponent(c.forcedLogits.file))
        let image = try VisionRGBImage(contentsOf: reference.appendingPathComponent(c.image.file))
        // The reference's prompt has each image's pad token expanded; the session expands it again from the image.
        var rendered: [Int] = []
        for id in c.promptTokens where !(id == imagePad && rendered.last == imagePad) { rendered.append(id) }
        var referenceGreedy = c.greedyTokens
        while let last = referenceGreedy.last, [151645, 151643].contains(last) { referenceGreedy.removeLast() }
        let reply = try await session.generate(renderedTokenIDs: rendered, images: [image], maximumTokens: c.greedyTokens.count)
        progress("generated \(c.name)")
        noteMemory(&report)
        let (divergence, margin, accepted) = judgeGreedy(reply.tokens, referenceGreedy, c.forcedTokens, c.greedyTokens,
                                                         forced, c.forcedLogits.vocab, c.forcedLogits.rows)
        report.cases.append(CaseResult(name: c.name, promptTokens: rendered.count, forcedRows: nil, forcedMinimumCosine: nil,
            forcedArgmaxMismatches: nil, forcedMismatchesAtNearTies: nil, greedyTokens: reply.tokens,
            referenceGreedyTokens: c.greedyTokens, greedyFirstDivergence: divergence, greedyDivergenceReferenceMargin: margin,
            greedyAccepted: accepted, timeToFirstTokenSeconds: reply.timeToFirstTokenSeconds, totalSeconds: reply.totalSeconds,
            footprintPeakMiB: footprint().peak, passed: accepted))
    }
    await session.unload()
    progress("unloaded")
}

/// Greedy agreement: identical up to the reference end, or first diverging where the reference's own teacher-forced row
/// (same prefix on both sides) is a near-tie that ranks the native token in its top 5.
private func judgeGreedy(_ native: [Int], _ referenceGreedy: [Int], _ forcedTokens: [Int], _ greedyTokens: [Int],
                         _ forced: [Float], _ vocab: Int, _ rows: Int) -> (Int?, Double?, Bool) {
    let divergence = zip(native, referenceGreedy).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        ?? (native.count != referenceGreedy.count ? min(native.count, referenceGreedy.count) : nil)
    guard let d = divergence else { return (nil, nil, true) }
    guard d < rows, d < native.count, Array(forcedTokens.prefix(d)) == Array(greedyTokens.prefix(d)) else { return (d, nil, false) }
    var onehot = [Float](repeating: 0, count: vocab)
    onehot[native[d]] = 1
    let row = compareRow(onehot, forced[(d * vocab)..<((d + 1) * vocab)])
    return (d, row.2, row.2 < 0.1 && row.3 < 5)
}

/// (cosine, argmax equal, reference top-1 minus top-2 probability, native token's rank in the reference)
private func compareRow(_ native: [Float], _ slice: ArraySlice<Float>) -> (Double, Bool, Double, Int) {
    let reference = Array(slice)
    var dot = 0.0, nn = 0.0, rr = 0.0
    for (a, b) in zip(native, reference) { dot += Double(a) * Double(b); nn += Double(a) * Double(a); rr += Double(b) * Double(b) }
    let nativeBest = native.indices.max { native[$0] < native[$1] }!
    let referenceBest = reference.indices.max { reference[$0] < reference[$1] }!
    let sorted = reference.map(Double.init).sorted(by: >)
    let denominator = reference.reduce(0.0) { $0 + exp(Double($1) - sorted[0]) }
    let margin = (1 - exp(sorted[1] - sorted[0])) / denominator
    let value = Double(reference[nativeBest])
    return (dot / (nn.squareRoot() * rr.squareRoot()), nativeBest == referenceBest, margin, reference.filter { Double($0) > value }.count)
}

private func floats(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    return data.withUnsafeBytes { bytes in
        stride(from: 0, to: data.count, by: 4).map { Float(bitPattern: bytes.loadUnaligned(fromByteOffset: $0, as: UInt32.self).littleEndian) }
    }
}

private func noteMemory(_ report: inout DeviceReport) {
    report.footprintPeakMiB = max(report.footprintPeakMiB, footprint().peak)
    report.availableMemoryMiBMinimum = min(report.availableMemoryMiBMinimum, availableMemory())
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
    print(String(format: "LANGUAGE_DEVICE_PROGRESS %@ footprint=%.0fMiB peak=%.0fMiB available=%.0fMiB", stage, memory.current, memory.peak, availableMemory()))
    fflush(stdout)
}

extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
