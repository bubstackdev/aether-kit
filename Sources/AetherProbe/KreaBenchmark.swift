import AetherKit
import AetherKrea
import AetherKreaCoreAI
import AetherKreaTokenizers
import CoreAI
import CryptoKit
import Darwin
import Foundation
import Metal
import Synchronization

enum BenchmarkProfile: String, Codable, Sendable { case encoded, provided }

struct KreaBenchmarkPlan: Codable, Sendable {
    let schema: Int
    let bundleFingerprint: String
    let warmupIterations: Int
    let measuredIterations: Int
    let cases: [BenchmarkCase]
    // Missing in the original plan; preserve its encoded protocol.
    var profile: BenchmarkProfile? = nil
    var effectiveProfile: BenchmarkProfile { profile ?? .encoded }

    func validate() throws {
        guard schema == 1, benchmarkIsHash(bundleFingerprint), (1...10).contains(warmupIterations),
              (3...100).contains(measuredIterations), (1...16).contains(cases.count),
              Set(cases.map(\.id)).count == cases.count else {
            throw AetherError.invalidRequest("benchmark needs schema 1, unique cases, 1...10 warmups and 3...100 measurements")
        }
        for item in cases {
            guard !item.id.isEmpty, !item.prompt.isEmpty, item.steps == 8, benchmarkIsHash(item.referenceSHA256) else {
                throw AetherError.invalidRequest("benchmark cases need IDs, prompts, eight steps and pinned reference hashes")
            }
            _ = try item.configuration
        }
    }
}

struct BenchmarkCase: Codable, Sendable {
    let id: String
    let prompt: String
    let width: Int
    let height: Int
    let steps: Int
    let seed: UInt64
    let reference: String
    let referenceSHA256: String
    var configuration: KreaConfiguration { get throws {
        try KreaConfiguration(width: width, height: height, steps: steps, seed: seed)
    } }
}

struct BenchmarkExpected: Sendable {
    let conditioning: String
    let steps: [String]
    let pixels: String
    let shape: [Int]

    static func load(_ data: Data, item: BenchmarkCase, noiseSHA256: String) throws -> Self {
        struct Reference: Decodable {
            let `case`: String
            let prompt: String
            let width: Int
            let height: Int
            let steps: Int
            let arm: String
            let residency: String
            let noiseSHA256: String
            let runs: [Run]
            struct Run: Decodable {
                let encodedTextSHA256: String
                let stepLatentsSHA256: [String]
                let pixelsSHA256: String
                let pixelShape: [Int]
            }
        }
        let ref = try JSONDecoder().decode(Reference.self, from: data)
        guard ref.case == item.id, ref.prompt == item.prompt, ref.width == item.width, ref.height == item.height,
              ref.steps == 8, ref.arm == "device-encoded", ref.residency == "balanced",
              ref.noiseSHA256 == noiseSHA256, let run = ref.runs.first,
              run.stepLatentsSHA256.count == 8, run.stepLatentsSHA256.allSatisfy(benchmarkIsHash),
              benchmarkIsHash(run.encodedTextSHA256), benchmarkIsHash(run.pixelsSHA256),
              run.pixelShape == [3, item.height, item.width] else {
            throw AetherError.invalidRequest("benchmark reference does not match workload \(item.id)")
        }
        return Self(conditioning: run.encodedTextSHA256, steps: run.stepLatentsSHA256,
                    pixels: run.pixelsSHA256, shape: run.pixelShape)
    }
}

struct BenchmarkMemory: Codable, Sendable {
    var physicalFootprintBytes: UInt64
    var residentBytes: UInt64
    var metalAllocatedBytes: UInt64?
    var processLifetimePeakFootprintBytes: UInt64
    var processLifetimePeakResidentBytes: UInt64

    static func read(device: (any MTLDevice)?) throws -> Self {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { throw AetherError.unsupported("task memory counters unavailable: \(result)") }
        return Self(physicalFootprintBytes: info.phys_footprint, residentBytes: info.resident_size,
                    metalAllocatedBytes: device.map { UInt64($0.currentAllocatedSize) },
                    processLifetimePeakFootprintBytes: UInt64(max(0, info.ledger_phys_footprint_peak)),
                    processLifetimePeakResidentBytes: info.resident_size_peak)
    }

    mutating func merge(_ other: Self) {
        physicalFootprintBytes = max(physicalFootprintBytes, other.physicalFootprintBytes)
        residentBytes = max(residentBytes, other.residentBytes)
        if let value = other.metalAllocatedBytes { metalAllocatedBytes = max(metalAllocatedBytes ?? 0, value) }
        processLifetimePeakFootprintBytes = max(processLifetimePeakFootprintBytes, other.processLifetimePeakFootprintBytes)
        processLifetimePeakResidentBytes = max(processLifetimePeakResidentBytes, other.processLifetimePeakResidentBytes)
    }
}

final class BenchmarkMemorySampler: Sendable {
    struct State: Sendable {
        var samples = 0
        var peak: BenchmarkMemory?
        var error: String?
    }
    private let state = Mutex(State())
    private let read: @Sendable () throws -> BenchmarkMemory
    init(read: @escaping @Sendable () throws -> BenchmarkMemory) { self.read = read }
    var snapshot: State { state.withLock { $0 } }
    func sample() {
        do {
            let memory = try read()
            state.withLock {
                $0.samples += 1
                if $0.peak == nil { $0.peak = memory } else { $0.peak?.merge(memory) }
            }
        } catch { state.withLock { $0.error = String(describing: error) } }
    }
}

struct BenchmarkIteration: Codable, Sendable {
    let caseID: String
    let role: String
    let iteration: Int
    var profile: BenchmarkProfile? = nil
    var passed = false
    var error: String?
    var totalSeconds: Double?
    var observerSeconds: Double = 0
    var stages: [String: Double] = [:]
    var conditioningSHA256: String?
    var stepLatentsSHA256: [String] = []
    var pixelsSHA256: String?
    var selectionIdentity: String?
    var memorySamples = 0
    var sampledPeakMemory: BenchmarkMemory?
}

// Stage clocks stop on callback entry and resume after validation, preventing
// reference hashing from being charged to the following pipeline stage.
final class BenchmarkIterationJournal: Sendable {
    struct State: Sendable {
        var row: BenchmarkIteration
        var anchor: ContinuousClock.Instant
        var phase = "conditioning"
        var nextStep = 0
        var textRows = 0
    }
    private let state: Mutex<State>
    private let expected: BenchmarkExpected
    private let memory: BenchmarkMemorySampler
    init(caseID: String, role: String, iteration: Int, expected: BenchmarkExpected,
         memory: BenchmarkMemorySampler, started: ContinuousClock.Instant, profile: BenchmarkProfile = .encoded) {
        self.expected = expected
        self.memory = memory
        var row = BenchmarkIteration(caseID: caseID, role: role, iteration: iteration)
        row.profile = profile
        state = Mutex(State(row: row, anchor: started))
    }
    var row: BenchmarkIteration { state.withLock { $0.row } }

    func accept(_ event: KreaGenerationEvent) throws {
        let entered = ContinuousClock.now
        try state.withLock { value in
            defer {
                let ended = ContinuousClock.now
                value.row.observerSeconds += AetherProbe.seconds(entered.duration(to: ended))
                value.anchor = ended
            }
            let elapsed = AetherProbe.seconds(value.anchor.duration(to: entered))
            switch event {
            case .conditioningReady(let conditioning):
                guard value.phase == "conditioning" else { throw AetherError.invalidTensor("benchmark conditioning order") }
                value.row.stages["conditioning"] = elapsed
                let hash = floatHash(conditioning.tensor.values.map(Float.init))
                value.row.conditioningSHA256 = hash
                guard hash == expected.conditioning else { throw AetherError.invalidTensor("benchmark conditioning hash mismatch") }
                value.textRows = conditioning.rows
                value.phase = "denoiserSetup"
            case .denoisersReady(let selection):
                guard value.phase == "denoiserSetup", !selection.specialized,
                      selection.imageTokens == (expected.shape[1] / 16) * (expected.shape[2] / 16),
                      selection.textRows == value.textRows else {
                    throw AetherError.invalidTensor("benchmark requires ordered dynamic denoisers")
                }
                value.row.stages["denoiserSetup"] = elapsed
                value.row.selectionIdentity = selection.identity
                value.phase = "step"
            case .step(let index, let total, let latents):
                guard value.phase == "step", index == value.nextStep, total == 8, index < expected.steps.count,
                      latents.shape == [(expected.shape[1] / 16) * (expected.shape[2] / 16), 64] else {
                    throw AetherError.invalidTensor("benchmark sampling order")
                }
                value.row.stages["step\(index)"] = elapsed
                let hash = floatHash(latents.values)
                value.row.stepLatentsSHA256.append(hash)
                guard hash == expected.steps[index] else { throw AetherError.invalidTensor("benchmark latent hash mismatch at step \(index)") }
                value.nextStep += 1
                if value.nextStep == 8 { value.phase = "decode" }
            case .decoded:
                guard value.phase == "decode", value.nextStep == 8 else { throw AetherError.invalidTensor("benchmark decode order") }
                value.row.stages["decode"] = elapsed
                value.phase = "completion"
            }
            memory.sample()
        }
    }

    func finish(_ image: GeneratedImage, ended: ContinuousClock.Instant, total: Double) throws {
        try state.withLock { value in
            value.row.totalSeconds = total
            guard value.phase == "completion", image.pixelsCHW.shape == expected.shape else {
                throw AetherError.invalidTensor("benchmark generation incomplete")
            }
            value.row.stages["completion"] = AetherProbe.seconds(value.anchor.duration(to: ended))
            let hash = floatHash(image.pixelsCHW.values)
            value.row.pixelsSHA256 = hash
            guard hash == expected.pixels else { throw AetherError.invalidTensor("benchmark pixel hash mismatch") }
            let sample = memory.snapshot
            value.row.memorySamples = sample.samples
            value.row.sampledPeakMemory = sample.peak
            guard sample.samples > 0, sample.error == nil else {
                throw AetherError.unsupported("benchmark memory sampling failed: \(sample.error ?? "no samples")")
            }
            value.row.passed = true
        }
    }

    func fail(_ error: any Error, total: Double) -> BenchmarkIteration {
        state.withLock { value in
            value.row.passed = false
            value.row.error = String(describing: error)
            value.row.totalSeconds = total
            let sample = memory.snapshot
            value.row.memorySamples = sample.samples
            value.row.sampledPeakMemory = sample.peak
            return value.row
        }
    }
}

struct BenchmarkConditioner: Sendable {
    let encode: @Sendable () async throws -> KreaConditioning
    let unload: @Sendable () async -> Void
}

struct BenchmarkConditioningSetup: Codable, Sendable {
    let caseID: String
    let secondsIncludingLoadHashAndUnload: Double
    let conditioningSHA256: String
    let excludedFromIterationTimings: Bool
}

enum BenchmarkConditioningPreparation {
    static func prepare(caseID: String, expected: String,
                        makeEncoder: @Sendable () async throws -> BenchmarkConditioner) async throws
        -> (conditioning: KreaConditioning, record: BenchmarkConditioningSetup) {
        try Task.checkCancellation()
        let started = ContinuousClock.now
        let encoder = try await makeEncoder()
        do {
            let conditioning = try await encoder.encode()
            let hash = floatHash(conditioning.tensor.values.map(Float.init))
            guard hash == expected else { throw AetherError.invalidTensor("benchmark setup conditioning hash mismatch") }
            try Task.checkCancellation()
            // Release encoder ownership before acquiring the denoiser set.
            await encoder.unload()
            return (conditioning, BenchmarkConditioningSetup(caseID: caseID,
                secondsIncludingLoadHashAndUnload: AetherProbe.seconds(started.duration(to: .now)),
                conditioningSHA256: hash, excludedFromIterationTimings: true))
        } catch {
            await encoder.unload()
            throw error
        }
    }
}

struct BenchmarkOperation: Sendable {
    let generate: @Sendable (@escaping @Sendable (KreaGenerationEvent) throws -> Void) async throws -> GeneratedImage
    let unload: @Sendable () async -> Void
}

enum BenchmarkLoop {
    static func run(item: BenchmarkCase, expected: BenchmarkExpected, warmups: Int, measurements: Int,
                    profile: BenchmarkProfile = .encoded, operation: BenchmarkOperation, readMemory: @escaping @Sendable () throws -> BenchmarkMemory,
                    onStart: (String, Int) throws -> Void,
                    onIteration: (BenchmarkIteration) throws -> Void) async throws {
        do {
            for index in 0..<(warmups + measurements) {
                try Task.checkCancellation()
                let role = index < warmups ? "warmup" : "measured"
                let ordinal = index < warmups ? index : index - warmups
                try onStart(role, ordinal)
                let memory = BenchmarkMemorySampler(read: readMemory)
                memory.sample()
                let sampler = Task.detached(priority: .utility) {
                    while !Task.isCancelled {
                        do { try await Task.sleep(for: .milliseconds(50)) } catch { break }
                        memory.sample()
                    }
                }
                let started = ContinuousClock.now
                let journal = BenchmarkIterationJournal(caseID: item.id, role: role, iteration: ordinal,
                                                        expected: expected, memory: memory, started: started, profile: profile)
                do {
                    let image = try await operation.generate(journal.accept)
                    let ended = ContinuousClock.now
                    memory.sample()
                    sampler.cancel()
                    await sampler.value
                    try journal.finish(image, ended: ended, total: AetherProbe.seconds(started.duration(to: ended)))
                } catch {
                    sampler.cancel()
                    await sampler.value
                    let row = journal.fail(error, total: AetherProbe.seconds(started.duration(to: .now)))
                    try onIteration(row)
                    throw error
                }
                try onIteration(journal.row)
            }
            await operation.unload()
        } catch {
            await operation.unload()
            throw error
        }
    }
}

struct BenchmarkStatistics: Codable, Sendable {
    let count: Int
    let minimum: Double
    let median: Double
    let mean: Double
    let maximum: Double
    static func summarize(_ values: [Double]) -> Self? {
        guard !values.isEmpty, values.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
        return Self(count: values.count, minimum: sorted[0], median: median,
                    mean: sorted.reduce(0, +) / Double(sorted.count), maximum: sorted[sorted.count - 1])
    }
    static func measured(_ rows: [BenchmarkIteration], caseID: String) -> [String: Self] {
        let rows = rows.filter { $0.caseID == caseID && $0.role == "measured" && $0.passed }
        guard !rows.isEmpty, Set(rows.map(\.profile)).count == 1 else { return [:] }
        var metrics: [String: [Double]] = [:]
        for row in rows {
            if let total = row.totalSeconds { metrics["totalIncludingObservers", default: []].append(total) }
            metrics["observer", default: []].append(row.observerSeconds)
            for (name, duration) in row.stages { metrics[name, default: []].append(duration) }
            metrics["pipelineExcludingObservers", default: []].append(row.stages.values.reduce(0, +))
            metrics["sampling", default: []].append((0..<8).reduce(0) { $0 + (row.stages["step\($1)"] ?? 0) })
        }
        return metrics.compactMapValues(summarize)
    }
}

func benchmarkIsHash(_ value: String) -> Bool {
    value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}
func benchmarkHash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
func benchmarkFileHash(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let chunk = try file.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
