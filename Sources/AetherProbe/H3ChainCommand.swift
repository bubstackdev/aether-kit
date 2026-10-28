import AetherKit
import AetherH3
import AetherH3CoreAI
import Darwin
import Foundation
import Metal
import Synchronization

@available(macOS 27, iOS 27, *)
private struct H3ChainLimits: Encodable {
    var captureNativeSubmissions = 6
    var qualificationNativeSubmissions = 38
    var qualificationCyclesPerCase = 3
    let physicalFootprintBytes = H3LifecycleLimits().physicalFootprintBytes
    let metalAllocatedBytes = H3LifecycleLimits().metalAllocatedBytes
    let settledGrowthBytes = H3LifecycleLimits().settledGrowthBytes
    let settleSeconds = H3LifecycleLimits().settleSeconds
    func check(_ value: BenchmarkMemory) throws { try H3LifecycleLimits().check(value) }
    func checkGrowth(baseline: BenchmarkMemory, current: BenchmarkMemory) throws {
        try H3LifecycleLimits().checkGrowth(baseline: baseline, current: current)
    }
}

@available(macOS 27, iOS 27, *)
final class H3ChainTrace: Sendable {
    private struct State {
        var events: [String] = []
        var action: (@Sendable () -> Void)?
    }
    private let state = Mutex(State())
    private let target: Int
    init(target: Int = 1) { self.target = target }
    var events: [String] { state.withLock { $0.events } }
    func record(_ event: H3ChainEvent) {
        let action = state.withLock { value -> (@Sendable () -> Void)? in
            value.events.append(event.block.map { "b\($0)." + event.event.rawValue } ?? event.event.rawValue)
            guard event.block == target, event.event == .nativeSubmitted, let action = value.action else { return nil }
            value.action = nil; value.events.append("cancellationRequested"); return action
        }
        action?()
    }
    func arm(_ action: @escaping @Sendable () -> Void) { state.withLock { $0.events = []; $0.action = action } }
    func append(_ event: String) { state.withLock { $0.events.append(event) } }
    static func validate(_ events: [String], closing: Bool, target: Int = 1) throws {
        let expected = (0...target).flatMap { ["b\($0).nativeSubmitted", "b\($0).nativeCompleted"] }
        guard events.filter({ $0.hasPrefix("b") }) == expected else {
            throw AetherError.invalidModel("H3 chain cancelled before the intended boundary or submitted a later block")
        }
        let prefix = "b\(target)."
        let selected = events.filter { !$0.hasPrefix("b") || $0.hasPrefix(prefix) }.map { $0.replacingOccurrences(of: prefix, with: "") }
        try H3LifecycleTrace.validate(selected, closing: closing)
    }
}

@available(macOS 27, iOS 27, *)
private struct H3ChainOutput: Encodable, Sendable {
    let block: Int
    let name = "hidden_out"
    let dtype = "bfloat16"
    let shape: [Int]
    let path: String?
    let sha256: String
    let inputSHA256: String
    let comparison: H3ReferenceComparison?
}
@available(macOS 27, iOS 27, *)
private struct H3ChainRun: Encodable {
    let name: String
    let cycle: Int
    let role: String
    let outputs: [H3ChainOutput]
}
@available(macOS 27, iOS 27, *)
private struct H3ChainCancellation: Encodable {
    let name: String
    let cycle: Int
    let concurrentUnload: Bool
    let events: [String]
    let passed: Bool
}
@available(macOS 27, iOS 27, *)
private struct H3ChainReport: Encodable {
    let schema = 1
    var scope = "Three corrected production blocks at 19333 rows and both captured T2VA cardinalities; independent per-block CPU checks at captured native contexts, not independent CPU chain/trajectory parity or full generation"
    var limits = H3ChainLimits()
    let mode: String
    let chainManifestPath: String
    var bundleSHA256: String?
    var phaseManifestSHA256: String?
    var chainManifestSHA256: String?
    var passed = false
    var captureComplete = false
    var readyForQualification = false
    var nativeExecuted = false
    var selectedFilesVerified = false
    var stage = "preparation"
    var boundBufferBytes: [String: Int] = [:]
    var numericalCheckpointBlocks: [Int] = [0,1,2]
    var streamedWeights = false
    var runs: [H3ChainRun] = []
    var cancellations: [H3ChainCancellation] = []
    var resources: [String: BenchmarkMemory] = [:]
    var sampledPeak: BenchmarkMemory?
    var memorySamples = 0
    var memorySamplingError: String?
    var error: String?
}

/// Holds only metadata. The native backend waits for each callback and does not
/// accumulate fifty CPU snapshots while this consumer hashes/checks/saves one.
@available(macOS 27, iOS 27, *)
private actor H3ChainOutputRecorder {
    let item: H3ChainProbePlan.Case
    let directory: URL
    let capture: Bool
    let name: String
    let cycle: Int
    let role: String
    let knownBlock0: String?
    let limit: Double
    let sampler: BenchmarkMemorySampler
    private(set) var outputs: [H3ChainOutput] = []
    init(item: H3ChainProbePlan.Case, directory: URL, capture: Bool, name: String,
         cycle: Int, role: String, knownBlock0: String?, limit: Double, sampler: BenchmarkMemorySampler) {
        self.item = item; self.directory = directory; self.capture = capture; self.name = name
        self.cycle = cycle; self.role = role; self.knownBlock0 = knownBlock0; self.limit = limit; self.sampler = sampler
    }
    func accept(block: Int, value: H3TensorData) throws {
        try Task.checkCancellation()
        guard block == outputs.count, block < item.steps.count, value.spec == item.base.output else {
            throw AetherError.invalidTensor("H3 output consumer saw an out-of-order or incompatible block")
        }
        let step = item.steps[block]
        let comparison = try step.reference.map { try H3ReferenceComparison(actual: value,
            reference: $0.read(), rows: item.base.referenceRows, limit: limit) }
        let hash = value.sha256
        let contextBlocks: Set<Int> = item.streamsWeights ? [0,1,2,23,24,48,49] : [0,1,2]
        let path = capture && contextBlocks.contains(block) ? directory.appendingPathComponent("\(name)-block\(block).raw") : nil
        if let path { try value.bytes.write(to: path, options: .atomic) }
        let inputHash = outputs.last?.sha256 ?? item.base.inputs.first { $0.name == "hidden" }!.sha256
        outputs.append(H3ChainOutput(block: block, shape: value.spec.shape, path: path?.path,
            sha256: hash, inputSHA256: inputHash, comparison: comparison))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(H3ChainRun(name: name, cycle: cycle, role: role, outputs: outputs))
            .write(to: directory.appendingPathComponent("progress.json"), options: .atomic)
        guard comparison?.passed != false,
              capture || !item.referenceBlocks.contains(block) || comparison != nil,
              step.expectedOutputSHA256 == nil || step.expectedOutputSHA256 == hash,
              block != 0 || knownBlock0 == hash else {
            throw AetherError.invalidTensor("H3 block failed a CPU checkpoint or pinned output hash")
        }
        try H3LifecycleLimits().check(BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice()))
        let memory = sampler.snapshot
        if let error = memory.error { throw AetherError.unsupported(error) }
        if let peak = memory.peak { try H3LifecycleLimits().check(peak) }
    }
}

@available(macOS 27, iOS 27, *)
enum H3ChainCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-h3-chain" else { return false }
        guard arguments.count == 6, ["--prepare-only", "--capture", "--run"].contains(arguments[5]) else {
            throw AetherError.invalidRequest("use qualify-h3-chain BUNDLE.json PHASES.json CHAIN.json NEW_OUTPUT_DIRECTORY --prepare-only|--capture|--run")
        }
        let output = URL(fileURLWithPath: arguments[4], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else { throw AetherError.invalidRequest("H3 chain output must be a new directory") }
        let chainURL = URL(fileURLWithPath: arguments[3]).standardizedFileURL
        var report = H3ChainReport(mode: arguments[5], chainManifestPath: chainURL.path)
        var active: H3ChainSession?
        var sampler: BenchmarkMemorySampler?
        var sampling: Task<Void, Never>?
        func write() throws {
            if let state = sampler?.snapshot {
                report.sampledPeak = state.peak; report.memorySamples = state.samples; report.memorySamplingError = state.error
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        func sample(_ key: String) throws -> BenchmarkMemory {
            let value = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
            report.resources[key] = value; try write()
            try report.limits.check(value)
            if let state = sampler?.snapshot {
                if let error = state.error { throw AetherError.unsupported(error) }
                if let peak = state.peak { try report.limits.check(peak) }
            }
            return value
        }
        do {
            try write()
            let plan = try H3ChainProbePlan(bundle: URL(fileURLWithPath: arguments[1]),
                phases: URL(fileURLWithPath: arguments[2]), chain: chainURL, requireReferences: arguments[5] == "--run")
            let full = plan.cases.allSatisfy(\.streamsWeights)
            if full {
                report.streamedWeights = true; report.numericalCheckpointBlocks = [0,1,2,24,49]
                report.scope = "Fifty corrected production blocks at 19333 rows and two T2VA cardinalities; one resident weight set, every-block hashes/finite values and CPU numerical checkpoints 0,1,2,24,49. Not all-block CPU numerical parity or full generation."
                report.limits.captureNativeSubmissions = 100; report.limits.qualificationNativeSubmissions = 250
                report.limits.qualificationCyclesPerCase = 2
            }
            report.bundleSHA256 = plan.blockPlan.header.manifestSHA256
            report.phaseManifestSHA256 = plan.blockPlan.phaseManifestSHA256; report.chainManifestSHA256 = plan.manifestSHA256
            report.boundBufferBytes = Dictionary(uniqueKeysWithValues: plan.cases.map { ($0.base.name, $0.boundBytes) })
            report.readyForQualification = plan.cases.allSatisfy { item in item.steps.allSatisfy { $0.expectedOutputSHA256 != nil && (!item.referenceBlocks.contains($0.block) || $0.reference != nil) } }
            try plan.verify(); report.selectedFilesVerified = true
            try verifyManifests(arguments, plan)
            if arguments[5] == "--prepare-only" {
                report.stage = "ready"; report.passed = true
                try write(); try AetherProbe.printJSON(report); return true
            }
            let capture = arguments[5] == "--capture"
            let memorySampler = BenchmarkMemorySampler { try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice()) }
            sampler = memorySampler
            sampling = Task {
                while !Task.isCancelled {
                    memorySampler.sample()
                    do { try await Task.sleep(for: .milliseconds(20)) } catch { break }
                }
            }
            let block0Hashes = [
                "body-0": "47881fda1fd57203540abf9eb4bd8f8311ce30002444b7814a5c431a9b30d084",
                "body-1": "72d57bec5667eef4e53975bd50713a2442fdb8e9dd915174ebb6df0d37a53c27",
            ]
            var fullSettledBaseline: BenchmarkMemory?
            for item in plan.cases {
                var settledBaseline: BenchmarkMemory?
                var firstHashes: [String]?
                for cycle in 1...(capture ? 1 : (full && item.base.name == "body-0" ? 1 : report.limits.qualificationCyclesPerCase)) {
                    let key = "\(item.base.name).cycle\(cycle)"
                    report.stage = key + ".load"; try write(); _ = try sample(key + ".beforeLoad")
                    let target = full ? 24 : 1
                    let trace = H3ChainTrace(target: target)
                    report.nativeExecuted = true; try write()
                    let session = try await H3ChainSession.open(probe: item, executionObserver: { trace.record($0) })
                    active = session
                    _ = try sample(key + ".afterLoad")
                    func checkedRun(_ role: String) async throws {
                        report.stage = key + "." + role; try write()
                        let recorder = H3ChainOutputRecorder(item: item, directory: output, capture: capture,
                            name: item.base.name, cycle: cycle, role: role, knownBlock0: block0Hashes[item.base.name],
                            limit: plan.blockPlan.relativeL2Limit, sampler: memorySampler)
                        try await session.run(onOutput: { try await recorder.accept(block: $0, value: $1) })
                        let outputs = await recorder.outputs
                        guard outputs.count == item.steps.count else { throw AetherError.invalidTensor("incomplete H3 chain") }
                        report.runs.append(H3ChainRun(name: item.base.name, cycle: cycle, role: role, outputs: outputs)); try write()
                        let hashes = outputs.map(\.sha256)
                        guard firstHashes == nil || firstHashes == hashes else { throw AetherError.invalidTensor("H3 chain repeated/recovered outputs changed bytes") }
                        firstHashes = hashes
                        _ = try sample(key + "." + role)
                    }
                    if capture {
                        try await checkedRun("capture")
                    } else {
                        try await checkedRun(cycle == 1 ? "run1" : (cycle == 2 ? "reopenedRecovery" : "afterClosingRecovery"))
                        if !full && cycle == 1 { try await checkedRun("run2") }
                        if full ? cycle == 1 : cycle < 3 {
                            let closing = full ? item.base.name == "body-1" : cycle == 2
                            report.stage = key + ".cancellation"; try write()
                            let events = try await cancelledRun(session, trace: trace, closing: closing)
                            let valid = (try? H3ChainTrace.validate(events, closing: closing, target: target)) != nil
                            report.cancellations.append(H3ChainCancellation(name: item.base.name, cycle: cycle,
                                concurrentUnload: closing, events: events, passed: valid)); try write()
                            try H3ChainTrace.validate(events, closing: closing, target: target)
                            if !closing {
                                guard await session.state == .ready else { throw AetherError.invalidModel("cancelled H3 chain did not recover its slot") }
                                try await checkedRun("retainedRecovery")
                            }
                        }
                    }
                    await session.unload(); active = nil
                    guard await session.state == .closed else { throw AetherError.invalidModel("H3 chain did not close") }
                    var previous = 0
                    report.stage = key + ".settle"; try write()
                    for seconds in report.limits.settleSeconds {
                        if seconds > previous { try await Task.sleep(for: .seconds(seconds - previous)) }
                        let memory = try sample(key + ".afterUnload\(seconds)s")
                        if seconds == 5 {
                            if full {
                                if let baseline = fullSettledBaseline { try report.limits.checkGrowth(baseline: baseline, current: memory) }
                                else { fullSettledBaseline = memory }
                            } else if cycle == 1 { settledBaseline = memory }
                            else if let settledBaseline { try report.limits.checkGrowth(baseline: settledBaseline, current: memory) }
                        }
                        previous = seconds
                    }
                }
            }
            try verifyManifests(arguments, plan)
            sampling?.cancel(); await sampling?.value
            report.stage = capture ? "capturedAwaitingCPUReferences" : "complete"
            report.captureComplete = capture; report.passed = !capture
            try write(); try AetherProbe.printJSON(report)
        } catch {
            if let active { await active.unload() }
            sampling?.cancel(); await sampling?.value
            report.stage = "failed"; report.passed = false; report.error = String(describing: error)
            try write(); throw error
        }
        return true
    }
    private static func verifyManifests(_ arguments: [String], _ plan: H3ChainProbePlan) throws {
        let expected = [plan.blockPlan.header.manifestSHA256, plan.blockPlan.phaseManifestSHA256, plan.manifestSHA256]
        for (path, hash) in zip(arguments[1...3], expected) {
            guard try h3SHA256(Data(contentsOf: URL(fileURLWithPath: path))) == hash else { throw AetherError.invalidModel("H3 chain manifest changed") }
        }
    }
    private static func cancelledRun(_ session: H3ChainSession, trace: H3ChainTrace, closing: Bool) async throws -> [String] {
        let (start, continuation) = AsyncStream.makeStream(of: Void.self)
        let unload = Mutex<Task<Void, Never>?>(nil)
        let task = Task {
            for await _ in start { break }
            do { try await session.run(onOutput: { _, _ in }); trace.append("runSucceeded") }
            catch is CancellationError { trace.append("runCancelled") }
            catch { trace.append("runFailed"); throw error }
        }
        trace.arm {
            task.cancel()
            if closing { unload.withLock { $0 = Task { await session.unload() } } }
        }
        continuation.yield(()); continuation.finish()
        do { try await task.value }
        catch { let closingTask = unload.withLock { $0 }; await closingTask?.value; throw error }
        let closingTask = unload.withLock { $0 }; await closingTask?.value
        return trace.events
    }
}
