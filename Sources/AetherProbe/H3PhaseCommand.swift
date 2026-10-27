import AetherKit
import AetherH3
import AetherH3CoreAI
import Darwin
import Foundation
import Metal
import Synchronization

@available(macOS 27, iOS 27, *)
private struct PhaseRun: Encodable {
    let name: String
    let role: String
    let comparisons: [String: H3TensorComparison]
}
@available(macOS 27, iOS 27, *)
private struct PhaseCancellation: Encodable {
    let name: String
    let concurrentUnload: Bool
    let events: [String]
    let passed: Bool
}
@available(macOS 27, iOS 27, *)
private struct PhaseLimits: Encodable {
    let sessionOpens = 10
    let retainedCancellations = 5
    let concurrentUnloadCancellations = 5
    let physicalFootprintBytes: UInt64 = 20 << 30
    let metalAllocatedBytes: UInt64 = 18 << 30
    let settledGrowthBytes: UInt64 = 512 << 20
    let settleSeconds = [0, 1, 5]
    func check(_ memory: BenchmarkMemory) throws { try H3LifecycleLimits().check(memory) }
    func checkGrowth(baseline: BenchmarkMemory, current: BenchmarkMemory) throws {
        try H3LifecycleLimits().checkGrowth(baseline: baseline, current: current)
    }
}
@available(macOS 27, iOS 27, *)
private struct PhaseReport: Encodable {
    let schema = 1
    let scope = "Fixed prepare, time-0/1 and heads-0/1 boundaries with complete original CPU references, repeat/recovery hashes and submitted-work cancellation. Not a linked backbone, complete Turbo8 trajectory, conditioner, decoder or benchmark."
    let mode: String
    let nativeSubmissionBudget = 30
    let numericalComparisonBudget = 28
    let limits = PhaseLimits()
    var passed = false
    var stage = "preparation"
    var nativeExecuted = false
    var selectedFilesVerified = false
    var bundleSHA256: String?
    var phaseManifestSHA256: String?
    var cpuRecordSHA256: String?
    var boundBufferBytes: [String: Int] = [:]
    var resources: [String: BenchmarkMemory] = [:]
    var sampledPeak: BenchmarkMemory?
    var memorySamples = 0
    var memorySamplingError: String?
    var runs: [PhaseRun] = []
    var cancellations: [PhaseCancellation] = []
    var error: String?
}

@available(macOS 27, iOS 27, *)
enum H3PhaseCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-h3-phases" else { return false }
        guard arguments.count == 6, ["--prepare-only", "--run"].contains(arguments[5]) else {
            throw AetherError.invalidRequest("use qualify-h3-phases BUNDLE.json PHASES.json CPU_CHECKS.json NEW_OUTPUT_DIRECTORY --prepare-only|--run")
        }
        let output = URL(fileURLWithPath: arguments[4], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else { throw AetherError.invalidRequest("H3 phase output must be a new directory") }
        var report = PhaseReport(mode: arguments[5])
        var active: H3PhaseSession?
        var sampler: BenchmarkMemorySampler?
        var sampling: Task<Void, Never>?
        func write() throws {
            if let snapshot = sampler?.snapshot {
                report.sampledPeak = snapshot.peak; report.memorySamples = snapshot.samples
                report.memorySamplingError = snapshot.error
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        func sample(_ name: String) throws -> BenchmarkMemory {
            let memory = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
            report.resources[name] = memory; try write()
            try report.limits.check(memory)
            if let error = sampler?.snapshot.error { throw AetherError.unsupported(error) }
            if let peak = sampler?.snapshot.peak { try report.limits.check(peak) }
            return memory
        }
        func verifyManifests(_ plan: H3PhaseProbePlan) throws {
            for (index, expected) in [(1, plan.header.manifestSHA256), (2, plan.phaseManifestSHA256), (3, plan.cpuRecordSHA256)] {
                guard try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[index]))) == expected else {
                    throw AetherError.invalidModel("H3 outer manifests changed during preparation/qualification")
                }
            }
        }
        do {
            try write()
            let plan = try H3PhaseProbePlan(bundle: URL(fileURLWithPath: arguments[1]),
                phases: URL(fileURLWithPath: arguments[2]), cpuRecord: URL(fileURLWithPath: arguments[3]))
            report.bundleSHA256 = plan.header.manifestSHA256
            report.phaseManifestSHA256 = plan.phaseManifestSHA256; report.cpuRecordSHA256 = plan.cpuRecordSHA256
            report.boundBufferBytes = Dictionary(uniqueKeysWithValues: plan.cases.map { ($0.name, $0.boundBytes) })
            try write(); try plan.verify(); report.selectedFilesVerified = true
            try verifyManifests(plan)
            if arguments[5] == "--prepare-only" {
                report.stage = "ready"; report.passed = true
                try write(); try AetherProbe.printJSON(report); return true
            }
            let memorySampler = BenchmarkMemorySampler { try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice()) }
            sampler = memorySampler
            sampling = Task {
                while !Task.isCancelled {
                    memorySampler.sample()
                    do { try await Task.sleep(for: .milliseconds(20)) } catch { break }
                }
            }
            for item in plan.cases {
                let trace = H3LifecycleTrace()
                var firstHashes: [String: String]?
                var settledBaseline: BenchmarkMemory?
                for cycle in 1...2 {
                    let key = "\(item.name).cycle\(cycle)"
                    report.stage = key + ".load"; try write()
                    _ = try sample(key + ".beforeLoad")
                    report.nativeExecuted = true; try write()
                    let session = try await H3PhaseSession.open(probe: item, executionObserver: { trace.record($0) })
                    active = session; _ = try sample(key + ".afterLoad")
                    func checkedRun(_ role: String) async throws {
                        report.stage = key + "." + role; try write()
                        let outputs = try await session.run()
                        guard Set(outputs.keys) == Set(item.references.map(\.name)) else {
                            throw AetherError.invalidTensor("H3 outer outputs differ from planned names")
                        }
                        var comparisons: [String: H3TensorComparison] = [:]
                        for file in item.references {
                            let comparison = try H3TensorComparison(actual: outputs[file.name]!, reference: file.read(), limit: plan.relativeL2Limit)
                            comparisons[file.name] = comparison
                        }
                        report.runs.append(PhaseRun(name: item.name, role: key + "." + role, comparisons: comparisons))
                        try write()
                        let hashes = comparisons.mapValues(\.actualSHA256)
                        guard comparisons.values.allSatisfy(\.passed), firstHashes == nil || firstHashes == hashes else {
                            throw AetherError.invalidTensor("H3 outer CPU comparison or repeat/recovery hash failed")
                        }
                        firstHashes = hashes
                        _ = try sample(key + ".after." + role)
                    }
                    if cycle == 1 {
                        try await checkedRun("run1"); try await checkedRun("run2")
                        for closing in [false, true] {
                            report.stage = key + (closing ? ".cancelAndUnload" : ".cancelRetained"); try write()
                            let events = try await cancelledRun(session, trace: trace, closing: closing)
                            let passed = (try? H3LifecycleTrace.validate(events, closing: closing)) != nil
                            report.cancellations.append(PhaseCancellation(name: item.name, concurrentUnload: closing, events: events, passed: passed))
                            try write(); try H3LifecycleTrace.validate(events, closing: closing)
                            if !closing {
                                guard await session.state == .ready else { throw AetherError.invalidModel("H3 outer cancellation did not restore readiness") }
                                try await checkedRun("retainedRecovery")
                            }
                        }
                    } else { try await checkedRun("reopenedRecovery") }
                    await session.unload(); active = nil
                    guard await session.state == .closed else { throw AetherError.invalidModel("H3 outer session did not close") }
                    report.stage = key + ".settle"; try write()
                    var previous = 0
                    for seconds in report.limits.settleSeconds {
                        if seconds > previous { try await Task.sleep(for: .seconds(seconds - previous)) }
                        let memory = try sample(key + ".afterUnload\(seconds)s")
                        if seconds == report.limits.settleSeconds.last {
                            if let baseline = settledBaseline { try report.limits.checkGrowth(baseline: baseline, current: memory) }
                            else { settledBaseline = memory }
                        }
                        previous = seconds
                    }
                }
            }
            guard report.runs.count == 20, report.cancellations.count == 10,
                  report.runs.reduce(0, { $0 + $1.comparisons.count }) == report.numericalComparisonBudget else {
                throw AetherError.invalidModel("H3 outer qualification did not complete its fixed plan")
            }
            try verifyManifests(plan)
            sampling?.cancel(); await sampling?.value
            if let error = sampler?.snapshot.error { throw AetherError.unsupported(error) }
            if let peak = sampler?.snapshot.peak { try report.limits.check(peak) }
            report.stage = "complete"; report.passed = true
            try write(); try AetherProbe.printJSON(report)
        } catch {
            if let active { await active.unload() }
            sampling?.cancel(); await sampling?.value
            report.stage = "failed"; report.passed = false; report.error = String(describing: error)
            try write(); throw error
        }
        return true
    }

    private static func cancelledRun(_ session: H3PhaseSession, trace: H3LifecycleTrace,
                                     closing: Bool) async throws -> [String] {
        let (start, continuation) = AsyncStream.makeStream(of: Void.self)
        let unload = Mutex<Task<Void, Never>?>(nil)
        let task = Task {
            for await _ in start { break }
            do {
                _ = try await session.run()
                trace.append("runSucceeded")
            } catch is CancellationError { trace.append("runCancelled") }
            catch { trace.append("runFailed"); throw error }
        }
        trace.arm {
            task.cancel()
            if closing { unload.withLock { $0 = Task { await session.unload() } } }
        }
        continuation.yield(()); continuation.finish()
        do { try await task.value }
        catch {
            let closingTask = unload.withLock { $0 }; await closingTask?.value
            throw error
        }
        let closingTask = unload.withLock { $0 }; await closingTask?.value
        return trace.events
    }
}
