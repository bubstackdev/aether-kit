import AetherKit
import AetherH3
import AetherH3CoreAI
import Darwin
import Foundation
import Metal
import Synchronization

/// Fixed bounds are published before native acquisition. This is a short
/// stability check, not a claim that all native allocations have been freed.
@available(macOS 27, iOS 27, *)
struct H3LifecycleLimits: Encodable {
    let cycles = 4
    let runsPerSession = 2
    let physicalFootprintBytes: UInt64 = 20 << 30
    let metalAllocatedBytes: UInt64 = 18 << 30
    let settledGrowthBytes: UInt64 = 512 << 20
    let settleSeconds = [0, 1, 5]

    func check(_ memory: BenchmarkMemory) throws {
        guard memory.physicalFootprintBytes <= physicalFootprintBytes,
              memory.processLifetimePeakFootprintBytes <= physicalFootprintBytes,
              let metal = memory.metalAllocatedBytes, metal <= metalAllocatedBytes else {
            throw AetherError.unsupported("H3 lifecycle memory exceeded a predefined bound or Metal counters are unavailable")
        }
    }
    func checkGrowth(baseline: BenchmarkMemory, current: BenchmarkMemory) throws {
        func bounded(_ a: UInt64, _ b: UInt64) -> Bool { b <= a || b - a <= settledGrowthBytes }
        guard bounded(baseline.physicalFootprintBytes, current.physicalFootprintBytes),
              bounded(baseline.residentBytes, current.residentBytes),
              let a = baseline.metalAllocatedBytes, let b = current.metalAllocatedBytes, bounded(a, b) else {
            throw AetherError.unsupported("H3 settled memory grew by more than the predefined 512 MiB bound")
        }
    }
}

@available(macOS 27, iOS 27, *)
final class H3LifecycleTrace: Sendable {
    private struct State {
        var events: [String] = []
        var onSubmit: (@Sendable () -> Void)?
    }
    private let state = Mutex(State())
    var events: [String] { state.withLock { $0.events } }
    func record(_ event: H3BlockEvent) {
        let action = state.withLock { value -> (@Sendable () -> Void)? in
            value.events.append(event.rawValue)
            guard event == .nativeSubmitted, let action = value.onSubmit else { return nil }
            value.onSubmit = nil
            value.events.append("cancellationRequested")
            return action
        }
        action?()
    }
    func append(_ event: String) { state.withLock { $0.events.append(event) } }
    func arm(_ action: @escaping @Sendable () -> Void) {
        state.withLock { $0.events = []; $0.onSubmit = action }
    }
    static func validate(_ events: [String], closing: Bool) throws {
        func position(_ name: String) throws -> Int {
            guard events.filter({ $0 == name }).count == 1, let index = events.firstIndex(of: name) else {
                throw AetherError.invalidModel("H3 cancellation did not observe exactly one \(name): \(events)")
            }
            return index
        }
        let submitted = try position("nativeSubmitted"), cancelled = try position("cancellationRequested")
        let completed = try position("nativeCompleted"), released = try position("slotReleased")
        let returned = try position("runCancelled")
        guard submitted < cancelled, cancelled < completed, completed < released, released < returned else {
            throw AetherError.invalidModel("H3 cancellation released ownership before joined completion")
        }
        if closing {
            let start = try position("unloadStarted"), end = try position("unloaded")
            guard cancelled < start, start < completed, completed < end else {
                throw AetherError.invalidModel("H3 concurrent unload was not observed while native work was pending")
            }
        }
    }
}

@available(macOS 27, iOS 27, *)
private struct H3LifecycleRun: Encodable {
    let name: String
    let cycle: Int
    let role: String
    let fullOutputSHA256: String
    let comparison: H3ReferenceComparison
}
@available(macOS 27, iOS 27, *)
private struct H3LifecycleCancellation: Encodable {
    let name: String
    let cycle: Int
    let concurrentUnload: Bool
    let events: [String]
    let passed: Bool
}
@available(macOS 27, iOS 27, *)
private struct H3LifecycleReport: Encodable {
    let schema = 1
    let scope = "Corrected block-0 lifecycle at two T2VA cardinalities; submitted-work cancellation, retained/reopened recovery and bounded memory stability. Not full generation or a benchmark."
    let mode: String
    let limits = H3LifecycleLimits()
    var passed = false
    var stage = "preparation"
    var nativeExecuted = false
    var selectedFilesVerified = false
    var bundleSHA256: String?
    var phaseManifestSHA256: String?
    var expectedOutputSHA256: [String: String] = [:]
    var resources: [String: BenchmarkMemory] = [:]
    var sampledPeak: BenchmarkMemory?
    var memorySamples = 0
    var memorySamplingError: String?
    var runs: [H3LifecycleRun] = []
    var cancellations: [H3LifecycleCancellation] = []
    var error: String?
}

@available(macOS 27, iOS 27, *)
enum H3LifecycleCommand {
    static func run(_ arguments: [String]) async throws -> Bool {
        guard arguments.first == "qualify-h3-lifecycle" else { return false }
        guard arguments.count == 5, ["--prepare-only", "--run"].contains(arguments[4]) else {
            throw AetherError.invalidRequest("use qualify-h3-lifecycle BUNDLE.json PHASES.json NEW_OUTPUT_DIRECTORY --prepare-only|--run")
        }
        let output = URL(fileURLWithPath: arguments[3], isDirectory: true).standardizedFileURL
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard mkdir(output.path, 0o755) == 0 else { throw AetherError.invalidRequest("H3 lifecycle output must be a new directory") }
        var report = H3LifecycleReport(mode: arguments[4])
        // These are the complete-output hashes qualified before this experiment,
        // rather than baselines learned from a possibly broken recovery run.
        report.expectedOutputSHA256 = [
            "body-0": "47881fda1fd57203540abf9eb4bd8f8311ce30002444b7814a5c431a9b30d084",
            "body-1": "72d57bec5667eef4e53975bd50713a2442fdb8e9dd915174ebb6df0d37a53c27",
        ]
        var active: H3BlockSession?
        var sampler: BenchmarkMemorySampler?
        var sampling: Task<Void, Never>?
        func write() throws {
            if let state = sampler?.snapshot {
                report.sampledPeak = state.peak; report.memorySamples = state.samples
                report.memorySamplingError = state.error
            }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: output.appendingPathComponent("report.json"), options: .atomic)
        }
        func sample(_ key: String) throws -> BenchmarkMemory {
            let value = try BenchmarkMemory.read(device: MTLCreateSystemDefaultDevice())
            report.resources[key] = value
            try write() // Keep the violating sample if the next guard rejects it.
            try report.limits.check(value)
            if let error = sampler?.snapshot.error { throw AetherError.unsupported(error) }
            if let peak = sampler?.snapshot.peak { try report.limits.check(peak) }
            return value
        }
        do {
            try write()
            let plan = try H3BlockProbePlan(bundle: URL(fileURLWithPath: arguments[1]), phases: URL(fileURLWithPath: arguments[2]))
            report.bundleSHA256 = plan.header.manifestSHA256; report.phaseManifestSHA256 = plan.phaseManifestSHA256
            try plan.verify(); report.selectedFilesVerified = true
            try verifyManifests(arguments, plan: plan)
            if arguments[4] == "--prepare-only" {
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
            var settled: [String: BenchmarkMemory] = [:]
            for cycle in 1...report.limits.cycles {
                for item in plan.cases {
                    let key = "\(item.name).cycle\(cycle)"
                    report.stage = key + ".load"; try write()
                    _ = try sample(key + ".beforeLoad")
                    let trace = H3LifecycleTrace()
                    report.nativeExecuted = true; try write()
                    let session = try await H3BlockSession.open(probe: item, executionObserver: { trace.record($0) })
                    active = session
                    _ = try sample(key + ".afterLoad")
                    let reference = try item.reference.read()
                    func checkedRun(_ role: String) async throws {
                        report.stage = key + "." + role; try write()
                        let value = try await session.run()
                        let comparison = try H3ReferenceComparison(actual: value, reference: reference,
                            rows: item.referenceRows, limit: plan.relativeL2Limit)
                        report.runs.append(H3LifecycleRun(name: item.name, cycle: cycle, role: role,
                            fullOutputSHA256: value.sha256, comparison: comparison))
                        try write()
                        guard comparison.passed, value.sha256 == report.expectedOutputSHA256[item.name] else {
                            throw AetherError.invalidTensor("H3 lifecycle output failed the CPU oracle or previously qualified full-output hash")
                        }
                    }
                    for repetition in 1...report.limits.runsPerSession {
                        try await checkedRun("run\(repetition)")
                        _ = try sample(key + ".afterRun\(repetition)")
                    }
                    if cycle == 2 || cycle == 3 {
                        let closing = cycle == 3
                        report.stage = key + ".cancellation"; try write()
                        let events = try await cancelledRun(session, trace: trace, closing: closing)
                        let valid = (try? H3LifecycleTrace.validate(events, closing: closing)) != nil
                        report.cancellations.append(H3LifecycleCancellation(name: item.name, cycle: cycle,
                            concurrentUnload: closing, events: events, passed: valid))
                        try write()
                        try H3LifecycleTrace.validate(events, closing: closing)
                        if !closing {
                            guard await session.state == .ready else { throw AetherError.invalidModel("cancelled H3 session was not ready") }
                            try await checkedRun("retainedRecovery")
                        }
                    }
                    _ = try sample(key + ".beforeUnload")
                    await session.unload(); active = nil
                    guard await session.state == .closed else { throw AetherError.invalidModel("H3 session did not close") }
                    report.stage = key + ".settle"; try write()
                    var previous = 0
                    for seconds in report.limits.settleSeconds {
                        if seconds > previous { try await Task.sleep(for: .seconds(seconds - previous)) }
                        let memory = try sample(key + ".afterUnload\(seconds)s")
                        if seconds == report.limits.settleSeconds.last {
                            if cycle == 2 { settled[item.name] = memory }
                            if cycle > 2, let baseline = settled[item.name] {
                                try report.limits.checkGrowth(baseline: baseline, current: memory)
                            }
                        }
                        previous = seconds
                    }
                }
            }
            try verifyManifests(arguments, plan: plan)
            sampling?.cancel(); await sampling?.value
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

    private static func verifyManifests(_ arguments: [String], plan: H3BlockProbePlan) throws {
        guard try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[1]))) == plan.header.manifestSHA256,
              try h3SHA256(Data(contentsOf: URL(fileURLWithPath: arguments[2]))) == plan.phaseManifestSHA256 else {
            throw AetherError.invalidModel("H3 lifecycle manifests changed")
        }
    }

    private static func cancelledRun(_ session: H3BlockSession, trace: H3LifecycleTrace,
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
