import AetherKit
import AetherH3CoreAI
import Foundation
import Synchronization
import Testing
@testable import AetherProbe

@Test func h3LifecycleRequiresExplicitModeAndKeepsPreparationFailuresNonNative() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    await #expect(throws: AetherError.self) {
        try await H3LifecycleCommand.run(["qualify-h3-lifecycle", "/missing", "/missing", root.path, "--unknown"])
    }
    #expect(!FileManager.default.fileExists(atPath: root.path))
    #expect(try await H3LifecycleCommand.run(["other"]) == false)
    await #expect(throws: (any Error).self) {
        try await H3LifecycleCommand.run(["qualify-h3-lifecycle", "/missing", "/missing", root.path, "--prepare-only"])
    }
    let data = try Data(contentsOf: root.appendingPathComponent("report.json"))
    let raw = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(raw["passed"] as? Bool == false)
    #expect(raw["nativeExecuted"] as? Bool == false)
    #expect(raw["stage"] as? String == "failed")
    await #expect(throws: AetherError.self) {
        try await H3LifecycleCommand.run(["qualify-h3-lifecycle", "/missing", "/missing", root.path, "--run"])
    }
    #expect(try Data(contentsOf: root.appendingPathComponent("report.json")) == data)
}

@Test func h3CancellationTraceRejectsMissedSubmissionEarlyReleaseAndLateUnload() throws {
    let valid = ["nativeSubmitted", "cancellationRequested", "unloadStarted", "nativeCompleted", "slotReleased", "unloaded", "runCancelled"]
    try H3LifecycleTrace.validate(valid, closing: true)
    try H3LifecycleTrace.validate(valid.filter { !["unloadStarted", "unloaded"].contains($0) }, closing: false)
    for invalid in [
        valid.filter { $0 != "nativeSubmitted" },
        ["nativeSubmitted", "nativeCompleted", "cancellationRequested", "slotReleased", "runCancelled"],
        ["nativeSubmitted", "cancellationRequested", "slotReleased", "nativeCompleted", "runCancelled"],
        ["nativeSubmitted", "cancellationRequested", "nativeCompleted", "unloadStarted", "slotReleased", "unloaded", "runCancelled"],
        valid + ["nativeSubmitted"],
        valid.filter { $0 != "runCancelled" } + ["runSucceeded"],
    ] {
        #expect(throws: AetherError.self) { try H3LifecycleTrace.validate(invalid, closing: true) }
    }
    let trace = H3LifecycleTrace()
    let calls = Mutex(0)
    trace.arm { calls.withLock { $0 += 1 } }
    trace.record(.nativeSubmitted); trace.record(.nativeCompleted); trace.record(.nativeSubmitted)
    #expect(calls.withLock { $0 } == 1)
    #expect(trace.events == ["nativeSubmitted", "cancellationRequested", "nativeCompleted", "nativeSubmitted"])
}

@Test func h3LifecycleBoundsRejectFootprintPeaksMetalGrowthAndResidentGrowth() throws {
    let limits = H3LifecycleLimits()
    let baseline = BenchmarkMemory(physicalFootprintBytes: 12 << 30, residentBytes: 2 << 30,
        metalAllocatedBytes: 1 << 30, processLifetimePeakFootprintBytes: 13 << 30,
        processLifetimePeakResidentBytes: 2 << 30)
    try limits.check(baseline)
    try limits.checkGrowth(baseline: baseline, current: baseline)
    var lower = baseline; lower.physicalFootprintBytes = 1
    try limits.checkGrowth(baseline: baseline, current: lower)
    for field in ["footprint", "peak", "metal", "missingMetal"] {
        var bad = baseline
        switch field {
        case "footprint": bad.physicalFootprintBytes = limits.physicalFootprintBytes + 1
        case "peak": bad.processLifetimePeakFootprintBytes = limits.physicalFootprintBytes + 1
        case "metal": bad.metalAllocatedBytes = limits.metalAllocatedBytes + 1
        default: bad.metalAllocatedBytes = nil
        }
        #expect(throws: AetherError.self) { try limits.check(bad) }
    }
    for field in ["footprint", "resident", "metal"] {
        var bad = baseline
        switch field {
        case "footprint": bad.physicalFootprintBytes += limits.settledGrowthBytes + 1
        case "resident": bad.residentBytes += limits.settledGrowthBytes + 1
        default: bad.metalAllocatedBytes! += limits.settledGrowthBytes + 1
        }
        #expect(throws: AetherError.self) { try limits.checkGrowth(baseline: baseline, current: bad) }
    }
}
