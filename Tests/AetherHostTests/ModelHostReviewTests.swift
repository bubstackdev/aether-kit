import Foundation
import Synchronization
import Testing
@testable import AetherHost

/// Second pre-release review, items 4 and 10 (ModelHost).
@Suite("ModelHost review regressions")
struct ModelHostReviewTests {
    /// Item 4: a request whose model type didn't match pinned the model before the cast threw, so it could never
    /// unload (idle, pressure or requested).
    @Test func aTypeMismatchLeavesNoPin() async throws {
        let host = ModelHost(policy: .manual, monitor: MemoryPressureMonitor(source: .manual))
        let ref = testRef()
        try await host.withModel(ref, load: { 42 }) { (_: Int) in }
        await #expect(throws: AetherError.self) {
            try await host.withModel(ref, load: { "different type" }) { (_: String) in }
        }
        #expect(await host.residency().first?.pinned == false)
        await host.unload(ref)
        #expect(await host.residency().isEmpty)
    }

    /// Item 10: an already-cancelled caller still loaded the model and ran its body.
    @Test func aCancelledCallerLoadsAndRunsNothing() async throws {
        let host = ModelHost(policy: .manual, monitor: MemoryPressureMonitor(source: .manual))
        let ref = testRef()
        let loads = Mutex(0)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await host.withModel(ref, load: { loads.withLock { $0 += 1 }; return 42 }) { (_: Int) in "body ran" }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(loads.withLock { $0 } == 0)
    }

    /// Whether `task` ends within `seconds`, without cancelling it.
    static func ends<T: Sendable>(_ task: Task<T, any Error>, within seconds: Double) async -> Bool {
        let done = Locked(false)
        let watcher = Task { _ = try? await task.value; done.value = true }
        let deadline = ContinuousClock.now + .seconds(seconds)
        while !done.value, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(5)) }
        _ = watcher
        return done.value
    }

    /// Third review, P2-8: a caller cancelled while its load ran kept waiting until the loader returned, then threw
    /// with the host `.loaded` for nobody. It stops waiting at once; with no caller left the load is detached and its
    /// model dropped, and the next request loads afresh.
    @Test(.timeLimit(.minutes(1))) func aCallerCancelledDuringItsLoadStopsWaitingAndTheModelIsDropped() async throws {
        let host = ModelHost(policy: .manual, monitor: MemoryPressureMonitor(source: .manual))
        let ref = testRef(), gate = RequestGate()  // the loader ignores cancellation until the gate opens
        let caller = Task { try await host.withModel(ref, load: { await gate.pass(); return 42 }) { (_: Int) in "ran" } }
        while await gate.entered == 0 { await Task.yield() }
        caller.cancel()
        #expect(await Self.ends(caller, within: 2), "the cancelled caller must not wait for the loader")
        await gate.open()
        await #expect(throws: CancellationError.self) { _ = try await caller.value }
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: ref) == .unloaded(nil))
        #expect(await host.residency().isEmpty)
        #expect(try await host.withModel(ref, load: { 7 }) { (value: Int) in value } == 7)
    }

    /// Third review, P2-8: a load another caller still waits for carries on when one caller is cancelled.
    @Test(.timeLimit(.minutes(1))) func aLoadOthersWaitForSurvivesOneCancelledCaller() async throws {
        let host = ModelHost(policy: .manual, monitor: MemoryPressureMonitor(source: .manual))
        let ref = testRef(), gate = RequestGate()
        let load: @Sendable () async throws -> Int = { await gate.pass(); return 42 }
        let first = Task { try await host.withModel(ref, load: load) { (value: Int) in value } }
        while await gate.entered == 0 { await Task.yield() }
        let second = Task { try await host.withModel(ref, load: load) { (value: Int) in value } }
        while await Self.pins(host, ref) < 2 { await Task.yield() }
        first.cancel()
        #expect(await Self.ends(first, within: 2))
        await #expect(throws: CancellationError.self) { _ = try await first.value }
        await gate.open()
        #expect(try await second.value == 42)
        let loads = await gate.entered
        #expect(await host.lifecycleState(of: ref) == .loaded && loads == 1)
    }

    static func pins(_ host: ModelHost, _ ref: ModelRef) async -> Int { await host.pinCount(ref) }
}

/// Second pre-release review, item 13: a timeout timer that fired before its request was queued left no marker, so
/// the request queued after timing out (and waited for a slot no timer would end).
@Suite("GPU admission timeout race")
struct GPUAdmissionTimeoutRaceTests {
    @Test(.timeLimit(.minutes(1))) func aTimeoutThatFiresBeforeQueueingRefusesTheRequest() async throws {
        let admission = try GPUAdmission()
        let held = try await admission.admit(owner: "holder")
        let late = Task {
            try await admission.admitQueued(owner: "waiter", timeout: .milliseconds(1)) {
                try? await Task.sleep(for: .milliseconds(100))  // the timer fires first
            }
        }
        try await Task.sleep(for: .milliseconds(300))
        held.release()  // before the fix the request had queued after its timeout, and now took this slot
        await #expect(throws: AetherError.self) { _ = try await late.value.release() }
        #expect(admission.snapshot.queued == 0 && admission.snapshot.running == 0)
    }
}
