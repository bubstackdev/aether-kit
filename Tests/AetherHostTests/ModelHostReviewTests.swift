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
