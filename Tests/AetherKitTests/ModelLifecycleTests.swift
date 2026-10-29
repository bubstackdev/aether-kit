@testable import AetherKit
import Foundation
import Testing

/// Sleeps that end only when the test fires them, so idle timers run on the test's schedule.
final class ManualSleeper: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [CheckedContinuation<Void, any Error>] = []

    var pending: Int { lock.withLock { waiters.count } }

    func sleep(_ duration: Duration) async throws {
        try await withCheckedThrowingContinuation { continuation in lock.withLock { waiters.append(continuation) } }
    }

    /// Ends every sleep started so far; a timer whose generation is stale then does nothing.
    func fire() {
        let ready = lock.withLock { () -> [CheckedContinuation<Void, any Error>] in defer { waiters = [] }; return waiters }
        for waiter in ready { waiter.resume() }
    }
}

/// A release that records its reasons and unloads unless told the model is busy.
final class RecordingRelease: @unchecked Sendable {
    private let lock = NSLock()
    private var reasons: [ModelUnloadReason] = []
    var busy = false
    var recorded: [ModelUnloadReason] { lock.withLock { reasons } }

    func attach(to lifecycle: ModelLifecycle) {
        lifecycle.setRelease { [weak self, weak lifecycle] reason in
            guard let self, let lifecycle, !lock.withLock({ busy }) else { return false }
            lock.withLock { reasons.append(reason) }
            lifecycle.unloaded(reason)
            return true
        }
    }
}

/// Waits until `lifecycle` reaches a state `done` accepts (at most 5 s).
func settle(_ lifecycle: ModelLifecycle, until done: @escaping @Sendable (ModelLifecycleState) -> Bool) async throws {
    try await withThrowingTaskGroup(of: Bool.self) { group in
        group.addTask { for await state in lifecycle.states where done(state) { return true }; return false }
        group.addTask { try await Task.sleep(for: .seconds(5)); return false }
        let reached = try await group.next() ?? false
        group.cancelAll()
        #expect(reached, "lifecycle stayed at \(lifecycle.current)")
    }
}

struct ModelLifecycleTests {
    private let monitor = MemoryPressureMonitor(systemSource: false)
    private let sleeper = ManualSleeper()

    private func lifecycle(_ policy: ModelLifecyclePolicy = ModelLifecyclePolicy(idleUnloadAfter: .seconds(60))) -> (ModelLifecycle, RecordingRelease) {
        let lifecycle = ModelLifecycle(policy: policy, monitor: monitor, sleep: { [sleeper] in try await sleeper.sleep($0) })
        let release = RecordingRelease()
        release.attach(to: lifecycle)
        return (lifecycle, release)
    }

    @Test func defaultsDependOnThePlatform() {
        #if os(iOS)
        #expect(ModelLifecyclePolicy.platformDefault.idleUnloadAfter == .seconds(60))
        #else
        #expect(ModelLifecyclePolicy.platformDefault.idleUnloadAfter == .seconds(300))
        #endif
        #expect(ModelLifecyclePolicy.platformDefault.releaseUnderMemoryPressure)
        #expect(ModelLifecyclePolicy.manual == ModelLifecyclePolicy(idleUnloadAfter: nil, releaseUnderMemoryPressure: false))
    }

    @Test func anIdleModelUnloadsWhenItsTimerFires() async throws {
        let (model, release) = lifecycle()
        #expect(model.current == .unloaded(nil))
        model.loading()
        model.loaded()
        #expect(model.current == .loaded)
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await settle(model) { $0 == .unloaded(.idle) }
        #expect(release.recorded == [.idle])
    }

    @Test func aRequestStopsTheTimerAndTheEndRestartsIt() async throws {
        let (model, release) = lifecycle()
        model.loaded()
        while sleeper.pending == 0 { await Task.yield() }
        model.running()
        sleeper.fire()  // the stale timer finds the model running
        try await Task.sleep(for: .milliseconds(50))
        #expect(model.current == .running && release.recorded.isEmpty)
        model.idle()
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await settle(model) { $0 == .unloaded(.idle) }
    }

    @Test func aReleaseTheModelRefusesLeavesItLoaded() async throws {
        let (model, release) = lifecycle()
        release.busy = true  // the product saw a request start under its own isolation
        model.loaded()
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await Task.sleep(for: .milliseconds(50))
        #expect(model.current == .loaded && release.recorded.isEmpty)
    }

    @Test func pressureReleasesAnIdleModelAndARunningOneOnceItEnds() async throws {
        let (idle, idleRelease) = lifecycle()
        let (busy, busyRelease) = lifecycle()
        idle.loaded()
        busy.running()
        monitor.deliver(.warning)
        try await settle(idle) { $0 == .unloaded(.memoryPressure) }
        #expect(idleRelease.recorded == [.memoryPressure])
        #expect(busy.current == .running && busyRelease.recorded.isEmpty)
        busy.idle()  // still under pressure
        try await settle(busy) { $0 == .unloaded(.memoryPressure) }
        monitor.deliver(.normal)
        let (later, laterRelease) = lifecycle()
        later.loaded()
        try await Task.sleep(for: .milliseconds(50))
        #expect(later.current == .loaded && laterRelease.recorded.isEmpty)
    }

    @Test func aSustainedWarningReleasesOncePerNotice() async throws {
        let (model, release) = lifecycle(ModelLifecyclePolicy(idleUnloadAfter: nil))
        model.loaded()
        monitor.deliver(.warning)
        try await settle(model) { $0 == .unloaded(.memoryPressure) }
        // A request loads it again while the level stays at warning: it isn't released after every request.
        model.loading()
        model.running()
        model.idle()
        try await Task.sleep(for: .milliseconds(50))
        #expect(model.current == .loaded && release.recorded == [.memoryPressure])
        // A new notice releases it again.
        monitor.deliver(.critical)
        try await settle(model) { $0 == .unloaded(.memoryPressure) }
        #expect(release.recorded == [.memoryPressure, .memoryPressure])
        monitor.deliver(.normal)
    }

    @Test func aNoticeThatFindsTheModelUnloadedDoesNotReleaseItsNextLoad() async throws {
        let (model, release) = lifecycle(ModelLifecyclePolicy(idleUnloadAfter: nil))
        monitor.deliver(.warning)  // unloaded: nothing to release
        model.loading()
        model.loaded()
        try await Task.sleep(for: .milliseconds(50))
        #expect(model.current == .loaded && release.recorded.isEmpty)
        // A model created during the warning keeps its first load too.
        let (later, laterRelease) = lifecycle(ModelLifecyclePolicy(idleUnloadAfter: nil))
        later.loaded()
        try await Task.sleep(for: .milliseconds(50))
        #expect(later.current == .loaded && laterRelease.recorded.isEmpty)
        monitor.deliver(.normal)
    }

    @Test func aFailedLoadKeepsTheLastUnloadReason() {
        let (model, _) = lifecycle(.manual)
        model.loading()
        model.notLoaded()
        #expect(model.current == .unloaded(nil))
        model.loaded()
        model.unloaded(.idle)
        model.loading()
        model.notLoaded()  // a refused or failed reload
        #expect(model.current == .unloaded(.idle))
    }

    @Test func criticalPressureRefusesLoadsUntilItEases() throws {
        let (model, _) = lifecycle()
        try model.admitLoad()
        monitor.deliver(.warning)
        try model.admitLoad()
        monitor.deliver(.critical)
        #expect(throws: AetherError.memoryPressure) { try model.admitLoad() }
        monitor.deliver(.normal)
        try model.admitLoad()
    }

    @Test func aManualPolicyNeverReleasesByItself() async throws {
        let (model, release) = lifecycle(.manual)
        model.loaded()
        monitor.deliver(.critical)
        sleeper.fire()
        try await Task.sleep(for: .milliseconds(50))
        #expect(sleeper.pending == 0 && model.current == .loaded && release.recorded.isEmpty)
        monitor.deliver(.normal)
    }

    @Test func streamsStartWithTheCurrentValue() async throws {
        let (model, _) = lifecycle(.manual)
        model.loading()
        var states = model.states.makeAsyncIterator()
        #expect(await states.next() == .loading)
        model.loaded()
        #expect(await states.next() == .loaded)
        var levels = monitor.levels.makeAsyncIterator()
        #expect(await levels.next() == .normal)
        monitor.deliver(.warning)
        #expect(await levels.next() == .warning)
        monitor.deliver(.normal)
    }
}
