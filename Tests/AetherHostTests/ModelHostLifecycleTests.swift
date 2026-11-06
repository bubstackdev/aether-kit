import Foundation
import Synchronization
import Testing
@testable import AetherHost

// Ported from AetherKit 0.0.7's ModelLifecycleTests and CriticalEasingTests. There, a product owned the model and the
// lifecycle asked it to release; here the host owns the model, so "the product refused because a request started"
// is the host's own pin check, and the cases drive requests through `withModel`.

/// Sleeps that end only when the test fires them, so idle timers run on the test's schedule.
final class ManualSleeper: Sendable {
    private let waiters = Mutex<[CheckedContinuation<Void, any Error>]>([])

    var pending: Int { waiters.withLock { $0.count } }

    func sleep(_ duration: Duration) async throws {
        try await withCheckedThrowingContinuation { continuation in waiters.withLock { $0.append(continuation) } }
    }

    /// Ends every sleep started so far; a timer whose generation is stale then does nothing.
    func fire() {
        let ready = waiters.withLock { waiters in defer { waiters = [] }; return waiters }
        for waiter in ready { waiter.resume() }
    }
}

/// A clock and an available-memory reading the test sets.
final class FakeDevice: Sendable {
    private struct State { var instant = ContinuousClock.now; var available: UInt64? }
    private let state: Mutex<State>
    init(available: UInt64?) { state = Mutex(State(available: available)) }
    var now: ContinuousClock.Instant { state.withLock { $0.instant } }
    var available: UInt64? { state.withLock { $0.available } }
    func advance(_ duration: Duration) { state.withLock { $0.instant += duration } }
    func set(available: UInt64?) { state.withLock { $0.available = available } }
    var clock: HostClock { HostClock(now: { self.now }, sleep: { try await Task.sleep(for: $0) }) }
}

final class Locked<Value: Sendable>: Sendable {
    private let stored: Mutex<Value>
    init(_ value: Value) { stored = Mutex(value) }
    var value: Value {
        get { stored.withLock { $0 } }
        set { stored.withLock { $0 = newValue } }
    }
}

/// A stand-in for an engine actor: the host only holds and drops it.
final class FakeModel: Sendable {
    let generation: Int
    init(generation: Int) { self.generation = generation }
}

/// Counts loads and can fail the next ones.
final class Loader: Sendable {
    struct Failure: Error {}
    private let state = Mutex((loads: 0, failNext: 0))
    var loads: Int { state.withLock { $0.loads } }
    func failNext(_ count: Int) { state.withLock { $0.failNext = count } }
    func load() throws -> FakeModel {
        try state.withLock { state in
            if state.failNext > 0 { state.failNext -= 1; throw Failure() }
            state.loads += 1
            return FakeModel(generation: state.loads)
        }
    }
}

/// Holds a request open until the test opens it.
actor RequestGate {
    private var isOpen = false
    private var waiting: [CheckedContinuation<Void, Never>] = []
    private(set) var entered = 0

    func pass() async {
        entered += 1
        if isOpen { return }
        await withCheckedContinuation { waiting.append($0) }
    }

    func open() {
        isOpen = true
        for continuation in waiting { continuation.resume() }
        waiting = []
    }
}

func testRef(_ id: String = UUID().uuidString) -> ModelRef {
    ModelRef(catalogID: id, repo: "owner/\(id)", revision: String(repeating: "a", count: 40), variant: "macos",
             driver: .decoderLM, assets: [])
}

/// Waits until `ref` reaches a state `done` accepts (at most 5 s).
func settle(_ host: ModelHost, _ ref: ModelRef, until done: @escaping @Sendable (ModelLifecycleState) -> Bool) async throws {
    let states = await host.lifecycleStates(of: ref)
    let reached = try await withThrowingTaskGroup(of: Bool.self) { group in
        group.addTask { for await state in states where done(state) { return true }; return false }
        group.addTask { try await Task.sleep(for: .seconds(5)); return false }
        let reached = try await group.next() ?? false
        group.cancelAll()
        return reached
    }
    let state = await host.lifecycleState(of: ref)
    #expect(reached, "\(ref.catalogID) stayed at \(state)")
}

@Suite("ModelHost lifecycle")
struct ModelHostLifecycleTests {
    private let monitor = MemoryPressureMonitor(source: .manual)
    private let sleeper = ManualSleeper()
    private let loader = Loader()

    private func host(_ policy: ModelLifecyclePolicy = ModelLifecyclePolicy(idleUnloadAfter: .seconds(60))) -> ModelHost {
        ModelHost(policy: policy, monitor: monitor, clock: HostClock(now: { .now }, sleep: { [sleeper] in try await sleeper.sleep($0) }))
    }

    private func use(_ host: ModelHost, _ ref: ModelRef, policy: ModelLifecyclePolicy? = nil) async throws {
        try await host.withModel(ref, policy: policy, load: loader.load) { _ in }
    }

    /// Starts a request that stays running until `gate` opens, and returns once the host reports it running.
    private func hold(_ host: ModelHost, _ ref: ModelRef, _ gate: RequestGate) async throws -> Task<Void, any Error> {
        let task = Task { [loader] in try await host.withModel(ref, load: loader.load) { _ in await gate.pass() } }
        try await settle(host, ref) { $0 == .running }
        return task
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
        let host = host(), ref = testRef()
        #expect(await host.lifecycleState(of: ref) == .unloaded(nil))
        try await use(host, ref)
        #expect(await host.lifecycleState(of: ref) == .loaded)
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await settle(host, ref) { $0 == .unloaded(.idle) }
        #expect(await host.residency().isEmpty)
    }

    /// Also covers 0.0.7's "a release the model refuses leaves it loaded": the stale timer finds the model pinned.
    @Test func aRequestStopsTheTimerAndTheEndRestartsIt() async throws {
        let host = host(), ref = testRef(), gate = RequestGate()
        try await use(host, ref)
        while sleeper.pending == 0 { await Task.yield() }
        let request = try await hold(host, ref, gate)
        sleeper.fire()  // the stale timer finds the model running
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: ref) == .running)
        await gate.open()
        try await request.value
        #expect(await host.lifecycleState(of: ref) == .loaded && loader.loads == 1)
        while sleeper.pending == 0 { await Task.yield() }
        sleeper.fire()
        try await settle(host, ref) { $0 == .unloaded(.idle) }
    }

    @Test func pressureReleasesAnIdleModelAndARunningOneOnceItEnds() async throws {
        let host = host(), idle = testRef(), busy = testRef(), gate = RequestGate()
        try await use(host, idle)
        let request = try await hold(host, busy, gate)
        await monitor.deliver(.warning)
        try await settle(host, idle) { $0 == .unloaded(.memoryPressure) }
        #expect(await host.lifecycleState(of: busy) == .running)
        await gate.open()  // still under pressure
        try await request.value
        try await settle(host, busy) { $0 == .unloaded(.memoryPressure) }
        await monitor.deliver(.normal)
        let later = testRef()
        try await use(host, later)
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: later) == .loaded)
    }

    @Test func aSustainedWarningReleasesOncePerNotice() async throws {
        let host = host(ModelLifecyclePolicy(idleUnloadAfter: nil)), ref = testRef()
        try await use(host, ref)
        await monitor.deliver(.warning)
        try await settle(host, ref) { $0 == .unloaded(.memoryPressure) }
        // A request loads it again while the level stays at warning: it isn't released after every request.
        try await use(host, ref)
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: ref) == .loaded && loader.loads == 2)
        // A new notice releases it again.
        await monitor.deliver(.critical)
        try await settle(host, ref) { $0 == .unloaded(.memoryPressure) }
        await monitor.deliver(.normal)
    }

    @Test func aNoticeThatFindsTheModelUnloadedDoesNotReleaseItsNextLoad() async throws {
        let host = host(ModelLifecyclePolicy(idleUnloadAfter: nil)), ref = testRef()
        #expect(await host.lifecycleState(of: ref) == .unloaded(nil))
        _ = await host.lifecycleStates(of: ref)  // known to the host, but not loaded
        await monitor.deliver(.warning)  // unloaded: nothing to release
        try await use(host, ref)
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: ref) == .loaded)
        // A model first seen during the warning keeps its first load too.
        let later = testRef()
        try await use(host, later)
        try await Task.sleep(for: .milliseconds(50))
        #expect(await host.lifecycleState(of: later) == .loaded)
        await monitor.deliver(.normal)
    }

    @Test func aFailedLoadKeepsTheLastUnloadReason() async throws {
        let host = host(.manual), ref = testRef()
        loader.failNext(1)
        await #expect(throws: Loader.Failure.self) { try await use(host, ref) }
        #expect(await host.lifecycleState(of: ref) == .unloaded(nil))
        try await use(host, ref)
        await host.unload(ref)
        #expect(await host.lifecycleState(of: ref) == .unloaded(.requested))
        loader.failNext(1)
        await #expect(throws: Loader.Failure.self) { try await use(host, ref) }  // a failed reload
        #expect(await host.lifecycleState(of: ref) == .unloaded(.requested))
    }

    @Test func criticalPressureRefusesLoadsUntilItEases() async throws {
        let host = host(.manual), resident = testRef(), ref = testRef()
        try await use(host, resident)
        await monitor.deliver(.warning)
        try await use(host, ref)
        await host.unload(ref)
        await monitor.deliver(.critical)
        await #expect { try await use(host, ref) } throws: { error in
            if case AetherError.memoryPressure = error { return true }
            return false
        }
        #expect(await host.lifecycleState(of: ref) == .unloaded(.requested))
        // A model that is already loaded keeps serving.
        try await use(host, resident)
        await monitor.deliver(.normal)
        try await use(host, ref)
    }

    @Test func aManualPolicyNeverReleasesByItself() async throws {
        let host = host(.manual), ref = testRef()
        try await use(host, ref)
        await monitor.deliver(.critical)
        sleeper.fire()
        try await Task.sleep(for: .milliseconds(50))
        #expect(sleeper.pending == 0)
        #expect(await host.lifecycleState(of: ref) == .loaded)
        await monitor.deliver(.normal)
    }

    @Test func streamsStartWithTheCurrentValue() async throws {
        let host = host(.manual), ref = testRef()
        var states = await host.lifecycleStates(of: ref).makeAsyncIterator()
        #expect(await states.next() == .unloaded(nil))
        try await use(host, ref)
        var seen: [ModelLifecycleState] = []
        while seen.last != .loaded, let next = await states.next() { seen.append(next) }
        #expect(seen.last == .loaded)
        var levels = await monitor.levels().makeAsyncIterator()
        #expect(await levels.next() == .normal)
        await monitor.deliver(.warning)
        #expect(await levels.next() == .warning)
        await monitor.deliver(.normal)
    }

    // MARK: Added with the host: joined loads and deferred unloads.

    @Test func concurrentFirstUsesJoinOneLoad() async throws {
        let host = host(.manual), ref = testRef()
        let generations = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<8 {
                group.addTask { [loader] in try await host.withModel(ref, load: loader.load) { $0.generation } }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        #expect(loader.loads == 1 && Set(generations) == [1])
        #expect(await host.lifecycleState(of: ref) == .loaded)
    }

    @Test func anUnloadRequestedWhileRunningWaitsForTheRequest() async throws {
        let host = host(.manual), ref = testRef(), gate = RequestGate()
        let request = try await hold(host, ref, gate)
        await host.unload(ref)
        #expect(await host.lifecycleState(of: ref) == .running)
        await gate.open()
        try await request.value
        #expect(await host.lifecycleState(of: ref) == .unloaded(.requested))
    }
}

@Suite("Critical pressure easing")
struct CriticalEasingTests {
    @Test func criticalEasesOnceTheAppsMemoryHasDoubled() async throws {
        let device = FakeDevice(available: 600)
        let monitor = MemoryPressureMonitor(source: .manual, clock: device.clock, availableMemory: { device.available },
                                            kernelLevel: { nil })
        await monitor.deliver(.critical)
        device.set(available: 1_100)
        #expect(await monitor.level == .critical)
        device.set(available: 1_200)
        #expect(await monitor.level == .normal)
        let host = ModelHost(policy: .manual, monitor: monitor, clock: device.clock)
        try await host.withModel(testRef(), load: { FakeModel(generation: 1) }) { _ in }
    }

    @Test func criticalExpiresAfterItsLastNotice() async {
        let device = FakeDevice(available: nil)  // a Mac: no per-app reading
        let monitor = MemoryPressureMonitor(source: .manual, clock: device.clock, availableMemory: { device.available },
                                            kernelLevel: { nil })
        await monitor.deliver(.critical)
        device.advance(.seconds(45))
        await monitor.deliver(.critical)  // a repeated notice renews it
        device.advance(.seconds(45))
        #expect(await monitor.level == .critical)
        device.advance(.seconds(15))
        #expect(await monitor.level == .normal)
    }

    @Test func whereTheKernelStillReadsCriticalTheExpiryWaits() async {
        let device = FakeDevice(available: nil)
        let kernel = Locked(MemoryPressureLevel.critical)
        let monitor = MemoryPressureMonitor(source: .manual, clock: device.clock, availableMemory: { device.available },
                                            kernelLevel: { kernel.value })
        await monitor.deliver(.critical)
        device.advance(.seconds(61))
        #expect(await monitor.level == .critical)  // a sustained critical on a Mac
        device.advance(.seconds(30))
        kernel.value = .normal
        #expect(await monitor.level == .critical)  // the clock restarted at the first expiry
        device.advance(.seconds(31))
        #expect(await monitor.level == .normal)
    }

    @Test func aFreshCriticalAfterEasingIsANewNotice() async throws {
        let device = FakeDevice(available: 600)
        let monitor = MemoryPressureMonitor(source: .manual, clock: device.clock, availableMemory: { device.available },
                                            kernelLevel: { nil })
        let host = ModelHost(policy: ModelLifecyclePolicy(idleUnloadAfter: nil), monitor: monitor, clock: device.clock)
        let ref = testRef(), loader = Loader()
        try await host.withModel(ref, load: loader.load) { _ in }
        await monitor.deliver(.critical)
        try await settle(host, ref) { $0 == .unloaded(.memoryPressure) }
        device.advance(.seconds(61))
        #expect(await monitor.level == .normal)
        try await host.withModel(ref, load: loader.load) { _ in }
        await monitor.deliver(.critical)
        try await settle(host, ref) { $0 == .unloaded(.memoryPressure) }
        #expect(loader.loads == 2)
    }
}
