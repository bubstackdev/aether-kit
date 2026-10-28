import AetherKit
@testable import AetherLanguageCoreAI
import Foundation
import Synchronization
import Testing

/// Drives `LanguageRequestAdmission` on an actor exactly as `LanguageGenerationSession` does: each request is admitted,
/// reloads in front if the budget is spent, does its work, and on ending past the budget starts the idle reload. The
/// budget and the reload are fakes: `spend()` marks the budget spent, and a reload clears it after `reload` returns.
private actor FakeSession {
    let admission = LanguageRequestAdmission()
    private let gpu: GPUAdmissionOwner?
    private let slowReload: @Sendable () async -> Void
    private(set) var spent = false
    private(set) var reloads = 0
    init(gpu: GPUAdmissionOwner?, reload: @escaping @Sendable () async -> Void = {}) { self.gpu = gpu; self.slowReload = reload }

    func spend() { spent = true }

    private func reload() async throws {
        await slowReload()
        try Task.checkCancellation()
        spent = false
        reloads += 1
    }

    /// A request; `spends` marks the budget spent by its end, and `scratch` reserves its scratch.
    func request(spends: Bool = false, scratch: () throws -> ResourceReservation? = { nil },
                 work: @Sendable () async -> Void = {}) async throws {
        let admitted = try await admission.begin(gpu: gpu, reserveScratch: scratch, reloadNeeded: { self.spent },
                                                 reload: { try await self.reload() })
        defer { admitted.release() }
        await work()
        if spends { spent = true }
        startIdleReloadIfNeeded()
    }

    func startIdleReloadIfNeeded() {
        guard admission.shouldStartIdleReload(reloadNeeded: spent) else { return }
        admission.idleReloadStarted(Task {
            await self.admission.runIdleReload(gpu: self.gpu, reloadNeeded: { self.spent }, reload: { try await self.reload() })
        })
    }

    func unload() async { await admission.close() }
    var counts: [Int] { [admission.idleReloads, admission.requestReloads] }
    var closing: Bool { admission.withdrawn }
    var wouldStartIdleReload: Bool { admission.shouldStartIdleReload(reloadNeeded: spent) }
}

/// Opens once, from any task.
private final class Gate: Sendable {
    private let state = Mutex<(open: Bool, waiters: [CheckedContinuation<Void, Never>])>((false, []))
    func wait() async {
        await withCheckedContinuation { continuation in
            let resume = state.withLock { s -> Bool in
                if s.open { return true }
                s.waiters.append(continuation)
                return false
            }
            if resume { continuation.resume() }
        }
    }
    func open() {
        let waiters = state.withLock { s -> [CheckedContinuation<Void, Never>] in
            s.open = true
            defer { s.waiters = [] }
            return s.waiters
        }
        waiters.forEach { $0.resume() }
    }
    var isOpen: Bool { state.withLock { $0.open } }
}

private func until(_ condition: @Sendable () async -> Bool) async {
    while await !condition() { await Task.yield() }
}

struct LanguageRequestAdmissionTests {
    /// Another owner holds the GPU: a request waiting for admission fails with `closed` as soon as the session
    /// unloads, before any reload, and never takes the GPU.
    @Test(.timeLimit(.minutes(1))) func unloadWithdrawsARequestQueuedBehindAnotherOwner() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let other = try await gpu.admit(owner: "other")
        let session = FakeSession(gpu: gpu.owner("language"))
        await session.spend()
        let request = Task { try await session.request() }
        await until { gpu.snapshot.queued == 1 }
        await session.unload()
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(await session.reloads == 0)
        other.release()
        #expect(gpu.snapshot.running == 0)
    }

    /// A reload in front of a request runs under the request's grant. An unload during it ends the request with `closed`
    /// when the reload returns, without its work, and the grant is released; later requests are refused.
    @Test(.timeLimit(.minutes(1))) func unloadDuringAReloadInFrontEndsTheRequestWhenTheReloadReturns() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let reloading = Gate(), finishReload = Gate()
        let runningDuringReload = Mutex<Int?>(nil)
        let session = FakeSession(gpu: gpu.owner("language"), reload: {
            runningDuringReload.withLock { $0 = gpu.snapshot.running }
            reloading.open()
            await finishReload.wait()
        })
        await session.spend()
        let worked = Mutex(false)
        let request = Task { try await session.request(work: { worked.withLock { $0 = true } }) }
        await reloading.wait()
        #expect(runningDuringReload.withLock { $0 } == 1)
        let unload = Task { await session.unload() }
        await until { await session.closing }
        finishReload.open()
        await unload.value
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(!worked.withLock { $0 })
        #expect(gpu.snapshot.running == 0)
        await #expect(throws: LanguageGenerationError.closed) { try await session.request() }
    }

    /// Without a GPU admission, an unload during a reload in front still ends the request when the reload returns.
    @Test(.timeLimit(.minutes(1))) func unloadDuringAReloadWithoutGPUAdmissionEndsTheRequest() async throws {
        let reloading = Gate(), finishReload = Gate()
        let session = FakeSession(gpu: nil, reload: { reloading.open(); await finishReload.wait() })
        await session.spend()
        let request = Task { try await session.request() }
        await reloading.wait()
        let unload = Task { await session.unload() }
        await until { await session.closing }
        finishReload.open()
        await unload.value
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
    }

    /// A request that spends the budget starts the reload at idle, under its own grant; the next request runs without one.
    @Test(.timeLimit(.minutes(1))) func reloadRunsAtIdleUnderAGrant() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let runningDuringReload = Mutex<Int?>(nil)
        let session = FakeSession(gpu: gpu.owner("language"), reload: { runningDuringReload.withLock { $0 = gpu.snapshot.running } })
        try await session.request(spends: true)
        await until { await session.reloads == 1 }
        #expect(runningDuringReload.withLock { $0 } == 1)
        try await session.request()
        #expect(await session.counts == [1, 0])
        #expect(gpu.snapshot.running == 0)
    }

    /// A request arriving while the idle reload runs waits for it and does not reload again.
    @Test(.timeLimit(.minutes(1))) func requestDuringAnIdleReloadWaitsForIt() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let reloading = Gate(), finishReload = Gate()
        let session = FakeSession(gpu: gpu.owner("language"), reload: { reloading.open(); await finishReload.wait() })
        try await session.request(spends: true)
        await reloading.wait()
        let ran = Mutex(false)
        let request = Task { try await session.request(work: { ran.withLock { $0 = true } }) }
        await until { gpu.snapshot.queued == 1 }
        #expect(!ran.withLock { $0 })
        finishReload.open()
        try await request.value
        #expect(ran.withLock { $0 })
        #expect(await session.counts == [1, 0])
    }

    /// Unload during an idle reload queued behind another owner returns at once: its admission is withdrawn, it never
    /// reloads, and the GPU queue is left empty.
    @Test(.timeLimit(.minutes(1))) func unloadWithdrawsAnIdleReloadQueuedBehindAnotherOwner() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let session = FakeSession(gpu: gpu.owner("language"))
        let other = try await gpu.admit(owner: "other")
        await session.spend()
        await session.startIdleReloadIfNeeded()
        await until { gpu.snapshot.queued == 1 }
        await session.unload()
        #expect(await session.reloads == 0)
        #expect(gpu.snapshot.queued == 0)
        other.release()
        #expect(gpu.snapshot.running == 0)
    }

    /// Unload during an idle reload in progress cancels it: unload waits only for that reload, whose result is dropped.
    @Test(.timeLimit(.minutes(1))) func unloadCancelsAnIdleReloadInProgress() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let reloading = Gate(), finishReload = Gate()
        let session = FakeSession(gpu: gpu.owner("language"), reload: { reloading.open(); await finishReload.wait() })
        try await session.request(spends: true)
        await reloading.wait()
        let unload = Task { await session.unload() }
        await until { await session.closing }
        finishReload.open()
        await unload.value
        #expect(await session.reloads == 0)        // cancelled: the fake drops its result after the load returns
        #expect(gpu.snapshot.running == 0)
    }

    /// A request admitted before the idle reload has its grant must not wait for it (the idle reload would wait for the
    /// request's grant): it cancels the idle reload and reloads in front. Exactly one reload happens either way.
    @Test(.timeLimit(.minutes(1))) func requestOvertakingAPendingIdleReloadReloadsInFront() async throws {
        for _ in 0..<50 {
            let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
            let session = FakeSession(gpu: gpu.owner("language"))
            await session.spend()
            // Start the idle reload and a request in the same actor turn: whichever is admitted first, both finish. A
            // deadlock ignores cancellation, so a watchdog fails the test instead of letting it hang.
            let finished = Mutex(false)
            let request = Task { try await session.requestAfterStartingIdleReload(); finished.withLock { $0 = true } }
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while !finished.withLock({ $0 }) && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
            guard finished.withLock({ $0 }) else {
                Issue.record("The request and the idle reload deadlocked on the GPU grant")
                return
            }
            try await request.value
            await until { await session.counts.reduce(0, +) >= 1 }
            await session.unload()
            #expect(await session.counts.reduce(0, +) == 1)
            #expect(await session.reloads == 1)
            #expect(gpu.snapshot.running == 0 && gpu.snapshot.queued == 0)
        }
    }

    /// A request's scratch is reserved under its grant before a reload in front: a refusal costs no load and releases
    /// the grant, and an admitted request holds its scratch until it ends.
    @Test(.timeLimit(.minutes(1))) func scratchIsReservedBeforeAReloadInFront() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let coordinator = try ResourceCoordinator(capacityBytes: 10)
        let session = FakeSession(gpu: gpu.owner("language"))
        await session.spend()
        await #expect(throws: ResourceAdmissionError.self) {
            try await session.request(scratch: { try coordinator.reserve(bytes: 11) })
        }
        #expect(await session.reloads == 0)
        #expect(gpu.snapshot.running == 0 && coordinator.snapshot.reservedBytes == 0)
        let held = Mutex<UInt64>(0)
        try await session.request(scratch: { try coordinator.reserve(bytes: 4) },
                                  work: { held.withLock { $0 = coordinator.snapshot.reservedBytes } })
        #expect(await session.reloads == 1)
        #expect(held.withLock { $0 } == 4 && coordinator.snapshot.reservedBytes == 0)
        #expect(gpu.snapshot.running == 0)
    }

    /// No idle reload starts once the session is closing.
    @Test func noIdleReloadAfterClose() async {
        let session = FakeSession(gpu: nil)
        await session.spend()
        #expect(await session.wouldStartIdleReload)
        await session.unload()
        #expect(await !session.wouldStartIdleReload)
    }
}

extension FakeSession {
    func requestAfterStartingIdleReload() async throws {
        startIdleReloadIfNeeded()
        try await request()
    }
}
