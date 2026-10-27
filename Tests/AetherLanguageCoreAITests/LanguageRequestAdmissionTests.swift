import AetherKit
@testable import AetherLanguageCoreAI
import Foundation
import Synchronization
import Testing

/// Drives `LanguageRequestAdmission` (the start of every `LanguageGenerationSession` request) on an actor, as the
/// session does, with a fake slow memory-budget reload.
private actor FakeSession {
    let admission = LanguageRequestAdmission()
    private let gpu: GPUAdmissionOwner?
    private let reload: @Sendable () async -> Void
    init(gpu: GPUAdmissionOwner?, reload: @escaping @Sendable () async -> Void) { self.gpu = gpu; self.reload = reload }

    /// A request: admission, the reload under the grant, then (if still open) its work under the grant.
    func request(work: @Sendable () async -> Void = {}) async throws {
        let grant = try await admission.begin(gpu: gpu, prepare: { await self.reload() })
        defer { grant?.release() }
        await work()
    }
    func unload() { admission.withdraw() }
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
}

struct LanguageRequestAdmissionTests {
    /// Another owner holds the GPU: a request waiting for admission fails with `closed` as soon as the session
    /// unloads, before any reload, and never takes the GPU.
    @Test(.timeLimit(.minutes(1))) func unloadWithdrawsARequestQueuedBehindAnotherOwner() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let other = try await gpu.admit(owner: "other")
        let reloads = Mutex(0)
        let session = FakeSession(gpu: gpu.owner("language"), reload: { reloads.withLock { $0 += 1 } })
        let request = Task { try await session.request() }
        while gpu.snapshot.queued == 0 { await Task.yield() }
        await session.unload()
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(reloads.withLock { $0 } == 0)
        other.release()
        #expect(gpu.snapshot.running == 0)
    }

    /// The reload runs under the request's grant (never while another owner holds the GPU). An unload during it
    /// ends the request with `closed` when the reload returns, without its work, and the grant is released.
    @Test(.timeLimit(.minutes(1))) func unloadDuringASlowReloadEndsTheRequestWhenTheReloadReturns() async throws {
        let gpu = try GPUAdmission(maximumConcurrent: 1, waiting: .queue())
        let reloading = Gate(), finishReload = Gate()
        let runningDuringReload = Mutex<Int?>(nil)
        let session = FakeSession(gpu: gpu.owner("language"), reload: {
            runningDuringReload.withLock { $0 = gpu.snapshot.running }
            reloading.open()
            await finishReload.wait()
        })
        let worked = Mutex(false)
        let request = Task { try await session.request(work: { worked.withLock { $0 = true } }) }
        await reloading.wait()
        #expect(runningDuringReload.withLock { $0 } == 1)
        await session.unload()
        finishReload.open()
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
        #expect(!worked.withLock { $0 })
        #expect(gpu.snapshot.running == 0)
        // Later requests are refused without queueing.
        await #expect(throws: LanguageGenerationError.closed) { try await session.request() }
    }

    /// Without a GPU admission, an unload during the reload still ends the request when the reload returns.
    @Test(.timeLimit(.minutes(1))) func unloadDuringAReloadWithoutGPUAdmissionEndsTheRequest() async throws {
        let reloading = Gate(), finishReload = Gate()
        let session = FakeSession(gpu: nil, reload: { reloading.open(); await finishReload.wait() })
        let request = Task { try await session.request() }
        await reloading.wait()
        await session.unload()
        finishReload.open()
        await #expect(throws: LanguageGenerationError.closed) { try await request.value }
    }
}
