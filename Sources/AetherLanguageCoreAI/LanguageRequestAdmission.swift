import AetherKit

/// Request admission and memory-budget reloads, shared by the native session and CPU tests. Owned
/// by, and only used on, its session's actor.
///
/// A request, once validated, is admitted to the GPU (when the session has a ``GPUAdmission``
/// owner) and then, still under its grant, reloads the model first if the budget is spent (a
/// multi-GB load never runs while another owner holds the GPU).
///
/// The reload normally happens earlier, at idle: when a request ends past the budget, the session
/// starts `runIdleReload` in a task, which reloads under its own grant. A request arriving while
/// that reload runs waits for it. A request arriving before the idle reload has its grant cancels
/// it and reloads in front instead (waiting would deadlock on the request's own grant). A
/// cancelled or failed idle reload leaves the budget spent, so the next request reloads in front.
///
/// `close()` is the session's unload: a request or idle reload waiting for admission is withdrawn
/// at once, later admissions are refused, an idle reload is cancelled (an in-progress load
/// finishes, then is dropped), and a request inside its reload fails with `closed` when the reload
/// returns. So close never waits behind the GPU queue, at most for one model load.
final class LanguageRequestAdmission {
    private var pending: [ObjectIdentifier: GPUAdmissionAttempt] = [:]
    private(set) var withdrawn = false
    private var idleTask: Task<Void, Never>?
    private var idleReloading = false
    private(set) var idleReloads = 0
    private(set) var requestReloads = 0

    /// Refuses later admissions and withdraws every waiting one.
    func withdraw() {
        withdrawn = true
        for attempt in pending.values { attempt.withdraw() }
    }

    /// Unload: withdraws, cancels an idle reload and waits for it to end.
    func close(isolation: isolated (any Actor)? = #isolation) async {
        withdraw()
        idleTask?.cancel()
        await idleTask?.value
    }

    /// The start of a request: admission, then (under the grant) a reload when `reloadNeeded()`.
    func begin(gpu: GPUAdmissionOwner?, reloadNeeded: () -> Bool, reload: () async throws -> Void,
               isolation: isolated (any Actor)? = #isolation) async throws -> GPUAdmissionGrant? {
        let grant = try await admit(gpu)
        do {
            if let idleTask {
                // Reloading: wait for it. Not yet holding its grant: cancel it and reload here.
                if !idleReloading { idleTask.cancel(); withdrawIdleAttempt() }
                await idleTask.value
            }
            if !withdrawn, reloadNeeded() {
                try await reload()
                requestReloads += 1
            }
            guard !withdrawn else { throw LanguageGenerationError.closed }
            return grant
        } catch {
            grant?.release()
            throw error
        }
    }

    /// Whether a request that just ended should start an idle reload.
    func shouldStartIdleReload(reloadNeeded: Bool) -> Bool { reloadNeeded && !withdrawn && idleTask == nil }

    /// Records the session's idle reload task (started after `shouldStartIdleReload`).
    func idleReloadStarted(_ task: Task<Void, Never>) { idleTask = task }

    /// The idle reload task's body: admission, then the reload under its grant.
    func runIdleReload(gpu: GPUAdmissionOwner?, reloadNeeded: () -> Bool, reload: () async throws -> Void,
                       isolation: isolated (any Actor)? = #isolation) async {
        defer { idleTask = nil; idleReloading = false; idleAttempt = nil }
        guard !Task.isCancelled, !withdrawn else { return }
        let grant: GPUAdmissionGrant?
        do { grant = try await admit(gpu, idle: true) } catch { return }
        defer { grant?.release() }
        guard !Task.isCancelled, !withdrawn, reloadNeeded() else { return }
        idleReloading = true
        do {
            try await reload()
            idleReloads += 1
        } catch {}
    }

    private var idleAttempt: GPUAdmissionAttempt?
    private func withdrawIdleAttempt() { idleAttempt?.withdraw() }

    private func admit(_ gpu: GPUAdmissionOwner?, idle: Bool = false,
                       isolation: isolated (any Actor)? = #isolation) async throws -> GPUAdmissionGrant? {
        guard !withdrawn else { throw LanguageGenerationError.closed }
        guard let gpu else { return nil }
        let attempt = GPUAdmissionAttempt(gpu)
        let key = ObjectIdentifier(attempt)
        pending[key] = attempt
        if idle { idleAttempt = attempt }
        defer { pending[key] = nil; if idle { idleAttempt = nil } }
        return try await attempt.admit(orThrow: { LanguageGenerationError.closed })
    }
}
