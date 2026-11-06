public import Foundation
import Synchronization

/// In-process GPU admission, shared explicitly by the work that should take turns. Ported from AetherKit 0.0.7.
///
/// There is no process-global default: concurrency is a host decision, so a host creates one and passes it (or an
/// ``owner(_:)`` handle) to every engine or pipeline that should queue behind it. Requests are served by a
/// ``FairInferenceQueue``: FIFO per owner and round-robin across owners, with at most `maximumConcurrent` grants
/// running. A request either queues (optionally with a timeout; cancellation leaves the queue) or fails fast.
///
/// With `crossProcessLease`, the admission also holds a ``SharedGPULease`` while any grant runs, so cooperating
/// processes (Atelier, the lab wrappers, Aether tools) are excluded. Use it only in unsandboxed hosts: a sandboxed
/// app's temporary directory is private, so the lease would exclude nothing outside it.
///
/// The lease is polled, so under steady in-process load another process could starve. The admission therefore keeps
/// an owned lease across consecutive (or overlapping) grants for at most `crossProcessTurn`. After that it starts no
/// new grant on that lease: running grants drain, the lease is released, and the admission waits
/// `crossProcessHandOffGap` before re-acquiring, so a process polling at least that often can take a turn. Before each
/// grant the held lease is re-checked: an inherited lease's holder must still be named by the lease and alive.
///
/// The type is a `Sendable` class over a `Mutex` rather than an actor because a grant's release must be synchronous
/// (idempotent, and run from `deinit`); the lock is never held across a suspension or file I/O.
///
/// Refusals are ``AetherError/admission(_:)``: `.busy`, `.timedOut`, the queue's limits, or a lease failure.
public final class GPUAdmission: Sendable {
    public enum Waiting: Sendable, Equatable {
        /// Wait for a slot, up to `timeout` when given.
        case queue(timeout: Duration? = nil)
        /// Refuse immediately unless a slot is free and nothing is queued ahead.
        case failFast
    }

    public let maximumConcurrent: Int
    public let waiting: Waiting
    public let crossProcessLease: SharedGPULease.Configuration?
    public let crossProcessTurn: Duration
    public let crossProcessHandOffGap: Duration

    private enum Step: Sendable, Equatable { case granted, acquireLease(after: Duration) }
    private struct State: Sendable {
        var queue: FairInferenceQueue<String, Void>
        var waiters: [UUID: CheckedContinuation<Step, any Error>] = [:]
        /// Requests abandoned (cancelled or timed out) before they were queued, with the error their enqueue throws.
        var cancelledBeforeQueueing: [UUID: any Error] = [:]
        var lease: SharedGPULease?
        var leaseGrants: Set<UUID> = []
        var leaseSince: ContinuousClock.Instant?
        var handOffUntil: ContinuousClock.Instant?
        /// One task at a time acquires or releases the lease; no other request starts meanwhile.
        var leaseBusy = false
        /// The turn expired: no grant joins the held lease; running grants drain and then the lease is handed off.
        var draining = false
    }
    private let state: Mutex<State>

    public init(maximumConcurrent: Int = 1, waiting: Waiting = .queue(), maximumOutstanding: Int = 64,
                maximumOutstandingPerOwner: Int = 8, crossProcessLease: SharedGPULease.Configuration? = nil,
                crossProcessTurn: Duration = .seconds(30), crossProcessHandOffGap: Duration = .milliseconds(500)) throws(AetherError) {
        let queue = try FairInferenceQueue<String, Void>(
            maximumOutstanding: maximumOutstanding, maximumOutstandingPerOwner: maximumOutstandingPerOwner,
            maximumRunning: maximumConcurrent)
        self.maximumConcurrent = maximumConcurrent
        self.waiting = waiting
        self.crossProcessLease = crossProcessLease
        self.crossProcessTurn = crossProcessTurn
        self.crossProcessHandOffGap = crossProcessHandOffGap
        state = Mutex(State(queue: queue))
    }

    public struct Snapshot: Sendable, Equatable {
        public let running: Int
        public let queued: Int
        public let holdsCrossProcessLease: Bool

        public init(running: Int, queued: Int, holdsCrossProcessLease: Bool) {
            self.running = running
            self.queued = queued
            self.holdsCrossProcessLease = holdsCrossProcessLease
        }
    }

    public var snapshot: Snapshot {
        state.withLock { Snapshot(running: $0.queue.runningCount, queued: $0.queue.queuedCount, holdsCrossProcessLease: $0.lease != nil) }
    }

    /// A handle naming the owner whose requests share one fairness lane.
    public func owner(_ name: String) -> GPUAdmissionOwner { GPUAdmissionOwner(admission: self, name: name) }

    /// Admits one unit of GPU work for `owner`. Hold the grant until the work's native submissions have joined.
    public func admit(owner: String, waiting: Waiting? = nil) async throws -> GPUAdmissionGrant {
        switch waiting ?? self.waiting {
        case .failFast: return try admitImmediately(owner: owner)
        case .queue(let timeout): return try await admitQueued(owner: owner, timeout: timeout, beforeEnqueue: nil)
        }
    }

    /// The held lease and whether it is still valid, checked outside the state lock (it reads the pid file and signals
    /// the holder).
    private struct LeaseCheck { let lease: SharedGPULease; let valid: Bool }

    private func checkLease() -> LeaseCheck? {
        guard let lease = state.withLock({ $0.lease }) else { return nil }
        return LeaseCheck(lease: lease, valid: lease.isStillValid)
    }

    /// Joins the held lease if it is still valid; drops it otherwise. A lease installed after `check` was taken has just
    /// been acquired and counts as valid.
    private static func joinLease(_ state: inout State, _ id: UUID, _ check: LeaseCheck?) -> Bool {
        guard let lease = state.lease else { return false }
        guard check.map({ $0.lease !== lease || $0.valid }) ?? true else {
            state.lease = nil
            state.leaseGrants = []
            state.leaseSince = nil
            state.draining = false
            return false
        }
        state.leaseGrants.insert(id)
        return true
    }

    /// True once an owned lease has been held for its whole turn.
    private func turnExpired(_ state: State, at now: ContinuousClock.Instant) -> Bool {
        guard let lease = state.lease, lease.holding == .owned, let since = state.leaseSince else { return false }
        return since.duration(to: now) >= crossProcessTurn
    }

    private func admitImmediately(owner: String) throws -> GPUAdmissionGrant {
        try Task.checkCancellation()
        let id = UUID()
        let check = crossProcessLease == nil ? nil : checkLease()
        let mustAcquire = try state.withLock { state throws(AetherError) -> Bool in
            guard state.queue.queuedCount == 0, state.queue.runningCount < maximumConcurrent, !state.leaseBusy,
                  !state.draining else {
                throw .admission(.busy)
            }
            if crossProcessLease != nil, turnExpired(state, at: .now) {
                // The turn is over: running grants drain, then the lease is handed off. With none running, the pump
                // that follows every release hands it off now.
                if !state.leaseGrants.isEmpty { state.draining = true }
                throw .admission(.busy)
            }
            try state.queue.enqueue(id: id, owner: owner, work: ())
            guard state.queue.next()?.id == id else { preconditionFailure("an idle fair queue must start its only ticket") }
            guard crossProcessLease != nil, !Self.joinLease(&state, id, check) else { return false }
            state.leaseBusy = true
            return true
        }
        if mustAcquire, let configuration = crossProcessLease {
            // The lease's mkdir, reclaim and flock run outside the state lock; `leaseBusy` keeps others out meanwhile.
            do {
                let lease = try SharedGPULease.tryAcquire(configuration)
                installLease(lease, for: id)
            } catch {
                state.withLock { state in
                    state.leaseBusy = false
                    try? state.queue.release(id: id)
                }
                pump()
                throw error
            }
            pump()
        }
        return GPUAdmissionGrant(admission: self, id: id, owner: owner)
    }

    private func installLease(_ lease: SharedGPULease, for id: UUID) {
        state.withLock { state in
            state.lease = lease
            state.leaseGrants = [id]
            state.leaseSince = ContinuousClock.now
            state.handOffUntil = nil
            state.leaseBusy = false
            state.draining = false
        }
    }

    /// Queues with `timeout`. `beforeEnqueue` runs between starting the timer and queueing (tests widen the window).
    func admitQueued(
        owner: String, timeout: Duration?, beforeEnqueue: (@Sendable () async -> Void)?
    ) async throws -> GPUAdmissionGrant {
        let id = UUID()
        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now.advanced(by: max(.zero, $0)) }
        let timer = timeout.map { duration in
            Task { [weak self] in
                try? await Task.sleep(for: duration)
                // The timer can fire before the request is queued: leave a marker then too, or the request would queue
                // after it timed out.
                if !Task.isCancelled { self?.abandon(id, with: AetherError.admission(.timedOut), beforeQueueing: true) }
            }
        }
        defer {
            timer?.cancel()
            state.withLock { _ = $0.cancelledBeforeQueueing.removeValue(forKey: id) }
        }
        await beforeEnqueue?()
        let step: Step = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let refusal: (any Error)? = state.withLock { state in
                    if let abandoned = state.cancelledBeforeQueueing.removeValue(forKey: id) { return abandoned }
                    do { try state.queue.enqueue(id: id, owner: owner, work: ()) } catch { return error }
                    state.waiters[id] = continuation
                    return nil
                }
                if let refusal { continuation.resume(throwing: refusal) } else { pump() }
            }
        } onCancel: {
            abandon(id, with: CancellationError(), beforeQueueing: true)
        }
        if case .acquireLease(let delay) = step, let configuration = crossProcessLease {
            do {
                if delay > .zero {
                    if let deadline, clock.now.advanced(by: delay) >= deadline { throw AetherError.admission(.timedOut) }
                    try await Task.sleep(for: delay)
                }
                let remaining = deadline.map { max(.zero, clock.now.duration(to: $0)) }
                installLease(try await SharedGPULease.acquire(configuration, timeout: remaining), for: id)
            } catch {
                state.withLock { state in
                    state.leaseBusy = false
                    try? state.queue.release(id: id)
                }
                pump()
                throw error
            }
            pump()
        }
        let grant = GPUAdmissionGrant(admission: self, id: id, owner: owner)
        if Task.isCancelled { grant.release(); throw CancellationError() }
        return grant
    }

    /// Removes a still-queued request. Before it is queued, cancellation leaves a marker the enqueue step honors.
    private func abandon(_ id: UUID, with error: any Error, beforeQueueing: Bool) {
        let waiter: CheckedContinuation<Step, any Error>? = state.withLock { state in
            switch state.queue.cancel(id: id) {
            case .removedQueued: return state.waiters.removeValue(forKey: id)
            case .notFound: if beforeQueueing { state.cancelledBeforeQueueing[id] = error }; return nil
            case .drainRunning: return nil
            }
        }
        waiter?.resume(throwing: error)
    }

    /// Starts as many queued requests as slots allow. While one request acquires (or the admission releases) the
    /// cross-process lease, no other starts, so at most one task ever waits on the lease.
    private func pump() {
        let check = crossProcessLease == nil ? nil : checkLease()
        let (started, handOff) = state.withLock { state -> ([(CheckedContinuation<Step, any Error>, Step)], SharedGPULease?) in
            var started: [(CheckedContinuation<Step, any Error>, Step)] = []
            var handOff: SharedGPULease?
            while true {
                if crossProcessLease != nil {
                    if state.leaseBusy || state.draining { break }
                    let now = ContinuousClock.now
                    if state.queue.queuedCount > 0, turnExpired(state, at: now) {
                        if state.leaseGrants.isEmpty {
                            // Nothing runs on the lease: hand it off now.
                            handOff = state.lease
                            state.lease = nil
                            state.leaseSince = nil
                            state.handOffUntil = now.advanced(by: crossProcessHandOffGap)
                            state.leaseBusy = true
                        } else {
                            // Start nothing more on this lease; it is handed off once running grants drain.
                            state.draining = true
                        }
                        break
                    }
                }
                guard let ticket = state.queue.next() else { break }
                guard let waiter = state.waiters.removeValue(forKey: ticket.id) else {
                    try? state.queue.release(id: ticket.id)
                    continue
                }
                if crossProcessLease == nil || Self.joinLease(&state, ticket.id, check) {
                    started.append((waiter, .granted))
                } else {
                    state.leaseBusy = true
                    let delay = state.handOffUntil.map { max(.zero, ContinuousClock.now.duration(to: $0)) } ?? .zero
                    started.append((waiter, .acquireLease(after: delay)))
                }
            }
            return (started, handOff)
        }
        for (waiter, step) in started { waiter.resume(returning: step) }
        if let handOff {
            handOff.release()
            state.withLock { $0.leaseBusy = false }
            pump()
        }
    }

    fileprivate func release(_ id: UUID) {
        let finished: SharedGPULease? = state.withLock { state in
            guard (try? state.queue.release(id: id)) != nil else { return nil }
            guard state.leaseGrants.remove(id) != nil, state.leaseGrants.isEmpty, let lease = state.lease else { return nil }
            let now = ContinuousClock.now
            let othersWaiting = state.queue.queuedCount > 0
            if othersWaiting, !state.draining, lease.holding == .owned, !turnExpired(state, at: now) {
                return nil  // Hand the lease straight to the next in-process request.
            }
            let handOff = (othersWaiting || state.draining) && lease.holding == .owned
            state.lease = nil
            state.leaseSince = nil
            state.draining = false
            if handOff { state.handOffUntil = now.advanced(by: crossProcessHandOffGap) }
            state.leaseBusy = true
            return lease
        }
        if let finished {
            // File I/O outside the state lock; `leaseBusy` holds back acquisition until the directory is gone.
            finished.release()
            state.withLock { $0.leaseBusy = false }
        }
        pump()
    }
}

/// One admitted unit of GPU work. Release after the work's native submissions have joined, including after
/// cancellation or failure. Releasing is idempotent and also happens on deinit.
public final class GPUAdmissionGrant: Sendable {
    public let owner: String
    private let admission: GPUAdmission
    private let id: UUID
    private let released = Mutex(false)

    fileprivate init(admission: GPUAdmission, id: UUID, owner: String) {
        self.admission = admission
        self.id = id
        self.owner = owner
    }

    public func release() {
        guard released.withLock({ value in defer { value = true }; return !value }) else { return }
        admission.release(id)
    }

    deinit { release() }
}

/// What an engine or pipeline accepts: a shared ``GPUAdmission`` and the owner name its requests are queued under.
public struct GPUAdmissionOwner: Sendable {
    public let admission: GPUAdmission
    public let name: String

    public init(admission: GPUAdmission, name: String) {
        self.admission = admission
        self.name = name
    }

    public func admit(waiting: GPUAdmission.Waiting? = nil) async throws -> GPUAdmissionGrant {
        try await admission.admit(owner: name, waiting: waiting)
    }
}

/// One operation's admission, which the operation's owner can withdraw while it is still waiting, so an unload never
/// waits behind the GPU queue. A withdrawal after the grant but before `admit` returns releases the grant.
package final class GPUAdmissionAttempt: Sendable {
    private enum Phase: Sendable { case idle, waiting(Task<GPUAdmissionGrant, any Error>), granted, withdrawn }
    private let phase = Mutex(Phase.idle)
    private let owner: GPUAdmissionOwner

    package init(_ owner: GPUAdmissionOwner) { self.owner = owner }

    /// Throws `closed()` when withdrawn.
    package func admit(orThrow closed: @Sendable () -> any Error) async throws -> GPUAdmissionGrant {
        let owner = owner
        let task: Task<GPUAdmissionGrant, any Error>? = phase.withLock { phase in
            guard case .idle = phase else { return nil }
            let task = Task { try await owner.admit() }
            phase = .waiting(task)
            return task
        }
        guard let task else { throw closed() }
        let grant: GPUAdmissionGrant
        do {
            grant = try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
        } catch {
            if isWithdrawn { throw closed() }
            throw error
        }
        let kept = phase.withLock { phase -> Bool in
            guard case .waiting = phase else { return false }
            phase = .granted
            return true
        }
        guard kept else { grant.release(); throw closed() }
        return grant
    }

    private var isWithdrawn: Bool { phase.withLock { if case .withdrawn = $0 { true } else { false } } }

    /// Withdraws a request that has not been granted yet. Has no effect once granted.
    package func withdraw() {
        let task: Task<GPUAdmissionGrant, any Error>? = phase.withLock { phase in
            switch phase {
            case .idle: phase = .withdrawn; return nil
            case .waiting(let task): phase = .withdrawn; return task
            case .granted, .withdrawn: return nil
            }
        }
        task?.cancel()
    }
}
