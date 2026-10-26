import Foundation
import Synchronization

public enum GPUAdmissionError: Error, Sendable, Equatable, CustomStringConvertible {
    /// Fail-fast admission found every slot in use, or work already queued ahead.
    case busy
    /// A queued admission reached its timeout before a slot was free.
    case timedOut
    /// The global or per-owner outstanding limit refused the request.
    case limit(InferenceAdmissionError)

    public var description: String {
        switch self {
        case .busy: "The GPU is in use by other work in this process."
        case .timedOut: "Timed out waiting for the GPU."
        case .limit(let reason): "GPU admission limit reached (\(reason))."
        }
    }
}

/// Shared in-process GPU admission for model sessions. Hosts create one and pass it, explicitly, to every session that
/// should take turns; nothing is shared by default, and sessions without one keep their behavior.
///
/// Requests are served by a ``FairInferenceQueue``: FIFO per owner and round-robin across owners, with at most
/// `maximumConcurrent` grants running. A request either queues (optionally with a timeout; cancellation leaves the
/// queue) or fails fast with ``GPUAdmissionError/busy``.
///
/// With `crossProcessLease`, the admission also holds a ``SharedGPULease`` while any grant runs, so other cooperating
/// processes (Atelier, Kit tools) are excluded. Lease errors surface as ``SharedGPULeaseError``. Use it only in
/// unsandboxed hosts: a sandboxed app's temporary directory is private, so the lease would exclude nothing outside it.
/// The lease is polled; there is no fairness between processes.
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

    private enum Step: Sendable { case granted, acquireLease }
    private struct State: Sendable {
        var queue: FairInferenceQueue<String, Void>
        var waiters: [UUID: CheckedContinuation<Step, any Error>] = [:]
        var cancelledBeforeQueueing: Set<UUID> = []
        var lease: SharedGPULease?
        var leaseUsers = 0
        var leaseAcquiring = false
    }
    private let state: Mutex<State>

    public init(maximumConcurrent: Int = 1, waiting: Waiting = .queue(), maximumOutstanding: Int = 64,
                maximumOutstandingPerOwner: Int = 8, crossProcessLease: SharedGPULease.Configuration? = nil) throws {
        let queue: FairInferenceQueue<String, Void>
        do {
            queue = try FairInferenceQueue(maximumOutstanding: maximumOutstanding,
                                           maximumOutstandingPerOwner: maximumOutstandingPerOwner,
                                           maximumRunning: maximumConcurrent)
        } catch let error as InferenceAdmissionError { throw GPUAdmissionError.limit(error) }
        self.maximumConcurrent = maximumConcurrent
        self.waiting = waiting
        self.crossProcessLease = crossProcessLease
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
        case .queue(let timeout): return try await admitQueued(owner: owner, timeout: timeout)
        }
    }

    private func admitImmediately(owner: String) throws -> GPUAdmissionGrant {
        try Task.checkCancellation()
        let id = UUID()
        try state.withLock { state in
            guard state.queue.queuedCount == 0, state.queue.runningCount < maximumConcurrent, !state.leaseAcquiring else {
                throw GPUAdmissionError.busy
            }
            do { try state.queue.enqueue(id: id, owner: owner, work: ()) }
            catch let error as InferenceAdmissionError { throw GPUAdmissionError.limit(error) }
            guard state.queue.next()?.id == id else { preconditionFailure("an idle fair queue must start its only ticket") }
            guard let configuration = crossProcessLease else { return }
            if state.lease == nil {
                do { state.lease = try SharedGPULease.tryAcquire(configuration) }
                catch { try? state.queue.release(id: id); throw error }
            }
            state.leaseUsers += 1
        }
        return GPUAdmissionGrant(admission: self, id: id, owner: owner)
    }

    private func admitQueued(owner: String, timeout: Duration?) async throws -> GPUAdmissionGrant {
        let id = UUID()
        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now.advanced(by: max(.zero, $0)) }
        let timer = timeout.map { duration in
            Task { [weak self] in
                try? await Task.sleep(for: duration)
                if !Task.isCancelled { self?.abandon(id, with: GPUAdmissionError.timedOut, beforeQueueing: false) }
            }
        }
        defer {
            timer?.cancel()
            state.withLock { _ = $0.cancelledBeforeQueueing.remove(id) }
        }
        let step: Step = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let refusal: (any Error)? = state.withLock { state in
                    if state.cancelledBeforeQueueing.remove(id) != nil { return CancellationError() }
                    do { try state.queue.enqueue(id: id, owner: owner, work: ()) }
                    catch let error as InferenceAdmissionError { return GPUAdmissionError.limit(error) }
                    catch { return error }
                    state.waiters[id] = continuation
                    return nil
                }
                if let refusal { continuation.resume(throwing: refusal) } else { pump() }
            }
        } onCancel: {
            abandon(id, with: CancellationError(), beforeQueueing: true)
        }
        if step == .acquireLease, let configuration = crossProcessLease {
            do {
                let remaining = deadline.map { max(.zero, clock.now.duration(to: $0)) }
                let lease = try await SharedGPULease.acquire(configuration, timeout: remaining)
                state.withLock { state in
                    state.lease = lease
                    state.leaseUsers += 1
                    state.leaseAcquiring = false
                }
            } catch {
                state.withLock { state in
                    state.leaseAcquiring = false
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
            case .notFound: if beforeQueueing { state.cancelledBeforeQueueing.insert(id) }; return nil
            case .drainRunning: return nil
            }
        }
        waiter?.resume(throwing: error)
    }

    /// Starts as many queued requests as slots allow. While one request acquires the cross-process lease, no other
    /// starts, so at most one task ever waits on the lease.
    private func pump() {
        let started: [(CheckedContinuation<Step, any Error>, Step)] = state.withLock { state in
            var started: [(CheckedContinuation<Step, any Error>, Step)] = []
            while !(crossProcessLease != nil && state.leaseAcquiring), let ticket = state.queue.next() {
                guard let waiter = state.waiters.removeValue(forKey: ticket.id) else {
                    try? state.queue.release(id: ticket.id)
                    continue
                }
                if crossProcessLease == nil {
                    started.append((waiter, .granted))
                } else if state.lease != nil {
                    state.leaseUsers += 1
                    started.append((waiter, .granted))
                } else {
                    state.leaseAcquiring = true
                    started.append((waiter, .acquireLease))
                }
            }
            return started
        }
        for (waiter, step) in started { waiter.resume(returning: step) }
    }

    fileprivate func release(_ id: UUID) {
        state.withLock { state in
            guard (try? state.queue.release(id: id)) != nil else { return }
            guard crossProcessLease != nil, state.lease != nil else { return }
            state.leaseUsers -= 1
            if state.leaseUsers == 0 {
                state.lease?.release()
                state.lease = nil
            }
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

/// What a session accepts: a shared ``GPUAdmission`` and the owner name its requests are queued under.
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
