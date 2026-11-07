import AetherCore
import Synchronization

/// One turn at a time, in arrival order (DESIGN.md §6.6): a busy resource queues its callers instead of failing them.
///
/// The runtime's one serialization primitive. The text generator and chat session, the image, video and speech
/// pipelines, the stateless drivers and graph chains all take their turns through one; GPU priority ordering is
/// separate (`GPUAdmission`): a pipeline takes the GPU first, then its own turn.
///
/// - **FIFO:** turns are granted in the order callers arrived.
/// - **Cancellation while waiting:** the caller leaves the queue at once and ``withTurn(_:)`` throws
///   `CancellationError`, without waiting for its turn to come. A caller already cancelled when it arrives, or
///   cancelled just as its turn is handed to it, also throws and passes the turn on: a cancelled request never runs.
/// - **Cancellation while holding:** the body decides how to stop; the turn is released when it returns or throws.
/// - **Bound:** `limit` callers may wait (nil: any number); one more is refused with
///   ``AetherError/AdmissionFailure/queueFull(limit:)`` before it queues.
///
/// `Sendable`: the queue is behind a mutex, so a waiter's cancellation handler can remove it from any thread. The body
/// runs on the caller's executor (an actor's, typically), so it may use the actor's state.
public final class SerialGate: Sendable {
    private enum Slot {
        /// Queued; its continuation not made yet.
        case pending
        case waiting(CheckedContinuation<Void, any Error>)
        /// Handed the turn (or cancelled) before its continuation was made.
        case granted, cancelled
    }

    private struct State {
        var busy = false
        var nextID: UInt64 = 0
        var order: [UInt64] = []
        var slots: [UInt64: Slot] = [:]
    }

    /// The most callers that may wait for a turn (nil: no bound).
    public let limit: Int?
    private let state = Mutex(State())

    public init(limit: Int? = nil) {
        precondition((limit ?? 0) >= 0, "a SerialGate's limit can't be negative")
        self.limit = limit
    }

    /// Whether a turn is running.
    public var isBusy: Bool { state.withLock { $0.busy } }

    /// How many callers are waiting for a turn.
    public var waiting: Int { state.withLock { $0.order.count } }

    /// Runs `body` when every earlier caller's turn has ended. Throws `CancellationError` if the task is cancelled
    /// before the turn begins (leaving the queue at once), and `admission(.queueFull)` when `limit` callers already
    /// wait. The turn is released when `body` returns or throws.
    public func withTurn<R>(_ body: () async throws -> R) async throws -> R {
        try await enter(cancellable: true)
        defer { leave() }
        return try await body()
    }

    /// Runs `body` in a turn that neither cancellation nor the bound refuses: for an unload, which waits for the
    /// running request and then runs.
    public func withTurnIgnoringCancellation<R>(_ body: () async throws -> R) async rethrows -> R {
        try? await enter(cancellable: false)
        defer { leave() }
        return try await body()
    }

    private enum Arrival { case granted, queued(UInt64), full(Int) }

    private func enter(cancellable: Bool) async throws {
        if cancellable { try Task.checkCancellation() }
        let arrival: Arrival = state.withLock { s in
            if !s.busy {
                s.busy = true
                return .granted
            }
            if cancellable, let limit, s.order.count >= limit { return .full(limit) }
            s.nextID += 1
            s.order.append(s.nextID)
            s.slots[s.nextID] = .pending
            return .queued(s.nextID)
        }
        switch arrival {
        case .granted: return
        case .full(let limit): throw AetherError.admission(.queueFull(limit: limit))
        case .queued(let id):
            guard cancellable else { return try await wait(id) }
            try await withTaskCancellationHandler { try await wait(id) } onCancel: { cancel(id) }
            // Cancelled just as the turn was handed over: pass it on rather than run.
            if Task.isCancelled {
                leave()
                throw CancellationError()
            }
        }
    }

    private func wait(_ id: UInt64) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
            let slot: Slot? = state.withLock { s in
                switch s.slots[id] {
                case .pending?:
                    s.slots[id] = .waiting(continuation)
                    return nil
                case let other:
                    s.slots[id] = nil
                    return other
                }
            }
            switch slot {
            case nil: break
            case .granted?: continuation.resume()
            default: continuation.resume(throwing: CancellationError())
            }
        }
    }

    /// A waiter's task was cancelled: it leaves the queue now (unless the turn already reached it).
    private func cancel(_ id: UInt64) {
        let continuation: CheckedContinuation<Void, any Error>? = state.withLock { s in
            guard let i = s.order.firstIndex(of: id) else { return nil }
            s.order.remove(at: i)
            if case .waiting(let continuation)? = s.slots[id] {
                s.slots[id] = nil
                return continuation
            }
            s.slots[id] = .cancelled
            return nil
        }
        continuation?.resume(throwing: CancellationError())
    }

    private func leave() {
        let continuation: CheckedContinuation<Void, any Error>? = state.withLock { s in
            guard !s.order.isEmpty else {
                s.busy = false
                return nil
            }
            let id = s.order.removeFirst()
            if case .waiting(let continuation)? = s.slots[id] {
                s.slots[id] = nil
                return continuation
            }
            s.slots[id] = .granted
            return nil
        }
        continuation?.resume()
    }
}
