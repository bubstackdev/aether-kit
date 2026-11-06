/// One turn at a time, in arrival order (DESIGN.md §6.6): a busy resource queues its callers instead of failing them.
///
/// The runtime's single busy policy. The decoder engine and the image pipelines serialize their callers; a chain (and
/// any driver that shares graphs across chains) takes its turns through one gate, so a second caller waits rather than
/// meeting `.unavailable` on a lent arena (0.0.x's H3 graphs threw `sessionBusy`).
///
/// Non-`Sendable`: it belongs to the actor that owns the graphs it guards. A caller cancelled while waiting still
/// takes its turn when it comes (and the body is expected to check cancellation first), so turns are never lost.
public final class SerialGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    public init() {}

    /// Whether a turn is running.
    public var isBusy: Bool { busy }

    /// How many callers are waiting for a turn.
    public var waiting: Int { waiters.count }

    /// Runs `body` when every earlier caller's turn has ended.
    public func withTurn<R>(_ body: () async throws -> R) async rethrows -> R {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
        defer {
            if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
        }
        return try await body()
    }
}
