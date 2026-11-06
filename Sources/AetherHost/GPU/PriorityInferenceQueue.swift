public import Foundation

/// Transport-independent, bounded admission by priority: the highest ``GPUPriority`` waiting starts first, FIFO within
/// a level. Running work is never preempted, and there is no aging (a `.low` ticket starts only when no `.medium` or
/// `.high` one is waiting). Ported from AetherKit 0.0.7's fair queue, whose round-robin across owners it replaces.
///
/// A value type, owned by the consumer's actor or lock; it neither starts, cancels nor unloads native work. The caller
/// releases a running ticket only after native work has drained: cancelling a running ticket deliberately keeps its
/// compute slot. ``GPUAdmission`` is Aether's own consumer. `Owner` is a label for the per-owner limit, not a
/// credential; authorize work before admitting it. Refusals are ``AetherError/admission(_:)``.
public struct PriorityInferenceQueue<Owner: Hashable & Sendable, Work: Sendable>: Sendable {
    public struct Ticket: Sendable {
        public let id: UUID
        public let owner: Owner
        public let work: Work
        /// Changed by ``PriorityInferenceQueue/setPriority(_:of:)`` while the ticket waits.
        public fileprivate(set) var priority: GPUPriority
    }

    public enum Cancellation: Sendable, Equatable { case notFound, removedQueued, drainRunning }

    public let maximumOutstanding: Int
    public let maximumOutstandingPerOwner: Int
    public let maximumRunning: Int
    private var tickets: [UUID: Ticket] = [:]
    /// Queued tickets in arrival order.
    private var pending: [UUID] = []
    private var counts: [Owner: Int] = [:]
    private var running: Set<UUID> = []

    public init(maximumOutstanding: Int = 64, maximumOutstandingPerOwner: Int = 8, maximumRunning: Int = 1) throws(AetherError) {
        guard maximumOutstanding > 0, maximumOutstandingPerOwner > 0, maximumRunning > 0,
              maximumRunning <= maximumOutstanding else { throw .admission(.invalidLimits) }
        self.maximumOutstanding = maximumOutstanding
        self.maximumOutstandingPerOwner = maximumOutstandingPerOwner
        self.maximumRunning = maximumRunning
    }

    public var outstandingCount: Int { tickets.count }
    public var runningCount: Int { running.count }
    public var queuedCount: Int { pending.count }

    /// Counts against the global and per-owner limits, including running tickets.
    public mutating func enqueue(id: UUID, owner: Owner, priority: GPUPriority = .medium, work: Work) throws(AetherError) {
        guard tickets[id] == nil else { throw .admission(.duplicateID) }
        guard tickets.count < maximumOutstanding else { throw .admission(.globalLimit) }
        guard counts[owner, default: 0] < maximumOutstandingPerOwner else { throw .admission(.ownerLimit) }
        tickets[id] = Ticket(id: id, owner: owner, work: work, priority: priority)
        counts[owner, default: 0] += 1
        pending.append(id)
    }

    /// Changes a queued ticket's priority; it keeps its arrival order within the new level. Returns false (and changes
    /// nothing) for a running or unknown ticket.
    @discardableResult
    public mutating func setPriority(_ priority: GPUPriority, of id: UUID) -> Bool {
        guard !running.contains(id), tickets[id] != nil else { return false }
        tickets[id]?.priority = priority
        return true
    }

    /// Starts the next ticket when a running slot is free: the earliest queued ticket of the highest waiting priority.
    public mutating func next() -> Ticket? {
        guard running.count < maximumRunning, !pending.isEmpty else { return nil }
        var best = 0
        for index in pending.indices.dropFirst() where tickets[pending[index]]!.priority > tickets[pending[best]]!.priority {
            best = index
        }
        let id = pending.remove(at: best)
        running.insert(id)
        return tickets[id]
    }

    /// Removes queued work immediately. Running work returns `drainRunning` and keeps its slot until ``release(id:)``.
    public mutating func cancel(id: UUID) -> Cancellation {
        guard let ticket = tickets[id] else { return .notFound }
        if running.contains(id) { return .drainRunning }
        pending.removeAll { $0 == id }
        remove(ticket)
        return .removedQueued
    }

    /// Call after joined completion, including cancelled or failed native work.
    public mutating func release(id: UUID) throws(AetherError) {
        guard running.remove(id) != nil, let ticket = tickets[id] else { throw .admission(.notRunning) }
        remove(ticket)
    }

    private mutating func remove(_ ticket: Ticket) {
        tickets.removeValue(forKey: ticket.id)
        let remaining = counts[ticket.owner, default: 1] - 1
        if remaining == 0 { counts.removeValue(forKey: ticket.owner) } else { counts[ticket.owner] = remaining }
    }
}
