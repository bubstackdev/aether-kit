public import Foundation

/// Transport-independent, bounded admission with per-owner fairness: FIFO within an owner, round-robin across owners.
/// Ported from AetherKit 0.0.7, which adopted it from an earlier host-consumer admission draft.
///
/// A value type, owned by the consumer's actor or lock; it neither starts, cancels nor unloads native work. The caller
/// releases a running ticket only after native work has drained: cancelling a running ticket deliberately keeps its
/// compute slot. ``GPUAdmission`` is Aether's own consumer. `Owner` is a label, not a credential; authorize work before
/// admitting it. Refusals are ``AetherError/admission(_:)``.
public struct FairInferenceQueue<Owner: Hashable & Sendable, Work: Sendable>: Sendable {
    public struct Ticket: Sendable {
        public let id: UUID
        public let owner: Owner
        public let work: Work
    }

    public enum Cancellation: Sendable, Equatable { case notFound, removedQueued, drainRunning }

    public let maximumOutstanding: Int
    public let maximumOutstandingPerOwner: Int
    public let maximumRunning: Int
    private var tickets: [UUID: Ticket] = [:]
    private var pending: [Owner: [UUID]] = [:]
    private var readyOwners: [Owner] = []
    private var counts: [Owner: Int] = [:]
    private var running: Set<UUID> = []
    private var lastOwner: Owner?

    public init(maximumOutstanding: Int = 64, maximumOutstandingPerOwner: Int = 8, maximumRunning: Int = 1) throws(AetherError) {
        guard maximumOutstanding > 0, maximumOutstandingPerOwner > 0, maximumRunning > 0,
              maximumRunning <= maximumOutstanding else { throw .admission(.invalidLimits) }
        self.maximumOutstanding = maximumOutstanding
        self.maximumOutstandingPerOwner = maximumOutstandingPerOwner
        self.maximumRunning = maximumRunning
    }

    public var outstandingCount: Int { tickets.count }
    public var runningCount: Int { running.count }
    public var queuedCount: Int { tickets.count - running.count }

    /// Counts against the global and per-owner limits, including running tickets.
    public mutating func enqueue(id: UUID, owner: Owner, work: Work) throws(AetherError) {
        guard tickets[id] == nil else { throw .admission(.duplicateID) }
        guard tickets.count < maximumOutstanding else { throw .admission(.globalLimit) }
        guard counts[owner, default: 0] < maximumOutstandingPerOwner else { throw .admission(.ownerLimit) }
        tickets[id] = Ticket(id: id, owner: owner, work: work)
        counts[owner, default: 0] += 1
        if pending[owner] == nil { readyOwners.append(owner) }
        pending[owner, default: []].append(id)
    }

    /// Starts the next ticket when a running slot is free: round-robin among owners with queued work, FIFO within each.
    public mutating func next() -> Ticket? {
        guard running.count < maximumRunning, !readyOwners.isEmpty else { return nil }
        // A new owner arriving while the last owner's request runs gets the next turn instead of waiting behind that
        // owner's already queued burst.
        if readyOwners.count > 1, readyOwners.first == lastOwner { readyOwners.append(readyOwners.removeFirst()) }
        let owner = readyOwners.removeFirst()
        guard var ids = pending[owner], !ids.isEmpty else { return nil }
        let id = ids.removeFirst()
        if ids.isEmpty { pending.removeValue(forKey: owner) } else { pending[owner] = ids; readyOwners.append(owner) }
        lastOwner = owner
        running.insert(id)
        return tickets[id]
    }

    /// Removes queued work immediately. Running work returns `drainRunning` and keeps its slot until ``release(id:)``.
    public mutating func cancel(id: UUID) -> Cancellation {
        guard let ticket = tickets[id] else { return .notFound }
        if running.contains(id) { return .drainRunning }
        pending[ticket.owner]?.removeAll { $0 == id }
        if pending[ticket.owner]?.isEmpty == true {
            pending.removeValue(forKey: ticket.owner)
            readyOwners.removeAll { $0 == ticket.owner }
        }
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
