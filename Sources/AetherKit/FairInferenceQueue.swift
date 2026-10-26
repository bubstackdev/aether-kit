import Foundation

public enum InferenceAdmissionError: Error, Sendable, Equatable {
    case invalidLimits, duplicateID, globalLimit, ownerLimit, notRunning
}

/// Transport-independent, bounded admission with per-owner fairness: FIFO within an owner, round-robin across owners.
///
/// A Sendable value owned by the consumer's coordination actor or lock; it neither starts, cancels nor unloads native
/// work. The caller must release a running ticket only after native work has drained: cancelling a running ticket
/// deliberately keeps its compute slot. ``GPUAdmission`` is the Kit's own consumer. Kit does not interpret `Owner` as a
/// credential; authorize work before admitting it.
///
/// Adopted from the transport-independent admission design drafted for a host consumer in an earlier local line.
public struct FairInferenceQueue<Owner: Hashable & Sendable, Work: Sendable>: Sendable {
    public struct Ticket: Sendable {
        public let id: UUID
        public let owner: Owner
        public let work: Work
    }
    public typealias Failure = InferenceAdmissionError
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

    public init(maximumOutstanding: Int = 64, maximumOutstandingPerOwner: Int = 8, maximumRunning: Int = 1) throws {
        guard maximumOutstanding > 0, maximumOutstandingPerOwner > 0, maximumRunning > 0,
              maximumRunning <= maximumOutstanding else { throw Failure.invalidLimits }
        self.maximumOutstanding = maximumOutstanding
        self.maximumOutstandingPerOwner = maximumOutstandingPerOwner
        self.maximumRunning = maximumRunning
    }

    public var outstandingCount: Int { tickets.count }
    public var runningCount: Int { running.count }
    public var queuedCount: Int { tickets.count - running.count }

    /// Counts against the global and per-owner limits, including running tickets.
    public mutating func enqueue(id: UUID, owner: Owner, work: Work) throws {
        guard tickets[id] == nil else { throw Failure.duplicateID }
        guard tickets.count < maximumOutstanding else { throw Failure.globalLimit }
        guard counts[owner, default: 0] < maximumOutstandingPerOwner else { throw Failure.ownerLimit }
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
    public mutating func release(id: UUID) throws {
        guard running.remove(id) != nil, let ticket = tickets[id] else { throw Failure.notRunning }
        remove(ticket)
    }

    private mutating func remove(_ ticket: Ticket) {
        tickets.removeValue(forKey: ticket.id)
        let remaining = counts[ticket.owner, default: 1] - 1
        if remaining == 0 { counts.removeValue(forKey: ticket.owner) } else { counts[ticket.owner] = remaining }
    }
}
