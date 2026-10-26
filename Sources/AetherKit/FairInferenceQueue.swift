import Foundation

public enum InferenceAdmissionError: Error, Equatable {
    case invalidLimits, duplicateID, globalLimit, ownerLimit, notRunning
}

/// Transport-independent admission for shared inference. The caller owns runtime
/// tasks and must release a running ticket only after native work has drained.
/// Cancelling a running ticket deliberately does not release its compute slot.
public struct FairInferenceQueue<Owner: Hashable & Sendable, Work: Sendable>: Sendable {
    public struct Ticket: Sendable {
        public let id: UUID
        public let owner: Owner
        public let work: Work
    }
    public typealias Failure = InferenceAdmissionError
    public enum Cancellation: Sendable, Equatable { case notFound, removedQueued, drainRunning }
    private let maximumOutstanding: Int
    private let maximumOutstandingPerOwner: Int
    private let maximumRunning: Int
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

    public mutating func enqueue(id: UUID, owner: Owner, work: Work) throws {
        guard tickets[id] == nil else { throw Failure.duplicateID }
        guard tickets.count < maximumOutstanding else { throw Failure.globalLimit }
        guard counts[owner, default: 0] < maximumOutstandingPerOwner else { throw Failure.ownerLimit }
        let ticket = Ticket(id: id, owner: owner, work: work)
        tickets[id] = ticket; counts[owner, default: 0] += 1
        if pending[owner] == nil { readyOwners.append(owner) }
        pending[owner, default: []].append(id)
    }
    /// Round-robin among users with queued work; FIFO within each user.
    public mutating func next() -> Ticket? {
        guard running.count < maximumRunning, !readyOwners.isEmpty else { return nil }
        // A new user arriving while the last user's request runs gets the next
        // turn instead of sitting behind that user's already queued burst.
        if readyOwners.count > 1, readyOwners.first == lastOwner { readyOwners.append(readyOwners.removeFirst()) }
        let owner = readyOwners.removeFirst()
        guard var ids = pending[owner], !ids.isEmpty else { return nil }
        let id = ids.removeFirst()
        if ids.isEmpty { pending.removeValue(forKey: owner) }
        else { pending[owner] = ids; readyOwners.append(owner) }
        lastOwner = owner; running.insert(id)
        return tickets[id]
    }
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
        if remaining == 0 { counts.removeValue(forKey: ticket.owner) }
        else { counts[ticket.owner] = remaining }
    }
}
