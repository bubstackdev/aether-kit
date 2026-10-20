import Foundation
import Synchronization

/// Admission estimates, not measured process memory or an OS allocation guarantee.
public enum ResourceAdmissionError: Error, Sendable, Equatable {
    case invalidByteCount
    case insufficientCapacity(requested: UInt64, available: UInt64)
    case releasedReservation
}

public struct ResourceSnapshot: Sendable, Equatable {
    public let capacityBytes: UInt64
    public let reservedBytes: UInt64
    public let reservationCount: Int
    public var availableBytes: UInt64 { capacityBytes - reservedBytes }
}

/// An explicitly shared, fail-fast budget. No eviction, hidden singleton or scheduling policy.
/// All ledger changes are synchronous transactions; none suspend while capacity is reserved.
public final class ResourceCoordinator: Sendable {
    private struct Ledger: Sendable {
        var entries: [UUID: UInt64] = [:]
        var total: UInt64 = 0
    }
    public let capacityBytes: UInt64
    private let ledger = Mutex(Ledger())

    public init(capacityBytes: UInt64) throws {
        guard capacityBytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        self.capacityBytes = capacityBytes
    }

    public var snapshot: ResourceSnapshot {
        ledger.withLock { ResourceSnapshot(capacityBytes: capacityBytes, reservedBytes: $0.total, reservationCount: $0.entries.count) }
    }

    public func reserve(bytes: UInt64) throws -> ResourceReservation {
        guard bytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        let id = UUID()
        try ledger.withLock { state in
            let available = capacityBytes - state.total
            guard bytes <= available else {
                throw ResourceAdmissionError.insufficientCapacity(requested: bytes, available: available)
            }
            state.entries[id] = bytes
            state.total += bytes
        }
        return ResourceReservation(coordinator: self, id: id)
    }

    fileprivate func resize(_ id: UUID, to bytes: UInt64) throws {
        guard bytes > 0 else { throw ResourceAdmissionError.invalidByteCount }
        try ledger.withLock { state in
            guard let old = state.entries[id] else { throw ResourceAdmissionError.releasedReservation }
            let available = capacityBytes - (state.total - old)
            guard bytes <= available else {
                throw ResourceAdmissionError.insufficientCapacity(requested: bytes, available: available)
            }
            state.entries[id] = bytes
            state.total = state.total - old + bytes
        }
    }

    fileprivate func release(_ id: UUID) {
        ledger.withLock { state in
            if let bytes = state.entries.removeValue(forKey: id) { state.total -= bytes }
        }
    }
}

/// Keep this lease alive until all work using its allocations has joined. Grow before allocating;
/// shrink only after releasing the old storage. Releasing is idempotent and also happens on deinit.
/// Replacement allocations require a reservation for the old and new storage's overlapping lifetime.
public final class ResourceReservation: Sendable {
    private let coordinator: ResourceCoordinator
    private let id: UUID
    fileprivate init(coordinator: ResourceCoordinator, id: UUID) { self.coordinator = coordinator; self.id = id }
    public func resize(to bytes: UInt64) throws { try coordinator.resize(id, to: bytes) }
    public func release() { coordinator.release(id) }
    deinit { coordinator.release(id) }
}
