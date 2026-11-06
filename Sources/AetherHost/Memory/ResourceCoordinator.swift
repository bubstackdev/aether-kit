public import AetherCore
public import Foundation
import Synchronization

/// An explicitly shared, fail-fast memory budget (ported from AetherKit 0.0.7). A host creates one and passes it to
/// every pipeline that should share the capacity; there is no process-global default, eviction or scheduling policy.
///
/// Generative pipelines reserve their measured peak (``ResidencyPlan``'s `peakBytes`, from AetherRuntime) before
/// anything loads, and per stage as they go, so a run that cannot fit is refused before it allocates. Reservations are
/// admission estimates, not an allocation guarantee from the OS. Every change is one synchronous transaction under a
/// `Mutex`; nothing suspends while capacity is reserved.
///
/// Refusals: ``AetherError/capacity(what:needed:available:)`` when the bytes don't fit, and
/// ``AetherError/admission(_:)`` (`invalidByteCount`, `reservationReleased`) for a misuse.
public final class ResourceCoordinator: Sendable {
    public struct Snapshot: Sendable, Hashable {
        public let capacityBytes: UInt64
        public let reservedBytes: UInt64
        public let reservationCount: Int
        public var availableBytes: UInt64 { capacityBytes - reservedBytes }
    }

    private struct Ledger: Sendable {
        var entries: [UUID: UInt64] = [:]
        var total: UInt64 = 0
    }

    public let capacityBytes: UInt64
    private let ledger = Mutex(Ledger())

    public init(capacityBytes: UInt64) throws(AetherError) {
        guard capacityBytes > 0 else { throw .admission(.invalidByteCount) }
        self.capacityBytes = capacityBytes
    }

    public var snapshot: Snapshot {
        ledger.withLock { Snapshot(capacityBytes: capacityBytes, reservedBytes: $0.total, reservationCount: $0.entries.count) }
    }

    /// Reserves `bytes`, or refuses without changing anything.
    public func reserve(bytes: UInt64) throws(AetherError) -> ResourceReservation {
        guard bytes > 0 else { throw .admission(.invalidByteCount) }
        let id = UUID()
        try ledger.withLock { state throws(AetherError) in
            let available = capacityBytes - state.total
            guard bytes <= available else { throw Self.refusal(bytes, available) }
            state.entries[id] = bytes
            state.total += bytes
        }
        return ResourceReservation(coordinator: self, id: id)
    }

    fileprivate func resize(_ id: UUID, to bytes: UInt64) throws(AetherError) {
        guard bytes > 0 else { throw .admission(.invalidByteCount) }
        try ledger.withLock { state throws(AetherError) in
            guard let old = state.entries[id] else { throw .admission(.reservationReleased) }
            let available = capacityBytes - (state.total - old)
            guard bytes <= available else { throw Self.refusal(bytes, available) }
            state.entries[id] = bytes
            state.total = state.total - old + bytes
        }
    }

    fileprivate func release(_ id: UUID) {
        ledger.withLock { state in
            if let bytes = state.entries.removeValue(forKey: id) { state.total -= bytes }
        }
    }

    private static func refusal(_ bytes: UInt64, _ available: UInt64) -> AetherError {
        .capacity(what: "reserved memory", needed: Int64(clamping: bytes), available: Int64(clamping: available))
    }
}

/// Reserved capacity. Keep it alive until all work using the allocations has joined. Grow before allocating; shrink
/// only after the old storage is released (a replacement needs both reserved while they overlap). Releasing is
/// idempotent and also happens on deinit.
public final class ResourceReservation: Sendable {
    private let coordinator: ResourceCoordinator
    private let id: UUID

    fileprivate init(coordinator: ResourceCoordinator, id: UUID) {
        self.coordinator = coordinator
        self.id = id
    }

    /// Changes the reservation to `bytes`, or refuses without changing anything.
    public func resize(to bytes: UInt64) throws(AetherError) { try coordinator.resize(id, to: bytes) }

    public func release() { coordinator.release(id) }

    deinit { coordinator.release(id) }
}
