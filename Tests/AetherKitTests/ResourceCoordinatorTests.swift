import AetherKit
import Testing

@Test func reservationsRefuseGrowthWithoutChangingLedger() throws {
    let budget = try ResourceCoordinator(capacityBytes: 100)
    let first = try budget.reserve(bytes: 40)
    let second = try budget.reserve(bytes: 50)
    #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: 60, available: 50)) { try first.resize(to: 60) }
    #expect(budget.snapshot.reservedBytes == 90)
    second.release()
    try first.resize(to: 100)
    #expect(budget.snapshot.availableBytes == 0)
    #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: 1, available: 0)) { try budget.reserve(bytes: 1) }
    first.release(); first.release()
    #expect(budget.snapshot.reservedBytes == 0)
    #expect(budget.snapshot.reservationCount == 0)
    #expect(throws: ResourceAdmissionError.releasedReservation) { try first.resize(to: 1) }
}

@Test func reservationsRejectZeroAndDoNotOverflow() throws {
    #expect(throws: ResourceAdmissionError.invalidByteCount) { try ResourceCoordinator(capacityBytes: 0) }
    let budget = try ResourceCoordinator(capacityBytes: UInt64.max)
    #expect(throws: ResourceAdmissionError.invalidByteCount) { try budget.reserve(bytes: 0) }
    let lease = try budget.reserve(bytes: UInt64.max)
    #expect(throws: ResourceAdmissionError.insufficientCapacity(requested: UInt64.max, available: 0)) {
        try budget.reserve(bytes: UInt64.max)
    }
    try lease.resize(to: 1)
    #expect(budget.snapshot.reservedBytes == 1)
    lease.release()
}

@Test func reservationDeinitReturnsCapacity() throws {
    let budget = try ResourceCoordinator(capacityBytes: 1)
    do {
        let lease = try budget.reserve(bytes: 1)
        withExtendedLifetime(lease) { #expect(budget.snapshot.reservedBytes == 1) }
    }
    #expect(budget.snapshot.reservedBytes == 0)
}

@Test func competingReservationsNeverOvercommit() async throws {
    let budget = try ResourceCoordinator(capacityBytes: 10)
    let leases = await withTaskGroup(of: ResourceReservation?.self, returning: [ResourceReservation].self) { group in
        for _ in 0..<100 { group.addTask { try? budget.reserve(bytes: 1) } }
        var result: [ResourceReservation] = []
        for await lease in group { if let lease { result.append(lease) } }
        return result
    }
    #expect(leases.count == 10)
    #expect(budget.snapshot.reservedBytes == 10)
    for lease in leases { lease.release() }
    #expect(budget.snapshot.reservedBytes == 0)
}
