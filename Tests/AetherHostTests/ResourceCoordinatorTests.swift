import AetherCore
import Testing
@testable import AetherHost

// 0.0.x's ResourceCoordinatorTests, on AetherError.

private func refusal(_ error: AetherError) -> (needed: Int64, available: Int64)? {
    if case .capacity(_, let needed, let available) = error { (needed, available) } else { nil }
}

private func expectRefused(_ needed: Int64, _ available: Int64, _ body: () throws(AetherError) -> Void) {
    do {
        try body()
        Issue.record("expected a capacity refusal")
    } catch {
        #expect(refusal(error)! == (needed, available))
    }
}

@Test func reservationsRefuseGrowthWithoutChangingTheLedger() throws {
    let budget = try ResourceCoordinator(capacityBytes: 100)
    let first = try budget.reserve(bytes: 40)
    let second = try budget.reserve(bytes: 50)
    expectRefused(60, 50) { () throws(AetherError) in try first.resize(to: 60) }
    #expect(budget.snapshot.reservedBytes == 90)
    second.release()
    try first.resize(to: 100)
    #expect(budget.snapshot.availableBytes == 0)
    expectRefused(1, 0) { () throws(AetherError) in _ = try budget.reserve(bytes: 1) }
    first.release(); first.release()
    #expect(budget.snapshot.reservedBytes == 0 && budget.snapshot.reservationCount == 0)
    #expect { try first.resize(to: 1) } throws: { ($0 as? AetherError).map { "\($0)" } == "\(AetherError.admission(.reservationReleased))" }
}

@Test func reservationsRejectZeroAndDoNotOverflow() throws {
    #expect(throws: AetherError.self) { try ResourceCoordinator(capacityBytes: 0) }
    let budget = try ResourceCoordinator(capacityBytes: .max)
    #expect(throws: AetherError.self) { try budget.reserve(bytes: 0) }
    let lease = try budget.reserve(bytes: .max)
    expectRefused(.max, 0) { () throws(AetherError) in _ = try budget.reserve(bytes: .max) }
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
    #expect(leases.count == 10 && budget.snapshot.reservedBytes == 10)
    for lease in leases { lease.release() }
    #expect(budget.snapshot.reservedBytes == 0)
}
