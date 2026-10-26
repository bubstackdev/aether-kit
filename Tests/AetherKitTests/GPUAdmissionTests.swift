import AetherKit
import Darwin
import Foundation
import Synchronization
import Testing

private final class Order: Sendable {
    private let log = Mutex<[String]>([])
    func append(_ value: String) { log.withLock { $0.append(value) } }
    var values: [String] { log.withLock { $0 } }
}

/// Polls until `condition` holds, so tests do not depend on scheduling order.
private func eventually(_ condition: () -> Bool) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(10))
    while !condition() {
        guard ContinuousClock.now < deadline else { Issue.record("condition never held"); return }
        try await Task.sleep(for: .milliseconds(5))
    }
}

private func isBusy(_ error: any Error) -> Bool { (error as? GPUAdmissionError) == .busy }

@Test func queuedGrantsAreFairAndSerialized() async throws {
    let admission = try GPUAdmission()
    let first = try await admission.admit(owner: "a")
    let order = Order()
    var waiters: [Task<Void, any Error>] = []
    for (owner, label) in [("a", "a1"), ("a", "a2"), ("b", "b1")] {
        let queued = admission.snapshot.queued
        waiters.append(Task {
            let grant = try await admission.admit(owner: owner)
            order.append(label)
            try await Task.sleep(for: .milliseconds(10))
            grant.release()
        })
        try await eventually { admission.snapshot.queued == queued + 1 }
    }
    #expect(admission.snapshot == .init(running: 1, queued: 3, holdsCrossProcessLease: false))
    first.release()
    for waiter in waiters { try await waiter.value }
    // "a" just ran, so the new owner "b" goes before a's queued burst.
    #expect(order.values == ["b1", "a1", "a2"])
    #expect(admission.snapshot == .init(running: 0, queued: 0, holdsCrossProcessLease: false))
}

@Test func failFastRefusesWhileBusyOrWhenWorkIsQueuedAhead() async throws {
    let admission = try GPUAdmission(waiting: .failFast)
    let held = try await admission.admit(owner: "a")
    await #expect { try await admission.admit(owner: "b") } throws: { isBusy($0) }
    let queued = Task { try await admission.admit(owner: "c", waiting: .queue()) }
    try await eventually { admission.snapshot.queued == 1 }
    held.release()
    let next = try await queued.value
    #expect(next.owner == "c")
    next.release()
    let again = try await admission.admit(owner: "b")
    again.release()
    held.release()
    #expect(admission.snapshot.running == 0)
}

@Test func queuedAdmissionTimesOutAndLeavesTheQueue() async throws {
    let admission = try GPUAdmission(waiting: .queue(timeout: .milliseconds(100)))
    let held = try await admission.admit(owner: "a")
    await #expect(throws: GPUAdmissionError.timedOut) { try await admission.admit(owner: "b") }
    #expect(admission.snapshot.queued == 0)
    held.release()
    let next = try await admission.admit(owner: "b")
    next.release()
}

@Test func cancellingAQueuedAdmissionLeavesTheQueue() async throws {
    let admission = try GPUAdmission()
    let held = try await admission.admit(owner: "a")
    let waiter = Task { try await admission.admit(owner: "b") }
    try await eventually { admission.snapshot.queued == 1 }
    waiter.cancel()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    #expect(admission.snapshot == .init(running: 1, queued: 0, holdsCrossProcessLease: false))
    // A task cancelled before it asks is refused without queueing.
    let early = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await admission.admit(owner: "c")
    }
    await #expect(throws: CancellationError.self) { try await early.value }
    held.release()
    #expect(admission.snapshot == .init(running: 0, queued: 0, holdsCrossProcessLease: false))
}

@Test func concurrentSlotsAndIdempotentRelease() async throws {
    let admission = try GPUAdmission(maximumConcurrent: 2, waiting: .failFast)
    let one = try await admission.admit(owner: "a")
    let two = try await admission.admit(owner: "a")
    await #expect { try await admission.admit(owner: "b") } throws: { isBusy($0) }
    one.release(); one.release()
    #expect(admission.snapshot.running == 1)
    do {
        let three = try await admission.admit(owner: "b")
        #expect(admission.snapshot.running == 2)
        _ = three
    }
    // Deinit released the third grant.
    #expect(admission.snapshot.running == 1)
    two.release()
    #expect(admission.snapshot.running == 0)
}

@Test func limitsAndConfigurationMapToAdmissionErrors() async throws {
    #expect(throws: GPUAdmissionError.limit(.invalidLimits)) { try GPUAdmission(maximumConcurrent: 0) }
    let admission = try GPUAdmission(maximumOutstandingPerOwner: 1)
    let held = try await admission.admit(owner: "a")
    await #expect(throws: GPUAdmissionError.limit(.ownerLimit)) { try await admission.admit(owner: "a") }
    held.release()
}

@Test func ownerHandlesShareOneAdmission() async throws {
    let admission = try GPUAdmission(waiting: .failFast)
    let krea = admission.owner("krea"), language = admission.owner("language")
    let grant = try await krea.admit()
    #expect(grant.owner == "krea")
    await #expect { try await language.admit() } throws: { isBusy($0) }
    grant.release()
    try await language.admit().release()
}

// MARK: Cross-process backend, on private lease directories only.

@Test func theCrossProcessLeaseIsHeldWhileAnyGrantRuns() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let admission = try GPUAdmission(maximumConcurrent: 2, crossProcessLease: leaseConfiguration(directory))
    let one = try await admission.admit(owner: "a")
    #expect(pidText(directory) == String(getpid()) && admission.snapshot.holdsCrossProcessLease)
    let two = try await admission.admit(owner: "b")
    one.release()
    #expect(FileManager.default.fileExists(atPath: directory.path))
    two.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path) && !admission.snapshot.holdsCrossProcessLease)
    let failFast = try await admission.admit(owner: "c", waiting: .failFast)
    #expect(pidText(directory) == String(getpid()))
    failFast.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func anotherProcessHoldingTheLeaseBlocksAdmission() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    // Another holder, as Atelier or a wrapper would leave it: this test process's parent is alive.
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try Data(String(getppid()).utf8).write(to: directory.appendingPathComponent("pid"))
    let admission = try GPUAdmission(maximumConcurrent: 2, crossProcessLease: leaseConfiguration(directory))
    await #expect(throws: SharedGPULeaseError.held(holder: getppid())) { try await admission.admit(owner: "a", waiting: .failFast) }
    await #expect(throws: SharedGPULeaseError.timedOut(holder: getppid())) {
        try await admission.admit(owner: "a", waiting: .queue(timeout: .milliseconds(100)))
    }
    #expect(admission.snapshot == .init(running: 0, queued: 0, holdsCrossProcessLease: false))
    // Two queued requests: one waits on the lease, the other waits in the queue, and both start once it frees.
    let first = Task { try await admission.admit(owner: "a") }
    let second = Task { try await admission.admit(owner: "b") }
    try await eventually { admission.snapshot.running == 1 && admission.snapshot.queued == 1 }
    await #expect { try await admission.admit(owner: "c", waiting: .failFast) } throws: { isBusy($0) }
    try FileManager.default.removeItem(at: directory)
    let grants = [try await first.value, try await second.value]
    #expect(pidText(directory) == String(getpid()) && admission.snapshot.running == 2)
    grants.forEach { $0.release() }
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}

@Test func cancellingTheLeaseWaiterHandsTheTurnToTheNextRequest() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try Data(String(getppid()).utf8).write(to: directory.appendingPathComponent("pid"))
    let admission = try GPUAdmission(crossProcessLease: leaseConfiguration(directory))
    let first = Task { try await admission.admit(owner: "a") }
    try await eventually { admission.snapshot.running == 1 }
    let second = Task { try await admission.admit(owner: "b") }
    try await eventually { admission.snapshot.queued == 1 }
    first.cancel()
    await #expect(throws: CancellationError.self) { try await first.value }
    try await eventually { admission.snapshot.running == 1 && admission.snapshot.queued == 0 }
    try FileManager.default.removeItem(at: directory)
    let grant = try await second.value
    #expect(grant.owner == "b" && pidText(directory) == String(getpid()))
    grant.release()
}

@Test func anInheritedLeaseIsRecheckedBeforeEveryGrant() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    // A wrapper-like ancestor (this test's parent) holds the lease and hands it to this process.
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try Data(String(getppid()).utf8).write(to: directory.appendingPathComponent("pid"))
    let admission = try GPUAdmission(maximumConcurrent: 2,
                                     crossProcessLease: leaseConfiguration(directory, handedOffBy: getppid()))
    let first = try await admission.admit(owner: "a", waiting: .failFast)
    #expect(admission.snapshot.holdsCrossProcessLease && pidText(directory) == String(getppid()))
    // The holder's lease is replaced by an unrelated live holder: the next grant must not ride on it.
    try Data("1".utf8).write(to: directory.appendingPathComponent("pid"), options: .atomic)
    await #expect(throws: SharedGPULeaseError.held(holder: 1)) { try await admission.admit(owner: "b", waiting: .failFast) }
    first.release()
    // Once the holder is gone, a grant takes a lease of its own.
    try FileManager.default.removeItem(at: directory)
    let owned = try await admission.admit(owner: "b")
    #expect(pidText(directory) == String(getpid()))
    owned.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}

@Test func anOwnedLeasePassesStraightToTheNextRequestWithinItsTurn() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let admission = try GPUAdmission(crossProcessLease: leaseConfiguration(directory), crossProcessTurn: .seconds(60))
    let first = try await admission.admit(owner: "a")
    let next = Task { try await admission.admit(owner: "b") }
    try await eventually { admission.snapshot.queued == 1 }
    let watcher = LeaseWatcher(directory)
    first.release()
    let grant = try await next.value
    #expect(watcher.stop() == 0) // The lease directory never disappeared between the two grants.
    grant.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}

@Test func afterItsTurnTheLeaseIsReleasedWithAGapAnotherProcessCanUse() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let admission = try GPUAdmission(crossProcessLease: leaseConfiguration(directory), crossProcessTurn: .zero,
                                     crossProcessHandOffGap: .seconds(2))
    let first = try await admission.admit(owner: "a")
    let next = Task { try await admission.admit(owner: "b") }
    try await eventually { admission.snapshot.queued == 1 }
    first.release()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    // An outside process takes the lease inside the gap; the queued request then waits for it.
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try Data(String(getppid()).utf8).write(to: directory.appendingPathComponent("pid"))
    try await Task.sleep(for: .milliseconds(2300))
    #expect(admission.snapshot.running == 1 && pidText(directory) == String(getppid()))
    try FileManager.default.removeItem(at: directory)
    let grant = try await next.value
    #expect(pidText(directory) == String(getpid()))
    grant.release()
}

/// Counts moments the lease directory is missing, sampled from a background thread.
private final class LeaseWatcher: @unchecked Sendable {
    private let missing = Mutex(0)
    private let running = Mutex(true)
    private let finished = DispatchSemaphore(value: 0)
    private let firstMissingAt = Mutex<ContinuousClock.Instant?>(nil)
    init(_ directory: URL) {
        Thread.detachNewThread { [self] in
            while running.withLock({ $0 }) {
                if !FileManager.default.fileExists(atPath: directory.path) {
                    missing.withLock { $0 += 1 }
                    firstMissingAt.withLock { if $0 == nil { $0 = .now } }
                }
                usleep(200)
            }
            finished.signal()
        }
    }
    /// When the directory was first seen missing.
    var firstMissing: ContinuousClock.Instant? { firstMissingAt.withLock { $0 } }
    func stop() -> Int {
        running.withLock { $0 = false }
        finished.wait()
        return missing.withLock { $0 }
    }
}

@Test func aWithdrawnAttemptStopsWaitingAndNeverKeepsAGrant() async throws {
    let admission = try GPUAdmission()
    let held = try await admission.admit(owner: "a")
    struct Closed: Error {}
    let waiting = GPUAdmissionAttempt(admission.owner("b"))
    let task = Task { try await waiting.admit(orThrow: { Closed() }) }
    try await eventually { admission.snapshot.queued == 1 }
    waiting.withdraw()
    await #expect(throws: Closed.self) { try await task.value }
    #expect(admission.snapshot == .init(running: 1, queued: 0, holdsCrossProcessLease: false))
    // Withdrawn before asking: refused without queueing.
    let early = GPUAdmissionAttempt(admission.owner("c"))
    early.withdraw()
    await #expect(throws: Closed.self) { try await early.admit(orThrow: { Closed() }) }
    held.release()
    // Withdrawing after the grant has no effect.
    let granted = GPUAdmissionAttempt(admission.owner("d"))
    let grant = try await granted.admit(orThrow: { Closed() })
    granted.withdraw()
    #expect(admission.snapshot.running == 1)
    grant.release()
    #expect(admission.snapshot.running == 0)
}

@Test(.timeLimit(.minutes(1))) func overlappingGrantsStillHandTheLeaseOffAfterTheTurn() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let turn = Duration.milliseconds(300), hold = Duration.milliseconds(120)
    let admission = try GPUAdmission(maximumConcurrent: 2, crossProcessLease: leaseConfiguration(directory),
                                     crossProcessTurn: turn, crossProcessHandOffGap: .milliseconds(400))
    let first = try await admission.admit(owner: "a")
    let start = ContinuousClock.now
    let watcher = LeaseWatcher(directory)
    // Two slots and a steady stream of overlapping grants: some grant is always running, so without draining the
    // lease would never be released.
    let workers = (0..<16).map { index in
        Task {
            try await Task.sleep(for: .milliseconds(40 * index))
            let grant = try await admission.admit(owner: "w\(index % 2)")
            try await Task.sleep(for: hold)
            grant.release()
        }
    }
    try await Task.sleep(for: hold)
    first.release()
    for worker in workers { try await worker.value }
    let released = try #require(watcher.firstMissing)
    _ = watcher.stop()
    // Released within the turn plus one grant's duration (with scheduling slack), while grants were still queued.
    #expect(start.duration(to: released) <= turn + hold + .milliseconds(250))
    #expect(admission.snapshot == .init(running: 0, queued: 0, holdsCrossProcessLease: false))
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}
