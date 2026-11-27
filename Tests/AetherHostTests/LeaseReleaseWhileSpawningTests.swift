@testable import AetherHost
import Darwin
import Foundation
import Synchronization
import Testing

// Atelier lab #160: a process being spawned holds a copy of every descriptor of this one until its program loads
// (O_CLOEXEC ones included), and a flock lock goes only with the last copy. So a lock given back with `close` alone
// could still be held an instant later: a holder's probe saw a waiter that had gone, a second reclaimer a reclaim
// that had ended. Each case gives a lock back hundreds of times while other threads spawn processes and checks it is
// gone at once every time. All three failed on `close` alone.

private let cycles = 2_000

/// Threads that keep spawning a short-lived process until `stop()`.
private final class Spawner: Sendable {
    private let running = Atomic(true)
    private let threads = Atomic(0)
    init(threads count: Int = 4) {
        for _ in 0..<count {
            threads.add(1, ordering: .relaxed)
            Thread.detachNewThread { [self] in
                while running.load(ordering: .relaxed) {
                    let process = Process()
                    process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
                    try? process.run()
                    process.waitUntilExit()
                }
                threads.subtract(1, ordering: .relaxed)
            }
        }
    }
    func stop() {
        running.store(false, ordering: .relaxed)
        while threads.load(ordering: .relaxed) > 0 { usleep(1_000) }
    }
}

/// Whether anyone holds a lock on `path`: an exclusive non-blocking flock, given back at once.
private func held(_ path: String) -> Bool {
    let descriptor = open(path, O_RDONLY | O_CLOEXEC)
    guard descriptor >= 0 else { return false }
    defer { unlockAndClose(descriptor) }
    return flock(descriptor, LOCK_EX | LOCK_NB) != 0
}

private func sibling(_ directory: URL, _ suffix: String) -> String {
    directory.deletingLastPathComponent().appendingPathComponent(directory.lastPathComponent + suffix).path
}

@Test func aFreedWaiterMarkIsGoneAtOnceWhileProcessesAreSpawned() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let spawner = Spawner(); defer { spawner.stop() }
    var stillHeld = 0
    for _ in 0..<cycles {
        var mark: WaiterMark? = WaiterMark(directory: directory)
        withExtendedLifetime(mark) {}
        mark = nil
        if held(sibling(directory, ".waiters")) { stillHeld += 1 }
    }
    #expect(stillHeld == 0)
}

@Test func aReclaimersSerialLockIsGoneAtOnceWhileProcessesAreSpawned() throws {
    // No lease: each call takes the serial lock, finds nothing to reclaim and gives it back.
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let spawner = Spawner(); defer { spawner.stop() }
    var stillHeld = 0
    for _ in 0..<cycles {
        #expect(!SharedGPULease.clearStale(directory))
        if held(sibling(directory, ".reclaim")) { stillHeld += 1 }
    }
    #expect(stillHeld == 0)
}

@Test func theWrapperProbeLeavesItsLockFreeAtOnceWhileProcessesAreSpawned() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let wrapper = directory.deletingLastPathComponent().appendingPathComponent("wrapper.lock")
    #expect(FileManager.default.createFile(atPath: wrapper.path, contents: nil))
    let spawner = Spawner(); defer { spawner.stop() }
    var stillHeld = 0
    for _ in 0..<cycles {
        #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: wrapper))
        if held(wrapper.path) { stillHeld += 1 }
    }
    #expect(stillHeld == 0)
}
