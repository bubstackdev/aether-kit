@testable import AetherHost
import Darwin
import Foundation
import Synchronization
import Testing

// Ported from AetherKit 0.0.7. Every case uses its own temporary lock directory; none may touch the shared
// /tmp/atelier-gpu.lock. The protocol cases mirror Atelier's SharedGPULockTests so both implementations are held to the
// same contract, and the cross-process cases run a helper process and a wrapper-style shell holder.

func privateLeaseDirectory() throws -> URL {
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent("aether-lease-test-\(UUID())", isDirectory: true)
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    let directory = parent.appendingPathComponent("gpu.lock", isDirectory: true)
    #expect(directory.path != SharedGPULease.sharedDirectory.path)
    return directory
}

func removeLeaseParent(_ directory: URL) { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }

func leaseConfiguration(_ directory: URL, handedOffBy: Int32? = nil) -> SharedGPULease.Configuration {
    SharedGPULease.Configuration(directory: directory, handedOffBy: handedOffBy, pollInterval: .milliseconds(20))
}

/// An externally created lease, as another implementation would leave it.
private func externalLease(_ directory: URL, holder: String?) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    if let holder { try Data(holder.utf8).write(to: directory.appendingPathComponent("pid")) }
}

func pidText(_ directory: URL) -> String? {
    try? String(contentsOf: directory.appendingPathComponent("pid"), encoding: .utf8)
}

/// Entries beside the lock, ignoring the reclaimers' persistent `.reclaim` flock file.
func leaseSiblings(_ directory: URL) throws -> [String] {
    try FileManager.default.contentsOfDirectory(atPath: directory.deletingLastPathComponent().path)
        // The reclaim and waiters files persist by design (deleting one a process still holds would split it).
        .filter { !$0.hasSuffix(".reclaim") && !$0.hasSuffix(".waiters") }.sorted()
}

/// The admission failure an error carries, if any.
func admissionFailure(_ error: any Error) -> AetherError.AdmissionFailure? {
    if case AetherError.admission(let failure) = error { return failure }
    return nil
}

private func isHeld(_ error: any Error, by holder: Int32? = nil) -> Bool {
    guard case .leaseHeld(let found)? = admissionFailure(error) else { return false }
    return holder == nil || found == holder
}

/// A pid that existed and has been reaped, so signal 0 reports ESRCH.
func deadPID() throws -> pid_t {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/true")
    try process.run()
    process.waitUntilExit()
    let pid = process.processIdentifier
    #expect(kill(pid, 0) != 0 && errno == ESRCH)
    return pid
}

private let ownPID = String(getpid())

/// A reclaim lock this process just released can stay held for an instant: a child that another test is spawning
/// shares the open descriptor between fork and exec. Reclaim fails safe (reports the lease held) in that window, so
/// assertions that expect success after a release retry briefly.
private func clearsStaleSoon(_ directory: URL) -> Bool {
    let deadline = Date().addingTimeInterval(5)
    repeat {
        if SharedGPULease.clearStale(directory) { return true }
        usleep(10_000)
    } while Date() < deadline
    return false
}

@Test func leaseNamesThisProcessAndReleaseRemovesIt() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let first = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(first.holding == .owned && first.isHeld)
    #expect(pidText(directory) == ownPID)
    #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["pid"])
    // A live holder, this process included, is never reclaimed.
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0, by: getpid()) }
    #expect(pidText(directory) == ownPID)
    first.release()
    #expect(!first.isHeld)
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    first.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func deinitReleasesTheLease() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    do {
        let lease = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
        #expect(SharedGPULease.holder(of: directory) == getpid())
        _ = lease
    }
    #expect(!FileManager.default.fileExists(atPath: directory.path))
}

@Test func liveOrUnsignalableHoldersAreNeverReclaimed() throws {
    // launchd is alive and not signalable by this user (EPERM), which is never treated as stale.
    for value in [ownPID, "1", "\(getppid())\n"] {
        let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
        try externalLease(directory, holder: value)
        #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0) }
        #expect(!SharedGPULease.clearStale(directory))
        #expect(pidText(directory) == value)
        #expect(try leaseSiblings(directory) == ["gpu.lock"])
    }
}

@Test func leasesWithoutAUsablePIDAreNeverReclaimed() throws {
    for value in [nil, "", "not-a-pid", "0", "-4", "99999999999"] as [String?] {
        let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
        try externalLease(directory, holder: value)
        #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { error in
            admissionFailure(error) == .leaseHeld(holder: nil)
        }
        #expect(pidText(directory) == value)
        #expect(try leaseSiblings(directory) == ["gpu.lock"])
    }
}

@Test func aLeaseNamingADeadProcessIsReclaimedAndRenamed() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try externalLease(directory, holder: String(try deadPID()) + "\n")
    let reclaimed = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(pidText(directory) == ownPID)
    #expect(try leaseSiblings(directory) == ["gpu.lock"])
    reclaimed.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func reclaimRestoresALeaseThatChangedHolderBeforeTheMove() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try externalLease(directory, holder: String(try deadPID()))
    // Another host replaced the stale lease's pid between this host's read and its move aside.
    let cleared = SharedGPULease.clearStale(directory) {
        try? Data(ownPID.utf8).write(to: directory.appendingPathComponent("pid"), options: .atomic)
    }
    #expect(!cleared)
    #expect(pidText(directory) == ownPID)
    #expect(try leaseSiblings(directory) == ["gpu.lock"])
}

@Test func reclaimNeverReplacesALeaseCreatedSinceByAnOutsideRemover() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try externalLease(directory, holder: String(try deadPID()))
    // An rm-based remover (the lab's gpu.sh) deletes the stale lease and a new holder takes it before the move.
    let cleared = SharedGPULease.clearStale(directory) {
        try? FileManager.default.removeItem(at: directory)
        try? externalLease(directory, holder: ownPID)
    }
    #expect(!cleared)
    #expect(pidText(directory) == ownPID)
    #expect(try leaseSiblings(directory) == ["gpu.lock"])
    // The newer lease is still a live lease to every acquirer.
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0, by: getpid()) }
}

@Test func aConcurrentReclaimerHoldsOffASecondOne() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let dead = String(try deadPID())
    try externalLease(directory, holder: dead)
    let reclaim = directory.deletingLastPathComponent().appendingPathComponent("gpu.lock.reclaim").path
    // Close-on-exec, so a child process another test spawns meanwhile cannot inherit and keep holding this flock.
    let other = open(reclaim, O_RDONLY | O_CREAT | O_CLOEXEC, 0o600)
    try #require(other >= 0 && flock(other, LOCK_EX) == 0)
    #expect(!SharedGPULease.clearStale(directory))
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0) }
    #expect(pidText(directory) == dead)
    close(other)
    #expect(clearsStaleSoon(directory))
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func aReclaimFileThatIsNotThisUsersRegularFileIsNeverUsed() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let dead = String(try deadPID())
    try externalLease(directory, holder: dead)
    let reclaim = directory.deletingLastPathComponent().appendingPathComponent("gpu.lock.reclaim")
    // A symlink planted in its place is never followed.
    try FileManager.default.createSymbolicLink(atPath: reclaim.path, withDestinationPath: "/dev/null")
    #expect(!SharedGPULease.clearStale(directory))
    #expect(pidText(directory) == dead)
    // Nor is anything but a regular file.
    try FileManager.default.removeItem(at: reclaim)
    try FileManager.default.createDirectory(at: reclaim, withIntermediateDirectories: false)
    #expect(!SharedGPULease.clearStale(directory))
    #expect(pidText(directory) == dead)
    // Nor a hard link to another file: it is left untouched, not tightened or locked.
    try FileManager.default.removeItem(at: reclaim)
    let target = directory.deletingLastPathComponent().appendingPathComponent("own-file")
    try Data().write(to: target)
    try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: target.path)
    try #require(link(target.path, reclaim.path) == 0)
    #expect(!SharedGPULease.clearStale(directory))
    #expect(pidText(directory) == dead)
    let targetMode = try #require(try FileManager.default.attributesOfItem(atPath: target.path)[.posixPermissions] as? Int)
    #expect(targetMode & 0o777 == 0o644)
    try FileManager.default.removeItem(at: target)
    // This user's private file is used; a fresh one is created 0600.
    try FileManager.default.removeItem(at: reclaim)
    #expect(SharedGPULease.clearStale(directory))
    let mode = try #require(try FileManager.default.attributesOfItem(atPath: reclaim.path)[.posixPermissions] as? Int)
    #expect(mode & 0o077 == 0)
}

@Test func thisUsersOwnOpenReclaimFileIsMadePrivateAndUsed() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let dead = String(try deadPID())
    try externalLease(directory, holder: dead)
    let reclaim = directory.deletingLastPathComponent().appendingPathComponent("gpu.lock.reclaim")
    func mode() throws -> Int { try #require(try FileManager.default.attributesOfItem(atPath: reclaim.path)[.posixPermissions] as? Int) }
    try Data().write(to: reclaim)
    try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: reclaim.path)
    // Repair never bypasses serialization: a reclaimer holding the old file still holds off this one.
    let other = open(reclaim.path, O_RDONLY | O_CLOEXEC)
    try #require(other >= 0 && flock(other, LOCK_EX) == 0)
    #expect(!SharedGPULease.clearStale(directory))
    #expect(pidText(directory) == dead)
    #expect(try mode() & 0o777 == 0o600)
    close(other)
    try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: reclaim.path)
    #expect(clearsStaleSoon(directory))
    #expect(try leaseSiblings(directory).isEmpty)
    #expect(try mode() & 0o777 == 0o600)
}

@Test func releaseLeavesALeaseThatNoLongerNamesThisProcess() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let lease = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    try Data("424242".utf8).write(to: directory.appendingPathComponent("pid"), options: .atomic)
    lease.release()
    #expect(pidText(directory) == "424242")
}

@Test func aMissingParentDirectoryIsUnavailableNotHeld() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let nested = directory.appendingPathComponent("missing/gpu.lock", isDirectory: true)
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(nested)) } throws: { error in
        guard case .leaseUnavailable(_, ENOENT)? = admissionFailure(error) else { return false }
        return true
    }
}

@Test func waitingAcquisitionTimesOutAndHonorsCancellation() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    let clock = ContinuousClock(), start = clock.now
    await #expect { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .milliseconds(150)) } throws: { error in
        admissionFailure(error) == .leaseTimedOut(holder: getpid())
    }
    #expect(start.duration(to: clock.now) >= .milliseconds(140))
    // Zero timeout is one attempt.
    await #expect { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .zero) } throws: { error in
        if case .leaseTimedOut? = admissionFailure(error) { return true }
        return false
    }
    let waiter = Task { try await SharedGPULease.acquire(leaseConfiguration(directory)) }
    try await Task.sleep(for: .milliseconds(80))
    waiter.cancel()
    await #expect(throws: CancellationError.self) { try await waiter.value }
    #expect(pidText(directory) == ownPID)
    held.release()
}

@Test func waitingAcquisitionTakesTheLeaseOnceReleasedAndReportsTheHolder() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    let reported = BusyLog()
    let waiter = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), onBusy: { reported.append($0) }) }
    // Release only once the waiter has reported the holder, however slowly it starts.
    let deadline = Date().addingTimeInterval(30)
    while reported.values.isEmpty, Date() < deadline { usleep(5_000) }
    held.release()
    let lease = try await waiter.value
    #expect(lease.holding == .owned && pidText(directory) == ownPID)
    #expect(reported.values == [.process(getpid())])
    lease.release()
}

final class BusyLog: Sendable {
    private let log = Mutex<[SharedGPULease.Busy]>([])
    func append(_ value: SharedGPULease.Busy) { log.withLock { $0.append(value) } }
    var values: [SharedGPULease.Busy] { log.withLock { $0 } }
}

@Test func busyReportsNameAnAncestorWithoutHandOffAndAPidLessLease() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try externalLease(directory, holder: String(getppid()))
    let reported = BusyLog()
    await #expect {
        try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .milliseconds(60)) { reported.append($0) }
    } throws: { admissionFailure($0) == .leaseTimedOut(holder: getppid()) }
    #expect(reported.values.first == .ancestorWithoutHandOff(getppid()))
    // A pid-less lease is reported as such, repeatedly while it persists, and is never reclaimed.
    try FileManager.default.removeItem(at: directory.appendingPathComponent("pid"))
    let pidless = BusyLog()
    var configuration = leaseConfiguration(directory)
    configuration.busyReportInterval = .milliseconds(40)
    await #expect {
        try await SharedGPULease.acquire(configuration, timeout: .milliseconds(300)) { pidless.append($0) }
    } throws: { admissionFailure($0) == .leaseTimedOut(holder: nil) }
    #expect(pidless.values.count >= 2 && pidless.values.allSatisfy { $0 == .unidentified })
    #expect(FileManager.default.fileExists(atPath: directory.path))
}

@Test func anExplicitHandOffIsInheritedOnlyFromTheNamedLiveAncestor() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    try externalLease(directory, holder: String(getppid()))
    // No hand-off: an ancestor's lease is held like any other.
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0, by: getppid()) }
    // A hand-off naming someone else is not accepted.
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory, handedOffBy: 1)) } throws: { isHeld($0, by: getppid()) }
    let inherited = try SharedGPULease.tryAcquire(leaseConfiguration(directory, handedOffBy: getppid()))
    #expect(inherited.holding == .inherited(holder: getppid()) && inherited.isStillValid)
    #expect(inherited.handOffEnvironment == ["AETHER_GPU_LEASE_HOLDER": String(getppid())])
    inherited.release()
    #expect(pidText(directory) == String(getppid()))
    // Once the lease names anyone else, or is gone, the inherited lease no longer protects this process.
    try Data("1".utf8).write(to: directory.appendingPathComponent("pid"), options: .atomic)
    #expect(!inherited.isStillValid)
    try FileManager.default.removeItem(at: directory)
    #expect(!inherited.isStillValid)
    let owned = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(owned.isStillValid && owned.handOffEnvironment == ["AETHER_GPU_LEASE_HOLDER": String(getpid())])
    owned.release()
    #expect(!owned.isStillValid)
}

@Test func ancestorDetectionWalksTheParentChain() throws {
    #expect(SharedGPULease.isAncestor(getppid()))
    #expect(!SharedGPULease.isAncestor(getpid()))
    #expect(!SharedGPULease.isAncestor(1))
    #expect(!SharedGPULease.isAncestor(try deadPID()))
}

// MARK: Cross-process cases: a Kit helper process and a shell holder using the wrapper's protocol.

private final class BundleToken {}

private func helperURL() throws -> URL {
    let url = Bundle(for: BundleToken.self).bundleURL.deletingLastPathComponent().appendingPathComponent("AetherGPULeaseHelper")
    try #require(FileManager.default.isExecutableFile(atPath: url.path), "build the AetherGPULeaseHelper target")
    return url
}

/// Starts a process and waits for `marker` on its standard output.
/// Children never inherit a hand-off from the test runner's environment; tests pass one explicitly.
private func start(_ executable: URL, _ arguments: [String], until marker: String?,
                   environment: [String: String] = [:]) throws -> (Process, Pipe) {
    let process = Process(), output = Pipe()
    process.executableURL = executable
    process.arguments = arguments
    var inherited = ProcessInfo.processInfo.environment
    inherited.removeValue(forKey: SharedGPULease.holderEnvironmentKey)
    process.environment = inherited.merging(environment) { $1 }
    process.standardOutput = output
    process.standardError = FileHandle.nullDevice
    try process.run()
    if let marker { try waitFor(marker: marker, on: output) }
    return (process, output)
}

/// Blocks until `marker` appears on a child's standard output.
private func waitFor(marker: String, on output: Pipe) throws {
    var text = Data()
    while !text.contains(Data(marker.utf8)) {
        let chunk = output.fileHandleForReading.availableData
        if chunk.isEmpty { break }
        text.append(chunk)
    }
    try #require(text.contains(Data(marker.utf8)), "the child did not report \(marker)")
}

private func startHelper(_ mode: String, _ directory: URL, holding: Bool = true) throws -> Process {
    let (process, _) = try start(try helperURL(), [mode, directory.path], until: "ready")
    if holding { #expect(pidText(directory) == String(process.processIdentifier)) }
    return process
}

/// Runs to completion and returns status and trimmed standard output.
private func runToEnd(_ executable: URL, _ arguments: [String], environment: [String: String] = [:]) throws -> (Int32, String) {
    let (process, output) = try start(executable, arguments, until: nil, environment: environment)
    let data = output.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines))
}

/// Waits for exit, killing a child that outlives the deadline so no test can hang.
private func exited(_ process: Process, within seconds: Double = 10) -> Bool {
    let deadline = Date().addingTimeInterval(seconds)
    while process.isRunning, Date() < deadline { usleep(10_000) }
    guard !process.isRunning else { kill(process.processIdentifier, SIGKILL); process.waitUntilExit(); return false }
    return true
}

/// The development wrapper's lease steps (`with-gpu-lock.sh`), against a private directory.
private let wrapperHolder = #"""
lease=$1
until mkdir $lease 2>/dev/null; do sleep 0.05; done
print -n $$ > $lease/pid.tmp.$$ && mv -f $lease/pid.tmp.$$ $lease/pid
release() { [[ "$(cat $lease/pid 2>/dev/null)" == "$$" ]] && rm -rf $lease }
trap "release; exit 130" INT TERM HUP
print ready
while true; do sleep 0.05; done
"""#

@Test func anotherKitProcessContendsForTheLease() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let helper = try helperURL()
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    // Fail-fast and timed waits in another process see this process as the holder.
    #expect(try runToEnd(helper, ["try", directory.path]) == (2, "held \(getpid())"))
    #expect(try runToEnd(helper, ["wait", directory.path, "0.2"]) == (3, "timedOut \(getpid())"))
    // A waiting process takes the lease once this one releases it.
    let (waiter, output) = try start(helper, ["wait", directory.path, "10"], until: nil)
    try await Task.sleep(for: .milliseconds(300))
    #expect(waiter.isRunning)
    held.release()
    #expect(exited(waiter) && waiter.terminationStatus == 0)
    #expect(String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).contains("acquired"))
    #expect(try leaseSiblings(directory).isEmpty)
    // And the other way around: a lease another process holds times out here, then frees on its termination.
    let holder = try startHelper("default", directory)
    await #expect { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .milliseconds(100)) } throws: { error in
        admissionFailure(error) == .leaseTimedOut(holder: holder.processIdentifier)
    }
    let waiting = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .seconds(10)) }
    kill(holder.processIdentifier, SIGTERM)
    #expect(exited(holder))
    let lease = try await waiting.value
    #expect(pidText(directory) == ownPID)
    lease.release()
}

@Test func aChildInheritsOnlyAnExplicitHandOffFromAnAncestor() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let helper = try helperURL()
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    let handOff = held.handOffEnvironment
    #expect(try runToEnd(helper, ["try", directory.path]) == (2, "held \(getpid())"))
    // Being an ancestor is not enough: without the hand-off the child waits like any other process.
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"]) == (2, "held \(getpid())"))
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"], environment: ["AETHER_GPU_LEASE_HOLDER": "1"])
            == (2, "held \(getpid())"))
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"], environment: handOff) == (0, "inherited \(getpid())"))
    // Through an intermediate shell, as under a wrapper; the inherited lease is left in place.
    let shell = URL(fileURLWithPath: "/bin/sh")
    #expect(try runToEnd(shell, ["-c", "\"$0\" try \"$1\" inherit; echo status=$?", helper.path, directory.path],
                         environment: handOff) == (0, "inherited \(getpid())\nstatus=0"))
    #expect(pidText(directory) == ownPID)
    held.release()
    // A hand-off naming a sibling holder is refused: the sibling is not an ancestor.
    let sibling = try startHelper("default", directory)
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"],
                         environment: ["AETHER_GPU_LEASE_HOLDER": String(sibling.processIdentifier)])
            == (2, "held \(sibling.processIdentifier)"))
    kill(sibling.processIdentifier, SIGTERM)
    #expect(exited(sibling))
}

@Test func aWrapperStyleHolderAndTheKitLeaseExcludeEachOther() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let zsh = URL(fileURLWithPath: "/bin/zsh")
    let (wrapper, _) = try start(zsh, ["-c", wrapperHolder, "_", directory.path], until: "ready")
    #expect(pidText(directory) == String(wrapper.processIdentifier))
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0, by: wrapper.processIdentifier) }
    let waiting = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .seconds(10)) }
    try await Task.sleep(for: .milliseconds(150))
    kill(wrapper.processIdentifier, SIGTERM)
    #expect(exited(wrapper) && wrapper.terminationStatus == 130)
    let lease = try await waiting.value
    #expect(pidText(directory) == ownPID)
    // While the Kit holds it, the wrapper's mkdir fails, and it gets the lease after release.
    let (second, output) = try start(zsh, ["-c", wrapperHolder, "_", directory.path], until: nil)
    try await Task.sleep(for: .milliseconds(200))
    #expect(pidText(directory) == ownPID)
    lease.release()
    // "ready" follows the wrapper's trap installation; signalling earlier would kill it before it can release.
    try waitFor(marker: "ready", on: output)
    #expect(pidText(directory) == String(second.processIdentifier))
    kill(second.processIdentifier, SIGTERM)
    #expect(exited(second) && second.terminationStatus == 130)
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func aKilledWrapperStyleHoldersLeaseIsReclaimed() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let (wrapper, _) = try start(URL(fileURLWithPath: "/bin/zsh"), ["-c", wrapperHolder, "_", directory.path], until: "ready")
    kill(wrapper.processIdentifier, SIGKILL)
    #expect(exited(wrapper))
    #expect(pidText(directory) == String(wrapper.processIdentifier))
    let next = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(pidText(directory) == ownPID)
    next.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test(arguments: [SIGTERM, SIGINT, SIGHUP])
func defaultTerminationReleasesTheLeaseBeforeTheProcessDies(signal: Int32) throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("default", directory)
    kill(process.processIdentifier, signal)
    #expect(exited(process))
    #expect(process.terminationReason == .uncaughtSignal && process.terminationStatus == signal)
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func anOwnTerminationHandlerIsNotPreemptedByTheLease() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("handled", directory)
    kill(process.processIdentifier, SIGTERM)
    #expect(exited(process))
    #expect(process.terminationReason == .exit && process.terminationStatus == 42)
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func anIgnoredSignalStaysIgnoredWhileTheLeaseIsHeld() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("ignored", directory)
    kill(process.processIdentifier, SIGHUP)
    // Give an unwanted release or termination ample time to show, then check twice.
    usleep(500_000)
    #expect(process.isRunning)
    #expect(pidText(directory) == String(process.processIdentifier))
    kill(process.processIdentifier, SIGHUP)
    usleep(200_000)
    #expect(process.isRunning)
    #expect(pidText(directory) == String(process.processIdentifier))
    kill(process.processIdentifier, SIGTERM)
    #expect(exited(process))
    #expect(process.terminationReason == .uncaughtSignal && process.terminationStatus == SIGTERM)
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func aKilledHoldersLeaseIsReclaimedByTheNextHost() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("default", directory)
    kill(process.processIdentifier, SIGKILL)
    #expect(exited(process))
    #expect(pidText(directory) == String(process.processIdentifier))
    let next = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(pidText(directory) == ownPID)
    next.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func releaseKeepsAHandlerInstalledAfterTheLease() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("late", directory, holding: false)
    #expect(try leaseSiblings(directory).isEmpty)
    kill(process.processIdentifier, SIGTERM)
    #expect(exited(process))
    #expect(process.terminationReason == .exit && process.terminationStatus == 43)
}

@Test func signalReleaseCoversEveryLeaseStillHeldAfterAnotherReleases() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let process = try startHelper("pair", directory, holding: false)
    let second = directory.deletingLastPathComponent().appendingPathComponent("gpu.lock.second")
    #expect(!FileManager.default.fileExists(atPath: directory.path))
    #expect(pidText(second) == String(process.processIdentifier))
    kill(process.processIdentifier, SIGTERM)
    #expect(exited(process))
    #expect(process.terminationReason == .uncaughtSignal && process.terminationStatus == SIGTERM)
    #expect(try leaseSiblings(directory).isEmpty)
}

// MARK: scripts/gpu-lease, the repo wrapper for GPU work outside the kit (Forge checks), against a private directory.

private func gpuLeaseScript() throws -> URL {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let url = root.appendingPathComponent("scripts/gpu-lease")
    try #require(FileManager.default.isExecutableFile(atPath: url.path), "scripts/gpu-lease is missing or not executable")
    return url
}

/// A command for the wrapper to run: reports ready, then runs until signalled or orphaned.
private let leaseScriptChild = ["/bin/sh", "-c", "echo ready; while kill -0 $PPID 2>/dev/null; do sleep 0.05; done"]

@Test func theLeaseScriptsLeaseIsBusyForTheKitAndReleasedOnTermination() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let script = try gpuLeaseScript(), helper = try helperURL()
    let (wrapper, _) = try start(script, ["--directory", directory.path, "--"] + leaseScriptChild, until: "ready")
    let holder = wrapper.processIdentifier
    #expect(pidText(directory) == String(holder))
    #expect { try SharedGPULease.tryAcquire(leaseConfiguration(directory)) } throws: { isHeld($0, by: holder) }
    #expect(try runToEnd(helper, ["try", directory.path]) == (2, "held \(holder)"))
    let waiting = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .seconds(10)) }
    try await Task.sleep(for: .milliseconds(150))
    // SIGTERM goes to the command; the wrapper releases once it exits, and dies of the same signal.
    kill(holder, SIGTERM)
    #expect(exited(wrapper))
    #expect(wrapper.terminationReason == .uncaughtSignal && wrapper.terminationStatus == SIGTERM)
    let lease = try await waiting.value
    #expect(pidText(directory) == ownPID)
    lease.release()
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func theLeaseScriptWaitsWhileTheKitHoldsTheLease() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let script = try gpuLeaseScript()
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    // Busy: the command never runs, and the wrapper times out with 75.
    #expect(try runToEnd(script, ["--quiet", "--timeout", "0.3", "--directory", directory.path, "--", "/bin/echo", "ran"]) == (75, ""))
    // A waiting wrapper runs its command once the kit releases, and passes its exit status through.
    let (waiter, output) = try start(script, ["--quiet", "--directory", directory.path, "--", "/bin/sh", "-c", "echo ran; exit 3"],
                                     until: nil)
    try await Task.sleep(for: .milliseconds(400))
    #expect(waiter.isRunning)
    #expect(pidText(directory) == ownPID)
    held.release()
    #expect(exited(waiter) && waiter.terminationReason == .exit && waiter.terminationStatus == 3)
    #expect(String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self) == "ran\n")
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func aKitProcessUnderTheLeaseScriptInheritsItsLease() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let script = try gpuLeaseScript(), helper = try helperURL()
    let wrapped = ["--directory", directory.path, "--", "/bin/sh", "-c", "echo $AETHER_GPU_LEASE_HOLDER $PPID; \"$0\" try \"$1\" inherit",
                   helper.path, directory.path]
    let (status, output) = try runToEnd(script, wrapped)
    let lines = output.split(separator: "\n").map(String.init)
    try #require(lines.count == 2)
    // The child's hand-off names the wrapper (the shell's parent), and the kit process inherits from it.
    let fields = lines[0].split(separator: " ")
    #expect(status == 0 && fields.count == 2 && fields[0] == fields[1])
    #expect(lines[1] == "inherited \(fields[0])")
    // Without accepting the hand-off the kit process sees the wrapper as the holder; its status passes through.
    let (refused, held) = try runToEnd(script, ["--directory", directory.path, "--", helper.path, "try", directory.path])
    #expect(refused == 2 && held.hasPrefix("held "))
    // A wrapper inside a wrapper just runs its command under the outer lease.
    let (nested, inner) = try runToEnd(script, ["--directory", directory.path, "--", script.path, "--timeout", "0", "--directory",
                                                directory.path, "--", helper.path, "try", directory.path, "inherit"])
    #expect(nested == 0 && inner.hasPrefix("inherited "))
    #expect(try leaseSiblings(directory).isEmpty)
}

@Test func theLeaseScriptReclaimsOnlyALeaseNamingADeadProcess() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let script = try gpuLeaseScript()
    let quick = ["--quiet", "--timeout", "0", "--directory", directory.path, "--", "/bin/echo", "ran"]
    // Pid-less and live-holder leases are never reclaimed.
    try externalLease(directory, holder: nil)
    #expect(try runToEnd(script, quick) == (75, ""))
    try FileManager.default.removeItem(at: directory)
    try externalLease(directory, holder: "1")
    #expect(try runToEnd(script, quick) == (75, ""))
    #expect(pidText(directory) == "1")
    try FileManager.default.removeItem(at: directory)
    // A dead holder's lease is reclaimed, and the wrapper's own lease is gone after its command.
    try externalLease(directory, holder: String(try deadPID()))
    #expect(try runToEnd(script, quick) == (0, "ran"))
    #expect(try leaseSiblings(directory).isEmpty)
    // And the kit reclaims a killed wrapper's lease.
    let (wrapper, _) = try start(script, ["--directory", directory.path, "--"] + leaseScriptChild, until: "ready")
    kill(wrapper.processIdentifier, SIGKILL)
    #expect(exited(wrapper))
    #expect(pidText(directory) == String(wrapper.processIdentifier))
    let next = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(pidText(directory) == ownPID)
    next.release()
    #expect(try leaseSiblings(directory).isEmpty)
}


// MARK: Waiters (C2)

@Test func aWaitingAcquisitionIsVisibleToTheHolderUntilItStopsWaiting() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let noWrapper = directory.deletingLastPathComponent().appendingPathComponent("no-wrapper.lock")
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    let reported = BusyLog()
    let waiter = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), onBusy: { reported.append($0) }) }
    let deadline = Date().addingTimeInterval(30)
    while reported.values.isEmpty, Date() < deadline { usleep(5_000) }
    #expect(SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    // Probing doesn't disturb the waiter, and asking again gives the same answer.
    #expect(SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    held.release()
    let lease = try await waiter.value
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    lease.release()
}

@Test func aCancelledOrTimedOutWaiterStopsBeingAWaiter() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let noWrapper = directory.deletingLastPathComponent().appendingPathComponent("no-wrapper.lock")
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    await #expect(throws: AetherError.self) { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .milliseconds(100)) }
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    let reported = BusyLog()
    let waiter = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), onBusy: { reported.append($0) }) }
    let deadline = Date().addingTimeInterval(30)
    while reported.values.isEmpty, Date() < deadline { usleep(5_000) }
    #expect(held.othersWaiting || SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    waiter.cancel()
    _ = try? await waiter.value
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    // A one-attempt acquisition never marks itself.
    await #expect(throws: AetherError.self) { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .zero) }
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    held.release()
}

@Test func aLegacyWrapperHoldingItsLockCountsAsAWaiter() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let wrapper = directory.deletingLastPathComponent().appendingPathComponent("wrapper.lock")
    #expect(FileManager.default.createFile(atPath: wrapper.path, contents: Data()))
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: wrapper))
    let queued = open(wrapper.path, O_RDONLY | O_CLOEXEC)
    defer { close(queued) }
    #expect(flock(queued, LOCK_EX | LOCK_NB) == 0)
    #expect(SharedGPULease.hasWaiters(directory: directory, legacyWrapper: wrapper))
    flock(queued, LOCK_UN)
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: wrapper))
}

@Test func aWaitersFileThatIsNotThisUsersPrivateFileIsNeverUsed() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let parent = directory.deletingLastPathComponent()
    let noWrapper = parent.appendingPathComponent("no-wrapper.lock")
    // A hard link to another file of this user's (two links) is refused, so holding it can't fake a waiter.
    let other = parent.appendingPathComponent("other")
    #expect(FileManager.default.createFile(atPath: other.path, contents: Data()))
    let waiters = parent.appendingPathComponent(directory.lastPathComponent + ".waiters")
    #expect(link(other.path, waiters.path) == 0)
    let held = open(other.path, O_RDONLY | O_CLOEXEC)
    defer { close(held) }
    #expect(flock(held, LOCK_SH | LOCK_NB) == 0)
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
}

@Test func aWaitingLeaseScriptIsVisibleToTheHolder() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let noWrapper = directory.deletingLastPathComponent().appendingPathComponent("no-wrapper.lock")
    let script = try gpuLeaseScript()
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    let (waiter, _) = try start(script, ["--quiet", "--directory", directory.path, "--", "/bin/echo", "ran"], until: nil)
    let deadline = Date().addingTimeInterval(30)
    while !SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper), Date() < deadline { usleep(10_000) }
    #expect(held.othersWaiting || SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
    held.release()
    #expect(exited(waiter) && waiter.terminationStatus == 0)
    #expect(!SharedGPULease.hasWaiters(directory: directory, legacyWrapper: noWrapper))
}
