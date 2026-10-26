@testable import AetherKit
import Darwin
import Foundation
import Synchronization
import Testing

// Every case uses its own temporary lock directory. None may touch the shared /tmp/atelier-gpu.lock.
// The protocol cases mirror Atelier's SharedGPULockTests so both implementations are held to the same contract.

func privateLeaseDirectory() throws -> URL {
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent("aether-lease-test-\(UUID())", isDirectory: true)
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    let directory = parent.appendingPathComponent("gpu.lock", isDirectory: true)
    #expect(directory.path != SharedGPULease.sharedDirectory.path)
    return directory
}

func removeLeaseParent(_ directory: URL) { try? FileManager.default.removeItem(at: directory.deletingLastPathComponent()) }

func leaseConfiguration(_ directory: URL, inherit: Bool = false) -> SharedGPULease.Configuration {
    SharedGPULease.Configuration(directory: directory, inheritsAncestorLease: inherit, pollInterval: .milliseconds(20))
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
        .filter { $0 != "gpu.lock.reclaim" && $0 != "gpu.lock.second.reclaim" }.sorted()
}

private func isHeld(_ error: any Error, by holder: Int32? = nil) -> Bool {
    guard case SharedGPULeaseError.held(let found) = error else { return false }
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
            guard case SharedGPULeaseError.held(nil) = error else { return false }
            return true
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
        guard case SharedGPULeaseError.unavailable(_, ENOENT) = error else { return false }
        return true
    }
}

@Test func waitingAcquisitionTimesOutAndHonorsCancellation() async throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    let clock = ContinuousClock(), start = clock.now
    await #expect { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .milliseconds(150)) } throws: { error in
        guard case SharedGPULeaseError.timedOut(getpid()) = error else { return false }
        return true
    }
    #expect(start.duration(to: clock.now) >= .milliseconds(140))
    // Zero timeout is one attempt.
    await #expect { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .zero) } throws: { error in
        if case SharedGPULeaseError.timedOut = error { return true }
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
    let reported = HolderLog()
    let waiter = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), onBusy: { reported.append($0) }) }
    try await Task.sleep(for: .milliseconds(150))
    held.release()
    let lease = try await waiter.value
    #expect(lease.holding == .owned && pidText(directory) == ownPID)
    #expect(reported.values == [getpid()])
    lease.release()
}

final class HolderLog: Sendable {
    private let log = Mutex<[Int32?]>([])
    func append(_ value: Int32?) { log.withLock { $0.append(value) } }
    var values: [Int32?] { log.withLock { $0 } }
}

@Test func toolPolicyReadsTheDocumentedEnvironment() throws {
    let standard = try SharedGPULease.ToolPolicy.environment([:])
    #expect(standard.configuration?.directory.path == SharedGPULease.sharedDirectory.path)
    #expect(standard.configuration?.inheritsAncestorLease == true && standard.timeout == nil)
    #expect(try SharedGPULease.ToolPolicy.environment(["AETHER_GPU_LEASE": "on"]).configuration != nil)
    #expect(try SharedGPULease.ToolPolicy.environment(["AETHER_GPU_LEASE": "0"]).configuration == nil)
    #expect(try SharedGPULease.ToolPolicy.environment(["AETHER_GPU_LEASE": "OFF"]).configuration == nil)
    let custom = try SharedGPULease.ToolPolicy.environment(["AETHER_GPU_LEASE_PATH": "/private/tmp/x.lock", "AETHER_GPU_LEASE_TIMEOUT": "2.5"])
    #expect(custom.configuration?.directory.path == "/private/tmp/x.lock" && custom.timeout == .milliseconds(2500))
    for bad in [["AETHER_GPU_LEASE": "maybe"], ["AETHER_GPU_LEASE_PATH": "relative.lock"], ["AETHER_GPU_LEASE_TIMEOUT": "-1"],
                ["AETHER_GPU_LEASE_TIMEOUT": "soon"]] {
        #expect { try SharedGPULease.ToolPolicy.environment(bad) } throws: { error in
            if case SharedGPULeaseError.invalidEnvironment = error { return true }
            return false
        }
    }
    #expect(SharedGPULease.ToolPolicy(configuration: nil, timeout: nil) == (try SharedGPULease.ToolPolicy.environment(["AETHER_GPU_LEASE": "off"])))
}

@Test func disabledToolPolicyTakesNothing() async throws {
    #expect(try await SharedGPULease.ToolPolicy(configuration: nil, timeout: nil).acquire() == nil)
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
private func start(_ executable: URL, _ arguments: [String], until marker: String?) throws -> (Process, Pipe) {
    let process = Process(), output = Pipe()
    process.executableURL = executable
    process.arguments = arguments
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
private func runToEnd(_ executable: URL, _ arguments: [String]) throws -> (Int32, String) {
    let (process, output) = try start(executable, arguments, until: nil)
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
        guard case SharedGPULeaseError.timedOut(holder.processIdentifier) = error else { return false }
        return true
    }
    let waiting = Task { try await SharedGPULease.acquire(leaseConfiguration(directory), timeout: .seconds(10)) }
    kill(holder.processIdentifier, SIGTERM)
    #expect(exited(holder))
    let lease = try await waiting.value
    #expect(pidText(directory) == ownPID)
    lease.release()
}

@Test func aChildInheritsOnlyWhenAskedAndOnlyFromAnAncestor() throws {
    let directory = try privateLeaseDirectory(); defer { removeLeaseParent(directory) }
    let helper = try helperURL()
    let held = try SharedGPULease.tryAcquire(leaseConfiguration(directory))
    #expect(try runToEnd(helper, ["try", directory.path]) == (2, "held \(getpid())"))
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"]) == (0, "inherited \(getpid())"))
    // Through an intermediate shell, as under a wrapper; the inherited lease is left in place.
    let shell = URL(fileURLWithPath: "/bin/sh")
    #expect(try runToEnd(shell, ["-c", "\"$0\" try \"$1\" inherit; echo status=$?", helper.path, directory.path])
            == (0, "inherited \(getpid())\nstatus=0"))
    #expect(pidText(directory) == ownPID)
    held.release()
    // A sibling holder is not an ancestor.
    let sibling = try startHelper("default", directory)
    #expect(try runToEnd(helper, ["try", directory.path, "inherit"]) == (2, "held \(sibling.processIdentifier)"))
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
    usleep(300_000)
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
