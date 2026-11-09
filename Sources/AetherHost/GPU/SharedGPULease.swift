import Darwin
import Dispatch
public import Foundation
import Synchronization

/// A cooperative, cross-process GPU lease, wire-compatible with Atelier's `SharedGPULock` and the lab wrappers
/// (`with-gpu-lock.sh`, `gpu.sh`). Ported from AetherKit 0.0.7 (`docs/gpu-lease.md` there); the protocol is unchanged.
///
/// The lease is a directory created atomically with `mkdir`; its holder writes its decimal pid, without a newline,
/// to `pid` inside it (written to `pid.tmp.<pid>` and renamed into place). Release moves the directory aside under a
/// unique name, only while `pid` still names this process, and then deletes the moved copy, so the lease path never
/// holds a lease without a pid. A lease whose pid names a process that no longer exists (`kill(pid, 0)` fails with
/// `ESRCH`) is reclaimed: reclaimers serialize on a sibling `<name>.reclaim` flock file (this user's single-link
/// regular file, mode 0600), move the lease aside atomically, delete the moved copy only if it still names the same
/// dead process, and otherwise restore it with `renamex_np(RENAME_EXCL)` so a lease created since is never replaced.
/// A missing or unreadable pid (a holder between `mkdir` and its pid write) is never stale and never reclaimed.
///
/// Default-handled SIGINT, SIGTERM and SIGHUP release held leases before the process terminates; a signal the process
/// already handles or ignores is left alone, so hosts install their own signal handling *before* the first lease.
/// Releasing is idempotent and also happens on deinit.
///
/// The lease coordinates only processes that share the directory. A sandboxed app (and any iOS app) gets a private
/// temporary directory, so there it excludes nothing outside the app: use in-process ``GPUAdmission`` alone there.
/// Nothing here is a process-global default: callers choose the directory (``Configuration``).
///
/// Errors are ``AetherError/admission(_:)`` with ``AetherError/AdmissionFailure/leaseHeld(holder:)``,
/// ``AetherError/AdmissionFailure/leaseTimedOut(holder:)`` or ``AetherError/AdmissionFailure/leaseUnavailable(path:errno:)``.
public final class SharedGPULease: Sendable {
    /// The directory Atelier, the lab scripts and Aether tools share on one Mac.
    public static let sharedDirectory = URL(fileURLWithPath: "/tmp/atelier-gpu.lock", isDirectory: true)

    /// The environment variable a holder sets for a child process it hands its lease to (``handOffEnvironment``).
    /// Aether never reads the environment itself: a tool that honours hand-offs passes the value as
    /// ``Configuration/handedOffBy``.
    public static let holderEnvironmentKey = "AETHER_GPU_LEASE_HOLDER"

    /// How this process holds the lease.
    public enum Holding: Sendable, Equatable {
        /// This process created the lease and its pid file names this process.
        case owned
        /// An ancestor process handed this process its lease explicitly (``holderEnvironmentKey``), for example a
        /// wrapper running this process as its GPU job. Release leaves it alone.
        case inherited(holder: Int32)
    }

    /// Why a lease could not be taken, as reported to `onBusy`.
    public enum Busy: Sendable, Equatable {
        /// A live process holds the lease.
        case process(Int32)
        /// An ancestor of this process holds the lease but did not hand it to this process. It may be doing its own GPU
        /// work, so this process waits; a wrapper that runs this process as its GPU job should set
        /// ``SharedGPULease/holderEnvironmentKey`` to its pid.
        case ancestorWithoutHandOff(Int32)
        /// The lease has no readable pid. Either a holder is between `mkdir` and its pid write, or a holder was killed
        /// in that window and abandoned it. It is never reclaimed automatically (the shared protocol has no rule for
        /// it); if no process is using the GPU, remove the directory by hand.
        case unidentified

        public var holder: Int32? {
            switch self {
            case .process(let pid), .ancestorWithoutHandOff(let pid): pid
            case .unidentified: nil
            }
        }
    }

    /// Where and how to acquire.
    public struct Configuration: Sendable, Equatable {
        public var directory: URL
        /// The process that handed this process its lease, if any. The lease is inherited only while its pid names
        /// exactly this process and that process is a live ancestor of this one.
        public var handedOffBy: Int32?
        /// How often a waiting acquisition retries.
        public var pollInterval: Duration
        /// How often a waiting acquisition repeats `onBusy` while the holder is unchanged.
        public var busyReportInterval: Duration

        public init(directory: URL = SharedGPULease.sharedDirectory, handedOffBy: Int32? = nil,
                    pollInterval: Duration = .milliseconds(200), busyReportInterval: Duration = .seconds(60)) {
            self.directory = directory
            self.handedOffBy = handedOffBy
            self.pollInterval = pollInterval
            self.busyReportInterval = busyReportInterval
        }
    }

    public let directory: URL
    public let holding: Holding
    private let owner: pid_t
    private let held: Mutex<Bool>
    private let signalToken = UUID()

    private init(directory: URL, holding: Holding) {
        self.directory = directory
        self.holding = holding
        owner = getpid()
        held = Mutex(holding == .owned)
    }

    /// True until an owned lease is released. An inherited lease is never held by this object.
    public var isHeld: Bool { held.withLock { $0 } }

    /// Whether the lease still protects this process: an owned lease not yet released whose pid file still names this
    /// process, or an inherited lease whose pid file still names its live holder.
    public var isStillValid: Bool {
        switch holding {
        case .owned: isHeld && Self.readHolder(directory) == owner
        case .inherited(let holder): Self.readHolder(directory) == holder && !Self.isDead(holder)
        }
    }

    /// The environment to give a child process that should run as part of this process's GPU work. The child inherits
    /// the lease only while this process still holds it.
    public var handOffEnvironment: [String: String] {
        switch holding {
        case .owned: [Self.holderEnvironmentKey: String(owner)]
        case .inherited(let holder): [Self.holderEnvironmentKey: String(holder)]
        }
    }

    /// One attempt: acquires, reclaims a stale lease, or throws `.admission(.leaseHeld(holder:))`.
    public static func tryAcquire(_ configuration: Configuration = Configuration()) throws(AetherError) -> SharedGPULease {
        switch try attempt(configuration) {
        case .acquired(let lease): return lease
        case .busy(let busy): throw .admission(.leaseHeld(holder: busy.holder))
        }
    }

    /// Waits until the lease is free, retrying every `pollInterval`. A `nil` timeout waits indefinitely; a zero timeout
    /// makes one attempt. Cancelling the calling task stops waiting with `CancellationError`. `onBusy` is called when
    /// the lease is first found held, whenever the holder changes, and every `busyReportInterval` while it does not.
    public static func acquire(
        _ configuration: Configuration = Configuration(), timeout: Duration? = nil,
        onBusy: (@Sendable (Busy) -> Void)? = nil
    ) async throws -> SharedGPULease {
        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now.advanced(by: max(.zero, $0)) }
        var reported: (busy: Busy, at: ContinuousClock.Instant)?
        // Held while waiting so a holder can see it (hasWaiters); closing it on any exit releases the shared lock.
        var waiting: WaiterMark?
        defer { withExtendedLifetime(waiting) {} }
        while true {
            try Task.checkCancellation()
            let busy: Busy
            switch try attempt(configuration) {
            case .acquired(let lease):
                waiting = nil
                return lease
            case .busy(let current): busy = current
            }
            if waiting == nil, timeout != .zero { waiting = WaiterMark(directory: configuration.directory) }
            let now = clock.now
            if reported?.busy != busy || reported.map({ $0.at.duration(to: now) >= configuration.busyReportInterval }) == true {
                reported = (busy, now)
                onBusy?(busy)
            }
            if let deadline {
                guard now < deadline else { throw AetherError.admission(.leaseTimedOut(holder: busy.holder)) }
                try await Task.sleep(for: min(configuration.pollInterval, now.duration(to: deadline)))
            } else {
                try await Task.sleep(for: configuration.pollInterval)
            }
        }
    }

    /// The process an existing lease names, if its pid file is present and well formed.
    public static func holder(of directory: URL = sharedDirectory) -> Int32? { readHolder(directory) }

    /// The legacy wrapper lock the lab's `with-gpu-lock.sh` holds exclusively while a job waits in its queue.
    public static var legacyWrapperLock: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/aether-kit/gpu.lock")
    }

    /// Whether another process is waiting for the lease at `directory`: a waiting ``acquire(_:timeout:onBusy:)`` or
    /// `scripts/gpu-lease` holds a shared lock on the lease's `<name>.waiters` file (this user's single-link, mode-0600
    /// sibling, as `<name>.reclaim`) for as long as it waits; or, with `legacyWrapper`, a lab wrapper holds that lock
    /// file exclusively. A holder that keeps the GPU between its own jobs (a warm session) can ask this between jobs
    /// and step aside. Never blocks; a file this user can't use reads as no waiter.
    public static func hasWaiters(directory: URL = sharedDirectory, legacyWrapper: URL? = legacyWrapperLock) -> Bool {
        if let legacyWrapper {
            let descriptor = open(legacyWrapper.path, O_RDONLY | O_SHLOCK | O_NONBLOCK | O_CLOEXEC)
            if descriptor >= 0 { close(descriptor) } else if errno == EWOULDBLOCK { return true }
        }
        guard let waiters = privateSibling(directory, suffix: ".waiters", create: false) else { return false }
        defer { close(waiters) }
        guard flock(waiters, LOCK_EX | LOCK_NB) == 0 else { return errno == EWOULDBLOCK }
        flock(waiters, LOCK_UN)
        return false
    }

    /// Whether another process waits for this lease (``hasWaiters(directory:legacyWrapper:)`` on its directory).
    public var othersWaiting: Bool { Self.hasWaiters(directory: directory) }

    /// Releases an owned lease only if the pid file still names this process. Idempotent; an inherited lease is left.
    public func release() {
        held.withLock { value in
            guard value else { return }
            value = false
            Self.remove(directory, ifHeldBy: owner)
        }
        LeaseSignalRelease.unregister(signalToken)
    }

    deinit { release() }

    // MARK: Protocol

    private enum Attempt { case acquired(SharedGPULease), busy(Busy) }

    private static func attempt(_ configuration: Configuration) throws(AetherError) -> Attempt {
        let directory = configuration.directory
        if let lease = try create(directory) { return .acquired(lease) }
        let holder = readHolder(directory)
        if let holder, let handedOffBy = configuration.handedOffBy, holder == handedOffBy, !isDead(holder),
           isAncestor(holder) {
            return .acquired(SharedGPULease(directory: directory, holding: .inherited(holder: holder)))
        }
        if clearStale(directory), let lease = try create(directory) { return .acquired(lease) }
        guard let current = readHolder(directory) else { return .busy(.unidentified) }
        return .busy(isAncestor(current) ? .ancestorWithoutHandOff(current) : .process(current))
    }

    /// `nil` when the directory already exists.
    private static func create(_ directory: URL) throws(AetherError) -> SharedGPULease? {
        guard mkdir(directory.path, 0o755) == 0 else {
            let code = errno
            if code == EEXIST { return nil }
            throw .admission(.leaseUnavailable(path: directory.path, errno: code))
        }
        let pid = getpid()
        let temporary = directory.appendingPathComponent("pid.tmp.\(pid)").path
        let final = directory.appendingPathComponent("pid").path
        var failure: Int32 = 0
        let descriptor = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o644)
        if descriptor < 0 {
            failure = errno
        } else {
            let bytes = Array(String(pid).utf8)
            let count = bytes.withUnsafeBytes { write(descriptor, $0.baseAddress, $0.count) }
            if count != bytes.count { failure = count < 0 ? errno : EIO }
            if close(descriptor) != 0, failure == 0 { failure = errno }
            if failure == 0, rename(temporary, final) != 0 { failure = errno }
        }
        guard failure == 0 else {
            // Never leave a pid-less lease behind: move the half-made lease aside before deleting it.
            removeAside(directory)
            throw .admission(.leaseUnavailable(path: directory.path, errno: failure))
        }
        let lease = SharedGPULease(directory: directory, holding: .owned)
        LeaseSignalRelease.register(lease.signalToken) { [weak lease] in lease?.release() }
        return lease
    }

    /// Moves the lease aside only while it names `owner`, restores it if the moved copy names someone else (a lease
    /// created since an outside remover deleted ours), and otherwise deletes the moved copy.
    private static func remove(_ directory: URL, ifHeldBy owner: pid_t) {
        guard readHolder(directory) == owner else { return }
        let aside = asideURL(directory, "released")
        guard rename(directory.path, aside.path) == 0 else { return }
        guard readHolder(aside) == owner else {
            restore(aside, to: directory)
            return
        }
        try? FileManager.default.removeItem(at: aside)
    }

    /// Removes the lease this process just created but could not name. Only a lease that is still unnamed or names
    /// this process is moved; a moved copy that names anyone else is restored without replacing a newer lease.
    private static func removeAside(_ directory: URL) {
        let pid = getpid()
        func ours(_ url: URL) -> Bool { readHolder(url).map { $0 == pid } ?? true }
        guard ours(directory) else { return }
        let aside = asideURL(directory, "abandoned")
        guard rename(directory.path, aside.path) == 0 else { return }
        guard ours(aside) else {
            restore(aside, to: directory)
            return
        }
        try? FileManager.default.removeItem(at: aside)
    }

    private static func restore(_ aside: URL, to directory: URL) {
        if renamex_np(aside.path, directory.path, UInt32(RENAME_EXCL)) != 0 {
            FileHandle.standardError.write(Data("aether: could not restore a GPU lease; left \(aside.path)\n".utf8))
        }
    }

    private static func asideURL(_ directory: URL, _ reason: String) -> URL {
        directory.deletingLastPathComponent().appendingPathComponent("\(directory.lastPathComponent).\(reason)-\(UUID().uuidString)")
    }

    static func readHolder(_ directory: URL) -> pid_t? {
        guard let text = try? String(contentsOf: directory.appendingPathComponent("pid"), encoding: .utf8),
              let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 else { return nil }
        return pid
    }

    /// True only when signal 0 reports that no such process exists. A live process another user owns (EPERM) is not stale.
    static func isDead(_ pid: pid_t) -> Bool { kill(pid, 0) != 0 && errno == ESRCH }

    /// Walks this process's parent chain. launchd (pid 1) is never treated as a holder.
    static func isAncestor(_ candidate: pid_t, of start: pid_t = getpid()) -> Bool {
        guard candidate > 1 else { return false }
        var current = start
        for _ in 0..<64 {
            guard let parent = parentProcess(of: current), parent > 1 else { return false }
            if parent == candidate { return true }
            current = parent
        }
        return false
    }

    /// `sysctl(KERN_PROC_PID)`. A sandboxed or iOS process may not be allowed to ask; then a hand-off never matches.
    static func parentProcess(of pid: pid_t) -> pid_t? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        guard sysctl(&name, 4, &info, &size, nil, 0) == 0, size == MemoryLayout<kinfo_proc>.stride else { return nil }
        let parent = info.kp_eproc.e_ppid
        return parent > 0 ? parent : nil
    }

    /// Opens `<directory>.<suffix>` beside the lease (created if asked), or nil. /tmp is shared: a file another user
    /// created could be held forever to block everyone, and a hard link to one of this user's files would pass the uid
    /// check. Only this user's single-link regular file is used, made private if an older build created it 0644.
    static func privateSibling(_ directory: URL, suffix: String, create: Bool) -> Int32? {
        let path = directory.deletingLastPathComponent().appendingPathComponent(directory.lastPathComponent + suffix).path
        let descriptor = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | (create ? O_CREAT : 0), 0o600)
        guard descriptor >= 0 else { return nil }
        var owner = stat()
        guard fstat(descriptor, &owner) == 0, owner.st_mode & S_IFMT == S_IFREG, owner.st_uid == getuid(),
              owner.st_nlink == 1 else { close(descriptor); return nil }
        if owner.st_mode & 0o077 != 0 {
            guard fchmod(descriptor, 0o600) == 0, fstat(descriptor, &owner) == 0, owner.st_mode & 0o077 == 0 else {
                close(descriptor); return nil
            }
        }
        return descriptor
    }

    /// Atelier's stale reclaim, step for step. Returns true only when it removed a lease naming a dead process.
    /// `beforeMove` runs between the holder check and the move aside (tests use it to model a racing replacement).
    static func clearStale(_ directory: URL, beforeMove: () -> Void = {}) -> Bool {
        let parent = directory.deletingLastPathComponent()
        guard let serial = privateSibling(directory, suffix: ".reclaim", create: true) else { return false }
        defer { close(serial) }
        // Another reclaimer is mid-reclaim; report the lease as held rather than wait.
        guard flock(serial, LOCK_EX | LOCK_NB) == 0 else { return false }
        guard let dead = readHolder(directory), isDead(dead) else { return false }
        beforeMove()
        let aside = parent.appendingPathComponent(directory.lastPathComponent + ".abandoned-\(UUID().uuidString)")
        guard rename(directory.path, aside.path) == 0 else { return false }
        guard readHolder(aside) == dead else {
            // The lease changed between the read and the move (only an outside remover can free the path while the
            // reclaim lock is held). Never replace a lease created since.
            restore(aside, to: directory)
            return false
        }
        return (try? FileManager.default.removeItem(at: aside)) != nil
    }
}

/// Signals are process-wide, so one handler set serves every lease this process owns. Handlers are installed when
/// the first lease registers, only for signals still at their default disposition, and removed when the last lease
/// releases. On a signal, every registered lease releases and the signal is re-raised with its default action.
///
/// This is the one piece of process-global mutable state in AetherHost, and the system forces it: a signal's
/// disposition belongs to the process, not to any object, so the set of leases to release on SIGTERM has to be
/// reachable from a process-wide handler. It holds no configuration, only the live leases.
///
/// Removing restores `SIG_DFL` only where the disposition is still `SIG_IGN`. That keeps a handler function another
/// component installed while a lease was held, but it cannot tell this code's `SIG_IGN` from a `SIG_IGN` installed
/// later (the usual Swift pattern of `SIG_IGN` plus a `DispatchSource`). Such handling must be installed before the
/// first lease is acquired, which makes the lease leave that signal alone.
enum LeaseSignalRelease {
    private struct Handler {
        let signal: Int32
        let source: any DispatchSourceSignal
    }

    private struct State {
        var releases: [UUID: @Sendable () -> Void] = [:]
        var handlers: [Handler] = []
    }

    private static let state = Mutex(State())

    static func register(_ token: UUID, release: @escaping @Sendable () -> Void) {
        state.withLock { state in
            if state.releases.isEmpty { state.handlers = install() }
            state.releases[token] = release
        }
    }

    static func unregister(_ token: UUID) {
        state.withLock { state in
            guard state.releases.removeValue(forKey: token) != nil, state.releases.isEmpty else { return }
            for handler in state.handlers {
                handler.source.cancel()
                var current = sigaction()
                if sigaction(handler.signal, nil, &current) == 0,
                   unsafeBitCast(current.__sigaction_u.__sa_handler, to: Int.self) == unsafeBitCast(SIG_IGN, to: Int.self) {
                    Darwin.signal(handler.signal, SIG_DFL)
                }
            }
            state.handlers = []
        }
    }

    private static func install() -> [Handler] {
        // Default termination skips deinit: release first, then terminate as before. A signal the process already
        // handles or ignores is left alone, so an owner with its own shutdown releases through it and nohup stays in
        // force. Such owners must install their handling before acquiring.
        [SIGINT, SIGTERM, SIGHUP].compactMap { signal in
            var current = sigaction()
            guard sigaction(signal, nil, &current) == 0, current.__sigaction_u.__sa_handler == nil else { return nil }
            let source = DispatchSource.makeSignalSource(signal: signal, queue: .global())
            source.setEventHandler {
                let releases = state.withLock { Array($0.releases.values) }
                releases.forEach { $0() }
                Darwin.signal(signal, SIG_DFL)
                raise(signal)
            }
            Darwin.signal(signal, SIG_IGN)
            source.resume()
            return Handler(signal: signal, source: source)
        }
    }
}

/// A waiting acquisition's mark: a shared lock on the lease's `<name>.waiters` file, released when this is freed.
/// Shared, so any number of waiters coexist; a holder's exclusive probe (``SharedGPULease/hasWaiters(directory:legacyWrapper:)``)
/// fails while any is held. A file this user can't use leaves the waiter unmarked (it still waits).
final class WaiterMark: Sendable {
    private let descriptor: Int32?

    init(directory: URL) {
        guard let file = SharedGPULease.privateSibling(directory, suffix: ".waiters", create: true) else { descriptor = nil; return }
        // A holder's probe holds the exclusive lock only for an instant: try a few times rather than block.
        for _ in 0..<20 {
            if flock(file, LOCK_SH | LOCK_NB) == 0 { descriptor = file; return }
            usleep(1_000)
        }
        close(file)
        descriptor = nil
    }

    deinit { if let descriptor { close(descriptor) } }
}
