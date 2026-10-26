import Darwin
import Dispatch
import Foundation
import Synchronization

/// Failures of the cross-process GPU lease. `holder` is the process the lease names, when it names one.
public enum SharedGPULeaseError: Error, Sendable, Equatable, CustomStringConvertible {
    /// Another live process (or this process) holds the lease.
    case held(holder: Int32?)
    /// A waiting acquisition reached its timeout while the lease was still held.
    case timedOut(holder: Int32?)
    /// The lease directory could not be created for a reason other than an existing lease.
    case unavailable(path: String, errno: Int32)
    /// A tool environment variable has a value the policy does not recognize.
    case invalidEnvironment(String)

    public var description: String {
        switch self {
        case .held(let holder): "The shared GPU lease is held by \(holder.map { "process \($0)" } ?? "an unidentified holder")."
        case .timedOut(let holder): "Timed out waiting for the shared GPU lease held by \(holder.map { "process \($0)" } ?? "an unidentified holder")."
        case .unavailable(let path, let code): "Cannot create the GPU lease at \(path): \(String(cString: strerror(code)))."
        case .invalidEnvironment(let reason): "Invalid GPU lease environment: \(reason)"
        }
    }
}

/// A cooperative, cross-process GPU lease, wire-compatible with Atelier's `SharedGPULock` and the lab wrappers.
///
/// The lease is a directory created atomically with `mkdir`; its holder writes its decimal pid, without a newline,
/// to `pid` inside it (written to a temporary file and renamed into place). Release removes the directory only while
/// `pid` still names this process. A lease whose pid names a process that no longer exists is reclaimed: reclaimers
/// serialize on a sibling `<name>.reclaim` flock file (this user's single-link regular file, mode 0600), move the
/// lease aside atomically, delete the moved copy only if it still names the same dead process and otherwise restore
/// it without replacing any lease created since. A missing or unreadable pid (a holder between `mkdir` and its pid
/// write) is never stale.
///
/// Default-handled SIGINT, SIGTERM and SIGHUP release held leases before the process terminates; a signal the process
/// already handles or ignores is left alone. Releasing is idempotent and also happens on deinit.
///
/// The lease coordinates only processes that share the directory. Sandboxed apps get a private temporary directory,
/// so there the lease excludes nothing outside the app. Nothing here is a process-global default: callers choose the
/// directory, and Kit tools pass it explicitly through ``ToolPolicy``.
public final class SharedGPULease: Sendable {
    /// The directory Atelier, the lab scripts and Kit tools share on one Mac.
    public static let sharedDirectory = URL(fileURLWithPath: "/tmp/atelier-gpu.lock", isDirectory: true)

    /// How this process holds the lease.
    public enum Holding: Sendable, Equatable {
        /// This process created the lease and its pid file names this process.
        case owned
        /// An ancestor process (for example a wrapper running this process as its GPU job) holds the lease.
        /// Release leaves it alone.
        case inherited(holder: Int32)
    }

    /// Where and how to acquire.
    public struct Configuration: Sendable, Equatable {
        public var directory: URL
        /// Treat a lease held by an ancestor process as held on this process's behalf. Kit tools enable this so a
        /// command run under a wrapper that already holds the lease does not wait for its own parent.
        public var inheritsAncestorLease: Bool
        /// How often a waiting acquisition retries.
        public var pollInterval: Duration

        public init(directory: URL = SharedGPULease.sharedDirectory, inheritsAncestorLease: Bool = false,
                    pollInterval: Duration = .milliseconds(200)) {
            self.directory = directory
            self.inheritsAncestorLease = inheritsAncestorLease
            self.pollInterval = pollInterval
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

    /// One attempt: acquires, reclaims a stale lease, or throws ``SharedGPULeaseError/held(holder:)``.
    public static func tryAcquire(_ configuration: Configuration = Configuration()) throws -> SharedGPULease {
        switch try attempt(configuration) {
        case .acquired(let lease): return lease
        case .busy(let holder): throw SharedGPULeaseError.held(holder: holder)
        }
    }

    /// Waits until the lease is free, retrying every `pollInterval`. A `nil` timeout waits indefinitely; a zero timeout
    /// makes one attempt. Cancelling the calling task stops waiting with `CancellationError`. `onBusy` is called when
    /// the lease is first found held and whenever its holder changes, for progress messages.
    public static func acquire(
        _ configuration: Configuration = Configuration(), timeout: Duration? = nil,
        onBusy: (@Sendable (Int32?) -> Void)? = nil
    ) async throws -> SharedGPULease {
        let clock = ContinuousClock()
        let deadline = timeout.map { clock.now.advanced(by: max(.zero, $0)) }
        var reported: Int32?? = .none
        while true {
            try Task.checkCancellation()
            let holder: Int32?
            switch try attempt(configuration) {
            case .acquired(let lease): return lease
            case .busy(let current): holder = current
            }
            if reported != .some(holder) { reported = .some(holder); onBusy?(holder) }
            if let deadline {
                let now = clock.now
                guard now < deadline else { throw SharedGPULeaseError.timedOut(holder: holder) }
                try await Task.sleep(for: min(configuration.pollInterval, now.duration(to: deadline)))
            } else {
                try await Task.sleep(for: configuration.pollInterval)
            }
        }
    }

    /// The process an existing lease names, if its pid file is present and well formed.
    public static func holder(of directory: URL = sharedDirectory) -> Int32? { readHolder(directory) }

    /// Removes the lease only if this object created it and the pid file still names this process. Idempotent.
    public func release() {
        held.withLock { value in
            guard value else { return }
            value = false
            if Self.readHolder(directory) == owner { try? FileManager.default.removeItem(at: directory) }
        }
        LeaseSignalRelease.unregister(signalToken)
    }

    deinit { release() }

    // MARK: Protocol

    private enum Attempt { case acquired(SharedGPULease), busy(Int32?) }

    private static func attempt(_ configuration: Configuration) throws -> Attempt {
        let directory = configuration.directory
        if let lease = try create(directory) { return .acquired(lease) }
        let holder = readHolder(directory)
        if configuration.inheritsAncestorLease, let holder, isAncestor(holder) {
            return .acquired(SharedGPULease(directory: directory, holding: .inherited(holder: holder)))
        }
        guard clearStale(directory) else { return .busy(holder) }
        if let lease = try create(directory) { return .acquired(lease) }
        return .busy(readHolder(directory))
    }

    /// `nil` when the directory already exists.
    private static func create(_ directory: URL) throws -> SharedGPULease? {
        guard mkdir(directory.path, 0o755) == 0 else {
            let code = errno
            if code == EEXIST { return nil }
            throw SharedGPULeaseError.unavailable(path: directory.path, errno: code)
        }
        let pid = getpid()
        let temporary = directory.appendingPathComponent("pid.tmp.\(pid)").path
        let final = directory.appendingPathComponent("pid").path
        let written: Bool = {
            let descriptor = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o644)
            guard descriptor >= 0 else { return false }
            let bytes = Array(String(pid).utf8)
            let count = bytes.withUnsafeBytes { write(descriptor, $0.baseAddress, $0.count) }
            return close(descriptor) == 0 && count == bytes.count && rename(temporary, final) == 0
        }()
        guard written else {
            let code = errno
            try? FileManager.default.removeItem(at: directory)
            throw SharedGPULeaseError.unavailable(path: directory.path, errno: code)
        }
        let lease = SharedGPULease(directory: directory, holding: .owned)
        LeaseSignalRelease.register(lease.signalToken) { [weak lease] in lease?.release() }
        return lease
    }

    static func readHolder(_ directory: URL) -> pid_t? {
        guard let text = try? String(contentsOf: directory.appendingPathComponent("pid"), encoding: .utf8),
              let pid = pid_t(text.trimmingCharacters(in: .whitespacesAndNewlines)), pid > 0 else { return nil }
        return pid
    }

    /// True only when signal 0 reports that no such process exists. A live process another user owns (EPERM) is not stale.
    static func isDead(_ pid: pid_t) -> Bool { kill(pid, 0) != 0 && errno == ESRCH }

    /// Walks this process's parent chain. launchd (pid 1) is never treated as a holder's descendant relationship.
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

    static func parentProcess(of pid: pid_t) -> pid_t? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, pid]
        guard sysctl(&name, 4, &info, &size, nil, 0) == 0, size == MemoryLayout<kinfo_proc>.stride else { return nil }
        let parent = info.kp_eproc.e_ppid
        return parent > 0 ? parent : nil
    }

    /// Atelier's stale reclaim, step for step. Returns true only when it removed a lease naming a dead process.
    /// `beforeMove` runs between the holder check and the move aside (tests use it to model a racing replacement).
    static func clearStale(_ directory: URL, beforeMove: () -> Void = {}) -> Bool {
        let parent = directory.deletingLastPathComponent()
        let reclaim = parent.appendingPathComponent(directory.lastPathComponent + ".reclaim").path
        let serial = open(reclaim, O_RDONLY | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard serial >= 0 else { return false }
        defer { close(serial) }
        // /tmp is shared: a reclaim file another user created could be held forever to block every reclaim, and a
        // hard link to one of this user's files would pass the uid check. Use only this user's single-link regular
        // file, made private if an older build created it 0644; otherwise report the lease as held.
        var owner = stat()
        guard fstat(serial, &owner) == 0, owner.st_mode & S_IFMT == S_IFREG, owner.st_uid == getuid(),
              owner.st_nlink == 1 else { return false }
        if owner.st_mode & 0o077 != 0 {
            guard fchmod(serial, 0o600) == 0, fstat(serial, &owner) == 0, owner.st_mode & 0o077 == 0 else { return false }
        }
        // Another reclaimer is mid-reclaim; report the lease as held rather than wait.
        guard flock(serial, LOCK_EX | LOCK_NB) == 0 else { return false }
        guard let dead = readHolder(directory), isDead(dead) else { return false }
        beforeMove()
        let aside = parent.appendingPathComponent(directory.lastPathComponent + ".abandoned-\(UUID().uuidString)")
        guard rename(directory.path, aside.path) == 0 else { return false }
        guard readHolder(aside) == dead else {
            // The lease changed between the read and the move (only an outside remover can free the path while the
            // reclaim lock is held). Never replace a lease created since.
            if renamex_np(aside.path, directory.path, UInt32(RENAME_EXCL)) != 0 {
                FileHandle.standardError.write(Data("aetherkit: could not restore a GPU lease; left \(aside.path)\n".utf8))
            }
            return false
        }
        return (try? FileManager.default.removeItem(at: aside)) != nil
    }
}

extension SharedGPULease {
    /// The environment contract Kit tools (`aether-probe`, the benchmark script, the opt-in native suites) use.
    ///
    /// - `AETHER_GPU_LEASE`: unset, `1` or `on` takes the lease; `0` or `off` disables it for special cases.
    /// - `AETHER_GPU_LEASE_PATH`: an absolute lease directory instead of ``sharedDirectory``.
    /// - `AETHER_GPU_LEASE_TIMEOUT`: seconds to wait; `0` fails fast. Unset waits indefinitely.
    ///
    /// Tools always inherit a lease an ancestor process holds, so they compose with wrappers that hold it already.
    public struct ToolPolicy: Sendable, Equatable {
        /// `nil` when the lease is disabled.
        public var configuration: Configuration?
        public var timeout: Duration?

        public init(configuration: Configuration?, timeout: Duration?) {
            self.configuration = configuration
            self.timeout = timeout
        }

        public static func environment(_ environment: [String: String] = ProcessInfo.processInfo.environment) throws -> ToolPolicy {
            switch environment["AETHER_GPU_LEASE"]?.lowercased() {
            case nil, "", "1", "on": break
            case "0", "off": return ToolPolicy(configuration: nil, timeout: nil)
            case let value?: throw SharedGPULeaseError.invalidEnvironment("AETHER_GPU_LEASE=\(value); use on or off")
            }
            var configuration = Configuration(inheritsAncestorLease: true)
            if let path = environment["AETHER_GPU_LEASE_PATH"], !path.isEmpty {
                guard path.hasPrefix("/") else {
                    throw SharedGPULeaseError.invalidEnvironment("AETHER_GPU_LEASE_PATH must be absolute")
                }
                configuration.directory = URL(fileURLWithPath: path, isDirectory: true)
            }
            var timeout: Duration?
            if let text = environment["AETHER_GPU_LEASE_TIMEOUT"], !text.isEmpty {
                guard let seconds = Double(text), seconds.isFinite, seconds >= 0 else {
                    throw SharedGPULeaseError.invalidEnvironment("AETHER_GPU_LEASE_TIMEOUT must be non-negative seconds")
                }
                timeout = .milliseconds(Int64((seconds * 1000).rounded()))
            }
            return ToolPolicy(configuration: configuration, timeout: timeout)
        }

        /// `nil` when disabled; otherwise an owned or inherited lease.
        public func acquire(onBusy: (@Sendable (Int32?) -> Void)? = nil) async throws -> SharedGPULease? {
            guard let configuration else { return nil }
            return try await SharedGPULease.acquire(configuration, timeout: timeout, onBusy: onBusy)
        }
    }
}

/// Signals are process-wide, so one handler set serves every lease this process owns. Handlers are installed when
/// the first lease registers, only for signals still at their default disposition, and removed when the last lease
/// releases (restoring only this code's own `SIG_IGN`, never a handler installed since). On a signal, every
/// registered lease releases and the signal is re-raised with its default action.
enum LeaseSignalRelease {
    private struct State {
        var releases: [UUID: @Sendable () -> Void] = [:]
        var sources: [(signal: Int32, source: any DispatchSourceSignal)] = []
    }
    private static let state = Mutex(State())

    static func register(_ token: UUID, release: @escaping @Sendable () -> Void) {
        state.withLock { state in
            if state.releases.isEmpty { state.sources = install() }
            state.releases[token] = release
        }
    }

    static func unregister(_ token: UUID) {
        state.withLock { state in
            guard state.releases.removeValue(forKey: token) != nil, state.releases.isEmpty else { return }
            for handler in state.sources {
                handler.source.cancel()
                var current = sigaction()
                if sigaction(handler.signal, nil, &current) == 0,
                   unsafeBitCast(current.__sigaction_u.__sa_handler, to: Int.self) == unsafeBitCast(SIG_IGN, to: Int.self) {
                    Darwin.signal(handler.signal, SIG_DFL)
                }
            }
            state.sources = []
        }
    }

    private static func install() -> [(signal: Int32, source: any DispatchSourceSignal)] {
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
            return (signal, source)
        }
    }
}
