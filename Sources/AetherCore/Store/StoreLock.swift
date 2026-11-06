import Foundation
import Darwin

/// An advisory `flock` on one install location (`<repo>/<revision>`), shared by every `ModelStore` on the same root,
/// in this process or another. Ported from AetherKit 0.0.7's `LocalAssetStore` locks.
///
/// - an install holds it exclusively from before its first byte until the bundle is renamed into place;
/// - `remove` and stage recovery hold it exclusively;
/// - a reader lease (``BundleReadLease``) holds it shared for as long as the model uses the files.
///
/// The lock file is this user's single-link regular file, opened without following links; it stays in place after
/// release so every process keeps locking the same inode. The descriptor closes (and the lock drops) on deinit, and on
/// process exit.
final class StoreLock: Sendable {
    private let descriptor: Int32

    private init(descriptor: Int32) { self.descriptor = descriptor }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    /// One attempt; nil when another holder's lock conflicts.
    static func attempt(_ url: URL, exclusive: Bool) throws(AetherError) -> StoreLock? {
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        } catch {
            throw .runtime(underlying: error)
        }
        let fd = open(url.path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw .runtime(underlying: POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1 else {
            close(fd)
            throw .manifestInvalid(field: "store", reason: "'\(url.path)' is not a private lock file")
        }
        guard flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0 else {
            let code = errno
            close(fd)
            if code == EWOULDBLOCK { return nil }
            throw .runtime(underlying: POSIXError(POSIXErrorCode(rawValue: code) ?? .EIO))
        }
        return StoreLock(descriptor: fd)
    }

    /// Waits for an exclusive lock, polling so that cancellation stays prompt (a blocking `flock` would hold a thread).
    static func exclusive(_ url: URL, poll: Duration = .milliseconds(100)) async throws -> StoreLock {
        while true {
            if let lock = try attempt(url, exclusive: true) { return lock }
            try await Task.sleep(for: poll)
        }
    }
}

/// Keeps an installed bundle's files in place while a model uses them: ``ModelStore/remove(_:)`` (from any process
/// sharing the store root) refuses with ``AetherError/inUse(resource:)`` while a lease is held. Several leases may be
/// held at once. There is no release method: the owner of the loaded model keeps the lease, and dropping the last
/// reference releases it.
public final class BundleReadLease: Sendable {
    public let bundle: InstalledBundle
    private let lock: StoreLock

    init(bundle: InstalledBundle, lock: StoreLock) {
        self.bundle = bundle
        self.lock = lock
    }
}
