import CryptoKit
import Darwin
import Foundation

/// Immutable app-owned installations. This store copies only declared portable files. It does not
/// download models, modify external checkpoints, or manage Apple's specialization cache.
/// The root must be private to the host; advisory locks coordinate cooperating store instances/processes.
public struct LocalAssetStore: Sendable {
    public let root: URL
    private let onCopiedChunk: @Sendable () throws -> Void

    public init(root: URL) throws { try self.init(root: root, onCopiedChunk: {}) }

    // Controlled boundary for interruption/interleaving tests; not a public progress or scheduling API.
    init(root: URL, onCopiedChunk: @escaping @Sendable () throws -> Void) throws {
        self.onCopiedChunk = onCopiedChunk
        guard root.isFileURL else { throw AssetStoreError.unsafePath(root.path) }
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        try directory(self.root)
        for name in ["locks", "staging", "installed"] { try directory(self.root.appendingPathComponent(name)) }
    }

    /// Copies and hashes in a private stage, then atomically publishes. A retry verifies an existing
    /// installation instead of overwriting it. Cancellation waits for the worker to close its files/lock.
    public func install(_ manifest: AssetManifest, from source: URL) async throws -> String {
        try Task.checkCancellation()
        try manifest.validate()
        guard source.isFileURL else { throw AssetStoreError.unsafePath(source.path) }
        let worker = Task.detached { try installSynchronously(manifest, from: source) }
        return try await withTaskCancellationHandler { try await worker.value } onCancel: { worker.cancel() }
    }

    /// Acquires reader protection before reading metadata and hashing installed bytes. Keep the lease
    /// for the entire model lifetime. Separate store values and processes obey the same lock.
    public func acquire(_ installationID: String) async throws -> AssetReadLease {
        try Task.checkCancellation()
        let worker = Task.detached { () throws -> AssetReadLease in
            try Task.checkCancellation()
            let lock = try lock(installationID, exclusive: false)
            let destination = destination(installationID)
            let manifest = try verifiedManifest(at: destination, expectedID: installationID)
            try verify(manifest, at: destination.appendingPathComponent("files"))
            try Task.checkCancellation()
            return AssetReadLease(installationID: installationID, manifest: manifest,
                                  directory: destination.appendingPathComponent("files"), lock: lock)
        }
        return try await withTaskCancellationHandler {
            let lease = try await worker.value
            try Task.checkCancellation()
            return lease
        } onCancel: { worker.cancel() }
    }

    /// Fails with busy while another process is installing or any reader holds a lease.
    public func remove(_ installationID: String) throws {
        let held = try lock(installationID, exclusive: true)
        defer { withExtendedLifetime(held) {} }
        let target = destination(installationID)
        guard FileManager.default.fileExists(atPath: target.path) else { throw AssetStoreError.notInstalled }
        guard target.resolvingSymlinksInPath().path == target.path else { throw AssetStoreError.unsafePath(target.path) }
        // Explicit removal also recovers corrupt installations; their untrusted metadata is not needed.
        try FileManager.default.removeItem(at: target)
    }

    /// Removes recognizable unfinished stages only after acquiring their installation's exclusive lock.
    /// Active imports/readers are skipped. Unknown entries (including older UUID-only stages) are left
    /// untouched because their ownership cannot be proven. Does not remove published installations.
    /// The host must keep the store root private, as for install/acquire/remove.
    @discardableResult
    public func removeAbandonedStages() throws -> Int {
        try Task.checkCancellation()
        let staging = try confined("staging", under: root)
        let entries = try FileManager.default.contentsOfDirectory(atPath: staging.path).sorted()
        var removed = 0
        for name in entries {
            try Task.checkCancellation()
            let parts = name.split(separator: ".", omittingEmptySubsequences: false)
            guard parts.count == 2, parts[0].utf8.count == 64,
                  parts[0].utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
                  let uuid = UUID(uuidString: String(parts[1])), uuid.uuidString == parts[1] else { continue }
            let held: StoreLock
            do { held = try lock(String(parts[0]), exclusive: true) }
            catch AssetStoreError.busy { continue }
            defer { withExtendedLifetime(held) {} }
            // Recheck under the same lock used through install's publish/rollback boundary. Another
            // cleanup may already have removed the entry after our directory snapshot was taken.
            let stage = try confined(name, under: staging)
            let attributes: [FileAttributeKey: Any]
            do { attributes = try FileManager.default.attributesOfItem(atPath: stage.path) }
            catch let error as CocoaError where error.code == .fileReadNoSuchFile || error.code == .fileNoSuchFile { continue }
            guard attributes[.type] as? FileAttributeType == .typeDirectory else {
                throw AssetStoreError.unsafePath(stage.path)
            }
            try Task.checkCancellation()
            try FileManager.default.removeItem(at: stage)
            removed += 1
        }
        return removed
    }

    private func installSynchronously(_ manifest: AssetManifest, from source: URL) throws -> String {
        try Task.checkCancellation()
        let metadata = try manifest.canonicalData()
        guard metadata.count < 32 << 20 else { throw AssetStoreError.invalidManifest("installation metadata exceeds limit") }
        let id = digest(metadata)
        let held = try lock(id, exclusive: true)
        defer { withExtendedLifetime(held) {} }
        let target = destination(id)
        if FileManager.default.fileExists(atPath: target.path) {
            let existing = try verifiedManifest(at: target, expectedID: id)
            try verify(existing, at: target.appendingPathComponent("files"))
            try Task.checkCancellation()
            return id
        }
        let stage = root.appendingPathComponent("staging/\(id).\(UUID().uuidString)", isDirectory: true)
        try directory(stage)
        defer { try? FileManager.default.removeItem(at: stage) }
        let payload = stage.appendingPathComponent("files", isDirectory: true)
        try directory(payload)
        let source = source.standardizedFileURL.resolvingSymlinksInPath()
        for file in manifest.files {
            try Task.checkCancellation()
            let input = try confined(file.path, under: source)
            let output = payload.appendingPathComponent(file.path)
            try directory(output.deletingLastPathComponent())
            try stream(file, from: input, to: output, onCopiedChunk: onCopiedChunk)
        }
        try metadata.write(to: stage.appendingPathComponent("manifest.json"), options: .withoutOverwriting)
        try Task.checkCancellation()
        // POSIX rename is atomic on the same volume; no cancellation check after this commit point.
        guard rename(stage.path, target.path) == 0 else { throw AssetStoreError.fileSystem(errno) }
        return id
    }

    private func destination(_ id: String) -> URL { root.appendingPathComponent("installed/\(id)", isDirectory: true) }

    private func lock(_ id: String, exclusive: Bool) throws -> StoreLock {
        guard id.utf8.count == 64, id.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw AssetStoreError.invalidManifest("invalid installation identity")
        }
        return try StoreLock(url: root.appendingPathComponent("locks/\(id)"), exclusive: exclusive)
    }

    private func verifiedManifest(at target: URL, expectedID: String) throws -> AssetManifest {
        guard FileManager.default.fileExists(atPath: target.path) else { throw AssetStoreError.notInstalled }
        let url = try confined("manifest.json", under: target)
        let input = try openRegular(url)
        defer { try? input.close() }
        guard let bytes = try input.read(upToCount: 32 << 20), bytes.count < 32 << 20 else {
            throw AssetStoreError.invalidManifest("installation metadata exceeds limit")
        }
        let manifest = try JSONDecoder().decode(AssetManifest.self, from: bytes)
        guard try manifest.installationID == expectedID else { throw AssetStoreError.integrityMismatch("manifest") }
        return manifest
    }

    private func verify(_ manifest: AssetManifest, at payload: URL) throws {
        for file in manifest.files { try stream(file, from: confined(file.path, under: payload), to: nil) }
    }
}

/// No explicit release method: native/model ownership should retain this object. The last reference
/// closes the shared lock, so callers cannot accidentally release protection while another owner uses it.
public final class AssetReadLease: Sendable {
    public let installationID: String
    public let manifest: AssetManifest
    public let directory: URL
    private let lock: StoreLock
    fileprivate init(installationID: String, manifest: AssetManifest, directory: URL, lock: StoreLock) {
        self.installationID = installationID; self.manifest = manifest; self.directory = directory; self.lock = lock
    }
}

private final class StoreLock: Sendable {
    private let descriptor: Int32
    init(url: URL, exclusive: Bool) throws {
        let fd = open(url.path, O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw AssetStoreError.fileSystem(errno) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_nlink == 1 else {
            close(fd); throw AssetStoreError.unsafePath(url.path)
        }
        guard flock(fd, (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB) == 0 else {
            let code = errno; close(fd)
            if code == EWOULDBLOCK { throw AssetStoreError.busy }
            throw AssetStoreError.fileSystem(code)
        }
        descriptor = fd
    }
    deinit { flock(descriptor, LOCK_UN); close(descriptor) }
}

private func directory(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    guard url.standardizedFileURL.path == url.standardizedFileURL.resolvingSymlinksInPath().path else {
        throw AssetStoreError.unsafePath(url.path)
    }
}

private func confined(_ path: String, under root: URL) throws -> URL {
    guard AssetManifest.validPath(path) else { throw AssetStoreError.unsafePath(path) }
    let candidate = root.appendingPathComponent(path).standardizedFileURL
    guard candidate.resolvingSymlinksInPath().path == candidate.path else { throw AssetStoreError.unsafePath(path) }
    return candidate
}

private func openRegular(_ url: URL) throws -> FileHandle {
    let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
    guard fd >= 0 else { throw AssetStoreError.fileSystem(errno) }
    var info = stat()
    guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG else {
        close(fd); throw AssetStoreError.unsafePath(url.path)
    }
    return FileHandle(fileDescriptor: fd, closeOnDealloc: true)
}

private func stream(_ file: AssetFile, from source: URL, to destination: URL?,
                    onCopiedChunk: @Sendable () throws -> Void = {}) throws {
    let input = try openRegular(source)
    defer { try? input.close() }
    var output: FileHandle?
    if let destination {
        let fd = open(destination.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw AssetStoreError.fileSystem(errno) }
        output = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    }
    defer { try? output?.close() }
    var hash = SHA256(), remaining = file.byteCount
    while remaining > 0 {
        try Task.checkCancellation()
        try autoreleasepool {
            guard let bytes = try input.read(upToCount: Int(min(remaining, 1 << 20))), !bytes.isEmpty else {
                throw AssetStoreError.integrityMismatch(file.path)
            }
            hash.update(data: bytes)
            try output?.write(contentsOf: bytes)
            remaining -= UInt64(bytes.count)
        }
        try onCopiedChunk()
    }
    guard try input.read(upToCount: 1)?.isEmpty != false,
          hash.finalize().map({ String(format: "%02x", $0) }).joined() == file.sha256 else {
        throw AssetStoreError.integrityMismatch(file.path)
    }
    try output?.synchronize()
}
