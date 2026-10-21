import Darwin
import Foundation

public enum ExternalFileError: Error, Sendable, Equatable {
    case invalidFile
    case notLeased
    case invalidRange
    case fileChanged
    case busy
    case fileSystem(Int32)
}

/// Pins read-only file descriptors for caller-owned immutable files; no registration, copies or deletion.
/// Reads use pread on these descriptors, so path replacement cannot redirect a leased read. Shared advisory
/// locks protect against cooperating writers; other cache tools need not honor them. Size/mtime changes
/// are refused, but callers must still use content hashes and keep the files immutable during use.
public final class ExternalFileLease: Sendable {
    private let bindings: [String: PinnedExternalFile]

    public init(files: [URL]) throws {
        guard !files.isEmpty, files.count <= 100_000 else { throw ExternalFileError.invalidFile }
        var bindings: [String: PinnedExternalFile] = [:]
        var canonical: [String: PinnedExternalFile] = [:]
        for url in files {
            try Task.checkCancellation()
            let key = try Self.key(url)
            if bindings[key] != nil { continue }
            let resolved = url.standardizedFileURL.resolvingSymlinksInPath()
            let file: PinnedExternalFile
            if let existing = canonical[resolved.path] { file = existing }
            else {
                file = try PinnedExternalFile(resolved)
                canonical[resolved.path] = file
            }
            bindings[key] = file
        }
        self.bindings = bindings
    }

    public func byteCount(of url: URL) throws -> UInt64 { try file(url).validatedByteCount() }

    /// No intermediate Data allocation. Concurrent calls have independent offsets; the pointer never escapes.
    public func read(into destination: UnsafeMutableRawBufferPointer, from url: URL, offset: UInt64) throws {
        try Task.checkCancellation()
        let file = try file(url), size = try file.validatedByteCount()
        guard offset <= size, UInt64(destination.count) <= size - offset else { throw ExternalFileError.invalidRange }
        if destination.isEmpty { return }
        guard let address = destination.baseAddress else { throw ExternalFileError.invalidRange }
        var done = 0
        while done < destination.count {
            try Task.checkCancellation()
            let count = pread(file.descriptor, address + done, min(destination.count - done, 64 << 20), off_t(offset + UInt64(done)))
            if count < 0, errno == EINTR { continue }
            guard count > 0 else {
                if count == 0 { throw ExternalFileError.invalidRange }
                throw ExternalFileError.fileSystem(errno)
            }
            done += count
        }
        _ = try file.validatedByteCount()
        try Task.checkCancellation()
    }

    private func file(_ url: URL) throws -> PinnedExternalFile {
        guard let file = bindings[try Self.key(url)] else { throw ExternalFileError.notLeased }
        return file
    }
    private static func key(_ url: URL) throws -> String {
        guard url.isFileURL, url.host == nil || url.host == "" || url.host == "localhost",
              url.query == nil, url.fragment == nil, !url.path.utf8.contains(0) else { throw ExternalFileError.invalidFile }
        return url.standardizedFileURL.path
    }
}

private final class PinnedExternalFile: Sendable {
    struct Identity: Equatable, Sendable {
        let device: dev_t
        let inode: ino_t
        let size: Int64
        let modifiedSeconds: Int
        let modifiedNanoseconds: Int
        init(_ info: stat) {
            device = info.st_dev; inode = info.st_ino; size = info.st_size
            modifiedSeconds = info.st_mtimespec.tv_sec; modifiedNanoseconds = info.st_mtimespec.tv_nsec
        }
    }
    let descriptor: Int32
    private let identity: Identity
    init(_ url: URL) throws {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw ExternalFileError.fileSystem(errno) }
        do {
            guard flock(fd, LOCK_SH | LOCK_NB) == 0 else {
                if errno == EWOULDBLOCK { throw ExternalFileError.busy }
                throw ExternalFileError.fileSystem(errno)
            }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size >= 0 else {
                throw ExternalFileError.invalidFile
            }
            descriptor = fd; identity = Identity(info)
        } catch { close(fd); throw error }
    }
    func validatedByteCount() throws -> UInt64 {
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw ExternalFileError.fileSystem(errno) }
        guard Identity(info) == identity else { throw ExternalFileError.fileChanged }
        return UInt64(identity.size)
    }
    deinit { flock(descriptor, LOCK_UN); close(descriptor) }
}
