import CryptoKit
import Foundation

/// One bundle download: staging, resume, verification, and the final rename. Runs off the store's actor.
struct Installer: Sendable {
    let transport: any HTTPTransport
    let endpoint: URL
    let ref: ModelRef
    let files: [CatalogFile]
    let staging: URL
    let destination: URL

    private static let progressGranularity: Int64 = 1 << 20

    /// Downloads every file into `staging`, verifies each SHA-256, writes the receipt, and renames the directory
    /// to `destination`. Returns `destination`.
    @concurrent
    func run(progress: @escaping @Sendable (InstallProgress) -> Void) async throws -> URL {
        let fm = FileManager.default
        for file in files { try Self.checkPath(file.path) }
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try Self.excludeFromBackup(staging)

        let total = files.reduce(Int64(0)) { $0 + $1.size }
        var done: Int64 = 0
        progress(InstallProgress(completedBytes: 0, totalBytes: total, file: nil))
        for file in files {
            try Task.checkCancellation()
            try await fetch(file, before: done, total: total, progress: progress)
            done += file.size
            progress(InstallProgress(completedBytes: done, totalBytes: total, file: file.path))
        }

        let receipt = InstallReceipt(ref: ref, files: files, installedAt: Date())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(receipt).write(to: staging.appending(path: InstallReceipt.fileName), options: .atomic)

        // A directory at `destination` without a receipt is an interrupted earlier attempt: replace it.
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: staging, to: destination)  // rename(2): atomic on one volume
        try Self.excludeFromBackup(destination)
        return destination
    }

    /// Brings one staged file to its verified final state, resuming a partial download when there is one.
    private func fetch(
        _ file: CatalogFile, before done: Int64, total: Int64,
        progress: @escaping @Sendable (InstallProgress) -> Void
    ) async throws {
        let local = staging.appending(path: file.path)
        let fm = FileManager.default
        try fm.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)

        var allowRetry = true
        while true {
            let existing = (try? fm.attributesOfItem(atPath: local.path)[.size] as? Int64) ?? nil
            let resumed: Bool
            switch existing {
            case let size? where size == file.size:
                resumed = true  // complete from an earlier attempt; verify below
            case let size? where size > 0 && size < file.size:
                resumed = true
                try await download(file, to: local, from: size, before: done, total: total, progress: progress)
            default:
                if existing != nil { try fm.removeItem(at: local) }
                resumed = false
                try await download(file, to: local, from: 0, before: done, total: total, progress: progress)
            }

            let actual = try Self.sha256(of: local)
            if actual == file.sha256.lowercased() { return }
            try? fm.removeItem(at: local)
            // A resumed file may have had a corrupt prefix: fetch it once from scratch before giving up.
            if resumed && allowRetry {
                allowRetry = false
                continue
            }
            throw AetherError.integrity(file: file.path, expected: file.sha256, actual: actual)
        }
    }

    private func download(
        _ file: CatalogFile, to local: URL, from offset: Int64, before done: Int64, total: Int64,
        progress: @escaping @Sendable (InstallProgress) -> Void
    ) async throws {
        var headers: [String: String] = [:]
        if offset > 0 { headers["Range"] = "bytes=\(offset)-" }
        let url = remoteURL(for: file.path)

        let head: HTTPResponseHead
        let body: AsyncThrowingStream<Data, any Error>
        do {
            (head, body) = try await transport.get(url, headers: headers)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw AetherError.download(file: file.path, reason: .transport(Self.describe(error)))
        }

        var start: Int64
        switch head.status {
        case 206 where offset > 0:
            guard let rangeStart = Self.contentRangeStart(head.headers["content-range"]), rangeStart == offset else {
                throw AetherError.download(
                    file: file.path, reason: .unexpectedContent("Content-Range '\(head.headers["content-range"] ?? "")' does not start at \(offset)"))
            }
            start = offset
        case 200:
            start = 0  // the server ignored the range: start over
        default:
            throw AetherError.download(file: file.path, reason: .http(status: head.status))
        }

        if start == 0 {
            FileManager.default.createFile(atPath: local.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: local)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(start))
        try handle.seekToEnd()

        var written = start
        var reported = start
        do {
            for try await chunk in body {
                guard written + Int64(chunk.count) <= file.size else {
                    throw AetherError.download(
                        file: file.path, reason: .unexpectedContent("more than the catalog's \(file.size) bytes"))
                }
                try handle.write(contentsOf: chunk)
                written += Int64(chunk.count)
                if written - reported >= Self.progressGranularity {
                    reported = written
                    progress(InstallProgress(completedBytes: done + written, totalBytes: total, file: file.path))
                }
            }
        } catch let error as AetherError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // The bytes so far stay in staging; the next attempt resumes from them.
            throw AetherError.download(file: file.path, reason: .transport(Self.describe(error)))
        }
        guard written == file.size else {
            throw AetherError.download(
                file: file.path, reason: .transport("connection closed after \(written) of \(file.size) bytes"))
        }
    }

    func remoteURL(for path: String) -> URL {
        var url = endpoint.appending(path: ref.repo).appending(path: "resolve").appending(path: ref.revision)
        for component in (ref.subtree + "/" + path).split(separator: "/") {
            url = url.appending(path: String(component))
        }
        return url
    }

    // MARK: - Helpers

    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func excludeFromBackup(_ url: URL) throws {
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var url = url
        try url.setResourceValues(values)
    }

    /// Catalog paths are relative and may not climb out of the bundle.
    static func checkPath(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.hasPrefix("/"), !components.contains(where: { $0 == ".." || $0 == "." || $0.isEmpty })
        else {
            throw AetherError.manifestInvalid(field: "files", reason: "'\(path)' is not a safe relative path")
        }
    }

    /// `bytes 100-199/200` → 100.
    static func contentRangeStart(_ header: String?) -> Int64? {
        guard let header, header.hasPrefix("bytes ") else { return nil }
        return header.dropFirst(6).split(separator: "-").first.flatMap { Int64($0) }
    }

    static func describe(_ error: any Error) -> String {
        if let url = error as? URLError { return "URLError \(url.code.rawValue): \(url.localizedDescription)" }
        return "\(error)"
    }
}
