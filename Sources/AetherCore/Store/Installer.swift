import CryptoKit
import Foundation

/// One install: every asset file of a variant into staging (with resume and SHA-256 verification), the receipt,
/// then one rename into place. Runs off the store's actor.
struct Installer: Sendable {
    /// One file to fetch: where it lands (relative to the install root) and where it comes from.
    struct Item: Sendable {
        var asset: String
        var file: CatalogFile
        var local: String
        var remote: URL
    }

    let transport: any HTTPTransport
    let ref: ModelRef
    let items: [Item]
    let staging: URL
    let destination: URL

    private static let progressGranularity: Int64 = 1 << 20

    init(transport: any HTTPTransport, endpoint: URL, ref: ModelRef, staging: URL, destination: URL) throws {
        self.transport = transport
        self.ref = ref
        self.staging = staging
        self.destination = destination
        var items: [Item] = []
        var byLocal: [String: String] = [:]
        for asset in ref.assets {
            for file in asset.files {
                try Self.checkPath(file.path)
                let root = InstalledLayout.assetRoot(asset, mainRepo: ref.repo, in: URL(fileURLWithPath: "/"))
                let local = String(root.appending(path: file.path).path.dropFirst())
                // Two assets of one repo may overlap; the same file twice is fine, two different files are not.
                if let previous = byLocal[local] {
                    guard previous == file.sha256 else {
                        throw AetherError.manifestInvalid(field: "assets.\(asset.name)", reason: "'\(local)' is listed twice with different hashes")
                    }
                    continue
                }
                byLocal[local] = file.sha256
                var remote = endpoint.appending(path: asset.repo).appending(path: "resolve").appending(path: asset.revision)
                for component in (asset.subtree + "/" + file.path).split(separator: "/") {
                    remote = remote.appending(path: String(component))
                }
                items.append(Item(asset: asset.name, file: file, local: local, remote: remote))
            }
        }
        self.items = items
    }

    /// Downloads every file into `staging`, verifies each SHA-256, writes the receipt, and renames the directory to
    /// `destination`. Returns `destination`.
    @concurrent
    func run(progress: @escaping @Sendable (InstallProgress) -> Void) async throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try Self.excludeFromBackup(staging)

        let total = items.reduce(Int64(0)) { $0 + $1.file.size }
        var done: Int64 = 0
        progress(InstallProgress(completedBytes: 0, totalBytes: total, file: nil))
        for item in items {
            try Task.checkCancellation()
            try await fetch(item, before: done, total: total, progress: progress)
            done += item.file.size
            progress(InstallProgress(completedBytes: done, totalBytes: total, file: item.local))
        }

        let receipt = InstallReceipt(ref: ref, installedAt: Date())
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let receiptURL = InstallReceipt.url(in: staging)
        try fm.createDirectory(at: receiptURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try encoder.encode(receipt).write(to: receiptURL, options: .atomic)

        // A directory at `destination` without a receipt is an interrupted earlier attempt: replace it.
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: staging, to: destination)  // rename(2): atomic on one volume
        try Self.excludeFromBackup(destination)
        return destination
    }

    /// Brings one staged file to its verified final state, resuming a partial download when there is one.
    private func fetch(
        _ item: Item, before done: Int64, total: Int64, progress: @escaping @Sendable (InstallProgress) -> Void
    ) async throws {
        let local = staging.appending(path: item.local)
        let file = item.file
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
                try await download(item, to: local, from: size, before: done, total: total, progress: progress)
            default:
                if existing != nil { try fm.removeItem(at: local) }
                resumed = false
                try await download(item, to: local, from: 0, before: done, total: total, progress: progress)
            }

            let actual = try Self.sha256(of: local)
            if actual == file.sha256.lowercased() { return }
            try? fm.removeItem(at: local)
            // A resumed file may have had a corrupt prefix: fetch it once from scratch before giving up.
            if resumed && allowRetry {
                allowRetry = false
                continue
            }
            throw AetherError.integrity(file: item.local, expected: file.sha256, actual: actual)
        }
    }

    private func download(
        _ item: Item, to local: URL, from offset: Int64, before done: Int64, total: Int64,
        progress: @escaping @Sendable (InstallProgress) -> Void
    ) async throws {
        let file = item.file
        var headers: [String: String] = [:]
        if offset > 0 { headers["Range"] = "bytes=\(offset)-" }

        let head: HTTPResponseHead
        let body: AsyncThrowingStream<Data, any Error>
        do {
            (head, body) = try await transport.get(item.remote, headers: headers)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw AetherError.download(file: item.local, reason: .transport(Self.describe(error)))
        }

        let start: Int64
        switch head.status {
        case 206 where offset > 0:
            guard let rangeStart = Self.contentRangeStart(head.headers["content-range"]), rangeStart == offset else {
                throw AetherError.download(
                    file: item.local,
                    reason: .unexpectedContent("Content-Range '\(head.headers["content-range"] ?? "")' does not start at \(offset)"))
            }
            start = offset
        case 200:
            start = 0  // the server ignored the range: start over
        default:
            throw AetherError.download(file: item.local, reason: .http(status: head.status))
        }

        if start == 0 { FileManager.default.createFile(atPath: local.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: local)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(start))
        try handle.seekToEnd()

        var written = start
        var reported = start
        do {
            for try await chunk in body {
                guard written + Int64(chunk.count) <= file.size else {
                    throw AetherError.download(file: item.local, reason: .unexpectedContent("more than the catalog's \(file.size) bytes"))
                }
                try handle.write(contentsOf: chunk)
                written += Int64(chunk.count)
                if written - reported >= Self.progressGranularity {
                    reported = written
                    progress(InstallProgress(completedBytes: done + written, totalBytes: total, file: item.local))
                }
            }
        } catch let error as AetherError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // The bytes so far stay in staging; the next attempt resumes from them.
            throw AetherError.download(file: item.local, reason: .transport(Self.describe(error)))
        }
        guard written == file.size else {
            throw AetherError.download(file: item.local, reason: .transport("connection closed after \(written) of \(file.size) bytes"))
        }
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

    /// Catalog paths are relative and may not climb out of the asset.
    static func checkPath(_ path: String) throws {
        guard isSafeRelativePath(path) else {
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
