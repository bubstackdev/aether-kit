import CryptoKit
import Foundation
import Synchronization

/// One install: every asset file of a variant into staging (with resume and SHA-256 verification), the receipt,
/// then one rename into place. Runs off the store's actor, under the install location's exclusive ``StoreLock``.
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
    let layout: StoreLayout
    let headers: @Sendable (URL) -> [String: String]
    let diskSpace: @Sendable (URL) throws -> Int64

    private static let progressGranularity: Int64 = 1 << 20

    init(
        transport: any HTTPTransport, endpoint: URL, ref: ModelRef, layout: StoreLayout,
        headers: @escaping @Sendable (URL) -> [String: String], diskSpace: @escaping @Sendable (URL) throws -> Int64
    ) throws {
        self.transport = transport
        self.ref = ref
        self.layout = layout
        self.staging = layout.staging(for: ref)
        self.destination = layout.bundle(for: ref)
        self.headers = headers
        self.diskSpace = diskSpace
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
    ///
    /// Holds the install location's lock throughout, so two processes never fetch into the same staging directory. A
    /// process that waited for the lock and finds the bundle installed (another process finished it) returns at once.
    @concurrent
    func run(progress: @escaping @Sendable (InstallProgress) -> Void) async throws -> URL {
        let lock = try await StoreLock.exclusive(layout.lock(for: ref))
        defer { withExtendedLifetime(lock) {} }
        if let receipt = InstallReceipt.read(at: destination), receipt.ref.variant == ref.variant { return destination }

        let fm = FileManager.default
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        try Self.excludeFromBackup(staging)

        let total = items.reduce(Int64(0)) { $0 + $1.file.size }
        // Refuse before the first byte if the volume can't hold what is still missing. Install is a rename, so the
        // missing bytes are all it needs.
        let staged = items.reduce(Int64(0)) { sum, item in
            let size = Self.size(of: staging.appending(path: item.local)) ?? 0
            return sum + (size <= item.file.size ? size : 0)
        }
        let needed = total - staged
        if needed > 0 {
            let available = try diskSpace(staging)
            guard available >= needed else { throw AetherError.capacity(what: "disk space", needed: needed, available: available) }
        }

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

        // A directory at `destination` without this variant's receipt is an interrupted or superseded earlier
        // attempt: move it out of the way in one rename, so a crash never leaves half of it looking installed.
        try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: destination.path) { try layout.discard(destination) }
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

        var refetchAllowed = true
        var restartAllowed = true
        while true {
            let existing = Self.size(of: local)
            let resumed: Bool
            switch existing {
            case let size? where size == file.size:
                resumed = true  // complete from an earlier attempt; verify below
            case let size? where size > 0 && size < file.size:
                resumed = true
                do {
                    try await download(item, to: local, from: size, before: done, total: total, progress: progress)
                } catch AetherError.download(_, .http(416)) where restartAllowed {
                    // The server can't satisfy the range: the partial file doesn't match what it holds. Start over, once.
                    restartAllowed = false
                    try? fm.removeItem(at: local)
                    continue
                }
            default:
                if existing != nil { try fm.removeItem(at: local) }
                resumed = false
                try await download(item, to: local, from: 0, before: done, total: total, progress: progress)
            }

            guard Self.size(of: local) == file.size else {
                try? fm.removeItem(at: local)
                throw AetherError.download(file: item.local, reason: .unexpectedContent("not the catalog's \(file.size) bytes"))
            }
            let actual = try Self.sha256(of: local)
            if actual == file.sha256.lowercased() { return }
            try? fm.removeItem(at: local)
            // A resumed file may have had a corrupt prefix: fetch it once from scratch before giving up.
            if resumed && refetchAllowed {
                refetchAllowed = false
                continue
            }
            throw AetherError.integrity(file: item.local, expected: file.sha256, actual: actual)
        }
    }

    private func download(
        _ item: Item, to local: URL, from offset: Int64, before done: Int64, total: Int64,
        progress: @escaping @Sendable (InstallProgress) -> Void
    ) async throws {
        let request = DownloadRequest(
            url: item.remote, headers: headers(item.remote), destination: local, offset: offset, size: item.file.size,
            file: item.local)
        let throttle = ProgressThrottle(start: offset, granularity: Self.progressGranularity)
        try await transport.download(request) { bytes in
            if throttle.admit(bytes) {
                progress(InstallProgress(completedBytes: done + bytes, totalBytes: total, file: item.local))
            }
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

    static func size(of url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
    }
}

/// Admits a progress report once at least `granularity` bytes have arrived since the last one.
private final class ProgressThrottle: Sendable {
    private let reported: Mutex<Int64>
    private let granularity: Int64

    init(start: Int64, granularity: Int64) {
        reported = Mutex(start)
        self.granularity = granularity
    }

    func admit(_ bytes: Int64) -> Bool {
        reported.withLock { last in
            guard bytes - last >= granularity else { return false }
            last = bytes
            return true
        }
    }
}
