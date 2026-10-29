import Foundation

/// Fetches one file of a catalog entry. A transport writes `request.url`'s bytes to `destination`, appending from the
/// file's current size when `resume` is true (and restarting it when the server can't resume), and reports the file's
/// total bytes on disk as they arrive. It never decides what is trusted: preparation checks every file's size and
/// `LocalAssetStore` hashes every byte it installs (docs/model-preparation.md).
public protocol ModelTransport: Sendable {
    func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
               progress: @escaping @Sendable (UInt64) -> Void) async throws
}

public struct TransportRequest: Sendable, Equatable {
    public let url: URL
    /// The file as the catalog entry pins it.
    public let file: AssetFile
    /// Extra request headers, such as an authorization header the app supplies. Kit never stores them.
    public let headers: [String: String]
    public init(url: URL, file: AssetFile, headers: [String: String] = [:]) {
        self.url = url; self.file = file; self.headers = headers
    }
}

public enum PreparationError: Error, Sendable, Equatable {
    case insufficientSpace(needed: UInt64, available: UInt64)
    case integrityMismatch(path: String)
    case http(status: Int, path: String)
    case network(String)
    case invalidEntry(String)
    /// Another run is preparing the same installation.
    case busy
    case fileSystem(Int32)

    /// Whether running the same preparation again can succeed without anything else changing.
    public var isRetryable: Bool {
        switch self {
        case .network, .busy, .integrityMismatch: true
        case let .http(status, _): status == 408 || status == 429 || status >= 500
        case .insufficientSpace, .invalidEntry, .fileSystem: false
        }
    }
}

public enum PreparationEvent: Sendable, Equatable {
    case checking
    /// Overall and per-file download progress. Coalesced; each file's final count is always delivered.
    case downloading(path: String, fileBytes: UInt64, fileTotal: UInt64, bytesDone: UInt64, bytesTotal: UInt64)
    /// Copying into the store, hashing every byte against the entry.
    case installing(bytesDone: UInt64, bytesTotal: UInt64)
    case installed(installationID: String)
    case failed(PreparationError)
    case cancelled

    public var isTerminal: Bool {
        switch self {
        case .installed, .failed, .cancelled: true
        case .checking, .downloading, .installing: false
        }
    }
}

public enum PreparationStatus: Sendable, Equatable {
    case installed(installationID: String)
    /// Downloaded data is on disk and a run will resume from it.
    case partial(bytesPresent: UInt64, bytesTotal: UInt64)
    case notInstalled(bytesTotal: UInt64)
}

/// Prepares catalog entries into a `LocalAssetStore`: check, download into a per-installation area kept across
/// launches, install (the store's copy and hash pass is where bytes become trusted), then clean up.
public struct ModelPreparation: Sendable {
    public let store: LocalAssetStore
    let transport: any ModelTransport
    let headers: @Sendable (URL) -> [String: String]
    let availableCapacity: @Sendable (URL) throws -> UInt64
    let progressInterval: Duration

    /// `headers` supplies per-request headers (for example a Hugging Face bearer token for a gated repository).
    public init(store: LocalAssetStore, transport: any ModelTransport,
                headers: @escaping @Sendable (URL) -> [String: String] = { _ in [:] }) {
        self.init(store: store, transport: transport, headers: headers,
                  availableCapacity: Self.volumeAvailableCapacity, progressInterval: .milliseconds(100))
    }

    init(store: LocalAssetStore, transport: any ModelTransport, headers: @escaping @Sendable (URL) -> [String: String],
         availableCapacity: @escaping @Sendable (URL) throws -> UInt64, progressInterval: Duration) {
        self.store = store; self.transport = transport; self.headers = headers
        self.availableCapacity = availableCapacity; self.progressInterval = progressInterval
    }

    /// Request headers for a gated or private Hugging Face repository: the app's token as a bearer authorization, sent
    /// only to huggingface.co. The transport forwards headers only to the original host, so a redirect to the CDN that
    /// serves the bytes never sees the token. Kit doesn't store it.
    public static func huggingFaceAuthorization(token: String) -> @Sendable (URL) -> [String: String] {
        { url in url.host?.lowercased() == "huggingface.co" ? ["Authorization": "Bearer " + token] : [:] }
    }

    /// Where the entry stands, without network access or hashing.
    public func status(_ entry: ModelCatalogEntry) throws -> PreparationStatus {
        let id = try validated(entry)
        if store.isInstalled(id) { return .installed(installationID: id) }
        let present = try presentBytes(entry, id: id)
        return present > 0 ? .partial(bytesPresent: present, bytesTotal: entry.byteCount) : .notInstalled(bytesTotal: entry.byteCount)
    }

    /// Prepares the entry, ending in exactly one terminal event. Stopping the iteration cancels the run; downloaded
    /// bytes stay on disk for the next run.
    public func prepare(_ entry: ModelCatalogEntry) -> AsyncStream<PreparationEvent> {
        let id = try? entry.installationID
        let token = UUID()
        return AsyncStream { continuation in
            // The worker registers itself and removes its own record when it ends; nothing else writes the registry,
            // so a stream dropped after its terminal event can't leave a stale record behind.
            if let id { Self.runs.register(id, token) }
            let worker = Task {
                let event: PreparationEvent
                do {
                    event = .installed(installationID: try await run(entry, token: token) { continuation.yield($0) })
                } catch is CancellationError {
                    event = .cancelled
                } catch let error as PreparationError {
                    event = Task.isCancelled ? .cancelled : .failed(error)
                } catch {
                    event = Task.isCancelled ? .cancelled : .failed(Self.mapped(error))
                }
                // `run` has released the download lock by now, so a run waiting on this one can take it.
                if let id { Self.runs.remove(id, token) }
                continuation.yield(event)
                continuation.finish()
            }
            if let id { Self.runs.attach(id, token, worker) }
            continuation.onTermination = { _ in worker.cancel() }
        }
    }

    /// Every run this process has started and not yet finished, by installation.
    static let runs = RunRegistry()

    /// `prepare` for callers that only need the result: the installation ID, or the run's error.
    public func install(_ entry: ModelCatalogEntry) async throws -> String {
        for await event in prepare(entry) {
            switch event {
            case let .installed(id): return id
            case let .failed(error): throw error
            case .cancelled: throw CancellationError()
            case .checking, .downloading, .installing: continue
            }
        }
        throw CancellationError()
    }

    /// Removes the entry's downloaded bytes. Fails with busy while a run is preparing it.
    func discardDownloads(_ entry: ModelCatalogEntry) throws {
        let id = try validated(entry)
        let held: StoreLock
        do { held = try store.downloadLock(id) } catch AssetStoreError.busy { throw PreparationError.busy }
        defer { withExtendedLifetime(held) {} }
        try remove(store.downloads.appendingPathComponent(id, isDirectory: true))
    }

    // MARK: - The run

    private func run(_ entry: ModelCatalogEntry, token: UUID,
                     emit: @escaping @Sendable (PreparationEvent) -> Void) async throws -> String {
        let id = try validated(entry)
        emit(.checking)
        if try await verifiedExisting(entry, id: id, emit: emit) { return id }
        let held = try await downloadLock(id, token: token)
        defer { withExtendedLifetime(held) {} }
        // Another run (here or in another process) may have published while this one waited for the lock.
        if try await verifiedExisting(entry, id: id, emit: emit) { return id }
        let area = try downloadFiles(id)
        let present = try presentBytes(entry, id: id)
        // The download needs the missing bytes, and the install's copy needs the whole entry again.
        let needed = (entry.byteCount - present) + entry.byteCount
        let available = try availableCapacity(store.root)
        guard available >= needed else { throw PreparationError.insufficientSpace(needed: needed, available: available) }

        let throttle = Throttle(interval: progressInterval)
        var done = present
        for file in entry.manifest.files {
            try Task.checkCancellation()
            let target = try confinedDownload(file.path, in: area)
            var size = fileSize(target)
            if size > file.byteCount { try remove(target); size = 0 }
            if file.byteCount == 0, !FileManager.default.fileExists(atPath: target.path) {
                // Nothing to fetch, but the install needs the file; it still checks the empty file's hash.
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                guard FileManager.default.createFile(atPath: target.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
                    throw PreparationError.fileSystem(errno)
                }
            }
            if size < file.byteCount {
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                // Saturating: the area can change under a background task that lands after `present` was summed.
                let before = done - min(done, size)
                let url = try entry.url(for: file)
                let request = TransportRequest(url: url, file: file, headers: headers(url))
                let report: @Sendable (UInt64) -> Void = { bytes in
                    let clamped = min(bytes, file.byteCount)
                    if throttle.admit() {
                        emit(.downloading(path: file.path, fileBytes: clamped, fileTotal: file.byteCount,
                                          bytesDone: before + clamped, bytesTotal: entry.byteCount))
                    }
                }
                try await transport.fetch(request, to: target, resume: size > 0, progress: report)
                try Task.checkCancellation()
                size = fileSize(target)
                guard size == file.byteCount else {
                    try remove(target)
                    throw PreparationError.integrityMismatch(path: file.path)
                }
                done = before + size
                emit(.downloading(path: file.path, fileBytes: size, fileTotal: file.byteCount,
                                  bytesDone: done, bytesTotal: entry.byteCount))
            }
        }
        emit(.installing(bytesDone: 0, bytesTotal: entry.byteCount))
        let installThrottle = Throttle(interval: progressInterval)
        do {
            _ = try await store.install(entry.manifest, from: area) { bytes in
                if installThrottle.admit() { emit(.installing(bytesDone: bytes, bytesTotal: entry.byteCount)) }
            }
        } catch AssetStoreError.integrityMismatch(let path) where entry.manifest.files.contains(where: { $0.path == path }) {
            // The downloaded bytes don't hash to the entry: drop that file so the next run fetches it again.
            try? remove(try confinedDownload(path, in: area))
            throw PreparationError.integrityMismatch(path: path)
        }
        try? FileManager.default.removeItem(at: area.deletingLastPathComponent())
        return id
    }

    // MARK: - Helpers

    /// The installation's download lock. Busy means another run is preparing it, unless the holder is a run this
    /// process has cancelled and that is still unwinding: then wait for it (at most `stoppedRunWait`) and try again.
    private func downloadLock(_ id: String, token: UUID) async throws -> StoreLock {
        do { return try store.downloadLock(id) } catch AssetStoreError.busy {}
        let stopped = Self.runs.cancelledRuns(id, excluding: token)
        guard !stopped.isEmpty else {
            // The holder may have released the lock between our attempt and the registry check.
            do { return try store.downloadLock(id) } catch AssetStoreError.busy { throw PreparationError.busy }
        }
        let waited = await withTaskGroup(of: Bool.self) { group in
            group.addTask { for run in stopped { await run.value }; return true }
            group.addTask { try? await Task.sleep(for: Self.stoppedRunWait); return false }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        try Task.checkCancellation()
        guard waited else { throw PreparationError.busy }
        do { return try store.downloadLock(id) } catch AssetStoreError.busy { throw PreparationError.busy }
    }

    static let stoppedRunWait: Duration = .seconds(30)

    /// If the installation exists, hashes it (as a retried install would) and returns true. An installation that fails
    /// verification is removed so this run downloads it again; one that readers hold open reports busy.
    private func verifiedExisting(_ entry: ModelCatalogEntry, id: String,
                                  emit: @Sendable (PreparationEvent) -> Void) async throws -> Bool {
        guard store.isInstalled(id) else { return false }
        emit(.installing(bytesDone: 0, bytesTotal: entry.byteCount))
        do {
            _ = try await store.acquire(id)
            return true
        } catch AssetStoreError.notInstalled {
            return false  // another run removed it first
        } catch AssetStoreError.integrityMismatch, AssetStoreError.invalidManifest {
            do { try store.remove(id) } catch AssetStoreError.busy { throw PreparationError.busy } catch AssetStoreError.notInstalled {}
            return false
        }
    }

    private func validated(_ entry: ModelCatalogEntry) throws -> String {
        do {
            try entry.validate()
            return try entry.installationID
        } catch let error as AssetStoreError {
            throw PreparationError.invalidEntry("\(error)")
        }
    }

    /// The installation's download area (`downloads/<id>/files`), created on first use.
    private func downloadFiles(_ id: String) throws -> URL {
        let files = store.downloads.appendingPathComponent(id, isDirectory: true).appendingPathComponent("files", isDirectory: true)
        try FileManager.default.createDirectory(at: files, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        return files
    }

    private func presentBytes(_ entry: ModelCatalogEntry, id: String) throws -> UInt64 {
        let area = store.downloads.appendingPathComponent(id, isDirectory: true).appendingPathComponent("files", isDirectory: true)
        guard FileManager.default.fileExists(atPath: area.path) else { return 0 }
        return try entry.manifest.files.reduce(0) { sum, file in
            let size = fileSize(try confinedDownload(file.path, in: area))
            return sum + (size <= file.byteCount ? size : 0)
        }
    }

    /// A download path that stays inside the area: no traversal, and no symlink anywhere along it.
    private func confinedDownload(_ path: String, in area: URL) throws -> URL {
        guard AssetManifest.validPath(path) else { throw PreparationError.invalidEntry("unsafe path \(path)") }
        let candidate = area.appendingPathComponent(path).standardizedFileURL
        var probe = candidate
        while probe.path.count > area.path.count {
            if (try? probe.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true {
                throw PreparationError.invalidEntry("download path holds a link: \(path)")
            }
            probe.deleteLastPathComponent()
        }
        return candidate
    }

    private func fileSize(_ url: URL) -> UInt64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.uint64Value ?? 0
    }

    private func remove(_ url: URL) throws {
        do { try FileManager.default.removeItem(at: url) }
        catch let error as CocoaError where error.code == .fileNoSuchFile || error.code == .fileReadNoSuchFile {}
    }

    static func volumeAvailableCapacity(_ url: URL) throws -> UInt64 {
        let values = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return UInt64(max(0, values.volumeAvailableCapacityForImportantUsage ?? 0))
    }

    static func mapped(_ error: any Error) -> PreparationError {
        switch error {
        case let error as PreparationError: return error
        case let AssetStoreError.integrityMismatch(path): return .integrityMismatch(path: path)
        case AssetStoreError.busy: return .busy
        case let AssetStoreError.fileSystem(code): return .fileSystem(code)
        case let error as AssetStoreError: return .invalidEntry("\(error)")
        case let error as URLError: return .network(error.localizedDescription)
        case let error as CocoaError: return .fileSystem(Int32((error.userInfo[NSUnderlyingErrorKey] as? NSError)?.code ?? 0))
        default: return .network("\(error)")
        }
    }
}

/// Admits at most one progress event per interval.
private final class Throttle: @unchecked Sendable {
    private let interval: Duration
    private let lock = NSLock()
    private var last: ContinuousClock.Instant?
    init(interval: Duration) { self.interval = interval }
    func admit() -> Bool {
        lock.lock(); defer { lock.unlock() }
        let now = ContinuousClock.now
        if let last, now - last < interval { return false }
        last = now
        return true
    }
}

/// Runs by installation: each run's record and, once started, its worker task.
final class RunRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var runs: [String: [UUID: Task<Void, Never>?]] = [:]

    func register(_ id: String, _ token: UUID) { lock.withLock { runs[id, default: [:]][token] = .some(nil) } }

    /// Records the worker, unless it already finished and removed its record.
    func attach(_ id: String, _ token: UUID, _ worker: Task<Void, Never>) {
        lock.withLock { if runs[id]?[token] != nil { runs[id]?[token] = worker } }
    }

    func remove(_ id: String, _ token: UUID) {
        lock.withLock {
            runs[id]?[token] = nil
            if runs[id]?.isEmpty == true { runs[id] = nil }
        }
    }

    /// Workers of other runs for `id` that have been cancelled and haven't finished.
    func cancelledRuns(_ id: String, excluding token: UUID) -> [Task<Void, Never>] {
        lock.withLock {
            (runs[id] ?? [:]).compactMap { key, worker in key != token ? worker.flatMap { $0.isCancelled ? $0 : nil } : nil }
        }
    }

    func isEmpty(_ id: String) -> Bool { lock.withLock { runs[id] == nil } }
}
