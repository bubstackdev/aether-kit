public import Foundation

/// Downloads, verifies and installs model bundles (DESIGN.md §5).
///
/// - Layout: one variant is one install at `<root>/<repo>/<revision>/`, holding every asset of the variant (see
///   `InstalledBundle`), with Aether's files in `.aether/`. Default root `Application Support/Aether/Models`.
/// - Atomic installs: files land in `<root>/.staging/…`, then one rename moves the finished install into place. An
///   install is complete exactly when its receipt (`.aether/receipt.json`) is present.
/// - Every file's SHA-256 is checked against the catalog before the bundle is accepted.
/// - Interrupted downloads resume from the staged bytes with an HTTP `Range` request.
/// - Concurrent requests for one bundle join a single download, and every caller receives the progress.
/// - Bundles and staging are excluded from backup.
/// - With ``InstallOptions/allowStaleRevision``, a failed download falls back to a complete cached sibling revision.
///
/// Across processes (and across `ModelStore` values on one root), each install location `<repo>/<revision>` has an
/// advisory lock (`.locks/`): an install holds it exclusively, so a second process waits and then finds the bundle
/// installed; ``lease(_:)`` holds it shared, and ``remove(_:)`` refuses while any lease is held. Before the first
/// install or removal, the store recovers what an interrupted earlier run left behind
/// (``recoverAbandonedStages()``). Before downloading, it checks the volume has room for the bytes still missing.
///
/// Request headers come from `headers`, per URL: ``huggingFaceAuthorization(token:)`` sends a token to huggingface.co
/// only, and the transport drops the app's headers on a redirect to another host. The store never keeps a token.
public actor ModelStore {
    public nonisolated let root: URL
    private let transport: any HTTPTransport
    private let endpoint: URL
    private let headers: @Sendable (URL) -> [String: String]
    private let diskSpace: @Sendable (URL) throws -> Int64
    private nonisolated let layout: StoreLayout
    private var recovered = false
    private var jobs: [ModelRef: Job] = [:]
    /// Cancelled downloads still unwinding; a new download of the same ref waits for them (they share staging).
    private var unwinding: [ModelRef: Task<Void, Never>] = [:]

    private typealias Subscriber = AsyncThrowingStream<InstallEvent, any Error>.Continuation

    private struct Job {
        /// Distinguishes this download from a later one for the same ref (after a cancel).
        let generation = UUID()
        var subscribers: [UUID: (continuation: Subscriber, options: InstallOptions)] = [:]
        var latest: InstallProgress?
        var task: Task<Void, Never>?
    }

    public static let huggingFace = URL(string: "https://huggingface.co")!

    /// `headers` supplies per-request headers (for example ``huggingFaceAuthorization(token:)`` for a gated repo).
    /// `diskSpace` reports the free bytes on the volume holding a URL; tests inject it.
    public init(
        root: URL, transport: any HTTPTransport = URLSessionTransport(), endpoint: URL = ModelStore.huggingFace,
        headers: @escaping @Sendable (URL) -> [String: String] = { _ in [:] },
        diskSpace: @escaping @Sendable (URL) throws -> Int64 = { try Device.freeDiskSpace(at: $0) }
    ) {
        self.root = root
        self.transport = transport
        self.endpoint = endpoint
        self.headers = headers
        self.diskSpace = diskSpace
        layout = StoreLayout(root: root)
    }

    /// Request headers for a gated or private Hugging Face repository: the token as a bearer authorization, sent only
    /// to `huggingface.co`. The transports drop it on a redirect to any other host (the CDN that serves the bytes).
    public static func huggingFaceAuthorization(token: String) -> @Sendable (URL) -> [String: String] {
        { url in url.host?.lowercased() == "huggingface.co" ? ["Authorization": "Bearer " + token] : [:] }
    }

    /// `Application Support/Aether/Models` in the app's container.
    public static func defaultRoot() throws(AetherError) -> URL {
        do {
            return try FileManager.default.url(
                for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true
            ).appending(path: "Aether/Models", directoryHint: .isDirectory)
        } catch {
            throw .runtime(underlying: error)
        }
    }

    // MARK: - Queries

    /// The revision root of `ref`'s install: `<root>/<repo>/<revision>`.
    public nonisolated func bundleURL(for ref: ModelRef) -> URL { layout.bundle(for: ref) }

    /// The installed bundle for exactly `ref` (the whole installation identity: catalog id, driver, variant and every
    /// asset's repo, revision, subtree and files), if it is complete.
    public func installed(_ ref: ModelRef) -> InstalledBundle? {
        guard (try? ref.validateForStore()) != nil else { return nil }
        let url = bundleURL(for: ref)
        guard let receipt = InstallReceipt.read(at: url), receipt.ref == ref else { return nil }
        return InstalledBundle(ref: receipt.ref, url: url, isStale: false, storeRoot: root)
    }

    /// The newest complete install of the same model and variant at any other revision.
    public func staleSibling(of ref: ModelRef) -> InstalledBundle? {
        guard (try? ref.validateForStore()) != nil else { return nil }
        let repoDir = InstalledLayout.appendingComponents(root, ref.repo)
        guard let revisions = try? FileManager.default.contentsOfDirectory(atPath: repoDir.path) else { return nil }
        var best: (InstalledBundle, Date)?
        for revision in revisions where revision != ref.revision && !revision.hasPrefix(".") {
            let url = repoDir.appending(path: revision)
            guard let receipt = InstallReceipt.read(at: url), receipt.ref.catalogID == ref.catalogID,
                  receipt.ref.variant == ref.variant
            else { continue }
            if best == nil || receipt.installedAt > best!.1 {
                best = (InstalledBundle(ref: receipt.ref, url: url, isStale: true, storeRoot: root), receipt.installedAt)
            }
        }
        return best?.0
    }

    /// A reader lease on an installed bundle: while it is held, no process removes the bundle's files. Throws
    /// ``AetherError/inUse(resource:)`` while an install or removal of that location is running, and
    /// ``AetherError/unknownModel(id:)`` when the bundle is no longer installed.
    ///
    /// Pipelines take one by themselves for a bundle this store returned (``InstalledBundle/readLease()``).
    public nonisolated func lease(_ bundle: InstalledBundle) throws(AetherError) -> BundleReadLease {
        try layout.lease(bundle)
    }

    /// Deletes the install for `ref` (and any staged bytes for it). Refuses with ``AetherError/inUse(resource:)`` while
    /// a reader lease is held or an install of the same location is running, in this process or another.
    public func remove(_ ref: ModelRef) throws(AetherError) {
        try ref.validateForStore()
        recoverIfNeeded()
        guard let lock = try StoreLock.attempt(layout.lock(for: ref), exclusive: true) else {
            throw .inUse(resource: ref.description)
        }
        defer { withExtendedLifetime(lock) {} }
        let fm = FileManager.default
        for url in [bundleURL(for: ref), stagingURL(for: ref)] where fm.fileExists(atPath: url.path) {
            do { try layout.discard(url) } catch { throw .runtime(underlying: error) }
        }
    }

    /// Clears what an interrupted earlier run left behind. Runs by itself before the store's first install or removal;
    /// hosts may also call it during maintenance. Returns how many entries it removed.
    ///
    /// - discarded bundles still in `.trash/` (a removal or replacement interrupted mid-delete);
    /// - staged bytes whose bundle is now installed (they can't be resumed into anything);
    /// - revision directories without a receipt (never a complete install: installs appear by one rename, receipt
    ///   included).
    ///
    /// Each entry is removed only under its install location's exclusive lock, taken without waiting: a location
    /// another process is installing, removing or reading is skipped, and retried by a later call. Staged bytes of an
    /// install that isn't finished are kept for resume. Entries the store didn't create are left alone, and a symlink
    /// is never followed.
    @discardableResult
    public func recoverAbandonedStages() -> Int {
        recovered = true
        return layout.recover()
    }

    private func recoverIfNeeded() {
        if !recovered { recoverAbandonedStages() }
    }

    // MARK: - Install

    /// Installs `ref` and streams progress, ending with `.finished`. Concurrent calls for the same `ref` share one
    /// download; each caller's stream receives every progress update from the moment it joins. Cancelling a stream
    /// detaches that caller; the download is cancelled when the last caller leaves (staged bytes are kept for resume).
    public func installation(
        _ ref: ModelRef, options: InstallOptions = []
    ) -> AsyncThrowingStream<InstallEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<InstallEvent, any Error>.makeStream()
        do { try ref.validateForStore() } catch {
            continuation.finish(throwing: error)
            return stream
        }
        recoverIfNeeded()
        if let bundle = installed(ref) {
            continuation.yield(.finished(bundle))
            continuation.finish()
            return stream
        }
        let id = UUID()
        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            Task { await self.detach(id, from: ref) }
        }
        var job = jobs[ref] ?? Job()
        job.subscribers[id] = (continuation, options)
        if let latest = job.latest { continuation.yield(.progress(latest)) }
        if job.task == nil {
            let installer: Installer
            do {
                installer = try Installer(
                    transport: transport, endpoint: endpoint, ref: ref, layout: layout, headers: headers,
                    diskSpace: diskSpace)
            } catch {
                continuation.finish(throwing: error)
                return stream
            }
            let generation = job.generation
            let prior = unwinding[ref]
            job.task = Task {
                await prior?.value
                await self.drive(installer, for: ref, generation: generation)
            }
        }
        jobs[ref] = job
        return stream
    }

    /// Installs `ref`, reporting progress to `progress`, and returns the bundle.
    public func install(
        _ ref: ModelRef, options: InstallOptions = [], progress: (@Sendable (InstallProgress) -> Void)? = nil
    ) async throws -> InstalledBundle {
        for try await event in installation(ref, options: options) {
            switch event {
            case .progress(let p): progress?(p)
            case .finished(let bundle): return bundle
            }
        }
        throw CancellationError()
    }

    /// Resolves `id` in `catalog` for `device` and installs it.
    public func install(
        _ id: String, in catalog: Catalog = .builtin, for device: Device, options: InstallOptions = [],
        progress: (@Sendable (InstallProgress) -> Void)? = nil
    ) async throws -> InstalledBundle {
        let ref = try ModelRef.resolve(id, in: catalog, for: device)
        return try await install(ref, options: options, progress: progress)
    }

    // MARK: - Job plumbing

    nonisolated func stagingURL(for ref: ModelRef) -> URL { layout.staging(for: ref) }

    private func drive(_ installer: Installer, for ref: ModelRef, generation: UUID) async {
        let (updates, sink) = AsyncStream<InstallProgress>.makeStream(bufferingPolicy: .bufferingNewest(64))
        async let outcome = Self.execute(installer, sink: sink)
        for await update in updates { publish(update, for: ref, generation: generation) }
        finish(ref, generation: generation, with: await outcome)
    }

    @concurrent
    private static func execute(
        _ installer: Installer, sink: AsyncStream<InstallProgress>.Continuation
    ) async -> Result<URL, any Error> {
        defer { sink.finish() }
        do {
            return .success(try await installer.run { sink.yield($0) })
        } catch {
            return .failure(error)
        }
    }

    private func publish(_ update: InstallProgress, for ref: ModelRef, generation: UUID) {
        guard var job = jobs[ref], job.generation == generation else { return }
        job.latest = update
        jobs[ref] = job
        for subscriber in job.subscribers.values { subscriber.continuation.yield(.progress(update)) }
    }

    private func finish(_ ref: ModelRef, generation: UUID, with outcome: Result<URL, any Error>) {
        guard let job = jobs[ref], job.generation == generation else { return }
        jobs[ref] = nil
        switch outcome {
        case .success(let url):
            let bundle = InstalledBundle(ref: ref, url: url, isStale: false, storeRoot: root)
            for subscriber in job.subscribers.values {
                subscriber.continuation.yield(.finished(bundle))
                subscriber.continuation.finish()
            }
        case .failure(let error):
            let stale = Self.isDownloadFailure(error) ? staleSibling(of: ref) : nil
            for subscriber in job.subscribers.values {
                if let stale, subscriber.options.contains(.allowStaleRevision) {
                    subscriber.continuation.yield(.finished(stale))
                    subscriber.continuation.finish()
                } else {
                    subscriber.continuation.finish(throwing: error)
                }
            }
        }
    }

    private func detach(_ id: UUID, from ref: ModelRef) {
        guard var job = jobs[ref], job.subscribers.removeValue(forKey: id) != nil else { return }
        if job.subscribers.isEmpty {
            job.task?.cancel()
            jobs[ref] = nil
            if let task = job.task {
                unwinding[ref] = task
                Task { await task.value; self.doneUnwinding(ref, task) }
            }
        } else {
            jobs[ref] = job
        }
    }

    private func doneUnwinding(_ ref: ModelRef, _ task: Task<Void, Never>) {
        if unwinding[ref] == task { unwinding[ref] = nil }
    }

    private static func isDownloadFailure(_ error: any Error) -> Bool {
        if case AetherError.download = error { return true }
        return false
    }
}
