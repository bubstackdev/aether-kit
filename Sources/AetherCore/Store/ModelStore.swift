public import Foundation

/// Downloads, verifies and installs model bundles (DESIGN.md §5).
///
/// - Layout: `<root>/<repo>/<revision>/<subtree>`, default root `Application Support/Aether/Models`.
/// - Atomic installs: files land in `<root>/.staging/…`, then one rename moves the finished bundle into place. A
///   bundle is complete exactly when its receipt (`.aether-install.json`) is present.
/// - Every file's SHA-256 is checked against the catalog before the bundle is accepted.
/// - Interrupted downloads resume from the staged bytes with an HTTP `Range` request.
/// - Concurrent requests for one bundle join a single download, and every caller receives the progress.
/// - Bundles and staging are excluded from backup.
/// - With ``InstallOptions/allowStaleRevision``, a failed download falls back to a complete cached sibling revision.
public actor ModelStore {
    public nonisolated let root: URL
    private let transport: any HTTPTransport
    private let endpoint: URL
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

    public init(root: URL, transport: any HTTPTransport = URLSessionTransport(), endpoint: URL = ModelStore.huggingFace) {
        self.root = root
        self.transport = transport
        self.endpoint = endpoint
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

    public nonisolated func bundleURL(for ref: ModelRef) -> URL {
        var url = root
        for part in [ref.repo, ref.revision, ref.subtree] {
            for component in part.split(separator: "/") { url = url.appending(path: String(component)) }
        }
        return url
    }

    /// The installed bundle for exactly `ref`'s revision, if it is complete.
    public func installed(_ ref: ModelRef) -> InstalledBundle? {
        let url = bundleURL(for: ref)
        guard FileManager.default.fileExists(atPath: url.appending(path: InstallReceipt.fileName).path) else { return nil }
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    /// The newest complete installation of `ref`'s repo and subtree at any other revision.
    public func staleSibling(of ref: ModelRef) -> InstalledBundle? {
        let fm = FileManager.default
        var repoDir = root
        for component in ref.repo.split(separator: "/") { repoDir = repoDir.appending(path: String(component)) }
        guard let revisions = try? fm.contentsOfDirectory(atPath: repoDir.path) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var best: (InstalledBundle, Date)?
        for revision in revisions where revision != ref.revision && !revision.hasPrefix(".") {
            var candidate = ref
            candidate.revision = revision
            let url = bundleURL(for: candidate)
            guard let data = try? Data(contentsOf: url.appending(path: InstallReceipt.fileName)),
                  let receipt = try? decoder.decode(InstallReceipt.self, from: data),
                  receipt.ref.subtree == ref.subtree
            else { continue }
            if best == nil || receipt.installedAt > best!.1 {
                best = (InstalledBundle(ref: receipt.ref, url: url, isStale: true), receipt.installedAt)
            }
        }
        return best?.0
    }

    /// Deletes the installed bundle for `ref` (and any staged bytes for it).
    public func remove(_ ref: ModelRef) throws(AetherError) {
        let fm = FileManager.default
        for url in [bundleURL(for: ref), stagingURL(for: ref)] where fm.fileExists(atPath: url.path) {
            do { try fm.removeItem(at: url) } catch { throw .runtime(underlying: error) }
        }
    }

    // MARK: - Install

    /// Installs `ref` and streams progress, ending with `.finished`. Concurrent calls for the same `ref` share one
    /// download; each caller's stream receives every progress update from the moment it joins. Cancelling a stream
    /// detaches that caller; the download is cancelled when the last caller leaves (staged bytes are kept for resume).
    public func installation(
        _ ref: ModelRef, files: [CatalogFile], options: InstallOptions = []
    ) -> AsyncThrowingStream<InstallEvent, any Error> {
        let (stream, continuation) = AsyncThrowingStream<InstallEvent, any Error>.makeStream()
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
            let installer = Installer(
                transport: transport, endpoint: endpoint, ref: ref, files: files,
                staging: stagingURL(for: ref), destination: bundleURL(for: ref))
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
        _ ref: ModelRef, files: [CatalogFile], options: InstallOptions = [],
        progress: (@Sendable (InstallProgress) -> Void)? = nil
    ) async throws -> InstalledBundle {
        for try await event in installation(ref, files: files, options: options) {
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
        let (ref, variant) = try ModelRef.resolve(id, in: catalog, for: device)
        return try await install(ref, files: variant.files, options: options, progress: progress)
    }

    // MARK: - Job plumbing

    nonisolated func stagingURL(for ref: ModelRef) -> URL {
        let key = [ref.repo, ref.revision, ref.subtree].joined(separator: "@")
            .replacingOccurrences(of: "/", with: "~")
        return root.appending(path: ".staging", directoryHint: .isDirectory).appending(path: key)
    }

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
            let bundle = InstalledBundle(ref: ref, url: url, isStale: false)
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
