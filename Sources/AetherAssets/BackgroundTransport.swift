import Foundation

/// HTTPS through a background `URLSession`, so the system carries transfers on while an iOS app is suspended or
/// terminated (and in macOS apps). One download task per file.
///
/// - **Redirects.** Background sessions follow redirects themselves and never consult a delegate, so the transport
///   resolves them first, in the foreground: it follows each redirect on the original host with the app's headers and
///   stops at the first other host. The background task then fetches that URL, with the app's headers only if it is
///   still on the original host. A token for a model host never reaches the CDN it redirects to.
/// - **Resuming.** A partial file resumes with a `Range` request: a `206` that continues the file is appended, a `200`
///   or a `206` from byte 0 replaces it, a `416` for a partial file restarts it once. A stopped or failed task keeps the
///   system's resume data beside the file, and the next task for the same bytes continues from it; a task carrying the
///   app's headers keeps none, since resume data holds the request. A `403` or `410` (an expired signed link) retries
///   once with the redirect resolved again.
/// - **Relaunches.** Every task records where its bytes belong, so a task that finishes while no `fetch` is waiting
///   (the app was relaunched to receive it) still lands in the download area, and the next preparation run finds the
///   file complete. A later `fetch` for a file whose task is still running joins that task instead of starting another.
///
/// The app connects the transport to its lifecycle in one place (docs/model-preparation.md):
/// ```swift
/// // UIApplicationDelegate.application(_:handleEventsForBackgroundURLSession:completionHandler:)
/// BackgroundTransport.handleEvents(identifier: identifier, completionHandler: completionHandler)
/// // or SwiftUI: .backgroundTask(.urlSession(identifier)) { await BackgroundTransport.handleEvents(identifier: identifier) }
/// ```
public final class BackgroundTransport: ModelTransport, Sendable {
    public let identifier: String
    private let coordinator: BackgroundCoordinator
    private let resolver: URLSessionConfiguration

    /// `configure` adjusts the background configuration before the session is created (for example
    /// `isDiscretionary` or `allowsCellularAccess`). Every transport with the same identifier shares one session: pass
    /// the same `configure` here and to ``handleEvents(identifier:configure:completionHandler:)``, since whichever runs
    /// first in a process creates the session.
    public convenience init(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void = { _ in }) {
        self.init(identifier: identifier, configure: configure, resolver: .ephemeral)
    }

    init(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void, resolver: URLSessionConfiguration) {
        self.identifier = identifier
        self.resolver = resolver
        coordinator = BackgroundCoordinator.shared(identifier: identifier, configure: configure)
    }

    /// Hands the system's background-session events to Kit. Call it from the app's background-session hook; Kit calls
    /// `completionHandler` on the main queue once every event for the session has been delivered.
    public static func handleEvents(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void = { _ in },
                                    completionHandler: @escaping @Sendable () -> Void) {
        BackgroundCoordinator.shared(identifier: identifier, configure: configure, eventsCompletion: completionHandler)
    }

    /// The async form, for SwiftUI's `.backgroundTask(.urlSession(identifier))`: returns once every event for the
    /// session has been delivered.
    public static func handleEvents(identifier: String,
                                    configure: @escaping @Sendable (URLSessionConfiguration) -> Void = { _ in }) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            handleEvents(identifier: identifier, configure: configure) { continuation.resume() }
        }
    }

    public func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
                      progress: @escaping @Sendable (UInt64) -> Void) async throws {
        guard request.url.scheme?.lowercased() == "https" else {
            throw PreparationError.invalidEntry("not an HTTPS URL: \(request.url.absoluteString)")
        }
        let offset = resume ? URLSessionTransport.size(of: destination) : 0
        do {
            try await downloadRenewingExpiredLinks(request, to: destination, from: offset, progress: progress)
        } catch let PreparationError.http(status, _) where status == 416 && offset > 0 {
            // The server can't satisfy the range: the partial file doesn't match what it holds. Start over.
            try? FileManager.default.removeItem(at: destination)
            try await downloadRenewingExpiredLinks(request, to: destination, from: 0, progress: progress)
        }
    }

    /// A task keeps the URL it was created with, and a CDN's signed link can expire before a long-suspended or
    /// discretionary task (or one resumed from old resume data) reaches it. A 403 or 410 therefore retries once with
    /// the redirect resolved again and without the resume data.
    private func downloadRenewingExpiredLinks(_ request: TransportRequest, to destination: URL, from offset: UInt64,
                                              progress: @escaping @Sendable (UInt64) -> Void) async throws {
        do {
            try await download(request, to: destination, from: offset, progress: progress)
        } catch let PreparationError.http(status, _) where status == 403 || status == 410 {
            try? FileManager.default.removeItem(at: record(request, destination, offset).resumeDataURL)
            try await download(request, to: destination, from: offset, progress: progress)
        }
    }

    private func record(_ request: TransportRequest, _ destination: URL, _ offset: UInt64) -> BackgroundTaskRecord {
        BackgroundTaskRecord(destination: BackgroundTaskRecord.portablePath(destination.path), offset: offset,
                             byteCount: request.file.byteCount, path: request.file.path)
    }

    private func download(_ request: TransportRequest, to destination: URL, from offset: UInt64,
                          progress: @escaping @Sendable (UInt64) -> Void) async throws {
        try await coordinator.download(record: record(request, destination, offset), progress: progress) {
            try await RedirectResolver.resolve(request, configuration: resolver)
        }
    }
}

/// Follows redirects in the foreground with the app's headers while they stay on the original host, and returns the
/// URL a background task should fetch and the headers it may carry there. Each hop is probed with the same GET the
/// task will send (a one-byte range, abandoned once the response headers arrive), since a server may answer HEAD
/// differently or not at all.
enum RedirectResolver {
    static let maximumHops = 8

    static func resolve(_ request: TransportRequest, configuration: URLSessionConfiguration) async throws -> (url: URL, headers: [String: String]) {
        guard !request.headers.isEmpty else { return (request.url, [:]) }  // nothing to protect: let the system follow
        let host = request.url.host?.lowercased()
        let probe = HeadersOnly()
        let session = URLSession(configuration: configuration, delegate: probe, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var url = request.url
        for _ in 0..<maximumHops {
            guard url.host?.lowercased() == host else { return (url, [:]) }
            var hop = URLRequest(url: url)
            for (name, value) in request.headers { hop.setValue(value, forHTTPHeaderField: name) }
            hop.setValue("bytes=0-0", forHTTPHeaderField: "Range")
            hop.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            let response = try await probe.response(to: hop, in: session)
            guard let http = response as? HTTPURLResponse else { throw PreparationError.network("not an HTTP response") }
            switch http.statusCode {
            case 200..<300:
                return (url, request.headers)
            case 300..<400:
                guard let location = http.value(forHTTPHeaderField: "Location"),
                      let next = URL(string: location, relativeTo: url)?.absoluteURL,
                      next.scheme?.lowercased() == "https" else {
                    throw PreparationError.network("redirect away from HTTPS for \(request.file.path)")
                }
                url = next
            default:
                throw PreparationError.http(status: http.statusCode, path: request.file.path)
            }
        }
        throw PreparationError.network("too many redirects for \(request.file.path)")
    }

    /// Refuses redirects (the 3xx becomes the response) and cancels each task once its response headers arrive.
    private final class HeadersOnly: NSObject, URLSessionDataDelegate, @unchecked Sendable {
        private let lock = NSLock()
        private var pending: [Int: CheckedContinuation<URLResponse, any Error>] = [:]

        func response(to request: URLRequest, in session: URLSession) async throws -> URLResponse {
            let task = session.dataTask(with: request)
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    lock.withLock { pending[task.taskIdentifier] = continuation }
                    task.resume()
                }
            } onCancel: { task.cancel() }
        }

        private func finish(_ task: URLSessionTask, _ result: Result<URLResponse, any Error>) {
            lock.withLock { pending.removeValue(forKey: task.taskIdentifier) }?.resume(with: result)
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
            completionHandler(nil)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
            finish(dataTask, .success(response))
            completionHandler(.cancel)
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
            if let error = error as? URLError, error.code == .cancelled {
                finish(task, .failure(CancellationError()))
            } else {
                finish(task, .failure(error.map { PreparationError.network(($0 as NSError).localizedDescription) }
                                      ?? PreparationError.network("no response")))
            }
        }
    }
}

/// What a background task's bytes are for, kept in the task's description so it survives an app relaunch.
struct BackgroundTaskRecord: Codable, Equatable {
    /// The download path, relative to the home directory when it lies inside it: an iOS app's container path changes
    /// across updates, its home-relative paths don't.
    let destination: String
    let offset: UInt64
    let byteCount: UInt64
    let path: String

    var destinationURL: URL {
        destination.hasPrefix("/") ? URL(fileURLWithPath: destination)
            : URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(destination)
    }

    /// The system's resume data for this task's bytes, kept beside the file until a task continues from it.
    var resumeDataURL: URL {
        destinationURL.deletingLastPathComponent().appendingPathComponent(".\(destinationURL.lastPathComponent).resume-\(offset)")
    }

    static func portablePath(_ absolute: String) -> String {
        let home = URL(fileURLWithPath: NSHomeDirectory()).standardizedFileURL.resolvingSymlinksInPath().path + "/"
        let path = URL(fileURLWithPath: absolute).standardizedFileURL.resolvingSymlinksInPath().path
        return path.hasPrefix(home) ? String(path.dropFirst(home.count)) : path
    }

    var encoded: String { String(decoding: (try? JSONEncoder().encode(self)) ?? Data(), as: UTF8.self) }
    init(destination: String, offset: UInt64, byteCount: UInt64, path: String) {
        self.destination = destination; self.offset = offset; self.byteCount = byteCount; self.path = path
    }
    init?(encoded: String?) {
        guard let encoded, let value = try? JSONDecoder().decode(Self.self, from: Data(encoded.utf8)) else { return nil }
        self = value
    }

    /// Moves a finished download into place: appended when it continues the file, replacing it when it starts at byte 0
    /// (a `200`, or a `206` from the start). A task resumed from the system's resume data downloads into one file that
    /// also holds the bytes fetched before the stop, so where the file starts comes from the `Content-Range` end and
    /// the received size, not the range's start. A download larger than the pinned size is refused. Runs inside the
    /// delegate callback, before the system deletes `location`.
    func apply(downloadedFile location: URL, response: URLResponse?) throws {
        guard let http = response as? HTTPURLResponse else { throw PreparationError.network("not an HTTP response") }
        let target = destinationURL
        let received = URLSessionTransport.size(of: location)
        switch http.statusCode {
        case 206:
            guard let header = http.value(forHTTPHeaderField: "Content-Range"),
                  let range = URLSessionTransport.contentRange(header), range.total.map({ $0 == byteCount }) ?? true else {
                throw PreparationError.network("unexpected Content-Range for \(path)")
            }
            guard range.end < byteCount, received <= range.end + 1 else { throw PreparationError.integrityMismatch(path: path) }
            let start = range.end + 1 - received
            if start == 0 { try replace(target, with: location); return }
            guard start == offset, URLSessionTransport.size(of: target) == offset else {
                throw PreparationError.network("unexpected Content-Range for \(path)")
            }
            let output = try FileHandle(forWritingTo: target)
            defer { try? output.close() }
            try output.seekToEnd()
            let input = try FileHandle(forReadingFrom: location)
            defer { try? input.close() }
            while let chunk = try input.read(upToCount: 4 << 20), !chunk.isEmpty { try output.write(contentsOf: chunk) }
            try output.synchronize()
        case 200:
            guard received <= byteCount else { throw PreparationError.integrityMismatch(path: path) }
            try replace(target, with: location)
        default:
            throw PreparationError.http(status: http.statusCode, path: path)
        }
    }

    private func replace(_ target: URL, with location: URL) throws {
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
        try FileManager.default.moveItem(at: location, to: target)
    }

    /// Where a task's download starts in the file, once its response has arrived. A task's expected byte count covers
    /// its whole download, including what a resumed task fetched before it stopped, so when known it gives the start
    /// even if the first response sent the whole file and the resumed one honoured the range. Otherwise: byte 0 after
    /// a `200` or a `206` from the start, else the record's offset.
    func base(of response: URLResponse?, expectedBytes: Int64) -> UInt64 {
        if expectedBytes > 0, UInt64(expectedBytes) <= byteCount { return byteCount - UInt64(expectedBytes) }
        guard let http = response as? HTTPURLResponse else { return min(offset, byteCount) }
        if http.statusCode == 200 { return 0 }
        let range = http.value(forHTTPHeaderField: "Content-Range").flatMap(URLSessionTransport.contentRange)
        return range?.start == 0 ? 0 : min(offset, byteCount)
    }
}

/// One per session identifier: owns the background session and routes its tasks to waiting fetches.
final class BackgroundCoordinator: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private struct Waiter {
        let continuation: CheckedContinuation<Void, any Error>
        let progress: @Sendable (UInt64) -> Void
    }

    private static let registryLock = NSLock()
    nonisolated(unsafe) private static var registry: [String: BackgroundCoordinator] = [:]

    /// The coordinator for `identifier`, creating it (and its session) on first use. An events completion is stored
    /// before the session exists, so events the system delivers as soon as the session is created can't miss it.
    @discardableResult
    static func shared(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void,
                       eventsCompletion: (@Sendable () -> Void)? = nil) -> BackgroundCoordinator {
        registryLock.withLock {
            if let existing = registry[identifier] {
                if let eventsCompletion { existing.storeEventsCompletion(eventsCompletion) }
                return existing
            }
            let coordinator = BackgroundCoordinator()
            if let eventsCompletion { coordinator.storeEventsCompletion(eventsCompletion) }
            let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
            configuration.sessionSendsLaunchEvents = true
            configure(configuration)
            coordinator.session = URLSession(configuration: configuration, delegate: coordinator, delegateQueue: nil)
            registry[identifier] = coordinator
            return coordinator
        }
    }

    private let lock = NSLock()
    private var session: URLSession!
    private var waiters: [Int: [Waiter]] = [:]            // by task identifier
    private var applied: [Int: (any Error)?] = [:]        // a finished download's result, until didComplete
    private var failures: [Int: any Error] = [:]          // a task cancelled for a reason of ours
    private var outcomes: [Int: (any Error)?] = [:]       // completed tasks, for a fetch that joins late
    private var eventsCompletions: [@Sendable () -> Void] = []
    private var eventsFinishedUnclaimed = false

    private override init() { super.init() }

    func storeEventsCompletion(_ completion: @escaping @Sendable () -> Void) {
        let callNow: Bool = lock.withLock {
            if eventsFinishedUnclaimed { eventsFinishedUnclaimed = false; return true }
            eventsCompletions.append(completion)
            return false
        }
        if callNow { DispatchQueue.main.async { completion() } }
    }

    func download(record: BackgroundTaskRecord, progress: @escaping @Sendable (UInt64) -> Void,
                  resolve: () async throws -> (url: URL, headers: [String: String])) async throws {
        let task: URLSessionTask
        if let existing = await existingTask(for: record) { task = existing } else { task = try newTask(record, resolved: await resolve()) }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let outcome: (any Error)?? = lock.withLock {
                    if let outcome = outcomes[task.taskIdentifier] { return .some(outcome) }
                    waiters[task.taskIdentifier, default: []].append(Waiter(continuation: continuation, progress: progress))
                    return .none
                }
                switch outcome {
                case .some(.some(let error)): continuation.resume(throwing: error)
                case .some(.none): continuation.resume()
                case .none: if task.state == .suspended { task.resume() }
                }
            }
        } onCancel: {
            // Keep what the system downloaded, so the next task for these bytes continues from it. Resume data holds
            // the original request, so a task carrying the app's headers (a token) keeps none.
            if let download = task as? URLSessionDownloadTask, !Self.carriesAppHeaders(task) {
                download.cancel(byProducingResumeData: { data in if let data { try? data.write(to: record.resumeDataURL) } })
            } else {
                task.cancel()
            }
        }
    }

    /// Whether a task's request carries headers beyond the transport's own (`Range`, `Accept-Encoding`).
    static func carriesAppHeaders(_ task: URLSessionTask) -> Bool {
        (task.originalRequest?.allHTTPHeaderFields ?? [:]).keys.contains { !["range", "accept-encoding"].contains($0.lowercased()) }
    }

    /// A running task from this or an earlier launch that writes the same file from the same offset.
    private func existingTask(for record: BackgroundTaskRecord) async -> URLSessionTask? {
        let tasks = await session.allTasks
        return tasks.first { task in
            guard task.state == .running || task.state == .suspended, let other = BackgroundTaskRecord(encoded: task.taskDescription) else {
                return false
            }
            return other.destinationURL.standardizedFileURL.path == record.destinationURL.standardizedFileURL.path
                && other.offset == record.offset
        }
    }

    private func newTask(_ record: BackgroundTaskRecord, resolved: (url: URL, headers: [String: String])) throws -> URLSessionTask {
        let task: URLSessionDownloadTask
        if let data = try? Data(contentsOf: record.resumeDataURL) {
            try? FileManager.default.removeItem(at: record.resumeDataURL)
            task = session.downloadTask(withResumeData: data)
        } else {
            var urlRequest = URLRequest(url: resolved.url)
            for (name, value) in resolved.headers { urlRequest.setValue(value, forHTTPHeaderField: name) }
            if record.offset > 0 { urlRequest.setValue("bytes=\(record.offset)-", forHTTPHeaderField: "Range") }
            urlRequest.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            task = session.downloadTask(with: urlRequest)
        }
        task.taskDescription = record.encoded
        return task
    }

    // MARK: URLSessionDownloadDelegate

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) else { return }
        let written = UInt64(max(0, totalBytesWritten))
        let base = record.base(of: downloadTask.response, expectedBytes: totalBytesExpectedToWrite)
        guard written <= record.byteCount - base else {
            // More than the pinned remainder: wrong, and it could fill the disk.
            lock.withLock { failures[downloadTask.taskIdentifier] = PreparationError.integrityMismatch(path: record.path) }
            downloadTask.cancel()
            return
        }
        for waiter in lock.withLock({ waiters[downloadTask.taskIdentifier] ?? [] }) { waiter.progress(base + written) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let outcome: (any Error)?
        if let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) {
            do { try record.apply(downloadedFile: location, response: downloadTask.response); outcome = nil }
            catch { outcome = error }
        } else {
            outcome = PreparationError.network("background task without a record")
        }
        lock.withLock { applied[downloadTask.taskIdentifier] = .some(outcome) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let record = BackgroundTaskRecord(encoded: task.taskDescription)
        // A failed transfer keeps the system's resume data for the next task, unless it holds the app's headers.
        if let data = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data, let record,
           !Self.carriesAppHeaders(task) {
            try? data.write(to: record.resumeDataURL)
        }
        let (pending, outcome): ([Waiter], (any Error)?) = lock.withLock {
            let pending = waiters.removeValue(forKey: task.taskIdentifier) ?? []
            let finished = applied.removeValue(forKey: task.taskIdentifier)
            let ours = failures.removeValue(forKey: task.taskIdentifier)
            let outcome: (any Error)?
            if let ours {
                outcome = ours
            } else if let error = error as? URLError, error.code == .cancelled {
                outcome = CancellationError()
            } else if let error {
                outcome = PreparationError.network((error as NSError).localizedDescription)
            } else if let finished {
                outcome = finished
            } else {
                // No body was delivered (an error status): report it.
                let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
                outcome = PreparationError.http(status: status, path: record?.path ?? "")
            }
            outcomes[task.taskIdentifier] = .some(outcome)
            if outcomes.count > 256, let oldest = outcomes.keys.min() { outcomes[oldest] = nil }
            return (pending, outcome)
        }
        for waiter in pending {
            if let outcome { waiter.continuation.resume(throwing: outcome) } else { waiter.continuation.resume() }
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        let completions: [@Sendable () -> Void] = lock.withLock {
            defer { eventsCompletions = [] }
            if eventsCompletions.isEmpty { eventsFinishedUnclaimed = true }
            return eventsCompletions
        }
        guard !completions.isEmpty else { return }
        DispatchQueue.main.async { completions.forEach { $0() } }
    }
}
