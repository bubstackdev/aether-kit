public import Foundation
import Synchronization

/// ``HTTPTransport`` on a background `URLSession`, so the system carries a model's downloads on while an iOS app is
/// suspended or terminated (and in macOS apps). Ported from AetherKit 0.0.7, where it was qualified on an iPhone.
///
/// `download` runs one system download task per file; `get` (single responses, not file downloads) uses a
/// foreground ``URLSessionTransport``, since background sessions run no data tasks.
///
/// - **Redirects.** Background sessions follow redirects themselves and never ask a delegate, so the app's headers
///   would travel with them. When a request carries headers, the transport resolves its redirects first, in the
///   foreground: it follows each hop on the original host with the headers and stops at the first other host. Each hop
///   is probed with the GET the task will send (a one-byte range, abandoned once the headers arrive), not a HEAD. The
///   background task then fetches the final URL, carrying the headers only if it is still on the original host. A
///   `403` or `410` (an expired signed link) retries once with the redirect resolved again and without resume data.
/// - **Resuming.** A partial file resumes with a `Range` request: a `206` that continues the file is appended, a `200`
///   or a `206` from byte 0 replaces it, and a `416` reaches the store, which restarts the file once. A task writes
///   nothing to the file until it finishes, so a stopped or failed task keeps the system's resume data beside the
///   file, and the next task for the same bytes continues from it. A task carrying the app's headers keeps none,
///   since resume data holds the request.
/// - **Relaunches.** Every task records where its bytes belong (home-relative, which survives an iOS app update) in
///   its description, so a task that finishes while no `download` is waiting still lands in the staging directory,
///   and the next install finds the file complete. A `download` for a file whose task is still running, from this
///   launch or an earlier one, joins that task instead of starting another.
///
/// **One transport per identifier.** A process may hold only one session per background identifier, so the app
/// creates one `BackgroundTransport` at launch, keeps it for the process's lifetime, and hands the system's
/// background-session events to that same object:
/// ```swift
/// // UIApplicationDelegate.application(_:handleEventsForBackgroundURLSession:completionHandler:)
/// transport.handleEvents(completionHandler: completionHandler)
/// // or SwiftUI: .backgroundTask(.urlSession(identifier)) { await transport.handleEvents() }
/// ```
public final class BackgroundTransport: HTTPTransport, Sendable {
    public let identifier: String
    private let session: URLSession
    private let coordinator: BackgroundCoordinator
    private let resolver: URLSessionConfiguration
    private let foreground: URLSessionTransport

    /// `configure` adjusts the background configuration before the session is created (for example
    /// `isDiscretionary` or `allowsCellularAccess`).
    public convenience init(identifier: String, configure: (URLSessionConfiguration) -> Void = { _ in }) {
        let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
        configuration.sessionSendsLaunchEvents = true
        configure(configuration)
        self.init(identifier: identifier, configuration: configuration, resolver: .ephemeral)
    }

    /// Any session configuration: tests use an ephemeral one with a stub `URLProtocol`.
    init(identifier: String, configuration: URLSessionConfiguration, resolver: URLSessionConfiguration) {
        self.identifier = identifier
        self.resolver = resolver
        coordinator = BackgroundCoordinator()
        session = URLSession(configuration: configuration, delegate: coordinator, delegateQueue: nil)
        foreground = URLSessionTransport(session: URLSession(configuration: resolver))
    }

    /// Hands the system's background-session events to the transport. Call it from the app's background-session
    /// hook; `completionHandler` runs on the main queue once every event for the session has been delivered.
    public func handleEvents(completionHandler: @escaping @Sendable () -> Void) {
        coordinator.storeEventsCompletion(completionHandler)
    }

    /// The async form, for SwiftUI's `.backgroundTask(.urlSession(identifier))`.
    public func handleEvents() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            handleEvents { continuation.resume() }
        }
    }

    public func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>) {
        try await foreground.get(url, headers: headers)
    }

    public func download(_ request: DownloadRequest, progress: @escaping @Sendable (Int64) -> Void) async throws {
        let record = BackgroundTaskRecord(request)
        do {
            try await coordinator.download(record: record, in: session, progress: progress) {
                try await RedirectResolver.resolve(request, configuration: resolver)
            }
        } catch AetherError.download(_, .http(let status)) where status == 403 || status == 410 {
            // A task keeps the URL it was created with, and a CDN's signed link can expire before a long-suspended
            // task (or one resumed from old resume data) reaches it: resolve again, without the resume data.
            try? FileManager.default.removeItem(at: record.resumeDataURL)
            try await coordinator.download(record: record, in: session, progress: progress) {
                try await RedirectResolver.resolve(request, configuration: resolver)
            }
        }
    }
}

/// Follows redirects in the foreground with the app's headers while they stay on the original host, and returns the
/// URL a background task should fetch and the headers it may carry there.
enum RedirectResolver {
    static let maximumHops = 8

    static func resolve(_ request: DownloadRequest, configuration: URLSessionConfiguration) async throws -> (url: URL, headers: [String: String]) {
        guard !request.headers.isEmpty else { return (request.url, [:]) }  // nothing to protect: let the system follow
        let host = request.url.host?.lowercased()
        let wasSecure = request.url.scheme?.lowercased() == "https"
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
            let response: URLResponse
            do {
                response = try await probe.response(to: hop, in: session)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                throw AetherError.download(file: request.file, reason: .transport(describeTransportError(error)))
            }
            guard let http = response as? HTTPURLResponse else {
                throw AetherError.download(file: request.file, reason: .transport("not an HTTP response"))
            }
            switch http.statusCode {
            case 200..<300:
                return (url, request.headers)
            case 300..<400:
                guard let location = http.value(forHTTPHeaderField: "Location"),
                      let next = URL(string: location, relativeTo: url)?.absoluteURL,
                      !wasSecure || next.scheme?.lowercased() == "https"
                else {
                    throw AetherError.download(file: request.file, reason: .transport("redirect away from HTTPS"))
                }
                url = next
            default:
                throw AetherError.download(file: request.file, reason: .http(status: http.statusCode))
            }
        }
        throw AetherError.download(file: request.file, reason: .transport("too many redirects"))
    }

    /// Refuses redirects (the 3xx becomes the response) and cancels each task once its response headers arrive.
    private final class HeadersOnly: NSObject, URLSessionDataDelegate, Sendable {
        private struct State {
            var pending: [Int: CheckedContinuation<URLResponse, any Error>] = [:]
            /// Results that arrived before their continuation, as for a task cancelled before it started.
            var early: [Int: Result<URLResponse, any Error>] = [:]
        }
        private let state = Mutex(State())

        func response(to request: URLRequest, in session: URLSession) async throws -> URLResponse {
            let task = session.dataTask(with: request)
            return try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    let result: Result<URLResponse, any Error>? = state.withLock { state in
                        if let result = state.early.removeValue(forKey: task.taskIdentifier) { return result }
                        state.pending[task.taskIdentifier] = continuation
                        return nil
                    }
                    if let result { continuation.resume(with: result) } else { task.resume() }
                }
            } onCancel: { task.cancel() }
        }

        private func finish(_ task: URLSessionTask, _ result: Result<URLResponse, any Error>) {
            let continuation: CheckedContinuation<URLResponse, any Error>? = state.withLock { state in
                if let continuation = state.pending.removeValue(forKey: task.taskIdentifier) { return continuation }
                if state.early[task.taskIdentifier] == nil { state.early[task.taskIdentifier] = result }
                return nil
            }
            continuation?.resume(with: result)
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
                finish(task, .failure(error ?? URLError(.badServerResponse)))
            }
        }
    }
}

/// What a background task's bytes are for, kept in the task's description so it survives an app relaunch.
struct BackgroundTaskRecord: Codable, Hashable {
    /// The file's path, relative to the home directory when it lies inside it: an iOS app's container path changes
    /// across updates, its home-relative paths don't.
    let destination: String
    let offset: Int64
    let size: Int64
    /// The bundle-relative path, for errors.
    let file: String

    init(destination: String, offset: Int64, size: Int64, file: String) {
        self.destination = destination
        self.offset = offset
        self.size = size
        self.file = file
    }

    init(_ request: DownloadRequest) {
        self.init(destination: Self.portablePath(request.destination.path), offset: request.offset, size: request.size,
                  file: request.file)
    }

    init?(encoded: String?) {
        guard let encoded, let value = try? JSONDecoder().decode(Self.self, from: Data(encoded.utf8)) else { return nil }
        self = value
    }

    var encoded: String { String(decoding: (try? JSONEncoder().encode(self)) ?? Data(), as: UTF8.self) }

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

    private func failure(_ reason: AetherError.DownloadFailure) -> AetherError { .download(file: file, reason: reason) }

    /// Moves a finished download into place: appended when it continues the file, replacing it when it starts at byte 0
    /// (a `200`, or a `206` from the start). A task resumed from the system's resume data downloads into one file that
    /// also holds the bytes fetched before the stop, so where the file starts comes from the `Content-Range` end and
    /// the received size, not the range's start. A download larger than the pinned size is refused. Runs inside the
    /// delegate callback, before the system deletes `location`.
    func apply(downloadedFile location: URL, response: URLResponse?) throws(AetherError) {
        guard let http = response as? HTTPURLResponse else { throw failure(.transport("not an HTTP response")) }
        let target = destinationURL
        let received = Self.size(of: location)
        do {
            switch http.statusCode {
            case 206:
                guard let range = ContentRange(http.value(forHTTPHeaderField: "Content-Range")),
                      range.total.map({ $0 == size }) ?? true
                else {
                    throw failure(.unexpectedContent("unexpected Content-Range"))
                }
                guard range.end < size, received <= range.end + 1 else { throw failure(.unexpectedContent("more than the catalog's \(size) bytes")) }
                let start = range.end + 1 - received
                if start == 0 { try replace(target, with: location); return }
                guard start == offset, Self.size(of: target) == offset else {
                    throw failure(.unexpectedContent("Content-Range does not continue at \(offset)"))
                }
                let output = try FileHandle(forWritingTo: target)
                defer { try? output.close() }
                try output.seekToEnd()
                let input = try FileHandle(forReadingFrom: location)
                defer { try? input.close() }
                while let chunk = try input.read(upToCount: 4 << 20), !chunk.isEmpty { try output.write(contentsOf: chunk) }
                try output.synchronize()
            case 200:
                guard received <= size else { throw failure(.unexpectedContent("more than the catalog's \(size) bytes")) }
                try replace(target, with: location)
            default:
                throw failure(.http(status: http.statusCode))
            }
        } catch let error as AetherError {
            throw error
        } catch {
            throw .runtime(underlying: error)
        }
    }

    private func replace(_ target: URL, with location: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
        try fm.moveItem(at: location, to: target)
    }

    /// Where a task's download starts in the file, once its response has arrived. A task's expected byte count covers
    /// its whole download, including what a resumed task fetched before it stopped, so when known it gives the start
    /// even if the first response sent the whole file and the resumed one honoured the range. Otherwise: byte 0 after
    /// a `200` or a `206` from the start, else the record's offset. A content-encoded body's expected count is its
    /// encoded size, so it isn't used then.
    func base(of response: URLResponse?, expectedBytes: Int64) -> Int64 {
        let http = response as? HTTPURLResponse
        let encoding = http?.value(forHTTPHeaderField: "Content-Encoding")?.lowercased()
        if expectedBytes > 0, expectedBytes <= size, encoding == nil || encoding == "identity" {
            return size - expectedBytes
        }
        guard let http else { return min(offset, size) }
        if http.statusCode == 200 { return 0 }
        let range = ContentRange(http.value(forHTTPHeaderField: "Content-Range"))
        return range?.start == 0 ? 0 : min(offset, size)
    }

    static func size(of url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? 0
    }
}

/// The session's delegate: routes its tasks to waiting downloads, and lands every finished task's bytes.
final class BackgroundCoordinator: NSObject, URLSessionDownloadDelegate, Sendable {
    private struct Waiter {
        let continuation: CheckedContinuation<Void, any Error>
        let progress: @Sendable (Int64) -> Void
    }

    private struct State {
        var waiters: [Int: [Waiter]] = [:]          // by task identifier
        var applied: [Int: AetherError?] = [:]      // a finished download's result, until didComplete
        var failures: [Int: AetherError] = [:]      // a task cancelled for a reason of ours
        var outcomes: [Int: (any Error)?] = [:]     // completed tasks, for a download that joins late
        var eventsCompletions: [@Sendable () -> Void] = []
        var eventsFinishedUnclaimed = false
    }

    private let state = Mutex(State())

    func storeEventsCompletion(_ completion: @escaping @Sendable () -> Void) {
        let callNow: Bool = state.withLock { state in
            if state.eventsFinishedUnclaimed { state.eventsFinishedUnclaimed = false; return true }
            state.eventsCompletions.append(completion)
            return false
        }
        if callNow { DispatchQueue.main.async { completion() } }
    }

    func download(
        record: BackgroundTaskRecord, in session: URLSession, progress: @escaping @Sendable (Int64) -> Void,
        resolve: () async throws -> (url: URL, headers: [String: String])
    ) async throws {
        let task: URLSessionTask
        if let existing = await existingTask(for: record, in: session) {
            task = existing
        } else {
            task = Self.newTask(record, resolved: try await resolve(), in: session)
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let outcome: (any Error)?? = state.withLock { state in
                    if let outcome = state.outcomes[task.taskIdentifier] { return .some(outcome) }
                    state.waiters[task.taskIdentifier, default: []].append(Waiter(continuation: continuation, progress: progress))
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
        (task.originalRequest?.allHTTPHeaderFields ?? [:]).keys.contains { !RedirectPolicy.transportHeaders.contains($0.lowercased()) }
    }

    /// A running task, from this or an earlier launch, that writes the same file from the same offset.
    private func existingTask(for record: BackgroundTaskRecord, in session: URLSession) async -> URLSessionTask? {
        let tasks = await session.allTasks
        return tasks.first { task in
            guard task.state == .running || task.state == .suspended,
                  let other = BackgroundTaskRecord(encoded: task.taskDescription)
            else { return false }
            return other.destinationURL.standardizedFileURL.path == record.destinationURL.standardizedFileURL.path
                && other.offset == record.offset
        }
    }

    private static func newTask(
        _ record: BackgroundTaskRecord, resolved: (url: URL, headers: [String: String]), in session: URLSession
    ) -> URLSessionTask {
        let task: URLSessionDownloadTask
        if let data = try? Data(contentsOf: record.resumeDataURL) {
            try? FileManager.default.removeItem(at: record.resumeDataURL)
            task = session.downloadTask(withResumeData: data)
        } else {
            var request = URLRequest(url: resolved.url)
            for (name, value) in resolved.headers { request.setValue(value, forHTTPHeaderField: name) }
            if record.offset > 0 { request.setValue("bytes=\(record.offset)-", forHTTPHeaderField: "Range") }
            request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            task = session.downloadTask(with: request)
        }
        task.taskDescription = record.encoded
        return task
    }

    // MARK: URLSessionDownloadDelegate

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) else { return }
        let written = max(0, totalBytesWritten)
        let base = record.base(of: downloadTask.response, expectedBytes: totalBytesExpectedToWrite)
        guard written <= record.size - base else {
            // More than the pinned remainder: wrong, and it could fill the disk.
            state.withLock {
                $0.failures[downloadTask.taskIdentifier] = .download(
                    file: record.file, reason: .unexpectedContent("more than the catalog's \(record.size) bytes"))
            }
            downloadTask.cancel()
            return
        }
        for waiter in state.withLock({ $0.waiters[downloadTask.taskIdentifier] ?? [] }) { waiter.progress(base + written) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let outcome: AetherError?
        if let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) {
            do { try record.apply(downloadedFile: location, response: downloadTask.response); outcome = nil }
            catch { outcome = error }
        } else {
            outcome = .download(file: "", reason: .transport("background task without a record"))
        }
        state.withLock { $0.applied[downloadTask.taskIdentifier] = .some(outcome) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let record = BackgroundTaskRecord(encoded: task.taskDescription)
        // A failed transfer keeps the system's resume data for the next task, unless it holds the app's headers.
        if let data = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data, let record,
           !Self.carriesAppHeaders(task) {
            try? data.write(to: record.resumeDataURL)
        }
        let file = record?.file ?? ""
        let (pending, outcome): ([Waiter], (any Error)?) = state.withLock { state in
            let pending = state.waiters.removeValue(forKey: task.taskIdentifier) ?? []
            let finished = state.applied.removeValue(forKey: task.taskIdentifier)
            let ours = state.failures.removeValue(forKey: task.taskIdentifier)
            let outcome: (any Error)?
            if let ours {
                outcome = ours
            } else if let error = error as? URLError, error.code == .cancelled {
                outcome = CancellationError()
            } else if let error {
                outcome = AetherError.download(file: file, reason: .transport(describeTransportError(error)))
            } else if let finished {
                outcome = finished
            } else {
                // No body was delivered (an error status): report it.
                let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
                outcome = AetherError.download(file: file, reason: .http(status: status))
            }
            state.outcomes[task.taskIdentifier] = .some(outcome)
            if state.outcomes.count > 256, let oldest = state.outcomes.keys.min() { state.outcomes[oldest] = nil }
            return (pending, outcome)
        }
        for waiter in pending {
            if let outcome { waiter.continuation.resume(throwing: outcome) } else { waiter.continuation.resume() }
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        let completions: [@Sendable () -> Void] = state.withLock { state in
            defer { state.eventsCompletions = [] }
            if state.eventsCompletions.isEmpty { state.eventsFinishedUnclaimed = true }
            return state.eventsCompletions
        }
        guard !completions.isEmpty else { return }
        DispatchQueue.main.async { completions.forEach { $0() } }
    }
}
