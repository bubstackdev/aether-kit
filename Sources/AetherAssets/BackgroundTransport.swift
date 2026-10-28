import Foundation

/// HTTPS through a background `URLSession`, so the system carries transfers on while an iOS app is suspended or
/// terminated (and in macOS apps). One download task per file. A partial file resumes with a `Range` request, as in
/// the foreground transport: a `206` whose `Content-Range` starts at the file's size is appended, a `200` replaces the
/// file, anything else fails. Every task records where its bytes belong, so a task that finishes while no `fetch` is
/// waiting (the app was relaunched to receive it) still lands in the download area, and the next preparation run finds
/// the file complete. A later `fetch` for a file whose task is still running joins that task instead of starting
/// another.
///
/// The app connects the transport to its lifecycle in one place (docs/model-preparation.md):
/// ```swift
/// // UIApplicationDelegate.application(_:handleEventsForBackgroundURLSession:completionHandler:),
/// // or SwiftUI's .backgroundTask(.urlSession(identifier)) { ... }
/// BackgroundTransport.handleEvents(identifier: identifier, completionHandler: completionHandler)
/// ```
public final class BackgroundTransport: ModelTransport, Sendable {
    public let identifier: String
    private let coordinator: BackgroundCoordinator

    /// `configure` adjusts the background configuration before the session is created (for example
    /// `isDiscretionary` or `allowsCellularAccess`). Every transport with the same identifier shares one session; the
    /// first one created decides its configuration.
    public init(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void = { _ in }) {
        self.identifier = identifier
        coordinator = BackgroundCoordinator.shared(identifier: identifier, configure: configure)
    }

    /// Hands the system's background-session events to Kit. Call it from the app's background-session hook; Kit calls
    /// `completionHandler` on the main queue once every event for the session has been delivered.
    public static func handleEvents(identifier: String, completionHandler: @escaping @Sendable () -> Void) {
        BackgroundCoordinator.shared(identifier: identifier, configure: { _ in }).storeEventsCompletion(completionHandler)
    }

    public func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
                      progress: @escaping @Sendable (UInt64) -> Void) async throws {
        guard request.url.scheme?.lowercased() == "https" else {
            throw PreparationError.invalidEntry("not an HTTPS URL: \(request.url.absoluteString)")
        }
        let offset = resume ? URLSessionTransport.size(of: destination) : 0
        let record = BackgroundTaskRecord(destination: BackgroundTaskRecord.portablePath(destination.path), offset: offset,
                                          byteCount: request.file.byteCount, path: request.file.path)
        try await coordinator.download(request, record: record, progress: progress)
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

    static func portablePath(_ absolute: String) -> String {
        let home = URL(fileURLWithPath: NSHomeDirectory()).standardizedFileURL.path + "/"
        let path = URL(fileURLWithPath: absolute).standardizedFileURL.path
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

    /// Moves a finished download into place: appended after a matching `206`, replacing the file after a `200`.
    /// Runs inside the delegate callback, before the system deletes `location`.
    func apply(downloadedFile location: URL, response: URLResponse?) throws {
        guard let http = response as? HTTPURLResponse else { throw PreparationError.network("not an HTTP response") }
        let target = destinationURL
        switch http.statusCode {
        case 206:
            guard let header = http.value(forHTTPHeaderField: "Content-Range"),
                  let range = URLSessionTransport.contentRange(header), range.start == offset,
                  range.total.map({ $0 == byteCount }) ?? true,
                  URLSessionTransport.size(of: target) == offset else {
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
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            if FileManager.default.fileExists(atPath: target.path) { try FileManager.default.removeItem(at: target) }
            try FileManager.default.moveItem(at: location, to: target)
        default:
            throw PreparationError.http(status: http.statusCode, path: path)
        }
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

    static func shared(identifier: String, configure: @escaping @Sendable (URLSessionConfiguration) -> Void) -> BackgroundCoordinator {
        registryLock.withLock {
            if let existing = registry[identifier] { return existing }
            let coordinator = BackgroundCoordinator(identifier: identifier, configure: configure)
            registry[identifier] = coordinator
            return coordinator
        }
    }

    private let lock = NSLock()
    private var session: URLSession!
    private var waiters: [Int: [Waiter]] = [:]          // by task identifier
    private var outcomes: [Int: (any Error)?] = [:]     // a finished download's result, until didComplete
    private var eventsCompletion: (@Sendable () -> Void)?

    private init(identifier: String, configure: @Sendable (URLSessionConfiguration) -> Void) {
        super.init()
        let configuration = URLSessionConfiguration.background(withIdentifier: identifier)
        configuration.sessionSendsLaunchEvents = true
        configure(configuration)
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
    }

    func storeEventsCompletion(_ completion: @escaping @Sendable () -> Void) { lock.withLock { eventsCompletion = completion } }

    func download(_ request: TransportRequest, record: BackgroundTaskRecord,
                  progress: @escaping @Sendable (UInt64) -> Void) async throws {
        let task = await existingTask(for: record) ?? newTask(request, record: record)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let finished: Bool = lock.withLock {
                    if task.state == .completed { return true }
                    waiters[task.taskIdentifier, default: []].append(Waiter(continuation: continuation, progress: progress))
                    return false
                }
                if finished { continuation.resume() }  // the file landed before we joined; preparation checks its size
                if task.state == .suspended { task.resume() }
            }
        } onCancel: { task.cancel() }
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

    private func newTask(_ request: TransportRequest, record: BackgroundTaskRecord) -> URLSessionTask {
        var urlRequest = URLRequest(url: request.url)
        for (name, value) in request.headers { urlRequest.setValue(value, forHTTPHeaderField: name) }
        if record.offset > 0 { urlRequest.setValue("bytes=\(record.offset)-", forHTTPHeaderField: "Range") }
        urlRequest.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let task = session.downloadTask(with: urlRequest)
        task.taskDescription = record.encoded
        return task
    }

    // MARK: URLSessionDownloadDelegate

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) else { return }
        let total = record.offset + UInt64(max(0, totalBytesWritten))
        for waiter in lock.withLock({ waiters[downloadTask.taskIdentifier] ?? [] }) { waiter.progress(total) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        let outcome: (any Error)?
        if let record = BackgroundTaskRecord(encoded: downloadTask.taskDescription) {
            do { try record.apply(downloadedFile: location, response: downloadTask.response); outcome = nil }
            catch { outcome = error }
        } else {
            outcome = PreparationError.network("background task without a record")
        }
        lock.withLock { outcomes[downloadTask.taskIdentifier] = .some(outcome) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        guard let url = newRequest.url, url.scheme?.lowercased() == "https" else { completionHandler(nil); return }
        // As in the foreground transport: carry the range, and the app's headers only while the host is unchanged.
        var redirected = URLRequest(url: url)
        let original = task.originalRequest
        let sameHost = url.host?.lowercased() == original?.url?.host?.lowercased()
        for (name, value) in original?.allHTTPHeaderFields ?? [:]
            where sameHost || ["range", "accept-encoding"].contains(name.lowercased()) {
            redirected.setValue(value, forHTTPHeaderField: name)
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let (pending, outcome): ([Waiter], (any Error)?) = lock.withLock {
            let pending = waiters.removeValue(forKey: task.taskIdentifier) ?? []
            let finished = outcomes.removeValue(forKey: task.taskIdentifier)
            if let error = error as? URLError, error.code == .cancelled { return (pending, CancellationError()) }
            if let error { return (pending, PreparationError.network((error as NSError).localizedDescription)) }
            if let finished { return (pending, finished) }
            // A redirect refused, or a response without a body: report the status.
            let status = (task.response as? HTTPURLResponse)?.statusCode ?? 0
            let path = BackgroundTaskRecord(encoded: task.taskDescription)?.path ?? ""
            return (pending, PreparationError.http(status: status, path: path))
        }
        for waiter in pending {
            if let outcome { waiter.continuation.resume(throwing: outcome) } else { waiter.continuation.resume() }
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        guard let completion = lock.withLock({ () -> (@Sendable () -> Void)? in
            defer { eventsCompletion = nil }
            return eventsCompletion
        }) else { return }
        DispatchQueue.main.async { completion() }
    }
}
