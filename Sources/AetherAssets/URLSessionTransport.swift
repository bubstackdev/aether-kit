import Foundation

/// The built-in foreground transport: HTTPS through `URLSession`, resuming a partial file with an HTTP `Range` request.
/// A `206` whose `Content-Range` starts at the file's current size appends; a `200` restarts the file; a `416` for a
/// partial file restarts once without a range; anything else fails with `PreparationError.http`. Redirects must stay
/// on HTTPS, and request headers such as `Authorization` are only forwarded to the original host, so a token meant
/// for a model host never reaches the CDN it redirects to. Transfers run while the app does; see the background
/// transport for transfers that continue while an iOS app is suspended (docs/model-preparation.md).
public struct URLSessionTransport: ModelTransport {
    private let configuration: URLSessionConfiguration

    public init(configuration: URLSessionConfiguration = .default) {
        self.configuration = configuration
    }

    public func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
                      progress: @escaping @Sendable (UInt64) -> Void) async throws {
        guard request.url.scheme?.lowercased() == "https" else {
            throw PreparationError.invalidEntry("not an HTTPS URL: \(request.url.absoluteString)")
        }
        let existing = resume ? Self.size(of: destination) : 0
        do {
            try await transfer(request, to: destination, from: existing, progress: progress)
        } catch let PreparationError.http(status, _) where status == 416 && existing > 0 {
            // The server can't satisfy the range: the partial file doesn't match what it holds. Start over.
            try await transfer(request, to: destination, from: 0, progress: progress)
        }
    }

    private func transfer(_ request: TransportRequest, to destination: URL, from offset: UInt64,
                          progress: @escaping @Sendable (UInt64) -> Void) async throws {
        var urlRequest = URLRequest(url: request.url)
        for (name, value) in request.headers { urlRequest.setValue(value, forHTTPHeaderField: name) }
        if offset > 0 { urlRequest.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        urlRequest.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let delegate = try StreamingDelegate(destination: destination, offset: offset, request: request, progress: progress)
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        let task = session.dataTask(with: urlRequest)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                delegate.start(continuation)
                task.resume()
            }
        } onCancel: { task.cancel() }
    }

    static func size(of url: URL) -> UInt64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.uint64Value ?? 0
    }

    /// `bytes START-END/TOTAL` → (START, END, TOTAL); TOTAL may be `*`.
    static func contentRange(_ value: String) -> (start: UInt64, end: UInt64, total: UInt64?)? {
        let parts = value.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 1)
        guard parts.count == 2, parts[0].lowercased() == "bytes" else { return nil }
        let rangeAndTotal = parts[1].split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2, let dash = rangeAndTotal[0].firstIndex(of: "-"),
              let start = UInt64(rangeAndTotal[0][..<dash]),
              let end = UInt64(rangeAndTotal[0][rangeAndTotal[0].index(after: dash)...]), end >= start else { return nil }
        return (start, end, rangeAndTotal[1] == "*" ? nil : UInt64(rangeAndTotal[1]))
    }
}

extension ModelTransport where Self == URLSessionTransport {
    /// HTTPS in the foreground with the default session configuration.
    public static var urlSession: URLSessionTransport { URLSessionTransport() }
}

/// Writes one response to disk as it arrives and resumes the waiting `fetch` when the task ends.
private final class StreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let destination: URL
    private let offset: UInt64
    private let request: TransportRequest
    private let progress: @Sendable (UInt64) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, any Error>?
    private var handle: FileHandle?
    private var written: UInt64 = 0
    private var failure: (any Error)?

    init(destination: URL, offset: UInt64, request: TransportRequest, progress: @escaping @Sendable (UInt64) -> Void) throws {
        self.destination = destination; self.offset = offset; self.request = request; self.progress = progress
    }

    func start(_ continuation: CheckedContinuation<Void, any Error>) { lock.withLock { self.continuation = continuation } }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        do {
            guard let http = response as? HTTPURLResponse else { throw PreparationError.network("not an HTTP response") }
            let append: Bool
            switch http.statusCode {
            case 206:
                guard let header = http.value(forHTTPHeaderField: "Content-Range"),
                      let range = URLSessionTransport.contentRange(header),
                      range.start == offset || range.start == 0,
                      range.total.map({ $0 == request.file.byteCount }) ?? true else {
                    throw PreparationError.network("unexpected Content-Range for \(request.file.path)")
                }
                // A range from the start is the whole file: restart it, as for a 200.
                append = range.start == offset && offset > 0
            case 200:
                guard http.expectedContentLength < 0 || UInt64(http.expectedContentLength) == request.file.byteCount else {
                    throw PreparationError.integrityMismatch(path: request.file.path)
                }
                append = false
            default:
                throw PreparationError.http(status: http.statusCode, path: request.file.path)
            }
            if !append || !FileManager.default.fileExists(atPath: destination.path) {
                guard FileManager.default.createFile(atPath: destination.path, contents: nil,
                                                     attributes: [.posixPermissions: 0o600]) else {
                    throw PreparationError.fileSystem(errno)
                }
            }
            let handle = try FileHandle(forWritingTo: destination)
            if append { try handle.truncate(atOffset: offset); try handle.seekToEnd() }
            lock.withLock { self.handle = handle; self.written = append ? offset : 0 }
            completionHandler(.allow)
        } catch {
            lock.withLock { failure = error }
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let total: UInt64? = lock.withLock {
            guard failure == nil, let handle else { return nil }
            // Never write past the pinned size: a server sending more is wrong, and could fill the disk.
            guard written + UInt64(data.count) <= request.file.byteCount else {
                failure = PreparationError.integrityMismatch(path: request.file.path)
                dataTask.cancel()
                return nil
            }
            do {
                try handle.write(contentsOf: data)
                written += UInt64(data.count)
                return written
            } catch {
                failure = PreparationError.fileSystem(Int32((error as NSError).code))
                dataTask.cancel()
                return nil
            }
        }
        if let total { progress(total) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        guard let url = newRequest.url, url.scheme?.lowercased() == "https" else {
            lock.withLock { failure = PreparationError.network("redirect away from HTTPS for \(request.file.path)") }
            completionHandler(nil)
            task.cancel()
            return
        }
        // Carry the range; forward the app's headers only while the host is the one they were meant for.
        var redirected = URLRequest(url: url)
        let original = task.originalRequest
        for name in ["Range", "Accept-Encoding"] {
            if let value = original?.value(forHTTPHeaderField: name) { redirected.setValue(value, forHTTPHeaderField: name) }
        }
        if url.host?.lowercased() == request.url.host?.lowercased() {
            for (name, value) in request.headers { redirected.setValue(value, forHTTPHeaderField: name) }
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let (continuation, outcome): (CheckedContinuation<Void, any Error>?, (any Error)?) = lock.withLock {
            try? handle?.synchronize()
            try? handle?.close()
            handle = nil
            let pending = self.continuation
            self.continuation = nil
            if let failure { return (pending, failure) }
            if let error = error as? URLError, error.code == .cancelled { return (pending, CancellationError()) }
            return (pending, error.map { PreparationError.network(($0 as NSError).localizedDescription) })
        }
        if let outcome { continuation?.resume(throwing: outcome) } else { continuation?.resume() }
    }
}
