public import Foundation
import Synchronization

/// The status line and headers of a response. Header names are lowercased.
public struct HTTPResponseHead: Sendable, Hashable {
    public var status: Int
    public var headers: [String: String]

    public init(status: Int, headers: [String: String]) {
        self.status = status
        self.headers = Dictionary(headers.map { ($0.key.lowercased(), $0.value) }, uniquingKeysWith: { a, _ in a })
    }
}

/// One file a transport brings to its full size on disk.
public struct DownloadRequest: Sendable, Hashable {
    public var url: URL
    /// The app's headers for this request (an authorization header, say). Transports send them only to `url`'s host:
    /// a redirect to another host drops them.
    public var headers: [String: String]
    /// Where the bytes go.
    public var destination: URL
    /// The bytes of `destination` to keep and resume after; 0 starts the file over.
    public var offset: Int64
    /// The catalog's size for the file. A transport never writes past it.
    public var size: Int64
    /// The file's bundle-relative path, for errors.
    public var file: String

    public init(url: URL, headers: [String: String], destination: URL, offset: Int64, size: Int64, file: String) {
        self.url = url
        self.headers = headers
        self.destination = destination
        self.offset = offset
        self.size = size
        self.file = file
    }
}

/// How ``ModelStore`` reaches the network, so tests can substitute a stub and an app can choose a background session.
///
/// `get` streams one response. `download` brings one file to its catalog size; its default implementation streams
/// through `get`, and ``BackgroundTransport`` replaces it with system download tasks. Either way the transport only
/// moves bytes: the store checks each file's size and SHA-256 before accepting it.
public protocol HTTPTransport: Sendable {
    /// Starts a GET and returns once the response head arrives. Cancelling the body stream cancels the transfer.
    func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>)

    /// Writes `request.url`'s bytes to `request.destination`: appended after the first `request.offset` bytes when
    /// the server resumes with a matching `206`, from byte 0 after a `200` (or when `offset` is 0). Reports the file's
    /// bytes on disk as they arrive. Throws ``AetherError/download(file:reason:)`` (a `416` arrives as
    /// `.http(status: 416)`; the store restarts the file once) or `CancellationError`.
    func download(_ request: DownloadRequest, progress: @escaping @Sendable (Int64) -> Void) async throws
}

extension HTTPTransport {
    /// Streams the file through ``get(_:headers:)``.
    public func download(_ request: DownloadRequest, progress: @escaping @Sendable (Int64) -> Void) async throws {
        var headers = request.headers
        headers["Accept-Encoding"] = "identity"  // byte counts and hashes are of the stored file, not a coding
        if request.offset > 0 { headers["Range"] = "bytes=\(request.offset)-" }

        let head: HTTPResponseHead
        let body: AsyncThrowingStream<Data, any Error>
        do {
            (head, body) = try await get(request.url, headers: headers)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as AetherError {
            throw error
        } catch {
            throw AetherError.download(file: request.file, reason: .transport(describeTransportError(error)))
        }

        let start: Int64
        switch head.status {
        case 206:
            // A range that continues the file appends; one from byte 0 is the whole file again, so it restarts.
            guard let range = ContentRange(head.headers["content-range"]), range.start == request.offset || range.start == 0,
                  range.total.map({ $0 == request.size }) ?? true
            else {
                throw AetherError.download(
                    file: request.file,
                    reason: .unexpectedContent("Content-Range '\(head.headers["content-range"] ?? "")' does not continue at \(request.offset)"))
            }
            start = range.start
        case 200:
            start = 0  // the server ignored the range (or none was sent): start over
        default:
            throw AetherError.download(file: request.file, reason: .http(status: head.status))
        }

        let local = request.destination
        do {
            if start == 0 { FileManager.default.createFile(atPath: local.path, contents: nil) }
            let handle = try FileHandle(forWritingTo: local)
            defer { try? handle.close() }
            try handle.truncate(atOffset: UInt64(start))
            try handle.seekToEnd()

            var written = start
            do {
                for try await chunk in body {
                    // Never write past the pinned size: a server sending more is wrong, and could fill the disk.
                    guard written + Int64(chunk.count) <= request.size else {
                        throw AetherError.download(file: request.file, reason: .unexpectedContent("more than the catalog's \(request.size) bytes"))
                    }
                    try handle.write(contentsOf: chunk)
                    written += Int64(chunk.count)
                    progress(written)
                }
            } catch let error as AetherError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // The bytes so far stay on disk; the next attempt resumes from them.
                throw AetherError.download(file: request.file, reason: .transport(describeTransportError(error)))
            }
            guard written == request.size else {
                throw AetherError.download(file: request.file, reason: .transport("connection closed after \(written) of \(request.size) bytes"))
            }
        } catch let error as AetherError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw AetherError.runtime(underlying: error)
        }
    }
}

/// `bytes START-END/TOTAL`; TOTAL may be `*`.
struct ContentRange: Sendable, Hashable {
    var start: Int64
    var end: Int64
    var total: Int64?

    init?(_ header: String?) {
        guard let header else { return nil }
        let parts = header.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 1)
        guard parts.count == 2, parts[0].lowercased() == "bytes" else { return nil }
        let rangeAndTotal = parts[1].split(separator: "/", maxSplits: 1)
        guard rangeAndTotal.count == 2, let dash = rangeAndTotal[0].firstIndex(of: "-"),
              let start = Int64(rangeAndTotal[0][..<dash]),
              let end = Int64(rangeAndTotal[0][rangeAndTotal[0].index(after: dash)...]), start >= 0, end >= start
        else { return nil }
        if rangeAndTotal[1] == "*" {
            total = nil
        } else {
            guard let total = Int64(rangeAndTotal[1]) else { return nil }
            self.total = total
        }
        self.start = start
        self.end = end
    }
}

func describeTransportError(_ error: any Error) -> String {
    if let url = error as? URLError { return "URLError \(url.code.rawValue): \(url.localizedDescription)" }
    return "\(error)"
}

/// The redirect rule both URLSession transports share: never from HTTPS down to another scheme; carry the
/// transport's own headers (`Range`, `Accept-Encoding`); carry the app's headers only while the host is the one
/// they were meant for, so a token for the model host never reaches the CDN it redirects to.
enum RedirectPolicy {
    static let transportHeaders: Set<String> = ["range", "accept-encoding"]

    /// The request to follow `newRequest` with, or nil to refuse the redirect.
    static func follow(from original: URLRequest?, to newRequest: URLRequest) -> URLRequest? {
        guard let url = newRequest.url else { return nil }
        let wasSecure = original?.url?.scheme?.lowercased() == "https"
        if wasSecure, url.scheme?.lowercased() != "https" { return nil }
        var redirected = URLRequest(url: url)
        let sameHost = url.host?.lowercased() == original?.url?.host?.lowercased()
        for (name, value) in original?.allHTTPHeaderFields ?? [:] {
            if sameHost || transportHeaders.contains(name.lowercased()) { redirected.setValue(value, forHTTPHeaderField: name) }
        }
        return redirected
    }
}

/// ``HTTPTransport`` on `URLSession`, streaming chunks as they arrive (no whole-file buffering, no temp files).
public struct URLSessionTransport: HTTPTransport {
    private let session: URLSession

    public init(session: URLSession = URLSession(configuration: .default)) {
        self.session = session
    }

    public func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>) {
        var request = URLRequest(url: url)
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        let (raw, continuation) = AsyncThrowingStream<Data, any Error>.makeStream()
        let flow = FlowControl()
        let delegate = StreamingDelegate(body: continuation, flow: flow)
        let task = session.dataTask(with: request)
        task.delegate = delegate
        flow.task = DataTaskControl(task)
        continuation.onTermination = { _ in task.cancel() }
        // Backpressure: the delegate suspends the task once `FlowControl.highWater` bytes wait unread, and the
        // consumer's pulls resume it below `lowWater`, so a slow disk never lets the body pile up in memory. No chunk
        // is ever dropped.
        let pump = Pump(raw.makeAsyncIterator(), flow: flow)
        let body = AsyncThrowingStream<Data, any Error> { try await pump.next() }
        let head = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (response: CheckedContinuation<HTTPResponseHead, any Error>) in
                delegate.awaitResponse(response)
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
        return (head, body)
    }
}

/// What ``FlowControl`` needs of a data task: the seam that lets tests drive the interleavings deterministically.
protocol FlowControlledTask: AnyObject, Sendable {
    func suspend()
    func resume()
}

/// A data task as ``FlowControl`` drives it (a wrapper, so the package doesn't extend a Foundation class).
final class DataTaskControl: FlowControlledTask {
    let task: URLSessionDataTask
    init(_ task: URLSessionDataTask) { self.task = task }
    func suspend() { task.suspend() }
    func resume() { task.resume() }
}

/// The bytes received and not yet read, and whether the data task is suspended for them.
///
/// Each `suspend()`/`resume()` runs under the same lock as the bookkeeping that decided it, so the task's actual state
/// always matches `isSuspended`: a consumer that drains the buffer can't resume the task before the pending suspend
/// lands and leave it stalled with nothing buffered. Neither call re-enters ``FlowControl`` (URLSession delivers
/// delegate callbacks on its own queue), so holding the lock across them can't deadlock.
final class FlowControl: Sendable {
    static let highWater = 8 << 20
    static let lowWater = 2 << 20
    private struct State { var buffered = 0; var suspended = false; var task: (any FlowControlledTask)? }
    private let state = Mutex(State())

    var task: (any FlowControlledTask)? {
        get { state.withLock { $0.task } }
        set { state.withLock { $0.task = newValue } }
    }

    /// A chunk arrived: past the high-water mark the task is suspended until the consumer catches up.
    func received(_ count: Int) {
        state.withLock { state in
            state.buffered += count
            guard state.buffered > Self.highWater, !state.suspended, let task = state.task else { return }
            state.suspended = true
            task.suspend()
        }
    }

    /// The consumer read a chunk: below the low-water mark a suspended task resumes.
    func consumed(_ count: Int) {
        state.withLock { state in
            state.buffered -= count
            guard state.suspended, state.buffered < Self.lowWater, let task = state.task else { return }
            state.suspended = false
            task.resume()
        }
    }

    var bufferedBytes: Int { state.withLock { $0.buffered } }
    var isSuspended: Bool { state.withLock { $0.suspended } }
}

/// The consumer's side of the body: pulls one chunk at a time and reports it read. One consumer, as a stream has.
private final class Pump: @unchecked Sendable {
    private var iterator: AsyncThrowingStream<Data, any Error>.Iterator
    private let flow: FlowControl

    init(_ iterator: AsyncThrowingStream<Data, any Error>.Iterator, flow: FlowControl) {
        self.iterator = iterator
        self.flow = flow
    }

    func next() async throws -> Data? {
        let chunk = try await iterator.next()
        if let chunk { flow.consumed(chunk.count) }
        return chunk
    }
}

/// Bridges one data task's delegate callbacks into a response continuation and a body stream.
private final class StreamingDelegate: NSObject, URLSessionDataDelegate, Sendable {
    private let body: AsyncThrowingStream<Data, any Error>.Continuation
    private let flow: FlowControl
    /// The response continuation, and the first outcome. A task cancelled before it was resumed (the caller was
    /// already cancelled when `get` began) completes on the delegate queue without waiting for `resume()`, possibly
    /// before ``awaitResponse(_:)`` has installed the continuation; that outcome is kept for it rather than lost, or
    /// `get` would never return.
    private struct ResponseState {
        var continuation: CheckedContinuation<HTTPResponseHead, any Error>?
        var early: Result<HTTPResponseHead, any Error>?
        var answered = false
    }
    private let response = Mutex(ResponseState())

    init(body: AsyncThrowingStream<Data, any Error>.Continuation, flow: FlowControl) {
        self.body = body
        self.flow = flow
    }

    func awaitResponse(_ continuation: CheckedContinuation<HTTPResponseHead, any Error>) {
        let early = response.withLock { state -> Result<HTTPResponseHead, any Error>? in
            if let early = state.early {
                state.early = nil
                return early
            }
            state.continuation = continuation
            return nil
        }
        if let early { continuation.resume(with: early) }
    }

    /// Delivers the first outcome only: to the waiting continuation, or kept until one is installed.
    private func answer(_ outcome: Result<HTTPResponseHead, any Error>) {
        let waiting = response.withLock { state -> CheckedContinuation<HTTPResponseHead, any Error>? in
            guard !state.answered else { return nil }
            state.answered = true
            guard let continuation = state.continuation else {
                state.early = outcome
                return nil
            }
            state.continuation = nil
            return continuation
        }
        waiting?.resume(with: outcome)
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let http = response as? HTTPURLResponse
        var headers: [String: String] = [:]
        for (key, value) in http?.allHeaderFields ?? [:] {
            if let key = key as? String, let value = value as? String { headers[key] = value }
        }
        answer(.success(HTTPResponseHead(status: http?.statusCode ?? 0, headers: headers)))
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        guard let redirected = RedirectPolicy.follow(from: task.originalRequest, to: request) else {
            // A redirect from HTTPS to plain HTTP: refuse it, and fail rather than hand back the 3xx as the response.
            answer(.failure(URLError(.appTransportSecurityRequiresSecureConnection)))
            completionHandler(nil)
            task.cancel()
            return
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        body.yield(data)
        flow.received(data.count)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let error {
            answer(.failure(error))
            body.finish(throwing: error)
        } else {
            answer(.failure(URLError(.badServerResponse)))
            body.finish()
        }
    }
}
