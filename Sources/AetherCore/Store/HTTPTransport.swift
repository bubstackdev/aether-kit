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
        let (body, continuation) = AsyncThrowingStream<Data, any Error>.makeStream()
        let delegate = StreamingDelegate(body: continuation)
        let task = session.dataTask(with: request)
        task.delegate = delegate
        continuation.onTermination = { _ in task.cancel() }
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

/// Bridges one data task's delegate callbacks into a response continuation and a body stream.
private final class StreamingDelegate: NSObject, URLSessionDataDelegate, Sendable {
    private let body: AsyncThrowingStream<Data, any Error>.Continuation
    private let response = Mutex<CheckedContinuation<HTTPResponseHead, any Error>?>(nil)

    init(body: AsyncThrowingStream<Data, any Error>.Continuation) {
        self.body = body
    }

    func awaitResponse(_ continuation: CheckedContinuation<HTTPResponseHead, any Error>) {
        response.withLock { $0 = continuation }
    }

    private func takeResponse() -> CheckedContinuation<HTTPResponseHead, any Error>? {
        response.withLock { pending in
            defer { pending = nil }
            return pending
        }
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
        takeResponse()?.resume(returning: HTTPResponseHead(status: http?.statusCode ?? 0, headers: headers))
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        guard let redirected = RedirectPolicy.follow(from: task.originalRequest, to: request) else {
            // A redirect from HTTPS to plain HTTP: refuse it, and fail rather than hand back the 3xx as the response.
            takeResponse()?.resume(throwing: URLError(.appTransportSecurityRequiresSecureConnection))
            completionHandler(nil)
            task.cancel()
            return
        }
        completionHandler(redirected)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        body.yield(data)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        if let error {
            takeResponse()?.resume(throwing: error)
            body.finish(throwing: error)
        } else {
            takeResponse()?.resume(throwing: URLError(.badServerResponse))
            body.finish()
        }
    }
}
