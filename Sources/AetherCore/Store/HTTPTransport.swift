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

/// Streams HTTP GET bodies. ``ModelStore`` downloads through this, so tests can substitute a stub.
public protocol HTTPTransport: Sendable {
    /// Starts a GET and returns once the response head arrives. Cancelling the body stream cancels the transfer.
    func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>)
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
