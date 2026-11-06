import Foundation
import Synchronization
import AetherCore

/// An in-process HTTP fixture behind `URLProtocol`: each test gets its own host, so suites run in parallel.
/// It honours `Range`, records requests, and can fail or corrupt responses on demand.
final class StubSite: Sendable {
    struct Request: Sendable {
        var path: String
        var range: String?
        var authorization: String?
        var method = "GET"
    }

    struct Behavior: Sendable {
        /// Drop the connection after this many body bytes, for this many requests of the path.
        var failAfter: (bytes: Int, times: Int)?
        /// Serve these bytes instead (same length): a corrupt mirror.
        var corrupt: Data?
        /// Answer with this status and no body.
        var status: Int?
        /// Ignore `Range` and always send the whole file with 200.
        var ignoreRange = false
        /// Answer ranged requests with this status and no body (a `416`, say); unranged ones are served.
        var rangeStatus: Int?
        /// Answer a resumed request's `206` with a `Content-Range` starting here instead.
        var contentRangeStart: Int?
        /// Answer a ranged request with a `206` of the whole file (`bytes 0-…`).
        var partialFromStart = false
        /// Append these bytes to the body: more than the catalog's size.
        var extra: Data?
        /// Redirect (302) to this URL instead of serving.
        var redirect: URL?
    }

    private struct State {
        var files: [String: Data] = [:]
        var behaviors: [String: Behavior] = [:]
        var requests: [Request] = []
        var offline = false
    }

    let host: String
    private let state = Mutex(State())

    init() {
        host = "site-\(UUID().uuidString.lowercased()).stub"
        StubURLProtocol.register(self)
    }

    var endpoint: URL { URL(string: "https://\(host)")! }

    func serve(_ path: String, _ data: Data) { state.withLock { $0.files[path] = data } }
    func behave(_ path: String, _ behavior: Behavior) { state.withLock { $0.behaviors[path] = behavior } }
    func setOffline(_ offline: Bool) { state.withLock { $0.offline = offline } }
    var requests: [Request] { state.withLock { $0.requests } }

    func handle(_ request: URLRequest, _ proto: StubURLProtocol) {
        let path = request.url!.path
        let range = request.value(forHTTPHeaderField: "Range")
        let authorization = request.value(forHTTPHeaderField: "Authorization")
        let method = request.httpMethod ?? "GET"
        state.withLock { $0.requests.append(Request(path: path, range: range, authorization: authorization, method: method)) }
        respond(path: path, range: range, proto)
    }

    private func respond(path: String, range: String?, _ proto: StubURLProtocol) {
        let (file, behavior, offline) = state.withLock { s -> (Data?, Behavior, Bool) in
            var behavior = s.behaviors[path] ?? Behavior()
            if var fail = behavior.failAfter, fail.times > 0 {
                fail.times -= 1
                s.behaviors[path]?.failAfter = fail.times > 0 ? fail : nil
                behavior.failAfter = fail
            } else {
                behavior.failAfter = nil
            }
            return (s.files[path], behavior, s.offline)
        }
        if offline { return proto.fail(URLError(.notConnectedToInternet)) }
        if let target = behavior.redirect { return proto.redirect(to: target) }
        if let status = behavior.status { return proto.reply(status: status, headers: [:], body: Data()) }
        if range != nil, let status = behavior.rangeStatus { return proto.reply(status: status, headers: [:], body: Data()) }
        guard var body = behavior.corrupt ?? file else { return proto.reply(status: 404, headers: [:], body: Data()) }
        if let extra = behavior.extra { body += extra }
        var status = 200
        var headers = ["Content-Length": "\(body.count)"]
        if let range, !behavior.ignoreRange, range.hasPrefix("bytes="),
           let start = Int(range.dropFirst(6).split(separator: "-").first ?? "")
        {
            guard start < body.count else { return proto.reply(status: 416, headers: [:], body: Data()) }
            let from = behavior.partialFromStart ? 0 : start
            headers["Content-Range"] = "bytes \(behavior.contentRangeStart ?? from)-\(body.count - 1)/\(body.count)"
            body = body.subdata(in: from..<body.count)
            headers["Content-Length"] = "\(body.count)"
            status = 206
        }
        if let fail = behavior.failAfter {
            proto.reply(status: status, headers: headers, body: body.prefix(fail.bytes), thenFail: URLError(.networkConnectionLost))
        } else {
            proto.reply(status: status, headers: headers, body: body)
        }
    }
}

final class StubURLProtocol: URLProtocol {
    private static let sites = Mutex<[String: StubSite]>([:])

    static func register(_ site: StubSite) { sites.withLock { $0[site.host] = site } }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasSuffix(".stub") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let host = request.url?.host, let site = Self.sites.withLock({ $0[host] }) else {
            return fail(URLError(.cannotFindHost))
        }
        site.handle(request, self)
    }

    override func stopLoading() {}

    func reply(status: Int, headers: [String: String], body: Data, thenFail error: URLError? = nil) {
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // Deliver in pieces, as a network would.
        var offset = 0
        while offset < body.count {
            let end = min(offset + 256 * 1024, body.count)
            client?.urlProtocol(self, didLoad: body.subdata(in: offset..<end))
            offset = end
        }
        if let error {
            // URLSession drops body bytes that are still in flight when the protocol fails, so let them land first,
            // as they would on a real connection that dies later.
            Thread.sleep(forTimeInterval: 0.2)
            client?.urlProtocol(self, didFailWithError: error)
        } else {
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    func redirect(to target: URL) {
        let response = HTTPURLResponse(url: request.url!, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": target.absoluteString])!
        client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: response)
        // If the session refuses the redirect, the 302 itself is the response the task completes with.
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocolDidFinishLoading(self)
    }

    func fail(_ error: URLError) {
        client?.urlProtocol(self, didFailWithError: error)
    }
}

/// Wraps a transport so that no request starts until the test opens the gate.
struct GatedTransport: HTTPTransport {
    let inner: any HTTPTransport
    let gate: Gate

    func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>) {
        await gate.pass()
        return try await inner.get(url, headers: headers)
    }
}

actor Gate {
    private var isOpen = false
    private var waiting: [CheckedContinuation<Void, Never>] = []

    var waitingCount: Int { waiting.count }

    func pass() async {
        if isOpen { return }
        await withCheckedContinuation { waiting.append($0) }
    }

    func open() {
        isOpen = true
        for continuation in waiting { continuation.resume() }
        waiting = []
    }
}
