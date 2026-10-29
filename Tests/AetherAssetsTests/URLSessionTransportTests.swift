@testable import AetherAssets
import CryptoKit
import Foundation
import Testing

/// Serves scripted HTTP responses for one test's host. Hosts are unique per test, so parallel tests don't interact.
private final class ScriptedProtocol: URLProtocol, @unchecked Sendable {
    enum Reply {
        case file(Data, honourRange: Bool = true, contentRangeStart: UInt64? = nil, cutAfter: Int? = nil, partialFromStart: Bool = false)
        case status(Int)
        case redirect(to: URL)
    }
    struct Seen { let url: URL; let method: String; let headers: [String: String] }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var replies: [String: [Reply]] = [:]
    nonisolated(unsafe) private static var seen: [String: [Seen]] = [:]

    static func script(_ host: String, _ list: [Reply]) { lock.withLock { replies[host] = list } }
    static func requests(_ host: String) -> [Seen] { lock.withLock { seen[host] ?? [] } }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let url = request.url!, host = url.host!
        let reply: Reply? = Self.lock.withLock {
            Self.seen[host, default: []].append(Seen(url: url, method: request.httpMethod ?? "GET", headers: request.allHTTPHeaderFields ?? [:]))
            guard var list = Self.replies[host], !list.isEmpty else { return nil }
            let next = list.removeFirst(); Self.replies[host] = list
            return next
        }
        switch reply {
        case nil:
            send(status: 500, headers: [:], body: Data())
        case let .status(code):
            send(status: code, headers: [:], body: Data())
        case let .redirect(target):
            let response = HTTPURLResponse(url: url, statusCode: 302, httpVersion: "HTTP/1.1", headerFields: ["Location": target.absoluteString])!
            client?.urlProtocol(self, wasRedirectedTo: URLRequest(url: target), redirectResponse: response)
            // If the session's delegate refuses the redirect, the 302 itself is the response the task completes with.
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
        case let .file(data, _, _, _, true):
            send(status: 206, headers: ["Content-Range": "bytes 0-\(data.count - 1)/\(data.count)", "Content-Length": "\(data.count)"],
                 body: data)
        case let .file(data, honourRange, rangeStart, cutAfter, false):
            var start = 0
            if honourRange, let range = request.value(forHTTPHeaderField: "Range"),
               let value = range.split(separator: "=").last?.split(separator: "-").first, let offset = Int(value) {
                start = offset
            }
            if start > 0 {
                let body = data[start...]
                send(status: 206, headers: ["Content-Range": "bytes \(rangeStart ?? UInt64(start))-\(data.count - 1)/\(data.count)",
                                            "Content-Length": "\(body.count)"], body: Data(body), cutAfter: cutAfter)
            } else {
                send(status: 200, headers: ["Content-Length": "\(data.count)"], body: data, cutAfter: cutAfter)
            }
        }
    }

    private func send(status: Int, headers: [String: String], body: Data, cutAfter: Int? = nil) {
        // A content type, as real model hosts send, so the loading system doesn't hold the response back to sniff it.
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: headers.merging(["Content-Type": "application/octet-stream"]) { a, _ in a })!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        if let cutAfter {
            client?.urlProtocol(self, didLoad: body.prefix(cutAfter))
            client?.urlProtocol(self, didFailWithError: URLError(.networkConnectionLost))
            return
        }
        var position = body.startIndex
        while position < body.endIndex {
            let end = body.index(position, offsetBy: 32, limitedBy: body.endIndex) ?? body.endIndex
            client?.urlProtocol(self, didLoad: body[position..<end])
            position = end
        }
        client?.urlProtocolDidFinishLoading(self)
    }
}

private struct TransportFixture {
    let host = "models-\(UUID().uuidString.lowercased()).example"
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("transport-\(UUID().uuidString)")
    let data = Data((0..<1000).map { UInt8($0 % 253) })
    var destination: URL { directory.appendingPathComponent("file.bin") }
    var url: URL { URL(string: "https://\(host)/kit/file.bin")! }

    init() throws { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }

    var transport: URLSessionTransport {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ScriptedProtocol.self]
        return URLSessionTransport(configuration: configuration)
    }
    func request(headers: [String: String] = [:]) -> TransportRequest {
        TransportRequest(url: url, file: AssetFile(path: "file.bin", byteCount: UInt64(data.count), sha256: String(repeating: "0", count: 64)),
                         headers: headers)
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}

struct URLSessionTransportTests {
    @Test func aWholeFileArrivesWithProgress() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        ScriptedProtocol.script(fixture.host, [.file(fixture.data)])
        let seen = Recorder()
        try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: false) { seen.add($0) }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
        #expect(seen.values.last == UInt64(fixture.data.count) && seen.values == seen.values.sorted())
    }

    @Test func aPartialFileResumesWithARangeRequest() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        try fixture.data.prefix(400).write(to: fixture.destination)
        ScriptedProtocol.script(fixture.host, [.file(fixture.data)])
        try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: true) { _ in }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
        #expect(ScriptedProtocol.requests(fixture.host).first?.headers["Range"] == "bytes=400-")
    }

    @Test func aServerThatIgnoresTheRangeRestartsTheFile() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        try Data(repeating: 7, count: 400).write(to: fixture.destination)
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, honourRange: false)])
        try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: true) { _ in }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
    }

    @Test func aMismatchedContentRangeIsRefused() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        try fixture.data.prefix(400).write(to: fixture.destination)
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, contentRangeStart: 300)])
        await #expect(throws: PreparationError.self) {
            try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: true) { _ in }
        }
    }

    @Test func anUnsatisfiableRangeRestartsOnce() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        try fixture.data.prefix(400).write(to: fixture.destination)
        ScriptedProtocol.script(fixture.host, [.status(416), .file(fixture.data)])
        try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: true) { _ in }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
        #expect(ScriptedProtocol.requests(fixture.host).last?.headers["Range"] == nil)
    }

    @Test func httpErrorsCarryTheirStatus() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        ScriptedProtocol.script(fixture.host, [.status(404)])
        await #expect(throws: PreparationError.http(status: 404, path: "file.bin")) {
            try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: false) { _ in }
        }
    }

    @Test func aCutConnectionKeepsTheBytesThatArrived() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, cutAfter: 300)])
        await #expect(throws: PreparationError.self) {
            try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: false) { _ in }
        }
        // Whatever arrived before the cut is kept, and is a true prefix of the file (the loading system may hold back
        // the last chunk it had when the connection failed).
        let kept = try Data(contentsOf: fixture.destination)
        #expect(kept.count <= 300 && kept == fixture.data.prefix(kept.count))
    }

    @Test func redirectsStayOnHTTPSAndDropHeadersAtANewHost() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        let cdnHost = "cdn-\(UUID().uuidString.lowercased()).example"
        ScriptedProtocol.script(fixture.host, [.redirect(to: URL(string: "https://\(cdnHost)/signed/file.bin")!)])
        ScriptedProtocol.script(cdnHost, [.file(fixture.data)])
        try await fixture.transport.fetch(fixture.request(headers: ["Authorization": "Bearer secret"]),
                                          to: fixture.destination, resume: false) { _ in }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
        #expect(ScriptedProtocol.requests(fixture.host).first?.headers["Authorization"] == "Bearer secret")
        #expect(ScriptedProtocol.requests(cdnHost).first?.headers["Authorization"] == nil)

        let other = try TransportFixture()
        defer { other.remove() }
        ScriptedProtocol.script(other.host, [.redirect(to: URL(string: "http://\(cdnHost)/plain/file.bin")!)])
        await #expect(throws: PreparationError.self) {
            try await other.transport.fetch(other.request(), to: other.destination, resume: false) { _ in }
        }
    }

    @Test func plainHTTPIsRefusedBeforeConnecting() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        let request = TransportRequest(url: URL(string: "http://\(fixture.host)/file.bin")!, file: fixture.request().file)
        await #expect(throws: PreparationError.self) {
            try await fixture.transport.fetch(request, to: fixture.destination, resume: false) { _ in }
        }
        #expect(ScriptedProtocol.requests(fixture.host).isEmpty)
    }

    @Test func preparationOverHTTPSResumesAcrossACutAndInstalls() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        let file = AssetFile(path: "file.bin", byteCount: UInt64(fixture.data.count), sha256: sha256Hex(fixture.data))
        let entry = try ModelCatalogEntry(
            manifest: AssetManifest(assetIdentity: "example/1:https", executionContract: "example.bundle/1", files: [file]),
            source: .https(base: URL(string: "https://\(fixture.host)/kit")!))
        let preparation = ModelPreparation(store: try LocalAssetStore(root: fixture.directory.appendingPathComponent("store")),
                                           transport: fixture.transport, headers: { _ in [:] },
                                           availableCapacity: { _ in .max }, progressInterval: .zero)
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, cutAfter: 500), .file(fixture.data)])
        var first: PreparationEvent?
        for await event in preparation.prepare(entry) where event.isTerminal { first = event }
        guard case .failed? = first else { Issue.record("expected the cut to fail the first run, got \(String(describing: first))"); return }
        let id = try await preparation.install(entry)
        let lease = try await preparation.store.acquire(id)
        #expect(try Data(contentsOf: lease.directory.appendingPathComponent("file.bin")) == fixture.data)
        let ranges = ScriptedProtocol.requests(fixture.host).map { $0.headers["Range"] }
        #expect(ranges.count == 2 && ranges[0] == nil)
    }

    @Test func aBodyLargerThanThePinnedSizeIsStopped() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        // A 200 whose Content-Length disagrees with the entry is refused before anything is written.
        ScriptedProtocol.script(fixture.host, [.file(fixture.data + Data(repeating: 1, count: 500))])
        await #expect(throws: PreparationError.integrityMismatch(path: "file.bin")) {
            try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: false) { _ in }
        }
    }

    @Test func aPartialResponseFromTheStartRestartsTheFile() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        try Data(repeating: 9, count: 400).write(to: fixture.destination)
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, honourRange: false, partialFromStart: true)])
        try await fixture.transport.fetch(fixture.request(), to: fixture.destination, resume: true) { _ in }
        #expect(try Data(contentsOf: fixture.destination) == fixture.data)
    }

    private var resolverConfiguration: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ScriptedProtocol.self]
        return configuration
    }

    @Test func resolvingStopsAtTheFirstOtherHostAndDropsHeadersThere() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        let cdn = URL(string: "https://cdn-\(UUID().uuidString.lowercased()).example/signed/file.bin")!
        let sameHostHop = URL(string: "https://\(fixture.host)/kit/moved.bin")!
        ScriptedProtocol.script(fixture.host, [.redirect(to: sameHostHop), .redirect(to: cdn)])
        let resolved = try await RedirectResolver.resolve(fixture.request(headers: ["Authorization": "Bearer secret"]),
                                                          configuration: resolverConfiguration)
        #expect(resolved.url == cdn && resolved.headers.isEmpty)
        #expect(ScriptedProtocol.requests(fixture.host).allSatisfy {
            $0.headers["Authorization"] == "Bearer secret" && $0.method == "GET" && $0.headers["Range"] == "bytes=0-0"
        })
        #expect(ScriptedProtocol.requests(cdn.host!).isEmpty)
    }

    @Test func resolvingWithoutARedirectKeepsTheHeaders() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        ScriptedProtocol.script(fixture.host, [.status(200)])
        let same = try await RedirectResolver.resolve(fixture.request(headers: ["Authorization": "Bearer secret"]),
                                                      configuration: resolverConfiguration)
        #expect(same.url == fixture.url && same.headers == ["Authorization": "Bearer secret"])
    }

    @Test func resolvingRefusesPlainHTTPAndReportsErrors() async throws {
        let plain = try TransportFixture()
        defer { plain.remove() }
        ScriptedProtocol.script(plain.host, [.redirect(to: URL(string: "http://\(plain.host)/x")!)])
        await #expect(throws: PreparationError.self) {
            _ = try await RedirectResolver.resolve(plain.request(headers: ["Authorization": "x"]), configuration: resolverConfiguration)
        }
        let missing = try TransportFixture()
        defer { missing.remove() }
        ScriptedProtocol.script(missing.host, [.status(404)])
        await #expect(throws: PreparationError.http(status: 404, path: "file.bin")) {
            _ = try await RedirectResolver.resolve(missing.request(headers: ["Authorization": "x"]), configuration: resolverConfiguration)
        }
    }

    @Test func resolvingStopsAtTheHeadersOfAServerThatIgnoresTheRange() async throws {
        let fixture = try TransportFixture()
        defer { fixture.remove() }
        ScriptedProtocol.script(fixture.host, [.file(fixture.data, honourRange: false)])
        let resolved = try await RedirectResolver.resolve(fixture.request(headers: ["Authorization": "Bearer secret"]),
                                                          configuration: resolverConfiguration)
        #expect(resolved.url == fixture.url && resolved.headers == ["Authorization": "Bearer secret"])
    }

    @Test func resumeDataIsNotKeptForATaskCarryingAppHeaders() {
        var request = URLRequest(url: URL(string: "https://example.com/f")!)
        request.setValue("bytes=5-", forHTTPHeaderField: "Range")
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let plain = URLSession.shared.downloadTask(with: request)
        request.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        let authorized = URLSession.shared.downloadTask(with: request)
        defer { plain.cancel(); authorized.cancel() }
        #expect(!BackgroundCoordinator.carriesAppHeaders(plain))
        #expect(BackgroundCoordinator.carriesAppHeaders(authorized))
    }

    @Test func resolvingWithoutHeadersMakesNoRequest() async throws {
        let open = try TransportFixture()
        defer { open.remove() }
        #expect(try await RedirectResolver.resolve(open.request(), configuration: resolverConfiguration).url == open.url)
        #expect(ScriptedProtocol.requests(open.host).isEmpty)
    }

    @Test func contentRangeParsing() {
        let range = URLSessionTransport.contentRange("bytes 128-299/300")
        #expect(range?.start == 128 && range?.end == 299 && range?.total == 300)
        #expect(URLSessionTransport.contentRange("bytes 5-9/*")!.total == nil)
        #expect(URLSessionTransport.contentRange("items 1-2/3") == nil)
        #expect(URLSessionTransport.contentRange("bytes x-2/3") == nil)
        #expect(URLSessionTransport.contentRange("bytes 5-/9") == nil)
        #expect(URLSessionTransport.contentRange("bytes 9-5/10") == nil)
    }
}

private func sha256Hex(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var list: [UInt64] = []
    func add(_ value: UInt64) { lock.withLock { list.append(value) } }
    var values: [UInt64] { lock.withLock { list } }
}
