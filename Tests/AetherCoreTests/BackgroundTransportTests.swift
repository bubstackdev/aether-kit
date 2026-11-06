import Foundation
import Testing
@testable import AetherCore

// Ported from AetherKit 0.0.7's BackgroundTransportTests and the redirect-resolver cases of its URLSessionTransport
// tests. A real background session needs an app (and, for suspension, a device); the end-to-end cases here run the
// same coordinator on an ephemeral session with the stub protocol, through ModelStore.

@Suite("BackgroundTransport")
struct BackgroundTransportTests {
    private let data = Data((0..<600).map { UInt8($0 % 241) })

    private func response(_ status: Int, _ headers: [String: String] = [:]) -> HTTPURLResponse {
        HTTPURLResponse(url: URL(string: "https://example.com/f")!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
    }

    private func downloaded(_ bytes: Data, in directory: URL) throws -> URL {
        let url = directory.appendingPathComponent("tmp-\(UUID().uuidString)")
        try bytes.write(to: url)
        return url
    }

    static var stubConfiguration: URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return configuration
    }

    static func transport() -> BackgroundTransport {
        BackgroundTransport(identifier: "test-\(UUID())", configuration: stubConfiguration, resolver: stubConfiguration)
    }

    // MARK: What a finished task does with its bytes

    @Test func recordsSurviveTheirTaskDescriptionAndStayHomeRelative() {
        let home = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Store/.staging/x/f.bin").path
        let record = BackgroundTaskRecord(destination: BackgroundTaskRecord.portablePath(home), offset: 5, size: 9, file: "f.bin")
        #expect(record.destination == "Library/Store/.staging/x/f.bin")
        #expect(record.destinationURL.path == URL(fileURLWithPath: home).standardizedFileURL.path)
        #expect(BackgroundTaskRecord(encoded: record.encoded) == record)
        #expect(BackgroundTaskRecord(encoded: "not json") == nil)
        #expect(BackgroundTaskRecord.portablePath("/elsewhere/f.bin") == "/elsewhere/f.bin")
    }

    @Test func aMatchingPartialResponseIsAppended() async throws {
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("f.bin")
            try data.prefix(200).write(to: target)
            let record = BackgroundTaskRecord(destination: target.path, offset: 200, size: 600, file: "f.bin")
            try record.apply(downloadedFile: downloaded(data.suffix(400), in: directory), response: response(206, ["Content-Range": "bytes 200-599/600"]))
            #expect(try Data(contentsOf: target) == data)
        }
    }

    @Test func aTaskResumedFromResumeDataIsAppendedFromTheRecordOffset() async throws {
        // Stopped at byte 450 and resumed: the system's request starts there, but its file holds bytes 200 through 599.
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("f.bin")
            try data.prefix(200).write(to: target)
            let record = BackgroundTaskRecord(destination: target.path, offset: 200, size: 600, file: "f.bin")
            let resumed = response(206, ["Content-Range": "bytes 450-599/600"])
            #expect(record.base(of: resumed, expectedBytes: 400) == 200)
            #expect(record.base(of: resumed, expectedBytes: -1) == 200)
            // The first response sent the whole file; resumed, the server honoured the range.
            #expect(record.base(of: resumed, expectedBytes: 600) == 0)
            // A compressed body's expected count is its encoded size: read the response instead.
            let gzipped = response(206, ["Content-Range": "bytes 450-599/600", "Content-Encoding": "gzip"])
            #expect(record.base(of: gzipped, expectedBytes: 90) == 200)
            try record.apply(downloadedFile: downloaded(data.suffix(400), in: directory), response: resumed)
            #expect(try Data(contentsOf: target) == data)
        }
    }

    @Test func aPartialResponseFromTheStartReplacesTheFile() async throws {
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("f.bin")
            try Data(repeating: 9, count: 200).write(to: target)
            let record = BackgroundTaskRecord(destination: target.path, offset: 200, size: 600, file: "f.bin")
            let fromStart = response(206, ["Content-Range": "bytes 0-599/600"])
            #expect(record.base(of: fromStart, expectedBytes: -1) == 0)
            #expect(record.base(of: response(200), expectedBytes: -1) == 0)
            #expect(record.base(of: response(200), expectedBytes: 600) == 0)
            try record.apply(downloadedFile: downloaded(data, in: directory), response: fromStart)
            #expect(try Data(contentsOf: target) == data)
        }
    }

    @Test func aFullResponseReplacesTheFile() async throws {
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("sub/f.bin")
            let record = BackgroundTaskRecord(destination: target.path, offset: 0, size: 600, file: "f.bin")
            try record.apply(downloadedFile: downloaded(data, in: directory), response: response(200))
            #expect(try Data(contentsOf: target) == data)
        }
    }

    @Test func aMismatchedOrFailedResponseLeavesTheFileAlone() async throws {
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("f.bin")
            try data.prefix(200).write(to: target)
            let record = BackgroundTaskRecord(destination: target.path, offset: 200, size: 600, file: "f.bin")
            #expect(throws: AetherError.self) {
                try record.apply(downloadedFile: downloaded(data.suffix(300), in: directory),
                                 response: response(206, ["Content-Range": "bytes 300-599/600"]))
            }
            #expect {
                try record.apply(downloadedFile: downloaded(Data(), in: directory), response: response(503))
            } throws: { error in
                if case AetherError.download("f.bin", .http(503)) = error { return true }
                return false
            }
            #expect(try Data(contentsOf: target) == data.prefix(200))
        }
    }

    @Test func aDownloadLargerThanThePinnedRemainderIsRefused() async throws {
        try await withTemporaryDirectory { directory in
            let target = directory.appendingPathComponent("f.bin")
            let record = BackgroundTaskRecord(destination: target.path, offset: 0, size: 600, file: "f.bin")
            #expect {
                try record.apply(downloadedFile: downloaded(data + Data([1]), in: directory), response: response(200))
            } throws: { error in
                if case AetherError.download("f.bin", .unexpectedContent) = error { return true }
                return false
            }
            #expect(!FileManager.default.fileExists(atPath: target.path))
        }
    }

    @Test func resumeDataLivesBesideTheFileKeyedByOffset() {
        let record = BackgroundTaskRecord(destination: "/store/.staging/x/model/main.mlirb", offset: 128, size: 9, file: "p")
        #expect(record.resumeDataURL.path == "/store/.staging/x/model/.main.mlirb.resume-128")
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

    // MARK: Foreground redirect resolution

    private func request(_ site: StubSite, _ path: String, headers: [String: String]) -> DownloadRequest {
        DownloadRequest(url: site.endpoint.appending(path: path), headers: headers, destination: URL(fileURLWithPath: "/dev/null"),
                        offset: 0, size: 1000, file: "file.bin")
    }

    @Test func resolvingStopsAtTheFirstOtherHostAndDropsHeadersThere() async throws {
        let site = StubSite(), cdn = StubSite()
        let signed = cdn.endpoint.appending(path: "signed/file.bin")
        site.behave("/kit/file.bin", .init(redirect: site.endpoint.appending(path: "kit/moved.bin")))
        site.behave("/kit/moved.bin", .init(redirect: signed))
        let resolved = try await RedirectResolver.resolve(request(site, "kit/file.bin", headers: ["Authorization": "Bearer secret"]),
                                                          configuration: Self.stubConfiguration)
        #expect(resolved.url == signed && resolved.headers.isEmpty)
        #expect(site.requests.count == 2)
        #expect(site.requests.allSatisfy { $0.authorization == "Bearer secret" && $0.method == "GET" && $0.range == "bytes=0-0" })
        #expect(cdn.requests.isEmpty)
    }

    @Test func resolvingWithoutARedirectKeepsTheHeaders() async throws {
        let site = StubSite()
        site.serve("/kit/file.bin", data)
        let same = try await RedirectResolver.resolve(request(site, "kit/file.bin", headers: ["Authorization": "Bearer secret"]),
                                                      configuration: Self.stubConfiguration)
        #expect(same.url == site.endpoint.appending(path: "kit/file.bin") && same.headers == ["Authorization": "Bearer secret"])
    }

    @Test func resolvingStopsAtTheHeadersOfAServerThatIgnoresTheRange() async throws {
        let site = StubSite()
        site.serve("/kit/file.bin", data)
        site.behave("/kit/file.bin", .init(ignoreRange: true))
        let resolved = try await RedirectResolver.resolve(request(site, "kit/file.bin", headers: ["Authorization": "Bearer secret"]),
                                                          configuration: Self.stubConfiguration)
        #expect(resolved.headers == ["Authorization": "Bearer secret"])
    }

    @Test func resolvingRefusesPlainHTTPAndReportsErrors() async throws {
        let plain = StubSite()
        plain.behave("/x", .init(redirect: URL(string: "http://\(plain.host)/y")!))
        await #expect {
            _ = try await RedirectResolver.resolve(request(plain, "x", headers: ["Authorization": "x"]), configuration: Self.stubConfiguration)
        } throws: { error in
            if case AetherError.download("file.bin", .transport) = error { return true }
            return false
        }
        let missing = StubSite()
        await #expect {
            _ = try await RedirectResolver.resolve(request(missing, "x", headers: ["Authorization": "x"]), configuration: Self.stubConfiguration)
        } throws: { error in
            if case AetherError.download("file.bin", .http(404)) = error { return true }
            return false
        }
    }

    @Test func resolvingWithoutHeadersMakesNoRequest() async throws {
        let site = StubSite()
        let resolved = try await RedirectResolver.resolve(request(site, "kit/file.bin", headers: [:]), configuration: Self.stubConfiguration)
        #expect(resolved.url == site.endpoint.appending(path: "kit/file.bin"))
        #expect(site.requests.isEmpty)
    }

    // MARK: Through the store (added: the transport behind the HTTPTransport seam)

    @Test func theStoreInstallsThroughDownloadTasksAndResumes() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = ModelStoreTests.Bundle()
            bundle.serve(on: site)
            let store = ModelStore(root: root, transport: Self.transport(), endpoint: site.endpoint)
            let staged = store.stagingURL(for: bundle.ref).appending(path: ModelStoreTests.Bundle.weights)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bundle.contents[ModelStoreTests.Bundle.weights]!.prefix(70_000).write(to: staged)

            let installed = try await store.install(bundle.ref)
            try ModelStoreTests.expectInstalled(installed, bundle)
            #expect(site.requests.filter { $0.path == bundle.remotePath(ModelStoreTests.Bundle.weights) }.map(\.range) == ["bytes=70000-"])
        }
    }

    @Test func theStoresTokenNeverReachesTheCDN() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite(), cdn = StubSite()
            let bundle = ModelStoreTests.Bundle()
            bundle.serve(on: site)
            let weights = ModelStoreTests.Bundle.weights
            cdn.serve("/signed/weights", bundle.contents[weights]!)
            site.behave(bundle.remotePath(weights), .init(redirect: cdn.endpoint.appending(path: "signed/weights")))
            let host = site.host
            let store = ModelStore(root: root, transport: Self.transport(), endpoint: site.endpoint,
                                   headers: { $0.host == host ? ["Authorization": "Bearer secret"] : [:] })
            try ModelStoreTests.expectInstalled(try await store.install(bundle.ref), bundle)
            #expect(site.requests.allSatisfy { $0.authorization == "Bearer secret" })
            #expect(!cdn.requests.isEmpty && cdn.requests.allSatisfy { $0.authorization == nil })
        }
    }
}
