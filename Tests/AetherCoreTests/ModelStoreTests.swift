import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherCore

@Suite("ModelStore")
struct ModelStoreTests {
    /// A tiny two-asset install: a main asset with a multi-megabyte "weights" file (several progress steps) and a
    /// small nested file, plus a tokenizer asset from a foreign repo at its own pin.
    struct Bundle {
        let ref: ModelRef
        /// Local path relative to the revision root → contents.
        let contents: [String: Data]
        /// Local path → remote URL path.
        let remote: [String: String]

        static let foreignRepo = "vendor/tokenizer-repo"
        static let foreignRevision = String(repeating: "f", count: 40)

        init(revision: String = String(repeating: "1", count: 40)) {
            var generator = SplitMix(seed: 42)
            let weights = Data((0..<(2_500_000)).map { _ in UInt8(truncatingIfNeeded: generator.next()) })
            let main: [String: Data] = [
                "toy.aimodel/main.mlirb": weights,
                "metadata.json": Data(#"{"kind":"llm"}"#.utf8),
            ]
            let tokenizer: [String: Data] = ["tokenizer.json": Data(#"{"a":1}"#.utf8)]
            func files(_ map: [String: Data]) -> [CatalogFile] {
                map.keys.sorted().map { CatalogFile(path: $0, size: Int64(map[$0]!.count), sha256: Self.sha256(map[$0]!)) }
            }
            ref = ModelRef(
                catalogID: "toy", repo: "owner/toy-model", revision: revision, variant: "macos", driver: .decoderLM,
                assets: [
                    .init(name: "main", repo: "owner/toy-model", revision: revision, subtree: "macos/toy", files: files(main)),
                    .init(name: "tokenizer", repo: Self.foreignRepo, revision: Self.foreignRevision, subtree: "", files: files(tokenizer)),
                ])
            var contents: [String: Data] = [:]
            var remote: [String: String] = [:]
            for (path, data) in main {
                contents["macos/toy/\(path)"] = data
                remote["macos/toy/\(path)"] = "/owner/toy-model/resolve/\(revision)/macos/toy/\(path)"
            }
            for (path, data) in tokenizer {
                contents[".aether/assets/tokenizer/\(path)"] = data
                remote[".aether/assets/tokenizer/\(path)"] = "/\(Self.foreignRepo)/resolve/\(Self.foreignRevision)/\(path)"
            }
            self.contents = contents
            self.remote = remote
        }

        static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

        static let weights = "macos/toy/toy.aimodel/main.mlirb"

        func remotePath(_ local: String) -> String { remote[local]! }

        func serve(on site: StubSite) {
            for (local, data) in contents { site.serve(remote[local]!, data) }
        }
    }

    static func transport() -> URLSessionTransport {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        return URLSessionTransport(session: URLSession(configuration: configuration))
    }

    static func store(_ root: URL, _ site: StubSite, gate: Gate? = nil) -> ModelStore {
        let transport: any HTTPTransport = gate.map { GatedTransport(inner: Self.transport(), gate: $0) } ?? Self.transport()
        return ModelStore(root: root, transport: transport, endpoint: site.endpoint)
    }

    static func expectInstalled(_ installed: InstalledBundle, _ bundle: Bundle) throws {
        for (path, data) in bundle.contents {
            #expect(try Data(contentsOf: installed.url.appending(path: path)) == data, "\(path)")
        }
        #expect(installed.url(ofAsset: "tokenizer") == installed.url.appending(path: ".aether/assets/tokenizer"))
        #expect(installed.url(of: AssetPath(asset: "main", path: "metadata.json")) == installed.url.appending(path: "macos/toy/metadata.json"))
        let excluded = try installed.url.resourceValues(forKeys: [.isExcludedFromBackupKey]).isExcludedFromBackup
        #expect(excluded == true)
    }

    @Test func installsVerifiesAndPlacesAtomically() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            let updates = Mutex<[InstallProgress]>([])
            let installed = try await store.install(bundle.ref) { p in updates.withLock { $0.append(p) } }

            #expect(installed.url == root.appending(path: "owner/toy-model/\(bundle.ref.revision)"))
            #expect(!installed.isStale)
            try Self.expectInstalled(installed, bundle)
            #expect(FileManager.default.fileExists(atPath: installed.url.appending(path: ".aether/receipt.json").path))
            #expect(!FileManager.default.fileExists(atPath: store.stagingURL(for: bundle.ref).path), "staging was renamed away")

            let progress = updates.withLock { $0 }
            #expect(progress.count >= 3)
            #expect(progress.last?.completedBytes == bundle.ref.bytes)
            #expect(zip(progress, progress.dropFirst()).allSatisfy { $0.completedBytes <= $1.completedBytes })

            // Installed: a second call touches no network.
            let before = site.requests.count
            let again = try await store.install(bundle.ref)
            #expect(again == installed)
            #expect(site.requests.count == before)
        }
    }

    @Test func resumesAnInterruptedDownloadWithRange() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let weights = bundle.remotePath(Bundle.weights)
            site.behave(weights, .init(failAfter: (bytes: 1_000_000, times: 1)))
            let store = Self.store(root, site)

            await #expect {
                _ = try await store.install(bundle.ref)
            } throws: { error in
                if case AetherError.download(let file, .transport) = error { return file == Bundle.weights }
                return false
            }
            let partial = store.stagingURL(for: bundle.ref).appending(path: Bundle.weights)
            let kept = try #require(try FileManager.default.attributesOfItem(atPath: partial.path)[.size] as? Int)
            #expect(kept > 0 && kept <= 1_000_000)
            #expect(!FileManager.default.fileExists(atPath: store.bundleURL(for: bundle.ref).path))

            let installed = try await store.install(bundle.ref)
            try Self.expectInstalled(installed, bundle)
            let weightRequests = site.requests.filter { $0.path == weights }
            #expect(weightRequests.count == 2)
            #expect(weightRequests.first?.range == nil)
            #expect(weightRequests.last?.range == "bytes=\(kept)-")
        }
    }

    @Test func rejectsAHashMismatchAndInstallsNothing() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let path = ".aether/assets/tokenizer/tokenizer.json"
            site.behave(bundle.remotePath(path), .init(corrupt: Data(#"{"b":2}"#.utf8)))
            let store = Self.store(root, site)

            await #expect {
                _ = try await store.install(bundle.ref)
            } throws: { error in
                if case AetherError.integrity(let file, let expected, let actual) = error {
                    return file == path && expected == Bundle.sha256(bundle.contents[path]!) && actual != expected
                }
                return false
            }
            #expect(!FileManager.default.fileExists(atPath: store.bundleURL(for: bundle.ref).path))
            #expect(!FileManager.default.fileExists(atPath: store.stagingURL(for: bundle.ref).appending(path: path).path),
                    "the rejected file is not kept")
        }
    }

    @Test func refetchesAResumedFileWhosePrefixWasCorrupt() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            // A partial file from an earlier session, with bad bytes in it.
            let path = Bundle.weights
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 0xEE, count: 4096).write(to: staged)

            let installed = try await store.install(bundle.ref)
            try Self.expectInstalled(installed, bundle)
            let requests = site.requests.filter { $0.path == bundle.remotePath(path) }
            #expect(requests.map(\.range) == ["bytes=4096-", nil])
        }
    }

    @Test func serverIgnoringRangeRestartsTheFile() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let path = Bundle.weights
            site.behave(bundle.remotePath(path), .init(ignoreRange: true))
            let store = Self.store(root, site)
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bundle.contents[path]!.prefix(5000).write(to: staged)

            let installed = try await store.install(bundle.ref)
            try Self.expectInstalled(installed, bundle)
        }
    }

    @Test func concurrentCallersJoinOneDownloadAndAllSeeProgress() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let gate = Gate()
            let store = Self.store(root, site, gate: gate)

            let first = await store.installation(bundle.ref)
            let second = await store.installation(bundle.ref)
            async let a = Self.drain(first)
            async let b = Self.drain(second)
            // Both callers are attached before any byte moves.
            while await gate.waitingCount == 0 { try await Task.sleep(for: .milliseconds(5)) }
            await gate.open()

            let (resultA, resultB) = try await (a, b)
            #expect(resultA.bundle == resultB.bundle)
            try Self.expectInstalled(resultA.bundle, bundle)
            #expect(resultA.progress.count >= 2 && resultB.progress.count >= 2)
            #expect(resultA.progress.last?.completedBytes == resultB.progress.last?.completedBytes)
            for local in bundle.contents.keys {
                #expect(site.requests.filter { $0.path == bundle.remotePath(local) }.count == 1, "\(local)")
            }
        }
    }

    static func drain(_ stream: AsyncThrowingStream<InstallEvent, any Error>) async throws -> (bundle: InstalledBundle, progress: [InstallProgress]) {
        var progress: [InstallProgress] = []
        for try await event in stream {
            switch event {
            case .progress(let p): progress.append(p)
            case .finished(let bundle): return (bundle, progress)
            }
        }
        throw CancellationError()
    }

    @Test func staleRevisionIsOptIn() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let old = Bundle(revision: String(repeating: "1", count: 40))
            old.serve(on: site)
            let store = Self.store(root, site)
            let cached = try await store.install(old.ref)

            let new = Bundle(revision: String(repeating: "2", count: 40))
            site.setOffline(true)
            await #expect {
                _ = try await store.install(new.ref)
            } throws: { error in
                if case AetherError.download = error { return true }
                return false
            }

            let fallback = try await store.install(new.ref, options: .allowStaleRevision)
            #expect(fallback.isStale)
            #expect(fallback.url == cached.url)
            #expect(fallback.ref.revision == old.ref.revision)
        }
    }

    @Test func staleFallbackDoesNotMaskIntegrityFailures() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let old = Bundle(revision: String(repeating: "1", count: 40))
            old.serve(on: site)
            let store = Self.store(root, site)
            _ = try await store.install(old.ref)

            let new = Bundle(revision: String(repeating: "3", count: 40))
            new.serve(on: site)
            site.behave(new.remotePath("macos/toy/metadata.json"), .init(corrupt: Data(#"{"kind":"xxx"}"#.utf8)))
            await #expect {
                _ = try await store.install(new.ref, options: .allowStaleRevision)
            } throws: { error in
                if case AetherError.integrity = error { return true }
                return false
            }
        }
    }

    @Test func httpErrorsAreStructured() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            site.behave(bundle.remotePath("macos/toy/metadata.json"), .init(status: 503))
            let store = Self.store(root, site)
            await #expect {
                _ = try await store.install(bundle.ref)
            } throws: { error in
                if case AetherError.download("macos/toy/metadata.json", .http(503)) = error { return true }
                return false
            }
        }
    }

    @Test func unsafeCatalogPathsAreRefused() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let store = Self.store(root, site)
            var ref = Bundle().ref
            ref.assets[0].files = [CatalogFile(path: "../escape", size: 1, sha256: "")]
            await #expect(throws: AetherError.self) { _ = try await store.install(ref) }
        }
    }
}

/// Deterministic bytes for fixtures.
struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

extension ModelStoreTests {
    @Test func leavingCancelsAndALaterCallResumes() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let gate = Gate()
            let store = Self.store(root, site, gate: gate)

            // The only caller walks away before any byte moves: the download is cancelled.
            let abandoned = Task { try await Self.drain(await store.installation(bundle.ref)) }
            while await gate.waitingCount == 0 { try await Task.sleep(for: .milliseconds(5)) }
            abandoned.cancel()
            _ = await abandoned.result

            await gate.open()
            let installed = try await store.install(bundle.ref)
            try Self.expectInstalled(installed, bundle)
        }
    }
}
