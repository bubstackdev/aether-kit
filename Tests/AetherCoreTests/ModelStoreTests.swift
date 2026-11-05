import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherCore

@Suite("ModelStore")
struct ModelStoreTests {
    /// A tiny bundle: one multi-megabyte "weights" file (several progress steps), a small one, and a nested one.
    struct Bundle {
        let ref: ModelRef
        let files: [CatalogFile]
        let contents: [String: Data]

        init(revision: String = String(repeating: "1", count: 40)) {
            ref = ModelRef(catalogID: "toy", repo: "owner/toy-model", revision: revision, subtree: "macos/toy", variant: "macos")
            var generator = SplitMix(seed: 42)
            let weights = Data((0..<(2_500_000)).map { _ in UInt8(truncatingIfNeeded: generator.next()) })
            let contents = [
                "toy.aimodel/main.mlirb": weights,
                "metadata.json": Data(#"{"kind":"llm"}"#.utf8),
                "tokenizer/vocab.json": Data(#"{"a":1}"#.utf8),
            ]
            self.contents = contents
            files = contents.keys.sorted().map { path in
                CatalogFile(path: path, size: Int64(contents[path]!.count), sha256: Self.sha256(contents[path]!))
            }
        }

        static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

        func remotePath(_ file: String) -> String { "/\(ref.repo)/resolve/\(ref.revision)/\(ref.subtree)/\(file)" }

        func serve(on site: StubSite) {
            for (path, data) in contents { site.serve(remotePath(path), data) }
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
            let installed = try await store.install(bundle.ref, files: bundle.files) { p in updates.withLock { $0.append(p) } }

            #expect(installed.url == root.appending(path: "owner/toy-model/\(bundle.ref.revision)/macos/toy"))
            #expect(!installed.isStale)
            try Self.expectInstalled(installed, bundle)
            #expect(FileManager.default.fileExists(atPath: installed.url.appending(path: ".aether-install.json").path))
            #expect(!FileManager.default.fileExists(atPath: store.stagingURL(for: bundle.ref).path), "staging was renamed away")

            let progress = updates.withLock { $0 }
            #expect(progress.count >= 3)
            #expect(progress.last?.completedBytes == bundle.files.reduce(0) { $0 + $1.size })
            #expect(zip(progress, progress.dropFirst()).allSatisfy { $0.completedBytes <= $1.completedBytes })

            // Installed: a second call touches no network.
            let before = site.requests.count
            let again = try await store.install(bundle.ref, files: bundle.files)
            #expect(again == installed)
            #expect(site.requests.count == before)
        }
    }

    @Test func resumesAnInterruptedDownloadWithRange() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let weights = bundle.remotePath("toy.aimodel/main.mlirb")
            site.behave(weights, .init(failAfter: (bytes: 1_000_000, times: 1)))
            let store = Self.store(root, site)

            await #expect {
                _ = try await store.install(bundle.ref, files: bundle.files)
            } throws: { error in
                if case AetherError.download(let file, .transport) = error { return file == "toy.aimodel/main.mlirb" }
                return false
            }
            let partial = store.stagingURL(for: bundle.ref).appending(path: "toy.aimodel/main.mlirb")
            let kept = try #require(try FileManager.default.attributesOfItem(atPath: partial.path)[.size] as? Int)
            #expect(kept > 0 && kept <= 1_000_000)
            #expect(!FileManager.default.fileExists(atPath: store.bundleURL(for: bundle.ref).path))

            let installed = try await store.install(bundle.ref, files: bundle.files)
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
            let path = "tokenizer/vocab.json"
            site.behave(bundle.remotePath(path), .init(corrupt: Data(#"{"b":2}"#.utf8)))
            let store = Self.store(root, site)

            await #expect {
                _ = try await store.install(bundle.ref, files: bundle.files)
            } throws: { error in
                if case AetherError.integrity(let file, let expected, let actual) = error {
                    return file == path && expected == bundle.files.first { $0.path == path }!.sha256 && actual != expected
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
            let path = "toy.aimodel/main.mlirb"
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 0xEE, count: 4096).write(to: staged)

            let installed = try await store.install(bundle.ref, files: bundle.files)
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
            let path = "toy.aimodel/main.mlirb"
            site.behave(bundle.remotePath(path), .init(ignoreRange: true))
            let store = Self.store(root, site)
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bundle.contents[path]!.prefix(5000).write(to: staged)

            let installed = try await store.install(bundle.ref, files: bundle.files)
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

            let first = await store.installation(bundle.ref, files: bundle.files)
            let second = await store.installation(bundle.ref, files: bundle.files)
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
            for file in bundle.files {
                #expect(site.requests.filter { $0.path == bundle.remotePath(file.path) }.count == 1, "\(file.path)")
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
            let cached = try await store.install(old.ref, files: old.files)

            let new = Bundle(revision: String(repeating: "2", count: 40))
            site.setOffline(true)
            await #expect {
                _ = try await store.install(new.ref, files: new.files)
            } throws: { error in
                if case AetherError.download = error { return true }
                return false
            }

            let fallback = try await store.install(new.ref, files: new.files, options: .allowStaleRevision)
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
            _ = try await store.install(old.ref, files: old.files)

            let new = Bundle(revision: String(repeating: "3", count: 40))
            new.serve(on: site)
            site.behave(new.remotePath("metadata.json"), .init(corrupt: Data(#"{"kind":"xxx"}"#.utf8)))
            await #expect {
                _ = try await store.install(new.ref, files: new.files, options: .allowStaleRevision)
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
            site.behave(bundle.remotePath("metadata.json"), .init(status: 503))
            let store = Self.store(root, site)
            await #expect {
                _ = try await store.install(bundle.ref, files: bundle.files)
            } throws: { error in
                if case AetherError.download("metadata.json", .http(503)) = error { return true }
                return false
            }
        }
    }

    @Test func unsafeCatalogPathsAreRefused() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let store = Self.store(root, site)
            let ref = Bundle().ref
            await #expect(throws: AetherError.self) {
                _ = try await store.install(ref, files: [CatalogFile(path: "../escape", size: 1, sha256: "")])
            }
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
            let abandoned = Task { try await Self.drain(await store.installation(bundle.ref, files: bundle.files)) }
            while await gate.waitingCount == 0 { try await Task.sleep(for: .milliseconds(5)) }
            abandoned.cancel()
            _ = await abandoned.result

            await gate.open()
            let installed = try await store.install(bundle.ref, files: bundle.files)
            try Self.expectInstalled(installed, bundle)
        }
    }
}
