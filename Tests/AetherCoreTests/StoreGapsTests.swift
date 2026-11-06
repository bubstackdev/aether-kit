import Foundation
import Synchronization
import Testing
@testable import AetherCore

// The store gaps ported from AetherKit 0.0.7 (LocalAssetStore, URLSessionTransport and ModelPreparation tests), as
// specs on ModelStore: reader leases, the cross-process install lock, abandoned-stage recovery, the free-space check,
// the Hugging Face token's scope, redirects, and the 416 restart.

/// A transport that serves files from memory and records every request; never touches the network.
final class RecordingTransport: HTTPTransport {
    private let files: [String: Data]
    private let seen = Mutex<[(url: URL, headers: [String: String])]>([])

    init(files: [String: Data]) { self.files = files }

    var requests: [(url: URL, headers: [String: String])] { seen.withLock { $0 } }

    func get(_ url: URL, headers: [String: String]) async throws -> (HTTPResponseHead, AsyncThrowingStream<Data, any Error>) {
        seen.withLock { $0.append((url, headers)) }
        let (body, continuation) = AsyncThrowingStream<Data, any Error>.makeStream()
        guard let data = files[url.path] else {
            continuation.finish()
            return (HTTPResponseHead(status: 404, headers: [:]), body)
        }
        continuation.yield(data)
        continuation.finish()
        return (HTTPResponseHead(status: 200, headers: ["Content-Length": "\(data.count)"]), body)
    }
}

private func isInUse(_ error: any Error) -> Bool {
    if case AetherError.inUse = error { return true }
    return false
}

extension ModelStoreTests {
    // MARK: Reader leases

    @Test func readerLeasesAcrossStoresPreventRemoval() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let first = Self.store(root, site)
            let second = Self.store(root, site)
            let installed = try await first.install(bundle.ref)
            do {
                let lease = try second.lease(installed)
                #expect(lease.bundle == installed)
                await #expect { try await first.remove(bundle.ref) } throws: { isInUse($0) }
                // A second reader is allowed while the first lives.
                let peer = try first.lease(installed)
                withExtendedLifetime((lease, peer)) {}
            }
            try await first.remove(bundle.ref)
            #expect(await first.installed(bundle.ref) == nil)
            #expect { try second.lease(installed) } throws: { error in
                if case AetherError.unknownModel = error { return true }
                return false
            }
        }
    }

    @Test func removeRefusesWhileAnInstallRuns() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let gate = Gate()
            let store = Self.store(root, site, gate: gate)
            let install = Task { try await store.install(bundle.ref) }
            while await gate.waitingCount == 0 { try await Task.sleep(for: .milliseconds(5)) }
            await #expect { try await store.remove(bundle.ref) } throws: { isInUse($0) }
            await gate.open()
            try Self.expectInstalled(try await install.value, bundle)
        }
    }

    // MARK: Cross-process install lock

    @Test func anInstallAnotherStoreIsRunningIsWaitedForAndJoinedOnDisk() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let gate = Gate()
            // Two stores on one root stand in for two processes: they can't join in memory, only through the lock.
            let first = Self.store(root, site, gate: gate)
            let second = Self.store(root, site, gate: gate)
            let a = Task { try await first.install(bundle.ref) }
            while await gate.waitingCount == 0 { try await Task.sleep(for: .milliseconds(5)) }
            let b = Task { try await second.install(bundle.ref) }
            try await Task.sleep(for: .milliseconds(300))
            #expect(await gate.waitingCount == 1, "the second store waits for the lock, not the network")
            await gate.open()
            let (one, two) = try await (a.value, b.value)
            #expect(one == two)
            try Self.expectInstalled(two, bundle)
            for local in bundle.contents.keys {
                #expect(site.requests.filter { $0.path == bundle.remotePath(local) }.count == 1, "\(local)")
            }
        }
    }

    @Test func aLockHeldElsewhereHoldsTheInstallBack() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            // As another process would hold it: flock is per open file, so a second descriptor conflicts.
            var held: StoreLock? = try StoreLock.attempt(StoreLayout(root: root).lock(for: bundle.ref), exclusive: true)
            #expect(held != nil)
            let install = Task { try await store.install(bundle.ref) }
            try await Task.sleep(for: .milliseconds(300))
            #expect(site.requests.isEmpty)
            held = nil
            try Self.expectInstalled(try await install.value, bundle)
        }
    }

    // MARK: Abandoned-stage recovery

    @Test func recoveryClearsTrashSupersededStagesAndReceiptlessRevisionsOnly() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            let installed = try await store.install(bundle.ref)
            let fm = FileManager.default
            let layout = StoreLayout(root: root)

            // Staged bytes for a bundle that is now installed: nothing can resume into them.
            let superseded = layout.staging(for: bundle.ref).appending(path: "macos/toy")
            try fm.createDirectory(at: superseded, withIntermediateDirectories: true)
            try Data("partial".utf8).write(to: superseded.appending(path: "metadata.json"))
            // An unfinished install of another revision: kept for resume.
            let unfinished = Bundle(revision: String(repeating: "4", count: 40))
            let resumable = layout.staging(for: unfinished.ref)
            try fm.createDirectory(at: resumable, withIntermediateDirectories: true)
            // A half-deleted bundle in the trash, and a revision directory that never got its receipt.
            try fm.createDirectory(at: layout.trash.appending(path: UUID().uuidString), withIntermediateDirectories: true)
            let receiptless = layout.bundle(repo: "owner/toy-model", revision: String(repeating: "5", count: 40))
            try fm.createDirectory(at: receiptless.appending(path: "macos"), withIntermediateDirectories: true)
            // Entries the store didn't create.
            let unknown = root.appending(path: ".staging/not-a-stage")
            try fm.createDirectory(at: unknown, withIntermediateDirectories: true)
            let notRevision = root.appending(path: "owner/toy-model/main")
            try fm.createDirectory(at: notRevision, withIntermediateDirectories: true)

            #expect(await store.recoverAbandonedStages() == 3)
            #expect(!fm.fileExists(atPath: layout.staging(for: bundle.ref).path))
            #expect(!fm.fileExists(atPath: receiptless.path))
            #expect(try fm.contentsOfDirectory(atPath: layout.trash.path).isEmpty)
            #expect(fm.fileExists(atPath: resumable.path) && fm.fileExists(atPath: unknown.path) && fm.fileExists(atPath: notRevision.path))
            #expect(await store.recoverAbandonedStages() == 0)
            try Self.expectInstalled(installed, bundle)
        }
    }

    @Test func recoveryDefersToHoldersAndRetriesAfterRelease() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            let installed = try await store.install(bundle.ref)
            let stage = StoreLayout(root: root).staging(for: bundle.ref)
            try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: true)
            do {
                let reader = try store.lease(installed)
                let removed = await store.recoverAbandonedStages()
                withExtendedLifetime(reader) {
                    #expect(removed == 0)
                    #expect(FileManager.default.fileExists(atPath: stage.path))
                }
            }
            #expect(await store.recoverAbandonedStages() == 1)
        }
    }

    @Test func recoveryNeverFollowsASymlink() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            _ = try await store.install(bundle.ref)
            let outside = root.deletingLastPathComponent().appending(path: "outside-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: outside) }
            try Data("keep".utf8).write(to: outside.appending(path: "file"))
            // Planted where a superseded stage would be.
            try FileManager.default.createSymbolicLink(at: StoreLayout(root: root).staging(for: bundle.ref), withDestinationURL: outside)
            #expect(await store.recoverAbandonedStages() == 0)
            #expect(try Data(contentsOf: outside.appending(path: "file")) == Data("keep".utf8))
        }
    }

    @Test func recoveryRunsBeforeTheFirstInstall() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let trash = StoreLayout(root: root).trash
            try FileManager.default.createDirectory(at: trash.appending(path: UUID().uuidString), withIntermediateDirectories: true)
            _ = try await Self.store(root, site).install(bundle.ref)
            #expect(try FileManager.default.contentsOfDirectory(atPath: trash.path).isEmpty)
        }
    }

    // MARK: Free space

    @Test func notEnoughSpaceRefusesBeforeDownloading() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let tight = ModelStore(root: root, transport: Self.transport(), endpoint: site.endpoint, diskSpace: { _ in 1000 })
            await #expect {
                _ = try await tight.install(bundle.ref)
            } throws: { error in
                if case AetherError.capacity("disk space", bundle.ref.bytes, 1000) = error { return true }
                return false
            }
            #expect(site.requests.isEmpty)

            // Only the bytes still missing count: with all but 500 staged, 1000 free bytes are enough.
            let staged = StoreLayout(root: root).staging(for: bundle.ref)
            for (path, data) in bundle.contents {
                let url = staged.appending(path: path)
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try (path == Bundle.weights ? data.dropLast(500) : data).write(to: url)
            }
            try Self.expectInstalled(try await tight.install(bundle.ref), bundle)
        }
    }

    // MARK: Hugging Face token and redirects

    @Test func huggingFaceAuthorizationOnlyGoesToTheHub() {
        let headers = ModelStore.huggingFaceAuthorization(token: "hf_x")
        #expect(headers(URL(string: "https://huggingface.co/org/m/resolve/abc/f")!) == ["Authorization": "Bearer hf_x"])
        #expect(headers(URL(string: "https://cdn-lfs.huggingface.co/f")!).isEmpty)
        #expect(headers(URL(string: "https://cdn.example.com/f")!).isEmpty)
    }

    @Test func theTokenReachesEveryHubRequestAndNoOtherEndpoint() async throws {
        try await withTemporaryDirectory { root in
            let bundle = Bundle()
            let files = Dictionary(uniqueKeysWithValues: bundle.contents.map { (bundle.remotePath($0.key), $0.value) })
            let hub = RecordingTransport(files: files)
            let store = ModelStore(root: root.appending(path: "hub"), transport: hub, endpoint: ModelStore.huggingFace,
                                   headers: ModelStore.huggingFaceAuthorization(token: "hf_x"))
            try Self.expectInstalled(try await store.install(bundle.ref), bundle)
            #expect(hub.requests.count == bundle.contents.count)
            #expect(hub.requests.allSatisfy { $0.url.host == "huggingface.co" && $0.headers["Authorization"] == "Bearer hf_x" })

            let mirror = RecordingTransport(files: files)
            let other = ModelStore(root: root.appending(path: "mirror"), transport: mirror,
                                   endpoint: URL(string: "https://mirror.example")!,
                                   headers: ModelStore.huggingFaceAuthorization(token: "hf_x"))
            _ = try await other.install(bundle.ref)
            #expect(mirror.requests.allSatisfy { $0.headers["Authorization"] == nil })
        }
    }

    @Test func redirectsDropTheAppsHeadersAtANewHostAndNeverLeaveHTTPS() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite(), cdn = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let weights = bundle.remotePath(Bundle.weights)
            cdn.serve("/signed/weights", bundle.contents[Bundle.weights]!)
            site.behave(weights, .init(redirect: URL(string: "https://\(cdn.host)/signed/weights")!))
            let host = site.host
            let token: @Sendable (URL) -> [String: String] = { $0.host == host ? ["Authorization": "Bearer secret"] : [:] }
            let store = ModelStore(root: root, transport: Self.transport(), endpoint: site.endpoint, headers: token)
            try Self.expectInstalled(try await store.install(bundle.ref), bundle)
            #expect(site.requests.allSatisfy { $0.authorization == "Bearer secret" })
            #expect(cdn.requests.count == 1 && cdn.requests[0].authorization == nil)

            let plain = Bundle(revision: String(repeating: "6", count: 40))
            plain.serve(on: site)
            site.behave(plain.remotePath(Bundle.weights), .init(redirect: URL(string: "http://\(cdn.host)/signed/weights")!))
            await #expect {
                _ = try await store.install(plain.ref)
            } throws: { error in
                if case AetherError.download(Bundle.weights, .transport) = error { return true }
                return false
            }
            #expect(cdn.requests.count == 1, "the plain-HTTP hop is never made")
        }
    }

    // MARK: Range edge cases

    @Test func anUnsatisfiableRangeRestartsTheFileOnce() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let path = Bundle.weights
            site.behave(bundle.remotePath(path), .init(rangeStatus: 416))
            let store = Self.store(root, site)
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bundle.contents[path]!.prefix(5000).write(to: staged)

            try Self.expectInstalled(try await store.install(bundle.ref), bundle)
            #expect(site.requests.filter { $0.path == bundle.remotePath(path) }.map(\.range) == ["bytes=5000-", nil])
        }
    }

    @Test func aRangeThatDoesNotContinueTheFileIsRefused() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let path = Bundle.weights
            site.behave(bundle.remotePath(path), .init(contentRangeStart: 3000))
            let store = Self.store(root, site)
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bundle.contents[path]!.prefix(5000).write(to: staged)
            await #expect {
                _ = try await store.install(bundle.ref)
            } throws: { error in
                if case AetherError.download(path, .unexpectedContent) = error { return true }
                return false
            }
        }
    }

    @Test func aPartialResponseFromTheStartRestartsTheFile() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let path = Bundle.weights
            site.behave(bundle.remotePath(path), .init(partialFromStart: true))
            let store = Self.store(root, site)
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 9, count: 5000).write(to: staged)
            try Self.expectInstalled(try await store.install(bundle.ref), bundle)
        }
    }

    @Test func anOversizedStagedFileStartsOver() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            let store = Self.store(root, site)
            let path = ".aether/assets/tokenizer/tokenizer.json"
            let staged = store.stagingURL(for: bundle.ref).appending(path: path)
            try FileManager.default.createDirectory(at: staged.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(repeating: 0, count: 10_000).write(to: staged)
            try Self.expectInstalled(try await store.install(bundle.ref), bundle)
            #expect(site.requests.filter { $0.path == bundle.remotePath(path) }.map(\.range) == [nil])
        }
    }

    @Test func aBodyLargerThanThePinnedSizeIsStopped() async throws {
        try await withTemporaryDirectory { root in
            let site = StubSite()
            let bundle = Bundle()
            bundle.serve(on: site)
            site.behave(bundle.remotePath("macos/toy/metadata.json"), .init(extra: Data(repeating: 1, count: 500)))
            let store = Self.store(root, site)
            await #expect {
                _ = try await store.install(bundle.ref)
            } throws: { error in
                if case AetherError.download("macos/toy/metadata.json", .unexpectedContent) = error { return true }
                return false
            }
        }
    }

    @Test func contentRangeParsing() {
        let range = ContentRange("bytes 128-299/300")
        #expect(range?.start == 128 && range?.end == 299 && range?.total == 300)
        #expect(ContentRange("bytes 5-9/*") != nil && ContentRange("bytes 5-9/*")?.total == nil)
        #expect(ContentRange("items 1-2/3") == nil)
        #expect(ContentRange("bytes x-2/3") == nil)
        #expect(ContentRange("bytes 5-/9") == nil)
        #expect(ContentRange("bytes 9-5/10") == nil)
    }
}
