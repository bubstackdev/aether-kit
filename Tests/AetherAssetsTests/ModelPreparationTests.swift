@testable import AetherAssets
import CryptoKit
import Foundation
import Testing

private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

/// A transport that serves files from a local directory, with scripted failures per (path, attempt).
private final class ScriptedTransport: ModelTransport, @unchecked Sendable {
    enum Behavior { case normal, cutAfter(Int), corrupt, short, waitForRelease }
    struct Call: Equatable { let path: String; let resume: Bool; let offset: UInt64 }

    let source: URL
    private let lock = NSLock()
    private var script: [String: [Behavior]] = [:]
    private var log: [Call] = []
    private var released = false

    init(source: URL) { self.source = source }

    func on(_ path: String, _ behaviors: Behavior...) { lock.withLock { script[path] = behaviors } }
    func release() { lock.withLock { released = true } }
    var calls: [Call] { lock.withLock { log } }

    func fetch(_ request: TransportRequest, to destination: URL, resume: Bool,
               progress: @escaping @Sendable (UInt64) -> Void) async throws {
        let path = request.file.path
        let existing = (try? FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? NSNumber)?.uint64Value ?? 0
        let offset = resume ? existing : 0
        let behavior: Behavior = lock.withLock {
            log.append(Call(path: path, resume: resume, offset: offset))
            guard var queue = script[path], !queue.isEmpty else { return .normal }
            let next = queue.removeFirst(); script[path] = queue
            return next
        }
        var bytes = try Data(contentsOf: source.appendingPathComponent(path))
        switch behavior {
        case .corrupt: bytes[bytes.startIndex] ^= 0xff
        case .short: bytes.removeLast()
        default: break
        }
        if !resume { FileManager.default.createFile(atPath: destination.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        try handle.seekToEnd()
        var position = Int(offset)
        while position < bytes.count {
            try Task.checkCancellation()
            if case let .cutAfter(limit) = behavior, position >= limit { throw URLError(.networkConnectionLost) }
            if case .waitForRelease = behavior, position >= bytes.count / 2 {
                while !lock.withLock({ released }) {
                    try Task.checkCancellation()
                    try await Task.sleep(for: .milliseconds(5))
                }
            }
            let end = min(position + 16, bytes.count)
            try handle.write(contentsOf: bytes[position..<end])
            position = end
            progress(UInt64(position))
        }
    }
}

/// A source bundle, its catalog entry and an empty store.
private struct Fixture {
    let base = FileManager.default.temporaryDirectory.appendingPathComponent("preparation-\(UUID().uuidString)")
    var source: URL { base.appendingPathComponent("source") }
    var storeRoot: URL { base.appendingPathComponent("store") }
    let entry: ModelCatalogEntry

    init() throws {
        let contents: [String: Data] = [
            "registration.json": Data(#"{"model":"example"}"#.utf8),
            "model/main.mlirb": Data((0..<300).map { UInt8($0 % 251) }),
            "tokenizer/tokenizer.json": Data(String(repeating: "token ", count: 20).utf8),
        ]
        var files: [AssetFile] = []
        for (path, data) in contents {
            let url = base.appendingPathComponent("source").appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
            files.append(AssetFile(path: path, byteCount: UInt64(data.count), sha256: sha256(data)))
        }
        entry = try ModelCatalogEntry(
            manifest: AssetManifest(assetIdentity: "example/1:fixture", executionContract: "example.bundle/1", files: files),
            source: .https(base: URL(string: "https://models.example.com/fixture")!))
    }

    func preparation(_ transport: ScriptedTransport, capacity: UInt64 = .max) throws -> ModelPreparation {
        ModelPreparation(store: try LocalAssetStore(root: storeRoot), transport: transport, headers: { _ in [:] },
                         availableCapacity: { _ in capacity }, progressInterval: .zero)
    }

    func remove() { try? FileManager.default.removeItem(at: base) }
}

private func events(_ stream: AsyncStream<PreparationEvent>) async -> [PreparationEvent] {
    var all: [PreparationEvent] = []
    for await event in stream { all.append(event) }
    return all
}

struct ModelPreparationTests {
    @Test func aFreshEntryDownloadsInstallsAndCleansUp() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let preparation = try fixture.preparation(transport)
        #expect(try preparation.status(fixture.entry) == .notInstalled(bytesTotal: fixture.entry.byteCount))

        let all = await events(preparation.prepare(fixture.entry))
        let id = try fixture.entry.installationID
        #expect(all.first == .checking)
        #expect(all.last == .installed(installationID: id))
        #expect(all.filter(\.isTerminal).count == 1)
        let downloaded = all.compactMap { event -> UInt64? in
            if case let .downloading(_, _, _, done, total) = event { #expect(total == fixture.entry.byteCount); return done }
            return nil
        }
        #expect(downloaded == downloaded.sorted() && downloaded.last == fixture.entry.byteCount)
        #expect(all.contains(.installing(bytesDone: 0, bytesTotal: fixture.entry.byteCount)))

        #expect(try preparation.status(fixture.entry) == .installed(installationID: id))
        #expect(!FileManager.default.fileExists(atPath: preparation.store.downloads.appendingPathComponent(id).path))
        let lease = try await preparation.store.acquire(id)
        #expect(try Data(contentsOf: lease.directory.appendingPathComponent("model/main.mlirb"))
            == Data(contentsOf: fixture.source.appendingPathComponent("model/main.mlirb")))
    }

    @Test func aCutConnectionResumesFromTheBytesOnDisk() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .cutAfter(128))
        let preparation = try fixture.preparation(transport)

        let first = await events(preparation.prepare(fixture.entry))
        guard case let .failed(error)? = first.last else { Issue.record("expected a failure, got \(first)"); return }
        #expect(error == .network(URLError(.networkConnectionLost).localizedDescription) && error.isRetryable)
        guard case let .partial(present, total) = try preparation.status(fixture.entry) else {
            Issue.record("expected partial data"); return
        }
        #expect(present > 0 && present < total)

        let id = try await preparation.install(fixture.entry)
        #expect(try id == fixture.entry.installationID)
        let resumed = transport.calls.filter { $0.path == "model/main.mlirb" }
        #expect(resumed.count == 2 && resumed[1].resume && resumed[1].offset == 128)
    }

    @Test func bytesThatDontHashToTheEntryAreDroppedAndRefetched() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .corrupt)
        let preparation = try fixture.preparation(transport)

        let first = await events(preparation.prepare(fixture.entry))
        #expect(first.last == .failed(.integrityMismatch(path: "model/main.mlirb")))
        #expect(try !preparation.store.isInstalled(fixture.entry.installationID))
        _ = try await preparation.install(fixture.entry)
        let fetches = transport.calls.filter { $0.path == "model/main.mlirb" }
        #expect(fetches.count == 2 && !fetches[1].resume)
    }

    @Test func aFileOfTheWrongSizeIsRefused() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("tokenizer/tokenizer.json", .short)
        let preparation = try fixture.preparation(transport)
        let all = await events(preparation.prepare(fixture.entry))
        #expect(all.last == .failed(.integrityMismatch(path: "tokenizer/tokenizer.json")))
    }

    @Test func notEnoughSpaceRefusesBeforeDownloading() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let preparation = try fixture.preparation(transport, capacity: fixture.entry.byteCount)
        let all = await events(preparation.prepare(fixture.entry))
        #expect(all.last == .failed(.insufficientSpace(needed: 2 * fixture.entry.byteCount, available: fixture.entry.byteCount)))
        #expect(transport.calls.isEmpty)
    }

    @Test func anInstalledEntryNeedsNoTransport() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = try LocalAssetStore(root: fixture.storeRoot)
        _ = try await store.install(fixture.entry.manifest, from: fixture.source)
        let transport = ScriptedTransport(source: fixture.source)
        let preparation = try fixture.preparation(transport)
        let all = await events(preparation.prepare(fixture.entry))
        #expect(try all.last == .installed(installationID: fixture.entry.installationID))
        #expect(transport.calls.isEmpty)
    }

    @Test func aSecondRunForTheSameEntryIsBusy() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .waitForRelease)
        let preparation = try fixture.preparation(transport)
        let first = Task { await events(preparation.prepare(fixture.entry)) }
        while !transport.calls.contains(where: { $0.path == "model/main.mlirb" }) { try await Task.sleep(for: .milliseconds(5)) }
        let second = await events(preparation.prepare(fixture.entry))
        #expect(second.last == .failed(.busy))
        transport.release()
        #expect(try await first.value.last == .installed(installationID: fixture.entry.installationID))
    }

    @Test func stoppingTheIterationCancelsAndKeepsTheDownloadedBytes() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        transport.on("model/main.mlirb", .waitForRelease)
        let preparation = try fixture.preparation(transport)
        for await event in preparation.prepare(fixture.entry) {
            if case let .downloading(path, fileBytes, _, _, _) = event, path == "model/main.mlirb", fileBytes > 0 { break }
        }
        // The cancelled worker releases the download lock; wait until a status read sees the kept bytes.
        var status = try preparation.status(fixture.entry)
        for _ in 0..<200 {
            if case .partial = status, (try? preparation.store.downloadLock(fixture.entry.installationID)) != nil { break }
            try await Task.sleep(for: .milliseconds(5))
            status = try preparation.status(fixture.entry)
        }
        guard case .partial = status else { Issue.record("expected kept partial data, got \(status)"); return }
        transport.release()
        _ = try await preparation.install(fixture.entry)
        #expect(transport.calls.last { $0.path == "model/main.mlirb" }?.resume == true)
    }

    @Test func anOversizedPartialFileRestarts() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let transport = ScriptedTransport(source: fixture.source)
        let preparation = try fixture.preparation(transport)
        let area = preparation.store.downloads.appendingPathComponent(try fixture.entry.installationID)
            .appendingPathComponent("files/tokenizer")
        try FileManager.default.createDirectory(at: area, withIntermediateDirectories: true)
        try Data(repeating: 0, count: 10_000).write(to: area.appendingPathComponent("tokenizer.json"))
        _ = try await preparation.install(fixture.entry)
        #expect(transport.calls.first { $0.path == "tokenizer/tokenizer.json" }?.resume == false)
    }

    @Test func cleanupRemovesDownloadAreasOfPublishedInstallations() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = try LocalAssetStore(root: fixture.storeRoot)
        let id = try await store.install(fixture.entry.manifest, from: fixture.source)
        let leftover = store.downloads.appendingPathComponent(id).appendingPathComponent("files")
        try FileManager.default.createDirectory(at: leftover, withIntermediateDirectories: true)
        #expect(try store.removeAbandonedStages() == 1)
        #expect(!FileManager.default.fileExists(atPath: store.downloads.appendingPathComponent(id).path))
    }

    @Test func retryabilityFollowsTheCause() {
        #expect(PreparationError.http(status: 503, path: "a").isRetryable)
        #expect(PreparationError.http(status: 429, path: "a").isRetryable)
        #expect(!PreparationError.http(status: 404, path: "a").isRetryable)
        #expect(!PreparationError.insufficientSpace(needed: 2, available: 1).isRetryable)
        #expect(PreparationError.busy.isRetryable)
    }
}
