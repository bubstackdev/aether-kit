import CryptoKit
import Foundation
import Synchronization
import Testing
@testable import AetherCore

/// The location-free content identity and full verification of a bundle on disk outside any store: two small asset
/// files (one in a foreign repo's asset) and the Krea 2 fixture manifest.
@Suite("Bundle identity and verification")
struct BundleIdentityTests {
    static let manifestURL = URL(filePath: #filePath).deletingLastPathComponent()
        .appending(path: "Fixtures/krea2/vanilla.json")
    static let weights = Data(repeating: 7, count: 3 << 20)
    static let vocabulary = Data("{\"a\": 1}".utf8)

    static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    static func ref(weights: Data = weights, revision: String = "r1") -> ModelRef {
        ModelRef(
            catalogID: "fixture", repo: "owner/model", revision: revision, variant: "macos", driver: "krea2@1",
            assets: [
                .init(name: "main", repo: "owner/model", revision: revision, subtree: "",
                      files: [CatalogFile(path: "graph.aimodel/weights.bin", size: Int64(weights.count), sha256: sha256(weights))]),
                .init(name: "tokenizer", repo: "owner/tokenizer", revision: "t1", subtree: "",
                      files: [CatalogFile(path: "vocab.json", size: Int64(vocabulary.count), sha256: sha256(vocabulary))]),
            ])
    }

    /// Writes `ref`'s files (with `weights` as the graph's bytes) and the fixture manifest under `root`.
    static func write(_ ref: ModelRef, at root: URL, weights: Data = weights, manifest: Data? = nil) throws -> InstalledBundle {
        let bundle = InstalledBundle(ref: ref, url: root, isStale: false)
        let graph = try #require(bundle.url(of: AssetPath(asset: "main", path: "graph.aimodel/weights.bin")))
        let vocab = try #require(bundle.url(of: AssetPath(asset: "tokenizer", path: "vocab.json")))
        for (url, data) in [(graph, weights), (vocab, vocabulary)] {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url)
        }
        let manifestFile = Manifest.url(inBundle: root)
        try FileManager.default.createDirectory(at: manifestFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (manifest ?? Data(contentsOf: manifestURL)).write(to: manifestFile)
        return bundle
    }

    @Test func theIdentityIgnoresLocationRepoAndRevision() async throws {
        try await withTemporaryDirectory { dir in
            let one = try Self.write(Self.ref(), at: dir.appending(path: "store/owner/model/r1"))
            let restored = try Self.write(Self.ref(revision: "r2"), at: dir.appending(path: "archive/restored"))
            #expect(try one.contentIdentity() == restored.contentIdentity())
            #expect(try one.contentIdentity().count == 64)
        }
    }

    @Test func aChangedFileOrManifestGetsANewIdentity() async throws {
        try await withTemporaryDirectory { dir in
            let original = try Self.write(Self.ref(), at: dir.appending(path: "a"))
            let other = Data(repeating: 8, count: Self.weights.count)
            let replaced = try Self.write(Self.ref(weights: other), at: dir.appending(path: "b"), weights: other)
            var manifest = try Manifest.decode(Data(contentsOf: Self.manifestURL))
            manifest.license = "changed"
            let edited = try Self.write(Self.ref(), at: dir.appending(path: "c"), manifest: try manifest.encoded())
            let ids = try [original, replaced, edited].map { try $0.contentIdentity() }
            #expect(Set(ids).count == 3)
        }
    }

    @Test func capabilityIdentitiesDifferFromEachOtherAndThePlainOne() async throws {
        try await withTemporaryDirectory { dir in
            let bundle = try Self.write(Self.ref(), at: dir)
            let plain = try bundle.contentIdentity()
            let a = try bundle.contentIdentity(capability: "references")
            #expect(Set([plain, a, try bundle.contentIdentity(capability: "refine")]).count == 3)
            #expect(try bundle.contentIdentity(capability: "references") == a)
        }
    }

    @Test func verificationReadsEveryFileAndReportsProgress() async throws {
        try await withTemporaryDirectory { dir in
            let bundle = try Self.write(Self.ref(), at: dir)
            let seen = Mutex<[BundleVerificationProgress]>([])
            try bundle.verify { update in seen.withLock { $0.append(update) } }
            let updates = seen.withLock { $0 }
            let total = Int64(Self.weights.count + Self.vocabulary.count)
            #expect(updates.first == BundleVerificationProgress(completedBytes: 0, totalBytes: total, file: nil))
            #expect(updates.last?.completedBytes == total && updates.last?.fractionCompleted == 1)
            #expect(Set(updates.compactMap(\.file)) == [".aether/assets/tokenizer/vocab.json", "graph.aimodel/weights.bin"])
            #expect(zip(updates, updates.dropFirst()).allSatisfy { $0.completedBytes <= $1.completedBytes })
        }
    }

    @Test func verificationRefusesAMissingResizedOrAlteredFile() async throws {
        try await withTemporaryDirectory { dir in
            let bundle = try Self.write(Self.ref(), at: dir)
            let graph = try #require(bundle.url(of: AssetPath(asset: "main", path: "graph.aimodel/weights.bin")))

            var altered = Self.weights
            altered[altered.count - 1] = 9
            try altered.write(to: graph)
            #expect(Self.integrityActual(of: bundle) == Self.sha256(altered))

            try Self.weights.prefix(10).write(to: graph)
            #expect(Self.integrityActual(of: bundle) == "size 10")

            try FileManager.default.removeItem(at: graph)
            #expect(Self.integrityActual(of: bundle) == "missing")
        }
    }

    @Test func verificationStopsOnceCancelled() async throws {
        try await withTemporaryDirectory { dir in
            let bundle = try Self.write(Self.ref(), at: dir)
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                try bundle.verify()
            }
            await #expect(throws: CancellationError.self) { try await task.value }
        }
    }

    /// The `actual` of the integrity error `verify()` throws, or nil.
    static func integrityActual(of bundle: InstalledBundle) -> String? {
        do {
            try bundle.verify()
            return nil
        } catch AetherError.integrity(_, _, let actual) {
            return actual
        } catch {
            return nil
        }
    }
}
