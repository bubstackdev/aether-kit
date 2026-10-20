@testable import AetherAssets
import CryptoKit
import Foundation
import Testing

private struct Fixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    var source: URL { root.appendingPathComponent("source") }
    var storeRoot: URL { root.appendingPathComponent("store") }
    init() throws {
        try FileManager.default.createDirectory(at: source.appendingPathComponent("nested"), withIntermediateDirectories: true)
        try Data("model".utf8).write(to: source.appendingPathComponent("nested/model"))
        try Data().write(to: source.appendingPathComponent("empty"))
    }
    func cleanup() { try? FileManager.default.removeItem(at: root) }
    func manifest(contract: String = "test/1") throws -> AssetManifest {
        try AssetManifest(assetIdentity: "fixture@immutable-revision", executionContract: contract, files: [
            AssetFile(path: "nested/model", byteCount: 5, sha256: hash(Data("model".utf8))),
            AssetFile(path: "empty", byteCount: 0, sha256: hash(Data())),
        ])
    }
}
private func hash(_ value: Data) -> String { SHA256.hash(data: value).map { String(format: "%02x", $0) }.joined() }

@Test func assetIdentitySurvivesRelocationAndSeparatesExecutionContracts() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let manifest = try fixture.manifest()
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let id = try await store.install(manifest, from: fixture.source)
    #expect(id == (try manifest.installationID))
    #expect(try await store.install(manifest, from: fixture.source) == id)
    #expect(try fixture.manifest(contract: "references/1").installationID != id)
    let relocated = fixture.root.appendingPathComponent("relocated")
    try FileManager.default.moveItem(at: fixture.storeRoot, to: relocated)
    let moved = try LocalAssetStore(root: relocated)
    let lease = try await moved.acquire(id)
    #expect(lease.manifest == manifest)
    #expect(try Data(contentsOf: lease.directory.appendingPathComponent("nested/model")) == Data("model".utf8))
}

@Test func readersAcrossStoreInstancesPreventRemoval() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let first = try LocalAssetStore(root: fixture.storeRoot)
    let second = try LocalAssetStore(root: fixture.storeRoot)
    let id = try await first.install(fixture.manifest(), from: fixture.source)
    do {
        let lease = try await second.acquire(id)
        withExtendedLifetime(lease) {
            _ = #expect(throws: AssetStoreError.busy) { try first.remove(id) }
        }
        // A second reader is allowed while the first reader lives.
        let peer = try await first.acquire(id)
        withExtendedLifetime((lease, peer)) { #expect(peer.installationID == lease.installationID) }
    }
    try first.remove(id)
    await #expect(throws: AssetStoreError.notInstalled) { try await second.acquire(id) }
}

@Test func corruptImportLeavesPreviousInstallAndNoStage() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let previous = try await store.install(fixture.manifest(), from: fixture.source)
    try Data("wrong".utf8).write(to: fixture.source.appendingPathComponent("nested/model"))
    await #expect(throws: AssetStoreError.integrityMismatch("nested/model")) {
        try await store.install(fixture.manifest(contract: "new/1"), from: fixture.source)
    }
    let lease = try await store.acquire(previous)
    #expect(try Data(contentsOf: lease.directory.appendingPathComponent("nested/model")) == Data("model".utf8))
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.storeRoot.appendingPathComponent("staging").path).isEmpty)
}

@Test func installedCorruptionIsRejectedOnAcquireAndRetry() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let manifest = try fixture.manifest()
    let id = try await store.install(manifest, from: fixture.source)
    try Data("wrong".utf8).write(to: fixture.storeRoot.appendingPathComponent("installed/\(id)/files/nested/model"))
    await #expect(throws: AssetStoreError.integrityMismatch("nested/model")) { try await store.acquire(id) }
    await #expect(throws: AssetStoreError.integrityMismatch("nested/model")) { try await store.install(manifest, from: fixture.source) }
    // Explicit cleanup must work even when the metadata itself is corrupt.
    try Data("broken metadata".utf8).write(to: fixture.storeRoot.appendingPathComponent("installed/\(id)/manifest.json"))
    try store.remove(id)
    #expect(try await store.install(manifest, from: fixture.source) == id)
}

@Test func traversalCollisionsAndDecodedUnknownSchemasAreRejected() throws {
    let sha = hash(Data())
    for path in ["../outside", "/absolute", "a//b", "a/./b", "a/../b", "a\\b", "a/", ""] {
        #expect(throws: AssetStoreError.self) { try AssetManifest(assetIdentity: "a", executionContract: "v1", files: [AssetFile(path: path, byteCount: 0, sha256: sha)]) }
    }
    #expect(throws: AssetStoreError.self) {
        try AssetManifest(assetIdentity: "a", executionContract: "v1", files: ["a", "a/b"].map { AssetFile(path: $0, byteCount: 0, sha256: sha) })
    }
    let valid = try AssetManifest(assetIdentity: "a", executionContract: "v1", files: [AssetFile(path: "a", byteCount: 0, sha256: sha)])
    let encoded = try JSONEncoder().encode(valid)
    let json = String(decoding: encoded, as: UTF8.self).replacingOccurrences(of: "\"schemaVersion\":1", with: "\"schemaVersion\":2")
    #expect(throws: AssetStoreError.self) { try JSONDecoder().decode(AssetManifest.self, from: Data(json.utf8)) }
    var object = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    object["requiredFeatures"] = ["unsupported-recipe"]
    let required = try JSONSerialization.data(withJSONObject: object)
    #expect(throws: AssetStoreError.self) { try JSONDecoder().decode(AssetManifest.self, from: required) }
}

@Test func sourceSymlinksAndWrongFileSizesAreRejected() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let original = fixture.source.appendingPathComponent("nested/model")
    try FileManager.default.removeItem(at: original)
    try FileManager.default.createSymbolicLink(at: original, withDestinationURL: fixture.source.appendingPathComponent("empty"))
    await #expect(throws: AssetStoreError.unsafePath("nested/model")) {
        try await store.install(fixture.manifest(), from: fixture.source)
    }
    try FileManager.default.removeItem(at: original)
    try Data("model plus extra".utf8).write(to: original)
    await #expect(throws: AssetStoreError.integrityMismatch("nested/model")) {
        try await store.install(fixture.manifest(), from: fixture.source)
    }
}

@Test func cancelledInstallDoesNotPublish() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await store.install(fixture.manifest(), from: fixture.source)
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.storeRoot.appendingPathComponent("installed").path).isEmpty)
}


@Test(.timeLimit(.minutes(1))) func interruptedCopyJoinsAndUnlocksBeforeRetry() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let ordinary = try LocalAssetStore(root: fixture.storeRoot)
    let previous = try await ordinary.install(fixture.manifest(), from: fixture.source)
    let bytes = Data(repeating: 17, count: 3 << 20)
    try bytes.write(to: fixture.source.appendingPathComponent("large"))
    let manifest = try AssetManifest(assetIdentity: "large/1", executionContract: "test/1", files: [
        AssetFile(path: "large", byteCount: UInt64(bytes.count), sha256: hash(bytes))
    ])
    let started = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
    let paused = try LocalAssetStore(root: fixture.storeRoot, onCopiedChunk: {
        started.signal()
        guard resume.wait(timeout: .now() + 5) == .success else { throw AssetStoreError.busy }
    })
    let task = Task { try await paused.install(manifest, from: fixture.source) }
    let observed = await withCheckedContinuation { continuation in
        DispatchQueue.global().async {
            continuation.resume(returning: started.wait(timeout: .now() + 5) == .success)
        }
    }
    #expect(observed)
    await #expect(throws: AssetStoreError.busy) { try await ordinary.install(manifest, from: fixture.source) }
    task.cancel(); resume.signal()
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.storeRoot.appendingPathComponent("staging").path).isEmpty)
    let previousLease = try await ordinary.acquire(previous)
    let retried = try await ordinary.install(manifest, from: fixture.source)
    #expect(retried == (try manifest.installationID))
    withExtendedLifetime(previousLease) {}
}
