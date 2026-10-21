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

// A stage without a held installation lock models the filesystem state left by a dead importer.
private func abandonedStage(_ fixture: Fixture, id: String) throws -> URL {
    let stage = fixture.storeRoot.appendingPathComponent("staging/\(id).\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: stage.appendingPathComponent("files/nested"), withIntermediateDirectories: true)
    try Data("partial".utf8).write(to: stage.appendingPathComponent("files/nested/incomplete"))
    return stage
}

@Test func abandonedStageCleanupPreservesPublishedInstallsAndUnknownEntries() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let manifest = try fixture.manifest(), id = try manifest.installationID
    _ = try await store.install(manifest, from: fixture.source)
    let abandoned = try abandonedStage(fixture, id: id)
    let staging = fixture.storeRoot.appendingPathComponent("staging")
    let unknown = [UUID().uuidString, "unrecognized", id + ".not-a-uuid", String(repeating: "g", count: 64) + "." + UUID().uuidString]
    for name in unknown {
        try FileManager.default.createDirectory(at: staging.appendingPathComponent(name), withIntermediateDirectories: true)
    }
    #expect(try store.removeAbandonedStages() == 1)
    #expect(!FileManager.default.fileExists(atPath: abandoned.path))
    #expect(Set(try FileManager.default.contentsOfDirectory(atPath: staging.path)) == Set(unknown))
    #expect(try store.removeAbandonedStages() == 0)
    let installed = try await store.acquire(id)
    #expect(try Data(contentsOf: installed.directory.appendingPathComponent("nested/model")) == Data("model".utf8))
    withExtendedLifetime(installed) {}
}

@Test func abandonedStageCleanupDefersToReadersAndCanRetryAfterRelease() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let id = try await store.install(fixture.manifest(), from: fixture.source)
    let stage = try abandonedStage(fixture, id: id)
    do {
        let reader = try await store.acquire(id)
        let removed = try store.removeAbandonedStages()
        withExtendedLifetime(reader) {
            #expect(removed == 0)
            #expect(FileManager.default.fileExists(atPath: stage.path))
        }
    }
    #expect(try store.removeAbandonedStages() == 1)
    let reader = try await store.acquire(id)
    withExtendedLifetime(reader) { #expect(reader.installationID == id) }
}

@Test(.timeLimit(.minutes(1))) func abandonedStageCleanupSkipsLiveImportAndRecoversOtherIdentities() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let ordinary = try LocalAssetStore(root: fixture.storeRoot)
    let oldManifest = try fixture.manifest(contract: "abandoned/1")
    let orphan = try abandonedStage(fixture, id: oldManifest.installationID)
    let manifest = try AssetManifest(assetIdentity: "paused/1", executionContract: "test/1", files: [
        AssetFile(path: "nested/model", byteCount: 5, sha256: hash(Data("model".utf8)))
    ])
    let started = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
    let paused = try LocalAssetStore(root: fixture.storeRoot, onCopiedChunk: {
        started.signal()
        guard resume.wait(timeout: .now() + 10) == .success else { throw AssetStoreError.busy }
    })
    let task = Task { try await paused.install(manifest, from: fixture.source) }
    defer { resume.signal() }
    let observed = await withCheckedContinuation { continuation in
        DispatchQueue.global().async {
            continuation.resume(returning: started.wait(timeout: .now() + 5) == .success)
        }
    }
    try #require(observed)
    let before = try FileManager.default.contentsOfDirectory(atPath: fixture.storeRoot.appendingPathComponent("staging").path)
    #expect(before.count == 2)
    #expect(try ordinary.removeAbandonedStages() == 1)
    #expect(!FileManager.default.fileExists(atPath: orphan.path))
    #expect(try FileManager.default.contentsOfDirectory(atPath: fixture.storeRoot.appendingPathComponent("staging").path).count == 1)
    resume.signal()
    let id = try await task.value
    #expect(id == (try manifest.installationID))
    #expect(try ordinary.removeAbandonedStages() == 0)
    let reader = try await ordinary.acquire(id)
    withExtendedLifetime(reader) { #expect(reader.manifest == manifest) }
}

@Test func abandonedStageCleanupRefusesSymlinksAndNeverFollowsTheirTargets() throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let id = try fixture.manifest().installationID
    let link = fixture.storeRoot.appendingPathComponent("staging/\(id).\(UUID().uuidString)")
    try FileManager.default.createSymbolicLink(at: link, withDestinationURL: fixture.source)
    #expect(throws: (any Error).self) { try store.removeAbandonedStages() }
    #expect(try Data(contentsOf: fixture.source.appendingPathComponent("nested/model")) == Data("model".utf8))
    try FileManager.default.removeItem(at: link)
    let staging = fixture.storeRoot.appendingPathComponent("staging")
    try FileManager.default.removeItem(at: staging)
    try FileManager.default.createSymbolicLink(at: staging, withDestinationURL: fixture.source)
    #expect(throws: (any Error).self) { try store.removeAbandonedStages() }
    #expect(FileManager.default.fileExists(atPath: fixture.source.appendingPathComponent("nested/model").path))
}

@Test func cancelledAbandonedStageCleanupPreservesStageForRetry() async throws {
    let fixture = try Fixture(); defer { fixture.cleanup() }
    let store = try LocalAssetStore(root: fixture.storeRoot)
    let stage = try abandonedStage(fixture, id: fixture.manifest().installationID)
    let task = Task.detached {
        withUnsafeCurrentTask { $0?.cancel() }
        return try store.removeAbandonedStages()
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(FileManager.default.fileExists(atPath: stage.path))
    #expect(try store.removeAbandonedStages() == 1)
}
