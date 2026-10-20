import AetherAssets
import AetherKit
import CryptoKit
import Foundation

@main struct Consumer {
    static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("weight-free consumer fixture".utf8)
        try bytes.write(to: root.appendingPathComponent("fixture"))
        let manifest = try AssetManifest(assetIdentity: "consumer-fixture/1", executionContract: "example/1", files: [
            AssetFile(path: "fixture", byteCount: UInt64(bytes.count), sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
        ])
        let store = try LocalAssetStore(root: root.appendingPathComponent("assets"))
        let id = try await store.install(manifest, from: root)
        do {
            let lease = try await store.acquire(id)
            guard try Data(contentsOf: lease.directory.appendingPathComponent("fixture")) == bytes else { throw Failure.mismatch }
            do { try store.remove(id); throw Failure.missingProtection } catch AssetStoreError.busy {}
            withExtendedLifetime(lease) {}
        }
        try store.remove(id)
        let resources = try ResourceCoordinator(capacityBytes: 100)
        let lease = try resources.reserve(bytes: 60)
        guard resources.snapshot.availableBytes == 40 else { throw Failure.mismatch }
        lease.release()
        guard resources.snapshot.reservedBytes == 0 else { throw Failure.mismatch }
        print("Foundation consumer passed: verified install, reader protection, removal and reservation release.")
    }
    enum Failure: Error { case mismatch, missingProtection }
}
