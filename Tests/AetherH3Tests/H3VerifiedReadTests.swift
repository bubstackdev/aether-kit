import AetherAssets
import AetherH3
import AetherKit
import Darwin
import Foundation
import Testing

/// Cached tensor and embedding verification (``VerifiedFileHashes``) never lets bytes through that weren't hashed.
@Suite(.serialized) struct H3VerifiedReadTests {
    private let root: URL

    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    private func write(_ name: String, _ bytes: [UInt8]) throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data(bytes).write(to: url)
        return url
    }

    private func read(_ file: H3TensorFile) throws -> Data {
        var destination = Data(count: file.spec.byteCount)
        try destination.withUnsafeMutableBytes { try file.read(into: $0, verify: true) }
        return destination
    }

    @Test func leasedReadIdentifiesItsDescriptorNotThePath() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [2])
        let old = Data([128, 63, 0, 64]), new = Data([64, 64, 128, 64])
        let shard = try write("shard", Array(old))
        let lease = try ExternalFileLease(files: [shard])
        // The leased tensor declares the new bytes' digest; its descriptor still holds the old bytes.
        let leased = try H3TensorFile(name: "w", url: shard, spec: spec, sha256: h3SHA256(new)).holding(lease)
        let replacement = try write("replacement", Array(new))
        #expect(rename(replacement.path, shard.path) == 0)
        // The new file at the path is verified and recorded under the same key.
        let unleased = try H3TensorFile(name: "w", url: shard, spec: spec, sha256: h3SHA256(new))
        #expect(try read(unleased) == new)
        #expect(try read(unleased) == new)
        #expect(throws: AetherError.self) { try read(leased) }
        // A leased read of the old bytes under their own digest verifies, then hits.
        let original = try H3TensorFile(name: "w", url: shard, spec: spec, sha256: h3SHA256(old)).holding(lease)
        #expect(try read(original) == old)
        #expect(try read(original) == old)
    }

    @Test func sameDigestAtAnotherRangeIsHashedAgain() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [2])
        let shard = try write("shard", [128, 63, 0, 64, 64, 64, 128, 64])
        let digest = h3SHA256(Data([128, 63, 0, 64]))
        let first = try H3TensorFile(name: "a", segments: [.init(url: shard, offset: 0, length: 4)], spec: spec, sha256: digest)
        #expect(try read(first) == Data([128, 63, 0, 64]))
        #expect(try read(first) == Data([128, 63, 0, 64]))
        let other = try H3TensorFile(name: "b", segments: [.init(url: shard, offset: 4, length: 4)], spec: spec, sha256: digest)
        #expect(throws: AetherError.self) { try read(other) }
        // The same ranges in another order are another claim too.
        let swapped = try H3TensorFile(name: "c", segments: [.init(url: shard, offset: 2, length: 2), .init(url: shard, offset: 0, length: 2)],
                                       spec: spec, sha256: digest)
        #expect(throws: AetherError.self) { try read(swapped) }
        let leased = try other.holding(ExternalFileLease(files: [shard]))
        #expect(throws: AetherError.self) { try read(leased) }
    }

    @Test func changeDuringACachedReadIsHashed() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = try H3TensorSpec(scalarType: .bfloat16, shape: [2])
        let bytes = Data([128, 63, 0, 64])
        let shard = try write("shard", Array(bytes))
        let file = try H3TensorFile(name: "w", url: shard, spec: spec, sha256: h3SHA256(bytes))
        #expect(try read(file) == bytes)
        let path = shard.path
        // Rewritten in place after the cached copy: the read hashes again and refuses.
        var hooks = VerifiedFileHashes.Hooks()
        hooks.afterCachedRead = {
            let fd = open(path, O_WRONLY)
            _ = [UInt8]([0, 0]).withUnsafeBytes { pwrite(fd, $0.baseAddress, 2, 0) }
            close(fd)
        }
        #expect(throws: AetherError.self) { try VerifiedFileHashes.$hooks.withValue(hooks) { try read(file) } }
        // Restored bytes with a new status-change time: hashed again (and recorded again), never trusted as cached.
        try bytes.write(to: shard)
        var touched = VerifiedFileHashes.Hooks()
        touched.afterCachedRead = { _ = utimes(path, nil) }
        #expect(try read(file) == bytes)
        #expect(try VerifiedFileHashes.$hooks.withValue(touched) { try read(file) } == bytes)
    }

    @Test func leasedEmbeddingTableIdentifiesItsDescriptor() throws {
        defer { try? FileManager.default.removeItem(at: root) }
        let old = Data([128, 63, 0, 64]), new = Data([64, 64, 128, 64])
        let shard = try write("shard", Array(old))
        let lease = try ExternalFileLease(files: [shard])
        let leased = try H3EmbeddingTable(url: shard, offset: 0, rows: 1, width: 2, byteCount: 4, sha256: h3SHA256(new)).holding(lease)
        let replacement = try write("replacement", Array(new))
        #expect(rename(replacement.path, shard.path) == 0)
        let unleased = try H3EmbeddingTable(url: shard, offset: 0, rows: 1, width: 2, byteCount: 4, sha256: h3SHA256(new))
        try unleased.verify()
        try unleased.verify()
        #expect(throws: AetherError.self) { try leased.verify() }
        // Another offset with the same digest is another claim.
        let wide = try write("wide", Array(new + old))
        try H3EmbeddingTable(url: wide, offset: 0, rows: 1, width: 2, byteCount: 4, sha256: h3SHA256(new)).verify()
        #expect(throws: AetherError.self) {
            try H3EmbeddingTable(url: wide, offset: 4, rows: 1, width: 2, byteCount: 4, sha256: h3SHA256(new)).verify()
        }
    }
}
