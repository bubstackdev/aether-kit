@testable import AetherKit
import Darwin
import Foundation
import Testing

/// The verified-hash cache reuses a digest only while the file's identity is unchanged, and re-hashes on anything else.
@Suite(.serialized) struct VerifiedFileHashesTests {
    /// A counting hasher over a fresh directory, with the on-disk store inside it.
    private final class Fixture {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("verified-hashes-\(UUID().uuidString)")
        var hashes = 0
        let savedStore = VerifiedFileHashes.storeURL

        init() throws {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            VerifiedFileHashes.storeURL = directory.appendingPathComponent("store/verified-file-hashes.json")
            VerifiedFileHashes.resetMemory()
        }

        deinit {
            VerifiedFileHashes.storeURL = savedStore
            VerifiedFileHashes.resetMemory()
            try? FileManager.default.removeItem(at: directory)
        }

        func hash(_ url: URL) throws -> String {
            try VerifiedFileHashes.sha256(of: url) { url in
                self.hashes += 1
                return try LanguageModelManifest.fileHash(url)
            }
        }

        func write(_ name: String, byte: UInt8, count: Int = Int(VerifiedFileHashes.minimumCachedBytes)) throws -> URL {
            let url = directory.appendingPathComponent(name)
            try Data(repeating: byte, count: count).write(to: url)
            return url
        }
    }

    @Test func unchangedFileIsHashedOnceAcrossProcessesWorthOfMemory() throws {
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        let digest = try f.hash(url)
        #expect(digest == (try LanguageModelManifest.fileHash(url)))
        #expect(try f.hash(url) == digest)
        #expect(f.hashes == 1)
        // A new process: memory is empty, the store on disk still holds the entry.
        VerifiedFileHashes.resetMemory()
        #expect(try f.hash(url) == digest)
        #expect(f.hashes == 1)
    }

    @Test func changedContentWithRestoredTimesIsHashedAgain() throws {
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        let first = try f.hash(url)
        var before = stat()
        #expect(stat(url.path, &before) == 0)
        // Same size, then the modification time set back: the status-change time still moves.
        let handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Data([2]))
        try handle.close()
        var times = [timespec(tv_sec: before.st_atimespec.tv_sec, tv_nsec: before.st_atimespec.tv_nsec),
                     timespec(tv_sec: before.st_mtimespec.tv_sec, tv_nsec: before.st_mtimespec.tv_nsec)]
        #expect(utimensat(AT_FDCWD, url.path, &times, 0) == 0)
        let second = try f.hash(url)
        #expect(second != first)
        #expect(second == (try LanguageModelManifest.fileHash(url)))
        #expect(f.hashes == 2)
    }

    @Test func replacedFileIsHashedAgain() throws {
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        _ = try f.hash(url)
        let replacement = try f.write("replacement.bin", byte: 3)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: replacement)
        #expect(try f.hash(url) == (try LanguageModelManifest.fileHash(url)))
        #expect(f.hashes == 2)
    }

    @Test func smallFilesAndUnreadableStoresAlwaysHash() throws {
        let f = try Fixture()
        let small = try f.write("tokenizer.json", byte: 4, count: 1024)
        _ = try f.hash(small)
        _ = try f.hash(small)
        #expect(f.hashes == 2)
        let url = try f.write("model.bin", byte: 5)
        _ = try f.hash(url)
        VerifiedFileHashes.resetMemory()
        try Data("not json".utf8).write(to: try #require(VerifiedFileHashes.storeURL))
        _ = try f.hash(url)
        #expect(f.hashes == 4)
    }

    @Test func manifestVerificationUsesTheCache() throws {
        let f = try Fixture()
        let model = f.directory.appendingPathComponent("model.aimodel/main.mlirb")
        try FileManager.default.createDirectory(at: model.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 6, count: Int(VerifiedFileHashes.minimumCachedBytes)).write(to: model)
        let first = try LanguageModelManifest.verifiedHash(model)
        #expect(first == (try LanguageModelManifest.fileHash(model)))
        let entry = try JSONSerialization.jsonObject(with: Data(contentsOf: try #require(VerifiedFileHashes.storeURL))) as? [String: Any]
        let entries = try #require(entry?["entries"] as? [String: Any])
        #expect(entries.keys.contains(model.standardizedFileURL.resolvingSymlinksInPath().path))
    }

    @Test func claimsCoverEveryFileAndRecordOnlyPassingChecks() throws {
        let f = try Fixture()
        let a = try f.write("a.bin", byte: 1, count: 1024), b = try f.write("b.bin", byte: 2, count: 1024)
        var checks = 0
        var read: [URL] = []
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b, a]) { read = $0; checks += 1 }
        #expect(read.map(\.path) == [a, b].map { $0.resolvingSymlinksInPath().path })
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { _ in checks += 1 }
        #expect(checks == 1)
        // Another digest over the same files is another claim.
        try VerifiedFileHashes.verifyOnce("other", files: [a, b]) { _ in checks += 1 }
        #expect(checks == 2)
        // A failing check records nothing.
        struct Mismatch: Error {}
        #expect(throws: Mismatch.self) { try VerifiedFileHashes.verifyOnce("bad", files: [a]) { _ in throw Mismatch() } }
        #expect(throws: Mismatch.self) { try VerifiedFileHashes.verifyOnce("bad", files: [a]) { _ in throw Mismatch() } }
        // Any covered file changing (here the second) checks again, also in a new process.
        let handle = try FileHandle(forWritingTo: b)
        try handle.write(contentsOf: Data([9]))
        try handle.close()
        VerifiedFileHashes.flush()
        VerifiedFileHashes.resetMemory()
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { _ in checks += 1 }
        #expect(checks == 3)
        VerifiedFileHashes.flush()
        VerifiedFileHashes.resetMemory()
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { _ in checks += 1 }
        #expect(checks == 3)
        // A file that can't be identified is checked every time.
        let missing = f.directory.appendingPathComponent("missing.bin")
        try VerifiedFileHashes.verifyOnce("digest", files: [missing]) { _ in checks += 1 }
        try VerifiedFileHashes.verifyOnce("digest", files: [missing]) { _ in checks += 1 }
        #expect(checks == 5)
    }

    // MARK: review of #83

    @Test func symlinkRetargetedAfterResolvingHashesTheResolvedFile() throws {
        let f = try Fixture()
        let a = try f.write("a.bin", byte: 1), b = try f.write("b.bin", byte: 2)
        let link = f.directory.appendingPathComponent("model.bin")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: a)
        let linkPath = link.path, bPath = b.path
        var hooks = VerifiedFileHashes.Hooks()
        hooks.afterResolving = { _ in unlink(linkPath); symlink(bPath, linkPath) }
        // Resolved to a.bin, then the link moves to b.bin: the digest is a.bin's, recorded for a.bin.
        let first = try VerifiedFileHashes.$hooks.withValue(hooks) { try f.hash(link) }
        #expect(first == (try LanguageModelManifest.fileHash(a)))
        #expect(try f.hash(link) == (try LanguageModelManifest.fileHash(b)))
        #expect(try f.hash(a) == first)
        #expect(f.hashes == 2)
    }

    @Test func fileChangedWhileHashingIsNotRecorded() throws {
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        var hashes = 0
        let digest = try VerifiedFileHashes.sha256(of: url) { url in
            hashes += 1
            let digest = try LanguageModelManifest.fileHash(url)
            let handle = try FileHandle(forWritingTo: url)
            try handle.write(contentsOf: Data([7]))
            try handle.close()
            return digest
        }
        #expect(digest != (try LanguageModelManifest.fileHash(url)))
        #expect(try f.hash(url) == (try LanguageModelManifest.fileHash(url)))
        #expect(f.hashes == 1)
        // A claim whose file changes during the check isn't recorded either.
        var checks = 0
        try VerifiedFileHashes.verifyOnce("digest", files: [url]) { files in
            checks += 1
            let handle = try FileHandle(forWritingTo: files[0])
            try handle.write(contentsOf: Data([8]))
            try handle.close()
        }
        try VerifiedFileHashes.verifyOnce("digest", files: [url]) { _ in checks += 1 }
        #expect(checks == 2)
    }

    @Test func forgedMismatchedAndOtherSchemaStoresAreIgnored() throws {
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        let identity = try #require(VerifiedFileHashes.Identity(path: path))
        let forged = String(repeating: "0", count: 64)
        let store = try #require(VerifiedFileHashes.storeURL)
        try FileManager.default.createDirectory(at: store.deletingLastPathComponent(), withIntermediateDirectories: true)
        func plant(_ entry: VerifiedFileHashes.Entry, schema: Int = VerifiedFileHashes.Store.currentSchema) throws {
            var disk = VerifiedFileHashes.Store()
            disk.schema = schema
            disk.entries[path] = entry
            try JSONEncoder().encode(disk).write(to: store)
            VerifiedFileHashes.resetMemory()
        }
        let real = try LanguageModelManifest.fileHash(url)
        // Another file's identity (inode, then ctime), a digest that isn't a SHA-256, another schema: all re-hash.
        let moved = [
            VerifiedFileHashes.Identity(device: identity.device, inode: identity.inode + 1, size: identity.size,
                                        modificationNanoseconds: identity.modificationNanoseconds,
                                        statusChangeNanoseconds: identity.statusChangeNanoseconds),
            VerifiedFileHashes.Identity(device: identity.device, inode: identity.inode, size: identity.size,
                                        modificationNanoseconds: identity.modificationNanoseconds,
                                        statusChangeNanoseconds: identity.statusChangeNanoseconds - 1),
        ]
        for other in moved {
            try plant(.init(identity: other, sha256: forged))
            #expect(try f.hash(url) == real)
        }
        try plant(.init(identity: identity, sha256: "not a digest"))
        #expect(try f.hash(url) == real)
        try plant(.init(identity: identity, sha256: forged), schema: 2)
        #expect(try f.hash(url) == real)
        #expect(f.hashes == 4)
        // A claim planted under a store of another schema is ignored too.
        let a = try f.write("a.bin", byte: 3, count: 1024)
        let claim = try #require(VerifiedFileHashes.claim("digest", files: [a]))
        var disk = VerifiedFileHashes.Store()
        disk.schema = 2
        disk.claims[claim.key] = claim.identities
        try JSONEncoder().encode(disk).write(to: store)
        VerifiedFileHashes.resetMemory()
        #expect(!VerifiedFileHashes.isVerified(claim))
    }

    @Test func untrustedVolumesAlwaysCheck() throws {
        #expect(VerifiedFileHashes.isTrustedVolume(type: "apfs", flags: UInt32(MNT_LOCAL)))
        #expect(VerifiedFileHashes.isTrustedVolume(type: "hfs", flags: UInt32(MNT_LOCAL | MNT_RDONLY)))
        #expect(!VerifiedFileHashes.isTrustedVolume(type: "apfs", flags: 0))
        for type in ["smbfs", "nfs", "afpfs", "webdav", "macfuse", "exfat", "msdos"] {
            #expect(!VerifiedFileHashes.isTrustedVolume(type: type, flags: UInt32(MNT_LOCAL)))
        }
        let f = try Fixture()
        let url = try f.write("model.bin", byte: 1)
        let small = try f.write("a.bin", byte: 2, count: 1024)
        let fd = open(small.path, O_RDONLY)
        defer { close(fd) }
        #expect(VerifiedFileHashes.Identity(descriptor: fd) != nil)
        let untrusted: [@Sendable (String, UInt32) -> Bool] = [{ _, _ in false }, { type, _ in type == "smbfs" }]
        for check in untrusted {
            var hooks = VerifiedFileHashes.Hooks()
            hooks.trustsVolume = check
            var checks = 0
            try VerifiedFileHashes.$hooks.withValue(hooks) { () throws -> Void in
                let identity = VerifiedFileHashes.Identity(descriptor: fd)
                #expect(identity == nil)
                _ = try f.hash(url)
                _ = try f.hash(url)
                try VerifiedFileHashes.verifyOnce("digest", files: [small]) { _ in checks += 1 }
                try VerifiedFileHashes.verifyOnce("digest", files: [small]) { _ in checks += 1 }
            }
            #expect(checks == 2)
        }
        #expect(f.hashes == 4)
        // Nothing was recorded while the volume wasn't trusted.
        _ = try f.hash(url)
        #expect(f.hashes == 5)
    }

    @Test func entryAndClaimLimits() throws {
        func identity(_ n: Int64) -> VerifiedFileHashes.Identity {
            .init(device: 1, inode: UInt64(n), size: 1, modificationNanoseconds: n, statusChangeNanoseconds: n)
        }
        var store = VerifiedFileHashes.Store()
        for n in 0..<Int64(VerifiedFileHashes.maximumEntries + 10) {
            store.insert(.init(identity: identity(n), sha256: String(repeating: "a", count: 64)), at: "/f\(n)")
        }
        #expect(store.entries.count == VerifiedFileHashes.maximumEntries)
        // The most recently changed files stay.
        #expect(store.entries["/f\(VerifiedFileHashes.maximumEntries + 9)"] != nil && store.entries["/f9"] == nil)
        #expect(store.entries["/f10"] != nil)
        for n in 0..<Int64(VerifiedFileHashes.maximumClaims) { store.insert(claim: [identity(n)], at: "c\(n)") }
        #expect(store.claims.count == VerifiedFileHashes.maximumClaims)
        store.insert(claim: [identity(1_000_000)], at: "last")
        #expect(store.claims.count == 1 && store.claims["last"] == [identity(1_000_000)])
    }

    @Test func descriptorClaimsNeedEveryIdentity() throws {
        let f = try Fixture()
        let a = try f.write("a.bin", byte: 1, count: 1024)
        let fd = open(a.path, O_RDONLY)
        defer { close(fd) }
        #expect(VerifiedFileHashes.claim("k", identities: []) == nil)
        #expect(VerifiedFileHashes.claim("k", identities: [VerifiedFileHashes.Identity(descriptor: fd), nil]) == nil)
        let claim = try #require(VerifiedFileHashes.claim("k", identities: [VerifiedFileHashes.Identity(descriptor: fd)]))
        #expect(!VerifiedFileHashes.isVerified(claim))
        VerifiedFileHashes.recordVerified(claim, current: [nil])
        #expect(!VerifiedFileHashes.isVerified(claim))
        VerifiedFileHashes.recordVerified(claim, current: [VerifiedFileHashes.Identity(descriptor: fd)])
        #expect(VerifiedFileHashes.isVerified(claim))
    }

    @Test func testProcessesNeverUseTheUsersStore() throws {
        #expect(VerifiedFileHashes.isTestProcess)
        let url = try #require(VerifiedFileHashes.defaultStoreURL())
        #expect(url.path.hasPrefix(FileManager.default.temporaryDirectory.path))
        #expect(!url.path.contains("/Library/Caches/"))
    }
}
