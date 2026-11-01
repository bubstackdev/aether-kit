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
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b, a]) { checks += 1 }
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { checks += 1 }
        #expect(checks == 1)
        // Another digest over the same files is another claim.
        try VerifiedFileHashes.verifyOnce("other", files: [a, b]) { checks += 1 }
        #expect(checks == 2)
        // A failing check records nothing.
        struct Mismatch: Error {}
        #expect(throws: Mismatch.self) { try VerifiedFileHashes.verifyOnce("bad", files: [a]) { throw Mismatch() } }
        #expect(throws: Mismatch.self) { try VerifiedFileHashes.verifyOnce("bad", files: [a]) { throw Mismatch() } }
        // Any covered file changing (here the second) checks again, also in a new process.
        let handle = try FileHandle(forWritingTo: b)
        try handle.write(contentsOf: Data([9]))
        try handle.close()
        VerifiedFileHashes.flush()
        VerifiedFileHashes.resetMemory()
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { checks += 1 }
        #expect(checks == 3)
        VerifiedFileHashes.flush()
        VerifiedFileHashes.resetMemory()
        try VerifiedFileHashes.verifyOnce("digest", files: [a, b]) { checks += 1 }
        #expect(checks == 3)
        // A file that can't be identified is checked every time.
        let missing = f.directory.appendingPathComponent("missing.bin")
        try VerifiedFileHashes.verifyOnce("digest", files: [missing]) { checks += 1 }
        try VerifiedFileHashes.verifyOnce("digest", files: [missing]) { checks += 1 }
        #expect(checks == 5)
    }
}
