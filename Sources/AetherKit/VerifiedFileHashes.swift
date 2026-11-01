import Darwin
import Foundation

/// SHA-256 digests of files a registration verified, remembered by file identity so an unchanged multi-GB asset isn't
/// read again on every open (a language bundle's 7.5–7.8 GB `main.mlirb` took about 2.9 s of a 4.1 s warm open).
///
/// An entry is reused only while the file's identity is exactly what it was when it was hashed: the resolved path,
/// device, inode, size, modification time and status-change time (`st_ctimespec`, which the kernel updates on every
/// write, rename and attribute change and which no caller can set back). Anything else (a different identity, a
/// missing or unreadable entry, a cache file that doesn't decode) re-hashes. An entry is stored only when the
/// identity read after hashing equals the one read before, so a file changing while it is hashed is never recorded.
///
/// Entries live in memory for the process and in `<Caches>/AetherKit/verified-file-hashes.json` (the app's own caches
/// directory, written atomically) so separate processes, such as a helper per request, share them. A cache can be lost
/// or deleted at any time; it only ever saves work. Writing it is best effort: a failure costs the next open a re-hash.
enum VerifiedFileHashes {
    struct Identity: Codable, Equatable, Sendable {
        let device: Int64
        let inode: UInt64
        let size: Int64
        let modificationNanoseconds: Int64
        let statusChangeNanoseconds: Int64

        /// The identity of the file at `path` (following links, as hashing does), nil when it can't be read.
        init?(path: String) {
            var info = stat()
            guard stat(path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { return nil }
            device = Int64(info.st_dev)
            inode = UInt64(info.st_ino)
            size = Int64(info.st_size)
            modificationNanoseconds = Int64(info.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(info.st_mtimespec.tv_nsec)
            statusChangeNanoseconds = Int64(info.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(info.st_ctimespec.tv_nsec)
        }

        init(device: Int64, inode: UInt64, size: Int64, modificationNanoseconds: Int64, statusChangeNanoseconds: Int64) {
            self.device = device
            self.inode = inode
            self.size = size
            self.modificationNanoseconds = modificationNanoseconds
            self.statusChangeNanoseconds = statusChangeNanoseconds
        }
    }

    struct Entry: Codable, Equatable, Sendable {
        let identity: Identity
        let sha256: String
    }

    private struct Store: Codable {
        var schema = 1
        var entries: [String: Entry] = [:]
    }

    /// Small files (tokenizers, templates) cost less to hash than to look up; only files this large are cached.
    static let minimumCachedBytes: Int64 = 16 * 1024 * 1024
    /// Entries kept on disk; beyond it, the oldest by status-change time are dropped.
    static let maximumEntries = 256

    private static let memory = UnfairLock([String: Entry]())

    /// The cache file; nil disables the on-disk cache (the in-memory one still applies). Tests point it elsewhere.
    nonisolated(unsafe) static var storeURL: URL? = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
        .appendingPathComponent("AetherKit", isDirectory: true).appendingPathComponent("verified-file-hashes.json")

    /// The file's SHA-256 (lowercase hex): a remembered digest when the file's identity is unchanged, otherwise
    /// `hash(url)`, remembered when the identity held still while hashing.
    static func sha256(of url: URL, hash: (URL) throws -> String) throws -> String {
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard let before = Identity(path: path), before.size >= minimumCachedBytes else { return try hash(url) }
        if let entry = lookup(path), entry.identity == before { return entry.sha256 }
        let digest = try hash(url)
        if Identity(path: path) == before { remember(Entry(identity: before, sha256: digest), for: path) }
        return digest
    }

    /// Forgets every in-memory entry (tests).
    static func resetMemory() { memory.withLock { $0.removeAll() } }

    private static func lookup(_ path: String) -> Entry? {
        if let entry = memory.withLock({ $0[path] }) { return entry }
        guard let entry = readStore()?.entries[path] else { return nil }
        memory.withLock { $0[path] = entry }
        return entry
    }

    private static func remember(_ entry: Entry, for path: String) {
        memory.withLock { $0[path] = entry }
        guard let storeURL else { return }
        var store = readStore() ?? Store()
        store.entries[path] = entry
        if store.entries.count > maximumEntries {
            let keep = store.entries.sorted { $0.value.identity.statusChangeNanoseconds > $1.value.identity.statusChangeNanoseconds }
                .prefix(maximumEntries)
            store.entries = Dictionary(uniqueKeysWithValues: keep.map { ($0.key, $0.value) })
        }
        guard let data = try? JSONEncoder().encode(store) else { return }
        try? FileManager.default.createDirectory(at: storeURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: storeURL, options: .atomic)
    }

    private static func readStore() -> Store? {
        guard let storeURL, let data = try? Data(contentsOf: storeURL),
              let store = try? JSONDecoder().decode(Store.self, from: data), store.schema == 1 else { return nil }
        return store
    }
}
