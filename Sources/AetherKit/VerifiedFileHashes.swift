import Darwin
import Foundation

/// Integrity checks remembered by file identity, so an unchanged multi-GB asset isn't read and hashed again on every
/// open: a language bundle's `main.mlirb` (about 2.9 s of a 4.1 s warm open), Krea's six model streams (2.7 s per
/// renderer), H3's model streams and the in-place weights its graphs stream.
///
/// A remembered result is reused only while every file it covers has exactly the identity it had when it was checked:
/// the resolved path, device, inode, size, modification time and status-change time (`st_ctimespec`, which the kernel
/// updates on every write, rename and attribute change and which no caller can set back). Anything else (a different
/// identity, a missing entry, a store that doesn't decode) checks again. A result is recorded only when the identities
/// read after the check equal those read before it, so a file changing during the check is never recorded.
///
/// Two kinds of record:
/// - a file's digest (``sha256(of:hash:)``), for checks that compare a computed hash with an expected one;
/// - a claim (``claim(_:files:)``): "these bytes matched this expected digest", over one or more files, for checks
///   that hash a file, a byte range or several ranges against a declared digest (H3 tensors read in place from
///   checkpoint shards). Its key is the expected digest and the files, so a range of a file is covered by the identity
///   of the whole file.
///
/// Records live in memory for the process and in `<Caches>/AetherKit/verified-file-hashes.json` (the app's own caches
/// directory, written atomically) so separate processes share them. The store is written at most about once a second
/// while records arrive, and at ``flush()``. It can be lost or deleted at any time; it only ever saves work.
package enum VerifiedFileHashes {
    package struct Identity: Codable, Equatable, Sendable {
        let device: Int64
        let inode: UInt64
        let size: Int64
        let modificationNanoseconds: Int64
        let statusChangeNanoseconds: Int64

        /// The identity of the regular file at `path` (following links, as reading does), nil when it can't be read.
        init?(path: String) {
            var info = stat()
            guard stat(path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { return nil }
            device = Int64(info.st_dev)
            inode = UInt64(info.st_ino)
            size = Int64(info.st_size)
            modificationNanoseconds = Int64(info.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(info.st_mtimespec.tv_nsec)
            statusChangeNanoseconds = Int64(info.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(info.st_ctimespec.tv_nsec)
        }
    }

    struct Entry: Codable, Equatable, Sendable {
        let identity: Identity
        let sha256: String
    }

    /// A pending claim: the key and the files' identities read before the check.
    package struct Claim: Sendable {
        let key: String
        let paths: [String]
        let identities: [Identity]
    }

    private struct Store: Codable {
        var schema = 2
        var entries: [String: Entry] = [:]
        var claims: [String: [Identity]] = [:]
    }

    private struct State {
        var store = Store()
        var loaded = false
        var dirty = false
        var lastWrite = Date.distantPast
    }

    /// Files this small cost less to hash than to look up; ``sha256(of:hash:)`` always hashes them.
    static let minimumCachedBytes: Int64 = 16 * 1024 * 1024
    static let maximumEntries = 256
    static let maximumClaims = 32_768

    private static let state = UnfairLock(State())

    /// The cache file; nil disables the on-disk store (memory still applies). Tests point it elsewhere.
    nonisolated(unsafe) static var storeURL: URL? = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
        .appendingPathComponent("AetherKit", isDirectory: true).appendingPathComponent("verified-file-hashes.json")

    // MARK: digests

    /// The file's SHA-256 (lowercase hex): a remembered digest when the file's identity is unchanged, otherwise
    /// `hash(url)`, remembered when the identity held still while hashing.
    package static func sha256(of url: URL, hash: (URL) throws -> String) throws -> String {
        let path = resolved(url)
        guard let before = Identity(path: path), before.size >= minimumCachedBytes else { return try hash(url) }
        if let entry = withStore({ $0.entries[path] }), entry.identity == before { return entry.sha256 }
        let digest = try hash(url)
        if Identity(path: path) == before {
            update { store in
                store.entries[path] = Entry(identity: before, sha256: digest)
                if store.entries.count > maximumEntries {
                    let keep = store.entries.sorted { $0.value.identity.statusChangeNanoseconds > $1.value.identity.statusChangeNanoseconds }
                    store.entries = Dictionary(uniqueKeysWithValues: keep.prefix(maximumEntries).map { ($0.key, $0.value) })
                }
            }
        }
        return digest
    }

    // MARK: claims

    /// A claim that `files` (in order; repeats allowed) hold bytes matching `digest`, with the files' identities now.
    /// Nil when a file can't be identified: then check without recording.
    package static func claim(_ digest: String, files: [URL]) -> Claim? {
        var paths: [String] = []
        for path in files.map(resolved) where !paths.contains(path) { paths.append(path) }
        var identities: [Identity] = []
        for path in paths {
            guard let identity = Identity(path: path) else { return nil }
            identities.append(identity)
        }
        return Claim(key: ([digest] + paths).joined(separator: "\u{0}"), paths: paths, identities: identities)
    }

    /// Whether the claim was checked before with these exact identities.
    package static func isVerified(_ claim: Claim) -> Bool {
        withStore { $0.claims[claim.key] } == claim.identities
    }

    /// Records a check that just passed, when the files' identities are still those the claim read before it.
    package static func recordVerified(_ claim: Claim) {
        guard claim.paths.map({ Identity(path: $0) }) == claim.identities.map(Optional.some) else { return }
        update { store in
            store.claims[claim.key] = claim.identities
            if store.claims.count > maximumClaims { store.claims.removeAll() }   // a runaway store starts over
        }
    }

    /// Runs `check` unless the claim over `files` is already verified, and records it when `check` returns.
    package static func verifyOnce(_ digest: String, files: [URL], _ check: () throws -> Void) throws {
        guard let claim = claim(digest, files: files) else { return try check() }
        if isVerified(claim) { return }
        try check()
        recordVerified(claim)
    }

    // MARK: store

    /// Writes pending records now (best effort).
    package static func flush() {
        guard let (data, url) = state.withLock({ state -> (Data, URL)? in
            guard state.dirty, let url = storeURL, let data = try? JSONEncoder().encode(state.store) else { return nil }
            state.dirty = false
            state.lastWrite = Date()
            return (data, url)
        }) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    /// Forgets everything in memory, so the next lookup reads the store again (tests).
    static func resetMemory() { state.withLock { $0 = State() } }

    private static func resolved(_ url: URL) -> String { url.standardizedFileURL.resolvingSymlinksInPath().path }

    private static func withStore<T>(_ body: (Store) -> T) -> T {
        state.withLock { state in
            load(&state)
            return body(state.store)
        }
    }

    private static func update(_ body: (inout Store) -> Void) {
        let due = state.withLock { state -> Bool in
            load(&state)
            body(&state.store)
            state.dirty = true
            return Date().timeIntervalSince(state.lastWrite) >= 1
        }
        if due { flush() }
    }

    /// Merges the store on disk into memory once per process (memory wins: it holds this process's newer checks).
    private static func load(_ state: inout State) {
        guard !state.loaded else { return }
        state.loaded = true
        guard let url = storeURL, let data = try? Data(contentsOf: url),
              let disk = try? JSONDecoder().decode(Store.self, from: data), disk.schema == 2 else { return }
        state.store.entries.merge(disk.entries) { mine, _ in mine }
        state.store.claims.merge(disk.claims) { mine, _ in mine }
    }
}
