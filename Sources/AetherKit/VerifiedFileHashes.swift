import Darwin
import Foundation
import MachO

/// Integrity checks remembered by file identity, so an unchanged multi-GB asset isn't read and hashed again on every
/// open: a language bundle's `main.mlirb` (about 2.9 s of a 4.1 s warm open), Krea's six model streams (2.7 s per
/// renderer), H3's model streams and the in-place weights its graphs stream.
///
/// A remembered result is reused only while every file it covers has exactly the identity it had when it was checked:
/// device, inode, size, modification time and status-change time (`st_ctimespec`, which the kernel updates on every
/// write, rename and attribute change and which no caller can set back). Anything else (a different identity, a missing
/// entry, a store that doesn't decode or has another schema) checks again. A result is recorded only when the identities
/// read after the check equal those read before it, so a file changing during the check is never recorded.
///
/// An identity is always taken from the bytes' own source: ``Identity/init(descriptor:)`` (`fstat`) on the descriptor
/// a check or a cached read reads from, or, for path checks, ``Identity/init(path:)`` (`lstat`, refusing a link) on the
/// fully resolved path that is then hashed. Only files on a local APFS or HFS+ volume (`statfs`: `MNT_LOCAL` and the
/// file system type) are remembered; anything else (network, FUSE, exFAT, MS-DOS) always hashes.
///
/// Two kinds of record:
/// - a file's digest (``sha256(of:hash:)``), for checks that compare a computed hash with an expected one;
/// - a claim (``claim(_:files:)``, ``claim(_:identities:)``): "these bytes matched this expected digest", for checks
///   that hash a file, a byte range or several ranges against a declared digest (H3 tensors read in place from
///   checkpoint shards). Its key is chosen by the caller and must name everything the check covered: the expected
///   digest, the files, and every byte range in order. A range of a file is covered by the identity of the whole file.
///
/// Records live in memory for the process and in `~/Library/Caches/AetherKit/verified-file-hashes.json` (the user's
/// caches directory; inside the app's container when sandboxed), plain JSON written atomically, so separate processes
/// share them. The store is written at most about once a second while records arrive, and at ``flush()``.
///
/// Trust boundary: the store is not signed. It sits inside the same boundary as the bundle manifests and `main.hash`
/// declarations that name the expected digests: it is per user, and a process that can write it can already write
/// those manifests (and the model files themselves). A forged record can only claim what a forged manifest could.
/// The store is safe to delete at any time; it only ever saves work. Processes writing it concurrently are last writer
/// wins, which costs only re-hashing what the lost records covered. Writes through a `MAP_SHARED` mapping can update a
/// file's modification and status-change times late (at `msync` or unmap), so a file being written through a mapping
/// must not be verified while that is going on; model files are written once and then only read.
package enum VerifiedFileHashes {
    package struct Identity: Codable, Equatable, Sendable {
        let device: Int64
        let inode: UInt64
        let size: Int64
        let modificationNanoseconds: Int64
        let statusChangeNanoseconds: Int64

        /// The identity of the regular file at `path` itself (`lstat`: a link is refused, so pass a resolved path),
        /// nil when it can't be read or isn't on a volume the cache trusts.
        package init?(path: String) {
            var info = stat(), volume = statfs()
            guard lstat(path, &info) == 0, statfs(path, &volume) == 0 else { return nil }
            self.init(info, volume)
        }

        /// The identity of the open file `descriptor` (`fstat`): the file its bytes are read from, whatever its path
        /// names now. Nil when it isn't a regular file on a volume the cache trusts.
        package init?(descriptor: Int32) {
            var info = stat(), volume = statfs()
            guard fstat(descriptor, &info) == 0, fstatfs(descriptor, &volume) == 0 else { return nil }
            self.init(info, volume)
        }

        private init?(_ info: stat, _ volume: statfs) {
            guard (info.st_mode & S_IFMT) == S_IFREG, VerifiedFileHashes.isTrustedVolume(volume) else { return nil }
            device = Int64(info.st_dev)
            inode = UInt64(info.st_ino)
            size = Int64(info.st_size)
            modificationNanoseconds = Int64(info.st_mtimespec.tv_sec) * 1_000_000_000 + Int64(info.st_mtimespec.tv_nsec)
            statusChangeNanoseconds = Int64(info.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(info.st_ctimespec.tv_nsec)
        }

        /// Tests: an identity with these fields.
        init(device: Int64, inode: UInt64, size: Int64, modificationNanoseconds: Int64, statusChangeNanoseconds: Int64) {
            self.device = device; self.inode = inode; self.size = size
            self.modificationNanoseconds = modificationNanoseconds; self.statusChangeNanoseconds = statusChangeNanoseconds
        }
    }

    struct Entry: Codable, Equatable, Sendable {
        let identity: Identity
        let sha256: String
    }

    /// A pending claim: the key, the files a path claim reads (resolved), and the identities read before the check.
    package struct Claim: Sendable {
        let key: String
        /// The resolved files of a path claim (``claim(_:files:)``), deduplicated in order: the check must read these.
        package let urls: [URL]
        let identities: [Identity]

        /// Whether `current` (read the same way as the claim's identities) still equals them.
        package func holds(_ current: [Identity?]) -> Bool { current == identities.map(Optional.some) }
    }

    /// Test hooks, scoped to the task that sets them (`$hooks.withValue`), so concurrent suites never see them.
    package struct Hooks: Sendable {
        /// Is the volume one the cache trusts? Replaces the `statfs` check: (file system type, mount flags).
        package var trustsVolume: (@Sendable (String, UInt32) -> Bool)? = nil
        /// Runs after ``sha256(of:hash:)`` resolves a path, before it identifies and hashes it.
        package var afterResolving: (@Sendable (String) -> Void)? = nil
        /// Runs after a cached read (a claim hit) copied its bytes, before the identities are read again.
        package var afterCachedRead: (@Sendable () -> Void)? = nil
        package init() {}
    }
    @TaskLocal package static var hooks = Hooks()

    struct Store: Codable {
        static let currentSchema = 3
        var schema = Self.currentSchema
        var entries: [String: Entry] = [:]
        var claims: [String: [Identity]] = [:]

        /// Records a digest, keeping the `limit` most recently changed files.
        mutating func insert(_ entry: Entry, at path: String, limit: Int = VerifiedFileHashes.maximumEntries) {
            entries[path] = entry
            guard entries.count > limit else { return }
            let keep = entries.sorted { $0.value.identity.statusChangeNanoseconds > $1.value.identity.statusChangeNanoseconds }
            entries = Dictionary(uniqueKeysWithValues: keep.prefix(limit).map { ($0.key, $0.value) })
        }

        /// Records a claim; a store past `limit` claims starts over (a runaway store, not a working set).
        mutating func insert(claim identities: [Identity], at key: String, limit: Int = VerifiedFileHashes.maximumClaims) {
            claims[key] = identities
            if claims.count > limit { claims = [key: identities] }
        }
    }

    private struct State {
        var store = Store()
        var loaded = false
        var dirty = false
        var lastWrite = Date.distantPast
        var storeURL: URL? = VerifiedFileHashes.defaultStoreURL()
    }

    /// Files this small cost less to hash than to look up; ``sha256(of:hash:)`` always hashes them.
    static let minimumCachedBytes: Int64 = 16 * 1024 * 1024
    static let maximumEntries = 256
    static let maximumClaims = 32_768

    private static let state = UnfairLock(State())

    /// The cache file; nil disables the on-disk store (memory still applies).
    static var storeURL: URL? {
        get { state.withLock { $0.storeURL } }
        set { state.withLock { $0.storeURL = newValue } }
    }

    /// `~/Library/Caches/AetherKit/verified-file-hashes.json`; in a test process (an `.xctest` bundle is loaded), a
    /// store in a per-process temporary directory instead, so no test run reads or writes the user's real store.
    static func defaultStoreURL() -> URL? {
        let directory = isTestProcess
            ? FileManager.default.temporaryDirectory.appendingPathComponent("AetherKit-tests-\(getpid())", isDirectory: true)
            : FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?.appendingPathComponent("AetherKit", isDirectory: true)
        return directory?.appendingPathComponent("verified-file-hashes.json")
    }

    static let isTestProcess: Bool = (0..<_dyld_image_count()).contains { index in
        guard let name = _dyld_get_image_name(index) else { return false }
        return String(cString: name).contains(".xctest/")
    }

    /// Only local APFS and HFS+ volumes keep the guarantees identities rely on (stable inodes, kernel-maintained
    /// status-change times). Network, FUSE and other file systems always hash.
    static func isTrustedVolume(_ volume: statfs) -> Bool {
        var name = volume.f_fstypename
        let type = withUnsafeBytes(of: &name) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        if let check = hooks.trustsVolume { return check(type, volume.f_flags) }
        return isTrustedVolume(type: type, flags: volume.f_flags)
    }

    static func isTrustedVolume(type: String, flags: UInt32) -> Bool {
        flags & UInt32(MNT_LOCAL) != 0 && (type == "apfs" || type == "hfs")
    }

    // MARK: digests

    /// The file's SHA-256 (lowercase hex): a remembered digest when the file's identity is unchanged, otherwise
    /// `hash` of the resolved path (the same file whose identity is recorded), remembered when the identity held
    /// still while hashing.
    package static func sha256(of url: URL, hash: (URL) throws -> String) throws -> String {
        let path = resolved(url)
        hooks.afterResolving?(path)
        let file = URL(fileURLWithPath: path)
        guard let before = Identity(path: path), before.size >= minimumCachedBytes else { return try hash(file) }
        if let entry = withStore({ $0.entries[path] }), entry.identity == before { return entry.sha256 }
        let digest = try hash(file)
        if isSHA256(digest), Identity(path: path) == before {
            update { $0.insert(Entry(identity: before, sha256: digest), at: path) }
        }
        return digest
    }

    // MARK: claims

    /// A claim over `files` (resolved, deduplicated in order) with their identities now; the check must read
    /// ``Claim/urls``. `key` must name everything the check covers (digest, ranges); the paths are appended to it.
    /// Nil when a file can't be identified or isn't on a trusted volume: then check without recording.
    package static func claim(_ key: String, files: [URL]) -> Claim? {
        var paths: [String] = []
        for path in files.map(resolved) where !paths.contains(path) { paths.append(path) }
        var identities: [Identity] = []
        for path in paths {
            guard let identity = Identity(path: path) else { return nil }
            identities.append(identity)
        }
        return Claim(key: ([key] + paths).joined(separator: "\u{0}"), urls: paths.map { URL(fileURLWithPath: $0) },
                     identities: identities)
    }

    /// A claim whose identities the caller read from the descriptors it reads the bytes from. `key` must name
    /// everything the check covers: the digest, the files and every byte range in order. Nil when any is nil.
    package static func claim(_ key: String, identities: [Identity?]) -> Claim? {
        let known = identities.compactMap { $0 }
        guard !known.isEmpty, known.count == identities.count else { return nil }
        return Claim(key: key, urls: [], identities: known)
    }

    /// Whether the claim was checked before with these exact identities.
    package static func isVerified(_ claim: Claim) -> Bool {
        withStore { $0.claims[claim.key] } == claim.identities
    }

    /// Records a check that just passed, when the files' identities now (`current`, read the same way as the claim's)
    /// are still those the claim read before it.
    package static func recordVerified(_ claim: Claim, current: [Identity?]) {
        guard claim.holds(current) else { return }
        update { $0.insert(claim: claim.identities, at: claim.key) }
    }

    /// Records a path claim's passing check, when its files' identities are unchanged.
    package static func recordVerified(_ claim: Claim) {
        recordVerified(claim, current: claim.urls.map { Identity(path: $0.path) })
    }

    /// Runs `check` on the claim's resolved files unless the claim over `files` is already verified, and records it
    /// when `check` returns with the files unchanged. Without a claim, `check` gets `files` as given.
    package static func verifyOnce(_ key: String, files: [URL], _ check: ([URL]) throws -> Void) throws {
        guard let claim = claim(key, files: files) else { return try check(files) }
        if isVerified(claim) { return }
        try check(claim.urls)
        recordVerified(claim)
    }

    // MARK: store

    /// Writes pending records now (best effort).
    package static func flush() {
        guard let (data, url) = state.withLock({ state -> (Data, URL)? in
            guard state.dirty, let url = state.storeURL, let data = try? JSONEncoder().encode(state.store) else { return nil }
            state.dirty = false
            state.lastWrite = Date()
            return (data, url)
        }) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    /// Forgets everything in memory, so the next lookup reads the store again (tests). Keeps ``storeURL``.
    static func resetMemory() { state.withLock { $0 = State(storeURL: $0.storeURL) } }

    /// The path with every link resolved (as `BundleFiles.confined` resolves it).
    private static func resolved(_ url: URL) -> String { url.standardizedFileURL.resolvingSymlinksInPath().path }

    private static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

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
    /// A store of another schema, or entries whose digest isn't a SHA-256, are ignored.
    private static func load(_ state: inout State) {
        guard !state.loaded else { return }
        state.loaded = true
        guard let url = state.storeURL, let data = try? Data(contentsOf: url),
              let disk = try? JSONDecoder().decode(Store.self, from: data), disk.schema == Store.currentSchema else { return }
        state.store.entries.merge(disk.entries.filter { isSHA256($0.value.sha256) }) { mine, _ in mine }
        state.store.claims.merge(disk.claims) { mine, _ in mine }
    }
}
