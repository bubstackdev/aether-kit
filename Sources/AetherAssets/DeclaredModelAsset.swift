import CryptoKit
import Foundation
import Synchronization

/// A Core AI asset directory inside a local bundle, with its declared content identity.
/// `main.hash` declares the SHA-256 of `main.mlirb`; `metadata.json` is hashed on opening.
/// A declaration is not verification of the stream bytes: call `verify()` explicitly.
public struct DeclaredModelAsset: Sendable, Equatable {
    public let url: URL
    public let declaredSHA256: String
    public let metadataSHA256: String

    /// Identity used to key qualified resource estimates: stream declaration plus metadata hash.
    public var identity: String { declaredSHA256 + ":" + metadataSHA256 }

    public init(url: URL, declaredSHA256: String, metadataSHA256: String) {
        self.url = url; self.declaredSHA256 = declaredSHA256; self.metadataSHA256 = metadataSHA256
    }

    /// Opens `relative` under `root`, reading the 32-byte declaration and hashing the small metadata.
    /// `pinned` maps bundle-relative paths to `sha256:<hex>` provenance; mismatches refuse.
    public static func open(_ relative: String, in root: URL, pinned: [String: String] = [:]) throws -> Self {
        let directory = try BundleFiles.confined(relative, in: root, directory: true)
        _ = try BundleFiles.confined(relative + "/main.mlirb", in: root)
        let declaration = try Data(contentsOf: BundleFiles.confined(relative + "/main.hash", in: root))
        guard declaration.count == 32 else {
            throw AssetStoreError.invalidManifest("main.hash must contain 32 bytes: \(relative)")
        }
        let metadataHash = try BundleFiles.sha256(of: BundleFiles.confined(relative + "/metadata.json", in: root))
        let declared = declaration.map { String(format: "%02x", $0) }.joined()
        if let value = pinned[relative + "/main.mlirb"], value != "sha256:" + declared {
            throw AssetStoreError.integrityMismatch("model declaration differs from provenance: \(relative)")
        }
        if let value = pinned[relative + "/metadata.json"], value != "sha256:" + metadataHash {
            throw AssetStoreError.integrityMismatch("model metadata differs from provenance: \(relative)")
        }
        return Self(url: directory, declaredSHA256: declared, metadataSHA256: metadataHash)
    }

    /// Streams the model through SHA-256 against its declaration. This can read several GB.
    public func verify() throws { try verify(checkStop: { try Task.checkCancellation() }) }

    /// `checkStop` runs between chunks; throw from it to stop hashing (cancellation or a superseded pass).
    /// Every file is resolved inside the asset directory; symlinks may not escape it.
    public func verify(checkStop: () throws -> Void) throws {
        let declaration = try Data(contentsOf: BundleFiles.confined("main.hash", in: url))
        guard declaration.map({ String(format: "%02x", $0) }).joined() == declaredSHA256,
              try BundleFiles.sha256(of: BundleFiles.confined("main.mlirb", in: url), checkStop: checkStop) == declaredSHA256,
              try BundleFiles.sha256(of: BundleFiles.confined("metadata.json", in: url), checkStop: checkStop) == metadataSHA256 else {
            throw AssetStoreError.integrityMismatch("model hash mismatch: \(url.lastPathComponent)")
        }
    }

    /// Verifies independent assets on concurrent threads. Workers observe the calling task's
    /// cancellation, and a failure stops only later assets, so the reported error is always the
    /// first failing asset in input order, as a serial pass would report it.
    public static func verifyConcurrently(_ assets: [DeclaredModelAsset]) throws {
        struct Superseded: Error {}
        // `isCancelled` is documented as safe from other threads, and concurrentPerform returns
        // before the task reference expires.
        struct CallerTask: @unchecked Sendable { let task: UnsafeCurrentTask? }
        let failures = Mutex<[Int: any Error]>([:])
        withUnsafeCurrentTask { task in
            let caller = CallerTask(task: task)
            DispatchQueue.concurrentPerform(iterations: assets.count) { index in
                do {
                    try assets[index].verify {
                        if caller.task?.isCancelled == true { throw CancellationError() }
                        if failures.withLock({ $0.keys.contains { $0 < index } }) { throw Superseded() }
                    }
                } catch {
                    failures.withLock { $0[index] = error }
                }
            }
        }
        if let first = failures.withLock({ $0.min { $0.key < $1.key }?.value }) { throw first }
    }

    /// The three portable files of this asset, relative to `root`, for an installation manifest.
    public func portableFiles(relativeTo root: URL) throws -> [AssetFile] {
        let base = root.standardizedFileURL.path
        guard url.path.hasPrefix(base + "/") else { throw AssetStoreError.unsafePath(url.path) }
        let relative = String(url.path.dropFirst(base.count + 1))
        return try [
            BundleFiles.portable(relative + "/main.mlirb", in: root, sha256: declaredSHA256),
            BundleFiles.portable(relative + "/metadata.json", in: root, sha256: metadataSHA256),
            BundleFiles.portable(relative + "/main.hash", in: root),
        ]
    }
}

/// Confined-path and hashing helpers for model bundles opened in place.
public enum BundleFiles {
    /// Resolves `relative` under `root`, refusing absolute paths, traversal and symlinks that escape the bundle.
    public static func confined(_ relative: String, in root: URL, directory: Bool = false) throws -> URL {
        guard AssetManifest.validPath(relative) else { throw AssetStoreError.unsafePath(relative) }
        let base = root.standardizedFileURL.resolvingSymlinksInPath()
        let url = base.appendingPathComponent(relative).standardizedFileURL.resolvingSymlinksInPath()
        guard url.pathComponents.starts(with: base.pathComponents), url.pathComponents.count > base.pathComponents.count else {
            throw AssetStoreError.unsafePath(relative)
        }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
        guard directory ? values.isDirectory == true : values.isRegularFile == true else {
            throw AssetStoreError.invalidManifest("missing \(directory ? "directory" : "file"): \(relative)")
        }
        return url
    }

    /// Streams a file through SHA-256 in autoreleased chunks. `checkStop` runs between chunks
    /// (by default, the current task's cancellation).
    public static func sha256(of url: URL, checkStop: () throws -> Void = { try Task.checkCancellation() }) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var digest = SHA256()
        while try autoreleasepool(invoking: { () throws -> Bool in
            guard let data = try file.read(upToCount: 4 << 20), !data.isEmpty else { return false }
            try checkStop()
            digest.update(data: data)
            return true
        }) {}
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// A portable file entry. Pass a known hash to avoid rereading large streams; the store verifies copies.
    public static func portable(_ relative: String, in root: URL, sha256 known: String? = nil) throws -> AssetFile {
        let url = try confined(relative, in: root)
        guard url.path == root.standardizedFileURL.resolvingSymlinksInPath().appendingPathComponent(relative).path else {
            throw AssetStoreError.unsafePath("portable installation requires direct file paths: \(relative)")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let count = attributes[.size] as? NSNumber else { throw AssetStoreError.invalidManifest("missing size: \(relative)") }
        return AssetFile(path: relative, byteCount: count.uint64Value, sha256: try known ?? sha256(of: url))
    }
}
