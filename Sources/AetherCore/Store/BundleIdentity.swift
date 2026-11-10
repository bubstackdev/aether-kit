import CryptoKit
import Foundation

// A bundle's location-free identity and its full verification (DESIGN.md §5). The identity is computed from
// declarations alone (the reference's file sizes and SHA-256s, and the manifest's content digest), so it costs no
// model-file reads; ``InstalledBundle/verify(progress:)`` is what proves the bytes on disk match those declarations.

/// How far a ``InstalledBundle/verify(progress:)`` has read.
public struct BundleVerificationProgress: Sendable, Hashable {
    public var completedBytes: Int64
    public var totalBytes: Int64
    /// The file being read, relative to the bundle root.
    public var file: String?

    public init(completedBytes: Int64, totalBytes: Int64, file: String?) {
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
        self.file = file
    }

    public var fractionCompleted: Double {
        totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 1
    }
}

extension InstalledBundle {
    /// The identity's format: part of every identity, so a future format can never reproduce an old one.
    static let identityFormat = "aether-bundle-identity@1"

    /// The bundle's content identity: lowercase hex SHA-256 over the driver, the variant key, every asset's files
    /// (name, path, size and SHA-256, sorted) and the manifest's ``Manifest/contentDigest``. The manifest's measured
    /// memory envelopes (`h3.envelopes`) are not part of the digest, so recording them keeps the identity.
    ///
    /// Location-free: the store root, the folder, the repo and the revision are not part of it, so the same bytes give
    /// the same identity wherever they are installed or restored, while any changed file or manifest gives a new one.
    /// It is computed from declarations; call ``verify(progress:)`` to check the files against them. Reads only the
    /// manifest. Throws what ``Manifest/load(fromBundle:)`` throws.
    public func contentIdentity() throws(AetherError) -> String {
        Self.identity(ref: ref, manifestDigest: try Manifest.load(fromBundle: url).contentDigest)
    }

    /// An identity for one capability of the bundle (for example, a pipeline mode that uses graphs the plain mode
    /// doesn't): the ``contentIdentity()`` bound to `capability`, so two capabilities of one bundle never share an
    /// identity and neither equals the plain one.
    public func contentIdentity(capability: String) throws(AetherError) -> String {
        Self.capabilityIdentity(try contentIdentity(), capability: capability)
    }

    static func identity(ref: ModelRef, manifestDigest: String) -> String {
        var hasher = SHA256()
        func field(_ value: String) {
            hasher.update(data: Data(value.utf8))
            hasher.update(data: Data([0]))
        }
        field(identityFormat)
        field(ref.driver.description)
        field(ref.variant)
        for asset in ref.assets.sorted(by: { $0.name < $1.name }) {
            field("asset")
            field(asset.name)
            for file in asset.files.sorted(by: { $0.path < $1.path }) {
                field(file.path)
                field(String(file.size))
                field(file.sha256.lowercased())
            }
        }
        field("manifest")
        field(manifestDigest)
        return hex(hasher.finalize())
    }

    static func capabilityIdentity(_ identity: String, capability: String) -> String {
        var hasher = SHA256()
        for part in [identityFormat, "capability", identity, capability] {
            hasher.update(data: Data(part.utf8))
            hasher.update(data: Data([0]))
        }
        return hex(hasher.finalize())
    }

    /// Reads every file the reference declares and checks its size and SHA-256, then that the manifest decodes:
    /// the full check a store's install does, for a bundle opened from any folder (``init(ref:url:isStale:)``), such as
    /// one restored from an archive. Tens of GB for a large model: run it off the main actor.
    ///
    /// `progress` is called as bytes are read (about every 64 MiB and at each file's end). Stops with
    /// `CancellationError` between reads once the task is cancelled. Throws ``AetherError/integrity(file:expected:actual:)``
    /// for a missing file (`actual` "missing"), a wrong size (`actual` "size N") or a wrong hash, and
    /// ``AetherError/invalidReference(field:reason:)`` for a reference with an unsafe path.
    public func verify(progress: ((BundleVerificationProgress) -> Void)? = nil) throws {
        try ref.validateForStore()
        let files = Self.verificationList(ref)
        let total = files.reduce(Int64(0)) { $0 + $1.file.size }
        var completed: Int64 = 0
        var reported: Int64 = 0
        progress?(BundleVerificationProgress(completedBytes: 0, totalBytes: total, file: nil))
        for item in files {
            let local = url.appending(path: item.local)
            guard let size = Self.fileSize(local) else {
                throw AetherError.integrity(file: item.local, expected: item.file.sha256, actual: "missing")
            }
            guard size == item.file.size else {
                throw AetherError.integrity(file: item.local, expected: "size \(item.file.size)", actual: "size \(size)")
            }
            let handle = try FileHandle(forReadingFrom: local)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
                try Task.checkCancellation()
                hasher.update(data: chunk)
                completed += Int64(chunk.count)
                if completed - reported >= 64 << 20 {
                    reported = completed
                    progress?(BundleVerificationProgress(completedBytes: completed, totalBytes: total, file: item.local))
                }
            }
            let actual = hex(hasher.finalize())
            guard actual == item.file.sha256.lowercased() else {
                throw AetherError.integrity(file: item.local, expected: item.file.sha256, actual: actual)
            }
            reported = completed
            progress?(BundleVerificationProgress(completedBytes: completed, totalBytes: total, file: item.local))
        }
        _ = try Manifest.load(fromBundle: url)
    }

    /// Each declared file once, by its path under the revision root (two assets of one repo may list the same file).
    static func verificationList(_ ref: ModelRef) -> [(local: String, file: CatalogFile)] {
        var seen = Set<String>()
        var list: [(local: String, file: CatalogFile)] = []
        for asset in ref.assets {
            let root = InstalledLayout.assetRoot(asset, mainRepo: ref.repo, in: URL(fileURLWithPath: "/"))
            for file in asset.files {
                let local = String(root.appending(path: file.path).path.dropFirst())
                if seen.insert(local).inserted { list.append((local, file)) }
            }
        }
        return list
    }

    static func fileSize(_ url: URL) -> Int64? {
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true, let size = values.fileSize else { return nil }
        return Int64(size)
    }
}

private func hex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
}
