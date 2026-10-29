import Foundation

/// What an app prepares: the exact installation (every file's path, size and SHA-256) and where to fetch it.
/// The entry is the trust root: nothing it doesn't pin is installed, and its manifest is what `LocalAssetStore`
/// installs, so the installation ID is known before any byte is fetched (docs/model-preparation.md).
public struct ModelCatalogEntry: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let manifest: AssetManifest
    public let source: ModelSource
    public let displayName: String?

    public init(manifest: AssetManifest, source: ModelSource, displayName: String? = nil) throws {
        schemaVersion = 1
        self.manifest = manifest; self.source = source; self.displayName = displayName
        try validate()
    }

    private enum CodingKeys: String, CodingKey { case schemaVersion, manifest, source, displayName }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        manifest = try values.decode(AssetManifest.self, forKey: .manifest)
        source = try values.decode(ModelSource.self, forKey: .source)
        displayName = try values.decodeIfPresent(String.self, forKey: .displayName)
        try validate()
    }

    public func validate() throws {
        guard schemaVersion == 1 else { throw AssetStoreError.invalidManifest("unsupported catalog entry schema") }
        guard displayName.map({ !$0.isEmpty && $0.utf8.count <= 256 }) ?? true else {
            throw AssetStoreError.invalidManifest("invalid display name")
        }
        try manifest.validate()
        try source.validate()
        // Bounded so byte arithmetic (totals, free-space checks) can't overflow on a malformed entry.
        var total: UInt64 = 0
        for file in manifest.files {
            let (sum, overflow) = total.addingReportingOverflow(file.byteCount)
            guard !overflow, sum <= Self.maximumByteCount else { throw AssetStoreError.invalidManifest("entry exceeds the size limit") }
            total = sum
        }
    }

    /// 4 PiB: far above any model, far below UInt64 overflow in preparation's arithmetic.
    static let maximumByteCount: UInt64 = 1 << 52

    public var installationID: String { get throws { try manifest.installationID } }

    /// Total bytes of every file the entry installs.
    public var byteCount: UInt64 { manifest.files.reduce(0) { $0 + $1.byteCount } }

    /// Where to fetch `file`, which must be one of the entry's files.
    public func url(for file: AssetFile) throws -> URL {
        guard manifest.files.contains(file) else { throw AssetStoreError.invalidManifest("file not in entry: \(file.path)") }
        return try source.url(for: file.path)
    }
}

/// Where a catalog entry's files come from. Only immutable locations are accepted: HTTPS under a fixed base, or a
/// Hugging Face repository at a full commit revision (a branch or tag could change underneath the pinned hashes).
public enum ModelSource: Codable, Sendable, Equatable {
    case https(base: URL)
    case huggingFace(repository: String, revision: String, subdirectory: String?)

    private enum CodingKeys: String, CodingKey { case kind, base, repository, revision, subdirectory }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        switch try values.decode(String.self, forKey: .kind) {
        case "https": self = .https(base: try values.decode(URL.self, forKey: .base))
        case "huggingFace":
            self = .huggingFace(repository: try values.decode(String.self, forKey: .repository),
                                revision: try values.decode(String.self, forKey: .revision),
                                subdirectory: try values.decodeIfPresent(String.self, forKey: .subdirectory))
        case let kind: throw AssetStoreError.invalidManifest("unknown model source: \(kind)")
        }
        try validate()
    }

    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .https(base):
            try values.encode("https", forKey: .kind)
            try values.encode(base, forKey: .base)
        case let .huggingFace(repository, revision, subdirectory):
            try values.encode("huggingFace", forKey: .kind)
            try values.encode(repository, forKey: .repository)
            try values.encode(revision, forKey: .revision)
            try values.encodeIfPresent(subdirectory, forKey: .subdirectory)
        }
    }

    public func validate() throws {
        switch self {
        case let .https(base):
            guard base.scheme?.lowercased() == "https", let host = base.host, !host.isEmpty,
                  base.query == nil, base.fragment == nil, base.user == nil, base.password == nil else {
                throw AssetStoreError.invalidManifest("an HTTPS source needs an https URL without query, fragment or credentials")
            }
        case let .huggingFace(repository, revision, subdirectory):
            let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 2, parts.allSatisfy({ part in
                !part.isEmpty && part.utf8.count <= 96 && part != "." && part != ".."
                    && part.utf8.allSatisfy { Self.repositoryCharacter($0) }
            }) else {
                throw AssetStoreError.invalidManifest("a Hugging Face repository is owner/name")
            }
            guard revision.utf8.count == 40, revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw AssetStoreError.invalidManifest("a Hugging Face revision must be a full 40-character commit hash")
            }
            guard subdirectory.map(AssetManifest.validPath) ?? true else {
                throw AssetStoreError.invalidManifest("invalid Hugging Face subdirectory")
            }
        }
    }

    /// The download URL of a manifest-relative path.
    public func url(for path: String) throws -> URL {
        try validate()
        guard AssetManifest.validPath(path) else { throw AssetStoreError.unsafePath(path) }
        let relative: String
        let base: URL
        switch self {
        case let .https(root):
            base = root; relative = path
        case let .huggingFace(repository, revision, subdirectory):
            base = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/")!
            relative = subdirectory.map { $0 + "/" + path } ?? path
        }
        var url = base
        for component in relative.split(separator: "/") {
            url.appendPathComponent(String(component), isDirectory: false)
        }
        return url
    }

    private static func repositoryCharacter(_ byte: UInt8) -> Bool {
        (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte) || byte == 45 || byte == 46 || byte == 95
    }
}
