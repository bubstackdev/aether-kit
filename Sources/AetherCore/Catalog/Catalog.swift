public import Foundation

/// The compiled-in model catalog (DESIGN.md §5). There is no runtime catalog fetch: what ships is what was tested.
///
/// `catalog/catalog.json` is the source; `scripts/gen-catalog` records per-file SHA-256 from the Hugging Face API
/// and emits `Catalog+Generated.swift` (``builtin``).
public struct Catalog: Sendable, Hashable, Codable {
    public static let currentSchema = 1

    public var schema: Int
    public var entries: [CatalogEntry]

    public init(schema: Int = Catalog.currentSchema, entries: [CatalogEntry]) {
        self.schema = schema
        self.entries = entries
    }

    public func entry(_ id: String) throws(AetherError) -> CatalogEntry {
        guard let entry = entries.first(where: { $0.id == id }) else { throw .unknownModel(id: id) }
        return entry
    }

    /// Decodes `catalog.json` (tests and tools; the app uses ``builtin``).
    public static func decode(_ data: Data) throws(AetherError) -> Catalog {
        try decodeSanitized(Catalog.self, from: data)
    }
}

/// One model. Each variant is one install: every asset it lists lands under `<repo>/<revision>/`.
public struct CatalogEntry: Sendable, Hashable, Codable, Identifiable {
    public var id: String
    public var task: String
    /// Hugging Face repo, `owner/name`.
    public var repo: String
    /// Full commit SHA the files were hashed at.
    public var revision: String
    public var license: String
    /// `platform` or `platform-arch` (`macos`, `ios`, `ios-h18p`) → what to install.
    public var variants: [String: CatalogVariant]

    public init(
        id: String, task: String, repo: String, revision: String, license: String, variants: [String: CatalogVariant]
    ) {
        self.id = id
        self.task = task
        self.repo = repo
        self.revision = revision
        self.license = license
        self.variants = variants
    }
}

public struct CatalogVariant: Sendable, Hashable, Codable {
    /// The driver this variant loads with (it may differ between variants of one model).
    public var driver: DriverID
    /// Asset name → subtree and files. `main` is required.
    public var assets: [String: CatalogAsset]
    /// When set, only these Core AI architectures may use the variant.
    public var architectures: [String]?

    public init(driver: DriverID, assets: [String: CatalogAsset], architectures: [String]? = nil) {
        self.driver = driver
        self.assets = assets
        self.architectures = architectures
    }

    /// Sum of every asset's file sizes.
    public var bytes: Int64 { assets.values.reduce(0) { $0 + $1.bytes } }
}

public struct CatalogAsset: Sendable, Hashable, Codable {
    /// Defaults to the entry's repo (a foreign tokenizer source sets its own, with a pinned revision).
    public var repo: String?
    public var revision: String?
    /// Directory in the repo that is the asset root; `""` is the repo root.
    public var subtree: String
    /// Paths relative to ``subtree``.
    public var files: [CatalogFile]

    public init(repo: String? = nil, revision: String? = nil, subtree: String, files: [CatalogFile]) {
        self.repo = repo
        self.revision = revision
        self.subtree = subtree
        self.files = files
    }

    public var bytes: Int64 { files.reduce(0) { $0 + $1.size } }
}

public struct CatalogFile: Sendable, Hashable, Codable {
    /// Path relative to the asset root.
    public var path: String
    public var size: Int64
    /// Lowercase hex SHA-256 of the contents (the HF LFS `oid` for LFS files).
    public var sha256: String

    public init(path: String, size: Int64, sha256: String) {
        self.path = path
        self.size = size
        self.sha256 = sha256
    }
}
