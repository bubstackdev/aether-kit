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

    /// Decodes `catalog.json` (used by tests and tools; the app uses ``builtin``).
    public static func decode(_ data: Data) throws(AetherError) -> Catalog {
        do {
            return try JSONDecoder().decode(Catalog.self, from: data)
        } catch let error as DecodingError {
            throw AetherError(decoding: error)
        } catch {
            throw AetherError.manifestInvalid(field: "", reason: "\(error)")
        }
    }
}

public struct CatalogEntry: Sendable, Hashable, Codable, Identifiable {
    public var id: String
    public var task: String
    public var driver: DriverID
    /// Hugging Face repo, `owner/name`.
    public var repo: String
    /// Full commit SHA the files were hashed at.
    public var revision: String
    public var license: String
    /// Variant key (`macos`, `ios`, `ios-<arch>`) → what to download.
    public var variants: [String: CatalogVariant]

    public init(
        id: String, task: String, driver: DriverID, repo: String, revision: String, license: String,
        variants: [String: CatalogVariant]
    ) {
        self.id = id
        self.task = task
        self.driver = driver
        self.repo = repo
        self.revision = revision
        self.license = license
        self.variants = variants
    }
}

public struct CatalogVariant: Sendable, Hashable, Codable {
    /// Directory inside the repo that is the bundle root.
    public var subtree: String
    /// Sum of file sizes.
    public var bytes: Int64
    /// Paths relative to ``subtree``.
    public var files: [CatalogFile]
    /// When set, only these Core AI architectures may use the variant.
    public var architectures: [String]?

    public init(subtree: String, bytes: Int64, files: [CatalogFile], architectures: [String]? = nil) {
        self.subtree = subtree
        self.bytes = bytes
        self.files = files
        self.architectures = architectures
    }
}

public struct CatalogFile: Sendable, Hashable, Codable {
    /// Path relative to the bundle root.
    public var path: String
    public var size: Int64
    /// Lowercase hex SHA-256 of the file's contents (the HF LFS `oid` for LFS files).
    public var sha256: String

    public init(path: String, size: Int64, sha256: String) {
        self.path = path
        self.size = size
        self.sha256 = sha256
    }
}
