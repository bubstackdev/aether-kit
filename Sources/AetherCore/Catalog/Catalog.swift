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
    /// True for models trained for mature (adult) content, such as the Krea 2 realism merge: an app should require an
    /// explicit opt-in before offering them. Such repos are gated on Hugging Face, so installing needs a token.
    public var mature: Bool
    /// `platform` or `platform-arch` (`macos`, `ios`, `ios-h18p`) → what to install.
    public var variants: [String: CatalogVariant]

    public init(
        id: String, task: String, repo: String, revision: String, license: String, mature: Bool = false,
        variants: [String: CatalogVariant]
    ) {
        self.id = id
        self.task = task
        self.repo = repo
        self.revision = revision
        self.license = license
        self.mature = mature
        self.variants = variants
    }

    private enum CodingKeys: String, CodingKey { case id, task, repo, revision, license, mature, variants }

    /// `mature` is optional in JSON (absent means false).
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id), task: try c.decode(String.self, forKey: .task),
            repo: try c.decode(String.self, forKey: .repo), revision: try c.decode(String.self, forKey: .revision),
            license: try c.decode(String.self, forKey: .license),
            mature: try c.decodeIfPresent(Bool.self, forKey: .mature) ?? false,
            variants: try c.decode([String: CatalogVariant].self, forKey: .variants))
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

    /// Sum of every asset's file sizes, saturating at `Int64.max` (``ModelRef/validateForStore()`` refuses such sizes).
    public var bytes: Int64 { saturatingSum(assets.values.map(\.bytes)) }
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

    /// Sum of the file sizes, saturating at `Int64.max` (``ModelRef/validateForStore()`` refuses such sizes).
    public var bytes: Int64 { saturatingSum(files.map(\.size)) }
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

/// `values` summed, clamped to `Int64.min...Int64.max` instead of trapping on overflow.
func saturatingSum(_ values: some Sequence<Int64>) -> Int64 {
    values.reduce(0) { total, value in
        let (sum, overflow) = total.addingReportingOverflow(value)
        return overflow ? (value < 0 ? .min : .max) : sum
    }
}

/// The sum of `values`, or nil if it overflows `Int64`.
func checkedSum(_ values: some Sequence<Int64>) -> Int64? {
    var total: Int64 = 0
    for value in values {
        let (sum, overflow) = total.addingReportingOverflow(value)
        if overflow { return nil }
        total = sum
    }
    return total
}
