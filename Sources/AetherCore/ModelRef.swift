/// One model variant resolved for one device, with every asset it installs (DESIGN.md §5).
public struct ModelRef: Sendable, Hashable, Codable, CustomStringConvertible {
    /// One asset of the variant, with its repo and revision made explicit.
    public struct Asset: Sendable, Hashable, Codable {
        public var name: String
        public var repo: String
        public var revision: String
        public var subtree: String
        public var files: [CatalogFile]

        public init(name: String, repo: String, revision: String, subtree: String, files: [CatalogFile]) {
            self.name = name
            self.repo = repo
            self.revision = revision
            self.subtree = subtree
            self.files = files
        }
    }

    public var catalogID: String
    /// The main repo and revision: the install lives at `<repo>/<revision>/`.
    public var repo: String
    public var revision: String
    /// The catalog variant key (`macos`, `ios`, `ios-h18p`).
    public var variant: String
    public var driver: DriverID
    /// Sorted by name, `main` first.
    public var assets: [Asset]

    public init(catalogID: String, repo: String, revision: String, variant: String, driver: DriverID, assets: [Asset]) {
        self.catalogID = catalogID
        self.repo = repo
        self.revision = revision
        self.variant = variant
        self.driver = driver
        self.assets = assets.sorted { ($0.name == "main" ? "" : $0.name) < ($1.name == "main" ? "" : $1.name) }
    }

    public var description: String { "\(catalogID)[\(variant)] (\(repo)@\(revision.prefix(7)))" }

    public var bytes: Int64 { assets.reduce(0) { $0 + $1.files.reduce(0) { $0 + $1.size } } }

    public func asset(_ name: String) -> Asset? { assets.first { $0.name == name } }

    /// Picks the variant of `entry` for `device`, before anything downloads:
    /// 1. `<platform>-<architecture>` (the per-arch AOT subtree, e.g. `ios-h18p`) when the entry has it;
    /// 2. else the generic `<platform>` variant (`ios`, `macos`);
    /// 3. else `unsupportedDevice`.
    /// A variant that declares `architectures` is eligible only on those architectures.
    public static func resolve(_ entry: CatalogEntry, for device: Device) throws(AetherError) -> ModelRef {
        let platform = device.platform.rawValue
        for key in ["\(platform)-\(device.architecture)", platform] {
            guard let variant = entry.variants[key] else { continue }
            if let archs = variant.architectures, !archs.contains(device.architecture) { continue }
            guard variant.assets["main"] != nil else {
                throw .manifestInvalid(field: "variants.\(key).assets.main", reason: "every variant has a main asset")
            }
            let assets = variant.assets.map { name, asset in
                Asset(
                    name: name, repo: asset.repo ?? entry.repo, revision: asset.revision ?? entry.revision,
                    subtree: asset.subtree, files: asset.files)
            }
            return ModelRef(
                catalogID: entry.id, repo: entry.repo, revision: entry.revision, variant: key, driver: variant.driver,
                assets: assets)
        }
        throw .unsupportedDevice(catalogID: entry.id, platform: platform, architecture: device.architecture)
    }

    /// Looks `id` up in `catalog` and resolves it for `device`.
    public static func resolve(_ id: String, in catalog: Catalog = .builtin, for device: Device) throws(AetherError) -> ModelRef {
        try resolve(catalog.entry(id), for: device)
    }
}
