/// Where one model's bundle lives upstream, resolved for one device (DESIGN.md §5).
public struct ModelRef: Sendable, Hashable, Codable, CustomStringConvertible {
    public var catalogID: String
    /// Hugging Face repo, `owner/name`.
    public var repo: String
    /// Full commit SHA.
    public var revision: String
    /// Directory inside the repo that is the bundle root.
    public var subtree: String
    /// The catalog variant key it was resolved from (`macos`, `ios`, `ios-h18p`, …).
    public var variant: String

    public init(catalogID: String, repo: String, revision: String, subtree: String, variant: String) {
        self.catalogID = catalogID
        self.repo = repo
        self.revision = revision
        self.subtree = subtree
        self.variant = variant
    }

    public var description: String { "\(catalogID) (\(repo)@\(revision.prefix(7))/\(subtree))" }

    /// Picks the variant of `entry` for `device`:
    /// 1. `<platform>-<architecture>` (e.g. `ios-h18p`) when the entry lists it;
    /// 2. otherwise `<platform>` (`ios` or `macos`).
    ///
    /// A variant that declares `architectures` is only eligible on those architectures.
    public static func resolve(_ entry: CatalogEntry, for device: Device) throws(AetherError) -> (ModelRef, CatalogVariant) {
        let platform = device.platform.rawValue
        for key in ["\(platform)-\(device.architecture)", platform] {
            guard let variant = entry.variants[key] else { continue }
            if let archs = variant.architectures, !archs.contains(device.architecture) { continue }
            let ref = ModelRef(
                catalogID: entry.id, repo: entry.repo, revision: entry.revision, subtree: variant.subtree,
                variant: key)
            return (ref, variant)
        }
        throw .unsupportedDevice(catalogID: entry.id, platform: platform, architecture: device.architecture)
    }

    /// Looks `id` up in `catalog` and resolves it for `device`.
    public static func resolve(
        _ id: String, in catalog: Catalog = .builtin, for device: Device
    ) throws(AetherError) -> (ModelRef, CatalogVariant) {
        try resolve(catalog.entry(id), for: device)
    }
}
