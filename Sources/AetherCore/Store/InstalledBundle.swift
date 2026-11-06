public import Foundation

/// A complete, verified install of one model variant: every asset under `<store root>/<repo>/<revision>/`.
public struct InstalledBundle: Sendable, Hashable {
    public var ref: ModelRef
    /// The revision root, `<store root>/<repo>/<revision>`.
    public var url: URL
    /// True when this is a cached sibling revision returned by the `.allowStaleRevision` fallback, not `ref`'s pin.
    public var isStale: Bool

    public init(ref: ModelRef, url: URL, isStale: Bool) {
        self.ref = ref
        self.url = url
        self.isStale = isStale
    }

    public var catalogID: String { ref.catalogID }

    /// The directory of asset `name`, or nil if the variant has no such asset.
    public func url(ofAsset name: String) -> URL? {
        ref.asset(name).map { InstalledLayout.assetRoot($0, mainRepo: ref.repo, in: url) }
    }

    /// `"<asset>:<path>"` → a file URL (`"."` is the asset root itself).
    public func url(of path: AssetPath) -> URL? {
        guard let root = url(ofAsset: path.asset) else { return nil }
        return path.path == "." ? root : root.appending(path: path.path)
    }
}

/// Where each asset of an install lives (DESIGN.md §5):
/// - assets of the model's own repo keep their repo layout: `<revision root>/<subtree>/…`;
/// - foreign assets (a tokenizer from another repo) live in `<revision root>/.aether/assets/<name>/…`;
/// - Aether's own files (manifest, receipt) live in `<revision root>/.aether/`.
enum InstalledLayout {
    static func assetRoot(_ asset: ModelRef.Asset, mainRepo: String, in root: URL) -> URL {
        if asset.repo != mainRepo {
            return root.appending(path: Manifest.directory).appending(path: "assets").appending(path: asset.name)
        }
        return asset.subtree.isEmpty ? root : appendingComponents(root, asset.subtree)
    }

    static func appendingComponents(_ url: URL, _ path: String) -> URL {
        path.split(separator: "/").reduce(url) { $0.appending(path: String($1)) }
    }
}

public struct InstallOptions: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// When the download fails, return a complete cached sibling revision instead (flagged ``InstalledBundle/isStale``).
    public static let allowStaleRevision = InstallOptions(rawValue: 1 << 0)
}

public struct InstallProgress: Sendable, Hashable {
    public var completedBytes: Int64
    public var totalBytes: Int64
    /// The file being transferred, relative to the bundle root.
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

public enum InstallEvent: Sendable, Hashable {
    case progress(InstallProgress)
    case finished(InstalledBundle)
}

/// Written into `.aether/` just before the install is renamed into place; its presence means "complete and verified".
struct InstallReceipt: Codable, Sendable {
    static let fileName = "receipt.json"

    var ref: ModelRef
    var installedAt: Date

    static func url(in root: URL) -> URL {
        root.appending(path: Manifest.directory).appending(path: fileName)
    }
}
