public import Foundation

/// A complete, verified bundle on disk.
public struct InstalledBundle: Sendable, Hashable {
    public var ref: ModelRef
    /// The bundle root: `<store root>/<repo>/<revision>/<subtree>`.
    public var url: URL
    /// True when this is a cached sibling revision returned by the `.allowStaleRevision` fallback, not `ref`'s pin.
    public var isStale: Bool

    public init(ref: ModelRef, url: URL, isStale: Bool) {
        self.ref = ref
        self.url = url
        self.isStale = isStale
    }

    public var catalogID: String { ref.catalogID }
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

/// Written into a bundle just before it is renamed into place; its presence means "complete and verified".
struct InstallReceipt: Codable, Sendable {
    static let fileName = ".aether-install.json"

    var ref: ModelRef
    var files: [CatalogFile]
    var installedAt: Date
}
