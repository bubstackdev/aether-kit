/// The single error domain of Aether-Kit (DESIGN.md §11).
///
/// Cases are structured so callers can react without parsing strings. Family- or layer-specific detail travels
/// in an associated payload (``DownloadFailure``, ``TensorFailure``, or `underlying`), never in a separate public
/// error enum.
public enum AetherError: Error, Sendable, CustomStringConvertible {
    /// `driver` is a known driver id (MANIFEST.md §1) that no registered driver implements in this build.
    /// Raised before anything downloads.
    case unsupported(driver: DriverID)
    /// `driver` is not a driver id this build knows at all.
    case unknownDriver(DriverID)
    /// The catalog entry has no variant this device can run.
    case unsupportedDevice(catalogID: String, platform: String, architecture: String)
    /// The id is not in the catalog.
    case unknownModel(id: String)
    /// A manifest (or a catalog or zoo document) is malformed. `field` is the dotted path of the offending field.
    case manifestInvalid(field: String, reason: String)
    /// The manifest and the graph disagree. `field` is the dotted manifest path; the others describe both sides.
    case manifestMismatch(field: String, expected: String, found: String)
    /// A download failed. `file` is the bundle-relative path.
    case download(file: String, reason: DownloadFailure)
    /// A downloaded file's SHA-256 does not match the catalog.
    case integrity(file: String, expected: String, actual: String)
    /// Not enough room: context, memory or disk. `what` names the resource.
    case capacity(what: String, needed: Int64, available: Int64)
    /// A tensor binding or typed access was invalid (wrong type, shape, unbound name, or a buffer in use).
    case tensor(name: String, reason: TensorFailure)
    /// The underlying runtime (Core AI or the file system) failed.
    case runtime(underlying: any Error)

    public enum DownloadFailure: Sendable, Hashable, CustomStringConvertible {
        /// The server answered with an unexpected HTTP status.
        case http(status: Int)
        /// The transfer failed (offline, reset, timed out, cancelled).
        case transport(String)
        /// The server's bytes disagree with the catalog (for example a size mismatch).
        case unexpectedContent(String)

        public var description: String {
            switch self {
            case .http(let status): "HTTP \(status)"
            case .transport(let message): message
            case .unexpectedContent(let message): message
            }
        }
    }

    public enum TensorFailure: Sendable, Hashable, CustomStringConvertible {
        case typeMismatch(expected: String, found: String)
        case shapeMismatch(expected: [Int], found: [Int])
        case outOfBounds(index: Int, count: Int)
        case unbound
        case unknownName
        /// The buffer's storage is lent to a running graph, or the handle was released.
        case unavailable(String)
        case unsupported(String)

        public var description: String {
            switch self {
            case .typeMismatch(let e, let f): "type mismatch: expected \(e), found \(f)"
            case .shapeMismatch(let e, let f): "shape mismatch: expected \(e), found \(f)"
            case .outOfBounds(let i, let c): "index \(i) out of bounds for \(c) elements"
            case .unbound: "not bound"
            case .unknownName: "no such tensor in the graph signature"
            case .unavailable(let why): "unavailable: \(why)"
            case .unsupported(let why): "unsupported: \(why)"
            }
        }
    }

    public var description: String {
        switch self {
        case .unsupported(let driver):
            "Unsupported driver '\(driver)': no registered driver loads it"
        case .unknownDriver(let driver):
            "Unknown driver '\(driver)'"
        case .unsupportedDevice(let id, let platform, let arch):
            "'\(id)' has no variant for \(platform) (\(arch))"
        case .unknownModel(let id):
            "Unknown model '\(id)'"
        case .manifestInvalid(let field, let reason):
            "Invalid manifest field '\(field)': \(reason)"
        case .manifestMismatch(let field, let expected, let found):
            "Manifest field '\(field)' does not match the graph: expected \(expected), found \(found)"
        case .download(let file, let reason):
            "Download of '\(file)' failed: \(reason)"
        case .integrity(let file, let expected, let actual):
            "Integrity check failed for '\(file)': expected sha256 \(expected), got \(actual)"
        case .capacity(let what, let needed, let available):
            "Insufficient \(what): needed \(needed), available \(available)"
        case .tensor(let name, let reason):
            "Tensor '\(name)': \(reason)"
        case .runtime(let underlying):
            "Runtime failure: \(underlying)"
        }
    }
}
