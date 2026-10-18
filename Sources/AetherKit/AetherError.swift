/// Failures detected by Aether before or after native model execution.
/// Native framework errors retain their original type and underlying diagnostics.
public enum AetherError: Error, Sendable, Equatable, CustomStringConvertible {
    case invalidTensor(String)
    case invalidRequest(String)
    case invalidModel(String)
    case unsupported(String)
    case sessionBusy
    case sessionClosed

    public var description: String {
        switch self {
        case .invalidTensor(let reason): "Invalid tensor: \(reason)"
        case .invalidRequest(let reason): "Invalid request: \(reason)"
        case .invalidModel(let reason): "Invalid model: \(reason)"
        case .unsupported(let reason): "Unsupported: \(reason)"
        case .sessionBusy: "This session already has an inference in progress."
        case .sessionClosed: "This session is closing or has been unloaded."
        }
    }
}
