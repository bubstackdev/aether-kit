import AetherAssets
import AetherKit
import Foundation

/// What the model screen shows for a readiness state, and which actions it offers. A pure mapping, so the screen's
/// rules are tested without a view.
public struct ReadinessDisplay: Equatable, Sendable {
    public enum Action: String, CaseIterable, Sendable {
        /// Start, or resume, the download (`ModelReadiness.prepare()`).
        case download
        /// Stop a running preparation; downloaded bytes stay for a resume.
        case cancel
        /// Prepare again after a failure.
        case retry
        /// Remove the installation and any downloaded bytes.
        case delete
        /// Decide again, for example after freeing memory.
        case refresh
    }

    public enum Tone: Sendable { case neutral, busy, ready, problem }

    public var title: String
    public var detail: String
    /// 0 to 1 while a phase reports bytes; nil otherwise (an indeterminate spinner when `tone` is busy).
    public var progress: Double?
    public var actions: [Action]
    public var tone: Tone

    public init(title: String, detail: String, progress: Double? = nil, actions: [Action], tone: Tone) {
        self.title = title; self.detail = detail; self.progress = progress; self.actions = actions; self.tone = tone
    }

    public init(_ state: ModelReadiness.State) {
        switch state {
        case let .notInstalled(present, total):
            if present > 0 {
                self.init(title: "Paused", detail: "\(Self.bytes(present)) of \(Self.bytes(total)) downloaded. Resume to continue.",
                          progress: Self.fraction(present, total), actions: [.download, .delete], tone: .neutral)
            } else {
                self.init(title: "Not installed", detail: "Download \(Self.bytes(total)) to run the model on this device.",
                          actions: [.download], tone: .neutral)
            }
        case let .installing(progress):
            let detail = switch progress.phase {
            case .checking: "Checking free space"
            case .downloading: "Downloading \(Self.bytes(progress.bytesDone)) of \(Self.bytes(progress.bytesTotal))"
            case .installing: "Installing and checking \(Self.bytes(progress.bytesDone)) of \(Self.bytes(progress.bytesTotal))"
            @unknown default: "Preparing"
            }
            self.init(title: "Installing", detail: detail,
                      progress: progress.phase == .checking ? nil : progress.fraction, actions: [.cancel], tone: .busy)
        case .verifying:
            self.init(title: "Verifying", detail: "Checking every byte of the installed model.", actions: [], tone: .busy)
        case .ready:
            self.init(title: "Ready", detail: "Installed and verified.", actions: [.delete], tone: .ready)
        case let .unsupported(reason):
            self.init(title: "Not supported here", detail: Self.describe(reason), actions: [.refresh], tone: .problem)
        case let .failed(failure):
            self.init(title: "Failed", detail: Self.describe(failure),
                      actions: failure.isRetryable ? [.retry, .delete] : [.delete], tone: .problem)
        @unknown default:
            self.init(title: "Unknown state", detail: "\(state)", actions: [.refresh], tone: .problem)
        }
    }

    public static func describe(_ reason: ModelReadiness.Unsupported) -> String {
        switch reason {
        case .operatingSystem: "The model needs macOS 27 or iOS 27."
        case .platform: "This model build is for another platform."
        case let .memory(required, available):
            "The model needs \(bytes(required)) of memory; this device allows \(bytes(available))."
        @unknown default: "\(reason)"
        }
    }

    public static func describe(_ failure: ModelReadiness.Failure) -> String {
        switch failure {
        case let .preparation(error): describe(error)
        case let .verification(reason): "The installed model didn't verify: \(reason)"
        case let .invalidModel(reason): "This model can't run here: \(reason)"
        @unknown default: "\(failure)"
        }
    }

    public static func describe(_ error: PreparationError) -> String {
        switch error {
        case let .insufficientSpace(needed, available): "Needs \(bytes(needed)) free; \(bytes(available)) available."
        case let .integrityMismatch(path): "\(path) didn't match its checksum; it will download again."
        case let .http(status, path): "The server answered \(status) for \(path)."
        case let .network(reason): "Network error: \(reason)"
        case let .invalidEntry(reason): "The catalog entry is invalid: \(reason)"
        case .busy: "Another download of this model is running."
        case let .fileSystem(code): "File system error \(code)."
        @unknown default: "\(error)"
        }
    }

    public static func bytes(_ count: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(clamping: count), countStyle: .file)
    }

    static func fraction(_ done: UInt64, _ total: UInt64) -> Double { total == 0 ? 0 : min(1, Double(done) / Double(total)) }
}

/// The lifecycle strip: whether the model is loaded, and the process's memory pressure.
public struct LifecycleDisplay: Equatable, Sendable {
    public var model: String
    public var pressure: String
    public var isLoaded: Bool

    public init(state: ModelLifecycleState?, pressure: MemoryPressureLevel) {
        switch state {
        case nil: model = "No model open"; isLoaded = false
        case let .unloaded(reason)?:
            isLoaded = false
            model = switch reason {
            case nil: "Not loaded yet"
            case .requested?: "Unloaded"
            case .idle?: "Unloaded after idling"
            case .memoryPressure?: "Unloaded under memory pressure"
            @unknown default: "Unloaded"
            }
        case .loading?: model = "Loading"; isLoaded = false
        case .loaded?: model = "Loaded, idle"; isLoaded = true
        case .running?: model = "Running"; isLoaded = true
        @unknown default: model = "\(String(describing: state))"; isLoaded = false
        }
        self.pressure = switch pressure {
        case .normal: "Memory normal"
        case .warning: "Memory warning"
        case .critical: "Memory critical: loads refused"
        }
    }
}
