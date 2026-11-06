import Synchronization

/// How soon a request wants the GPU (``GPUAdmission``). Waiting requests start highest level first, in arrival order
/// within a level; a running request is never preempted.
public enum GPUPriority: Int, Sendable, Hashable, Comparable, CaseIterable {
    /// Starts only when no `.medium` or `.high` request is waiting. There is no aging: steady other work holds it back.
    case low
    /// The default.
    case medium
    /// Goes ahead of every waiting `.medium` and `.low` request.
    case high

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// The priority of a caller's GPU requests, which the caller can change while they wait. Pass one in a request (an
/// image, speech or video request, or a text `GenerationParameters`) and keep it: setting ``priority`` moves the
/// request within the line while it waits, and does nothing once it runs. A request without a job is `.medium`.
///
/// One job may cover several requests (a batch, say); a change applies to each of them still waiting. Jobs compare by
/// identity.
public final class GPUJob: Sendable, Hashable {
    private let value: Mutex<GPUPriority>

    public init(priority: GPUPriority = .medium) { value = Mutex(priority) }

    /// Read when the admission picks the next request to start, so a change takes effect at the next free slot.
    public var priority: GPUPriority {
        get { value.withLock { $0 } }
        set { value.withLock { $0 = newValue } }
    }

    public static func == (lhs: GPUJob, rhs: GPUJob) -> Bool { lhs === rhs }
    public func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }
}
