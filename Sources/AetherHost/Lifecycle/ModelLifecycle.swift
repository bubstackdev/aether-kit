/// When a loaded model unloads by itself (ported from AetherKit 0.0.7's `ModelLifecyclePolicy`).
public struct ModelLifecyclePolicy: Sendable, Hashable {
    /// Unload after the model has been idle this long; nil keeps it loaded until it's asked to unload or memory runs
    /// short.
    public var idleUnloadAfter: Duration?
    /// Unload the model, if idle, when the system reports memory pressure (warning or critical), once per notice.
    public var releaseUnderMemoryPressure: Bool

    public init(idleUnloadAfter: Duration?, releaseUnderMemoryPressure: Bool = true) {
        self.idleUnloadAfter = idleUnloadAfter
        self.releaseUnderMemoryPressure = releaseUnderMemoryPressure
    }

    /// Idle unload after 60 s on iOS and 5 min on the Mac, and release under memory pressure.
    public static var platformDefault: Self {
        #if os(iOS)
        Self(idleUnloadAfter: .seconds(60))
        #else
        Self(idleUnloadAfter: .seconds(300))
        #endif
    }

    /// Neither: the model stays loaded until it's asked to unload.
    public static let manual = Self(idleUnloadAfter: nil, releaseUnderMemoryPressure: false)
}

/// Why a model left memory.
public enum ModelUnloadReason: Sendable, Hashable {
    /// The app asked (``ModelHost/unload(_:)``).
    case requested
    /// The policy's idle time passed.
    case idle
    /// The system reported memory pressure.
    case memoryPressure
}

public enum ModelLifecycleState: Sendable, Hashable {
    /// Not loaded: never yet (nil), or unloaded for the reason given. The next request loads it.
    case unloaded(ModelUnloadReason?)
    case loading
    /// Loaded and idle.
    case loaded
    /// Serving at least one request.
    case running
}

/// The time source of ``ModelHost`` and ``MemoryPressureMonitor``: the idle timers sleep on it, and critical-pressure
/// easing reads it. Tests inject a clock they advance and sleeps they fire.
public struct HostClock: Sendable {
    public var now: @Sendable () -> ContinuousClock.Instant
    public var sleep: @Sendable (Duration) async throws -> Void

    public init(now: @escaping @Sendable () -> ContinuousClock.Instant,
                sleep: @escaping @Sendable (Duration) async throws -> Void) {
        self.now = now
        self.sleep = sleep
    }

    /// `ContinuousClock` and `Task.sleep`.
    public static let system = HostClock(now: { .now }, sleep: { try await Task.sleep(for: $0) })
}
