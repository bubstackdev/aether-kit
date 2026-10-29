import Dispatch
import Foundation

/// When a loaded model unloads by itself (docs/model-lifecycle.md).
public struct ModelLifecyclePolicy: Sendable, Equatable {
    /// Unload after the model has been idle this long; nil keeps it loaded until it's asked to unload or memory runs
    /// short.
    public var idleUnloadAfter: Duration?
    /// Unload the model, if idle, when the system reports memory pressure (warning or critical).
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

public enum ModelUnloadReason: Sendable, Equatable {
    /// The app asked (`unload()`).
    case requested
    /// The policy's idle time passed.
    case idle
    /// The system reported memory pressure.
    case memoryPressure
}

public enum ModelLifecycleState: Sendable, Equatable {
    /// Not loaded: never yet (nil), or unloaded for the reason given. The next request loads it.
    case unloaded(ModelUnloadReason?)
    case loading
    /// Loaded and idle.
    case loaded
    /// Serving a request.
    case running
}

public enum MemoryPressureLevel: Int, Sendable, Comparable {
    case normal, warning, critical

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// The process's memory pressure, as the system reports it. While it's critical, Kit's products refuse new loads.
public enum MemoryPressure {
    public static var level: MemoryPressureLevel { MemoryPressureMonitor.shared.level }
    /// The current level, then each change.
    public static var levels: AsyncStream<MemoryPressureLevel> { MemoryPressureMonitor.shared.levels }
}

/// The system's memory-pressure notices, fanned out to every registered model. The system source starts on first use.
package final class MemoryPressureMonitor: Sendable {
    package static let shared = MemoryPressureMonitor(systemSource: true)

    private struct Registered { weak var lifecycle: ModelLifecycle? }
    private struct State {
        var level = MemoryPressureLevel.normal
        var subscribers: [UUID: AsyncStream<MemoryPressureLevel>.Continuation] = [:]
        var models: [ObjectIdentifier: Registered] = [:]
        var source: (any DispatchSourceMemoryPressure)?
    }

    private let systemSource: Bool
    private let state = UnfairLock(State())
    private let queue = DispatchQueue(label: "aether.memory-pressure")

    /// `systemSource: false` gives a monitor that only `deliver` drives (tests and qualification).
    package init(systemSource: Bool) { self.systemSource = systemSource }

    package var level: MemoryPressureLevel {
        startIfNeeded()
        return state.withLock { $0.level }
    }

    package var levels: AsyncStream<MemoryPressureLevel> {
        startIfNeeded()
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let id = UUID()
            state.withLock {
                $0.subscribers[id] = continuation
                continuation.yield($0.level)
            }
            continuation.onTermination = { [weak self] _ in self?.state.withLock { _ = $0.subscribers.removeValue(forKey: id) } }
        }
    }

    package func register(_ lifecycle: ModelLifecycle) {
        startIfNeeded()
        state.withLock {
            $0.models = $0.models.filter { $0.value.lifecycle != nil }
            $0.models[ObjectIdentifier(lifecycle)] = Registered(lifecycle: lifecycle)
        }
    }

    /// Records a level and tells every registered model, as the system source does. Also the seam for tests and for
    /// qualification runs that can't raise real pressure.
    package func deliver(_ level: MemoryPressureLevel) {
        let models: [ModelLifecycle] = state.withLock { state in
            guard state.level != level else { return [] }
            state.level = level
            for continuation in state.subscribers.values { continuation.yield(level) }
            return state.models.values.compactMap(\.lifecycle)
        }
        for model in models { model.pressure(level) }
    }

    private func startIfNeeded() {
        guard systemSource else { return }
        state.withLock { state in
            guard state.source == nil else { return }
            let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: queue)
            source.setEventHandler { [weak self, weak source] in
                guard let self, let event = source?.data else { return }
                deliver(event.contains(.critical) ? .critical : event.contains(.warning) ? .warning : .normal)
            }
            source.activate()
            state.source = source
        }
    }
}

/// One model's lifecycle bookkeeping, shared by every product: its state and stream, the idle timer, memory-pressure
/// release and the load gate. The product owns the model. It reports each transition, and `release` unloads the model
/// only if it's still idle under the product's own isolation. When it does, it calls ``unloaded(_:)`` before any new
/// request can start a load, and returns true.
package final class ModelLifecycle: Sendable {
    package typealias Release = @Sendable (ModelUnloadReason) async -> Bool

    package let policy: ModelLifecyclePolicy
    private let monitor: MemoryPressureMonitor
    private let sleep: @Sendable (Duration) async throws -> Void
    private struct State {
        var current = ModelLifecycleState.unloaded(nil)
        var subscribers: [UUID: AsyncStream<ModelLifecycleState>.Continuation] = [:]
        var timer: Task<Void, Never>?
        var timerGeneration = 0
        var release: Release?
    }
    private let state = UnfairLock(State())

    package init(policy: ModelLifecyclePolicy, monitor: MemoryPressureMonitor = .shared,
                 sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }) {
        self.policy = policy
        self.monitor = monitor
        self.sleep = sleep
        monitor.register(self)
    }

    deinit {
        state.withLock { state in
            state.timer?.cancel()
            for continuation in state.subscribers.values { continuation.finish() }
        }
    }

    /// Sets how idle and pressure releases reach the model. Products set it once, right after creating the lifecycle.
    package func setRelease(_ release: @escaping Release) { state.withLock { $0.release = release } }

    package var current: ModelLifecycleState { state.withLock { $0.current } }

    package var states: AsyncStream<ModelLifecycleState> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let id = UUID()
            state.withLock {
                $0.subscribers[id] = continuation
                continuation.yield($0.current)
            }
            continuation.onTermination = { [weak self] _ in self?.state.withLock { _ = $0.subscribers.removeValue(forKey: id) } }
        }
    }

    /// Throws while memory pressure is critical: no new load starts then.
    package func admitLoad() throws {
        if monitor.level == .critical { throw AetherError.memoryPressure }
    }

    package func loading() { transition(to: .loading, idleTimer: false) }
    package func loaded() { settle() }
    package func running() { transition(to: .running, idleTimer: false) }
    /// A request ended and the model stays loaded.
    package func idle() { settle() }

    /// Loaded and idle: start the idle timer, and release at once if pressure is still raised (a notice that arrived
    /// while the model was busy).
    private func settle() {
        transition(to: .loaded, idleTimer: true)
        let level = monitor.level
        if level >= .warning { pressure(level) }
    }
    package func unloaded(_ reason: ModelUnloadReason?) { transition(to: .unloaded(reason), idleTimer: false) }

    private func transition(to next: ModelLifecycleState, idleTimer: Bool) {
        state.withLock { state in
            state.timerGeneration += 1
            state.timer?.cancel()
            state.timer = nil
            if state.current != next {
                state.current = next
                for continuation in state.subscribers.values { continuation.yield(next) }
            }
            guard idleTimer, let after = policy.idleUnloadAfter else { return }
            let generation = state.timerGeneration
            state.timer = Task { [weak self, sleep] in
                do { try await sleep(after) } catch { return }
                await self?.fireIdle(generation)
            }
        }
    }

    private func fireIdle(_ generation: Int) async {
        let release: Release? = state.withLock { state in
            guard state.timerGeneration == generation, state.current == .loaded else { return nil }
            return state.release
        }
        if let release { _ = await release(.idle) }
    }

    /// From the monitor: a warning or critical notice releases the model if it's idle and the policy allows.
    func pressure(_ level: MemoryPressureLevel) {
        guard level >= .warning, policy.releaseUnderMemoryPressure else { return }
        let release: Release? = state.withLock { $0.current == .loaded ? $0.release : nil }
        guard let release else { return }
        Task { _ = await release(.memoryPressure) }
    }
}
