import Dispatch
import Foundation
#if os(iOS)
import os
#endif

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

@nonexhaustive public enum ModelUnloadReason: Sendable, Equatable {
    /// The app asked (`unload()`).
    case requested
    /// The policy's idle time passed.
    case idle
    /// The system reported memory pressure.
    case memoryPressure
}

@nonexhaustive public enum ModelLifecycleState: Sendable, Equatable {
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

/// The process's memory pressure, as the system reports it: read when first used, then each notice. While it's critical,
/// Kit's products refuse new loads. iOS doesn't report a return to normal after an app's critical notice, so critical
/// also eases by itself (docs/model-lifecycle.md): once the app's available memory has recovered to twice what it was at
/// the notice (iOS), or 60 s after the last critical notice.
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
        /// Counts warning and critical notices: a model releases once per notice, not on every idle while the level
        /// stays raised.
        var notice = 0
        var subscribers: [UUID: AsyncStream<MemoryPressureLevel>.Continuation] = [:]
        var models: [ObjectIdentifier: Registered] = [:]
        var source: (any DispatchSourceMemoryPressure)?
        /// The last critical notice: when it came, and the app's available memory then (iOS).
        var critical: (at: ContinuousClock.Instant, available: UInt64?)?
    }

    /// Critical eases this long after the last critical notice, if nothing else has eased it.
    package static let criticalExpiry: Duration = .seconds(60)

    private let systemSource: Bool
    private let state = UnfairLock(State())
    private let queue = DispatchQueue(label: "aether.memory-pressure")
    private let now: @Sendable () -> ContinuousClock.Instant
    private let availableMemory: @Sendable () -> UInt64?
    private let kernelLevel: @Sendable () -> MemoryPressureLevel?

    /// `systemSource: false` gives a monitor that only `deliver` drives (tests and qualification). `availableMemory` is
    /// what the app may still allocate (iOS), or nil where there's no per-app limit. `kernelLevel` is the kernel's
    /// level, or nil where an app can't read it (iOS).
    package init(systemSource: Bool, now: @escaping @Sendable () -> ContinuousClock.Instant = { .now },
                 availableMemory: @escaping @Sendable () -> UInt64? = MemoryPressureMonitor.appAvailableMemory,
                 kernelLevel: @escaping @Sendable () -> MemoryPressureLevel? = MemoryPressureMonitor.systemLevel) {
        self.systemSource = systemSource
        self.now = now
        self.availableMemory = availableMemory
        self.kernelLevel = kernelLevel
    }

    package var level: MemoryPressureLevel { reading.level }

    /// The level and the count of raised notices so far.
    package var reading: (level: MemoryPressureLevel, notice: Int) {
        startIfNeeded()
        easeIfRecovered()
        return state.withLock { ($0.level, $0.notice) }
    }

    /// Ends a critical level the system won't end itself: once the app's available memory has doubled since the
    /// notice, or the expiry has passed. Reported as a return to normal. Where the kernel's level is readable (the
    /// Mac), an expiry while it still reads critical keeps the level critical and restarts the clock.
    private func easeIfRecovered() {
        guard let critical = state.withLock({ $0.level == .critical ? $0.critical : nil }) else { return }
        let recovered = critical.available.map { at in availableMemory().map { $0 >= at.saturatingDoubled } ?? false } ?? false
        let expired = now() - critical.at >= Self.criticalExpiry
        if expired, !recovered, kernelLevel() == .critical {
            let renewed = state.withLock { state -> Bool in
                guard state.critical?.at == critical.at else { return false }
                state.critical?.at = now()
                return true
            }
            if renewed { scheduleExpiryCheck() }
        } else if recovered || expired {
            apply(.normal, ifCriticalAt: critical.at)
        }
    }

    /// iOS: `os_proc_available_memory()`. Elsewhere nil.
    package static func appAvailableMemory() -> UInt64? {
        #if os(iOS)
        let available = os_proc_available_memory()
        return available > 0 ? UInt64(available) : nil
        #else
        nil
        #endif
    }

    package var levels: AsyncStream<MemoryPressureLevel> {
        startIfNeeded()
        easeIfRecovered()
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
    package func deliver(_ level: MemoryPressureLevel) { apply(level, ifCriticalAt: nil) }

    /// A level from a notice, or (with `ifCriticalAt`) an easing decided against that critical notice, which a newer
    /// critical notice cancels.
    private func apply(_ level: MemoryPressureLevel, ifCriticalAt expected: ContinuousClock.Instant?) {
        let available = level == .critical ? availableMemory() : nil
        let (models, notice): ([ModelLifecycle], Int) = state.withLock { state in
            if let expected, state.critical?.at != expected { return ([], state.notice) }
            if level == .critical { state.critical = (now(), available) }
            guard state.level != level else { return ([], state.notice) }
            state.level = level
            if level != .critical { state.critical = nil }
            if level >= .warning { state.notice += 1 }
            for continuation in state.subscribers.values { continuation.yield(level) }
            return (state.models.values.compactMap(\.lifecycle), state.notice)
        }
        if level >= .warning { for model in models { model.pressure(notice: notice) } }
        if level == .critical { scheduleExpiryCheck() }
    }

    /// So a stream reader sees an expiry even if nothing reads the level meanwhile (system source only).
    private func scheduleExpiryCheck() {
        guard systemSource else { return }
        Task { [weak self] in
            try? await Task.sleep(for: Self.criticalExpiry + .seconds(1))
            self?.easeIfRecovered()
        }
    }

    private func startIfNeeded() {
        guard systemSource else { return }
        let startedCritical = state.withLock { state -> Bool in
            guard state.source == nil else { return false }
            // A process started under pressure sees the current level (for the load gate), not a notice.
            state.level = kernelLevel() ?? .normal
            if state.level == .critical { state.critical = (now(), nil) }
            let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: queue)
            source.setEventHandler { [weak self, weak source] in
                guard let self, let event = source?.data else { return }
                deliver(event.contains(.critical) ? .critical : event.contains(.warning) ? .warning : .normal)
            }
            source.activate()
            state.source = source
            return state.level == .critical
        }
        if startedCritical { scheduleExpiryCheck() }
    }

    /// The kernel's current pressure level (1 normal, 2 warning, 4 critical), or nil where it can't be read (an iOS app).
    package static func systemLevel() -> MemoryPressureLevel? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("kern.memorystatus_vm_pressure_level", &value, &size, nil, 0) == 0, value >= 1 else { return nil }
        return value >= 4 ? .critical : value >= 2 ? .warning : .normal
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
        /// Why the model last unloaded, kept across failed or refused loads.
        var lastReason: ModelUnloadReason?
        /// The last pressure notice this model released for.
        var handledNotice = 0
    }
    private let state = UnfairLock(State())

    package init(policy: ModelLifecyclePolicy, monitor: MemoryPressureMonitor = .shared,
                 sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }) {
        self.policy = policy
        self.monitor = monitor
        self.sleep = sleep
        monitor.register(self)
        // Notices from before this model existed found nothing loaded to release. Read after registering, so a notice
        // in between is either delivered (and spent on the unloaded model) or counted here.
        let notice = monitor.reading.notice
        state.withLock { $0.handledNotice = max($0.handledNotice, notice) }
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

    /// Loaded and idle: start the idle timer, and release now for a raised notice this model hasn't released for yet
    /// (one that arrived while it was busy). A level that stays raised doesn't release it again after every request.
    private func settle() {
        transition(to: .loaded, idleTimer: true)
        let reading = monitor.reading
        if reading.level >= .warning { pressure(notice: reading.notice) }
    }
    package func unloaded(_ reason: ModelUnloadReason) {
        state.withLock { $0.lastReason = reason }
        transition(to: .unloaded(reason), idleTimer: false)
    }
    /// A load failed or was refused, or an operation ended holding nothing: unloaded, keeping the last reason.
    package func notLoaded() { transition(to: .unloaded(state.withLock { $0.lastReason }), idleTimer: false) }

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

    /// A raised notice releases the model, once, if it's idle and the policy allows. A busy model releases for it when
    /// it next goes idle; an unloaded one has nothing to release, so the notice is spent and its next load is kept.
    func pressure(notice: Int) {
        guard policy.releaseUnderMemoryPressure else { return }
        let release: Release? = state.withLock { state in
            guard state.handledNotice < notice else { return nil }
            if case .unloaded = state.current { state.handledNotice = notice; return nil }
            return state.current == .loaded ? state.release : nil
        }
        guard let release else { return }
        Task { [weak self] in
            if await release(.memoryPressure) { self?.state.withLock { $0.handledNotice = max($0.handledNotice, notice) } }
        }
    }
}

private extension UInt64 {
    var saturatingDoubled: UInt64 { self > .max / 2 ? .max : self * 2 }
}
