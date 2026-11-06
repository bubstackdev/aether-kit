import Darwin
import Dispatch
import Foundation

public enum MemoryPressureLevel: Int, Sendable, Comparable {
    case normal, warning, critical

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// The process's memory pressure, as the system reports it: read when first used, then each notice. Ported from
/// AetherKit 0.0.7.
///
/// - Each warning or critical notice counts once (``Reading/notice``): a ``ModelHost`` releases each idle model once
///   per notice, not again after every request while the level stays raised.
/// - While the level is critical, ``ModelHost`` refuses new loads.
/// - **Easing.** iOS doesn't report a return to normal after an app's critical notice, and an app can't read the
///   kernel's level, so critical also ends by itself, reported as `.normal`: once the app's available memory has
///   recovered to twice what it was at the notice (iOS), or 60 s after the last critical notice. A repeated critical
///   notice restarts that clock. Where the kernel's level is readable (a Mac), an expiry while it still reads critical
///   keeps the level critical and restarts the clock. An easing decided against an older critical notice is dropped.
///
/// The `.system` source is a `DispatchSource` memory-pressure source, started on first use from the kernel's current
/// level (a process started under pressure sees the level for the load gate, but that is not a notice). The `.manual`
/// source is driven only by ``deliver(_:)``: for tests, and for qualification runs that can't raise real pressure.
public actor MemoryPressureMonitor {
    public enum Source: Sendable { case system, manual }

    /// The level, and how many raised (warning or critical) notices have arrived so far.
    public struct Reading: Sendable, Hashable {
        public var level: MemoryPressureLevel
        public var notice: Int
    }

    /// The process's monitor. Memory pressure is per process, so one system source serves every host in it.
    public static let system = MemoryPressureMonitor(source: .system)

    /// Critical eases this long after the last critical notice, if nothing else has eased it.
    public static let criticalExpiry: Duration = .seconds(60)

    private let source: Source
    private let clock: HostClock
    private let availableMemory: @Sendable () -> UInt64?
    private let kernelLevel: @Sendable () -> MemoryPressureLevel?

    private var current = Reading(level: .normal, notice: 0)
    /// The last critical notice: when it came, and the app's available memory then (iOS).
    private var critical: (at: ContinuousClock.Instant, available: UInt64?)?
    private var subscribers: [UUID: AsyncStream<Reading>.Continuation] = [:]
    private var dispatchSource: (any DispatchSourceMemoryPressure)?
    private var events: Task<Void, Never>?

    /// `availableMemory` is what the app may still allocate (iOS), or nil where there's no per-app limit. `kernelLevel`
    /// is the kernel's level, or nil where an app can't read it (iOS).
    public init(source: Source, clock: HostClock = .system,
                availableMemory: @escaping @Sendable () -> UInt64? = MemoryPressureMonitor.appAvailableMemory,
                kernelLevel: @escaping @Sendable () -> MemoryPressureLevel? = MemoryPressureMonitor.systemLevel) {
        self.source = source
        self.clock = clock
        self.availableMemory = availableMemory
        self.kernelLevel = kernelLevel
    }

    deinit {
        dispatchSource?.cancel()
        events?.cancel()
        for continuation in subscribers.values { continuation.finish() }
    }

    public var level: MemoryPressureLevel { reading.level }

    public var reading: Reading {
        startIfNeeded()
        easeIfRecovered()
        return current
    }

    /// The current reading, then each change. Newest-value buffer.
    public func readings() -> AsyncStream<Reading> {
        startIfNeeded()
        easeIfRecovered()
        let (stream, continuation) = AsyncStream<Reading>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        subscribers[id] = continuation
        continuation.yield(current)
        continuation.onTermination = { [weak self] _ in Task { await self?.unsubscribe(id) } }
        return stream
    }

    /// The current level, then each change.
    public func levels() -> AsyncStream<MemoryPressureLevel> {
        let readings = readings()
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let relay = Task {
                var last: MemoryPressureLevel?
                for await reading in readings where reading.level != last {
                    last = reading.level
                    continuation.yield(reading.level)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in relay.cancel() }
        }
    }

    private func unsubscribe(_ id: UUID) { subscribers[id] = nil }

    /// Records a level as a system notice would. The seam for tests and qualification; the system source uses it too.
    public func deliver(_ level: MemoryPressureLevel) { apply(level, ifCriticalAt: nil) }

    /// A level from a notice, or (with `ifCriticalAt`) an easing decided against that critical notice, which a newer
    /// critical notice cancels. Receiving a raised notice and changing level are separate: every warning or critical
    /// notice counts, even one that repeats the current level (a second warning is a second notice, and a host
    /// releases idle models again for it); a change of level without a notice (an easing, a return to normal) doesn't.
    private func apply(_ level: MemoryPressureLevel, ifCriticalAt expected: ContinuousClock.Instant?) {
        if let expected, critical?.at != expected { return }
        if level == .critical { critical = (clock.now(), availableMemory()) }
        defer { if level == .critical { scheduleExpiryCheck() } }
        let isNotice = expected == nil && level >= .warning
        guard current.level != level || isNotice else { return }
        current.level = level
        if level != .critical { critical = nil }
        if isNotice { current.notice += 1 }
        for continuation in subscribers.values { continuation.yield(current) }
    }

    /// Ends a critical level the system won't end itself (see the type's documentation).
    private func easeIfRecovered() {
        guard current.level == .critical, let critical else { return }
        let recovered = critical.available.map { at in availableMemory().map { $0 >= at.saturatingDoubled } ?? false } ?? false
        let expired = clock.now() - critical.at >= Self.criticalExpiry
        if expired, !recovered, kernelLevel() == .critical {
            self.critical?.at = clock.now()
            scheduleExpiryCheck()
        } else if recovered || expired {
            apply(.normal, ifCriticalAt: critical.at)
        }
    }

    /// So a stream reader sees an expiry even if nothing reads the level meanwhile (system source only: a manual
    /// monitor eases when it is read).
    private func scheduleExpiryCheck() {
        guard source == .system else { return }
        Task { [weak self, clock] in
            try? await clock.sleep(Self.criticalExpiry + .seconds(1))
            await self?.easeIfRecovered()
        }
    }

    private func startIfNeeded() {
        guard source == .system, dispatchSource == nil else { return }
        current.level = kernelLevel() ?? .normal
        if current.level == .critical {
            critical = (clock.now(), nil)
            scheduleExpiryCheck()
        }
        // Notices go through one stream, so they reach the actor in the order the system sent them.
        let (notices, sink) = AsyncStream<MemoryPressureLevel>.makeStream()
        let pressure = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .global())
        pressure.setEventHandler { [weak pressure] in
            guard let event = pressure?.data else { return }
            sink.yield(event.contains(.critical) ? .critical : event.contains(.warning) ? .warning : .normal)
        }
        pressure.activate()
        dispatchSource = pressure
        events = Task { [weak self] in
            for await level in notices { await self?.deliver(level) }
        }
    }

    /// iOS: `os_proc_available_memory()`. Elsewhere nil.
    public static func appAvailableMemory() -> UInt64? {
        #if os(iOS)
        let available = Device.availableMemory()
        return available > 0 ? available : nil
        #else
        nil
        #endif
    }

    /// The kernel's current pressure level (1 normal, 2 warning, 4 critical), or nil where it can't be read (an iOS app).
    public static func systemLevel() -> MemoryPressureLevel? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("kern.memorystatus_vm_pressure_level", &value, &size, nil, 0) == 0, value >= 1 else { return nil }
        return value >= 4 ? .critical : value >= 2 ? .warning : .normal
    }
}

private extension UInt64 {
    var saturatingDoubled: UInt64 { self > .max / 2 ? .max : self * 2 }
}
