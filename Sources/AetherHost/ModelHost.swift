public import Foundation

/// The single place models are loaded (DESIGN.md §10). This is the first slice: per-model lifecycle (ported from
/// AetherKit 0.0.7's `ModelLifecycle`). Residency across models (measured bytes, a budget, LRU eviction) is M6; the
/// seam it needs is here: every resident records its pins and when it was last used, and ``release(_:reason:)`` is
/// the one way a model leaves memory.
///
/// - **Requests.** ``withModel(_:policy:load:_:)`` loads the model on first use (concurrent first uses join one load),
///   pins it for the body, and unpins it afterwards. A pinned model is `running` and is never released; an unpinned
///   one is `loaded` (idle).
/// - **Idle unload.** When a model goes idle the policy's timer starts; any new request stops it. When it fires on a
///   model that is still idle, the model unloads with reason `.idle`.
/// - **Memory pressure.** On each warning or critical notice, every idle model whose policy allows it unloads once,
///   with reason `.memoryPressure`. A model busy at the notice unloads as soon as its last request ends. A model that
///   was unloaded at the notice, or first seen after it, keeps its next load: while the level stays raised, requests
///   load and keep the model rather than turning every request into a load and an unload. Running requests are never
///   cancelled.
/// - **The load gate.** While pressure is critical, a load throws ``AetherError/memoryPressure``. A model that is
///   already loaded keeps serving.
///
/// Releasing drops the host's reference to the model; the model's owner (an engine actor) frees its native resources
/// when the last reference goes. Isolation makes "release only if idle" exact: the host decides it, so a release can
/// never race a request.
public actor ModelHost {
    /// The process's host, with the platform's default policy and the system memory-pressure monitor.
    public static let shared = ModelHost()

    /// The policy for models whose request names none.
    public let policy: ModelLifecyclePolicy
    private let monitor: MemoryPressureMonitor
    private let clock: HostClock
    private var residents: [ModelRef: Resident] = [:]
    private var observer: Task<Void, Never>?
    /// The last reading the observer applied.
    private var pressure = MemoryPressureMonitor.Reading(level: .normal, notice: 0)

    private struct Resident {
        var policy: ModelLifecyclePolicy
        var model: (any Sendable)?
        var loading: Task<any Sendable, any Error>?
        /// Requests holding the model (and the load in flight, which pins on behalf of its callers).
        var pins = 0
        var state = ModelLifecycleState.unloaded(nil)
        /// Why the model last unloaded, kept across failed or refused loads.
        var lastReason: ModelUnloadReason?
        /// The last pressure notice this model has released for, or found nothing to release.
        var handledNotice: Int
        var lastUsed: ContinuousClock.Instant
        var timer: Task<Void, Never>?
        var timerGeneration = 0
        var unloadRequested = false
        var subscribers: [UUID: AsyncStream<ModelLifecycleState>.Continuation] = [:]
    }

    public init(policy: ModelLifecyclePolicy = .platformDefault, monitor: MemoryPressureMonitor = .system,
                clock: HostClock = .system) {
        self.policy = policy
        self.monitor = monitor
        self.clock = clock
    }

    deinit {
        observer?.cancel()
        for resident in residents.values {
            resident.timer?.cancel()
            for continuation in resident.subscribers.values { continuation.finish() }
        }
    }

    // MARK: - Requests

    /// Runs `body` with the model for `ref`, loading it with `load` if it isn't resident. `policy` (when given) replaces
    /// the model's lifecycle policy from now on. `body` runs in the caller's isolation, not on the host.
    public nonisolated func withModel<Model: Sendable, Result>(
        _ ref: ModelRef, policy: ModelLifecyclePolicy? = nil,
        load: @escaping @Sendable () async throws -> Model,
        _ body: (Model) async throws -> Result
    ) async throws -> Result {
        let model: Model = try await acquire(ref, policy: policy, load: load)
        do {
            let result = try await body(model)
            await relinquish(ref)
            return result
        } catch {
            await relinquish(ref)
            throw error
        }
    }

    /// Unloads `ref` with reason `.requested`: now if it's idle, or when its last running request ends.
    public func unload(_ ref: ModelRef) {
        guard var resident = residents[ref] else { return }
        if resident.pins == 0 {
            release(ref, reason: .requested)
        } else {
            resident.unloadRequested = true
            residents[ref] = resident
        }
    }

    public func lifecycleState(of ref: ModelRef) -> ModelLifecycleState {
        residents[ref]?.state ?? .unloaded(nil)
    }

    /// The current state of `ref`, then each change. Newest-value buffer.
    public func lifecycleStates(of ref: ModelRef) async -> AsyncStream<ModelLifecycleState> {
        await ensureResident(ref, policy: nil)
        let (stream, continuation) = AsyncStream<ModelLifecycleState>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        residents[ref]?.subscribers[id] = continuation
        continuation.yield(residents[ref]?.state ?? .unloaded(nil))
        continuation.onTermination = { [weak self] _ in Task { await self?.unsubscribe(id, from: ref) } }
        return stream
    }

    // MARK: - Residency seam (M6)

    /// When each resident model was last used, and whether a request holds it. M6's LRU eviction reads this.
    func residency() -> [(ref: ModelRef, lastUsed: ContinuousClock.Instant, pinned: Bool)] {
        residents.compactMap { ref, resident in
            resident.model == nil ? nil : (ref, resident.lastUsed, resident.pins > 0)
        }
    }

    /// Unloads `ref` if it's resident and idle. Returns false if it's pinned, loading or not loaded. Every unload
    /// (idle, pressure, requested, and M6's eviction) goes through here.
    @discardableResult
    func release(_ ref: ModelRef, reason: ModelUnloadReason) -> Bool {
        guard var resident = residents[ref], resident.pins == 0, resident.loading == nil, resident.model != nil else {
            return false
        }
        resident.model = nil
        resident.unloadRequested = false
        resident.lastReason = reason
        residents[ref] = resident
        transition(ref, to: .unloaded(reason), idleTimer: false)
        return true
    }

    // MARK: - Internals

    private func acquire<Model: Sendable>(
        _ ref: ModelRef, policy: ModelLifecyclePolicy?, load: @escaping @Sendable () async throws -> Model
    ) async throws -> Model {
        await ensureResident(ref, policy: policy)
        while true {
            guard var resident = residents[ref] else { preconditionFailure("residents are never removed") }
            if let model = resident.model {
                resident.pins += 1
                resident.lastUsed = clock.now()
                residents[ref] = resident
                transition(ref, to: .running, idleTimer: false)
                return try cast(model, ref)
            }
            if let loading = resident.loading {
                // Join the load in flight; its initiator installs the model and reports the state.
                resident.pins += 1
                residents[ref] = resident
                do {
                    return try cast(try await loading.value, ref)
                } catch {
                    residents[ref]?.pins -= 1
                    settleFailedLoad(ref)
                    throw error
                }
            }
            // The load gate. Reading the monitor suspends, so look at the resident again afterwards. A notice the
            // observer hasn't applied yet is applied now, while the model is still unloaded: it has nothing to
            // release, so its next load is kept.
            let reading = await monitor.reading
            apply(reading)
            guard let again = residents[ref], again.model == nil, again.loading == nil else { continue }
            if reading.level == .critical {
                transition(ref, to: .unloaded(again.lastReason), idleTimer: false)
                throw AetherError.memoryPressure
            }
            let task = Task<any Sendable, any Error> { try await Self.runLoad(load) }
            resident = again
            resident.loading = task
            resident.pins += 1
            residents[ref] = resident
            transition(ref, to: .loading, idleTimer: false)
            do {
                let model = try await task.value
                residents[ref]?.loading = nil
                residents[ref]?.model = model
                residents[ref]?.lastUsed = clock.now()
                transition(ref, to: .running, idleTimer: false)
                return try cast(model, ref)
            } catch {
                residents[ref]?.loading = nil
                residents[ref]?.pins -= 1
                settleFailedLoad(ref)
                throw error
            }
        }
    }

    /// A failed load leaves the model unloaded, keeping why it last unloaded, once no caller waits on it.
    private func settleFailedLoad(_ ref: ModelRef) {
        guard let resident = residents[ref], resident.pins == 0, resident.model == nil, resident.loading == nil else { return }
        transition(ref, to: .unloaded(resident.lastReason), idleTimer: false)
    }

    @concurrent
    private static func runLoad<Model: Sendable>(_ load: @Sendable () async throws -> Model) async throws -> any Sendable {
        try await load()
    }

    private func cast<Model>(_ model: any Sendable, _ ref: ModelRef) throws(AetherError) -> Model {
        guard let typed = model as? Model else {
            throw .manifestMismatch(field: "driver", expected: "\(Model.self)", found: "\(type(of: model)) for \(ref)")
        }
        return typed
    }

    /// A request ended. The last one leaves the model idle: the idle timer starts, and a requested unload or a
    /// pressure notice that arrived while it ran releases it now.
    private func relinquish(_ ref: ModelRef) {
        guard var resident = residents[ref] else { return }
        resident.pins = max(0, resident.pins - 1)
        resident.lastUsed = clock.now()
        residents[ref] = resident
        guard resident.pins == 0, resident.model != nil else { return }
        if resident.unloadRequested {
            release(ref, reason: .requested)
            return
        }
        transition(ref, to: .loaded, idleTimer: true)
        if pressure.level >= .warning { relieve(ref, notice: pressure.notice) }
    }

    private func ensureResident(_ ref: ModelRef, policy: ModelLifecyclePolicy?) async {
        await observePressure()
        if residents[ref] == nil {
            // Notices from before this model was first seen found nothing loaded to release. Ask the monitor, not
            // the observer, which may not have applied the latest notice yet.
            let notice = await monitor.reading.notice
            if residents[ref] == nil {
                residents[ref] = Resident(policy: policy ?? self.policy, handledNotice: max(notice, pressure.notice),
                                          lastUsed: clock.now())
                return
            }
        }
        if let policy { residents[ref]?.policy = policy }
    }

    private func transition(_ ref: ModelRef, to next: ModelLifecycleState, idleTimer: Bool) {
        guard var resident = residents[ref] else { return }
        resident.timerGeneration += 1
        resident.timer?.cancel()
        resident.timer = nil
        if resident.state != next {
            resident.state = next
            for continuation in resident.subscribers.values { continuation.yield(next) }
        }
        if idleTimer, let after = resident.policy.idleUnloadAfter {
            let generation = resident.timerGeneration
            resident.timer = Task { [weak self, clock] in
                do { try await clock.sleep(after) } catch { return }
                await self?.fireIdle(ref, generation: generation)
            }
        }
        residents[ref] = resident
    }

    private func fireIdle(_ ref: ModelRef, generation: Int) {
        guard let resident = residents[ref], resident.timerGeneration == generation, resident.state == .loaded else { return }
        release(ref, reason: .idle)
    }

    private func unsubscribe(_ id: UUID, from ref: ModelRef) {
        residents[ref]?.subscribers[id] = nil
    }

    // MARK: - Memory pressure

    private func observePressure() async {
        guard observer == nil else { return }
        // Claim the slot before suspending, so concurrent first requests start one observer.
        observer = Task {}
        let readings = await monitor.readings()
        let current = await monitor.reading
        pressure = current
        observer = Task { [weak self] in
            for await reading in readings { await self?.apply(reading) }
        }
    }

    /// Applies a reading from the observer or the load gate; an older one than the last applied is ignored.
    private func apply(_ reading: MemoryPressureMonitor.Reading) {
        guard reading.notice >= pressure.notice else { return }
        let raised = reading.notice > pressure.notice
        pressure = reading
        guard raised, reading.level >= .warning else { return }
        for ref in residents.keys { relieve(ref, notice: reading.notice) }
    }

    /// A raised notice releases the model once, if it's idle and its policy allows. A busy model releases for it when
    /// its last request ends; an unloaded one has nothing to release, so the notice is spent and its next load kept.
    private func relieve(_ ref: ModelRef, notice: Int) {
        guard var resident = residents[ref], resident.policy.releaseUnderMemoryPressure, resident.handledNotice < notice else {
            return
        }
        switch resident.state {
        case .unloaded:
            resident.handledNotice = notice
            residents[ref] = resident
        case .loaded:
            resident.handledNotice = notice
            residents[ref] = resident
            release(ref, reason: .memoryPressure)
        case .loading, .running:
            break
        }
    }
}
