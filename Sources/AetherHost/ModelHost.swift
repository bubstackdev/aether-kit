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
/// - **Cancellation.** A caller cancelled while a load is in flight stops waiting at once (`CancellationError`). The
///   load carries on for the callers still waiting; when none remain it is cancelled and detached, and whatever it
///   returns is dropped, so the model doesn't stay resident for nobody. The next request starts a fresh load.
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
        var loading: Flight?
        /// Requests holding the model, and the callers waiting on the load in flight (one pin each).
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

    /// A load in flight and the callers waiting for it.
    private struct Flight {
        let id = UUID()
        var task: Task<Void, Never>?
        var waiters: [UUID: CheckedContinuation<any Sendable, any Error>] = [:]
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
        // A cancelled caller loads nothing and runs nothing.
        try Task.checkCancellation()
        let model: Model = try await acquire(ref, policy: policy, load: load)
        do {
            try Task.checkCancellation()
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

    /// Requests holding `ref` and callers waiting on its load (tests).
    func pinCount(_ ref: ModelRef) -> Int { residents[ref]?.pins ?? 0 }

    // MARK: - Internals

    private func acquire<Model: Sendable>(
        _ ref: ModelRef, policy: ModelLifecyclePolicy?, load: @escaping @Sendable () async throws -> Model
    ) async throws -> Model {
        await ensureResident(ref, policy: policy)
        while true {
            guard var resident = residents[ref] else { preconditionFailure("residents are never removed") }
            if let model = resident.model {
                // Cast before pinning: a type mismatch must not leave the model pinned for good.
                let typed: Model = try cast(model, ref)
                resident.pins += 1
                resident.lastUsed = clock.now()
                residents[ref] = resident
                transition(ref, to: .running, idleTimer: false)
                return typed
            }
            if let flight = resident.loading {
                // Join the load in flight; ``finishLoad(_:flight:_:)`` installs the model and reports the state.
                resident.pins += 1
                residents[ref] = resident
                let joined = try await awaitLoad(ref, flight: flight.id)
                do {
                    return try cast(joined, ref)
                } catch {
                    relinquish(ref)
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
            var flight = Flight()
            let id = flight.id
            // Unstructured: the load outlives a caller that stops waiting, for the callers still waiting.
            flight.task = Task {
                let result: Result<any Sendable, any Error>
                do { result = .success(try await Self.runLoad(load)) } catch { result = .failure(error) }
                finishLoad(ref, flight: id, result)
            }
            resident = again
            resident.loading = flight
            resident.pins += 1
            residents[ref] = resident
            transition(ref, to: .loading, idleTimer: false)
            let model = try await awaitLoad(ref, flight: id)
            do {
                return try cast(model, ref)
            } catch {
                // Loaded, but not the type this caller wants: the model stays resident, unpinned by this caller.
                relinquish(ref)
                throw error
            }
        }
    }

    /// Waits for load `flight` of `ref`, holding the pin the caller took. Cancelling the caller ends the wait at once
    /// and gives the pin back (``abandon(_:flight:waiter:)``); the load isn't awaited directly, since waiting on an
    /// unstructured task's value ignores the waiter's cancellation.
    private func awaitLoad(_ ref: ModelRef, flight: UUID) async throws -> any Sendable {
        let waiter = UUID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<any Sendable, any Error>) in
                guard residents[ref]?.loading?.id == flight else {
                    // The flight ended between the caller's pin and here. It can't (nothing suspends in between); if
                    // it did, give the pin back rather than leave it held.
                    relinquish(ref)
                    continuation.resume(throwing: CancellationError())
                    return
                }
                residents[ref]?.loading?.waiters[waiter] = continuation
                if Task.isCancelled { abandon(ref, flight: flight, waiter: waiter) }
            }
        } onCancel: {
            Task { await self.abandon(ref, flight: flight, waiter: waiter) }
        }
    }

    /// A waiter was cancelled: it stops waiting and unpins. With no waiter left, the load is cancelled and detached;
    /// the model is left unloaded, and the next request starts a fresh load.
    private func abandon(_ ref: ModelRef, flight: UUID, waiter: UUID) {
        guard var resident = residents[ref], var loading = resident.loading, loading.id == flight,
              let continuation = loading.waiters.removeValue(forKey: waiter) else { return }
        resident.pins = max(0, resident.pins - 1)
        if loading.waiters.isEmpty {
            loading.task?.cancel()
            resident.loading = nil
        } else {
            resident.loading = loading
        }
        residents[ref] = resident
        continuation.resume(throwing: CancellationError())
        settleFailedLoad(ref)
    }

    /// Load `flight` ended. If callers still wait, a model is installed (pinned once per waiter) or the failure is
    /// theirs; a detached load's result is dropped.
    private func finishLoad(_ ref: ModelRef, flight: UUID, _ result: Result<any Sendable, any Error>) {
        guard var resident = residents[ref], let loading = resident.loading, loading.id == flight else { return }
        resident.loading = nil
        switch result {
        case .success(let model):
            resident.model = model
            resident.lastUsed = clock.now()
            residents[ref] = resident
            transition(ref, to: .running, idleTimer: false)
            for continuation in loading.waiters.values { continuation.resume(returning: model) }
        case .failure(let error):
            resident.pins = max(0, resident.pins - loading.waiters.count)
            residents[ref] = resident
            settleFailedLoad(ref)
            for continuation in loading.waiters.values { continuation.resume(throwing: error) }
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
