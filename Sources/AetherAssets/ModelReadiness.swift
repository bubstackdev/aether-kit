import Darwin
import Foundation
#if os(iOS)
import os
#endif

/// What a product says about running a model here: from the entry before anything downloads, and from the installation
/// once it's verified.
public enum ModelRuntimeVerdict: Sendable, Equatable {
    case runnable
    case unsupported(ModelReadiness.Unsupported)
    /// The model isn't one the product can run (for example the wrong installation contract).
    case invalid(String)
}

/// A product's own conditions for running a model, beside the entry's requirements (`ModelRuntimeCheck.language` in
/// `AetherLanguage`). Both checks must be cheap: no hashing, no loading.
public struct ModelRuntimeCheck: Sendable {
    public let entry: @Sendable (ModelCatalogEntry) -> ModelRuntimeVerdict
    public let installation: @Sendable (AssetReadLease) -> ModelRuntimeVerdict

    public init(entry: @escaping @Sendable (ModelCatalogEntry) -> ModelRuntimeVerdict,
                installation: @escaping @Sendable (AssetReadLease) -> ModelRuntimeVerdict) {
        self.entry = entry
        self.installation = installation
    }

    /// No conditions beyond the entry's requirements.
    public static let none = ModelRuntimeCheck(entry: { _ in .runnable }, installation: { _ in .runnable })
}

/// One model's readiness, the single state an app shows for it (docs/model-readiness.md): not installed, installing
/// with progress, verifying, ready, unsupported or failed.
///
/// Creating one checks the entry and the store without network access or hashing. An installed model is verified at
/// once (every byte is hashed, as `LocalAssetStore.acquire` always does) and becomes ready holding its read lease.
/// Nothing downloads until ``prepare()``. Unsupported is decided before downloading where the entry allows: the
/// product's check (the OS, for language), then the entry's platforms and memory requirement.
public final class ModelReadiness: @unchecked Sendable {
    @nonexhaustive public enum State: Sendable, Equatable {
        /// Not installed. `bytesPresent` are downloaded bytes a ``prepare()`` resumes from.
        case notInstalled(bytesPresent: UInt64, bytesTotal: UInt64)
        case installing(Progress)
        /// Hashing the installation before it's used.
        case verifying
        /// Verified and runnable here. The lease keeps the installation in place while anyone holds it; open the model
        /// from it (for example `AetherLanguageModel(installation:)`). The first request loads the model.
        case ready(AssetReadLease)
        case unsupported(Unsupported)
        case failed(Failure)

        public static func == (lhs: State, rhs: State) -> Bool {
            switch (lhs, rhs) {
            case let (.notInstalled(a, b), .notInstalled(c, d)): a == c && b == d
            case let (.installing(a), .installing(b)): a == b
            case (.verifying, .verifying): true
            case let (.ready(a), .ready(b)): a === b
            case let (.unsupported(a), .unsupported(b)): a == b
            case let (.failed(a), .failed(b)): a == b
            default: false
            }
        }
    }

    public struct Progress: Sendable, Equatable {
        @nonexhaustive public enum Phase: Sendable, Equatable {
            /// Checking the store and free space.
            case checking
            /// Fetching; `bytesDone` includes bytes a previous run left.
            case downloading
            /// Copying and hashing into the store.
            case installing
        }

        public let phase: Phase
        public let bytesDone: UInt64
        public let bytesTotal: UInt64
        /// Progress through the current phase, 0 to 1.
        public var fraction: Double { bytesTotal == 0 ? 0 : min(1, Double(bytesDone) / Double(bytesTotal)) }

        public init(phase: Phase, bytesDone: UInt64, bytesTotal: UInt64) {
            self.phase = phase; self.bytesDone = bytesDone; self.bytesTotal = bytesTotal
        }
    }

    @nonexhaustive public enum Unsupported: Sendable, Equatable {
        /// The product doesn't run on this OS (the Core AI products need macOS 27 or iOS 27).
        case operatingSystem
        /// The model is for another platform (for example a macOS export on an iPhone).
        case platform
        /// The model's memory requirement exceeds the device's physical memory (Mac) or the app's memory limit (iOS).
        case memory(requiredBytes: UInt64, availableBytes: UInt64)
    }

    @nonexhaustive public enum Failure: Error, Sendable, Equatable {
        /// Preparing failed; ``prepare()`` again to retry when the error is retryable.
        case preparation(PreparationError)
        /// The installation didn't verify or couldn't be opened. ``prepare()`` replaces a corrupt installation.
        case verification(String)
        /// The product can't run this model (``ModelRuntimeVerdict/invalid(_:)``).
        case invalidModel(String)

        public var isRetryable: Bool {
            switch self {
            case let .preparation(error): error.isRetryable
            case .verification: true
            case .invalidModel: false
            }
        }
    }

    /// What a memory requirement is compared with here: the app's memory limit on iOS (its footprint plus what it may
    /// still allocate), physical memory on a Mac and for an iOS app running on a Mac. nil when iOS doesn't report a limit
    /// (the process isn't an app); memory requirements then go unchecked.
    public static var deviceMemoryLimit: UInt64? { Device.currentMemoryLimit() }

    /// The device the requirements are compared with.
    struct Device: Sendable {
        let platform: ModelRequirements.Platform
        let memoryLimit: @Sendable () -> UInt64?

        static let current = Device(platform: currentPlatform, memoryLimit: currentMemoryLimit)

        static var currentPlatform: ModelRequirements.Platform {
            #if os(iOS)
            .ios
            #else
            .macos
            #endif
        }

        static func currentMemoryLimit() -> UInt64? {
            #if os(iOS)
            let process = ProcessInfo.processInfo
            if process.isiOSAppOnMac || process.isMacCatalystApp { return process.physicalMemory }
            // Zero means "not an app" (or already past the limit): no limit to compare with.
            let available = os_proc_available_memory()
            guard available > 0 else { return nil }
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
            let result = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
            }
            let footprint = result == KERN_SUCCESS ? info.phys_footprint : 0
            return footprint + UInt64(available)
            #else
            ProcessInfo.processInfo.physicalMemory
            #endif
        }
    }

    public let entry: ModelCatalogEntry
    public let preparation: ModelPreparation
    private let check: ModelRuntimeCheck
    private let device: Device
    private let lock = NSLock()
    private var state: State
    private var subscribers: [UUID: AsyncStream<State>.Continuation] = [:]
    /// The running preparation or verification, and its generation: a result from an older generation is dropped.
    private var work: Task<Void, Never>?
    private var generation = 0
    /// While `uninstall()` removes the installation: `prepare()` waits for it, and `refresh()` does nothing.
    private var uninstalling = false
    private var prepareAfterUninstall = false

    public convenience init(entry: ModelCatalogEntry, preparation: ModelPreparation, check: ModelRuntimeCheck = .none) {
        self.init(entry: entry, preparation: preparation, check: check, device: .current)
    }

    init(entry: ModelCatalogEntry, preparation: ModelPreparation, check: ModelRuntimeCheck, device: Device) {
        self.entry = entry; self.preparation = preparation; self.check = check; self.device = device
        state = .verifying
        let screened = screen()
        lock.withLock { evaluate(screened) }
    }

    /// Releasing the readiness cancels its preparation or verification (downloaded bytes stay) and ends its streams.
    deinit {
        work?.cancel()
        for continuation in subscribers.values { continuation.finish() }
    }

    /// The current state.
    public var current: State { lock.withLock { state } }

    /// The current state, then each change. A slow reader sees the latest state rather than every progress step.
    public var states: AsyncStream<State> {
        AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            let id = UUID()
            lock.withLock {
                subscribers[id] = continuation
                continuation.yield(state)
            }
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                self.lock.withLock { self.subscribers[id] = nil }
            }
        }
    }

    /// Downloads and installs the model, then verifies it, when it's not installed or failed. Checks the entry again
    /// first, so an entry that has become unsupported is never downloaded. Does nothing while installing, verifying or
    /// ready, when unsupported (``refresh()`` decides again), or after `invalidModel` (the same model would fail the same
    /// way). During ``uninstall()`` it starts once the removal ends. With `BackgroundTransport`, call it at launch to join
    /// a transfer that continued while the app wasn't running.
    public func prepare() {
        let screened = screen()
        lock.withLock {
            if uninstalling { prepareAfterUninstall = true; return }
            switch state {
            case .notInstalled, .failed(.preparation), .failed(.verification):
                if let screened { set(screened) } else { start(preparing: true) }
            case .installing, .verifying, .ready, .unsupported, .failed(.invalidModel): break
            }
        }
    }

    /// Stops a running preparation; downloaded bytes stay for the next ``prepare()``.
    public func cancel() {
        let screened = screen()
        lock.withLock {
            guard case .installing = state else { return }
            generation += 1
            work?.cancel()
            work = nil
            evaluate(screened)
        }
    }

    /// Decides again when not installed, unsupported or failed: for example after another process installed the model,
    /// after the app's memory limit changed, or to verify an installation again without downloading.
    public func refresh() {
        let screened = screen()
        lock.withLock {
            guard !uninstalling else { return }
            switch state {
            case .notInstalled, .unsupported, .failed:
                generation += 1
                evaluate(screened)
            case .installing, .verifying, .ready: break
            }
        }
    }

    /// Removes the installation and any downloaded bytes (except those another run is using). The state reads
    /// `notInstalled` while it runs, because the
    /// lease this readiness holds is dropped first. Fails with `PreparationError.busy` while installing, verifying or
    /// already uninstalling, or while a model opened from the lease is still alive; the installation is then verified
    /// again.
    public func uninstall() async throws {
        let id = try entry.installationID
        let ours: Int = try lock.withLock {
            guard !uninstalling else { throw PreparationError.busy }
            switch state {
            case .installing, .verifying: throw PreparationError.busy
            case .notInstalled, .ready, .unsupported, .failed: break
            }
            generation += 1
            uninstalling = true
            set(.notInstalled(bytesPresent: 0, bytesTotal: entry.byteCount))
            return generation
        }
        var removed = false
        defer {
            // Whatever happened, decide from what's on disk, and start a preparation asked for meanwhile.
            let screened = screen()
            lock.withLock {
                uninstalling = false
                let resume = prepareAfterUninstall && removed
                prepareAfterUninstall = false
                if generation == ours {
                    generation += 1
                    evaluate(screened)
                    if resume, case .notInstalled = state { start(preparing: true) }
                }
            }
        }
        do {
            try preparation.store.remove(id)
        } catch AssetStoreError.notInstalled {
        } catch AssetStoreError.busy {
            throw PreparationError.busy
        }
        removed = true
        // Downloads another run is using stay with it: the installation is gone, which is what was asked.
        do { try preparation.discardDownloads(entry) } catch PreparationError.busy {}
    }

    // MARK: - Evaluation

    /// Under the lock: settles what can be known without network access or hashing, and verifies an installed model.
    /// `screened` is ``screen()``'s verdict, taken outside the lock.
    private func evaluate(_ screened: State?) {
        if let screened { set(screened); return }
        do {
            switch try preparation.status(entry) {
            case .installed: start(preparing: false)
            case let .partial(present, total): set(.notInstalled(bytesPresent: present, bytesTotal: total))
            case let .notInstalled(total): set(.notInstalled(bytesPresent: 0, bytesTotal: total))
            }
        } catch {
            set(.failed(.preparation(ModelPreparation.mapped(error))))
        }
    }

    /// Unsupported or invalid from the entry alone: the product's check, then the platforms, then memory. Reads only
    /// immutable properties, and runs outside the lock, so a check may read ``current``.
    private func screen() -> State? {
        switch check.entry(entry) {
        case let .unsupported(reason): return .unsupported(reason)
        case let .invalid(reason): return .failed(.invalidModel(reason))
        case .runnable: break
        }
        if let platforms = entry.requirements?.platforms, !platforms.contains(device.platform) { return .unsupported(.platform) }
        if let required = entry.requirements?.memoryBytes, let limit = device.memoryLimit(), required > limit {
            return .unsupported(.memory(requiredBytes: required, availableBytes: limit))
        }
        return nil
    }

    /// Under the lock. The task holds this readiness weakly, so releasing it cancels the work.
    private func start(preparing: Bool) {
        generation += 1
        let ours = generation
        set(preparing ? .installing(Progress(phase: .checking, bytesDone: 0, bytesTotal: entry.byteCount)) : .verifying)
        work?.cancel()
        let publish: @Sendable (State) -> Void = { [weak self] next in self?.publish(next, ours) }
        work = Task { [weak self, entry, preparation, check] in
            let settled = await Self.run(entry: entry, preparation: preparation, check: check, preparing: preparing, publish: publish)
            guard let self else { return }
            let screened = settled == nil ? screen() : nil
            lock.withLock {
                guard generation == ours else { return }
                work = nil
                if let settled { set(settled) } else { evaluate(screened) }
            }
        }
    }

    /// Prepares (when asked), verifies and checks the installation. nil means start over from the store's status (a
    /// cancelled run, or an installation that vanished).
    private static func run(entry: ModelCatalogEntry, preparation: ModelPreparation, check: ModelRuntimeCheck, preparing: Bool,
                            publish: @Sendable (State) -> Void) async -> State? {
        if preparing {
            var outcome: PreparationEvent?
            for await event in preparation.prepare(entry) {
                switch event {
                case .checking: publish(.installing(Progress(phase: .checking, bytesDone: 0, bytesTotal: entry.byteCount)))
                case let .downloading(_, _, _, done, total): publish(.installing(Progress(phase: .downloading, bytesDone: done, bytesTotal: total)))
                case let .installing(done, total): publish(.installing(Progress(phase: .installing, bytesDone: done, bytesTotal: total)))
                case .installed, .failed, .cancelled: outcome = event
                }
            }
            switch outcome {
            case .installed: break
            case let .failed(error): return .failed(.preparation(error))
            default: return nil
            }
            publish(.verifying)
        }
        let lease: AssetReadLease
        do {
            lease = try await preparation.store.acquire(try entry.installationID)
        } catch AssetStoreError.notInstalled {
            return nil
        } catch is CancellationError {
            return nil
        } catch {
            return .failed(.verification(String(describing: error)))
        }
        switch check.installation(lease) {
        case .runnable: return .ready(lease)
        case let .unsupported(reason): return .unsupported(reason)
        case let .invalid(reason): return .failed(.invalidModel(reason))
        }
    }

    private func publish(_ next: State, _ ours: Int) {
        lock.withLock { if generation == ours { set(next) } }
    }

    private func set(_ next: State) {
        guard next != state else { return }
        state = next
        for continuation in subscribers.values { continuation.yield(next) }
    }
}
