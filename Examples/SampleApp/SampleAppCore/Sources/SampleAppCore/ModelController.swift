import AetherAssets
import AetherKit
import AetherLanguage
import Foundation
import Observation

/// The model screen's logic: one catalog entry's readiness, the model opened once it's ready, its lifecycle state, and
/// the process's memory pressure.
///
/// It keeps no `ModelReadiness.State` itself: a ready state holds the installation's read lease, and anything holding the
/// lease keeps `uninstall()` from removing the installation. The lease lives only in readiness and in the open model.
@MainActor
@Observable
public final class ModelController {
    public let entry: ModelCatalogEntry
    public private(set) var display: ReadinessDisplay
    /// A short name for the current readiness state, for logs and the qualification record.
    public private(set) var stateName = "starting"
    public private(set) var model: AetherLanguageModel?
    public private(set) var lifecycleState: ModelLifecycleState?
    public private(set) var pressure: MemoryPressureLevel = MemoryPressure.level
    /// The last action that failed (for example a delete refused while the model is busy).
    public private(set) var actionError: String?
    /// While a delete runs, the screen offers no actions and further deletes are ignored.
    public private(set) var isDeleting = false
    public var lifecycle: LifecycleDisplay { LifecycleDisplay(state: lifecycleState, pressure: pressure) }

    /// Called on the main actor whenever the open model changes (opened when ready, nil when dropped). The app passes it
    /// to the chat and structured screens; it's awaited, so they let go of the model before a delete runs.
    @ObservationIgnored public var modelDidChange: (@MainActor (AetherLanguageModel?) async -> Void)?
    /// Every state change, for the qualification record.
    @ObservationIgnored public var stateDidChange: (@MainActor (String) -> Void)?
    /// Every lifecycle change of the open model, for the qualification record.
    @ObservationIgnored public var lifecycleDidChange: (@MainActor (ModelLifecycleState) -> Void)?

    @ObservationIgnored private let readiness: ModelReadiness
    @ObservationIgnored private let lifecyclePolicy: ModelLifecyclePolicy
    @ObservationIgnored private var observers: [Task<Void, Never>] = []
    @ObservationIgnored private var lifecycleObserver: Task<Void, Never>?

    public init(entry: ModelCatalogEntry, preparation: ModelPreparation, check: ModelRuntimeCheck = .language,
                lifecycle: ModelLifecyclePolicy = .platformDefault) {
        self.entry = entry
        readiness = ModelReadiness(entry: entry, preparation: preparation, check: check)
        lifecyclePolicy = lifecycle
        display = ReadinessDisplay(title: "Checking", detail: "", actions: [], tone: .busy)
    }

    /// The app's store: Application Support/Models in the app's container, kept out of iCloud and device backups.
    public static func defaultStore() throws -> LocalAssetStore {
        let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        var root = support.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try root.setResourceValues(values)
        return try LocalAssetStore(root: root)
    }

    /// The transport this platform uses: a background session on iOS, so downloads continue while the app is suspended;
    /// the foreground session on the Mac, where the app isn't suspended.
    public static func defaultTransport() -> any ModelTransport {
        #if os(iOS)
        BackgroundTransport(identifier: backgroundSessionIdentifier)
        #else
        URLSessionTransport.urlSession
        #endif
    }

    public static let backgroundSessionIdentifier = "aether-sample.model-downloads"

    /// Starts following readiness and memory pressure. With a background transport, a download the system carried on
    /// while the app wasn't running is joined by the first ``perform(_:)`` of `.download`.
    public func start() {
        guard observers.isEmpty else { return }
        let states = readiness.states
        observers.append(Task { [weak self] in
            for await state in states {
                guard let self else { return }
                await self.apply(state)
            }
        })
        observers.append(Task { [weak self] in
            for await level in MemoryPressure.levels {
                guard let self else { return }
                self.pressure = level
            }
        })
    }

    public func stop() {
        observers.forEach { $0.cancel() }
        observers = []
        lifecycleObserver?.cancel()
    }

    public func perform(_ action: ReadinessDisplay.Action) async {
        guard !isDeleting else { return }
        actionError = nil
        switch action {
        case .download, .retry: readiness.prepare()
        case .cancel: readiness.cancel()
        case .refresh: readiness.refresh()
        case .delete: await delete()
        }
    }

    /// Drops the model everywhere (screens first), then removes the installation and downloaded bytes.
    public func delete() async {
        guard !isDeleting else { return }
        isDeleting = true
        defer { isDeleting = false }
        await dropModel()
        do {
            try await readiness.uninstall()
        } catch {
            actionError = "Couldn't delete: \(error)"
        }
    }

    private func apply(_ state: ModelReadiness.State) async {
        display = ReadinessDisplay(state)
        stateName = Self.name(state)
        stateDidChange?(stateName)
        if case let .ready(lease) = state {
            guard model == nil, !isDeleting else { return }
            let model = AetherLanguageModel(installation: lease, lifecycle: lifecyclePolicy)
            guard model.availability == .available else {
                display = ReadinessDisplay(title: "Can't open the model", detail: "\(model.availability)", actions: [.delete], tone: .problem)
                return
            }
            self.model = model
            lifecycleState = model.lifecycleState
            let states = model.lifecycleStates
            lifecycleObserver = Task { [weak self] in
                for await state in states {
                    self?.lifecycleState = state
                    self?.lifecycleDidChange?(state)
                }
            }
            await modelDidChange?(model)
        } else if model != nil {
            await dropModel()
        }
    }

    private func dropModel() async {
        lifecycleObserver?.cancel()
        lifecycleObserver = nil
        guard model != nil else { return }
        await modelDidChange?(nil)
        model = nil
        lifecycleState = nil
    }

    public static func name(_ state: ModelReadiness.State) -> String {
        switch state {
        case let .notInstalled(present, total): "notInstalled(\(present)/\(total))"
        case let .installing(progress): "installing(\(progress.phase))"
        case .verifying: "verifying"
        case .ready: "ready"
        case let .unsupported(reason): "unsupported(\(reason))"
        case let .failed(failure): "failed(\(failure))"
        @unknown default: "\(state)"
        }
    }
}
