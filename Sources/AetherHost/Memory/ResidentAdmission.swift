public import AetherCore
public import AetherRuntime

/// A generative pipeline's graphs and the memory they are admitted against, across generations (DESIGN.md §6.4).
///
/// One per pipeline, owned by its actor (not `Sendable`, like ``GraphResidency``). Each generation:
/// 1. ``begin(profile:shape:)`` plans residency from the ledger (or the graphs' files), against the coordinator's
///    available bytes plus what this pipeline already holds, or against the memory budget plus, for a budget of what
///    the process may still allocate, the bytes this pipeline's resident graphs already take; then reserves the plan's
///    peak and returns the residency to run in (graphs kept from the last generation when the mode and profile match).
/// 2. ``finish()`` after success: the reservation shrinks to what stays resident (every phase's held bytes when all
///    graphs stay loaded, nothing when they were released per phase), so a second pipeline on the same coordinator
///    can't take memory these graphs still use.
/// 3. ``release()`` after a failure, on ``begin(profile:shape:)``'s refusal, and on unload: every graph is released and
///    so is the reservation.
///
/// A refusal releases the resident graphs too: the plan already counts them as this pipeline's, so a plan that
/// doesn't fit wouldn't fit with them either.
public final class ResidentAdmission {
    public struct Settings {
        public var memory: MemoryBudget
        public var coordinator: ResourceCoordinator?
        public var ledger: FootprintLedger
        public var probe: GPUMemoryProbe
        /// The phase a manifest without `residency` measures its graphs under (``GraphResidency/beginPhase(_:)``).
        public var unphasedPhase: Manifest.Residency
        /// How many distinct values of ``begin(profile:shape:)``'s `shape` may be specialized before the graphs are
        /// released: each new shape of a graph loaded without the reshape hint keeps its own specialization's memory,
        /// which a footprint measured at one shape doesn't include. Released graphs reload from the specialization
        /// cache.
        public var maximumShapes: Int

        public init(
            memory: MemoryBudget, coordinator: ResourceCoordinator?, ledger: FootprintLedger, probe: GPUMemoryProbe,
            unphasedPhase: Manifest.Residency = .perStep, maximumShapes: Int = 4
        ) {
            self.memory = memory
            self.coordinator = coordinator
            self.ledger = ledger
            self.probe = probe
            self.unphasedPhase = unphasedPhase
            self.maximumShapes = max(1, maximumShapes)
        }
    }

    public let manifest: Manifest
    public let settings: Settings
    let device: Device
    /// The ledger's model key: the manifest id, the weights' source revision and the device's Core AI architecture, so
    /// another export of the model, or the same bundle on another device, never reuses these measurements.
    public let modelKey: String
    private let load: GraphResidency.Loader
    private let fileBytes: (String) -> Int64
    public private(set) var residency: GraphResidency?
    private var reservation: ResourceReservation?
    private var reservedBytes: Int64 = 0
    /// What the resident graphs hold (0 when none are): the last generation's plan of what stays loaded.
    public private(set) var residentBytes: Int64 = 0
    private var shapes: Set<Int> = []
    /// The lowest phase baseline of graphs released since the next residency began: their memory returns seconds
    /// late, so the next phase is measured from here (an overestimate, the direction admission prefers).
    private var releasedFloor: Int64?

    public init(
        manifest: Manifest, device: Device, settings: Settings, load: @escaping GraphResidency.Loader,
        fileBytes: @escaping (String) -> Int64
    ) {
        self.manifest = manifest
        self.device = device
        self.settings = settings
        modelKey = "\(manifest.id)@\(manifest.provenance?.sourceRevision ?? "unpinned")/\(device.platform.rawValue)-\(device.architecture)"
        self.load = load
        self.fileBytes = fileBytes
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { residency?.residentRoles ?? [] }

    /// The plan's inputs for `profile`.
    public func inputs(profile: String) -> ResidencyPlan.Inputs {
        ResidencyPlan.inputs(
            manifest: manifest, ledger: settings.ledger, model: modelKey, profile: profile,
            unphasedPhase: settings.unphasedPhase, fileBytes: fileBytes)
    }

    /// The budget a plan must fit now.
    public func budget() -> Int64 {
        if let coordinator = settings.coordinator {
            // This pipeline's own reservation is capacity it may plan with.
            return Int64(clamping: coordinator.snapshot.availableBytes) + reservedBytes
        }
        let budget = settings.memory.bytes(on: device)
        // What the process may still allocate already excludes this pipeline's resident graphs, which the plan counts.
        if case .processAvailable = settings.memory.rule { return budget + residentBytes }
        return budget
    }

    /// Plans and reserves one generation at `profile` (and, for graphs whose shape varies with the request, `shape`),
    /// returning the residency to run it in. Refuses with `capacity` before anything loads, releasing what was loaded.
    public func begin(profile: String, shape: Int? = nil) throws(AetherError) -> GraphResidency {
        if let shape, !shapes.contains(shape), shapes.count >= settings.maximumShapes { release() }
        let plan: ResidencyPlan
        do {
            plan = try ResidencyPlan.choose(inputs(profile: profile), budget: budget())
            try reserve(plan.peakBytes)
        } catch {
            release()
            throw error
        }
        if let shape { shapes.insert(shape) }
        if let residency, residency.mode == plan.mode, residency.profile == profile { return residency }
        if let old = residency {
            note(old.floor)
            old.releaseAll()
            shapes = shape.map { [$0] } ?? []
        }
        let fresh = GraphResidency(
            manifest: manifest, mode: plan.mode, ledger: settings.ledger, model: modelKey, profile: profile, probe: settings.probe,
            releasedBaseline: releasedFloor, load: load)
        releasedFloor = nil
        residency = fresh
        return fresh
    }

    /// After a generation that succeeded: the reservation is resized to what stays resident, as measured by the
    /// generation that just ran.
    ///
    /// When the measurement is larger than the plan and the coordinator can't grow the reservation to it, the graphs
    /// can't stay admitted: they are released along with the reservation, and this returns `false`. The generation's
    /// result is still the caller's to return (the work already ran, within the peak reserved for it); only the
    /// residency is given up, so the next generation plans and loads from scratch rather than keep graphs the
    /// coordinator doesn't account for.
    @discardableResult
    public func finish() -> Bool {
        guard let residency else {
            release()
            return false
        }
        switch residency.mode {
        case .allResident:
            residentBytes = inputs(profile: residency.profile).phases.values.reduce(0) { $0 + $1.heldBytes }
        case .perPhase:
            residentBytes = 0
        }
        guard residentBytes > 0, residency.residentRoles.isEmpty == false else {
            reservation?.release()
            reservation = nil
            reservedBytes = 0
            return true
        }
        do {
            try reserve(residentBytes)
            return true
        } catch {
            release()
            return false
        }
    }

    /// Releases every graph and the reservation (after a failure, or to unload).
    public func release() {
        if let residency { note(residency.floor) }
        residency?.releaseAll()
        residency = nil
        reservation?.release()
        reservation = nil
        reservedBytes = 0
        residentBytes = 0
        shapes = []
    }

    private func note(_ floor: Int64?) {
        guard let floor else { return }
        releasedFloor = min(releasedFloor ?? floor, floor)
    }

    private func reserve(_ bytes: Int64) throws(AetherError) {
        guard let coordinator = settings.coordinator else { return }
        let bytes = max(1, bytes)
        if let reservation {
            try reservation.resize(to: UInt64(bytes))
        } else {
            reservation = try coordinator.reserve(bytes: UInt64(bytes))
        }
        reservedBytes = bytes
    }
}
