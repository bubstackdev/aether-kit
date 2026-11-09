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
/// 3. ``release()`` after a failure and on ``begin(profile:shape:)``'s refusal: every graph is released and so is the
///    reservation. ``unload()`` does the same and also drops the bundle's reader lease.
///
/// **Reader lease.** Given a bundle from a ``ModelStore``, the admission holds its reader lease
/// (``InstalledBundle/readLease()``) from creation until ``unload()`` or deinit, so the store refuses to remove files
/// the pipeline reads, lazily included (H3 streams weights from the bundle mid-generation). After ``unload()`` the next
/// ``begin(profile:shape:)`` takes the lease again, refusing with `inUse` or `unknownModel` before anything loads.
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
        /// How long a release counts as pending, and its clock (``ReleaseLog``; tests pass a fake clock).
        public var releaseWindow: Duration
        public var releaseClock: ReleaseLog.Clock

        public init(
            memory: MemoryBudget, coordinator: ResourceCoordinator?, ledger: FootprintLedger, probe: GPUMemoryProbe,
            unphasedPhase: Manifest.Residency = .perStep, maximumShapes: Int = 4,
            releaseWindow: Duration = ReleaseLog.defaultWindow, releaseClock: ReleaseLog.Clock = .continuous
        ) {
            self.releaseWindow = releaseWindow
            self.releaseClock = releaseClock
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
    /// The ledger's model key: the manifest id, the weights' source revision, the export's identity (the first 16 hex
    /// digits of ``Manifest/contentDigest``, so another export or quantization of the same weights has its own) and
    /// the device's platform and Core AI architecture, so another export of the model, or the same bundle on another
    /// device, never reuses these measurements.
    public let modelKey: String
    private let load: GraphResidency.Loader
    private let fileBytes: (String) -> Int64
    private let bundle: InstalledBundle?
    private var lease: BundleReadLease?
    /// Released graphs whose memory may still be returning, across this pipeline's residencies and generations.
    public let releases: ReleaseLog
    public private(set) var residency: GraphResidency?
    private var reservation: ResourceReservation?
    private var reservedBytes: Int64 = 0
    /// What the resident graphs hold (0 when none are): the last generation's plan of what stays loaded.
    public private(set) var residentBytes: Int64 = 0
    private var shapes: Set<Int> = []
    /// The lowest phase baseline of graphs released since the next residency began: their memory returns seconds
    /// late, so the next phase is measured from here (an overestimate, the direction admission prefers).
    private var releasedFloor: Int64?

    /// `bundle` is the installed bundle the graphs load from, if any: its reader lease is taken here (see above).
    public init(
        manifest: Manifest, device: Device, settings: Settings, bundle: InstalledBundle? = nil,
        load: @escaping GraphResidency.Loader, fileBytes: @escaping (String) -> Int64
    ) throws(AetherError) {
        self.manifest = manifest
        self.device = device
        self.settings = settings
        modelKey = Self.modelKey(manifest: manifest, device: device)
        self.load = load
        self.fileBytes = fileBytes
        self.bundle = bundle
        releases = ReleaseLog(window: settings.releaseWindow, clock: settings.releaseClock)
        lease = try bundle?.readLease()
    }

    /// Whether the bundle's reader lease is held now.
    public var holdsLease: Bool { lease != nil }

    /// The ledger key for `manifest`'s graphs on `device` (``modelKey``).
    public static func modelKey(manifest: Manifest, device: Device) -> String {
        let export = manifest.contentDigest.prefix(16)
        return "\(manifest.id)@\(manifest.provenance?.sourceRevision ?? "unpinned")#\(export)/\(device.platform.rawValue)-\(device.architecture)"
    }

    /// Roles currently loaded.
    public var residentRoles: [String] { residency?.residentRoles ?? [] }

    /// The plan's inputs for `profile`.
    public func inputs(profile: String) -> ResidencyPlan.Inputs {
        inputs(profile: profile, specializations: 1)
    }

    /// The plan's inputs for `profile` with `specializations` shapes resident.
    public func inputs(profile: String, specializations: Int) -> ResidencyPlan.Inputs {
        ResidencyPlan.inputs(
            manifest: manifest, ledger: settings.ledger, model: modelKey, profile: profile,
            specializations: specializations, unphasedPhase: settings.unphasedPhase, fileBytes: fileBytes)
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
    ///
    /// Each shape kept resident is planned as its own specialization (``FootprintLedger/Key/specializations``). A new
    /// shape whose added specialization doesn't fit with every graph resident releases the graphs and plans again
    /// with this shape alone (they reload from the specialization cache); only that plan failing refuses.
    public func begin(profile: String, shape: Int? = nil) throws(AetherError) -> GraphResidency {
        if lease == nil, let bundle { lease = try bundle.readLease() }
        if let shape, !shapes.contains(shape), shapes.count >= settings.maximumShapes { release() }
        let current = residency.map { $0.profile == profile } ?? false ? shapes : []
        var count = max(1, current.count + (shape.map { current.contains($0) ? 0 : 1 } ?? 0))
        var plan: ResidencyPlan
        do {
            // Keeping the shapes resident must fit with every graph resident: per phase, graphs are released at each
            // phase's end, so no specialization outlives its generation and keeping them buys nothing.
            let kept = count > 1
                ? try? ResidencyPlan.choose(inputs(profile: profile, specializations: count), budget: budget()) : nil
            if let kept, kept.mode == .allResident {
                plan = kept
            } else {
                if count > 1 {
                    // The new shape's specialization doesn't fit beside the others: start over with this shape alone.
                    release()
                    count = 1
                }
                plan = try ResidencyPlan.choose(inputs(profile: profile, specializations: 1), budget: budget())
            }
            try reserve(plan.peakBytes)
        } catch {
            release()
            throw error
        }
        if let shape { shapes.insert(shape) }
        let overlap = ResidencyPlan.overlap(inputs(profile: profile, specializations: count).phases)
        if let residency, residency.mode == plan.mode, residency.profile == profile {
            residency.overlapAllowance = overlap
            residency.specializations = count
            return residency
        }
        if let old = residency {
            note(old.floor)
            old.releaseAll()
            shapes = shape.map { [$0] } ?? []
        }
        let fresh = GraphResidency(
            manifest: manifest, mode: plan.mode, ledger: settings.ledger, model: modelKey, profile: profile, probe: settings.probe,
            releasedBaseline: releasedFloor, releases: releases, load: load)
        fresh.overlapAllowance = overlap
        fresh.specializations = 1
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
            residentBytes = inputs(profile: residency.profile, specializations: residency.specializations).phases.values
                .reduce(0) { $0 + $1.heldBytes }
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

    /// Releases every graph and the reservation, and drops the bundle's reader lease: the store may remove the bundle
    /// until the next ``begin(profile:shape:)``.
    public func unload() {
        release()
        lease = nil
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
