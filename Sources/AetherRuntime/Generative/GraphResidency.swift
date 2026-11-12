public import AetherCore
import Foundation
import Metal
import Synchronization

// Residency and measured footprints for the multi-graph generative drivers (DESIGN.md §6.4, MANIFEST.md §5.3).
//
// A driver's graphs each carry a `residency` phase (prompt → steps → image). Before a generation the driver asks
// ``ResidencyPlan/choose(_:budget:)`` whether everything can stay resident, or each phase's graphs must be released when
// the phase ends, or the request doesn't fit at all; it then reserves the plan's peak from the host's shared
// `ResourceCoordinator` (AetherHost). It plans with *measured* bytes: each phase's Metal high-water mark and what the
// phase still holds when it ends, kept in a ``FootprintLedger`` keyed by model and profile (a canvas, a frame count:
// whatever changes the shapes), and seedable with qualified values. Until a phase has been measured, its graph files'
// size stands in, and the plan says so (`isMeasured`).
//
// What the measurements have to capture (G0 gate, qwen3-0.6b's decoder on an M5 Max; 0.0.x saw the same with H3):
// - loading a graph allocates almost nothing (0.4 MiB for a 320 MiB graph): the memory arrives with its first run
//   (+1,387 MiB peak, +913 MiB held while the graph stays loaded), so a load delta measures nothing;
// - releasing a graph returns its memory only seconds later (about 4–5 s here; 0.0.x: 1–6 s), so the next phase
//   can start while the previous one's memory is still allocated.

/// The process's memory as admission sees it: the larger of `MTLDevice.currentAllocatedSize` and `phys_footprint`.
/// Neither counter is reliable alone: 0.0.x measured 400 MiB of footprint against 8.57 GiB of Metal for a 4B model,
/// while the H3 spike (lab/h3-spike) saw the footprint include all the Metal memory plus 85–150 MiB, and release show up
/// in the footprint about 1 s after the Metal counter. The larger one errs towards refusing. Injectable so the
/// bookkeeping is testable.
public struct GPUMemoryProbe: Sendable {
    public var allocatedBytes: @Sendable () -> Int64

    public init(allocatedBytes: @escaping @Sendable () -> Int64) { self.allocatedBytes = allocatedBytes }

    /// The larger of the default Metal device's `currentAllocatedSize` (0 without a GPU) and the process's
    /// `phys_footprint` (0 if it can't be read).
    public static let system: GPUMemoryProbe = {
        let device = MTLCreateSystemDefaultDevice()
        return GPUMemoryProbe { max(Int64(device?.currentAllocatedSize ?? 0), physFootprint()) }
    }()

    /// The process's `phys_footprint` from `TASK_VM_INFO`.
    static func physFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }
}

/// What one phase costs, above the allocations present when it began.
public struct PhaseFootprint: Sendable, Hashable {
    /// The highest level sampled during the phase.
    public var peakBytes: Int64
    /// The level when the phase ended, with its graphs still loaded (at most the peak).
    public var heldBytes: Int64

    public init(peakBytes: Int64, heldBytes: Int64) {
        self.peakBytes = max(0, peakBytes)
        self.heldBytes = max(0, min(heldBytes, peakBytes))
    }
}

/// Measured phase footprints, shared by whatever should learn from earlier generations (a host passes one to every
/// pipeline of a model; a driver may seed it with qualified values). Each entry keeps the largest values recorded.
/// In memory only.
public final class FootprintLedger: Sendable {
    public struct Key: Sendable, Hashable, CustomStringConvertible {
        /// The model (catalog id or manifest id).
        public var model: String
        /// What changes the shapes the graphs run at; `""` when nothing does.
        public var profile: String
        public var phase: Manifest.Residency
        /// How many shape specializations of the phase's graphs were resident when it was measured: graphs loaded
        /// without the reshape hint keep one per shape they ran at (``GraphResidency/specializations``), and each
        /// holds its own memory. 1 for a single shape, and for whatever doesn't vary by shape.
        public var specializations: Int = 1

        public init(model: String, profile: String, phase: Manifest.Residency) {
            self.model = model
            self.profile = profile
            self.phase = phase
        }

        public init(model: String, profile: String, phase: Manifest.Residency, specializations: Int) {
            self.init(model: model, profile: profile, phase: phase)
            self.specializations = max(1, specializations)
        }

        public var description: String {
            "\(model)[\(profile)].\(phase.rawValue)" + (specializations > 1 ? "×\(specializations)" : "")
        }
    }

    private let entries = Mutex<[Key: PhaseFootprint]>([:])

    public init() {}

    /// Records `footprint`, keeping the larger peak and the larger held bytes seen for `key`.
    public func record(_ key: Key, _ footprint: PhaseFootprint) {
        entries.withLock { entries in
            let old = entries[key]
            entries[key] = PhaseFootprint(peakBytes: max(old?.peakBytes ?? 0, footprint.peakBytes),
                                          heldBytes: max(old?.heldBytes ?? 0, footprint.heldBytes))
        }
    }

    public func footprint(_ key: Key) -> PhaseFootprint? { entries.withLock { $0[key] } }

    /// `key`'s footprint: measured, or for more specializations than were measured, an estimate from the same phase
    /// at fewer. From the largest count `j` below `key`'s that was measured, each further specialization adds what one
    /// added between 1 and `j`; with only one specialization measured, each adds that whole footprint's held bytes (a
    /// full copy of the phase's graphs, the conservative case). Nil when even one wasn't measured. The second value is
    /// false for an estimate.
    public func estimate(_ key: Key) -> (footprint: PhaseFootprint, measured: Bool)? {
        let all = entries.withLock { $0 }
        if let exact = all[key] { return (exact, true) }
        var one = key
        one.specializations = 1
        guard key.specializations > 1, let single = all[one] else { return nil }
        var base = (count: 1, footprint: single)
        for (other, footprint) in all where other.model == key.model && other.profile == key.profile
            && other.phase == key.phase && other.specializations > base.count && other.specializations < key.specializations {
            base = (other.specializations, footprint)
        }
        let extraHeld: Int64, extraPeak: Int64
        if base.count > 1 {
            let copies = Int64(base.count - 1)
            extraHeld = max(0, (base.footprint.heldBytes - single.heldBytes) / copies)
            extraPeak = max(0, (base.footprint.peakBytes - single.peakBytes) / copies)
        } else {
            extraHeld = single.heldBytes
            extraPeak = single.heldBytes
        }
        let more = Int64(key.specializations - base.count)
        return (PhaseFootprint(peakBytes: base.footprint.peakBytes + more * extraPeak,
                               heldBytes: base.footprint.heldBytes + more * extraHeld), false)
    }

    public var snapshot: [Key: PhaseFootprint] { entries.withLock { $0 } }
}

/// How a generation's graphs stay in memory, and the peak that implies.
public struct ResidencyPlan: Sendable, Hashable {
    public enum Mode: Sendable, Hashable {
        /// Every graph stays loaded across phases and generations.
        case allResident
        /// Each phase's graphs are released when the phase ends.
        case perPhase
    }

    /// What a plan is computed from: each phase's footprint, and whether every one of them was measured.
    public struct Inputs: Sendable, Hashable {
        public var phases: [Manifest.Residency: PhaseFootprint]
        public var isMeasured: Bool
        /// False when a graph has no residency: it then never leaves, so only ``Mode/allResident`` applies.
        public var allowsPerPhase: Bool

        public init(phases: [Manifest.Residency: PhaseFootprint], isMeasured: Bool, allowsPerPhase: Bool = true) {
            self.phases = phases
            self.isMeasured = isMeasured
            self.allowsPerPhase = allowsPerPhase
        }
    }

    public let mode: Mode
    public let peakBytes: Int64
    /// False when some phase was planned from its files' size (no measurement yet): the real peak is likely higher.
    public let isMeasured: Bool

    /// A plan whose peak was measured as a whole rather than computed from the phases (``ResidentAdmission``'s
    /// all-resident generations).
    package init(mode: Mode, peakBytes: Int64, isMeasured: Bool) {
        self.mode = mode
        self.peakBytes = peakBytes
        self.isMeasured = isMeasured
    }

    /// The peak of `mode`.
    /// - All resident: what every phase holds, plus the largest transient above that (`peak − held`).
    /// - Per phase: phases run in a cycle (prompt → steps → image → the next prompt), and a released phase's memory
    ///   comes back seconds late, so each phase's peak is counted on top of what the phase before it held.
    public static func peak(_ mode: Mode, _ phases: [Manifest.Residency: PhaseFootprint]) -> Int64 {
        switch mode {
        case .allResident:
            let held = phases.values.reduce(0) { $0 + $1.heldBytes }
            return held + (phases.values.map { $0.peakBytes - $0.heldBytes }.max() ?? 0)
        case .perPhase:
            let overlap = overlap(phases)
            return phases.map { phase, footprint in (overlap[phase] ?? 0) + footprint.peakBytes }.max() ?? 0
        }
    }

    /// Per phase, the released memory each phase's peak is counted on top of: what the phase before it in the cycle
    /// holds (none for a single phase). ``GraphResidency`` makes this true by waiting at a phase boundary while more
    /// than this is still returning (``ReleaseLog``).
    public static func overlap(_ phases: [Manifest.Residency: PhaseFootprint]) -> [Manifest.Residency: Int64] {
        let order = Manifest.Residency.allCases.filter { phases[$0] != nil }
        var overlap: [Manifest.Residency: Int64] = [:]
        for (i, phase) in order.enumerated() {
            overlap[phase] = order.count > 1 ? phases[order[(i + order.count - 1) % order.count]]!.heldBytes : 0
        }
        return overlap
    }

    /// Everything resident if that fits `budget`, else per phase if that fits, else
    /// ``AetherError/capacity(what:needed:available:)`` before anything loads.
    public static func choose(_ inputs: Inputs, budget: Int64) throws(AetherError) -> ResidencyPlan {
        let all = peak(.allResident, inputs.phases)
        if all <= budget { return ResidencyPlan(mode: .allResident, peakBytes: all, isMeasured: inputs.isMeasured) }
        guard inputs.allowsPerPhase else {
            throw .capacity(what: "memory for one generation (every graph resident)", needed: all, available: budget)
        }
        let phased = peak(.perPhase, inputs.phases)
        if phased <= budget { return ResidencyPlan(mode: .perPhase, peakBytes: phased, isMeasured: inputs.isMeasured) }
        throw .capacity(what: "memory for one generation (graphs released per phase)", needed: phased, available: budget)
    }

    /// The inputs for a manifest's graphs under `profile`: the ledger's measurements, or, for a phase not measured
    /// yet, the size of its graphs' files (`fileBytes` per role) as both its peak and its held bytes. Graphs without a
    /// residency count towards the phase they are loaded in when measured, and by their files otherwise. A manifest
    /// whose graphs have no residency at all is measured under `unphasedPhase` (the phase its driver opens, as Kokoro
    /// opens `perStep`), and that measurement is what it plans with.
    public static func inputs(
        manifest: Manifest, ledger: FootprintLedger, model: String? = nil, profile: String,
        unphasedPhase: Manifest.Residency = .perStep, fileBytes: (String) -> Int64
    ) -> Inputs {
        inputs(manifest: manifest, ledger: ledger, model: model, profile: profile, specializations: 1,
               unphasedPhase: unphasedPhase, fileBytes: fileBytes)
    }

    /// The inputs with `specializations` shape specializations resident (``FootprintLedger/estimate(_:)`` for a count
    /// not measured yet, which makes the inputs unmeasured).
    public static func inputs(
        manifest: Manifest, ledger: FootprintLedger, model: String? = nil, profile: String, specializations: Int,
        unphasedPhase: Manifest.Residency = .perStep, fileBytes: (String) -> Int64
    ) -> Inputs {
        func key(_ phase: Manifest.Residency) -> FootprintLedger.Key {
            .init(model: model ?? manifest.id, profile: profile, phase: phase, specializations: specializations)
        }
        var phases: [Manifest.Residency: PhaseFootprint] = [:]
        var measured = true
        for phase in Manifest.Residency.allCases {
            let roles = manifest.graphs.filter { $0.value.residency == phase }.map(\.key)
            guard !roles.isEmpty else { continue }
            if let (footprint, exact) = ledger.estimate(key(phase)) {
                phases[phase] = footprint
                if !exact { measured = false }
            } else {
                let files = roles.reduce(Int64(0)) { $0 + fileBytes($1) }
                phases[phase] = PhaseFootprint(peakBytes: files, heldBytes: files)
                measured = false
            }
        }
        let unphased = manifest.graphs.filter { $0.value.residency == nil }.map(\.key)
        if phases.isEmpty, !unphased.isEmpty,
           let (footprint, exact) = ledger.estimate(key(unphasedPhase)) {
            return Inputs(phases: [unphasedPhase: footprint], isMeasured: exact, allowsPerPhase: false)
        }
        // Measured phases already include them (they load in some phase); otherwise their files are added.
        if !unphased.isEmpty, !measured || phases.isEmpty {
            measured = false
            let files = unphased.reduce(Int64(0)) { $0 + fileBytes($1) }
            let into = phases.keys.min { $0.order < $1.order } ?? unphasedPhase
            let base = phases[into] ?? PhaseFootprint(peakBytes: 0, heldBytes: 0)
            phases[into] = PhaseFootprint(peakBytes: base.peakBytes + files, heldBytes: base.heldBytes + files)
        }
        return Inputs(phases: phases, isMeasured: measured, allowsPerPhase: unphased.isEmpty)
    }
}

extension Manifest.Residency {
    /// Position in a generation: prompt 0, steps 1, image 2.
    var order: Int { Self.allCases.firstIndex(of: self)! }
}

/// Released graphs whose memory may not be back yet: each release's bytes and when it happened. A released graph's
/// memory returns seconds late (about 5 s; 0.0.x saw 1–6 s), so a release counts as pending for `window` after it.
///
/// The per-phase plan counts each phase's peak on top of the phase before it only (``ResidencyPlan/overlap(_:)``).
/// Several short phases, or back-to-back generations (SD-Turbo's phases last well under a second), would overlap
/// more than that, so ``GraphResidency`` enforces a reclamation boundary: a per-phase phase begins only once no more
/// than the plan's overlap is pending (``reclaim(allowing:)``), waiting for the oldest releases to age out. The wait is
/// by time alone: the probe can't tell a returned release from other allocations.
///
/// One per ``ResidentAdmission`` (so it spans its residencies and generations); not `Sendable`, owned by the driver's
/// actor like the residency.
public final class ReleaseLog {
    /// The log's time source: tests advance a fake one.
    public struct Clock: Sendable {
        /// Monotonic time since some origin.
        public var now: @Sendable () -> Duration
        public var sleep: @Sendable (Duration) async throws -> Void

        public init(now: @escaping @Sendable () -> Duration, sleep: @escaping @Sendable (Duration) async throws -> Void) {
            self.now = now
            self.sleep = sleep
        }

        public static let continuous: Clock = {
            let origin = ContinuousClock.now
            return Clock(now: { ContinuousClock.now - origin }, sleep: { try await Task.sleep(for: $0) })
        }()
    }

    /// How long a release counts as pending (0.0.x's longest observed delay).
    public static let defaultWindow: Duration = .seconds(6)

    public let window: Duration
    private let clock: Clock
    private var entries: [(at: Duration, bytes: Int64)] = []

    public init(window: Duration = ReleaseLog.defaultWindow, clock: Clock = .continuous) {
        self.window = window
        self.clock = clock
    }

    /// Notes that graphs holding `bytes` were released now.
    public func record(_ bytes: Int64) {
        guard bytes > 0 else { return }
        entries.append((clock.now(), bytes))
    }

    /// The bytes released within the window.
    public var pendingBytes: Int64 {
        prune(clock.now())
        return entries.reduce(0) { $0 + $1.bytes }
    }

    /// Waits until at most `allowance` released bytes are pending, sleeping until the oldest releases age out. Throws
    /// `CancellationError` if the task is cancelled while it waits.
    public func reclaim(allowing allowance: Int64) async throws {
        while true {
            let now = clock.now()
            prune(now)
            guard entries.reduce(0, { $0 + $1.bytes }) > allowance, let oldest = entries.first else { return }
            try await clock.sleep(oldest.at + window - now)
        }
    }

    private func prune(_ now: Duration) {
        entries.removeAll { now - $0.at >= window }
    }
}

/// Loads a driver's graphs on demand, measures each phase, and releases graphs per a ``ResidencyPlan``.
///
/// Non-`Sendable`, like ``Graph``: the driver's actor owns it. A phase is measured from ``beginPhase(_:)`` to
/// ``endPhase(_:)``: the Metal level is sampled every `samplingInterval` on a background thread, and at each load, each
/// ``sample()`` and the end. Measurements are of process-wide allocations, so work running alongside in the process
/// inflates them; the ledger keeps the maximum, which errs towards refusing.
public final class GraphResidency {
    /// Loads graph `role`, or a bucket of it (a ``key(_:bucket:)``).
    public typealias Loader = (_ role: String) async throws(AetherError) -> any GraphBackend

    public let mode: ResidencyPlan.Mode
    public let ledger: FootprintLedger
    public let model: String
    public let profile: String
    private let residency: [String: Manifest.Residency?]
    private let probe: GPUMemoryProbe
    private let samplingInterval: Duration?
    private let load: Loader
    /// Releases whose memory may still be returning (shared with the admission's other residencies).
    public let releases: ReleaseLog
    /// Per phase, the released bytes the plan counted its peak on top of (``ResidencyPlan/overlap(_:)``); a per-phase
    /// phase begins once no more than this is pending. Absent for a phase: the ledger's held bytes of the phase before
    /// it.
    public var overlapAllowance: [Manifest.Residency: Int64] = [:]
    /// How many shape specializations the graphs hold in this generation (``FootprintLedger/Key/specializations``),
    /// set by whoever plans it (``ResidentAdmission``). All resident, each phase's footprint is recorded under it;
    /// per phase, graphs are released at every phase's end, so they're recorded as one.
    public var specializations: Int = 1
    private var recordedSpecializations: Int { mode == .allResident ? max(1, specializations) : 1 }
    private let phaseOrder: [Manifest.Residency]
    private var loaded: [String: any GraphBackend] = [:]
    private var open: (phase: Manifest.Residency, baseline: Int64, observed: Int64, meter: PeakMeter)?
    /// The baseline of graphs released and not yet measured past: the next phase measures from it.
    private var releasedBaseline: Int64?
    /// The lowest baseline any phase began at: where released memory, once returned, leaves the level.
    public private(set) var floor: Int64?
    /// All resident: what each phase's graphs hold so far. A phase that runs again (the next generation) adds to the
    /// same graphs (a new shape's specialization, say), so its footprint is the sum of its occurrences' growth, not the
    /// largest single one.
    private var heldSoFar: [Manifest.Residency: Int64] = [:]
    /// All resident: the specialization count each phase last ran at. A phase that runs again at the same count and
    /// loads nothing adds no graphs, so whatever its level grew by is working memory, not more held bytes.
    private var specializationsAtLastRun: [Manifest.Residency: Int] = [:]
    /// Whether the open phase has loaded a graph.
    private var loadedInPhase = false
    /// The lowest level a phase began at with no graph loaded: what the process held before this model's graphs.
    private var unloadedLevel: Int64?
    /// The highest level sampled since ``markWindow()`` (nil before any sample): what a whole generation reached.
    package private(set) var windowPeak: Int64?
    /// The roles resident at any point since ``markWindow()``: what a generation's ``windowPeak`` was reached with.
    package private(set) var windowRoles: Set<String> = []

    /// `samplingInterval` nil samples only at loads, ``sample()`` and the phase's end (tests). `releasedBaseline` is
    /// the baseline of graphs another residency of the model released just before (a profile change), whose memory
    /// may still be returning. `releases` is the log of released graphs (absent: a new one on the continuous clock).
    public init(
        manifest: Manifest, mode: ResidencyPlan.Mode, ledger: FootprintLedger, model: String? = nil, profile: String = "",
        probe: GPUMemoryProbe = .system, samplingInterval: Duration? = .milliseconds(5), releasedBaseline: Int64? = nil,
        releases: ReleaseLog = ReleaseLog(), load: @escaping Loader
    ) {
        self.releases = releases
        let phases = Set(manifest.graphs.values.compactMap(\.residency))
        phaseOrder = Manifest.Residency.allCases.filter(phases.contains)
        self.releasedBaseline = releasedBaseline
        self.mode = mode
        self.ledger = ledger
        self.model = model ?? manifest.id
        self.profile = profile
        residency = manifest.graphs.mapValues(\.residency)
        self.probe = probe
        self.samplingInterval = samplingInterval
        self.load = load
    }

    deinit { open?.meter.stop() }

    /// Roles currently loaded, sorted.
    public var residentRoles: [String] { loaded.keys.sorted() }

    /// Graph `role` (or a ``key(_:bucket:)``), loading it if it isn't resident.
    public func graph(_ role: String) async throws(AetherError) -> any GraphBackend {
        if let graph = loaded[role] { return graph }
        guard residency[Self.parse(role).role] != nil else {
            throw .manifestInvalid(field: "graphs.\(role)", reason: "the manifest has no such graph")
        }
        let graph = try await load(role)
        loaded[role] = graph
        windowRoles.insert(role)
        loadedInPhase = true
        sample()
        return graph
    }

    /// Graph `role` as its `buckets` entry `bucket` runs it (that entry's function or file), loading it if it isn't
    /// resident. Each bucket is its own graph, resident beside the role's others.
    public func graph(_ role: String, bucket: Int) async throws(AetherError) -> any GraphBackend {
        try await graph(Self.key(role, bucket: bucket))
    }

    /// The loader key of graph `role` at bucket `value`: `role[value]`.
    public static func key(_ role: String, bucket value: Int) -> String { "\(role)[\(value)]" }

    /// A loader key's role and bucket.
    public static func parse(_ key: String) -> (role: String, bucket: Int?) {
        guard key.hasSuffix("]"), let open = key.lastIndex(of: "["),
              let value = Int(key[key.index(after: open)..<key.index(before: key.endIndex)]) else { return (key, nil) }
        return (String(key[..<open]), value)
    }

    /// The manifest graph a loader key names: the role's, or its bucket's variant (``Manifest/Graph/bucket(_:)``).
    public static func spec(_ key: String, in manifest: Manifest) throws(AetherError) -> Manifest.Graph {
        let (role, bucket) = parse(key)
        guard let graph = manifest.graphs[role] else {
            throw .manifestInvalid(field: "graphs.\(role)", reason: "the manifest has no such graph")
        }
        guard let bucket else { return graph }
        guard let variant = graph.bucket(bucket) else {
            throw .manifestInvalid(field: "graphs.\(role).buckets", reason: "no function or file entry for bucket \(bucket)")
        }
        return variant
    }

    /// Samples the Metal level into the open phase now (after a run, say).
    public func sample() {
        let level = probe.allocatedBytes()
        open?.meter.observe(level)
        if open != nil { windowPeak = max(windowPeak ?? level, level) }
    }

    /// Starts a new ``windowPeak`` (the admission marks each generation).
    package func markWindow() {
        windowPeak = nil
        windowRoles = Set(loaded.keys)
    }

    /// Releases the resident graphs of `phase` alone, between phases, keeping every other phase's (a Krea 2 reference
    /// encode drops its vision tower and text encoder before the denoisers run again). The phase holds nothing
    /// afterwards: its next occurrence loads and is measured afresh. Its held bytes go in the release log.
    package func releasePhase(_ phase: Manifest.Residency) {
        let roles = loaded.keys.filter { (residency[Self.parse($0).role] ?? nil) == phase }
        guard !roles.isEmpty else { return }
        for role in roles { loaded[role] = nil }
        releases.record(heldSoFar[phase] ?? 0)
        heldSoFar[phase] = nil
        specializationsAtLastRun[phase] = nil
    }

    /// The lowest level a phase began at with no graph loaded (nil until one did).
    package var levelBeforeGraphs: Int64? { unloadedLevel }

    /// What the loaded graphs hold now, read on the probe at the call: the level now above the level before any graph
    /// of this residency loaded (``levelBeforeGraphs``), so Core AI's Metal memory and whatever the graphs kept since
    /// their last phase ended are counted. 0 with no graph loaded. Before any phase has begun (no level before the
    /// graphs is known) it is the held bytes measured so far.
    ///
    /// The probe reads the whole process: memory something else in the process allocated after these graphs loaded
    /// counts here too, and memory it freed since comes off. Between generations, with nothing else allocating, that is
    /// what the graphs hold.
    public var liveResidentBytes: Int64 {
        guard !loaded.isEmpty else { return 0 }
        guard let unloadedLevel else { return heldSoFar.values.reduce(0, +) }
        return max(0, probe.allocatedBytes() - unloadedLevel)
    }


    /// Starts measuring `phase`. A phase still open is ended first. After graphs were released (per phase, or by a
    /// profile change), the phase measures from the level before they loaded: their memory comes back seconds late,
    /// and a baseline that still includes it would under-count this phase by up to their held bytes. That
    /// overestimates instead, the direction admission prefers.
    ///
    /// Per phase, it first waits while more released memory may still be returning than the plan counted this phase
    /// on top of (``ReleaseLog/reclaim(allowing:)``); cancelling the task ends the wait with `CancellationError`.
    public func beginPhase(_ phase: Manifest.Residency) async throws {
        if let current = open { endPhase(current.phase) }
        if mode == .perPhase { try await releases.reclaim(allowing: allowance(phase)) }
        let observed = probe.allocatedBytes()
        var baseline = observed
        if let released = releasedBaseline {
            baseline = min(baseline, released)
            releasedBaseline = nil
        }
        floor = min(floor ?? baseline, baseline)
        if loaded.isEmpty { unloadedLevel = min(unloadedLevel ?? baseline, baseline) }
        let meter = PeakMeter(baseline, level: observed)
        if let samplingInterval { meter.start(probe: probe, every: samplingInterval) }
        loadedInPhase = false
        open = (phase, baseline, observed, meter)
    }

    /// The released bytes `phase` may begin on top of.
    private func allowance(_ phase: Manifest.Residency) -> Int64 {
        if let allowance = overlapAllowance[phase] { return allowance }
        guard phaseOrder.count > 1, let i = phaseOrder.firstIndex(of: phase) else { return 0 }
        let previous = phaseOrder[(i + phaseOrder.count - 1) % phaseOrder.count]
        return ledger.estimate(.init(model: model, profile: profile, phase: previous, specializations: recordedSpecializations))?
            .footprint.heldBytes ?? 0
    }

    /// Ends `phase`: records its footprint and, per phase, releases the graphs whose residency is `phase`.
    public func endPhase(_ phase: Manifest.Residency) {
        var started: Int64?
        var measuredHeld: Int64?
        if let current = open, current.phase == phase {
            started = current.observed
            current.meter.stop()
            let level = probe.allocatedBytes()
            current.meter.observe(level)
            windowPeak = max(windowPeak ?? level, current.meter.peak, level)
            if mode == .allResident {
                // Memory that came back while this phase ran (the level fell below where it began) was the other
                // phases' working memory, still allocated when they ended: they no longer hold it. It is taken off the
                // phase that ran last first, back through the cycle.
                var returned = max(0, current.observed - current.meter.minimum)
                if returned > 0, let i = phaseOrder.firstIndex(of: phase) {
                    for k in 1..<max(1, phaseOrder.count) where returned > 0 {
                        let other = phaseOrder[(i + phaseOrder.count - k) % phaseOrder.count]
                        let take = min(returned, heldSoFar[other] ?? 0)
                        heldSoFar[other] = (heldSoFar[other] ?? 0) - take
                        returned -= take
                    }
                }
            }
            let prior = mode == .allResident ? heldSoFar[phase] ?? 0 : 0
            var held: Int64, peak: Int64
            if mode == .allResident, !loadedInPhase, specializationsAtLastRun[phase] == recordedSpecializations {
                // Nothing loaded and no new specialization: the same graphs ran again. What the level grew by is the
                // run's working memory (a decoder's activations at the canvas size, say), which a later phase may
                // give back and this one take again: a transient, never more held bytes.
                // Its transient is what the level rose above everything the residency already counts as held (this
                // phase's included): working memory it takes back after another phase made it give it up is already
                // in its held bytes.
                held = prior
                let counted = (unloadedLevel ?? current.baseline) + heldSoFar.values.reduce(0, +)
                peak = prior + max(0, current.meter.peak - max(counted, unloadedLevel == nil ? current.baseline : counted))
            } else {
                // Memory another phase gave back while this one ran (working memory returned as this phase's graphs
                // loaded) would hide this phase's graphs: count the growth from the lowest level the phase reached.
                let from = min(current.baseline, current.meter.minimum)
                held = prior + (level - from)
                peak = prior + max(current.meter.peak - current.baseline, current.meter.rise, level - from)
            }
            if mode == .allResident { specializationsAtLastRun[phase] = recordedSpecializations }
            if mode == .allResident, let unloadedLevel {
                // The sum of occurrences' growth assumes each began with the phase's resident memory counted.
                // `phys_footprint` doesn't promise that: between requests part of it can drop out and come back when
                // the graph next runs (Depth Anything 3's gate: ~120 MiB per request, so 15 requests read 1.1 GiB
                // for a 330 MiB model). What the phase holds can't exceed the level above where the process stood
                // before any graph loaded, less what the other phases hold.
                let others = heldSoFar.filter { $0.key != phase }.values.reduce(0, +)
                held = min(held, max(prior, level - unloadedLevel - others))
                peak = min(peak, max(held, current.meter.peak - unloadedLevel - others))
            }
            measuredHeld = held
            if mode == .allResident { heldSoFar[phase] = held }
            ledger.record(.init(model: model, profile: profile, phase: phase, specializations: recordedSpecializations),
                          PhaseFootprint(peakBytes: peak, heldBytes: held))
            open = nil
        }
        if mode == .perPhase {
            let released = residentRoles.filter { (residency[$0] ?? nil) == phase }
            for role in released { loaded[role] = nil }
            if !released.isEmpty {
                releases.record(measuredHeld
                    ?? ledger.footprint(.init(model: model, profile: profile, phase: phase))?.heldBytes ?? 0)
            }
            // The next phase measures from where this one started, not from a level that may still hold these graphs.
            if !released.isEmpty, let start = started ?? floor { releasedBaseline = min(releasedBaseline ?? start, start) }
        }
    }

    /// Releases every graph (the driver unloads). An open phase is discarded unrecorded. What the graphs held goes in
    /// the release log: all resident, every phase's held bytes; per phase, the open phase's growth.
    public func releaseAll() {
        if !loaded.isEmpty {
            switch mode {
            case .allResident: releases.record(heldSoFar.values.reduce(0, +))
            case .perPhase: releases.record(open.map { probe.allocatedBytes() - $0.baseline } ?? 0)
            }
        }
        heldSoFar = [:]
        specializationsAtLastRun = [:]
        open?.meter.stop()
        open = nil
        loaded.removeAll()
    }
}

/// The high-water mark of a phase, fed by its owner and optionally by a polling thread.
private final class PeakMeter: Sendable {
    private struct State { var peak: Int64; var minimum: Int64; var rise: Int64 = 0; var running = false }
    private let state: Mutex<State>

    /// `baseline` is where the peak is counted from; `level` is the level when the phase began (at or above it).
    init(_ baseline: Int64, level: Int64? = nil) { state = Mutex(State(peak: baseline, minimum: level ?? baseline)) }

    var peak: Int64 { state.withLock { $0.peak } }
    /// The lowest level observed (from the phase's starting level on).
    var minimum: Int64 { state.withLock { $0.minimum } }
    /// The largest rise above an earlier lowest level: how far the phase's own allocations climbed after memory that
    /// was returning during it had come back.
    var rise: Int64 { state.withLock { $0.rise } }

    func observe(_ level: Int64) {
        state.withLock {
            $0.peak = max($0.peak, level)
            $0.minimum = min($0.minimum, level)
            $0.rise = max($0.rise, level - $0.minimum)
        }
    }

    func start(probe: GPUMemoryProbe, every interval: Duration) {
        state.withLock { $0.running = true }
        let (seconds, attoseconds) = interval.components
        let micros = UInt32(clamping: max(1, seconds * 1_000_000 + attoseconds / 1_000_000_000_000))
        let thread = Thread { [self] in
            while state.withLock({ $0.running }) {
                observe(probe.allocatedBytes())
                usleep(micros)
            }
        }
        thread.name = "aether.residency.sampler"
        thread.start()
    }

    func stop() { state.withLock { $0.running = false } }
}

extension GraphResidency {
    /// A loader for an installed bundle: each role's `step` function (its `function`, or a bucket's for a
    /// ``key(_:bucket:)``), with the manifest's compute and reshape hint.
    public static func bundleLoader(
        bundle: InstalledBundle, manifest: Manifest, device: Device,
        cachePolicy: GraphOptions.CachePolicy = .default, cache: GraphOptions.Cache = .default
    ) -> Loader {
        { key throws(AetherError) in
            let spec = try spec(key, in: manifest)
            let role = parse(key).role
            let options = try GraphOptions(manifest: spec, at: "graphs.\(role)", cachePolicy: cachePolicy, cache: cache)
            guard let function = spec.phases["step"] else {
                throw .manifestInvalid(field: "graphs.\(role).functions.step", reason: "the graph has no such phase")
            }
            guard let path = spec.file.resolve(architecture: device.architecture), let url = bundle.url(of: path) else {
                throw .manifestInvalid(field: "graphs.\(role).file", reason: "no file for architecture \(device.architecture)")
            }
            guard spec.phases.count > 1 else { return try await Graph(url: url, function: function, options: options) }
            // Several functions: one specialization, so they share the weights.
            let loaded = try await Graph.load(url: url, functions: Array(Set(spec.phases.values)).sorted(), options: options)
            return try GraphFunctionSet(functions: spec.phases.compactMapValues { loaded[$0] })
        }
    }

    /// Wraps `loader` so each graph is checked against its manifest role and `contract` before use (MANIFEST.md §2.1.1):
    /// its `step` function's signature (and, for a ``PhasedGraphBackend``, every other phase's) against the role's
    /// names, ranks and scalar kinds.
    public static func validating(_ loader: @escaping Loader, manifest: Manifest, contract: DriverContract) -> Loader {
        { key throws(AetherError) in
            let graph = try await loader(key)
            let role = parse(key).role
            guard let graphContract = contract.graph(role) else {
                throw .manifestInvalid(field: "graphs.\(role)", reason: "not a \(contract.driver) graph")
            }
            let spec = try spec(key, in: manifest)
            try SignatureValidator.validate(spec, phase: "step", at: "graphs.\(role)", against: graph.signature,
                                            contract: graphContract, tables: manifest.tables ?? [:])
            if let phased = graph as? any PhasedGraphBackend {
                for phase in spec.phases.keys.sorted() where phase != "step" {
                    guard let function = phased.function(forPhase: phase) else {
                        throw .manifestInvalid(field: "graphs.\(role).functions.\(phase)", reason: "the loaded graph has no such function")
                    }
                    try SignatureValidator.validate(spec, phase: phase, at: "graphs.\(role)", against: function.signature,
                                                    contract: graphContract, tables: manifest.tables ?? [:])
                }
            }
            return graph
        }
    }

    /// The size on disk of graph `role`'s file for `device` (0 when it isn't installed): the stand-in for a phase
    /// not yet measured.
    public static func fileBytes(bundle: InstalledBundle, manifest: Manifest, role: String, device: Device) -> Int64 {
        guard let path = manifest.graphs[role]?.file.resolve(architecture: device.architecture),
              let url = bundle.url(of: path) else { return 0 }
        return diskBytes(url)
    }
}

/// Bytes of the regular files under `url` (an `.aimodel` package is a directory).
func diskBytes(_ url: URL) -> Int64 {
    guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else {
        return 0
    }
    var total: Int64 = 0
    for case let file as URL in walker {
        let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        if values?.isRegularFile == true { total += Int64(values?.fileSize ?? 0) }
    }
    return total
}
