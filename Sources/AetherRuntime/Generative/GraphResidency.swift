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

/// The process's Metal allocations, read as `MTLDevice.currentAllocatedSize`. `phys_footprint` doesn't see Core AI's
/// Metal memory (0.0.x measured 400 MiB of footprint against 8.57 GiB of Metal for a 4B model), so footprints are
/// measured with this. Injectable so the bookkeeping is testable.
public struct GPUMemoryProbe: Sendable {
    public var allocatedBytes: @Sendable () -> Int64

    public init(allocatedBytes: @escaping @Sendable () -> Int64) { self.allocatedBytes = allocatedBytes }

    /// The default Metal device's `currentAllocatedSize` (0 without a GPU).
    public static let system: GPUMemoryProbe = {
        let device = MTLCreateSystemDefaultDevice()
        return GPUMemoryProbe { Int64(device?.currentAllocatedSize ?? 0) }
    }()
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

        public init(model: String, profile: String, phase: Manifest.Residency) {
            self.model = model
            self.profile = profile
            self.phase = phase
        }

        public var description: String { "\(model)[\(profile)].\(phase.rawValue)" }
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
            let order = Manifest.Residency.allCases.filter { phases[$0] != nil }
            return order.indices.map { i in
                let previous = order.count > 1 ? phases[order[(i + order.count - 1) % order.count]]!.heldBytes : 0
                return previous + phases[order[i]]!.peakBytes
            }.max() ?? 0
        }
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
    /// residency count towards the phase they are loaded in when measured, and by their files otherwise.
    public static func inputs(
        manifest: Manifest, ledger: FootprintLedger, model: String? = nil, profile: String,
        fileBytes: (String) -> Int64
    ) -> Inputs {
        var phases: [Manifest.Residency: PhaseFootprint] = [:]
        var measured = true
        for phase in Manifest.Residency.allCases {
            let roles = manifest.graphs.filter { $0.value.residency == phase }.map(\.key)
            guard !roles.isEmpty else { continue }
            if let footprint = ledger.footprint(.init(model: model ?? manifest.id, profile: profile, phase: phase)) {
                phases[phase] = footprint
            } else {
                let files = roles.reduce(Int64(0)) { $0 + fileBytes($1) }
                phases[phase] = PhaseFootprint(peakBytes: files, heldBytes: files)
                measured = false
            }
        }
        let unphased = manifest.graphs.filter { $0.value.residency == nil }.map(\.key)
        // Measured phases already include them (they load in some phase); otherwise their files are added.
        if !unphased.isEmpty, !measured || phases.isEmpty {
            measured = false
            let files = unphased.reduce(Int64(0)) { $0 + fileBytes($1) }
            let into = phases.keys.min { $0.order < $1.order } ?? .perStep
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

/// Loads a driver's graphs on demand, measures each phase, and releases graphs per a ``ResidencyPlan``.
///
/// Non-`Sendable`, like ``Graph``: the driver's actor owns it. A phase is measured from ``beginPhase(_:)`` to
/// ``endPhase(_:)``: the Metal level is sampled every `samplingInterval` on a background thread, and at each load, each
/// ``sample()`` and the end. Measurements are of process-wide allocations, so work running alongside in the process
/// inflates them; the ledger keeps the maximum, which errs towards refusing.
public final class GraphResidency {
    /// Loads graph `role`.
    public typealias Loader = (_ role: String) async throws(AetherError) -> any GraphBackend

    public let mode: ResidencyPlan.Mode
    public let ledger: FootprintLedger
    public let model: String
    public let profile: String
    private let residency: [String: Manifest.Residency?]
    private let probe: GPUMemoryProbe
    private let samplingInterval: Duration?
    private let load: Loader
    private var loaded: [String: any GraphBackend] = [:]
    private var open: (phase: Manifest.Residency, baseline: Int64, meter: PeakMeter)?

    /// `samplingInterval` nil samples only at loads, ``sample()`` and the phase's end (tests).
    public init(
        manifest: Manifest, mode: ResidencyPlan.Mode, ledger: FootprintLedger, model: String? = nil, profile: String = "",
        probe: GPUMemoryProbe = .system, samplingInterval: Duration? = .milliseconds(5), load: @escaping Loader
    ) {
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

    /// Graph `role`, loading it if it isn't resident.
    public func graph(_ role: String) async throws(AetherError) -> any GraphBackend {
        if let graph = loaded[role] { return graph }
        guard residency[role] != nil else {
            throw .manifestInvalid(field: "graphs.\(role)", reason: "the manifest has no such graph")
        }
        let graph = try await load(role)
        loaded[role] = graph
        sample()
        return graph
    }

    /// Samples the Metal level into the open phase now (after a run, say).
    public func sample() { open?.meter.observe(probe.allocatedBytes()) }

    /// Starts measuring `phase`. A phase still open is ended first.
    public func beginPhase(_ phase: Manifest.Residency) {
        if let current = open { endPhase(current.phase) }
        let baseline = probe.allocatedBytes()
        let meter = PeakMeter(baseline)
        if let samplingInterval { meter.start(probe: probe, every: samplingInterval) }
        open = (phase, baseline, meter)
    }

    /// Ends `phase`: records its footprint and, per phase, releases the graphs whose residency is `phase`.
    public func endPhase(_ phase: Manifest.Residency) {
        if let current = open, current.phase == phase {
            current.meter.stop()
            let level = probe.allocatedBytes()
            current.meter.observe(level)
            ledger.record(.init(model: model, profile: profile, phase: phase),
                          PhaseFootprint(peakBytes: current.meter.peak - current.baseline, heldBytes: level - current.baseline))
            open = nil
        }
        if mode == .perPhase {
            for role in residentRoles where (residency[role] ?? nil) == phase { loaded[role] = nil }
        }
    }

    /// Releases every graph (the driver unloads). An open phase is discarded unrecorded.
    public func releaseAll() {
        open?.meter.stop()
        open = nil
        loaded.removeAll()
    }
}

/// The high-water mark of a phase, fed by its owner and optionally by a polling thread.
private final class PeakMeter: Sendable {
    private struct State { var peak: Int64; var running = false }
    private let state: Mutex<State>

    init(_ baseline: Int64) { state = Mutex(State(peak: baseline)) }

    var peak: Int64 { state.withLock { $0.peak } }

    func observe(_ level: Int64) { state.withLock { $0.peak = max($0.peak, level) } }

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
    /// A loader for an installed bundle: each role's `step` function (its `function`), with the manifest's compute
    /// and reshape hint.
    public static func bundleLoader(
        bundle: InstalledBundle, manifest: Manifest, device: Device,
        cachePolicy: GraphOptions.CachePolicy = .default, cache: GraphOptions.Cache = .default
    ) -> Loader {
        { role throws(AetherError) in
            guard let spec = manifest.graphs[role] else {
                throw .manifestInvalid(field: "graphs.\(role)", reason: "the manifest has no such graph")
            }
            let options = try GraphOptions(manifest: spec, at: "graphs.\(role)", cachePolicy: cachePolicy, cache: cache)
            return try await Graph(
                bundle: bundle, manifest: manifest, role: role, architecture: device.architecture, options: options)
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
