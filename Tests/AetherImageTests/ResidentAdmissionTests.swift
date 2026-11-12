import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing

/// Pre-release review findings 1, 2, 3 and 5 (`ResidentAdmission`, AetherHost), on the SD-Turbo fixture manifest (files
/// 10 / 100 / 10 for the prompt, steps and image graphs) and a hand-moved memory level.
@Suite("Resident admission")
struct ResidentAdmissionTests {
    final class Level: Sendable {
        let bytes = Mutex<Int64>(1_000)
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in bytes.withLock { $0 } } }
    }

    final class Inert: GraphBackend {
        let signature = GraphSignature(function: "f", inputs: [], states: [], outputs: [])
        func run(_ bindings: inout Bindings) async throws(AetherError) {}
    }

    static let files: [String: Int64] = ["textEncoder": 10, "denoiser": 100, "vaeDecoder": 10]

    static func admission(
        memory: MemoryBudget = .bytes(1 << 40), coordinator: ResourceCoordinator? = nil, ledger: FootprintLedger = .init(),
        probe: GPUMemoryProbe = GPUMemoryProbe { 0 }, maximumShapes: Int = 4
    ) throws -> ResidentAdmission {
        try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: memory, coordinator: coordinator, ledger: ledger, probe: probe, maximumShapes: maximumShapes),
            load: { _ in Inert() }, fileBytes: { files[$0] ?? 0 })
    }

    /// Second code review, batch C item 2: each residency waits on the plan's overlap and shares the admission's
    /// release log, so a release by an earlier residency (a profile change here) still counts at the next boundary.
    @Test func residenciesShareTheReleaseLogAndThePlansOverlap() async throws {
        let slept = Mutex<[Duration]>([])
        let now = Mutex<Duration>(.zero)
        let clock = ReleaseLog.Clock(now: { now.withLock { $0 } },
                                     sleep: { d in now.withLock { $0 += d }; slept.withLock { $0.append(d) } })
        let level = Level()
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(115), coordinator: nil, ledger: .init(), probe: level.probe,
                            releaseClock: clock),
            load: { _ in Inert() }, fileBytes: { Self.files[$0] ?? 0 })
        let first = try admission.begin(profile: "a")
        #expect(first.mode == .perPhase && first.releases === admission.releases)
        #expect(first.overlapAllowance == [.perPrompt: 10, .perStep: 10, .perImage: 100])
        try await first.beginPhase(.perPrompt)
        _ = try await first.graph("textEncoder")
        level.bytes.withLock { $0 += 10 }
        first.endPhase(.perPrompt)
        try await first.beginPhase(.perStep)
        _ = try await first.graph("denoiser")
        level.bytes.withLock { $0 += 100 }
        admission.release()  // a failure mid-steps: 10 + 100 pending
        let second = try admission.begin(profile: "a")
        #expect(second !== first && second.releases === admission.releases)
        try await second.beginPhase(.perPrompt)  // may begin on top of 10 only: waits out both releases
        #expect(slept.withLock { $0 } == [ReleaseLog.defaultWindow] && admission.releases.pendingBytes == 0)
    }

    static func loadAll(_ residency: GraphResidency) async throws {
        for role in ["textEncoder", "denoiser", "vaeDecoder"] { _ = try await residency.graph(role) }
    }

    /// Finding 2: graphs that stay loaded keep their bytes reserved after the generation, and a second pipeline on the
    /// coordinator can't take them; unloading returns them. (Before: the reservation ended with the call.)
    @Test func residentGraphsStayReserved() async throws {
        let coordinator = try ResourceCoordinator(capacityBytes: 1_000)
        let admission = try Self.admission(coordinator: coordinator)
        let residency = try admission.begin(profile: "512x512")
        #expect(residency.mode == .allResident && coordinator.snapshot.reservedBytes == 120)
        try await Self.loadAll(residency)
        admission.finish()
        #expect(admission.residentBytes == 120 && coordinator.snapshot.reservedBytes == 120)
        #expect(throws: AetherError.self) { _ = try coordinator.reserve(bytes: 900) }
        // The next generation plans with its own reservation as capacity: 10 free + 120 its own.
        let other = try coordinator.reserve(bytes: 870)
        #expect(try admission.begin(profile: "512x512") === residency)
        admission.finish()
        admission.release()
        #expect(coordinator.snapshot.reservedBytes == 870 && admission.residentRoles.isEmpty)
        other.release()
    }

    /// A cancelled generation keeps graphs that were all measured, its reservation resized to what they hold (as after
    /// a success, but recording nothing of the cut-short run); graphs its open phase loaded were never measured, so a
    /// cancellation then releases everything, as a failure does.
    @Test func aCancellationKeepsMeasuredGraphsReserved() async throws {
        let coordinator = try ResourceCoordinator(capacityBytes: 1_000)
        let admission = try Self.admission(coordinator: coordinator)
        try await Self.loadAll(try admission.begin(profile: "512x512"))
        admission.finish()
        #expect(admission.residentBytes == 120)
        // The next generation, warm, cancelled during its steps.
        let warm = try admission.begin(profile: "512x512")
        try await warm.beginPhase(.perStep)
        #expect(admission.keepAfterCancellation())
        #expect(admission.residentRoles == ["denoiser", "textEncoder", "vaeDecoder"])
        #expect(admission.residentBytes == 120 && coordinator.snapshot.reservedBytes == 120)
        #expect(try admission.begin(profile: "512x512") === warm, "the next generation reuses the residency")
        admission.finish()

        let cold = try Self.admission(coordinator: coordinator)
        let residency = try cold.begin(profile: "512x512")
        try await residency.beginPhase(.perStep)
        _ = try await residency.graph("denoiser")
        #expect(cold.keepAfterCancellation() == false)
        #expect(cold.residentRoles.isEmpty && cold.residentBytes == 0)
        #expect(coordinator.snapshot.reservedBytes == 120, "only the first admission's graphs stay reserved")
        admission.release()
    }

    /// Third review, P1-2: when the generation measures more than it planned and the coordinator can't grow the
    /// reservation, `finish()` kept the graphs on the smaller reservation (held 200, reserved 1, 99 free to others).
    /// Now the graphs and the reservation are released and `finish()` says so.
    @Test func aMeasurementTheCoordinatorCantHoldReleasesTheResidency() async throws {
        let coordinator = try ResourceCoordinator(capacityBytes: 100)
        let ledger = FootprintLedger()
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(1 << 40), coordinator: coordinator, ledger: ledger, probe: GPUMemoryProbe { 0 }),
            load: { _ in Inert() }, fileBytes: { _ in 0 })
        let residency = try admission.begin(profile: "512x512")
        #expect(residency.mode == .allResident && coordinator.snapshot.reservedBytes == 1)
        try await Self.loadAll(residency)
        // What the run measured: far more than the 1-byte estimate, and more than the coordinator has.
        ledger.record(.init(model: admission.modelKey, profile: "512x512", phase: .perStep),
                      PhaseFootprint(peakBytes: 200, heldBytes: 200))
        #expect(admission.finish() == false)
        #expect(admission.residentRoles.isEmpty && residency.residentRoles.isEmpty)
        #expect(admission.residentBytes == 0 && coordinator.snapshot.reservedBytes == 0)
    }

    /// Finding 2: a refusal releases the resident graphs and the reservation (before: they stayed loaded, unaccounted).
    @Test func aRefusalReleasesWhatWasResident() async throws {
        let coordinator = try ResourceCoordinator(capacityBytes: 1_000)
        let ledger = FootprintLedger()
        let admission = try Self.admission(coordinator: coordinator, ledger: ledger)
        try await Self.loadAll(try admission.begin(profile: "512x512"))
        admission.finish()
        ledger.record(.init(model: admission.modelKey, profile: "1024x1024", phase: .perStep),
                      PhaseFootprint(peakBytes: 5_000, heldBytes: 5_000))
        #expect(throws: AetherError.self) { _ = try admission.begin(profile: "1024x1024") }
        #expect(admission.residentRoles.isEmpty && coordinator.snapshot.reservedBytes == 0)
    }

    /// Finding 3: a budget of what the process may still allocate already excludes this pipeline's resident graphs,
    /// which the all-resident plan counts again; their bytes are added back. A fixed budget isn't.
    @Test func processAvailableBudgetsAddBackResidentBytes() async throws {
        let device = LatentDiffusionTests.mac
        let available = try Self.admission(memory: MemoryBudget(.processAvailable))
        try await Self.loadAll(try available.begin(profile: "512x512"))
        available.finish()
        #expect(available.budget() == MemoryBudget(.processAvailable).bytes(on: device) + 120)
        let fixed = try Self.admission(memory: .bytes(500))
        try await Self.loadAll(try fixed.begin(profile: "512x512"))
        fixed.finish()
        #expect(fixed.budget() == 500)
    }

    /// Finding 5: graphs whose shape follows the request specialize once per shape; past `maximumShapes` distinct shapes
    /// they are released before the next generation, so memory doesn't grow past what was measured.
    @Test func newShapesPastTheLimitReleaseTheGraphs() async throws {
        let admission = try Self.admission(maximumShapes: 2)
        let first = try admission.begin(profile: "512x512", shape: 20)
        try await Self.loadAll(first)
        admission.finish()
        #expect(try admission.begin(profile: "512x512", shape: 31) === first)
        admission.finish()
        #expect(try admission.begin(profile: "512x512", shape: 20) === first)  // a shape already specialized
        admission.finish()
        let third = try admission.begin(profile: "512x512", shape: 44)
        #expect(third !== first && third.residentRoles.isEmpty)
    }

    /// Finding 1: a manifest whose graphs have no residency (Kokoro) plans with what it measured under its phase; before,
    /// `phases` stayed empty and the plan always came from the files.
    /// Second review (overlap with finding 5): measurements are keyed by the weights' revision and the device, not
    /// the manifest id alone.
    @Test func ledgerKeysNameTheRevisionAndTheDevice() throws {
        let mac = try Self.admission()
        let phone = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: Device(platform: .iOS, architecture: "h19p"),
            settings: .init(memory: .bytes(1 << 40), coordinator: nil, ledger: .init(), probe: GPUMemoryProbe { 0 }),
            load: { _ in Inert() }, fileBytes: { _ in 0 })
        #expect(mac.modelKey != phone.modelKey && mac.modelKey.hasPrefix("sd-turbo@"))
    }

    /// Third review, P2-11: the key also names the export, so another export or quantization of the same weights (same
    /// id and revision) doesn't reuse these measurements.
    @Test func ledgerKeysNameTheExport() throws {
        let manifest = try LatentDiffusionTests.manifest()
        var requantized = manifest
        requantized.precision = ["weights": .string("int8")]
        let mac = LatentDiffusionTests.mac
        #expect(ResidentAdmission.modelKey(manifest: manifest, device: mac) != ResidentAdmission.modelKey(manifest: requantized, device: mac))
        #expect(ResidentAdmission.modelKey(manifest: manifest, device: mac) == ResidentAdmission.modelKey(manifest: try LatentDiffusionTests.manifest(), device: mac))
        #expect(try Self.admission().modelKey == ResidentAdmission.modelKey(manifest: manifest, device: mac))
    }

    @Test func unphasedManifestsPlanWithTheirMeasurement() throws {
        let url = LatentDiffusionTests.manifestURL.deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "kokoro/aether.json")
        let kokoro = try Manifest.decode(Data(contentsOf: url))
        #expect(kokoro.graphs.values.allSatisfy { $0.residency == nil })
        let ledger = FootprintLedger()
        let unmeasured = ResidencyPlan.inputs(manifest: kokoro, ledger: ledger, profile: "frames<=1024") { _ in 1_000 }
        #expect(!unmeasured.isMeasured && ResidencyPlan.peak(.allResident, unmeasured.phases) == 2_000)
        ledger.record(.init(model: kokoro.id, profile: "frames<=1024", phase: .perStep),
                      PhaseFootprint(peakBytes: 3_400, heldBytes: 3_200))
        let measured = ResidencyPlan.inputs(manifest: kokoro, ledger: ledger, profile: "frames<=1024") { _ in 1_000 }
        #expect(measured.isMeasured && !measured.allowsPerPhase)
        #expect(try ResidencyPlan.choose(measured, budget: 1 << 40).peakBytes == 3_400)
    }
}
