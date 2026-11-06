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
        ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: memory, coordinator: coordinator, ledger: ledger, probe: probe, maximumShapes: maximumShapes),
            load: { _ in Inert() }, fileBytes: { files[$0] ?? 0 })
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
        let phone = ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: Device(platform: .iOS, architecture: "h19p"),
            settings: .init(memory: .bytes(1 << 40), coordinator: nil, ledger: .init(), probe: GPUMemoryProbe { 0 }),
            load: { _ in Inert() }, fileBytes: { _ in 0 })
        #expect(mac.modelKey != phone.modelKey && mac.modelKey.hasPrefix("sd-turbo@"))
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
