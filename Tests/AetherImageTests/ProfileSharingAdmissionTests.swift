import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing

/// Admission that keeps graphs across profiles (``ResidentAdmission/Settings/profileSharing``), on the SD-Turbo fixture
/// manifest (files 10 / 100 / 10) with the fakes of ``ShapeSpecializationAdmissionTests``: each graph adds its bytes
/// while loaded, and the denoiser adds a full copy for each further canvas it specializes at, as Core AI does for a
/// graph loaded without the reshape hint. The prompt phase (the text encoder) is shared: it never specializes by canvas.
///
/// Before (app #35): every profile change released every graph, so Atelier's 512² phone pictures between 768×1344
/// portraits reloaded the text encoder (5–6 s), the denoisers and decoder (2–3 s) and paid their first step's
/// specialization again (about 9 s), each way.
@Suite("Profile sharing admission")
struct ProfileSharingAdmissionTests {
    typealias Memory = ShapeSpecializationAdmissionTests.Memory
    typealias Held = ShapeSpecializationAdmissionTests.Held
    static let files = ShapeSpecializationAdmissionTests.files

    final class Loads: Sendable {
        let log = Mutex<[String]>([])
        func append(_ role: String) { log.withLock { $0.append(role) } }
        var all: [String] { log.withLock { $0 } }
    }

    static func admission(
        budget: Int64, ledger: FootprintLedger = .init(), memory: Memory, loads: Loads = Loads(),
        sharing: Set<Manifest.Residency>? = [.perPrompt]
    ) throws -> ResidentAdmission {
        var settings = ResidentAdmission.Settings(memory: .bytes(budget), coordinator: nil, ledger: ledger, probe: memory.probe)
        settings.profileSharing = sharing
        return try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac, settings: settings,
            load: { role in
                loads.append(role)
                return Held(files[role] ?? 0, copy: role == "denoiser" ? 100 : 0, memory: memory)
            },
            fileBytes: { files[$0] ?? 0 })
    }

    /// The canvas a profile names, as the denoiser's specialization key.
    static func canvas(_ profile: String) -> Int { profile == "512x512" ? 1 : 2 }

    /// One generation at `profile`: the prompt phase, then the denoiser and decoder, which specialize for the canvas.
    @discardableResult
    static func generate(_ admission: ResidentAdmission, _ profile: String) async throws -> GraphResidency {
        let residency = try admission.begin(profile: profile, shape: 7)
        for (phase, role) in [(Manifest.Residency.perPrompt, "textEncoder"), (.perStep, "denoiser"), (.perImage, "vaeDecoder")] {
            try await residency.beginPhase(phase)
            let graph = try await residency.graph(role) as! Held
            if phase != .perPrompt { graph.specialize(canvas(profile)) }
            residency.endPhase(phase)
        }
        admission.finish()
        return residency
    }

    /// An encode alone at `profile` (Krea 2's canvas-aware encode): the prompt phase only.
    static func encode(_ admission: ResidentAdmission, _ profile: String) async throws {
        let residency = try admission.begin(profile: profile, shape: 7)
        try await residency.beginPhase(.perPrompt)
        _ = try await residency.graph("textEncoder")
        residency.endPhase(.perPrompt)
        admission.finish()
    }

    @Test func aProfileChangeThatFitsKeepsEveryGraph() async throws {
        let memory = Memory(), ledger = FootprintLedger(), loads = Loads()
        let admission = try Self.admission(budget: 1 << 40, ledger: ledger, memory: memory, loads: loads)
        let first = try await Self.generate(admission, "768x1344")
        #expect(first.mode == .allResident && memory.above == 120)
        let second = try await Self.generate(admission, "512x512")
        #expect(second === first && second.profile == "512x512" && second.specializations == 2)
        #expect(memory.above == 220 && loads.all == ["textEncoder", "denoiser", "vaeDecoder"])
        // Back and forth: the same graphs, nothing loads, nothing more is held.
        for profile in ["768x1344", "512x512", "768x1344"] {
            #expect(try await Self.generate(admission, profile) === first)
        }
        #expect(memory.above == 220 && loads.all.count == 3 && admission.residentBytes == 220)

        // The prompt phase is measured once, under no profile; the others per profile and count.
        let model = admission.modelKey
        #expect(ledger.footprint(.init(model: model, profile: "", phase: .perPrompt))?.heldBytes == 10)
        #expect(ledger.footprint(.init(model: model, profile: "768x1344", phase: .perPrompt)) == nil)
        #expect(ledger.footprint(.init(model: model, profile: "768x1344", phase: .perStep))?.heldBytes == 100)
        #expect(ledger.footprint(.init(model: model, profile: "512x512", phase: .perStep, specializations: 2))?.heldBytes == 200)
    }

    @Test func aProfileThatDoesntFitBesideTheOthersKeepsTheSharedPhase() async throws {
        let memory = Memory(), loads = Loads()
        let admission = try Self.admission(budget: 150, memory: memory, loads: loads)
        let first = try await Self.generate(admission, "768x1344")
        // A second canvas would hold 220: only the denoiser and decoder go, the text encoder stays.
        let second = try await Self.generate(admission, "512x512")
        #expect(second === first && second.specializations == 1 && memory.above == 120)
        #expect(loads.all == ["textEncoder", "denoiser", "vaeDecoder", "denoiser", "vaeDecoder"])
        for profile in ["768x1344", "512x512"] {
            let residency = try await Self.generate(admission, profile)
            #expect(residency.mode == .allResident && memory.above <= 150)
        }
        #expect(loads.all.filter { $0 == "textEncoder" }.count == 1)
    }

    /// An encode for a canvas that never generates (or a conditioning-only profile) isn't counted as a specialization.
    @Test func anEncodeAloneAddsNoSpecialization() async throws {
        let memory = Memory(), loads = Loads()
        let admission = try Self.admission(budget: 1 << 40, memory: memory, loads: loads)
        try await Self.encode(admission, "conditioning")
        let residency = try await Self.generate(admission, "768x1344")
        #expect(residency.specializations == 1 && memory.above == 120)
        try await Self.encode(admission, "512x512")
        try await Self.encode(admission, "768x1344")
        #expect(try await Self.generate(admission, "768x1344").specializations == 1)
        #expect(loads.all == ["textEncoder", "denoiser", "vaeDecoder"])
    }

    /// Without sharing (the default, every pipeline but Krea 2's), a profile change releases every graph as before.
    @Test func withoutSharingAProfileChangeReleasesEverything() async throws {
        let memory = Memory(), loads = Loads()
        let admission = try Self.admission(budget: 1 << 40, memory: memory, loads: loads, sharing: nil)
        let first = try await Self.generate(admission, "768x1344")
        let second = try await Self.generate(admission, "512x512")
        #expect(second !== first && memory.above == 120)
        #expect(loads.all == ["textEncoder", "denoiser", "vaeDecoder", "textEncoder", "denoiser", "vaeDecoder"])
    }

    /// Per phase, nothing stays between generations, so there's nothing to keep: a profile change starts afresh.
    @Test func perPhaseResidencyStartsAfreshOnAProfileChange() async throws {
        let memory = Memory()
        let admission = try Self.admission(budget: 110, memory: memory)
        let first = try await Self.generate(admission, "768x1344")
        #expect(first.mode == .perPhase)
        let second = try await Self.generate(admission, "512x512")
        #expect(second !== first && second.mode == .perPhase && second.profile == "512x512")
    }
}
