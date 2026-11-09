import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing

/// Admission when a graph's run leaves working memory allocated after its phase ends, until a later load gives it
/// back, as measured 2026-11-09 on Krea 2 realism at 1024², 8 steps, M5 Max 64 GB (a 48 GiB budget). The decoder's
/// activations, about 11.5 GB for 52 MB of weights, were still allocated when the image phase ended, and came back
/// while the next prompt's text encoder loaded, so the level fell during that phase although the encoder added about
/// 7.8 GB. The next run allocated them again.
///
/// On the SD-Turbo fixture manifest (one graph per phase) at a tenth of a gigabyte per unit: text encoder 78, denoiser
/// 303 (a full copy for each further shape), decoder 1; a step run's working memory 20, a decode's 115. Every graph
/// resident, a generation peaks at 78 + 303 + 1 + 115 + 20 = 517 above the level before any graph, under a budget of
/// 560.
///
/// Before: the image phase counted its lingering activations as held and then, at the next run, added them again
/// (nothing had loaded), the prompt phase measured nothing for an encoder that loaded while they came back, and the
/// phases' sum no longer fit once the encoder was measured again. The admission released the graphs at every phase
/// from then on and planned per phase forever: every render loaded every graph again. The traced renders: a
/// prompt of 14 rows (rendered twice) and one of 29 (twice).
@Suite("Lingering working memory admission")
struct LingeringWorkingMemoryAdmissionTests {
    final class Memory: @unchecked Sendable {
        let level = Mutex<Int64>(1_000)
        struct Weak { weak var graph: Held? }
        private let graphs = Mutex<[Weak]>([])
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in level.withLock { $0 } } }
        func add(_ bytes: Int64) { level.withLock { $0 += bytes } }
        var above: Int64 { level.withLock { $0 } - 1_000 }
        func register(_ graph: Held) { graphs.withLock { $0.append(Weak(graph: graph)) } }
        func unregister(_ graph: Held) { graphs.withLock { $0.removeAll { $0.graph == nil || $0.graph === graph } } }
        /// A load gives back every graph's lingering working memory first, then allocates its own weights.
        func returnWorkingMemory() { for graph in graphs.withLock({ $0.compactMap(\.graph) }) { graph.returnWorkingMemory() } }
    }

    final class Held: GraphBackend, @unchecked Sendable {
        let signature = GraphSignature(function: "f", inputs: [], states: [], outputs: [])
        let memory: Memory
        let copy: Int64
        let work: Int64
        private let state = Mutex<(bytes: Int64, working: Int64, shapes: Set<Int>)>((0, 0, []))
        init(_ bytes: Int64, copy: Int64, work: Int64, memory: Memory) {
            self.memory = memory
            self.copy = copy
            self.work = work
            memory.returnWorkingMemory()
            // A load takes a while (seconds for Krea 2's segments): long enough for the residency's sampler to see the
            // level after the working memory came back and before the weights are allocated.
            Thread.sleep(forTimeInterval: 0.02)
            state.withLock { $0.bytes = bytes }
            memory.add(bytes)
            memory.register(self)
        }
        deinit {
            memory.unregister(self)
            memory.add(-state.withLock { $0.bytes + $0.working })
        }
        /// One run at `shape`: a new shape's specialization after the first adds a copy; the run's working memory is
        /// allocated unless it still lingers from the last run.
        func run(_ shape: Int) {
            let added: Int64 = state.withLock { s in
                var added: Int64 = 0
                if s.shapes.insert(shape).inserted, s.shapes.count > 1 { s.bytes += copy; added += copy }
                if s.working == 0 { s.working = work; added += work }
                return added
            }
            memory.add(added)
        }
        func returnWorkingMemory() { memory.add(-state.withLock { s in defer { s.working = 0 }; return s.working }) }
        func run(_ bindings: inout Bindings) async throws(AetherError) {}
    }

    static let weights: [String: Int64] = ["textEncoder": 78, "denoiser": 303, "vaeDecoder": 1]
    static let working: [String: Int64] = ["textEncoder": 0, "denoiser": 20, "vaeDecoder": 115]
    static let budget: Int64 = 560

    static func admission(memory: Memory) throws -> ResidentAdmission {
        try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(budget), coordinator: nil, ledger: FootprintLedger(), probe: memory.probe),
            load: { role in Held(weights[role] ?? 0, copy: role == "denoiser" ? 303 : 0, work: working[role] ?? 0, memory: memory) },
            fileBytes: { weights[$0] ?? 0 })
    }

    /// A Krea 2 render as its pipeline runs one with conditioning reuse: the prompt encoded on its own (planned against
    /// the last canvas, "conditioning" before any), then the steps and the image from that conditioning.
    final class Renders {
        let admission: ResidentAdmission
        let memory: Memory
        var lastProfile: String?
        var loads: [[String]] = []
        init(_ admission: ResidentAdmission, _ memory: Memory) { self.admission = admission; self.memory = memory }

        @discardableResult
        func render(shape: Int, encode: Bool = true) async throws -> GraphResidency {
            var loaded: [String] = []
            if encode {
                let residency = try admission.begin(profile: lastProfile ?? "conditioning", shape: shape)
                let before = Set(residency.residentRoles)
                try await residency.beginPhase(.perPrompt)
                (try await residency.graph("textEncoder") as! Held).run(shape)
                residency.endPhase(.perPrompt)
                admission.finish()
                loaded += Set(residency.residentRoles).subtracting(before)
            }
            let residency = try admission.begin(profile: "512x512", shape: shape)
            lastProfile = "512x512"
            let before = Set(residency.residentRoles)
            for (phase, role) in [(Manifest.Residency.perStep, "denoiser"), (.perImage, "vaeDecoder")] {
                try await residency.beginPhase(phase)
                (try await residency.graph(role) as! Held).run(shape)
                residency.endPhase(phase)
            }
            admission.finish()
            loads.append((loaded + Set(residency.residentRoles).subtracting(before)).sorted())
            return residency
        }
    }

    /// The traced sequence: 14, 14 again (another prompt), 29 (a new length), 29 again (the same prompt, conditioning
    /// reused). The graphs stay resident throughout; the new length releases them and reloads alone (one shape), as
    /// a second copy of the denoiser doesn't fit; the length repeated after it loads nothing.
    @Test func aRepeatedLengthKeepsEveryGraphResident() async throws {
        let memory = Memory()
        let renders = Renders(try Self.admission(memory: memory), memory)
        let mug = try await renders.render(shape: 14)
        #expect(mug.mode == .allResident)
        let bicycle = try await renders.render(shape: 14)
        #expect(bicycle === mug && bicycle.mode == .allResident && renders.loads[1] == ["textEncoder"])
        #expect(memory.above <= Self.budget)

        let lighthouseA = try await renders.render(shape: 29)
        #expect(lighthouseA !== bicycle, "a second copy doesn't fit: the graphs reload for the new length alone")
        #expect(lighthouseA.mode == .allResident && lighthouseA.specializations == 1)
        #expect(renders.loads[2] == ["denoiser", "textEncoder", "vaeDecoder"])

        let lighthouseB = try await renders.render(shape: 29, encode: false)
        #expect(lighthouseB === lighthouseA && lighthouseB.mode == .allResident)
        #expect(renders.loads[3] == [], "a repeated length loads nothing")
        #expect(memory.above <= Self.budget)
    }

    /// Many renders at one length: the phases' held bytes don't grow with each run when nothing loads.
    @Test func heldBytesDontAddUpWhenNothingLoads() async throws {
        let memory = Memory(), ledger = FootprintLedger()
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(1 << 40), coordinator: nil, ledger: ledger, probe: memory.probe),
            load: { role in Held(Self.weights[role] ?? 0, copy: 0, work: Self.working[role] ?? 0, memory: memory) },
            fileBytes: { Self.weights[$0] ?? 0 })
        let renders = Renders(admission, memory)
        for _ in 0..<5 { try await renders.render(shape: 14) }
        let image = FootprintLedger.Key(model: admission.modelKey, profile: "512x512", phase: .perImage)
        let step = FootprintLedger.Key(model: admission.modelKey, profile: "512x512", phase: .perStep)
        // Measured once as the graphs loaded (weights and the first run's working memory still allocated at the end).
        #expect(ledger.footprint(image)?.heldBytes == 116)
        #expect(ledger.footprint(step)?.heldBytes == 323)
        #expect(renders.loads.dropFirst().allSatisfy { $0 == ["textEncoder"] || $0 == [] })
    }

    /// The text encoder's load gives the decoder's working memory back, so the level falls over its phase: the
    /// encoder is still measured at its weights, counted from the lowest level the phase reached.
    @Test func aLoadIsMeasuredFromTheLowestLevelOfItsPhase() async throws {
        let memory = Memory(), ledger = FootprintLedger()
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(1 << 40), coordinator: nil, ledger: ledger, probe: memory.probe),
            load: { role in Held(Self.weights[role] ?? 0, copy: 0, work: Self.working[role] ?? 0, memory: memory) },
            fileBytes: { Self.weights[$0] ?? 0 })
        let renders = Renders(admission, memory)
        try await renders.render(shape: 14)
        try await renders.render(shape: 14)
        let prompt = FootprintLedger.Key(model: admission.modelKey, profile: "512x512", phase: .perPrompt)
        #expect(ledger.footprint(prompt)?.heldBytes == 78)
    }

    /// The phases' sum, seeded too high (another pipeline's conservative measurements, say), says per phase; a
    /// generation measured with every graph resident says it fits, and the graphs stay resident.
    @Test func aMeasuredGenerationKeepsTheGraphsResident() async throws {
        let memory = Memory()
        let admission = try Self.admission(memory: memory)
        let renders = Renders(admission, memory)
        try await renders.render(shape: 14)
        try await renders.render(shape: 14)
        for (phase, footprint) in [(Manifest.Residency.perStep, PhaseFootprint(peakBytes: 340, heldBytes: 330)),
                                   (.perImage, PhaseFootprint(peakBytes: 180, heldBytes: 150))] {
            admission.settings.ledger.record(.init(model: admission.modelKey, profile: "512x512", phase: phase), footprint)
        }
        #expect(try ResidencyPlan.choose(admission.inputs(profile: "512x512"), budget: Self.budget).mode == .perPhase)
        let next = try await renders.render(shape: 14, encode: false)
        #expect(next.mode == .allResident && renders.loads.last == [])
    }

    /// Per phase while another pipeline holds part of the memory, then all resident again once it gives it back: the
    /// per-phase generations' measurements don't keep the graphs per phase.
    @Test func aPerPhasePlanReturnsToAllResidentOnceItFits() async throws {
        let memory = Memory()
        let coordinator = try ResourceCoordinator(capacityBytes: UInt64(Self.budget))
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(1 << 40), coordinator: coordinator, ledger: FootprintLedger(), probe: memory.probe),
            load: { role in Held(Self.weights[role] ?? 0, copy: 0, work: Self.working[role] ?? 0, memory: memory) },
            fileBytes: { Self.weights[$0] ?? 0 })
        let renders = Renders(admission, memory)
        try await renders.render(shape: 14)
        try await renders.render(shape: 14)
        // Released while idle; another pipeline takes 60.
        admission.release()
        let other = try coordinator.reserve(bytes: 60)
        let squeezed = try await renders.render(shape: 14)
        #expect(squeezed.mode == .perPhase)
        try await renders.render(shape: 14, encode: false)
        other.release()
        let back = try await renders.render(shape: 14, encode: false)
        #expect(back.mode == .allResident)
        let again = try await renders.render(shape: 14, encode: false)
        #expect(again === back && renders.loads.last == [])
    }

    /// A generation measured with only some phases' graphs (an encode on its own, 78) says nothing about one that loads
    /// the rest: when the phases' sum doesn't fit, it plans per phase rather than all resident at 78.
    @Test func anEncodeAloneDoesntMeasureAWholeGeneration() async throws {
        let memory = Memory()
        let admission = try Self.admission(memory: memory)
        let renders = Renders(admission, memory)
        renders.lastProfile = "512x512"
        let residency = try admission.begin(profile: "512x512", shape: 14)
        try await residency.beginPhase(.perPrompt)
        (try await residency.graph("textEncoder") as! Held).run(14)
        residency.endPhase(.perPrompt)
        admission.finish()
        for (phase, footprint) in [(Manifest.Residency.perStep, PhaseFootprint(peakBytes: 340, heldBytes: 330)),
                                   (.perImage, PhaseFootprint(peakBytes: 180, heldBytes: 150))] {
            admission.settings.ledger.record(.init(model: admission.modelKey, profile: "512x512", phase: phase), footprint)
        }
        let next = try await renders.render(shape: 14, encode: false)
        #expect(next.mode == .perPhase)
    }
}

/// The masked Krea 2 export's traced sequence (2026-11-09, realism, 1024², 8 steps, a 48 GiB budget,
/// `Krea2TextMaskQualificationTests` timing): prompts of 27, 27, 8 and 258 rows, each encoded with the generation. The
/// denoisers run at one shape (the 512-row bucket), so admission sees one shape throughout; the text encoder runs each
/// prompt at its own length (with the reshape hint). Before the fix the repeat measured 6 GB more held with nothing
/// loaded (Metal 47.1 GB, resident 52 GB, over the 51.5 GB budget) and every later render reloaded every graph.
@Suite("Lingering working memory admission, one denoiser shape")
struct OneShapeAdmissionTests {
    typealias Memory = LingeringWorkingMemoryAdmissionTests.Memory
    typealias Held = LingeringWorkingMemoryAdmissionTests.Held

    /// At a tenth of a gigabyte: text encoder 78, denoisers 256 (one copy: the bucket's shape never changes), decoder 1;
    /// a step run's working memory 20, a decode's 115. A generation with every graph resident reaches
    /// 78 + 256 + 1 + 115 + 20 = 470 above the level before any graph, under a budget of 515 (48 GiB).
    static let weights: [String: Int64] = ["textEncoder": 78, "denoiser": 256, "vaeDecoder": 1]
    static let budget: Int64 = 515

    @Test func differentPromptsKeepTheDenoisersLoaded() async throws {
        let memory = Memory()
        let admission = try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(Self.budget), coordinator: nil, ledger: FootprintLedger(), probe: memory.probe),
            load: { role in
                Held(Self.weights[role] ?? 0, copy: role == "denoiser" ? 256 : 0,
                     work: LingeringWorkingMemoryAdmissionTests.working[role] ?? 0, memory: memory)
            },
            fileBytes: { Self.weights[$0] ?? 0 })
        let bucket = 512
        var residencies: [GraphResidency] = []
        var loads: [[String]] = []
        for rows in [27, 27, 8, 258, 27, 100] {
            let residency = try admission.begin(profile: "1024x1024", shape: bucket)
            let before = Set(residency.residentRoles)
            try await residency.beginPhase(.perPrompt)
            (try await residency.graph("textEncoder") as! Held).run(rows)          // its own length
            residency.endPhase(.perPrompt)
            for (phase, role) in [(Manifest.Residency.perStep, "denoiser"), (.perImage, "vaeDecoder")] {
                try await residency.beginPhase(phase)
                (try await residency.graph(role) as! Held).run(bucket)
                residency.endPhase(phase)
            }
            #expect(admission.finish())
            loads.append(Set(residency.residentRoles).subtracting(before).sorted())
            residencies.append(residency)
            #expect(residency.mode == .allResident, "\(rows) rows: \(residency.mode)")
            #expect(admission.residentBytes <= Self.budget, "\(rows) rows: resident \(admission.residentBytes)")
            #expect(memory.above <= Self.budget)
        }
        #expect(loads[0] == ["denoiser", "textEncoder", "vaeDecoder"])
        #expect(loads.dropFirst().allSatisfy { $0.isEmpty }, "no reloads after the first: \(loads)")
        #expect(residencies.allSatisfy { $0 === residencies[0] })
    }
}
