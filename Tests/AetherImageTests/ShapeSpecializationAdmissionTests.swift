import AetherCore
import AetherHost
import AetherRuntime
import Foundation
import Synchronization
import Testing

/// Admission for graphs that keep one specialization per shape (graphs loaded without the reshape hint; Krea 2's
/// denoisers at each prompt row count), on the SD-Turbo fixture manifest (files 10 / 100 / 10). Each fake graph adds its
/// bytes to the memory level while it is loaded, and the denoiser adds a full copy for each further shape it
/// specializes, as Core AI does: measured 2026-11-08 on Krea 2 realism, a second row count kept about 27 GB.
///
/// Before: the ledger kept the largest footprint per phase whatever the number of shapes behind it, so once a pipeline
/// had run two shapes, every later plan at that profile counted both, and refused where one shape fits. And a second
/// shape was planned at one shape's footprint, so it ran past the budget.
@Suite("Shape specialization admission")
struct ShapeSpecializationAdmissionTests {
    final class Memory: Sendable {
        let level = Mutex<Int64>(1_000)
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in level.withLock { $0 } } }
        func add(_ bytes: Int64) { level.withLock { $0 += bytes } }
        var above: Int64 { level.withLock { $0 } - 1_000 }
    }

    /// A graph holding `bytes` while loaded; `specialize(_:)` adds `copy` bytes for each new shape after the first.
    final class Held: GraphBackend, @unchecked Sendable {
        let signature = GraphSignature(function: "f", inputs: [], states: [], outputs: [])
        let memory: Memory
        let copy: Int64
        private let state = Mutex<(bytes: Int64, shapes: Set<Int>)>((0, []))
        init(_ bytes: Int64, copy: Int64, memory: Memory) {
            self.memory = memory
            self.copy = copy
            state.withLock { $0.bytes = bytes }
            memory.add(bytes)
        }
        deinit { memory.add(-state.withLock { $0.bytes }) }
        func specialize(_ shape: Int) {
            let added: Int64 = state.withLock { s in
                guard s.shapes.insert(shape).inserted, s.shapes.count > 1 else { return 0 }
                s.bytes += copy
                return copy
            }
            memory.add(added)
        }
        func run(_ bindings: inout Bindings) async throws(AetherError) {}
    }

    static let files: [String: Int64] = ["textEncoder": 10, "denoiser": 100, "vaeDecoder": 10]

    static func admission(budget: Int64, ledger: FootprintLedger = .init(), memory: Memory) throws -> ResidentAdmission {
        try ResidentAdmission(
            manifest: try LatentDiffusionTests.manifest(), device: LatentDiffusionTests.mac,
            settings: .init(memory: .bytes(budget), coordinator: nil, ledger: ledger, probe: memory.probe),
            load: { role in Held(files[role] ?? 0, copy: role == "denoiser" ? 100 : 0, memory: memory) },
            fileBytes: { files[$0] ?? 0 })
    }

    /// One generation at `shape`: each phase loads its graph, which specializes for the shape. Returns its residency.
    @discardableResult
    static func generate(_ admission: ResidentAdmission, shape: Int, memory: Memory) async throws -> GraphResidency {
        let residency = try admission.begin(profile: "512x512", shape: shape)
        for (phase, role) in [(Manifest.Residency.perPrompt, "textEncoder"), (.perStep, "denoiser"), (.perImage, "vaeDecoder")] {
            try await residency.beginPhase(phase)
            (try await residency.graph(role) as! Held).specialize(shape)
            residency.endPhase(phase)
        }
        admission.finish()
        return residency
    }

    @Test func aShapeThatDoesntFitBesideTheOthersReloadsAlone() async throws {
        let memory = Memory()
        let admission = try Self.admission(budget: 150, memory: memory)
        let first = try await Self.generate(admission, shape: 20, memory: memory)
        #expect(first.mode == .allResident && memory.above == 120)
        // A second shape would hold 220 all resident: the graphs are released and reload for it alone.
        let second = try await Self.generate(admission, shape: 31, memory: memory)
        #expect(second !== first && second.mode == .allResident && second.specializations == 1)
        #expect(memory.above == 120)
        // Back to the first shape, and on to a third: always one shape, never past the budget, never refused.
        for shape in [20, 44, 20] {
            let residency = try await Self.generate(admission, shape: shape, memory: memory)
            #expect(residency.mode == .allResident && memory.above <= 150)
        }
        #expect(admission.residentBytes == 120)
    }

    @Test func shapesThatFitStayResidentAndAreMeasuredAsMany() async throws {
        let memory = Memory(), ledger = FootprintLedger()
        let admission = try Self.admission(budget: 1 << 40, ledger: ledger, memory: memory)
        let first = try await Self.generate(admission, shape: 20, memory: memory)
        #expect(try await Self.generate(admission, shape: 31, memory: memory) === first)
        #expect(first.specializations == 2 && memory.above == 220)
        // A shape already specialized: the same residency and count, nothing new held.
        #expect(try await Self.generate(admission, shape: 20, memory: memory) === first)
        #expect(first.specializations == 2 && memory.above == 220 && admission.residentBytes == 220)

        let step = FootprintLedger.Key(model: admission.modelKey, profile: "512x512", phase: .perStep)
        #expect(ledger.footprint(step)?.heldBytes == 100)
        #expect(ledger.footprint(.init(model: admission.modelKey, profile: "512x512", phase: .perStep, specializations: 2))?
            .heldBytes == 200)
    }

    /// The poisoning: a ledger that has measured two shapes still plans one shape at one shape's footprint.
    @Test func twoMeasuredShapesDontRefuseOneShapeLater() async throws {
        let memory = Memory(), ledger = FootprintLedger()
        let roomy = try Self.admission(budget: 1 << 40, ledger: ledger, memory: memory)
        try await Self.generate(roomy, shape: 20, memory: memory)
        try await Self.generate(roomy, shape: 31, memory: memory)
        roomy.release()
        #expect(memory.above == 0)

        let tight = try Self.admission(budget: 150, ledger: ledger, memory: memory)
        let residency = try await Self.generate(tight, shape: 44, memory: memory)
        #expect(residency.mode == .allResident && memory.above == 120)
        // And a second shape there doesn't fit (measured 220), so it reloads rather than refuse.
        let next = try await Self.generate(tight, shape: 20, memory: memory)
        #expect(next !== residency && memory.above == 120)
    }


    @Test func unmeasuredCountsAreEstimatedFromFewer() {
        let ledger = FootprintLedger()
        let one = FootprintLedger.Key(model: "m", profile: "p", phase: .perStep)
        func key(_ n: Int) -> FootprintLedger.Key { .init(model: "m", profile: "p", phase: .perStep, specializations: n) }
        #expect(ledger.estimate(key(2)) == nil)
        ledger.record(one, PhaseFootprint(peakBytes: 130, heldBytes: 100))
        // One measured: each further shape is a full copy of the held bytes.
        let two = ledger.estimate(key(2))
        #expect(two?.measured == false && two?.footprint == PhaseFootprint(peakBytes: 230, heldBytes: 200))
        #expect(ledger.estimate(one)?.measured == true && ledger.estimate(one)?.footprint.heldBytes == 100)
        // Three measured: each further shape adds half the growth from one to three.
        ledger.record(key(3), PhaseFootprint(peakBytes: 190, heldBytes: 160))
        #expect(ledger.estimate(key(5))?.footprint == PhaseFootprint(peakBytes: 250, heldBytes: 220))
        #expect(key(2).description.hasSuffix("perStep×2") && one.description.hasSuffix("perStep"))
    }
}
