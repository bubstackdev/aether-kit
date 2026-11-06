import AetherCore
import CryptoKit
import Foundation
import ImageIO
import Synchronization
import Testing
@testable import AetherRuntime

// The generative foundation (G0). Bit patterns and digests are 0.0.x's recorded fixtures: its
// AetherKitTests/SeededGaussianNoiseTests, AetherKreaTests/KreaReferenceTests, AetherH3Tests/H3MediaTests and
// AetherFlux2Tests/Flux2ContractTests.

@Suite("Seeded Gaussian noise")
struct SeededNoiseTests {
    /// SHA-256 of the first 98,304 values' float32 bits, recorded by 0.0.x from its standalone Kokoro and H3 generators
    /// before they were folded into one.
    static let recordedStreams: [UInt64: String] = [
        0: "d3c374d012de4a5bd4ff3a84c72f32836e9b3a4486d611fc3ea24fe24eafb622",
        1: "f708c4cda415178d4f593f9359d4223c150343a3bd89f9f3369a91a06e10d0f9",
        3: "8fcfdf4713b1e4a5f35681e547331b2b42b079b2f4de5e5929cd087d756a6004",
        5: "e5b1e14c75de41e28d18c03b1c78750b7c94a984b6870592004c65c98a67379e",
        100: "845e0f4548a5d846e6fe3d36249d4475dc6aae854538270ee94b7ad7f7a2e0d9",
        1043: "4a79d10bb38b2936f2fdb3efc4cf5f979d8f1931c4bfb209f032138e6ef19219",
        0x5eed: "f0ce5715969c2a5048a598dc04e6734c5dfe9a47dc10f6e3328b4ac93145bfe4",
        0xdeadbeef: "08ca528ded9a348fa76ba9dc8a296f2eae759c5898de82c6168221d0173c738a",
        .max: "8374d6935fef05f0d0ccdc4505c379acbb29b91f61ca614e8b872852c11ae5b0",
    ]

    @Test func reproducesThe0_0_xStreamsAndIsPrefixStable() throws {
        for (seed, expected) in Self.recordedStreams {
            let bits = try SeededGaussianNoise.generate(seed: seed, count: 98_304).map(\.bitPattern)
            let digest = bits.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
            #expect(digest == expected, "seed \(seed)")
            for count in [0, 1, 2, 3, 7, 128, 5_400, 16_201] {
                #expect(try SeededGaussianNoise.generate(seed: seed, count: count).map(\.bitPattern) == Array(bits.prefix(count)),
                        "seed \(seed) count \(count)")
            }
        }
    }

    @Test func matchesTheKreaAndH3OracleBits() throws {
        // Krea: captured from independent Swift lab oracles.
        let krea: [(UInt64, [UInt32])] = [
            (0, [3202863069, 1045741727, 1076470663, 3204126899, 3212645673, 1072668933, 1048658635, 3219995906]),
            (42, [1054103078, 1059526174, 3211023016, 1068094895, 1071473486, 3220247501, 1057729992, 3218346802]),
            (71344, [1035851364, 1074166398, 1060776933, 3184718423, 1070719263, 3206437512, 1025533962, 1052832527]),
        ]
        for (seed, expected) in krea {
            #expect(try SeededGaussianNoise.generate(seed: seed, count: 8).map(\.bitPattern) == expected)
        }
        // H3: an independent Python implementation of the same stream.
        #expect(try SeededGaussianNoise.generate(seed: 1043, count: 7).map(\.bitPattern) ==
                [0xbff69b06, 0xbf0b5ad4, 0xbed666d3, 0xbfe6e40c, 0xbfb30e8b, 0xbe3d4e3d, 0x3f2010f4])
        #expect(try SeededGaussianNoise.generate(seed: 0, count: 3).map(\.bitPattern) == [0xbee7cfdd, 0x3e54c09f, 0x4029a387])
    }

    @Test func fillWritesTheSameStreamAndBadCountsAreRefused() throws {
        var buffer = [Float](repeating: .nan, count: 9)
        buffer.withUnsafeMutableBufferPointer { SeededGaussianNoise.fill($0, seed: 1043) }
        #expect(buffer.map(\.bitPattern) == (try SeededGaussianNoise.generate(seed: 1043, count: 9)).map(\.bitPattern))
        #expect(try SeededGaussianNoise.generate(seed: 9, count: 0).isEmpty)
        #expect(throws: AetherError.self) { try SeededGaussianNoise.generate(seed: 1, count: -1) }
    }
}

@Suite("Flow-matching schedule")
struct FlowMatchScheduleTests {
    static let krea = Manifest.Schedule(steps: 8, maxSteps: 50, sigmas: .init(shift: .init(rule: .exponential, mu: 1.15)))
    static let klein = Manifest.Schedule(steps: 4, maxSteps: 50, sigmas: .init(shift: .init(rule: .empirical, empirical: .init(
        lowSteps: 10, lowSlope: 8.73809524e-05, lowIntercept: 1.89833333, highSteps: 200, highSlope: 0.00016927,
        highIntercept: 0.45666666, tokenThreshold: 4300))))

    @Test func kreaMatchesThe0_0_xOracleBits() throws {
        let schedule = try FlowMatchSchedule(Self.krea)
        #expect(schedule.timesteps.map(\.bitPattern) == [
            1065353216, 1064627160, 1063751508, 1062674713, 1061318479, 1059557808, 1057180097, 1050619501,
        ])
        let deltas = zip(schedule.sigmas.dropFirst(), schedule.sigmas).map { ($0 - $1).bitPattern }
        #expect(deltas == [
            3174122112, 3176515648, 3179508184, 3181743696, 3184979192, 3188793276, 3192834602, 3198103149,
        ])
        #expect(try FlowMatchSchedule(Self.krea, steps: 1).sigmas == [1, 0])
        #expect(throws: AetherError.self) { try FlowMatchSchedule(Self.krea, steps: 0) }
        #expect(throws: AetherError.self) { try FlowMatchSchedule(Self.krea, steps: 51) }
        // Review finding 10: the direct initializer refuses bad values instead of trapping.
        #expect(throws: AetherError.self) { try FlowMatchSchedule(steps: 0, mu: nil) }
        #expect(throws: AetherError.self) { try FlowMatchSchedule(steps: 4, mu: .nan) }
        #expect(throws: AetherError.self) { try FlowMatchSchedule(steps: 4, mu: 1, trainTimesteps: 0) }
    }

    @Test func fluxEmpiricalShiftMatchesTheReferencePipeline() throws {
        // Observed in the unmodified FLUX.2 [klein] pipeline: 2 steps at 256 generated tokens.
        #expect(try FlowMatchSchedule(Self.klein, steps: 2, imageTokens: 256).sigmas == [1, 0.8787367343902588, 0])
        let four = try FlowMatchSchedule(Self.klein, imageTokens: 1024)
        // The pinned Diffusers v0.42.1 pipeline at 512x512, 4 steps (Forge/flux2/reference_flux2.py): sigmas and the
        // transformer's timestep inputs, bit for bit.
        #expect(four.sigmas.map(\.bitPattern) == [1065353216, 1064650005, 1063406755, 1060613595, 0])
        #expect(four.timesteps.map(\.bitPattern) == [1065353216, 1064650006, 1063406755, 1060613595])
        #expect(four.sigmas.count == 5 && four.sigmas.first == 1 && four.sigmas.last == 0)
        #expect(four.timesteps == Array(four.sigmas.dropLast()).map { $0 * 1000 / 1000 })
        #expect(zip(four.sigmas, four.sigmas.dropFirst()).allSatisfy { $0 > $1 })
        #expect(throws: AetherError.self) { try FlowMatchSchedule(Self.klein) }  // needs the token count
    }

    @Test func unshiftedIsTheBaseGrid() throws {
        let none = Manifest.Schedule(steps: 4, sigmas: .init(shift: .init(rule: .none)))
        #expect(try FlowMatchSchedule(none).sigmas == [1, Float(1.0 - 0.25), Float(0.5), 0.25, 0])
        #expect(try FlowMatchSchedule(none).mu == nil)
    }

    @Test func eulerRoundsTheProductAndLeavesLatentsUnchangedOnFailure() throws {
        let a = Float(bitPattern: 0x3f80_0001), b = Float(bitPattern: 0x3f7f_fffe)
        #expect(roundedProduct(a, b) - 1 == 0)
        #expect(Float(-1).addingProduct(a, b) != 0)  // what a fused multiply-add would give instead
        let one = try FlowMatchSchedule(Self.krea, steps: 1)
        var x: [Float] = [2]
        try one.advance(step: 0, latents: &x, velocity: [3])
        #expect(x == [-1])
        let four = try FlowMatchSchedule(Self.klein, imageTokens: 1024)
        var latents: [Float] = [0.1, -2, 3.25]
        let v: [Float] = [1.5, 0.75, -4]
        let delta = four.sigmas[1] - four.sigmas[0]
        let expected = zip(latents, v).map { x, v -> Float in let product = delta * v; return x + product }
        try four.advance(step: 0, latents: &latents, velocity: v)
        #expect(latents == expected)
        let before = latents
        for (step, velocity) in [(4, v), (0, [1]), (0, [1, .nan, 1])] {
            #expect(throws: AetherError.self) { try four.advance(step: step, latents: &latents, velocity: velocity) }
            #expect(latents == before)
        }
        var huge: [Float] = [1, .greatestFiniteMagnitude]
        #expect(throws: AetherError.self) {
            try one.advance(step: 0, latents: &huge, velocity: [1, -.greatestFiniteMagnitude])  // max + max overflows
        }
        #expect(huge == [1, .greatestFiniteMagnitude])
    }
}

@Suite("Generated media")
struct GeneratedMediaTests {
    @Test func imageQuantizationAndPNGMatchTheReferenceConvention() throws {
        // [3, 1, 2]: R = (-1, 1), G = (0, 2), B = (-2, 0).
        let image = try GeneratedImage(width: 2, height: 1, planarRGB: [-1, 1, 0, 2, -2, 0])
        #expect(image.rgb8() == [0, 128, 0, 255, 255, 128])
        let png = try image.pngData()
        let source = try #require(CGImageSourceCreateWithData(png as CFData, nil))
        let decoded = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        #expect(decoded.width == 2 && decoded.height == 1)
        let bytes = try #require(decoded.dataProvider?.data) as Data
        let stride = decoded.bitsPerPixel / 8  // ImageIO may expand RGB to RGBA
        #expect(Array(bytes[0..<3]) == [0, 128, 0] && Array(bytes[stride..<(stride + 3)]) == [255, 255, 128])
        #expect(throws: AetherError.self) { try GeneratedImage(width: 2, height: 1, planarRGB: [0, 0]) }
        #expect(throws: AetherError.self) { try GeneratedImage(width: 1, height: 1, planarRGB: [0, .nan, 0]) }
        #expect(throws: AetherError.self) { try GeneratedImage(width: 0, height: 1, planarRGB: []) }
    }

    @Test func wavHeadersAndSamples() throws {
        let mono = try GeneratedAudio(mono: [0, 1, -1, 0.5, 2], sampleRate: 24_000)
        let pcm = try mono.wavData()
        #expect(pcm.count == 44 + 10)
        #expect(String(decoding: pcm[0..<4], as: UTF8.self) == "RIFF" && String(decoding: pcm[36..<40], as: UTF8.self) == "data")
        func i16(_ data: Data, _ at: Int) -> Int16 { Int16(littleEndian: data[at..<(at + 2)].withUnsafeBytes { $0.loadUnaligned(as: Int16.self) }) }
        #expect((0..<5).map { i16(pcm, 44 + 2 * $0) } == [0, 32767, -32767, 16384, 32767])  // clipped, ties to even
        let stereo = try GeneratedAudio(channels: [[0.25, 3], [-0.5, 1]], sampleRate: 48_000)
        let float = try stereo.wavData(encoding: .float32)
        #expect(float.count == 12 + 8 + 18 + 12 + 8 + 16)
        let samples = (0..<4).map { i in float[58 + 4 * i..<62 + 4 * i].withUnsafeBytes { Float(bitPattern: UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))) } }
        #expect(samples == [0.25, -0.5, 3, 1])  // interleaved, unclipped
        let surround = try GeneratedAudio(channels: Array(repeating: [0], count: 3), sampleRate: 8_000)
        #expect(try surround.wavData().count == 12 + 8 + 40 + 12 + 8 + 6)
        #expect(try GeneratedAudio.concatenating([mono, mono]).sampleCount == 10)
        #expect(throws: AetherError.self) { try GeneratedAudio.concatenating([mono, stereo]) }
        #expect(throws: AetherError.self) { try GeneratedAudio(channels: [[0], [0, 1]], sampleRate: 8_000) }
        #expect(throws: AetherError.self) { try GeneratedAudio(mono: [.infinity], sampleRate: 8_000) }
        #expect(throws: AetherError.self) { try GeneratedAudio(mono: [0], sampleRate: 0) }
    }
}

@Suite("Residency and measured footprints")
struct ResidencyTests {
    static func krea() throws -> Manifest {
        try Manifest.decode(Data(contentsOf: Fixtures.zoo.deletingLastPathComponent().appending(path: "krea2/vanilla.json")))
    }

    static let phases: [Manifest.Residency: PhaseFootprint] = [
        .perPrompt: PhaseFootprint(peakBytes: 10, heldBytes: 8),
        .perStep: PhaseFootprint(peakBytes: 9, heldBytes: 6),
        .perImage: PhaseFootprint(peakBytes: 5, heldBytes: 2),
    ]

    @Test func peaksCountHeldMemoryAndLateReleases() {
        // All resident: 8 + 6 + 2 held, plus the largest transient (the encoder's 2 or the steps' 3).
        #expect(ResidencyPlan.peak(.allResident, Self.phases) == 16 + 3)
        // Per phase, each peak on top of what the phase before it still holds (cyclically):
        // prompt 2 (image) + 10, steps 8 + 9, image 6 + 5.
        #expect(ResidencyPlan.peak(.perPhase, Self.phases) == 17)
        #expect(ResidencyPlan.peak(.perPhase, [.perStep: PhaseFootprint(peakBytes: 4, heldBytes: 4)]) == 4)
        #expect(PhaseFootprint(peakBytes: 3, heldBytes: 5).heldBytes == 3)  // held is at most the peak
    }

    @Test func planPrefersResidentThenPerPhaseThenRefuses() throws {
        let inputs = ResidencyPlan.Inputs(phases: Self.phases, isMeasured: true)
        #expect(try ResidencyPlan.choose(inputs, budget: 19) == ResidencyPlan(mode: .allResident, peakBytes: 19, isMeasured: true))
        #expect(try ResidencyPlan.choose(inputs, budget: 18) == ResidencyPlan(mode: .perPhase, peakBytes: 17, isMeasured: true))
        do {
            _ = try ResidencyPlan.choose(inputs, budget: 16)
            Issue.record("expected a capacity refusal")
        } catch .capacity(_, let needed, let available) {
            #expect(needed == 17 && available == 16)
        }
        // A graph that never leaves rules out releasing per phase.
        let pinned = ResidencyPlan.Inputs(phases: Self.phases, isMeasured: true, allowsPerPhase: false)
        #expect(throws: AetherError.self) { try ResidencyPlan.choose(pinned, budget: 18) }
    }

    @Test func inputsUseMeasurementsKeyedByProfileAndFilesUntilMeasured() throws {
        let manifest = try Self.krea()
        let ledger = FootprintLedger()
        let key = { (profile: String, phase: Manifest.Residency) in
            FootprintLedger.Key(model: "krea-2-turbo", profile: profile, phase: phase)
        }
        ledger.record(key("1024x1024", .perStep), PhaseFootprint(peakBytes: 500, heldBytes: 300))
        ledger.record(key("1024x1024", .perStep), PhaseFootprint(peakBytes: 400, heldBytes: 350))  // keeps each max
        #expect(ledger.footprint(key("1024x1024", .perStep)) == PhaseFootprint(peakBytes: 500, heldBytes: 350))
        let files = { (_: String) in Int64(100) }
        let partial = ResidencyPlan.inputs(manifest: manifest, ledger: ledger, profile: "1024x1024", fileBytes: files)
        #expect(!partial.isMeasured && partial.allowsPerPhase)
        #expect(partial.phases == [
            .perPrompt: PhaseFootprint(peakBytes: 100, heldBytes: 100),  // one encoder file
            .perStep: PhaseFootprint(peakBytes: 500, heldBytes: 350),
            .perImage: PhaseFootprint(peakBytes: 100, heldBytes: 100),
        ])
        ledger.record(key("1024x1024", .perPrompt), PhaseFootprint(peakBytes: 1, heldBytes: 1))
        ledger.record(key("1024x1024", .perImage), PhaseFootprint(peakBytes: 1, heldBytes: 1))
        #expect(ResidencyPlan.inputs(manifest: manifest, ledger: ledger, profile: "1024x1024", fileBytes: files).isMeasured)
        // Another profile has measured nothing: four denoiser files stand in for the steps.
        let other = ResidencyPlan.inputs(manifest: manifest, ledger: ledger, profile: "768x1344", fileBytes: files)
        #expect(!other.isMeasured && other.phases[.perStep] == PhaseFootprint(peakBytes: 400, heldBytes: 400))
    }

    /// A graph that is never run.
    final class Inert: GraphBackend {
        let signature = GraphSignature(function: "segment", inputs: [], states: [], outputs: [])
        func run(_ bindings: inout Bindings) async throws(AetherError) {}
    }

    /// Allocations the test moves by hand.
    final class FakeGPU: Sendable {
        let allocated = Mutex<Int64>(1_000)
        var probe: GPUMemoryProbe { GPUMemoryProbe { [self] in allocated.withLock { $0 } } }
        func add(_ bytes: Int64) { allocated.withLock { $0 += bytes } }
    }

    @Test(arguments: [ResidencyPlan.Mode.allResident, .perPhase])
    func residencyLoadsOnDemandMeasuresPhasesAndReleases(mode: ResidencyPlan.Mode) async throws {
        let gpu = FakeGPU()
        let ledger = FootprintLedger()
        var loads: [String] = []
        let residency = GraphResidency(
            manifest: try Self.krea(), mode: mode, ledger: ledger, profile: "1024x1024", probe: gpu.probe,
            samplingInterval: nil
        ) { role in
            loads.append(role)
            gpu.add(1)  // a load allocates almost nothing
            return Inert()
        }
        let key = { (phase: Manifest.Residency) in FootprintLedger.Key(model: "krea-2-turbo", profile: "1024x1024", phase: phase) }

        residency.beginPhase(.perPrompt)
        _ = try await residency.graph("textEncoder")
        _ = try await residency.graph("textEncoder")  // resident: not loaded again
        gpu.add(900)  // the first run materializes the weights and scratch
        residency.sample()
        gpu.add(-300)  // part of it is transient
        residency.endPhase(.perPrompt)
        #expect(ledger.footprint(key(.perPrompt)) == PhaseFootprint(peakBytes: 901, heldBytes: 601))

        residency.beginPhase(.perStep)
        for role in ["denoiser0", "denoiser1", "denoiser2", "denoiser3"] { _ = try await residency.graph(role) }
        residency.endPhase(.perStep)
        // Per phase, the prompt's graphs were released but their memory (the fake never frees it) still counts: the
        // steps phase measures from where the prompt began, the overestimate admission prefers.
        #expect(ledger.footprint(key(.perStep)) == (mode == .allResident
            ? PhaseFootprint(peakBytes: 4, heldBytes: 4) : PhaseFootprint(peakBytes: 605, heldBytes: 605)))

        switch mode {
        case .allResident:
            #expect(residency.residentRoles == ["denoiser0", "denoiser1", "denoiser2", "denoiser3", "textEncoder"])
        case .perPhase:
            #expect(residency.residentRoles.isEmpty)
            _ = try await residency.graph("textEncoder")  // the next generation loads it again
            #expect(loads.filter { $0 == "textEncoder" }.count == 2)
        }
        await #expect(throws: AetherError.self) { _ = try await residency.graph("refiner") }
        residency.beginPhase(.perImage)
        residency.releaseAll()  // an open phase is discarded unrecorded
        #expect(residency.residentRoles.isEmpty && ledger.footprint(key(.perImage)) == nil)
    }

    /// Review finding 4: a released phase's memory returns seconds late. Measured from a baseline that still held it,
    /// the next phase came out low (here its held bytes negative) once that memory returned mid-phase.
    @Test func aPhaseAfterAReleaseMeasuresFromBeforeTheReleasedGraphs() async throws {
        let gpu = FakeGPU()
        let ledger = FootprintLedger()
        let residency = GraphResidency(manifest: try Self.krea(), mode: .perPhase, ledger: ledger, profile: "p",
                                       probe: gpu.probe, samplingInterval: nil) { _ in Inert() }
        residency.beginPhase(.perPrompt)       // 1000
        _ = try await residency.graph("textEncoder")
        gpu.add(600)
        residency.endPhase(.perPrompt)         // released; its 600 not yet returned
        residency.beginPhase(.perStep)         // the level is still 1600
        _ = try await residency.graph("denoiser0")
        gpu.add(200)
        residency.sample()                     // 1800
        gpu.add(-600)                          // the prompt's memory returns
        residency.endPhase(.perStep)           // 1200
        let steps = ledger.footprint(.init(model: "krea-2-turbo", profile: "p", phase: .perStep))
        #expect(steps == PhaseFootprint(peakBytes: 800, heldBytes: 200))
        #expect(residency.floor == 1000)
    }

    /// Second review, overlap with finding 4: resident graphs that grow on a phase's next occurrence (100, then 80
    /// more) were recorded as the largest single growth (100), not what they hold (180).
    @Test func residentGrowthAccumulatesAcrossOccurrences() async throws {
        let gpu = FakeGPU()
        let ledger = FootprintLedger()
        let manifest = try Self.krea()
        let residency = GraphResidency(manifest: manifest, mode: .allResident, ledger: ledger, profile: "p",
                                       probe: gpu.probe, samplingInterval: nil) { _ in Inert() }
        residency.beginPhase(.perPrompt)
        _ = try await residency.graph("textEncoder")
        gpu.add(100)
        residency.endPhase(.perPrompt)
        residency.beginPhase(.perPrompt)
        gpu.add(80)
        residency.endPhase(.perPrompt)
        #expect(ledger.footprint(.init(model: manifest.id, profile: "p", phase: .perPrompt))
            == PhaseFootprint(peakBytes: 180, heldBytes: 180))
    }

    @Test func theSamplerCatchesAPeakBetweenSamples() async throws {
        let gpu = FakeGPU()
        let ledger = FootprintLedger()
        let residency = GraphResidency(manifest: try Self.krea(), mode: .perPhase, ledger: ledger, probe: gpu.probe,
                                       samplingInterval: .milliseconds(1)) { _ in Inert() }
        residency.beginPhase(.perImage)
        gpu.add(700)
        try await Task.sleep(for: .milliseconds(100))
        gpu.add(-700)
        residency.endPhase(.perImage)
        #expect(ledger.footprint(.init(model: "krea-2-turbo", profile: "", phase: .perImage)) == PhaseFootprint(peakBytes: 700, heldBytes: 0))
    }
}

/// `epsilonEuler` (MANIFEST.md §5.2): 0.0.x's `EpsilonSchedule` cases, and the arithmetic against Diffusers' order.
@Suite("Epsilon Euler schedule")
struct EpsilonEulerScheduleTests {
    @Test func inputScalingAndUpdateMatch0_0_x() throws {
        let s = try EpsilonEulerSchedule(timesteps: [999, 499], sigmas: [2, 1, 0], initNoiseSigma: 2)
        #expect(try s.modelInput(step: 0, latents: [6]) == [6 / Float(5).squareRoot()])
        var x: [Float] = [6]
        try s.advance(step: 0, latents: &x, epsilon: [2])
        #expect(x == [4])
        x = [6]
        try s.advance(step: 1, latents: &x, epsilon: [2])
        #expect(x == [4])
        #expect(throws: AetherError.self) { try s.advance(step: 2, latents: &x, epsilon: [2]) }
        var noise: [Float] = [1, -0.5]
        noise.withUnsafeMutableBufferPointer { s.scaleInitialNoise($0) }
        #expect(noise == [2, -1])
    }

    @Test func invalidTablesAreRefused() {
        for (t, sigmas, initial): ([Float], [Float], Float) in [
            ([1], [0, 0], 1), ([1], [1, 0], .nan), ([], [0], 1), ([1, 2], [2, 1, 0], 1), ([1], [1, 0.5], 1),
        ] {
            #expect(throws: AetherError.self) { try EpsilonEulerSchedule(timesteps: t, sigmas: sigmas, initNoiseSigma: initial) }
        }
    }

    /// The update in Diffusers' float32 order, element by element, against a non-trivial table.
    @Test func updateFollowsDiffusersOrder() throws {
        let sigmas: [Float] = [14.614646911621094, 4.081730842590332, 1.6128870248794556, 0.6932054162025452, 0]
        let s = try EpsilonEulerSchedule(timesteps: [999, 749, 499, 249], sigmas: sigmas, initNoiseSigma: sigmas[0])
        let x0: [Float] = [0.3, -1.7, 12.5, -0.001], e: [Float] = [0.11, -2.5, 0.75, 3.25]
        for step in 0..<4 {
            var x = x0
            try s.advance(step: step, latents: &x, epsilon: e)
            for i in x0.indices {
                let product: Float = sigmas[step] * e[i]
                let original: Float = x0[i] - product
                let derivative: Float = (x0[i] - original) / sigmas[step]
                let dt: Float = sigmas[step + 1] - sigmas[step]
                let move: Float = derivative * dt
                #expect(x[i] == x0[i] + move)
            }
        }
        var bad: [Float] = [1]
        #expect(throws: AetherError.self) { try s.advance(step: 0, latents: &bad, epsilon: [.infinity]) }
        #expect(bad == [1])
    }

    @Test func aFlowMatchScheduleIsNotEpsilon() throws {
        let flow = Manifest.Schedule(steps: 4, sigmas: .init(shift: .init(rule: .none)))
        #expect(throws: AetherError.self) { try EpsilonEulerSchedule(flow) }
        let epsilon = Manifest.Schedule(
            kind: .epsilonEuler, steps: 1, tables: ["1": .init(timesteps: [999], sigmas: [1, 0], initNoiseSigma: 1)],
            modelInput: .scaled, timestep: .table, update: .eulerEpsilon)
        #expect(throws: AetherError.self) { try FlowMatchSchedule(epsilon) }
        #expect(try EpsilonEulerSchedule(epsilon).timesteps == [999])
    }
}
