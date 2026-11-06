import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §5.2–§5.3. The fixtures are the Krea exporter's own output (`export_krea2.py --sample-manifest`), so a
/// change to what the exporter emits shows up here.
@Suite("Generative manifest blocks")
struct GenerativeManifestTests {
    static func krea(_ variant: String) throws -> Data { try Fixtures.data("krea2/\(variant).json") }

    static func mutated(variant: String = "vanilla", _ edit: (inout [String: Any]) -> Void) throws -> Data {
        var json = try JSONSerialization.jsonObject(with: krea(variant)) as! [String: Any]
        edit(&json)
        return try JSONSerialization.data(withJSONObject: json)
    }

    static func edit(_ json: inout [String: Any], _ block: String, _ change: (inout [String: Any]) -> Void) {
        var value = json[block] as! [String: Any]
        change(&value)
        json[block] = value
    }

    @Test(arguments: ["vanilla", "realism"])
    func theExportersManifestsDecode(variant: String) throws {
        let manifest = try Manifest.decode(Self.krea(variant))
        #expect(manifest.driver.description == "krea2@1")
        let schedule = try #require(manifest.schedule)
        #expect(schedule == Manifest.Schedule(
            steps: 8, maxSteps: 50, trainTimesteps: 1000,
            sigmas: .init(shift: .init(rule: .exponential, mu: 1.15))))
        let latent = try #require(manifest.latent)
        #expect(latent.channels == 64 && latent.pixelsPerToken == 16 && latent.maxImageTokens == 4096)
        #expect(latent.packing == .patch2x2ChannelMajor && latent.noise == .init())
        #expect(manifest.graphs.mapValues(\.residency) == [
            "textEncoder": .perPrompt, "denoiser0": .perStep, "denoiser1": .perStep, "denoiser2": .perStep,
            "denoiser3": .perStep, "vaeDecoder": .perImage,
        ])
        #expect((manifest.provenance?.mergedAdapter?.sha256.hasPrefix("a6712629") ?? false) == (variant == "realism"))
        // The canonical encoding round-trips.
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }

    @Test func absentBlocksStayAbsent() throws {
        let manifest = try Manifest.decode(Data(ManifestTests.example.utf8))
        #expect(manifest.schedule == nil && manifest.latent == nil)
        #expect(manifest.graphs["decoder"]?.residency == nil)
    }

    @Test func closedVocabulariesRejectUnknownValues() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("schedule.kind", { Self.edit(&$0, "schedule") { $0["kind"] = "epsilonEuler" } }),
            ("schedule.timestep", { Self.edit(&$0, "schedule") { $0["timestep"] = "sigma" } }),
            ("schedule.update", { Self.edit(&$0, "schedule") { $0["update"] = "x + dt * v, float32" } }),
            ("schedule.sigmas.base", { Self.edit(&$0, "schedule") {
                var sigmas = $0["sigmas"] as! [String: Any]
                sigmas["base"] = "linspace(1, 0, steps)"
                $0["sigmas"] = sigmas
            } }),
            ("latent.packing", { Self.edit(&$0, "latent") { $0["packing"] = "patch2x2" } }),
            ("latent.noise.kind", { Self.edit(&$0, "latent") { $0["noise"] = ["kind": "philox", "order": "packedToken", "dtype": "float32"] } }),
            ("graphs.textEncoder.residency", { json in
                var graphs = json["graphs"] as! [String: Any]
                var text = graphs["textEncoder"] as! [String: Any]
                text["residency"] = "perRequest"
                graphs["textEncoder"] = text
                json["graphs"] = graphs
            }),
        ]
        for (expected, edit) in cases {
            expectField(expected) { _ = try Manifest.decode(try Self.mutated(edit)) }
        }
    }

    @Test func structuralRulesNameTheirField() throws {
        func shift(_ value: [String: Any]) -> (inout [String: Any]) -> Void {
            { Self.edit(&$0, "schedule") { s in
                var sigmas = s["sigmas"] as! [String: Any]
                sigmas["shift"] = value
                s["sigmas"] = sigmas
            } }
        }
        let empirical: [String: Any] = ["lowSteps": 10, "lowSlope": 8.73809524e-05, "lowIntercept": 1.89833333,
                                        "highSteps": 200, "highSlope": 0.00016927, "highIntercept": 0.45666666,
                                        "tokenThreshold": 4300]
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("schedule.steps", { Self.edit(&$0, "schedule") { $0["steps"] = 0 } }),
            ("schedule.maxSteps", { Self.edit(&$0, "schedule") { $0["maxSteps"] = 4 } }),
            ("schedule.trainTimesteps", { Self.edit(&$0, "schedule") { $0["trainTimesteps"] = 0 } }),
            ("schedule.sigmas.terminal", { Self.edit(&$0, "schedule") { s in
                var sigmas = s["sigmas"] as! [String: Any]
                sigmas["terminal"] = 0.002
                s["sigmas"] = sigmas
            } }),
            ("schedule.sigmas.shift.mu", shift(["rule": "exponential"])),
            ("schedule.sigmas.shift.empirical", shift(["rule": "empirical"])),
            ("schedule.sigmas.shift.empirical", shift(["rule": "empirical", "empirical": empirical.merging(["highSteps": 10]) { $1 }])),
            ("schedule.sigmas.shift", shift(["rule": "none", "mu": 1.0])),
            ("schedule.guidance.scale", { Self.edit(&$0, "schedule") { $0["guidance"] = ["kind": "none", "scale": 0.0] } }),
            ("schedule.guidance.scale", { Self.edit(&$0, "schedule") { $0["guidance"] = ["kind": "cfg"] } }),
            ("latent.channels", { Self.edit(&$0, "latent") { $0["channels"] = 62 } }),
            ("latent.pixelsPerToken", { Self.edit(&$0, "latent") { $0["pixelsPerToken"] = 15 } }),
            ("latent.noise.dtype", { Self.edit(&$0, "latent") { $0["noise"] = ["kind": "splitmix64BoxMuller", "order": "packedToken", "dtype": "float16"] } }),
            ("latent.maxImageTokens", { Self.edit(&$0, "latent") { $0["maxImageTokens"] = 0 } }),
            ("latent.layout", { Self.edit(&$0, "latent") { $0["layout"] = "" } }),
        ]
        for (expected, edit) in cases {
            expectField(expected) { _ = try Manifest.decode(try Self.mutated(edit)) }
        }
        // The valid alternatives decode.
        let flux = try Manifest.decode(try Self.mutated(shift(["rule": "empirical", "empirical": empirical])))
        #expect(flux.schedule?.sigmas.shift.empirical?.tokenThreshold == 4300)
        let unshifted = try Manifest.decode(try Self.mutated(shift(["rule": "none"])))
        #expect(unshifted.schedule?.sigmas.shift.rule == Manifest.Schedule.Shift.Rule.none)
        let cfg = try Manifest.decode(try Self.mutated { Self.edit(&$0, "schedule") { $0["guidance"] = ["kind": "cfg", "scale": 4.5] } })
        #expect(cfg.schedule?.guidance == .init(kind: .cfg, scale: 4.5))
    }

    @Test func empiricalShiftFollowsTheReferenceRule() {
        // FLUX.2 [klein]'s coefficients, as 0.0.x's Flux2ContractTests checked them.
        let shift = Manifest.Schedule.EmpiricalShift(
            lowSteps: 10, lowSlope: 8.73809524e-05, lowIntercept: 1.89833333, highSteps: 200, highSlope: 0.00016927,
            highIntercept: 0.45666666, tokenThreshold: 4300)
        let high = 0.00016927 * 4096 + 0.45666666, low = 8.73809524e-05 * 4096 + 1.89833333
        let slope = (high - low) / 190
        #expect(shift.mu(imageTokens: 4096, steps: 4) == slope * 4 + (high - 200 * slope))
        #expect(shift.mu(imageTokens: 5120, steps: 4) == 0.00016927 * 5120 + 0.45666666)
    }

    @Test func tokenGridChecksTheRequest() throws {
        let latent = try #require(try Manifest.decode(Self.krea("vanilla")).latent)
        #expect(try latent.tokenGrid(width: 1024, height: 1024) == (64, 64))
        #expect(try latent.tokenGrid(width: 768, height: 1344) == (84, 48))
        for (w, h) in [(0, 16), (17, 16), (1040, 1024), (Int.max - 15, Int.max - 15)] {
            #expect(throws: AetherError.self) { try latent.tokenGrid(width: w, height: h) }
        }
    }
}
