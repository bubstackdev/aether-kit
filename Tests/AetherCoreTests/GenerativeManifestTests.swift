import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §5.2–§5.3. The fixtures are the Krea and FLUX.2 exporters' own output (`--sample-manifest`), so a
/// change to what an exporter emits shows up here.
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

    /// The FLUX.2 [klein] exporter's own manifest (`export_flux2.py --sample-manifest`): empirical shift, 128-channel
    /// latents, and the optional reference encoder resident per prompt.
    @Test func theFlux2ExportersManifestDecodes() throws {
        let manifest = try Manifest.decode(try Fixtures.data("flux2/flux2-klein-4b.json"))
        #expect(manifest.driver.description == "flux2-klein@1")
        let schedule = try #require(manifest.schedule)
        #expect(schedule == Manifest.Schedule(
            steps: 4, maxSteps: 50, trainTimesteps: 1000,
            sigmas: .init(shift: .init(rule: .empirical, empirical: .init(
                lowSteps: 10, lowSlope: 8.73809524e-05, lowIntercept: 1.89833333, highSteps: 200,
                highSlope: 0.00016927, highIntercept: 0.45666666, tokenThreshold: 4300)))))
        let latent = try #require(manifest.latent)
        #expect(latent.channels == 128 && latent.pixelsPerToken == 16 && latent.maxImageTokens == 4096)
        #expect(latent.packing == .patch2x2ChannelMajor && latent.noise == .init())
        #expect(manifest.graphs.mapValues(\.residency) == [
            "textEncoder": .perPrompt, "denoiser0": .perStep, "denoiser1": .perStep, "denoiser2": .perStep,
            "vaeDecoder": .perImage, "vaeEncoder": .perPrompt,
        ])
        #expect(manifest.graphs["vaeDecoder"]?.expectFrequentReshapes == false)
        #expect(manifest.tokenizer?.chatTemplate?.description == "main:tokenizer/chat_template.jinja")
        #expect(manifest.license == "Apache-2.0" && manifest.provenance?.mergedAdapter == nil)
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }

    @Test func absentBlocksStayAbsent() throws {
        let manifest = try Manifest.decode(Data(ManifestTests.example.utf8))
        #expect(manifest.schedule == nil && manifest.latent == nil)
        #expect(manifest.graphs["decoder"]?.residency == nil)
    }

    @Test func closedVocabulariesRejectUnknownValues() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("schedule.kind", { Self.edit(&$0, "schedule") { $0["kind"] = "ddim" } }),
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
        #expect(flux.schedule?.sigmas?.shift.empirical?.tokenThreshold == 4300)
        let unshifted = try Manifest.decode(try Self.mutated(shift(["rule": "none"])))
        #expect(unshifted.schedule?.sigmas?.shift.rule == Manifest.Schedule.Shift.Rule.none)
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

    // MARK: - epsilonEuler and spatial latents (SD-Turbo, `export_sdturbo.py --sample-manifest`)

    static func sdTurbo(_ edit: ((inout [String: Any]) -> Void)? = nil) throws -> Data {
        let data = try Fixtures.data("sd-turbo/aether.json")
        guard let edit else { return data }
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        edit(&json)
        return try JSONSerialization.data(withJSONObject: json)
    }

    @Test func theSDTurboExportDecodes() throws {
        let manifest = try Manifest.decode(Self.sdTurbo())
        #expect(manifest.driver == .latentDiffusion)
        try DriverContract.latentDiffusion.validate(manifest)
        let schedule = try #require(manifest.schedule)
        #expect(schedule.kind == .epsilonEuler && schedule.steps == 1 && schedule.stepLimit == 4)
        #expect(schedule.modelInput == .scaled && schedule.timestep == .table && schedule.update == .eulerEpsilon)
        // The pinned scheduler's tables, as the 0.0.x spike recorded them (docs/evidence/sd-turbo-v006-2026-11-02.json).
        let recorded: [Int: ([Float], [Float])] = [
            1: ([999], [14.614646911621094, 0]),
            2: ([999, 499], [14.614646911621094, 1.6128870248794556, 0]),
            3: ([999, 666, 332], [14.614646911621094, 2.9183084964752197, 0.9292148947715759, 0]),
            4: ([999, 749, 499, 249], [14.614646911621094, 4.081730842590332, 1.6128870248794556, 0.6932054162025452, 0]),
        ]
        for (steps, (timesteps, sigmas)) in recorded {
            #expect(try schedule.table(steps: steps) == .init(timesteps: timesteps, sigmas: sigmas, initNoiseSigma: 14.614646911621094))
        }
        #expect(throws: AetherError.self) { try schedule.table(steps: 5) }
        let latent = try #require(manifest.latent)
        #expect(latent.packing == .spatial && latent.noise.order == .nchw && latent.channels == 4)
        let size = try #require(latent.defaultSize)
        #expect(size.width == 512 && size.height == 512)
        #expect(try latent.tokenGrid(width: 512, height: 512) == (64, 64))
        #expect(throws: AetherError.self) { try latent.tokenGrid(width: 256, height: 256) }
        #expect(manifest.graphs.mapValues(\.residency) == ["textEncoder": .perPrompt, "denoiser": .perStep, "vaeDecoder": .perImage])
        #expect(try Manifest.decode(manifest.encoded()) == manifest)
    }

    @Test func epsilonEulerRulesNameTheirField() throws {
        func schedule(_ change: @escaping (inout [String: Any]) -> Void) -> (inout [String: Any]) -> Void {
            { Self.edit(&$0, "schedule", change) }
        }
        func table(_ steps: String, _ change: @escaping (inout [String: Any]) -> Void) -> (inout [String: Any]) -> Void {
            schedule { s in
                var tables = s["tables"] as! [String: Any]
                var t = tables[steps] as! [String: Any]
                change(&t)
                tables[steps] = t
                s["tables"] = tables
            }
        }
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("schedule.tables", schedule { $0["tables"] = nil }),
            ("schedule.tables", schedule { $0["maxSteps"] = 5 }),
            ("schedule.sigmas", schedule { $0["sigmas"] = ["base": "linspace(1, 1/steps, steps)", "shift": ["rule": "none"], "terminal": 0] }),
            ("schedule.modelInput", schedule { $0["modelInput"] = nil }),
            ("schedule.timestep", schedule { $0["timestep"] = "sigma * trainTimesteps / trainTimesteps" }),
            ("schedule.update", schedule { $0["update"] = "x + (sigma[i+1] - sigma[i]) * velocity" }),
            ("schedule.tables.2.timesteps", table("2") { $0["timesteps"] = [999] }),
            ("schedule.tables.2.timesteps", table("2") { $0["timesteps"] = [499, 999] }),
            ("schedule.tables.3.sigmas", table("3") { $0["sigmas"] = [14.6, 2.9, 0.9, 0.1] }),
            ("schedule.tables.1.initNoiseSigma", table("1") { $0["initNoiseSigma"] = 0 }),
            ("latent.noise.order", { Self.edit(&$0, "latent") { $0["noise"] = ["kind": "splitmix64BoxMuller", "order": "packedToken", "dtype": "float32"] } }),
            ("latent.sizes.0", { Self.edit(&$0, "latent") { $0["sizes"] = [[500, 512]] } }),
            ("latent.sizes", { Self.edit(&$0, "latent") { $0["sizes"] = [] as [Any] } }),
        ]
        for (expected, edit) in cases {
            expectField(expected) { _ = try Manifest.decode(try Self.sdTurbo(edit)) }
        }
        // A flow-matching schedule must not carry epsilonEuler's fields.
        expectField("schedule.tables") {
            _ = try Manifest.decode(try Self.mutated { Self.edit(&$0, "schedule") { $0["tables"] = [:] as [String: Any] } })
        }
    }
}
