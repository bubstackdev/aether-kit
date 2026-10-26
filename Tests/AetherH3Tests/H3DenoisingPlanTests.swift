import AetherH3
import Foundation
import Testing

@Test func h3DenoisingPlanPinsEightStepsFiftyModulationGraphsAndStreamedWeights() throws {
    for fault in ["none","missing","hash","resident"] {
        let root = try phaseFixture(); defer { try? FileManager.default.removeItem(at: root) }
        let chain = try chainFixture(references: true, full: true); defer { try? FileManager.default.removeItem(at: chain) }
        func read(_ path: URL) throws -> [String: Any] { try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any] }
        let bundleURL = root.appendingPathComponent("block-bundle.json"), phaseURL = root.appendingPathComponent("block-phases.json"), cpuURL = root.appendingPathComponent("cpu.json")
        var bundle = try read(bundleURL), cpu = try read(cpuURL), manifest = try read(chain.appendingPathComponent("chain.json"))
        var models = (0..<50).map { ["path":"/old-location/modulation-\($0).aimodel","function":"main","sha256":String(repeating:"a",count:64)] }
        if fault == "missing" { models.removeLast() }
        if fault == "hash" { models[0]["sha256"] = "invalid" }
        bundle["modulations"] = models
        let data = try JSONSerialization.data(withJSONObject: bundle); try data.write(to: bundleURL)
        cpu["bundle_sha256"] = h3SHA256(data); try JSONSerialization.data(withJSONObject: cpu).write(to: cpuURL)
        manifest["bundle_sha256"] = h3SHA256(data); manifest["phase_manifest_sha256"] = try h3SHA256(Data(contentsOf: phaseURL))
        if fault == "resident" { manifest["weight_residency"] = "all_blocks" }
        let manifestURL = root.appendingPathComponent("chain.json"); try JSONSerialization.data(withJSONObject: manifest).write(to: manifestURL)
        func open() throws -> H3DenoisingPlan {
            try H3DenoisingPlan(bundle: bundleURL, phases: phaseURL, cpuRecord: cpuURL, chain: manifestURL, modulationDirectory: root.appendingPathComponent("models"))
        }
        if fault == "none" {
            let plan = try open()
            #expect(plan.modulations.count == 50 && plan.backbone.cases.allSatisfy(\.streamsWeights))
            #expect(plan.modulations[49].url.lastPathComponent == "modulation-49.aimodel")
            #expect(plan.schedule.steps.map { plan.layout.timesteps(for: $0).values.count } == [1,2,2,2,2,2,2,2])
        } else { #expect(throws: (any Error).self) { try open() } }
    }
}
