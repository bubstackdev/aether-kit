@testable import AetherH3
import Foundation
import Testing

func phaseFixture(fault: String? = nil) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    for name in ["block-bundle", "block-phases", "turbo8-schedule", "outer-phases"] {
        let source = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        try FileManager.default.copyItem(at: source, to: root.appendingPathComponent(name + ".json"))
    }
    func read(_ name: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(name + ".json"))) as? [String: Any])
    }
    var bundle = try read("block-bundle"), outer = try read("outer-phases"), phases = try read("block-phases")
    var cases = outer["cases"] as! [[String: Any]]
    bundle["prepare"] = cases[0]["model"]; bundle["time"] = cases[1]["model"]; bundle["heads"] = cases[2]["model"]
    let bundleData = try JSONSerialization.data(withJSONObject: bundle)
    try bundleData.write(to: root.appendingPathComponent("block-bundle.json"))
    // Exercise actual schedule/index bytes without materializing large fixtures.
    let layout = try H3Layout(geometry: H3Geometry(width: 960, height: 544, requestedFrames: 124), textTags: Array(repeating: 1, count: 49))
    let schedule = try H3Schedule()
    try FileManager.default.createDirectory(at: root.appendingPathComponent("assets"), withIntermediateDirectories: true)
    for ordinal in 0..<2 {
        let plan = layout.timesteps(for: schedule.steps[ordinal])
        var timeInputs = cases[1 + ordinal * 2]["inputs"] as! [[String: Any]]
        var headInputs = cases[2 + ordinal * 2]["inputs"] as! [[String: Any]]
        var timeValues = plan.values, indices = plan.indices
        if fault == "timestep", ordinal == 1 { timeValues[0] += 0.01 }
        if fault == "indices", ordinal == 1 { indices[49] = 1 - indices[49] }
        let timestepData = timeValues.withUnsafeBytes { Data($0) }, indexData = indices.withUnsafeBytes { Data($0) }
        timeInputs[0]["sha256"] = h3SHA256(timestepData)
        let index = headInputs.firstIndex { $0["name"] as? String == "indices" }!
        headInputs[index]["sha256"] = h3SHA256(indexData)
        try timestepData.write(to: root.appendingPathComponent(timeInputs[0]["path"] as! String))
        try indexData.write(to: root.appendingPathComponent(headInputs[index]["path"] as! String))
        cases[1 + ordinal * 2]["inputs"] = timeInputs; cases[2 + ordinal * 2]["inputs"] = headInputs
    }
    outer["cases"] = cases
    var cpu: [String: Any] = ["schema": 1, "passed": true, "gpu_work": false,
        "bundle_sha256": h3SHA256(bundleData), "source_recipe_sha256": String(repeating: "a", count: 64), "cases": cases]
    switch fault {
    case "cpu": cpu["passed"] = false
    case "gpu": cpu["gpu_work"] = true
    case "bundle": cpu["bundle_sha256"] = String(repeating: "0", count: 64)
    case "duplicate": cases.append(cases[0])
    case "missing": cases.removeLast()
    case "graph": var model = cases[0]["model"] as! [String: Any]; model["sha256"] = String(repeating: "0", count: 64); cases[0]["model"] = model
    case "precision": var inputs = cases[0]["inputs"] as! [[String: Any]]; inputs[2]["dtype"] = "float32"; cases[0]["inputs"] = inputs
    case "output": var outputs = cases[2]["outputs"] as! [[String: Any]]; outputs.removeLast(); cases[2]["outputs"] = outputs
    case "reference": var outputs = cases[0]["outputs"] as! [[String: Any]]; outputs[0]["sha256"] = String(repeating: "0", count: 64); cases[0]["outputs"] = outputs
    case "context":
        var inputs = cases[2]["inputs"] as! [[String: Any]]
        inputs[1]["sha256"] = String(repeating: "0", count: 64); cases[2]["inputs"] = inputs; cpu["cases"] = cases
    default: break
    }
    phases["cases"] = (phases["cases"] as! [[String: Any]]) + cases
    try JSONSerialization.data(withJSONObject: phases).write(to: root.appendingPathComponent("block-phases.json"))
    try JSONSerialization.data(withJSONObject: cpu).write(to: root.appendingPathComponent("cpu.json"))
    return root
}
private func phasePlan(_ root: URL) throws -> H3PhaseProbePlan {
    try H3PhaseProbePlan(bundle: root.appendingPathComponent("block-bundle.json"), phases: root.appendingPathComponent("block-phases.json"), cpuRecord: root.appendingPathComponent("cpu.json"))
}

@Test func h3OuterPlanLinksCorrectedGraphsCompleteCPUOutputsAndBackboneContext() throws {
    let root = try phaseFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let plan = try phasePlan(root)
    #expect(plan.cases.map(\.name) == ["prepare", "time-0", "heads-0", "time-1", "heads-1"])
    #expect(plan.cases.map(\.references.count) == [1,1,2,1,2])
    #expect(plan.cases.allSatisfy { $0.boundBytes < 1 << 30 })
    #expect(plan.cases[0].references[0].spec.scalarType == .bfloat16)
    try plan.verifyTimestepInputs()
}
@Test func h3OuterPlanRejectsStaleCPUAndBoundaryDeclarations() throws {
    for fault in ["cpu", "gpu", "bundle", "duplicate", "missing", "graph", "precision", "output", "reference", "context"] {
        let root = try phaseFixture(fault: fault); defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) { try phasePlan(root) }
    }
}
@Test func h3OuterPlanRejectsWrongScheduleValuesAndInRangeWrongRowIndices() throws {
    for fault in ["timestep", "indices"] {
        let root = try phaseFixture(fault: fault); defer { try? FileManager.default.removeItem(at: root) }
        let plan = try phasePlan(root)
        #expect(throws: (any Error).self) { try plan.verifyTimestepInputs() }
    }
}
@Test func h3CompleteComparisonHandlesBF16F32ZeroNormAndIncompatibleStorage() throws {
    func bf(_ bits: [UInt16]) throws -> H3TensorData {
        try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [bits.count]), bytes: bits.withUnsafeBytes { Data($0) })
    }
    let actual = try bf([0x3f80,0x4000]), reference = try bf([0x3f80,0x3f80]), zeros = try bf([0,0])
    let comparison = try H3TensorComparison(actual: actual, reference: reference)
    #expect(!comparison.passed && abs(comparison.relativeL2! - sqrt(0.5)) < 1e-12 && comparison.maximumAbsoluteError == 1)
    #expect(try H3TensorComparison(actual: actual, reference: actual).byteIdentical)
    #expect(try H3TensorComparison(actual: zeros, reference: zeros).relativeL2 == 0)
    #expect(try H3TensorComparison(actual: actual, reference: zeros).relativeL2 == nil)
    let values: [Float] = [1e30,-1e30]
    let f32 = try H3TensorData(spec: H3TensorSpec(scalarType: .float32, shape: [2]), bytes: values.withUnsafeBytes { Data($0) })
    #expect(try H3TensorComparison(actual: f32, reference: f32).passed)
    let changedValues: [Float] = [-1e30,-1e30]
    let changed = try H3TensorData(spec: f32.spec, bytes: changedValues.withUnsafeBytes { Data($0) })
    let f32Difference = try H3TensorComparison(actual: changed, reference: f32)
    #expect(!f32Difference.passed && abs(f32Difference.relativeL2! - sqrt(2)) < 1e-12)
    #expect(throws: (any Error).self) { try H3TensorComparison(actual: actual, reference: f32) }
    #expect(throws: (any Error).self) { try H3TensorComparison(actual: actual, reference: actual, limit: .nan) }
}
