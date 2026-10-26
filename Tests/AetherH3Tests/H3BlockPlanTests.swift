import AetherH3
import Foundation
import Testing

private func planFixture() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    for name in ["block-bundle", "block-phases", "turbo8-schedule"] {
        let src = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        try FileManager.default.copyItem(at: src, to: root.appendingPathComponent(name + ".json"))
    }
    return root
}

@Test func h3ProbePlanBindsCorrectedWeightsAndBothActualCardinalities() throws {
    let root = try planFixture(); defer { try? FileManager.default.removeItem(at: root) }
    let plan = try H3BlockProbePlan(bundle: root.appendingPathComponent("block-bundle.json"), phases: root.appendingPathComponent("block-phases.json"))
    #expect(plan.cases.count == 2)
    #expect(plan.cases.map(\.inputs.count) == [27,27])
    #expect(plan.cases.map { $0.inputs.first { $0.name == "temb" }!.spec.shape } == [[6,3,5376],[6,6,5376]])
    #expect(plan.cases.allSatisfy { $0.boundBytes < 2 << 30 && $0.referenceRows.count == 64 })
    #expect(plan.cases.allSatisfy { $0.inputs.filter { $0.name.hasPrefix("w") }.count == 22 })
}

@Test func h3ProbePlanRejectsAlteredBindingAndReferenceDeclarations() throws {
    for fault in ["weight", "graph", "missing", "table", "rows", "cardinality", "schedule"] {
        let root = try planFixture(); defer { try? FileManager.default.removeItem(at: root) }
        let phases = root.appendingPathComponent("block-phases.json")
        var raw = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: phases)) as? [String: Any])
        var cases = raw["cases"] as! [[String: Any]], inputs = cases[0]["inputs"] as! [[String: Any]]
        switch fault {
        case "weight": inputs[5]["sha256"] = String(repeating: "0", count: 64)
        case "graph": var graph = cases[0]["model"] as! [String: Any]; graph["sha256"] = String(repeating: "0", count: 64); cases[0]["model"] = graph
        case "missing": inputs.removeLast()
        case "table": inputs[1]["shape"] = [6,9,5376]
        case "rows": var outputs = cases[0]["outputs"] as! [[String: Any]]; outputs[0]["row_indices"] = [0,0]; cases[0]["outputs"] = outputs
        case "cardinality": cases.removeLast()
        default:
            let schedule = root.appendingPathComponent("turbo8-schedule.json")
            try Data("{}".utf8).write(to: schedule)
        }
        cases[0]["inputs"] = inputs; raw["cases"] = cases
        try JSONSerialization.data(withJSONObject: raw).write(to: phases)
        #expect(throws: (any Error).self) {
            try H3BlockProbePlan(bundle: root.appendingPathComponent("block-bundle.json"), phases: phases)
        }
    }
}

@Test func h3ReferenceComparisonSelectsOnlyCapturedRowsAndRejectsUndefinedError() throws {
    let actual = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1,3,1]), bytes: Data([128,63,0,64,64,64]))
    let reference = try H3TensorData(spec: H3TensorSpec(scalarType: .bfloat16, shape: [1,2,1]), bytes: Data([64,64,128,63]))
    let comparison = try H3ReferenceComparison(actual: actual, reference: reference, rows: [2,0])
    #expect(comparison.passed && comparison.byteIdentical && comparison.relativeL2 == 0)
    #expect(throws: (any Error).self) { try H3ReferenceComparison(actual: actual, reference: reference, rows: [0,3]) }
    let zeros = try H3TensorData(spec: reference.spec, bytes: Data(repeating: 0, count: 4))
    let bad = try H3ReferenceComparison(actual: actual, reference: zeros, rows: [2,0])
    #expect(!bad.passed && bad.relativeL2 == nil)
}
