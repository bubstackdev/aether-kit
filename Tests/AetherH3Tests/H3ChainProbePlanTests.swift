import AetherKit
import AetherH3
import Foundation
import Testing

func chainFixture(references: Bool, full: Bool = false) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    for name in ["block-bundle", "block-phases", "turbo8-schedule"] {
        let source = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        try FileManager.default.copyItem(at: source, to: root.appendingPathComponent(name + ".json"))
    }
    let phases = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("block-phases.json"))) as! [String: Any]
    let cases = (phases["cases"] as! [[String: Any]]).map { item -> [String: Any] in
        let inputs = item["inputs"] as! [[String: Any]], outputs = item["outputs"] as! [[String: Any]]
        let table = inputs.first { $0["name"] as? String == "temb" }!
        let steps = (0..<(full ? 50 : 3)).map { block -> [String: Any] in
            var step: [String: Any] = ["block": block, "temb": table]
            if block == 0 || (references && (!full || [0,1,2,24,49].contains(block))) { step["reference"] = outputs[0] }
            if references { step["expected_output_sha256"] = String(repeating: "a", count: 64) }
            return step
        }
        return ["name": item["name"]!, "steps": steps]
    }
    var manifest: [String: Any] = ["schema": full ? 2 : 1, "block_indices": full ? Array(0..<50) : [0,1,2],
        "bundle_sha256": try h3SHA256(Data(contentsOf: root.appendingPathComponent("block-bundle.json"))),
        "phase_manifest_sha256": try h3SHA256(Data(contentsOf: root.appendingPathComponent("block-phases.json"))),
        "predefined_relative_l2_limit": 0.02, "oracle_recipe_sha256": String(repeating: "a", count: 64), "cases": cases]
    if full { manifest["weight_residency"] = "one_block"; manifest["reference_blocks"] = [0,1,2,24,49] }
    try JSONSerialization.data(withJSONObject: manifest).write(to: root.appendingPathComponent("chain.json"))
    return root
}
private func inspectChain(_ root: URL, references: Bool) throws -> H3ChainProbePlan {
    try H3ChainProbePlan(bundle: root.appendingPathComponent("block-bundle.json"), phases: root.appendingPathComponent("block-phases.json"),
                         chain: root.appendingPathComponent("chain.json"), requireReferences: references)
}

@Test func h3ChainRequiresCompleteReferencesForQualificationAndBindsThreeDistinctWeightSets() throws {
    let root = try chainFixture(references: false); defer { try? FileManager.default.removeItem(at: root) }
    let plan = try inspectChain(root, references: false)
    #expect(plan.cases.count == 2)
    for item in plan.cases {
        #expect(item.steps.map(\.block) == [0,1,2])
        #expect(item.steps.map(\.weights.count) == [22,22,22])
        #expect(item.boundBytes < 4 << 30)
        #expect(Set(item.steps.map { $0.weights[1].sha256 }).count == 3)
        #expect(item.steps[0].reference != nil && item.steps[1].reference == nil)
    }
    #expect(throws: AetherKit.AetherError.self) { try inspectChain(root, references: true) }
    let complete = try chainFixture(references: true); defer { try? FileManager.default.removeItem(at: complete) }
    #expect(try inspectChain(complete, references: true).cases.allSatisfy { $0.steps.allSatisfy { $0.reference != nil && $0.expectedOutputSHA256 != nil } })
}

@Test func h3ChainRejectsStaleSourcesReorderedBlocksCardinalityAndContextHashes() throws {
    for fault in ["bundle", "phase", "order", "missing", "case", "table", "block0table", "reference", "hash", "limit"] {
        let root = try chainFixture(references: true); defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("chain.json")
        var raw = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
        var cases = raw["cases"] as! [[String: Any]], steps = cases[0]["steps"] as! [[String: Any]]
        switch fault {
        case "bundle": raw["bundle_sha256"] = String(repeating: "0", count: 64)
        case "phase": raw["phase_manifest_sha256"] = String(repeating: "0", count: 64)
        case "order": steps.swapAt(1,2)
        case "missing": steps.removeLast()
        case "case": cases[1]["name"] = cases[0]["name"]
        case "table": var table = steps[1]["temb"] as! [String: Any]; table["shape"] = [6,9,5376]; steps[1]["temb"] = table
        case "block0table": var table = steps[0]["temb"] as! [String: Any]; table["sha256"] = String(repeating: "0", count: 64); steps[0]["temb"] = table
        case "reference": var ref = steps[1]["reference"] as! [String: Any]; ref["row_indices"] = [0,0]; steps[1]["reference"] = ref
        case "hash": steps[1]["expected_output_sha256"] = "not-a-hash"
        default: raw["predefined_relative_l2_limit"] = 0.03
        }
        cases[0]["steps"] = steps; raw["cases"] = cases
        try JSONSerialization.data(withJSONObject: raw).write(to: path)
        #expect(throws: (any Error).self) { try inspectChain(root, references: true) }
    }
}


@Test func h3FiftyBlockPlanAdmitsOneResidentWeightSetAndRequiresAllHashesAndCheckpoints() throws {
    let root = try chainFixture(references: true, full: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let plan = try inspectChain(root, references: true)
    for item in plan.cases {
        #expect(item.streamsWeights && item.steps.count == 50)
        #expect(item.referenceBlocks == [0,1,2,24,49])
        #expect(item.steps.filter { $0.reference != nil }.count == 5)
        #expect(item.boundBytes < 2 << 30)
        #expect(item.steps.reduce(0) { $0 + $1.weights.reduce(0) { $0 + $1.spec.byteCount } } > 30 << 30)
    }
    let path = root.appendingPathComponent("chain.json")
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
    for fault in ["residency", "checkpoints", "missingHash", "missingReference", "order", "missingBlock"] {
        var raw = original
        var cases = raw["cases"] as! [[String: Any]], steps = cases[0]["steps"] as! [[String: Any]]
        switch fault {
        case "residency": raw["weight_residency"] = "all_blocks"
        case "checkpoints": raw["reference_blocks"] = [0,1,2]
        case "missingHash": steps[7].removeValue(forKey: "expected_output_sha256")
        case "missingReference": steps[24].removeValue(forKey: "reference")
        case "order": steps.swapAt(24,25)
        default: steps.remove(at: 25)
        }
        cases[0]["steps"] = steps; raw["cases"] = cases
        try JSONSerialization.data(withJSONObject: raw).write(to: path)
        #expect(throws: (any Error).self) { try inspectChain(root, references: true) }
    }
}
