import Foundation
import Testing
@testable import AetherCore

/// MANIFEST.md §5.8: a graph's `weights` (the inputs a weight table binds per block). The fixture is the H3
/// exporter's own output (`Forge/h3/export_h3.py --sample-manifest`).
@Suite("Block weights in the manifest")
struct WeightsManifestTests {
    static func h3() throws -> Data { try Fixtures.data("h3/minimax-h3.json") }

    static func mutated(_ role: String, _ edit: @escaping (inout [String: Any]) -> Void) throws -> Data {
        var json = try JSONSerialization.jsonObject(with: h3()) as! [String: Any]
        var graphs = json["graphs"] as! [String: Any]
        var graph = graphs[role] as! [String: Any]
        edit(&graph)
        graphs[role] = graph
        json["graphs"] = graphs
        return try JSONSerialization.data(withJSONObject: json)
    }

    @Test func theH3ExportersManifestDecodes() throws {
        let manifest = try Manifest.decode(try Self.h3())
        #expect(manifest.driver.description == "minimax-h3@1")
        let body = try #require(manifest.graphs["body"]?.weights)
        #expect(body.table == AssetPath(asset: "main", path: "weights/body.json"))
        #expect(body.inputs == (0..<22).map { "w\($0)" })
        #expect(body.blocks == 50)
        #expect(manifest.graphs["conditionerLayer"]?.weights?.inputs.count == 11)
        #expect(manifest.graphs["prepareText"]?.weights?.blocks == 1)
        #expect(manifest.graphs["heads"]?.weights == nil)
        #expect(manifest.graphs["body"]?.residency == .perStep)
        // Round trip: what decodes encodes back to the same graph.
        let again = try Manifest.decode(try JSONEncoder().encode(manifest))
        #expect(again.graphs["body"]?.weights == body)
    }

    @Test func weightRulesNameTheirField() throws {
        let cases: [(String, (inout [String: Any]) -> Void)] = [
            ("graphs.body.weights.inputs", { g in g["weights"] = ["table": "main:weights/body.json", "inputs": [String]()] }),
            ("graphs.body.weights.inputs", { g in g["weights"] = ["table": "main:weights/body.json", "inputs": ["w0", "w0"]] }),
            ("graphs.body.weights.inputs", { g in g["weights"] = ["table": "main:weights/body.json", "inputs": ["w0", "tags"]] }),
            ("graphs.body.weights.table", { g in g["weights"] = ["table": "other:weights/body.json", "inputs": ["w0"]] }),
            ("graphs.body.weights.table", { g in g["weights"] = ["table": "main:../body.json", "inputs": ["w0"]] }),
            ("graphs.body.weights.blocks", { g in g["weights"] = ["table": "main:weights/body.json", "inputs": ["w0"], "blocks": 0] }),
        ]
        for (expected, edit) in cases {
            expectField(expected) { _ = try Manifest.decode(try Self.mutated("body", edit)) }
        }
    }
}
