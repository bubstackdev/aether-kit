import AetherH3
import Foundation
import Testing

@Test func h3ProtectedProfileBindsBodyConditionerRefinerAndEmbeddingWithoutIdentityChanges() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let weight = root.appendingPathComponent("weight"), embedding = root.appendingPathComponent("embedding")
    let bytes = Data([128,63]), rows = Data([128,63, 0,64, 64,64, 128,64])
    try bytes.write(to: weight); try rows.write(to: embedding)
    let zero = String(repeating: "0", count: 64)
    let graph: [String: Any] = ["path": root.appendingPathComponent("unused.aimodel").path, "function": "main", "sha256": zero]
    func tensor(_ name: String, shape: [Int] = [1], dtype: String = "bfloat16") -> [String: Any] {
        ["name": name, "path": weight.path, "dtype": dtype, "shape": shape, "sha256": h3SHA256(bytes)]
    }
    let conditioner: [String: Any] = [
        "embedding_table": ["path": embedding.path, "offset": 0, "length": 8, "dtype": "bfloat16", "shape": [2,2], "sha256": h3SHA256(rows)],
        "layer": graph, "parameters": Array(repeating: (0..<11).map { tensor("w\($0)") }, count: 50), "tap": 50,
        "rope_tables": ["cos": tensor("cos", shape: [4096,128]), "sin": tensor("sin", shape: [4096,128])],
    ]
    let step: [String: Any] = ["timesteps": [0], "temb": tensor("temb", shape: [1,2688], dtype: "float32"),
                                "blocks": Array(repeating: tensor("table", shape: [6,3,5376]), count: 50)]
    var decoder = graph
    decoder.merge(["tile": 256, "window_latents": 7, "window_frames": 22, "window_stride": 5]) { _, new in new }
    let manifest: [String: Any] = [
        "schema": 3, "complete": false, "missing": ["audio_decoder"],
        "snapshot": H3ExportHeader.snapshot, "diffusers_commit": H3ExportHeader.diffusersCommit,
        "transformer_weight_layout": H3ExportHeader.transformerWeightLayout,
        "limits": ["rows": [1,1000], "cardinality": [1,3], "text_tokens": [1,4096]],
        "tokenizer": root.path, "conditioner": conditioner,
        "prepare": ["text": graph, "text_parameters": [tensor("w0")], "pack": graph],
        "body": graph, "heads": graph, "parameters": Array(repeating: (0..<22).map { tensor("w\($0)") }, count: 50),
        "video_decoder": decoder,
        "recipes": ["fixture": ["schedule": ["video": ["shift": 12, "timesteps": [0], "sigmas": [1,0]],
                                                  "audio": ["shift": 3, "timesteps": [0], "sigmas": [1,0]]], "tasks": ["t2va": [step]]]],
    ]
    let url = root.appendingPathComponent("profile.json")
    try JSONSerialization.data(withJSONObject: manifest).write(to: url)
    let original = try H3Profile(url: url), protected = try original.protectingWeights()
    #expect(protected.manifestSHA256 == original.manifestSHA256)
    #expect(protected.modelIdentity == original.modelIdentity)
    try FileManager.default.moveItem(at: weight, to: root.appendingPathComponent("old-weight"))
    try FileManager.default.moveItem(at: embedding, to: root.appendingPathComponent("old-embedding"))
    try Data([0,0]).write(to: weight); try Data(count: 8).write(to: embedding)
    #expect(try protected.parameters[49][21].read().floatValues() == [1])
    #expect(try protected.conditioner?.parameters[49][10].read().floatValues() == [1])
    #expect(try protected.prepare?.textParameters[0].read().floatValues() == [1])
    try protected.conditioner?.embedding.verify()
    #expect(try protected.conditioner?.embedding.rows([1]).floatValues() == [3,4])
    #expect(throws: (any Error).self) { try original.parameters[0][0].read() }
}
