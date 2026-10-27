import AetherKit
import AetherH3
import Foundation
import Testing

/// A minimal schema-3 manifest (weight-free graphs, one tiny weight file) with optional first-frame encoder entries.
private func profile(_ root: URL, extra: [String: Any]) throws -> H3Profile {
    let weight = root.appendingPathComponent("weight"), embedding = root.appendingPathComponent("embedding")
    let bytes = Data([128, 63]), rows = Data([128, 63, 0, 64, 64, 64, 128, 64])
    try bytes.write(to: weight); try rows.write(to: embedding)
    let zero = String(repeating: "0", count: 64)
    func graph(_ name: String) -> [String: Any] { ["path": root.appendingPathComponent(name + ".aimodel").path, "function": "main", "sha256": zero] }
    func tensor(_ name: String, shape: [Int] = [1], dtype: String = "bfloat16") -> [String: Any] {
        ["name": name, "path": weight.path, "dtype": dtype, "shape": shape, "sha256": h3SHA256(bytes)]
    }
    let conditioner: [String: Any] = [
        "embedding_table": ["path": embedding.path, "offset": 0, "length": 8, "dtype": "bfloat16", "shape": [2, 2], "sha256": h3SHA256(rows)],
        "layer": graph("layer"), "parameters": Array(repeating: (0..<11).map { tensor("w\($0)") }, count: 50), "tap": 50,
        "rope_tables": ["cos": tensor("cos", shape: [4096, 128]), "sin": tensor("sin", shape: [4096, 128])],
    ]
    let step: [String: Any] = ["timesteps": [0], "temb": tensor("temb", shape: [1, 2688], dtype: "float32"),
                               "blocks": Array(repeating: tensor("table", shape: [6, 3, 5376]), count: 50)]
    var decoder = graph("decoder")
    decoder.merge(["tile": 256, "window_latents": 7, "window_frames": 22, "window_stride": 5]) { _, new in new }
    var manifest: [String: Any] = [
        "schema": 3, "complete": false, "missing": ["audio_decoder"],
        "snapshot": H3ExportHeader.snapshot, "diffusers_commit": H3ExportHeader.diffusersCommit,
        "transformer_weight_layout": H3ExportHeader.transformerWeightLayout,
        "limits": ["rows": [1, 1000], "cardinality": [1, 3], "text_tokens": [1, 4096]],
        "tokenizer": root.path, "conditioner": conditioner,
        "prepare": ["text": graph("text"), "text_parameters": [tensor("w0")], "pack": graph("pack")],
        "body": graph("body"), "heads": graph("heads"), "parameters": Array(repeating: (0..<22).map { tensor("w\($0)") }, count: 50),
        "video_decoder": decoder,
        "recipes": ["fixture": ["schedule": ["video": ["shift": 12, "timesteps": [0], "sigmas": [1, 0]],
                                              "audio": ["shift": 3, "timesteps": [0], "sigmas": [1, 0]]], "tasks": ["t2va": [step]]]],
        "video_encoder_first": ["960x544": graph("whole-960x544"), "544x960": graph("whole-544x960")],
    ]
    manifest.merge(extra) { _, new in new }
    let url = root.appendingPathComponent(UUID().uuidString + ".json")
    try JSONSerialization.data(withJSONObject: manifest).write(to: url)
    return try H3Profile(url: url)
}

private func tiled(_ root: URL, tails: [String], tileSize: Int = 256) -> [String: Any] {
    let zero = String(repeating: "0", count: 64)
    func graph(_ name: String) -> [String: Any] { ["path": root.appendingPathComponent(name + ".aimodel").path, "function": "main", "sha256": zero] }
    return ["video_encoder_first_tiled": ["tile": graph("tile256"), "tile_size": tileSize, "moments_channels": 48,
                                          "tails": Dictionary(uniqueKeysWithValues: tails.map { ($0, graph("tail-" + $0)) })]]
}

@Test func h3KeyframeEncodingFollowsTheManifestTiledWhenDeclaredElseTheWholeFrameGraph() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).resolvingSymlinksInPath()
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    // Without tiled entries (every existing bundle): the whole-frame graph, unchanged.
    let plain = try profile(root, extra: [:])
    #expect(plain.tiledKeyframeEncoder == nil)
    guard case .whole(let model)? = try plain.keyframeEncoding(width: 960, height: 544) else { Issue.record("expected the whole-frame graph"); return }
    #expect(model.url.lastPathComponent == "whole-960x544.aimodel")
    #expect(try plain.keyframeEncoding(width: 288, height: 288) == nil)
    // With tiled entries: the tiled graphs for each declared canvas, the source split, and the whole-frame graph elsewhere.
    let split = try profile(root, extra: tiled(root, tails: ["960x544", "288x288"]))
    guard case .tiled(let tile, let tail, let tiling)? = try split.keyframeEncoding(width: 960, height: 544) else { Issue.record("expected tiled"); return }
    #expect(tile.url.lastPathComponent == "tile256.aimodel" && tail.url.lastPathComponent == "tail-960x544.aimodel")
    #expect(tiling == (try H3SpatialTiling.source(width: 960, height: 544)) && tiling.tileCount == 15)
    guard case .tiled(_, _, let small)? = try split.keyframeEncoding(width: 288, height: 288) else { Issue.record("expected tiled 288"); return }
    #expect(small.tileCount == 4)
    guard case .whole? = try split.keyframeEncoding(width: 544, height: 960) else { Issue.record("expected whole 544x960"); return }
    // Tiled entries outside the 256 px source tile contract are refused.
    #expect(throws: AetherError.self) { _ = try profile(root, extra: tiled(root, tails: ["960x544"], tileSize: 512)) }
    #expect(throws: AetherError.self) { _ = try profile(root, extra: tiled(root, tails: ["200x200"])) }
}

extension H3Profile.KeyframeEncoding: Equatable {
    public static func == (a: Self, b: Self) -> Bool {
        switch (a, b) {
        case (.whole(let x), .whole(let y)): x.url == y.url
        case (.tiled(let t1, let a1, let s1), .tiled(let t2, let a2, let s2)): t1.url == t2.url && a1.url == a2.url && s1 == s2
        default: false
        }
    }
}
