import AetherKit
import AetherH3
import CryptoKit
import Foundation
import Testing

/// Writes a structurally complete synthetic schema-1 bundle with tiny stand-in files.
private struct SyntheticBundle {
    let root: URL
    var manifest: [String: Any]
    init() throws {
        let base = FileManager.default.temporaryDirectory.appending(path: "h3-bundle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: false)
        root = base
        func model(_ name: String) throws -> [String: Any] {
            let dir = base.appending(path: "\(name).aimodel")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: false)
            let stream = Data("stream \(name)".utf8), digest = Data(SHA256.hash(data: stream))
            try stream.write(to: dir.appending(path: "main.mlirb")); try digest.write(to: dir.appending(path: "main.hash"))
            return ["path": dir.path, "function": "main", "sha256": digest.map { String(format: "%02x", $0) }.joined()]
        }
        func tensor(_ name: String, file: String, dtype: String = "bfloat16") throws -> [String: Any] {
            let url = base.appending(path: file), data = Data([1, 2, 3, 4])
            if !FileManager.default.fileExists(atPath: url.path) { try data.write(to: url) }
            return ["name": name, "path": url.path, "dtype": dtype, "shape": [dtype == "float32" ? 1 : 2],
                    "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
        }
        let body = try model("body"), modulation = try model("modulation")
        manifest = [
            "schema": 1, "complete": true, "snapshot": H3ExportHeader.snapshot, "diffusers_commit": H3ExportHeader.diffusersCommit,
            "recipe": "turbo8", "transformer_weight_layout": H3ExportHeader.transformerWeightLayout,
            "adapter": ["repository": "lightx2v/Minimax-h3-Turbo", "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
                        "sha256": "e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f",
                        "execution": "dynamic_lora", "rank": 128, "alpha": 8.0],
            "width": 960, "height": 544, "frames": 124, "text_rows": 49,
            "tokenizer": base.appending(path: "tokenizer.json").path, "schedule": base.appending(path: "schedule.json").path,
            "prepare": try model("prepare"), "time": try model("time"), "heads": try model("heads"),
            "bodies": [body, try model("body1")], "modulations": Array(repeating: modulation, count: 50),
            "parameters": try (0..<50).map { _ in try (0..<22).map { try tensor("w\($0)", file: "weight.raw") } },
            "rotary_cos": try tensor("cos", file: "cos.raw", dtype: "float32"), "rotary_sin": try tensor("sin", file: "sin.raw", dtype: "float32"),
            "video_decoder": try model("video"), "audio_decoder": try model("audio"),
            "video_window_latents": 7, "video_window_frames": 22, "video_window_stride": 5,
        ]
    }
    func write() throws -> URL {
        let url = root.appending(path: "bundle-\(UUID().uuidString).json")
        try JSONSerialization.data(withJSONObject: manifest).write(to: url)
        return url
    }
}

@Test func h3BundleOpensACompleteProfileAndVerifiesEveryStreamAndTensorOnce() throws {
    let synthetic = try SyntheticBundle(); defer { try? FileManager.default.removeItem(at: synthetic.root) }
    let bundle = try H3Bundle(url: synthetic.write())
    #expect(bundle.geometry.frames == 124 && bundle.textRows == 49 && bundle.header.recipe == .turbo8)
    #expect(bundle.models.count == 8)          // prepare, time, heads, video, audio, two bodies, and 50 identical modulations once
    #expect(bundle.tensors.count == 3)         // one shared weight file, cos, sin
    var calls = 0
    try bundle.verifyModels { _, total in calls += 1; #expect(total == 8) }
    #expect(calls == 8)
    try bundle.verifyTensors()
    #expect(try bundle.diskBytes() > 0)
}

@Test func h3BundleVerificationFailsOnAChangedStreamOrTensor() throws {
    let synthetic = try SyntheticBundle(); defer { try? FileManager.default.removeItem(at: synthetic.root) }
    let bundle = try H3Bundle(url: synthetic.write())
    try Data("changed".utf8).write(to: synthetic.root.appending(path: "heads.aimodel/main.mlirb"))
    #expect(throws: AetherError.self) { try bundle.verifyModels() }
    try Data([9, 9, 9, 9]).write(to: synthetic.root.appending(path: "weight.raw"))
    #expect(throws: AetherError.self) { try bundle.verifyTensors() }
}

@Test func h3BundleRefusesIncompleteOrStructurallyWrongProfiles() throws {
    var synthetic = try SyntheticBundle(); defer { try? FileManager.default.removeItem(at: synthetic.root) }
    let original = synthetic.manifest
    let mutations: [(String, ([String: Any]) -> [String: Any])] = [
        ("no QKV marker", { var m = $0; m.removeValue(forKey: "transformer_weight_layout"); return m }),
        ("49 modulations", { var m = $0; m["modulations"] = Array(($0["modulations"] as! [Any]).dropLast()); return m }),
        ("21 bindings", { var m = $0; var p = $0["parameters"] as! [[Any]]; p[3] = Array(p[3].dropLast()); m["parameters"] = p; return m }),
        ("no window contract", { var m = $0; m.removeValue(forKey: "video_window_latents"); return m }),
        ("unaligned frames", { var m = $0; m["frames"] = 120; return m }),
        ("relative path", { var m = $0; var a = $0["prepare"] as! [String: Any]; a["path"] = "models/prepare.aimodel"; m["prepare"] = a; return m }),
        ("49-layer tap", { var m = $0; let a = $0["prepare"]!; m["conditioner"] = ["embedding": a, "layers": Array(repeating: a, count: 49)]; return m }),
        ("int32 weight", { var m = $0; var t = $0["rotary_cos"] as! [String: Any]; t["dtype"] = "int32"; m["rotary_cos"] = t; return m }),
    ]
    for (name, mutate) in mutations {
        synthetic.manifest = mutate(original)
        #expect(throws: AetherError.self, "\(name)") { try H3Bundle(url: synthetic.write()) }
    }
}

/// Opt-in, metadata only: the prototype's older production export predates the corrected
/// per-head QKV conversion and must be refused before any weight is read.
@Test(.enabled(if: ProcessInfo.processInfo.environment["AETHER_H3_PROTOTYPE_BUNDLE"] != nil))
func h3BundleRefusesThePrototypesUncorrectedProductionExport() throws {
    let url = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["AETHER_H3_PROTOTYPE_BUNDLE"]))
    #expect(throws: AetherError.self) { try H3Bundle(url: url) }
}
