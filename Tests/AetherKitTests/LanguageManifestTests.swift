import AetherKit
import CryptoKit
import Foundation
import Testing

struct LanguageManifestTests {
    @Test func checksPlatformAndIntegrityBeforeNativeLoad() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("model.aimodel"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        let model = directory.appendingPathComponent("model.aimodel/main.mlirb")
        let tokenizer = directory.appendingPathComponent("tokenizer/tokenizer.json")
        try Data("fixture".utf8).write(to: model)
        try Data("{}".utf8).write(to: tokenizer)
        let values: [String: Any] = ["schemaVersion": 1, "modelDirectory": "model.aimodel", "tokenizerDirectory": "tokenizer",
            "sourceRepository": "example/model", "sourceRevision": String(repeating: "a", count: 40),
            "modelSHA256": try LanguageModelManifest.fileHash(model), "tokenizerSHA256": try LanguageModelManifest.fileHash(tokenizer),
            "maximumContextLength": 16384, "stopTokenIDs": [2], "platform": "macos", "quantization": "test"]
        let manifest = try JSONDecoder().decode(LanguageModelManifest.self, from: JSONSerialization.data(withJSONObject: values))
        #expect(try manifest.verify(in: directory, platform: .macos) == directory.appendingPathComponent("model.aimodel").resolvingSymlinksInPath())
        #expect(throws: AetherError.self) { try manifest.verify(in: directory, platform: .ios) }
        try Data("changed".utf8).write(to: model)
        #expect(throws: AetherError.self) { try manifest.verify(in: directory, platform: .macos) }
    }
    @Test func fileHashSpansChunksAndPartialTail() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        var raw = Data(repeating: 0x5a, count: 8 << 20)
        raw.append(contentsOf: 0..<17)
        try raw.write(to: url)
        #expect(try LanguageModelManifest.fileHash(url) == SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined())
    }
    @Test func exportBoundsDecodeAndBoundTheRegisteredContext() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("model.aimodel"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        let model = directory.appendingPathComponent("model.aimodel/main.mlirb")
        let tokenizer = directory.appendingPathComponent("tokenizer/tokenizer.json")
        try Data("fixture".utf8).write(to: model)
        try Data("{}".utf8).write(to: tokenizer)
        func manifest(context: Int, bounds: [String: Int]?) throws -> LanguageModelManifest {
            var values: [String: Any] = ["schemaVersion": 1, "modelDirectory": "model.aimodel", "tokenizerDirectory": "tokenizer",
                "sourceRepository": "example/model", "sourceRevision": String(repeating: "a", count: 40),
                "modelSHA256": try LanguageModelManifest.fileHash(model), "tokenizerSHA256": try LanguageModelManifest.fileHash(tokenizer),
                "maximumContextLength": context, "stopTokenIDs": [2], "platform": "macos", "quantization": "test"]
            if let bounds { values["exportBounds"] = bounds }
            return try JSONDecoder().decode(LanguageModelManifest.self, from: JSONSerialization.data(withJSONObject: values))
        }
        // Absent bounds stay nil (0.0.1 registrations decode unchanged).
        #expect(try manifest(context: 4096, bounds: nil).exportBounds == nil)
        let bounded = try manifest(context: 4096, bounds: ["maximumPrefillChunk": 256, "maximumCacheCapacity": 4096])
        #expect(bounded.exportBounds == LanguageExportBounds(maximumPrefillChunk: 256, maximumCacheCapacity: 4096))
        _ = try bounded.verify(in: directory, platform: .macos)
        // A context beyond the exported capacity, or invalid bounds, fail registration.
        #expect(throws: AetherError.self) {
            try manifest(context: 8192, bounds: ["maximumPrefillChunk": 256, "maximumCacheCapacity": 4096]).verify(in: directory, platform: .macos)
        }
        #expect(throws: AetherError.self) {
            try manifest(context: 1024, bounds: ["maximumPrefillChunk": 0, "maximumCacheCapacity": 4096]).verify(in: directory, platform: .macos)
        }
    }

    @Test func rejectsTraversalBeforeReadingAssets() throws {
        let values: [String: Any] = ["schemaVersion": 1, "modelDirectory": "../escape", "tokenizerDirectory": "tokenizer",
            "sourceRepository": "example/model", "sourceRevision": String(repeating: "a", count: 40),
            "modelSHA256": String(repeating: "a", count: 64), "tokenizerSHA256": String(repeating: "b", count: 64),
            "maximumContextLength": 16384, "stopTokenIDs": [2], "platform": "macos", "quantization": "test"]
        let manifest = try JSONDecoder().decode(LanguageModelManifest.self, from: JSONSerialization.data(withJSONObject: values))
        #expect(throws: AetherError.self) { try manifest.verify(in: URL(filePath: "/private/tmp"), platform: .macos) }
    }
}
