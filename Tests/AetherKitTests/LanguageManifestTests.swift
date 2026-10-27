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

    /// An image registration verifies its encoder asset, needs an image-chunk bound covering its
    /// largest image, and is refused when it asks for reshape-optimized encoder specialization (F-P3-26).
    @Test func visionRegistrationVerifiesAndRefusesFrequentReshapes() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        for sub in ["model.aimodel", "vision.aimodel", "tokenizer"] {
            try FileManager.default.createDirectory(at: directory.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        let model = directory.appendingPathComponent("model.aimodel/main.mlirb")
        let vision = directory.appendingPathComponent("vision.aimodel/main.mlirb")
        let tokenizer = directory.appendingPathComponent("tokenizer/tokenizer.json")
        try Data("text".utf8).write(to: model)
        try Data("vision".utf8).write(to: vision)
        try Data("{}".utf8).write(to: tokenizer)
        func manifest(reshapes: Bool?, imageChunk: Int? = 2048, visionHash: String? = nil) throws -> LanguageModelManifest {
            var registration: [String: Any] = ["modelDirectory": "vision.aimodel", "modelSHA256": try visionHash ?? LanguageModelManifest.fileHash(vision),
                "imageTokenID": 151655, "visionStartTokenID": 151652, "visionEndTokenID": 151653, "patchSize": 16, "mergeSize": 2,
                "temporalPatchSize": 2, "minimumPixels": 65536, "maximumPixels": 16777216, "imageMean": [0.5, 0.5, 0.5],
                "imageStandardDeviation": [0.5, 0.5, 0.5], "positionTableSide": 48, "maximumImageTokens": 1024,
                "imageGrids": [["height": 44, "width": 44], ["height": 36, "width": 48]]]
            if let reshapes { registration["expectFrequentReshapes"] = reshapes }
            var bounds: [String: Any] = ["maximumPrefillChunk": 512, "maximumCacheCapacity": 4096]
            if let imageChunk { bounds["maximumImageChunk"] = imageChunk }
            let values: [String: Any] = ["schemaVersion": 1, "modelDirectory": "model.aimodel", "tokenizerDirectory": "tokenizer",
                "sourceRepository": "Qwen/Qwen3-VL-2B-Instruct", "sourceRevision": String(repeating: "a", count: 40),
                "modelSHA256": try LanguageModelManifest.fileHash(model), "tokenizerSHA256": try LanguageModelManifest.fileHash(tokenizer),
                "maximumContextLength": 4096, "stopTokenIDs": [151645], "platform": "macos", "quantization": "fp16",
                "exportBounds": bounds, "vision": registration]
            return try JSONDecoder().decode(LanguageModelManifest.self, from: JSONSerialization.data(withJSONObject: values))
        }
        let ok = try manifest(reshapes: nil)
        _ = try ok.verify(in: directory, platform: .macos)
        #expect(try ok.verifyVision(in: directory) == directory.appendingPathComponent("vision.aimodel").resolvingSymlinksInPath())
        #expect(try ok.vision?.encoderExpectsFrequentReshapes() == false)
        _ = try manifest(reshapes: false).verify(in: directory, platform: .macos)
        #expect(throws: AetherError.self) { try manifest(reshapes: true).verify(in: directory, platform: .macos) }
        #expect(throws: AetherError.self) { try manifest(reshapes: true).vision?.encoderExpectsFrequentReshapes() }
        #expect(throws: AetherError.self) { try manifest(reshapes: nil, imageChunk: nil).verify(in: directory, platform: .macos) }
        #expect(throws: AetherError.self) { try manifest(reshapes: nil, imageChunk: 512).verify(in: directory, platform: .macos) }
        #expect(throws: AetherError.self) {
            try manifest(reshapes: nil, visionHash: String(repeating: "0", count: 64)).verifyVision(in: directory)
        }
        #expect(LanguageVisionRegistration.reshapesQualified == false)
    }

    @Test func visionRegistrationRefusesOverflowingImageBudget() throws {
        func registration(tokens: Int, patch: Int = 16, merge: Int = 2) throws -> LanguageVisionRegistration {
            let values: [String: Any] = ["modelDirectory": "vision.aimodel", "modelSHA256": String(repeating: "a", count: 64),
                "imageTokenID": 151655, "visionStartTokenID": 151652, "visionEndTokenID": 151653, "patchSize": patch, "mergeSize": merge,
                "temporalPatchSize": 2, "minimumPixels": 65536, "maximumPixels": 16777216, "imageMean": [0.5, 0.5, 0.5],
                "imageStandardDeviation": [0.5, 0.5, 0.5], "positionTableSide": 48, "maximumImageTokens": tokens,
                "imageGrids": [["height": 2, "width": 2]]]
            return try JSONDecoder().decode(LanguageVisionRegistration.self, from: JSONSerialization.data(withJSONObject: values))
        }
        let ok = try registration(tokens: 1024)
        try ok.validate()
        #expect(ok.imageTokenPixelBudget == 1024 * 32 * 32)
        // Each factor of tokens · (patch · merge)² can overflow; validation throws instead of trapping.
        for hostile in [try registration(tokens: Int.max / 2), try registration(tokens: 1, patch: Int.max / 2),
                        try registration(tokens: 1, patch: 1 << 32, merge: 1)] {
            #expect(hostile.imageTokenPixelBudget == nil)
            #expect(throws: AetherError.self) { try hostile.validate() }
        }
        // A budget below the minimum pixels is still refused.
        #expect(throws: AetherError.self) { try registration(tokens: 1).validate() }
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
