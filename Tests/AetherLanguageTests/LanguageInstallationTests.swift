import AetherAssets
@testable import AetherLanguage
import AetherKit
import CryptoKit
import Foundation
import Testing

private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

/// A language bundle whose registration pins real hashes, so it installs through `LocalAssetStore`.
private struct HashedBundle {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("language-install-\(UUID().uuidString)")

    init(registerTokenizerFiles: Bool, vision: Bool = false, wrongDeclaration: Bool = false) throws {
        let files = FileManager.default
        func write(_ data: Data, _ path: String) throws {
            let target = url.appendingPathComponent(path)
            try files.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
        }
        func asset(_ directory: String, graph: String) throws -> String {
            let stream = Data(graph.utf8)
            try write(stream, directory + "/main.mlirb")
            var declaration = Data(SHA256.hash(data: stream))
            if wrongDeclaration { declaration[0] ^= 0xff }
            try write(declaration, directory + "/main.hash")
            try write(Data(#"{"functions":["main"]}"#.utf8), directory + "/metadata.json")
            return sha256(stream)
        }
        let model = try asset("model.aimodel", graph: "language graph")
        let tokenizer = Data(#"{"model":{}}"#.utf8)
        try write(tokenizer, "tokenizer/tokenizer.json")
        try write(Data("{{ messages }}".utf8), "tokenizer/chat_template.jinja")
        try write(Data("unregistered".utf8), "tokenizer/extra.txt")
        try write(Data("finder".utf8), "tokenizer/.DS_Store")
        try write(Data(#"{"provenance":true}"#.utf8), "export-report.json")
        var registration: [String: Any] = [
            "schemaVersion": 1, "modelDirectory": "model.aimodel", "tokenizerDirectory": "tokenizer",
            "sourceRepository": "example/model", "sourceRevision": String(repeating: "a", count: 40),
            "modelSHA256": model, "tokenizerSHA256": sha256(tokenizer),
            "maximumContextLength": 2048, "stopTokenIDs": [2], "platform": "macos", "quantization": "fp16",
        ]
        if registerTokenizerFiles {
            registration["tokenizerFiles"] = ["tokenizer.json": sha256(tokenizer),
                                              "chat_template.jinja": sha256(Data("{{ messages }}".utf8))]
        }
        if vision {
            let encoder = try asset("vision.aimodel", graph: "vision graph")
            registration["exportBounds"] = ["maximumPrefillChunk": 256, "maximumCacheCapacity": 2048, "maximumImageChunk": 256]
            registration["vision"] = [
                "modelDirectory": "vision.aimodel", "modelSHA256": encoder, "imageTokenID": 10, "visionStartTokenID": 11,
                "visionEndTokenID": 12, "patchSize": 16, "mergeSize": 2, "temporalPatchSize": 2, "minimumPixels": 1024,
                "maximumPixels": 65536, "imageMean": [0.5, 0.5, 0.5], "imageStandardDeviation": [0.5, 0.5, 0.5],
                "positionTableSide": 48, "maximumImageTokens": 256, "imageGrids": [["height": 16, "width": 16]],
            ]
        }
        try write(JSONSerialization.data(withJSONObject: registration, options: [.sortedKeys]), "language-model.json")
    }

    func remove() { try? FileManager.default.removeItem(at: url) }
}

struct LanguageInstallationTests {
    @Test func registeredTokenizerFilesAreExactlyWhatInstalls() throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true)
        defer { bundle.remove() }
        let manifest = try LanguageModelManifest.installationManifest(in: bundle.url)
        #expect(manifest.files.map(\.path) == [
            "language-model.json", "model.aimodel/main.hash", "model.aimodel/main.mlirb", "model.aimodel/metadata.json",
            "tokenizer/chat_template.jinja", "tokenizer/tokenizer.json",
        ])
        let registration = try Data(contentsOf: bundle.url.appendingPathComponent("language-model.json"))
        #expect(manifest.assetIdentity == "aether-language/1:" + sha256(registration))
        #expect(manifest.executionContract == "aether.language-bundle/1")
        let stream = try Data(contentsOf: bundle.url.appendingPathComponent("model.aimodel/main.mlirb"))
        #expect(manifest.files.first { $0.path == "model.aimodel/main.mlirb" }?.sha256 == sha256(stream))
        #expect(manifest.files.first { $0.path == "model.aimodel/main.mlirb" }?.byteCount == UInt64(stream.count))
    }

    @Test func withoutARegisteredListEveryVisibleTokenizerFileInstalls() throws {
        let bundle = try HashedBundle(registerTokenizerFiles: false)
        defer { bundle.remove() }
        let paths = try LanguageModelManifest.installationManifest(in: bundle.url).files.map(\.path)
        #expect(paths.filter { $0.hasPrefix("tokenizer/") }
            == ["tokenizer/chat_template.jinja", "tokenizer/extra.txt", "tokenizer/tokenizer.json"])
        #expect(!paths.contains("export-report.json"))
    }

    @Test func imageBundlesInstallTheirVisionAsset() throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true, vision: true)
        defer { bundle.remove() }
        let paths = try LanguageModelManifest.installationManifest(in: bundle.url).files.map(\.path)
        #expect(paths.contains("vision.aimodel/main.mlirb") && paths.contains("vision.aimodel/main.hash")
            && paths.contains("vision.aimodel/metadata.json"))
    }

    @Test func aDeclarationThatDisagreesWithTheRegistrationRefuses() throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true, wrongDeclaration: true)
        defer { bundle.remove() }
        #expect(throws: AetherError.self) { try LanguageModelManifest.installationManifest(in: bundle.url) }
    }

    @Test func theManifestInstallsAndTheInstallationDescribesItselfIdentically() async throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true)
        let storeRoot = FileManager.default.temporaryDirectory.appendingPathComponent("language-store-\(UUID().uuidString)")
        defer { bundle.remove(); try? FileManager.default.removeItem(at: storeRoot) }
        let manifest = try LanguageModelManifest.installationManifest(in: bundle.url)
        let store = try LocalAssetStore(root: storeRoot)
        let id = try await store.install(manifest, from: bundle.url)
        #expect(try id == manifest.installationID)
        let lease = try await store.acquire(id)
        #expect(try LanguageModelManifest.installationManifest(in: lease.directory) == manifest)
    }

    @Test func aModelOpenedFromAnInstallationHoldsItsLease() async throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true)
        let storeRoot = FileManager.default.temporaryDirectory.appendingPathComponent("language-store-\(UUID().uuidString)")
        defer { bundle.remove(); try? FileManager.default.removeItem(at: storeRoot) }
        let store = try LocalAssetStore(root: storeRoot)
        let id = try await store.install(LanguageModelManifest.installationManifest(in: bundle.url), from: bundle.url)
        do {
            let model = AetherLanguageModel(installation: try await store.acquire(id), platform: .macos, coreAIAvailable: true)
            #expect(model.availability == AetherLanguageModel(bundle: bundle.url, platform: .macos, coreAIAvailable: true).availability)
            #expect(model.installation?.installationID == id && model.bundle == model.installation?.directory)
            let copy = model
            #expect(throws: AssetStoreError.busy) { try store.remove(id) }
            withExtendedLifetime(copy) {}
        }
        try store.remove(id)  // the last copy is gone, and so is the lease
    }

    @Test func anInstallationOfAnotherFamilyIsUnavailable() async throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true)
        let storeRoot = FileManager.default.temporaryDirectory.appendingPathComponent("language-store-\(UUID().uuidString)")
        defer { bundle.remove(); try? FileManager.default.removeItem(at: storeRoot) }
        let language = try LanguageModelManifest.installationManifest(in: bundle.url)
        let foreign = try AssetManifest(assetIdentity: language.assetIdentity, executionContract: "kokoro.speech-bundle/1",
                                        files: language.files)
        let store = try LocalAssetStore(root: storeRoot)
        let id = try await store.install(foreign, from: bundle.url)
        let model = AetherLanguageModel(installation: try await store.acquire(id), platform: .macos, coreAIAvailable: true)
        guard case .unavailable(.bundleInvalid) = model.availability else {
            Issue.record("expected bundleInvalid, got \(model.availability)"); return
        }
    }

    @Test func theLanguageReadinessCheckFollowsTheFacade() async throws {
        let bundle = try HashedBundle(registerTokenizerFiles: true)
        let storeRoot = FileManager.default.temporaryDirectory.appendingPathComponent("language-store-\(UUID().uuidString)")
        defer { bundle.remove(); try? FileManager.default.removeItem(at: storeRoot) }
        let manifest = try LanguageModelManifest.installationManifest(in: bundle.url)
        let entry = try ModelCatalogEntry(manifest: manifest, source: .https(base: URL(string: "https://example.com/m")!))
        let store = try LocalAssetStore(root: storeRoot)
        let lease = try await store.acquire(try await store.install(manifest, from: bundle.url))

        let here = ModelRuntimeCheck.language(platform: .macos, coreAIAvailable: true)
        #expect(here.entry(entry) == .runnable)
        #expect(here.installation(lease) == .runnable)
        // Below Core AI's OS nothing runs, before or after installing.
        let oldOS = ModelRuntimeCheck.language(platform: .macos, coreAIAvailable: false)
        #expect(oldOS.entry(entry) == .unsupported(.operatingSystem))
        #expect(oldOS.installation(lease) == .unsupported(.operatingSystem))
        // A macOS registration on an iPhone.
        #expect(ModelRuntimeCheck.language(platform: .ios, coreAIAvailable: true).installation(lease) == .unsupported(.platform))

        let foreign = try ModelCatalogEntry(
            manifest: AssetManifest(assetIdentity: manifest.assetIdentity, executionContract: "kokoro.speech-bundle/1", files: manifest.files),
            source: entry.source)
        guard case .invalid = here.entry(foreign) else { Issue.record("a speech entry passed the language check"); return }
    }
}
