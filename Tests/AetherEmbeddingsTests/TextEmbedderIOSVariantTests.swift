import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import CryptoKit
import Foundation
import Testing
@testable import AetherEmbeddings

/// EmbeddingGemma 300M loads on iOS (hermetic: no graph or tokenizer files, no GPU). The iOS variant's manifest
/// (`AetherCoreTests/Fixtures/m4-ios/embeddinggemma-300m/aether.json`, the file the device gate ran on an iPhone 19,4)
/// must be the one the catalog pins, then go through the real driver load and the pipeline's construction (contract,
/// memory admission with the platform default budget) on an iPhone `Device`. The embedder itself also reads the
/// tokenizer, which the gates and the device run cover.
@Suite("Text embedder iOS variant")
struct TextEmbedderIOSVariantTests {
    @Test func embeddingGemmaLoadsOnIOS() async throws {
        let phone = Device(platform: .iOS, architecture: "h19p")
        let id = "embeddinggemma-300m"
        let ref = try ModelRef.resolve(id, for: phone)
        #expect(ref.variant == "ios" && ref.driver == .textEmbedder)
        let data = try Data(contentsOf: URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appending(path: "AetherCoreTests/Fixtures/m4-ios/\(id)/aether.json"))
        let pinned = try #require(ref.asset("main")?.files.first { $0.path == ".aether/aether.json" })
        #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == pinned.sha256, "the pinned manifest")

        let url = FileManager.default.temporaryDirectory.appending(path: "aether-ios-\(id)-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: url) }
        try FileManager.default.createDirectory(at: url.appending(path: ".aether"), withIntermediateDirectories: true)
        try data.write(to: url.appending(path: ".aether/aether.json"))
        let model = try await TextEmbedderDriver.load(
            InstalledBundle(ref: ref, url: url, isStale: false),
            options: LoadOptions(device: phone, cachePolicy: .default, cache: .default))
        #expect(model.manifest.variant.platform == .iOS && model.device == phone)
        _ = try StatelessPipeline(
            manifest: model.manifest, device: phone, contract: .textEmbedder, options: PipelineOptions(),
            load: { role throws(AetherError) in throw .runtime(underlying: CocoaError(.featureUnsupported, userInfo: ["role": role])) },
            fileBytes: { _ in 0 })
    }
}
