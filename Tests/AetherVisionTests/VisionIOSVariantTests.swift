import AetherCore
@_spi(Testing) import AetherHost
import AetherRuntime
import CryptoKit
import Foundation
import Testing
@testable import AetherVision

/// The M4 vision exports load on iOS (hermetic: no graph files, no GPU). The iOS variant's manifest
/// (`AetherCoreTests/Fixtures/m4-ios/<id>/aether.json`, the file the device gate ran on an iPhone 19,4) must be the one
/// the catalog pins, then go through the real driver load and the pipeline's construction (contract, memory
/// admission with the platform default budget) on an iPhone `Device`. Graphs load on the first request, which the
/// gates and the device run cover.
@Suite("Vision iOS variants")
struct VisionIOSVariantTests {
    static let phone = Device(platform: .iOS, architecture: "h19p")
    static let fixtures = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "AetherCoreTests/Fixtures/m4-ios")

    /// The catalog's iOS ref for `id`, installed in a fresh directory holding only the fixture manifest.
    static func bundle(_ id: String) throws -> InstalledBundle {
        let ref = try ModelRef.resolve(id, for: phone)
        #expect(ref.variant == "ios")
        let data = try Data(contentsOf: fixtures.appending(path: "\(id)/aether.json"))
        let pinned = try #require(ref.asset("main")?.files.first { $0.path == ".aether/aether.json" })
        #expect(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == pinned.sha256, "\(id): the pinned manifest")
        let url = FileManager.default.temporaryDirectory.appending(path: "aether-ios-\(id)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url.appending(path: ".aether"), withIntermediateDirectories: true)
        try data.write(to: url.appending(path: ".aether/aether.json"))
        return InstalledBundle(ref: ref, url: url, isStale: false)
    }

    static func options() -> LoadOptions { LoadOptions(device: phone, cachePolicy: .default, cache: .default) }

    /// Builds the pipeline over a loader that must not be called.
    static func pipeline(_ manifest: Manifest) throws {
        let contract = try #require(DriverCatalog.contracts[manifest.driver])
        _ = try StatelessPipeline(
            manifest: manifest, device: phone, contract: contract, options: PipelineOptions(),
            load: { role throws(AetherError) in throw .runtime(underlying: CocoaError(.featureUnsupported, userInfo: ["role": role])) },
            fileBytes: { _ in 0 })
    }

    @Test func depthAnythingLoadsOnIOS() async throws {
        let bundle = try Self.bundle("depth-anything-3-small")
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try await DepthEstimatorDriver.load(bundle, options: Self.options())
        #expect(model.manifest.variant.platform == .iOS && model.device == Self.phone)
        _ = try model.estimator()
        try Self.pipeline(model.manifest)
    }

    @Test(arguments: [("yolox-s", DriverID.detectorDense), ("rf-detr-nano", .detectorDetr), ("rf-detr-seg-nano", .detectorDetr)])
    func detectorsLoadOnIOS(_ id: String, driver: DriverID) async throws {
        let bundle = try Self.bundle(id)
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        #expect(bundle.ref.driver == driver)
        let model = driver == .detectorDense
            ? try await DenseDetectorDriver.load(bundle, options: Self.options())
            : try await DetrDetectorDriver.load(bundle, options: Self.options())
        #expect(model.manifest.variant.platform == .iOS && model.manifest.driver == driver)
        _ = try model.detector()
        try Self.pipeline(model.manifest)
    }

    /// CLIP's encoder also reads the tokenizer directory, which the fixture bundle doesn't hold: the driver load
    /// (manifest and configuration) and the pipeline over its contract.
    @Test func clipLoadsOnIOS() async throws {
        let bundle = try Self.bundle("clip-vit-b32")
        defer { try? FileManager.default.removeItem(at: bundle.url) }
        let model = try await ImageTextEncoderDriver.load(bundle, options: Self.options())
        #expect(model.manifest.variant.platform == .iOS && model.manifest.driver == .clipJoint)
        try Self.pipeline(model.manifest)
    }
}
